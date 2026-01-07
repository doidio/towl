from typing import Dict, Literal

import cv2
import numpy as np
import pyvista as pv
import trimesh
import warp as wp
from chainner_ext import resize, ResizeFilter
from matplotlib import cm

wp.config.quiet = True


def itk_monkey_patch():
    import itk, locale
    locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')

    # 修正 itk.imread 读取 Direction 错误，导致 TotalSeg 分割错误
    # itk 默认 ForceOrthogonalDirection=True 不适用于从脚到头 (FFS) 或斜切 (gantry-tilt) 扫描
    _New = itk.ImageSeriesReader.New
    itk.ImageSeriesReader.New = lambda *a, **k: _New(*a, **{**k, 'ForceOrthogonalDirection': False})


itk_monkey_patch()


@wp.kernel
def _wp_closest_point_kernel(
        mesh_id: wp.uint64,
        points: wp.array(dtype=wp.vec3),
        out_closest: wp.array(dtype=wp.vec3),
        out_dist: wp.array(dtype=wp.float32),
        out_tid: wp.array(dtype=wp.int32)
):
    tid = wp.tid()
    p = points[tid]

    # 执行 Warp 内置的 Mesh 查询
    # max_dist 设为极大值以模拟无限制查询
    query = wp.mesh_query_point(mesh_id, p, 1e10)

    if query.result:
        # 计算查询点到最近点的向量
        closest_p = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
        out_closest[tid] = closest_p
        out_dist[tid] = wp.length(p - closest_p)
        out_tid[tid] = query.face
    else:
        out_tid[tid] = -1


def closest_point(mesh, points):
    """
    使用 NVIDIA Warp 替换 trimesh.proximity.closest_point
    """
    points = np.asanyarray(points, dtype=np.float32)  # Warp 常用 float32
    num_points = len(points)

    # 1. 缓存 Warp Mesh 对象，避免重复构建 BVH
    if not hasattr(mesh, '_warp_mesh') or mesh._warp_mesh is None:
        # 将 trimesh 数据传输到 Warp 数组
        faces = wp.array(mesh.faces.flatten(), dtype=wp.int32)
        verts = wp.array(mesh.vertices, dtype=wp.vec3)

        # 创建 Warp Mesh 句柄
        mesh_id = wp.Mesh(verts, faces)
        mesh._warp_mesh = mesh_id
    else:
        mesh_id = mesh._warp_mesh

    # 2. 准备输入/输出数据 (GPU Array)
    wp_points = wp.from_numpy(points, dtype=wp.vec3)
    out_closest = wp.zeros(num_points, dtype=wp.vec3)
    out_dist = wp.zeros(num_points, dtype=wp.float32)
    out_tid = wp.zeros(num_points, dtype=wp.int32)

    # 3. 启动 Kernel
    wp.launch(
        kernel=_wp_closest_point_kernel,
        dim=num_points,
        inputs=[mesh_id.id, wp_points, out_closest, out_dist, out_tid],
    )

    # 4. 将结果转回 NumPy 并保持接口一致
    return (
        out_closest.numpy().astype(np.float64),
        out_dist.numpy().astype(np.float64),
        out_tid.numpy().astype(np.int32)
    )


def trimesh_monkey_patch():
    trimesh.proximity.closest_point = closest_point


def tri_poly(tri: trimesh.Trimesh | tuple, cell_data: Dict[str, np.ndarray] = None):
    if isinstance(tri, trimesh.Trimesh):
        tri = (tri.vertices, tri.faces)
    _ = np.insert(tri[1].reshape(-1, 3), 0, 3, axis=1)

    pd = pv.PolyData(tri[0], _)

    if cell_data is not None:
        for name, data in cell_data.items():
            pd.cell_data[name] = data

    return pd


def diff_dmc(volume: wp.array3d(dtype=wp.float32), origin: np.ndarray, spacing: float | np.ndarray,
             threshold: float, ):
    # vertices, indices = wp.MarchingCubes.extract_surface_marching_cubes(
    #     -volume, -threshold, wp.vec3(origin), wp.vec3(origin + spacing * (np.array(volume.shape) - 1)),
    # )
    # return trimesh.Trimesh(vertices.numpy(), indices.numpy().reshape((-1, 3)))
    import torch
    from diso import DiffDMC
    vertices, indices = DiffDMC(dtype=torch.float32)(-wp.to_torch(volume), None, isovalue=-threshold)
    vertices, indices = vertices.cpu().numpy(), indices.cpu().numpy()
    vertices = vertices * spacing * (np.array(volume.shape) - 1) + origin
    return trimesh.Trimesh(vertices, indices)


def icp(a, b, initial=None, threshold=1e-5, max_iterations=20, **kwargs):
    """
    Apply the iterative closest point algorithm to align a point cloud with
    another point cloud or mesh. Will only produce reasonable results if the
    initial transformation is roughly correct. Initial transformation can be
    found by applying Procrustes' analysis to a suitable set of landmark
    points (often picked manually).

    Parameters
    ----------
    a : (n,3) float
      List of points in space.
    b : (m,3) float or Trimesh
      List of points in space or mesh.
    initial : (4,4) float
      Initial transformation.
    threshold : float
      Stop when change in cost is less than threshold
    max_iterations : int
      Maximum number of iterations
    kwargs : dict
      Args to pass to procrustes

    Returns
    ----------
    matrix : (4,4) float
      The transformation matrix sending a to b
    transformed : (n,3) float
      The image of a under the transformation
    cost : float
      The cost of the transformation
    """
    from scipy.spatial import cKDTree
    from trimesh import util
    from trimesh.transformations import transform_points
    from trimesh.registration import procrustes

    a = np.asanyarray(a, dtype=np.float64)
    if not util.is_shape(a, (-1, 3)):
        raise ValueError("points must be (n,3)!")

    if initial is None:
        initial = np.eye(4)

    is_mesh = util.is_instance_named(b, "Trimesh")
    if not is_mesh:
        b = np.asanyarray(b, dtype=np.float64)
        if not util.is_shape(b, (-1, 3)):
            raise ValueError("points must be (n,3)!")
        btree = cKDTree(b)

    # transform a under initial_transformation
    a = transform_points(a, initial)
    total_matrix = initial

    # start with infinite cost
    old_cost = np.inf

    # avoid looping forever by capping iterations
    for it in range(max_iterations):
        # Closest point in b to each point in a
        if is_mesh:
            closest, _distance, _faces = b.nearest.on_surface(a)
        else:
            _distances, ix = btree.query(a, 1)
            closest = b[ix]

        # align a with closest points
        matrix, transformed, cost = procrustes(a=a, b=closest, **kwargs)

        # update a with our new transformed points
        a = transformed
        total_matrix = np.dot(matrix, total_matrix)

        if old_cost - cost < threshold:
            break
        else:
            old_cost = cost

    return total_matrix, transformed, cost, it


trimesh_monkey_patch()


def fast_drr(a, ax, th=(0, 900), mode: Literal['mean', 'max'] = 'mean'):
    a = a.copy()
    c = th[0] <= a
    a *= c
    if mode == 'mean':
        a = a.sum(axis=ax)
        c = np.sum(c, axis=ax)
        c[np.where(c <= 0)] = 1
        a = a / c
    elif mode == 'max':
        a = a.max(axis=ax)

    sm = cm.ScalarMappable(cmap='grey')
    sm.set_clim(th)
    a = sm.to_rgba(a, bytes=True)

    return a[:, :, :3].copy()


def resize_uint8(img, shape):
    img = img.astype(np.float32) / 255.0
    img = resize(img, tuple(round(_) for _ in reversed(shape)), ResizeFilter.CubicMitchell, False)
    img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    return img


def cv2_line(img, p0, p1, color, thickness=1):
    cv2.line(img, p0[::-1], p1[::-1], color, thickness)


@wp.kernel
def compute_sdf(
        mesh: wp.uint64, vertices: wp.array1d(dtype=wp.vec3),
        sdf: wp.array1d(dtype=float), max_dist: wp.float32,
):
    i = wp.tid()

    p = vertices[i]
    q = wp.mesh_query_point_sign_normal(mesh, p, max_dist)
    closest = wp.mesh_eval_position(mesh, q.face, q.u, q.v)
    dxyz = p - closest
    d = q.sign * wp.length(dxyz)
    sdf[i] = d


@wp.kernel
def compose_op(
        image_a: wp.array3d(dtype=float), image_b: wp.array3d(dtype=float), image_c: wp.array3d(dtype=float),
        xform_a: wp.transform, volume_b: wp.uint64, spacing_a: wp.vec3, spacing_b: wp.vec3,
        ct_bone: float, ct_metal: float, swap: bool,
):
    i, j, k = wp.tid()
    a = image_a[i, j, k]

    pa = wp.cw_mul(spacing_a, wp.vec3(wp.float32(i), wp.float32(j), wp.float32(k)))
    pb = wp.transform_point(xform_a, pa)

    uvw = wp.cw_div(pb, spacing_b)
    b = wp.volume_sample_f(volume_b, uvw, wp.Volume.LINEAR)

    image_b[i, j, k] = b

    if swap:
        a, b = b, a

    image_c[i, j, k] = b if b > ct_metal else wp.min(a, ct_metal)


@wp.kernel
def contact_drr(
        image_a: wp.array3d(dtype=float), image_b: wp.array3d(dtype=float),
        image_3: wp.array3d(dtype=float), image_2: wp.array2d(dtype=wp.vec3ub),
        ct_bone: float, ct_metal: float, ct_bg: float, ct_min: float, ct_width: float,
        ax: int, swap: bool, proximal: wp.vec3i, distal: wp.vec3i,
):
    i, j = wp.tid()

    end = image_a.shape[ax]
    add = float(0.0)
    count = float(0.0)
    count_ = wp.vec2()

    for k in range(end):
        if ax == 1:
            ijk = wp.vec3i(i, wp.int32(k), j)
            ax_ = 2
        elif ax == 2:
            ijk = wp.vec3i(i, j, wp.int32(k))
            ax_ = 1
        else:
            ijk = wp.vec3i(wp.int32(k), i, j)
            ax_ = 2

        _3 = image_3[ijk[0], ijk[1], ijk[2]]
        if _3 > ct_min:
            add += _3
            count += 1.0

        if ijk[0] > proximal[0]:
            continue

        a = image_a[ijk[0], ijk[1], ijk[2]]
        b = image_b[ijk[0], ijk[1], ijk[2]]

        if swap:
            a, b = b, a

        if a >= ct_bone and b < ct_metal:  # 原位是骨
            for _ in range(2):
                ijk_ = ijk
                ijk_[ax_] += 1 if _ > 0 else -1
                if ijk_[ax_] < 0 or ijk_[ax_] >= image_a.shape[ax_]:
                    continue

                a_ = image_a[ijk_[0], ijk_[1], ijk_[2]]
                b_ = image_b[ijk_[0], ijk_[1], ijk_[2]]

                if swap:
                    a_, b_ = b_, a_

                if b_ >= ct_metal:  # 邻位有金属
                    count_[_] += 1.0

    if count > 0.0:
        grey = add / count
    else:
        grey = ct_bg

    grey = 255.0 * (grey - ct_min) / ct_width
    grey = wp.clamp(grey, 0.0, 255.0)

    if count_[0] > 1.0 and count_[1] > 1.0:
        r, g, b = 0.0, 255.0, 255.0
    elif count_[0] > 1.0:
        r, g, b = 0.0, 255.0, 0.0
    elif count_[1] > 1.0:
        r, g, b = 0.0, 0.0, 255.0
    elif count_[0] > 0.0:
        r, g, b = 127.0, 255.0, 127.0
    elif count_[1] > 0.0:
        r, g, b = 0.0, 127.0, 255.0
    else:
        r, g, b = grey, grey, grey

    image_2[i, j] = wp.vec3ub(wp.uint8(r), wp.uint8(g), wp.uint8(b))

