from typing import Dict

import numpy as np
import pyvista as pv
import trimesh
import warp as wp


def itk_monkey_patch():
    import itk, locale
    locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')

    # 修正 itk.imread 读取 Direction 错误，导致 TotalSeg 分割错误
    # itk 默认 ForceOrthogonalDirection=True 不适用于从脚到头 (FFS) 或斜切 (gantry-tilt) 扫描
    _New = itk.ImageSeriesReader.New
    itk.ImageSeriesReader.New = lambda *a, **k: _New(*a, **{**k, 'ForceOrthogonalDirection': False})


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
    trimesh_monkey_patch()

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


def winding_volume(mesh: trimesh.Trimesh, spacing: float, bounds: list, winding: float = 0.5):
    length = bounds[1] - bounds[0]
    size = (length / spacing).astype(int)
    size = np.max([size, [1, 1, 1]], axis=0)
    max_dist = np.linalg.norm(bounds[1] - bounds[0])

    mesh = wp.Mesh(wp.array(mesh.vertices, wp.vec3), wp.array(mesh.faces.flatten(), wp.int32), None, True)
    volume = wp.empty(shape=(*size,), dtype=wp.float32)

    wp.launch(mesh_winding_volume, volume.shape, [
        mesh.id, volume, wp.vec3(wp.float32(spacing)),
        wp.vec3(bounds[0]), max_dist, winding,
    ])

    return volume


@wp.kernel
def mesh_winding_volume(
        mesh: wp.uint64, array: wp.array(dtype=wp.float32, ndim=3),
        spacing: wp.vec3, origin: wp.vec3,
        max_dist: wp.float32, threshold: wp.float32,
):
    """计算网格的卷绕密度，适用于不封闭网格"""
    i, j, k = wp.tid()

    p = origin
    p += wp.cw_mul(spacing, wp.vec3(wp.float32(i), wp.float32(j), wp.float32(k)))

    q = wp.mesh_query_point_sign_winding_number(
        mesh, p, max_dist, wp.float32(2.0), threshold,
    )

    closest = wp.mesh_eval_position(mesh, q.face, q.u, q.v)

    d = -q.sign * wp.length(p - closest)

    array[i, j, k] = d


@wp.kernel
def extract_outlier_faces(
        mesh: wp.uint64, vertices: wp.array1d(dtype=wp.vec3),
        faces: wp.array1d(dtype=wp.vec3i), face_normals: wp.array1d(dtype=wp.vec3),
        mask: wp.array1d(dtype=wp.bool), max_dist: wp.float32,
):
    i = wp.tid()

    p0 = vertices[faces[i][0]]
    p1 = vertices[faces[i][1]]
    p2 = vertices[faces[i][2]]
    face_center = (p0 + p1 + p2) / 3.0

    n = wp.normalize(face_normals[i])
    q = wp.mesh_query_ray(mesh, face_center + n * 1e-6, n, max_dist)
    if q.result:
        mask[i] = False


@wp.kernel
def compare_loss(
        volume_a: wp.uint64, spacing_a: wp.vec3,
        volume_b: wp.uint64, spacing_b: wp.vec3,
        hu_bone: float, hu_metal: float,
        pos: wp.array1d(dtype=wp.vec3), rot: wp.array1d(dtype=wp.quat),
        loss: wp.array1d(dtype=float), count: wp.array1d(dtype=float),
):
    i, j, k = wp.tid()

    # 术前图像直接取值
    uvw_a = wp.vec3(wp.float32(i), wp.float32(j), wp.float32(k))
    grad_a = wp.vec3()
    voxel_a = wp.volume_sample_grad_f(volume_a, uvw_a, wp.Volume.LINEAR, grad_a)

    # 仅比较介于骨与金属之间的体素
    if voxel_a < hu_bone:
        return

    # 术后图像经过配准变换插值
    p_a = wp.cw_mul(spacing_a, uvw_a)
    p_b = wp.transform_point(wp.transform(pos[0], rot[0]), p_a)
    uvw_b = wp.cw_div(p_b, spacing_b)
    grad_b = wp.vec3()
    voxel_b = wp.volume_sample_grad_f(volume_b, uvw_b, wp.Volume.LINEAR, grad_b)

    # 仅比较介于骨与金属之间的体素
    if voxel_b > hu_metal:
        return

    # 累计值差异
    diff = wp.abs(voxel_a - voxel_b)

    # 累计梯度方向差异
    grad_a = wp.normalize(grad_a)
    grad_b = wp.normalize(grad_b)
    cos = wp.dot(grad_a, grad_b)
    cos = 1.0 - wp.clamp(cos, -1.0, 1.0)

    loss[0] += cos
    count[0] += 1.0


@wp.kernel
def average_loss(loss: wp.array1d(dtype=float), count: wp.array1d(dtype=float)):
    i = wp.tid()
    loss[i] /= count[i]


@wp.kernel
def normalize_quat(x: wp.array(dtype=wp.quat)):
    i = wp.tid()
    x[i] = wp.normalize(x[i])
