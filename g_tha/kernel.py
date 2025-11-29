import numpy as np
import trimesh
import warp as wp


def diff_dmc(volume: wp.array3d(dtype=wp.float32), origin: np.ndarray, spacing: float | np.ndarray,
             threshold: float, ):
    import torch
    from diso import DiffDMC
    vertices, indices = DiffDMC(dtype=torch.float32)(-wp.to_torch(volume), None, isovalue=-threshold)
    vertices, indices = vertices.cpu().numpy(), indices.cpu().numpy()
    vertices = vertices * spacing * (np.array(volume.shape) - [1, 1, 1]) + origin
    return trimesh.Trimesh(vertices, indices)


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

    # q = wp.mesh_query_ray(mesh, face_center - n * 1e-6, n, max_dist)
    #
    # if q.result:
    #     mask[i] = False


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
