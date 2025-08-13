import numpy as np
import trimesh
import warp as wp


def diff_dmc(volume: wp.types.array(dtype=wp.types.float32, ndim=3), spacing: float | np.ndarray,
             origin: np.ndarray, threshold: float):
    import torch
    from diso import DiffDMC
    vertices, indices = DiffDMC(dtype=torch.float32)(-wp.to_torch(volume), None, isovalue=-threshold)
    vertices, indices = vertices.cpu().numpy(), indices.cpu().numpy()
    vertices = vertices * spacing * np.array(volume.shape) + origin
    return trimesh.Trimesh(vertices, indices)


@wp.context.kernel
def region_sample(
        volume: wp.types.uint64, volume_spacing: wp.types.vec3,
        array: wp.types.array(ndim=3), origin: wp.types.vec3, spacing: float,
        xform: wp.types.transform,
):
    i, j, k = wp.tid()

    # 采样值
    p = origin + spacing * wp.types.vec3(float(i), float(j), float(k))
    p = wp.transform_point(xform, p)

    uvw = wp.cw_div(p, volume_spacing)
    pixel = wp.volume_sample_f(volume, uvw, wp.types.Volume.LINEAR)

    array[i, j, k] = array.dtype(pixel)


@wp.context.kernel
def planar_cut(
        array: wp.types.array(ndim=3), origin: wp.types.vec3, spacing: float,
        xform: wp.types.transform, center: wp.types.vec3, normal: wp.types.vec3, bone_threshold: float,
):
    i, j, k = wp.tid()
    pixel = array[i, j, k]

    p = origin + spacing * wp.types.vec3(float(i), float(j), float(k))
    p = wp.transform_point(xform, p)

    d = wp.dot(wp.normalize(normal), p - center)

    if pixel > bone_threshold:
        if d <= 0.0:
            pixel = d + bone_threshold
        elif d < spacing * 1.5 and pixel > bone_threshold:
            pixel = d + bone_threshold

    array[i, j, k] = array.dtype(pixel)


@wp.context.kernel
def region_raymarching(
        array: wp.types.array(ndim=2, dtype=wp.types.vec4ub), origin: wp.types.vec3, spacing: wp.types.vec3,
        x_axis: wp.types.vec3, y_axis: wp.types.vec3, z_axis: wp.types.vec3, z_length: float, z_length_alpha: float,
        volume: wp.types.uint64, volume_spacing: wp.types.vec3, xform: wp.types.transform,
        mesh: wp.types.uint64, threshold_min: float, threshold_max: float, window_min: float, window_max: float,
):
    i, j = wp.tid()

    x_axis = wp.normalize(x_axis)
    y_axis = wp.normalize(y_axis)
    z_axis = wp.normalize(z_axis)

    ray_start = origin
    ray_start += float(i) * spacing[0] * x_axis
    ray_start += float(j) * spacing[1] * y_axis

    step = spacing[2] * z_axis
    gray_sum = float(0)

    n = int(wp.ceil(z_length / spacing[2]))
    grey_count = float(0)

    alpha_n = int(wp.ceil(z_length_alpha / spacing[2]))
    alpha_count = float(0)

    highlight_count = float(0)

    p = ray_start
    for k in range(n):
        p += step

        uvw = wp.cw_div(wp.transform_point(xform, p), volume_spacing)
        pixel = wp.volume_sample_f(volume, uvw, wp.types.Volume.LINEAR)

        if threshold_min <= pixel:
            gray_sum += pixel
            grey_count += 1.0

            if threshold_max <= pixel:
                highlight_count += 1.0
                if k < alpha_n:
                    alpha_count += 1.0

    if grey_count > 0:
        gray_sum /= grey_count

    grey = wp.round((gray_sum - window_min) / (window_max - window_min) * 255.0)
    grey = wp.clamp(grey, 0.0, 255.0)

    q = wp.mesh_query_ray(mesh, ray_start, z_axis, wp.types.float32(z_length))

    mesh_pbr = wp.types.vec3(255.0, 225.0, 0.0)
    highlight_color = wp.types.vec3(255.0, 0.0, 0.0)
    intersect_color = wp.types.vec3(255.0, 175.0, 0.0)

    if highlight_count > 0 and q.result:
        rgb = wp.types.vec3(grey) * 0.5 + intersect_color * 0.5
    elif highlight_count > 0:
        rgb = wp.types.vec3(grey) * 0.5 + highlight_color * 0.5
    elif q.result:
        rgb = wp.types.vec3(grey) * 0.5 + mesh_pbr * 0.5
    else:
        rgb = wp.types.vec3(grey)

    if alpha_count > 0:
        a = 255.0
    else:
        a = 0.0

    array[i, j] = array.dtype(wp.types.uint8(rgb[0]), wp.types.uint8(rgb[1]), wp.types.uint8(rgb[2]), wp.types.uint8(a))
