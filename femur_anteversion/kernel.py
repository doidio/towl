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
        array: wp.types.array(ndim=2, dtype=wp.types.vec3ub), origin: wp.types.vec3, spacing: wp.types.vec3,
        x_axis: wp.types.vec3, y_axis: wp.types.vec3, z_length: float,
        volume: wp.types.uint64, volume_spacing: wp.types.vec3, xform: wp.types.transform,
        mesh: wp.types.uint64,
        threshold_min: float, window_min: float, window_max: float,
):
    i, j = wp.tid()

    x_axis = wp.normalize(x_axis)
    y_axis = wp.normalize(y_axis)

    ray_start = origin
    ray_start += float(i) * spacing[0] * x_axis
    ray_start += float(j) * spacing[1] * y_axis

    z_axis = wp.cross(x_axis, y_axis)
    z_axis = wp.normalize(z_axis)

    step = spacing[2] * z_axis
    pixel_sum = float(0)

    n = int(wp.ceil(z_length / spacing[2]))
    pixel_n = float(0)

    p = ray_start
    for k in range(n):
        p += step

        uvw = wp.cw_div(wp.transform_point(xform, p), volume_spacing)
        pixel = wp.volume_sample_f(volume, uvw, wp.types.Volume.LINEAR)

        if threshold_min <= pixel:
            pixel_sum += pixel
            pixel_n += 1.0

    if pixel_n > 0:
        pixel_sum /= pixel_n

    pixel_sum = wp.round((pixel_sum - window_min) / (window_max - window_min) * 255.0)
    pixel_sum = wp.clamp(pixel_sum, 0.0, 255.0)

    array[i, j] = array.dtype(wp.types.uint8(pixel_sum))
