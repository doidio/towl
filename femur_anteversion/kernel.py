import numpy as np
import trimesh
import warp as wp


@wp.context.kernel
def femur_proximal_region(
        volume: wp.types.uint64, spacing: wp.types.vec3,
        region: wp.types.array(ndim=3), region_origin: wp.types.vec3, region_spacing: float, xform: wp.types.transform,
        neck_hcut_center: wp.types.vec3, neck_hcut_normal: wp.types.vec3,
        neck_vcut_center: wp.types.vec3, neck_vcut_normal: wp.types.vec3,
        bone_threshold: float,
):
    i, j, k = wp.tid()

    # 采样值
    p = region_origin + region_spacing * wp.types.vec3(float(i), float(j), float(k))
    p = wp.transform_point(xform, p)

    uvw = wp.cw_div(p, spacing)
    pixel = wp.volume_sample_f(volume, uvw, wp.types.Volume.LINEAR)

    # 横切股骨颈
    d = wp.dot(wp.normalize(neck_hcut_normal), p - neck_hcut_center)

    if pixel > bone_threshold:
        if d <= 0.0:
            pixel = d + bone_threshold
        elif d < region_spacing * 1.5 and pixel > bone_threshold:
            pixel = d + bone_threshold

    # 竖切股骨颈
    d = wp.dot(wp.normalize(neck_vcut_normal), p - neck_vcut_center)

    if pixel > bone_threshold:
        if d <= 0.0:
            pixel = d + bone_threshold
        elif d < region_spacing * 1.5 and pixel > bone_threshold:
            pixel = d + bone_threshold

    region[i, j, k] = region.dtype(pixel)


def diff_dmc(volume: wp.types.array(dtype=wp.types.float32, ndim=3), spacing: float | np.ndarray,
             origin: np.ndarray, threshold: float):
    import torch
    from diso import DiffDMC
    vertices, indices = DiffDMC(dtype=torch.float32)(-wp.to_torch(volume), None, isovalue=-threshold)
    vertices, indices = vertices.cpu().numpy(), indices.cpu().numpy()
    vertices = vertices * spacing * np.array(volume.shape) + origin
    return trimesh.Trimesh(vertices, indices)
