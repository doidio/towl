import warp as wp


@wp.kernel
def region_sample_F_W(
        volume: wp.uint64, volume_spacing: wp.vec3, volume_origin: wp.vec3,
        array: wp.array(ndim=3), spacing: wp.vec3, origin: wp.vec3, modality_threshold: float,
):
    i, j, k = wp.tid()

    p = wp.cw_mul(spacing, wp.vec3(wp.float32(i), wp.float32(j), wp.float32(k)))
    p += origin - volume_origin

    uvw = wp.cw_div(p, volume_spacing)
    pixel = wp.volume_sample_f(volume, uvw, wp.Volume.LINEAR)

    if array[i, j, k] == 0.0 or pixel == 0.0:
        array[i, j, k] = wp.max(array[i, j, k], array.dtype(pixel))

    # 差异较小，FW模态相同，选更亮以弱化接缝
    elif wp.abs(array[i, j, k] - pixel) < modality_threshold:
        array[i, j, k] = wp.max(array[i, j, k], array.dtype(pixel))

    # 差异较大，FW模态交换，选一侧以避免丢失F
    else:
        pass


@wp.kernel
def region_sample_in_opp(
        volume: wp.uint64, volume_spacing: wp.vec3, volume_origin: wp.vec3,
        array: wp.array(ndim=3), spacing: wp.vec3, origin: wp.vec3,
):
    i, j, k = wp.tid()

    p = wp.cw_mul(spacing, wp.vec3(wp.float32(i), wp.float32(j), wp.float32(k)))
    p += origin - volume_origin

    uvw = wp.cw_div(p, volume_spacing)
    pixel = wp.volume_sample_f(volume, uvw, wp.Volume.LINEAR)

    array[i, j, k] = wp.max(array[i, j, k], array.dtype(pixel))
