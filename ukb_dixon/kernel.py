import warp as wp


@wp.kernel
def region_sample(
        volume: wp.uint64, volume_spacing: wp.vec3, volume_origin: wp.vec3,
        array: wp.array(ndim=3), spacing: wp.vec3, origin: wp.vec3,
):
    i, j, k = wp.tid()

    p = wp.cw_mul(spacing, wp.vec3(wp.float32(i), wp.float32(j), wp.float32(k)))
    p += origin - volume_origin

    uvw = wp.cw_div(p, volume_spacing)
    pixel = wp.volume_sample_f(volume, uvw, wp.Volume.LINEAR)

    array[i, j, k] = wp.max(array[i, j, k], array.dtype(pixel))
