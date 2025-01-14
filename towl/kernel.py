import warp as wp

wp.config.verbose = False
wp.config.print_launches = False
wp.init()

pixel_type = wp.float32


@wp.kernel
def volume_sample(volume: wp.uint64, image: wp.array(dtype=pixel_type, ndim=3),
                  offset_uvw: wp.vec3, spacing_uvw: wp.vec3):
    i, j, k = wp.tid()
    ijk = wp.vec3(float(i), float(j), float(k))
    uvw = wp.cw_mul(spacing_uvw, ijk) + offset_uvw
    pixel = wp.volume_sample_f(volume, uvw, wp.Volume.LINEAR)
    image[i, j, k] = pixel_type(pixel)


@wp.kernel
def volume_parallel_xray(
        volume: wp.uint64, image: wp.array(dtype=wp.uint8, ndim=2),
        volume_origin: wp.vec3, volume_spacing: wp.vec3,
        image_origin: wp.vec3, image_iso_spacing: float,
        image_x_axis: wp.vec3, image_y_axis: wp.vec3, image_z_axis: wp.vec3,
        image_z_depth: float,
        threshold_min: float,
        window_min: float, window_max: float,
):
    i, j = wp.tid()

    ray_position = image_origin
    ray_position += float(i) * image_iso_spacing * image_x_axis
    ray_position += float(j) * image_iso_spacing * image_y_axis

    ray_step = image_iso_spacing * image_z_axis
    pixel = float(0)

    n = int(image_z_depth / image_iso_spacing)
    pixel_n = float(0)

    for k in range(n):
        ray_position += ray_step

        uvw = wp.cw_div(ray_position - volume_origin, volume_spacing)
        p = wp.volume_sample_f(volume, uvw, wp.Volume.LINEAR)

        if threshold_min <= p:
            pixel += p
            pixel_n += 1.0

    if pixel_n > 0:
        pixel /= pixel_n

    pixel = wp.round((pixel - window_min) / (window_max - window_min) * 255.0)
    pixel = wp.clamp(pixel, 0.0, 255.0)

    image[i, j] = wp.uint8(pixel)
