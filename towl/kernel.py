import warp as wp

wp.config.verbose = False
wp.config.print_launches = False
wp.init()


@wp.kernel
def volume_sample_positions(
        volume: wp.uint64, volume_origin: wp.vec3, volume_spacing: wp.vec3,
        positions: wp.array(dtype=wp.vec3), values: wp.array(dtype=float),
):
    i = wp.tid()
    uvw = wp.cw_div(positions[i] - volume_origin, volume_spacing)
    values[i] = wp.volume_sample_f(volume, uvw, wp.Volume.LINEAR)


@wp.kernel
def volume_query_rays(
        volume: wp.uint64, volume_origin: wp.vec3, volume_spacing: wp.vec3,
        ray_starts: wp.array(dtype=wp.vec3), ray_directions: wp.array(dtype=wp.vec3),
        ray_step_length: float, ray_max_length: float, ray_stop_threshold: float,
        ray_stops: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    n = int(ray_max_length / ray_step_length)
    for _ in range(n):
        p = ray_starts[i] + ray_directions[i] * ray_step_length * float(_)
        uvw = wp.cw_div(p - volume_origin, volume_spacing)
        if wp.volume_sample_f(volume, uvw, wp.Volume.LINEAR) > ray_stop_threshold:
            ray_stops[i] = p
            return


@wp.kernel
def volume_ray_parallel(
        volume: wp.uint64, volume_origin: wp.vec3, volume_spacing: wp.vec3,
        image: wp.array(dtype=wp.uint8, ndim=2), image_origin: wp.vec3, image_iso_spacing: float,
        image_x_axis: wp.vec3, image_y_axis: wp.vec3, image_z_axis: wp.vec3,
        image_z_depth: wp.float32,
        threshold_min: float, window_min: float, window_max: float,
):
    i, j = wp.tid()

    ray_start = image_origin
    ray_start += float(i) * image_iso_spacing * image_x_axis
    ray_start += float(j) * image_iso_spacing * image_y_axis

    step = image_iso_spacing * image_z_axis
    pixel = float(0)

    n = int(image_z_depth / image_iso_spacing)
    pixel_n = float(0)

    position = ray_start
    for k in range(n):
        position += step

        uvw = wp.cw_div(position - volume_origin, volume_spacing)
        p = wp.volume_sample_f(volume, uvw, wp.Volume.LINEAR)

        if threshold_min <= p:
            pixel += p
            pixel_n += 1.0

    if pixel_n > 0:
        pixel /= pixel_n

    pixel = wp.round((pixel - window_min) / (window_max - window_min) * 255.0)
    pixel = wp.clamp(pixel, 0.0, 255.0)

    image[i, j] = wp.uint8(pixel)


@wp.kernel
def volume_slice(
        volume: wp.uint64, volume_origin: wp.vec3, volume_spacing: wp.vec3,
        image: wp.array(dtype=wp.uint8, ndim=2), image_origin: wp.vec3, image_iso_spacing: float,
        image_x_axis: wp.vec3, image_y_axis: wp.vec3,
        window_min: float, window_max: float,
):
    i, j = wp.tid()

    position = image_origin
    position += float(i) * image_iso_spacing * image_x_axis
    position += float(j) * image_iso_spacing * image_y_axis

    uvw = wp.cw_div(position - volume_origin, volume_spacing)
    pixel = wp.volume_sample_f(volume, uvw, wp.Volume.LINEAR)

    pixel = wp.round((pixel - window_min) / (window_max - window_min) * 255.0)
    pixel = wp.clamp(pixel, 0.0, 255.0)

    image[i, j] = wp.uint8(pixel)


@wp.kernel
def mesh_ray_parallel(
        mesh: wp.uint64, base_color: wp.vec3,
        image: wp.array(dtype=wp.vec3, ndim=2), image_origin: wp.vec3, image_iso_spacing: float,
        image_x_axis: wp.vec3, image_y_axis: wp.vec3, image_z_axis: wp.vec3,
        image_z_depth: wp.float32,
):
    i, j = wp.tid()

    ray_start = image_origin
    ray_start += float(i) * image_iso_spacing * image_x_axis
    ray_start += float(j) * image_iso_spacing * image_y_axis

    q = wp.mesh_query_ray(
        mesh, ray_start, image_z_axis, image_z_depth,
    )
    if q.result:
        # 物理光照参数
        view_dir = -wp.normalize(image_z_axis)  # 视线方向
        normal = wp.normalize(q.normal)

        metallic_factor = 0.3

        # 基础材质参数
        ambient_intensity = 0.1
        ambient = ambient_intensity * base_color
        shininess = 64  # 高光锐度
        specular_strength = 0.8 * metallic_factor

        # 光照计算
        light_dir = wp.normalize(-image_z_axis)

        # 漫反射
        diffuse = wp.max(wp.dot(normal, light_dir), 0.0)

        # Blinn-Phong 高光
        H = wp.normalize(light_dir + view_dir)
        NdotH = wp.max(wp.dot(normal, H), 0.0)
        specular = specular_strength * wp.pow(NdotH, float(shininess))

        # 菲涅尔效应增强金属感
        fresnel = (1.0 - wp.max(wp.dot(view_dir, normal), 0.0)) ** 5.0
        specular += fresnel * metallic_factor

        # 漫反射分量（应用基础颜色和金属度）
        diffuse_term = (1.0 - metallic_factor) * diffuse * base_color

        # 高光分量（默认白色高光）
        specular_color = wp.vec3(specular, specular, specular)

        # 组合光照分量
        final_color = ambient + diffuse_term + specular_color
        final_color = wp.vec3(
            wp.clamp(final_color[0], 0.0, 1.0),
            wp.clamp(final_color[1], 0.0, 1.0),
            wp.clamp(final_color[2], 0.0, 1.0)
        )

        image[i, j] = final_color * 255.0


@wp.kernel
def mesh_slice(
        mesh: wp.uint64, mesh_max_dist: wp.float32, base_color: wp.vec3, opacity: float,
        image: wp.array(dtype=wp.vec3, ndim=2), image_origin: wp.vec3, image_iso_spacing: float,
        image_x_axis: wp.vec3, image_y_axis: wp.vec3,
):
    i, j = wp.tid()

    position = image_origin
    position += float(i) * image_iso_spacing * image_x_axis
    position += float(j) * image_iso_spacing * image_y_axis

    opacity = wp.clamp(opacity, 0.0, 1.0)

    if mesh > wp.uint64(0):
        if wp.mesh_query_point_sign_winding_number(
                mesh, position, mesh_max_dist, wp.float32(2.0), wp.float32(0.5),
        ).sign < 0:
            image[i, j] = opacity * base_color * 255.0 + (1.0 - opacity) * image[i, j]
