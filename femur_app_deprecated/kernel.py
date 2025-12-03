import numpy as np
import warp as wp

wp.config.verbose = False
wp.config.print_launches = False
wp.init()


def volume_dual_marching_cubes(volume: wp.array(dtype=wp.float32, ndim=3), spacing: float | np.ndarray,
                               origin: np.ndarray,
                               threshold: float):
    import torch
    from diso import DiffDMC
    vertices, indices = DiffDMC(dtype=torch.float32)(-wp.to_torch(volume), None, isovalue=-threshold)
    vertices, indices = vertices.cpu().numpy(), indices.cpu().numpy()
    vertices = vertices * spacing * np.array(volume.shape) + origin
    return vertices, indices


def volume_marching_cubes(volume: wp.array(dtype=wp.float32, ndim=3), spacing: float | np.ndarray, origin: np.ndarray,
                          threshold: float):
    max_cubes = volume.shape[0] * volume.shape[1] * volume.shape[2]
    mc = wp.MarchingCubes(volume.shape[0], volume.shape[1], volume.shape[2], max_cubes, max_cubes)
    mc.surface(volume, threshold)
    vertices = mc.verts.numpy() * spacing + origin
    indices = mc.indices.numpy()
    return vertices, indices


@wp.kernel
def transform_body(
        body_q: wp.array(dtype=wp.transform), body: int,
        in_points: wp.array(dtype=wp.vec3), out_points: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    out_points[i] = wp.transform_point(body_q[body], in_points[i])


@wp.kernel
def volume_sample(
        volume: wp.uint64, volume_origin: wp.vec3, volume_spacing: wp.vec3,
        sample: wp.array(dtype=wp.float32, ndim=3), sample_origin: wp.vec3, sample_spacing: float,
        sample_x_axis: wp.vec3, sample_y_axis: wp.vec3, sample_z_axis: wp.vec3,
        skip_bounds: bool,
):
    i, j, k = wp.tid()

    if skip_bounds:
        if i == 0 or i == sample.shape[0] - 1:
            return
        if j == 0 or j == sample.shape[1] - 1:
            return
        if k == 0 or k == sample.shape[2] - 1:
            return

    p = sample_origin
    p += float(i) * sample_spacing * sample_x_axis
    p += float(j) * sample_spacing * sample_y_axis
    p += float(k) * sample_spacing * sample_z_axis

    uvw = wp.cw_div(p - volume_origin, volume_spacing)
    pixel = wp.volume_sample_f(volume, uvw, wp.Volume.LINEAR)
    sample[i, j, k] = wp.float32(pixel)


@wp.kernel
def volume_sample_grad(
        volume: wp.uint64, volume_origin: wp.vec3, volume_spacing: wp.vec3,
        sample: wp.array(dtype=wp.float32, ndim=3), sample_grad: wp.array(dtype=wp.vec3, ndim=3),
        sample_origin: wp.vec3, sample_spacing: float,
        sample_x_axis: wp.vec3, sample_y_axis: wp.vec3, sample_z_axis: wp.vec3,
        skip_bounds: bool,
):
    i, j, k = wp.tid()

    if skip_bounds:
        if i == 0 or i == sample.shape[0] - 1:
            return
        if j == 0 or j == sample.shape[1] - 1:
            return
        if k == 0 or k == sample.shape[2] - 1:
            return

    p = sample_origin
    p += float(i) * sample_spacing * sample_x_axis
    p += float(j) * sample_spacing * sample_y_axis
    p += float(k) * sample_spacing * sample_z_axis

    uvw = wp.cw_div(p - volume_origin, volume_spacing)
    grad = wp.vec3()
    pixel = wp.volume_sample_grad_f(volume, uvw, wp.Volume.LINEAR, grad)
    sample[i, j, k] = wp.float32(pixel)
    sample_grad[i, j, k] = grad


@wp.kernel
def volume_clip_plane(
        volume_array: wp.array(dtype=wp.float32, ndim=3), volume_origin: wp.vec3, volume_spacing: float,
        plane_center: wp.vec3, plane_normal: wp.vec3, threshold: float,
):
    i, j, k = wp.tid()

    p = volume_origin + wp.vec3(float(i), float(j), float(k)) * volume_spacing

    d = wp.dot(wp.normalize(plane_normal), p - plane_center)

    if d <= 0.0:
        volume_array[i, j, k] = d + threshold
    elif d < volume_spacing * 1.5 and volume_array[i, j, k] > threshold:
        volume_array[i, j, k] = d + threshold


@wp.kernel
def volume_query_rays(
        volume: wp.uint64, volume_origin: wp.vec3, volume_spacing: wp.vec3,
        ray_starts: wp.array(dtype=wp.vec3), ray_directions: wp.array(dtype=wp.vec3),
        ray_step_length: float, ray_max_length: float, ray_stop_threshold: float,
        ray_stops: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    n = int(ray_max_length / ray_step_length)
    p = ray_starts[i]
    for _ in range(n):
        last_p = p
        p += ray_directions[i] * ray_step_length
        uvw = wp.cw_div(p - volume_origin, volume_spacing)
        if wp.volume_sample_f(volume, uvw, wp.Volume.LINEAR) > ray_stop_threshold:
            ray_stops[i] = last_p
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


@wp.kernel
def femoral_prothesis_collide(
        volume_array: wp.array(dtype=wp.float32, ndim=3), volume_origin: wp.vec3, volume_spacing: wp.vec3,
        bound_threshold: float,
        mesh: wp.uint64, mesh_com: wp.vec3,
        body_f: wp.array(dtype=wp.spatial_vector), body: int,
        neck_plane_center: wp.vec3, neck_plane_outer: wp.vec3,
        canal_plane_center: wp.vec3, canal_plane_outer: wp.vec3,
):
    i, j, k = wp.tid()
    position = wp.cw_mul(wp.vec3(float(i), float(j), float(k)), volume_spacing) + volume_origin

    if wp.dot(position - neck_plane_center, neck_plane_outer) > 0:  # 颈口外力
        f = wp.normalize(neck_plane_outer) * 9.8e2
        q = wp.cross(position - mesh_com, f)
        body_f[body] += wp.spatial_vector(q[0], q[1], q[2], f[0], f[1], f[2])
    # elif wp.dot(position - canal_plane_center, canal_plane_outer) < 0:  # 髓内碰撞
    #     if volume_array[i, j, k] > bound_threshold:
    #         query = wp.mesh_query_point_sign_normal(mesh, position, wp.float32(1e6), wp.float32(1e-3))
    #
    #         if query.sign < 0:
    #             force_position = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
    #             f = (position - force_position) * 0.001
    #             q = wp.cross(force_position - mesh_com, f)
    #             body_f[body] += wp.spatial_vector(q[0], q[1], q[2], f[0], f[1], f[2])


# @wp.kernel
# def femoral_prothesis_collide_(
#         volume_array: wp.array(dtype=wp.float32, ndim=3), volume_origin: wp.vec3, volume_spacing: wp.vec3,
#         bound_threshold: float,
#         body_q: wp.array(dtype=wp.transform), body_f: wp.array(dtype=wp.spatial_vector),
#         ps_body: int, ps_com: wp.vec3, ps_gravity: float, ps_mesh: wp.uint64,
#         neck_plane_center: wp.vec3, neck_plane_outer: wp.vec3,
#         canal_plane_center: wp.vec3, canal_plane_outer: wp.vec3,
# ):
#     i = wp.tid()
#
#     c = wp.transform_point(body_q[ps_body], ps_vertices[i])
#     ps_com = wp.transform_point(body_q[ps_body], ps_com)
#
#     if wp.dot(c - neck_plane_center, neck_plane_outer) > 0:  # 颈口外力
#         f = wp.normalize(neck_plane_outer) * ps_gravity
#         q = wp.cross(c - ps_com, f)
#         body_f[ps_body] += wp.spatial_vector(q[0], q[1], q[2], f[0], f[1], f[2])
#     elif wp.dot(c - canal_plane_center, canal_plane_outer) < 0:  # 髓内碰撞
#         uvw = wp.cw_div(c - volume_origin, volume_spacing)
#         grad = wp.vec3()
#         pixel = wp.volume_sample_grad_f(volume, uvw, wp.Volume.LINEAR, grad)
#
#         if pixel > bound_threshold:
#             f = wp.normalize(-grad) * ps_gravity
#             q = wp.cross(c - ps_com, f)
#             body_f[ps_body] += wp.spatial_vector(q[0], q[1], q[2], f[0], f[1], f[2])


@wp.kernel
def prothesis_render(
        volume: wp.uint64, volume_origin: wp.vec3, volume_spacing: wp.vec3, bound_threshold: float,
        image: wp.array(dtype=wp.vec3, ndim=2), image_origin: wp.vec3, image_iso_spacing: float,
        image_depth: wp.float32,
        image_x_axis: wp.vec3, image_y_axis: wp.vec3, image_z_axis: wp.vec3,
        threshold_min: float, window_min: float, window_max: float,
        mesh: wp.uint64, base_color: wp.vec3,
        cut_plane_center: wp.vec3, cut_plane_normal: wp.vec3,
):
    i, j = wp.tid()

    # volume
    ray_start = image_origin
    ray_start += float(i) * image_iso_spacing * image_x_axis
    ray_start += float(j) * image_iso_spacing * image_y_axis

    step = image_iso_spacing * image_z_axis
    pixel = float(0)

    n = int(image_depth / image_iso_spacing)
    pixel_n = float(0)
    contact_a_n = float(0)
    contact_p_n = float(0)

    cut_plane_normal = wp.normalize(cut_plane_normal)

    position = ray_start
    for k in range(n):
        position += step

        uvw = wp.cw_div(position - volume_origin, volume_spacing)
        p = wp.volume_sample_f(volume, uvw, wp.Volume.LINEAR)

        if threshold_min <= p:
            pixel_n += 1.0

            d = wp.dot(cut_plane_normal, position - cut_plane_center)
            if d < 0.0:
                pixel += p

                if p > bound_threshold:
                    query = wp.mesh_query_point_sign_normal(mesh, position, wp.float32(1e6), wp.float32(1e-3))
                    closest = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
                    normal = wp.mesh_eval_face_normal(mesh, query.face)
                    if wp.length(position - closest) < image_iso_spacing * 2.0:
                        if wp.dot(normal, image_z_axis) < 0.0:
                            contact_a_n += 1.0
                        else:
                            contact_p_n += 1.0

    if pixel_n > 0:
        pixel /= pixel_n

    pixel = wp.round((pixel - window_min) / (window_max - window_min) * 255.0)
    pixel = wp.clamp(pixel, 0.0, 255.0)

    if wp.max(contact_a_n, contact_p_n) > 0.0:
        if contact_a_n > contact_p_n:
            image[i, j] = wp.vec3(255.0, 0.0, 0.0)
        else:
            image[i, j] = wp.vec3(127.0, 0.0, 0.0)
    else:
        image[i, j] = wp.vec3(pixel)

    # mesh
    q = wp.mesh_query_ray(
        mesh, ray_start, image_z_axis, image_depth,
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

        if wp.max(contact_a_n, contact_p_n) > 0.0:
            image[i, j] = 0.5 * image[i, j] + 0.5 * final_color * 255.0
        else:
            image[i, j] = final_color * 255.0
