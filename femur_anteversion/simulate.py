import warnings

import newton
import numpy as np
import pyglet.window.key
import trimesh
import warp as wp

from kernel import winding_volume, diff_dmc


def main(
        dynamic_mesh: trimesh.Trimesh, static_meshes: list, cfg: dict, headless: bool,
        dynamic_q, z_dead, op_side,
):
    builder = newton.ModelBuilder('Z')
    for mesh in (dynamic_mesh,):
        builder.add_shape_mesh(
            mesh=newton.Mesh(mesh.vertices, mesh.faces.flatten()),
            body=builder.add_body(),
            cfg=newton.ModelBuilder.ShapeConfig(**cfg),
        )
    for mesh in static_meshes:
        builder.add_shape_mesh(
            mesh=newton.Mesh(mesh.vertices, mesh.faces.flatten()),
            body=builder.add_body(),
            cfg=newton.ModelBuilder.ShapeConfig(**cfg),
        )

    model = builder.finalize()
    solver = newton.solvers.SolverXPBD(model, iterations=10)
    state_0, state_1 = model.state(), model.state()
    control = model.control()

    viewer = newton.viewer.ViewerGL(headless=headless)
    viewer.show_ui = True
    viewer.show_contacts = True
    viewer.renderer.draw_wireframe = True
    viewer.set_model(model)
    viewer.set_camera(wp.vec3(0, -4, 3), 0, -265)
    viewer.camera.fov = 30

    fps = 100
    frame_dt = 1.0 / fps
    sim_dt = frame_dt / (substeps := 10)
    sim_time = 0.0

    z_frame, z_min = 0, dynamic_q[2]

    for active in range(len(static_meshes)):
        if dynamic_q is None:
            break

        static_q = [(_ - active, 0, 0, 0, 0, 0, 1) for _ in range(len(static_meshes))]
        state_0.body_q.assign((dynamic_q, *static_q))

        last = wp.transform_inverse(wp.transform(*dynamic_q))
        stat = []

        force = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        contacts = model.collide(state_0, rigid_contact_margin=0.001)
        for fid in range(5000):
            while viewer.is_paused():
                viewer.begin_frame(sim_time)
                viewer.end_frame()
            for _ in range(substeps):
                state_0.clear_forces()
                state_1.clear_forces()

                state_0.body_f.assign(force)

                # 限速，避免过大回弹
                body_qd = state_0.body_qd.numpy()
                angular = np.max(np.linalg.norm(np.rad2deg(body_qd[:, :3]), axis=1))
                linear = np.max(np.linalg.norm(body_qd[:, 3:], axis=1))

                while angular > 30 or linear * 1e2 > 1e2:
                    state_0.body_qd.assign(body_qd * 0.0)
                    body_qd = state_0.body_qd.numpy()
                    angular = np.max(np.linalg.norm(np.rad2deg(body_qd[:, :3]), axis=1))
                    linear = np.max(np.linalg.norm(body_qd[:, 3:], axis=1))

                solver.step(state_0, state_1, control, contacts, sim_dt)
                state_0, state_1 = state_1, state_0

                body_q = state_0.body_q.numpy()
                body_q[1:] = static_q
                state_0.body_q.assign(body_q)

                contacts = model.collide(state_0, rigid_contact_margin=0.001)

            sim_time += frame_dt

            if viewer.is_running():
                viewer.begin_frame(sim_time)
                viewer.log_state(state_0)
                viewer.log_contacts(contacts, state_0)
                viewer.end_frame()

            body_q = state_0.body_q.numpy()
            body_qd = state_0.body_qd.numpy()
            angular = np.max(np.linalg.norm(np.rad2deg(body_qd[:, :3]), axis=1))
            linear = np.max(np.linalg.norm(body_qd[:, 3:], axis=1))

            # 失败，假体速度崩溃或位置过低
            if np.isnan(body_qd[0]).any():
                warnings.warn(f'velocity crash {body_qd[0]}')
                dynamic_q = None
                break

            if np.min(body_q[0, 2]) < z_dead:
                warnings.warn(f'position crash {body_q[0]}')
                dynamic_q = None
                break

            contact_count = contacts.rigid_contact_count.numpy()[0]
            if contact_count > 0:
                force = [[50.0 * op_side, 0.0, 0.0, 0.0, 0.0, -50.0]]  # 贴紧前内侧压应力骨小梁
                substeps = 50
                sim_dt = frame_dt / substeps

                # if z_min < body_q[0, 2]:  # 即将稳定时施加力矩(后倾)迫使柄压紧近端后方股骨距
                #     force = [[50.0 * op_side, -50.0, -50.0 * op_side, 0.0, 0.0, -50.0]]

            if last is not None:
                delta = wp.transform(*body_q[0]) * last
                p = wp.transform_get_translation(delta)
                p = wp.length(p) * 1e2
                q = wp.transform_get_rotation(delta)
                _, angle = wp.quat_to_axis_angle(q)
                angle = np.rad2deg(angle)
                stat.append([p, angle])

            last = wp.transform_inverse(body_q[0])

            if len(stat) > 30:
                stat.pop(0)
                mean = np.mean(stat, 0)
            else:
                mean = (np.inf, np.inf)

            z = body_q[0, 2]
            if z_min > z:
                z_min = z
                z_frame = fid

            print(
                f'sim frame {fid} contact {contact_count} change {mean[0]:.3f} mm {mean[1]:.3f} z {fid - z_frame}%')

            # 接触稳定，位置变化 mm，旋转变化 deg
            if contact_count > 0 and fid - z_frame > 99:
                dynamic_q = body_q[0]
                break

    viewer.close()
    return dynamic_q


def simulate_volume(
        dynamic_mesh: trimesh.Trimesh, density: float, region_mesh: trimesh.Trimesh,
        region: wp.array3d(), spacing: float, origin: np.ndarray, bone_threshold: list,
        z_dead, op_side,
        headless: bool,
):
    builder = newton.ModelBuilder('Z', 0)

    # 假体
    _ = winding_volume(dynamic_mesh, spacing, spacing)
    dynamic_mesh = diff_dmc(_[0], _[1], spacing, 0.0)
    dynamic_mesh = max(dynamic_mesh.split(), key=lambda c: c.area)

    dynamic_mesh.vertices *= 1e-2
    builder.add_shape_mesh(
        mesh=(_ := newton.Mesh(dynamic_mesh.vertices, dynamic_mesh.faces.flatten())),
        body=builder.add_body(),
        cfg=newton.ModelBuilder.ShapeConfig(mu=0.0, restitution=0.0),
        key='dynamic',
    )
    dynamic_mass = _.mass

    dynamic_mesh = wp.Mesh(
        wp.array(dynamic_mesh.vertices, wp.vec3),
        wp.array(dynamic_mesh.faces.flatten(), wp.int32),
        support_winding_number=True,
    )

    # 体素场
    builder.add_shape_mesh(
        mesh=newton.Mesh(region_mesh.vertices, region_mesh.faces.flatten()),
        body=-1,
        cfg=newton.ModelBuilder.ShapeConfig(mu=0.0, restitution=0.0, has_shape_collision=False),
        key='static',
    )

    max_dist = wp.length(wp.vec3(region.shape) * spacing * 1e-2)

    # 模拟
    model = builder.finalize()
    solver = newton.solvers.SolverXPBD(model, iterations=1)
    state_0, state_1 = model.state(), model.state()
    control = model.control()

    viewer = newton.viewer.ViewerGL(headless=headless)
    viewer.show_ui = True
    viewer.show_contacts = True
    viewer.renderer.draw_wireframe = True
    viewer.set_model(model)
    # viewer.set_camera(wp.vec3(0, -4, 3), 0, -265)
    viewer.set_camera(wp.vec3(-4, 0, 3), 0, -354)
    viewer.camera.fov = 30

    fps = 100
    frame_dt = 1.0 / fps
    sim_dt = frame_dt / (substeps := 10)

    z_frame, z_min = 0, np.max(region_mesh.vertices[:, 2])

    state_0.body_q.assign((_ := (0, 0, z_min, 0, 0, 0, 1),))

    sim_time = 0.0
    viewer.begin_frame(sim_time)
    viewer.log_state(state_0)
    viewer.end_frame()
    viewer._paused = not headless

    def callback(symbol, _):
        nonlocal fid, density
        if symbol == pyglet.window.key.MINUS:
            density = wp.clamp(density - 10, 10, 500)
            fid = 0
        elif symbol == pyglet.window.key.EQUAL:
            density = wp.clamp(density + 10, 10, 500)
            fid = 0

    viewer.renderer.register_key_release(callback)

    dynamic_q = None
    fid = 0
    while fid < 500:
        fid += 1

        while viewer.is_paused():
            viewer.begin_frame(sim_time)
            viewer.end_frame()

        for sid in range(substeps):
            state_0.clear_forces()
            state_1.clear_forces()

            body_q = state_0.body_q.numpy()
            body_qd = state_0.body_qd.numpy()

            # 失败，假体速度崩溃
            if np.isnan(_ := body_qd[0]).any():
                warnings.warn(f'Invalid velocity {_.tolist()}')
                return None

            # 失败，位置过低
            if np.min(_ := body_q[0, 2]) < z_dead * 1e-2:
                warnings.warn(f'Invalid position {_.tolist()}')
                return None

            # 限速，避免过大回弹
            angular = np.rad2deg(body_qd[0, :3])
            linear = np.array(body_qd[0, 3:]) * 1e2
            angular = np.linalg.norm(angular)
            linear = np.linalg.norm(linear)

            if angular > 30 or linear > 90:
                state_0.body_qd.assign(body_qd * 0.0)

            # if (fid * substeps + sid) % 600 == 0:
            #     state_0.body_qd.assign(body_qd * 0.0)

            # 体素场力，与假体重力平衡
            _ = dynamic_mass * density * 1e3
            body_f = wp.array([wp.spatial_vector(0, 0, 0, 0, 0, -_)], dtype=wp.spatial_vector)

            count = wp.zeros((2,), float)
            wp.launch(voxel_push, region.shape, [
                region, spacing * 1e-2, wp.vec3(origin * 1e-2), bone_threshold[0], bone_threshold[1],
                wp.uint64(dynamic_mesh.id), max_dist,
                wp.transform_inverse(wp.transform(*state_0.body_q.numpy()[0])),
                wp.vec3(model.body_com.numpy()[0]),
                body_f, count,
            ])
            count = [int(_) for _ in count.numpy()]

            # 失败，假体受力崩溃
            if np.isnan(_ := body_f.numpy()).any():
                warnings.warn(f'Invalid force {_.tolist()}')
                return None

            body_f = body_f.numpy() * 1e1
            state_0.body_f.assign(body_f)

            # 统计平衡态
            dynamic_q = body_q[0]
            dynamic_q[:3] *= 1e2

            contacts = model.collide(state_0, rigid_contact_margin=0.0)
            solver.step(state_0, state_1, control, contacts, sim_dt)
            state_0, state_1 = state_1, state_0

        sim_time += frame_dt

        if viewer.is_running():
            viewer.begin_frame(sim_time)
            viewer.log_state(state_0)
            viewer.end_frame()

        print(f'Sim frame {fid} contacts {count} vel {angular:.3f} {linear:.3f} force {body_f.tolist()}')

    viewer.close()
    return dynamic_q, density


@wp.kernel
def voxel_push(
        image: wp.array3d(), spacing: float, origin: wp.vec3, min_threshold: float, max_threshold: float,
        mesh: wp.uint64, max_dist: float,
        body_q_inv: wp.transform, body_com: wp.vec3,
        body_f: wp.array(dtype=wp.spatial_vector), count: wp.array(dtype=float),
):
    i, j, k = wp.tid()

    pixel = float(image[i, j, k])

    # 忽略密度小于骨骼的体素
    if pixel < min_threshold:
        return

    # 查找体素距离最近的网格表面点
    p = wp.vec3(float(i), float(j), float(k)) * spacing + origin
    p = wp.transform_point(body_q_inv, p)
    q = wp.mesh_query_point_sign_winding_number(mesh, p, max_dist)

    if not q.result:
        return

    closest = wp.mesh_eval_position(mesh, q.face, q.u, q.v)

    # 作用力，位体素指向假体，穿透体素反向
    if q.sign > 0:
        f = closest - p
    else:
        f = p - closest

    # 忽略下沉作用的体素
    if f[2] < 0:
        return

    d = wp.length(f) * 1e2
    d_sign = d * q.sign

    # 忽略陷入假体过深的体素
    if d_sign < -2.0:
        return

    # 忽略距离过远的体素
    if d_sign > 2.0:
        return

    if d > 0:
        f /= d

    # 计算力和力矩
    force = (f / (1.0 + d) + wp.vec3(0.0, 0.0, 1e-3)) * pixel
    torque = wp.cross(p - body_com, force)

    # if d_sign > 0:
    #     force *= 0.0

    if d_sign > 0:
        body_f[0] += wp.spatial_vector(torque, force)
        count[1] += 1.0
    else:
        count[0] += 1.0
