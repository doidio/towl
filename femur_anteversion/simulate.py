import warnings
from pathlib import Path

import newton
import numpy as np
import trimesh
import warp as wp
from imgui_bundle import imgui
from scipy.spatial.transform import Rotation


def simulate_volume(
        prothesis_meshes: list[trimesh.Trimesh], prothesis_files: list[Path], body_i: int,
        reaming: list[float], region_meshes: list[trimesh.Trimesh],
        region: wp.array3d(), spacing: float, origin: np.ndarray, bone_threshold: list,
        z_dead, op_side, title: str, headless: bool,
):
    builder = newton.ModelBuilder('Z', 0)

    # 假体
    body_meshes = []
    for mesh in prothesis_meshes:
        builder.add_shape_mesh(
            mesh=(_ := newton.Mesh(mesh.vertices, mesh.faces.flatten())),
            body=builder.add_body(),
            cfg=newton.ModelBuilder.ShapeConfig(mu=0.0, restitution=0.0),
            key='dynamic',
        )

        body_meshes.append(wp.Mesh(
            wp.array(mesh.vertices, wp.vec3),
            wp.array(mesh.faces.flatten(), wp.int32),
            support_winding_number=True,
        ))
    body_n = len(body_meshes)

    # 骨骼
    for _ in region_meshes:
        builder.add_shape_mesh(
            mesh=newton.Mesh(_.vertices, _.faces.flatten()),
            body=-1,
            cfg=newton.ModelBuilder.ShapeConfig(mu=0.0, restitution=0.0),
            key='static',
        )

    max_dist = wp.length(wp.vec3(region.shape) * spacing * 1e-2)

    # 模拟
    model = builder.finalize()
    solver = newton.solvers.SolverXPBD(model, iterations=1)
    state_0, state_1 = model.state(), model.state()
    control = model.control()

    viewer = newton.viewer.ViewerGL(headless=headless)
    viewer.renderer.set_title(title)
    viewer.show_ui = True
    viewer.show_contacts = True
    viewer.renderer.draw_wireframe = True
    # viewer.renderer.draw_shadows = False
    viewer.set_model(model)
    # viewer.set_camera(wp.vec3(0, -5, 3), 0, 90)
    viewer.set_camera(wp.vec3(-5, 0, 3.5), 0, 0)
    viewer.camera.fov = 30

    fps = 100
    frame_dt = 1.0 / fps
    sim_dt = frame_dt / (substeps := 10)

    z_frame, z_init = 0, np.max([_.bounds[1][2] for _ in region_meshes])

    state_0.body_q.assign(_ := [[_ - body_i, _ - body_i, z_init, 0, 0, 0, 1] for _ in range(body_n)])
    prothesis_q = np.array(_[body_i])

    sim_time = 0.0
    viewer.begin_frame(sim_time)
    viewer.log_state(state_0)
    viewer.end_frame()
    viewer._paused = not headless

    user_apply = False

    def panel(_):
        nonlocal user_apply, body_i, body_n, sim_time, fid, depth_max, reaming, contacts
        imgui.separator()
        imgui.text(f'frame: {fid} / {total_frame}')

        imgui.text(f'prothesis: {prothesis_files[body_i].name}')
        if imgui.button('<'):
            q, qd = state_0.body_q.numpy(), state_0.body_qd.numpy()
            q, qd = q[body_i], qd[body_i]
            body_i = (body_i - 1) % body_n
            q = [q if _ == body_i else [_ - body_i, _ - body_i, z_init, 0, 0, 0, 1] for _ in range(body_n)]
            qd = [qd if _ == body_i else [0, 0, 0, 0, 0, 0] for _ in range(body_n)]
            state_0.body_q.assign(q)
            state_0.body_qd.assign(qd)
            fid, depth_max = 0, 0
        imgui.same_line()
        if imgui.button('>'):
            q, qd = state_0.body_q.numpy(), state_0.body_qd.numpy()
            q, qd = q[body_i], qd[body_i]
            body_i = (body_i + 1) % body_n
            q = [q if _ == body_i else [_ - body_i, _ - body_i, z_init, 0, 0, 0, 1] for _ in range(body_n)]
            qd = [qd if _ == body_i else [0, 0, 0, 0, 0, 0] for _ in range(body_n)]
            state_0.body_q.assign(q)
            state_0.body_qd.assign(qd)
            fid, depth_max = 0, 0

        imgui.text(f'reaming: {reaming[0]} {reaming[1]} {reaming[2]}')
        if imgui.button('A'):
            reaming[0] = int(np.clip(reaming[0] - 50, -2000, 2000))
            fid, depth_max = 0, 0
        imgui.same_line()
        if imgui.button('L'):
            reaming[1] = int(np.clip(reaming[1] - 50, -5000, 5000))
            fid, depth_max = 0, 0
        imgui.same_line()
        if imgui.button('S'):
            reaming[2] = int(np.clip(reaming[2] - 50, 0, 5000))
            fid, depth_max = 0, 0

        if imgui.button('P'):
            reaming[0] = int(np.clip(reaming[0] + 50, -5000, 5000))
            fid, depth_max = 0, 0
        imgui.same_line()
        if imgui.button('M'):
            reaming[1] = int(np.clip(reaming[1] + 50, -5000, 5000))
            fid, depth_max = 0, 0
        imgui.same_line()
        if imgui.button('I'):
            reaming[2] = int(np.clip(reaming[2] + 50, 0, 5000))
            fid, depth_max = 0, 0

        imgui.text(f'removed: {contacts[0]}')
        imgui.text(f'covered: {contacts[1]}')

        r = Rotation.from_quat(wp.transform_get_rotation(wp.transform(*prothesis_q)), scalar_first=False)
        r = r.as_euler('ZYX', degrees=True)
        imgui.text(f'depth: {(z_init - prothesis_q[2]) * 1e2:.3f} / {depth_max * 1e2:.3f} mm')
        imgui.text(f'torsion: {r[0] * op_side:.3f} deg')
        if imgui.button('replay'):
            state_0.body_q.assign([[_ - body_i, _ - body_i, z_init, 0, 0, 0, 1] for _ in range(body_n)])
            fid, depth_max = 0, 0
        imgui.same_line()
        if imgui.button('apply & exit'):
            user_apply = True

    viewer.register_ui_callback(panel, position='stats')

    fid, depth_max, contacts = 0, 0, (0, 0)
    while fid < (total_frame := 10000):
        while viewer.is_paused():
            viewer.begin_frame(sim_time)
            viewer.log_state(state_0)
            viewer.end_frame()

            if not viewer.is_running():
                break
        if not viewer.is_running():
            break

        if user_apply:
            viewer.close()
            prothesis_q[:3] *= 1e2
            return prothesis_q, prothesis_files[body_i], prothesis_meshes[body_i], reaming, contacts

        fid += 1

        for sid in range(substeps):
            state_0.clear_forces()
            state_1.clear_forces()

            body_q = state_0.body_q.numpy()
            body_qd = state_0.body_qd.numpy()

            # 失败，假体速度崩溃
            if np.isnan(_ := body_qd[body_i]).any():
                warnings.warn(f'Invalid velocity {_.tolist()}')
                return None

            # 失败，位置过低
            if np.min(_ := body_q[body_i, 2]) < z_dead * 1e-2:
                warnings.warn(f'Invalid position {_.tolist()}')
                return None

            if deeper := (depth_max < (_ := z_init - body_q[body_i, 2])):
                depth_max = _

            # 限速，避免过大回弹
            angular = np.rad2deg(body_qd[body_i, :3])
            linear = np.array(body_qd[body_i, 3:]) * 1e2
            angular = np.linalg.norm(angular)
            linear = np.linalg.norm(linear)

            if angular > 30 or linear > 90:
                state_0.body_qd.assign(body_qd * 0.0)

            # 假体预载力
            body_f = np.array([(0.0,) * 6] * body_n)
            if contacts[1] > 0 and not deeper:  # 到底后施加末端力矩
                body_f[body_i] = [reaming[0] * op_side, reaming[1], 0, 0, 0, 0]
            body_f[body_i, 5] -= reaming[2]  # 重力

            body_f = wp.array([wp.spatial_vector(*_) for _ in body_f], dtype=wp.spatial_vector)

            # 体素场支反力，与假体预载力平衡
            _ = wp.zeros((2,), float)
            wp.launch(voxel_force, region.shape, [
                region, spacing * 1e-2, wp.vec3(origin * 1e-2), bone_threshold[0],
                wp.uint64(body_meshes[body_i].id), max_dist,
                wp.transform_inverse(wp.transform(*state_0.body_q.numpy()[body_i])),
                wp.vec3(model.body_com.numpy()[body_i]),
                body_f, body_i, _,
            ])
            contacts = [int(_) for _ in _.numpy()]  # 穿透与接触体素数量

            # 失败，假体合力崩溃
            if np.isnan(_ := body_f.numpy()).any():
                warnings.warn(f'Invalid force {_.tolist()}')
                return None

            # 放大假体合力，减少振荡
            body_f = body_f.numpy() * 1e1
            state_0.body_f.assign(body_f)

            # 假体位姿
            prothesis_q = body_q[body_i]

            solver.step(state_0, state_1, control, None, sim_dt)  # noqa
            state_0, state_1 = state_1, state_0

        sim_time += frame_dt

        if viewer.is_running():
            viewer.begin_frame(sim_time)
            viewer.log_state(state_0)
            viewer.end_frame()

    if viewer.is_running():
        viewer.close()
    return None


@wp.kernel
def voxel_force(
        image: wp.array3d(), spacing: float, origin: wp.vec3, min_hu: float,
        mesh: wp.uint64, max_dist: float,
        body_q_inv: wp.transform, body_com: wp.vec3,
        body_f: wp.array(dtype=wp.spatial_vector), body_i: int,
        count: wp.array(dtype=float),
):
    i, j, k = wp.tid()

    hu = float(image[i, j, k])

    if hu < min_hu:
        return

    # 查找体素距离最近的网格表面点
    p = wp.vec3(float(i), float(j), float(k)) * spacing + origin
    p = wp.transform_point(body_q_inv, p)
    q = wp.mesh_query_point_sign_winding_number(mesh, p, max_dist)

    if not q.result:  # noqa
        return

    closest = wp.mesh_eval_position(mesh, q.face, q.u, q.v)  # noqa

    if q.sign < 0:  # noqa
        f = p - closest
        count[0] += 1.0  # 穿透体素
    else:
        f = closest - p

    # 忽略下沉作用的体素
    if f[2] < 0:
        return

    # 忽略穿透过深和距离过远的体素
    d = wp.length(f) * 1e2

    if d > 2.0:
        return

    if q.sign >= 0:  # noqa
        count[1] += 1.0  # 接触体素

    # 接触力
    if d > 0:
        force = f / d / (1.0 + d)
    else:
        force = wp.vec3()

    # 上浮力
    # force[2] += 1e-3

    # 骨密度，QCT(mg/cm³) = HU · 0.7 + 17.8
    force *= hu * 0.7 + 17.8

    # 力矩 = 力 × 质心
    torque = wp.cross(p - body_com, force)

    body_f[body_i] += wp.spatial_vector(torque, force)
