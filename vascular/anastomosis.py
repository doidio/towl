# CUDA driver & Toolkit & Microsoft Visual C++
# pip install vtk itk pyglet trimesh rtree scikit-learn
# pip install torch --index-url https://download.pytorch.org/whl/cu129
# pip install wheel diso
# pip install -U --pre warp-lang --extra-index-url=https://pypi.nvidia.com/
# pip uninstall newton newton-physics -y
# pip install -U git+https://github.com/newton-physics/newton.git@6950f379de428b368141916f17ee5d6d432e6d98

import argparse
import json
import locale
from pathlib import Path

import newton
import numpy as np
import pyvista as pv
import tomlkit
import warp as wp
from vtkmodules.vtkCommonMath import vtkMatrix4x4
from vtkmodules.vtkCommonTransforms import vtkTransform

from kernel import centerline_sdf, diff_dmc, tri_poly, sdf_smooth, FluidSPH, fluid_pressure, centerline_sdf_end

locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')

inactive_centerline = {}

active_centerline = {}
active_radius = {}

part_transform = {}
part_matrix = {}


def main(cfg_path: str):
    cfg_path = Path(cfg_path)

    _ = cfg_path.read_text('utf-8')
    cfg = tomlkit.loads(_)

    c = 2
    pl = pv.Plotter(title='血管吻合血流动力学模拟', shape=(1, c), border_width=0, window_size=(800 * c, 800))

    camera = pl.camera
    for _ in range(c):
        pl.subplot(0, _)
        pl.camera = camera
        pl.enable_parallel_projection()
        pl.add_camera_orientation_widget()
        pl.enable_hidden_line_removal()
        pl.enable_depth_peeling()
        pl.camera_position = 'xz'

    # 读取SlicerVMTK模块导出的血管中心线数据
    for part in ('髂内动脉', '肾动脉'):
        cfg[part] = cfg.get(part, {})

        inactive_centerline[part] = []

        centerline = Path(cfg[part]['中心线'])

        bounds = []
        for f in centerline.rglob('*.json'):
            segment = json.loads(f.read_text(encoding='utf-8'))
            centers = segment['markups'][0]['controlPoints']
            radius_raw = [0] * len(centers)
            for _ in segment['markups'][0]['measurements']:
                if _['name'] == 'Radius':
                    radius_raw = _['controlPointValues']

            assert len(centers) == len(radius_raw)

            centers = [_['position'] for _ in centers]

            r_min = radius_raw[0]
            radius = []
            for p, r in zip(centers, radius_raw):
                if r_min > r:
                    r_min = r
                radius.append(r_min)
                bounds.append(p)

            # 提取吻合分支和段落
            if f.name == cfg[part]['吻合分支']:
                begin, end = cfg[part]['吻合段落']
                assert 0 <= begin < end <= 1.0

                begin, end = int(np.floor(len(centers) * begin)), int(np.ceil(len(centers) * end))

                active_centerline[part] = centers[begin:end]
                active_radius[part] = radius[begin:end]

            if len(centers):
                inactive_centerline[part].append(centers)

    # 肾动脉始端连接髂内动脉末端，中心线方向对齐
    c0 = (
        [active_centerline['髂内动脉'][-1][_] for _ in range(3)],
        [active_centerline['肾动脉'][0][_] for _ in range(3)],
    )
    c1 = (
        [active_centerline['髂内动脉'][-1][_] - active_centerline['髂内动脉'][-2][_] for _ in range(3)],
        [active_centerline['肾动脉'][1][_] - active_centerline['肾动脉'][0][_] for _ in range(3)],
    )

    m = vtkMatrix4x4()
    m.SetData((
        1.0, 0.0, 0.0, -c0[1][0],
        0.0, 1.0, 0.0, -c0[1][1],
        0.0, 0.0, 1.0, -c0[1][2],
        0.0, 0.0, 0.0, 1.0,
    ))

    _ = np.identity(4)
    _[:3, :3] = np.array(wp.quat_to_matrix(wp.quat_between_vectors(wp.vec3(c1[1]), wp.vec3(c1[0])))).reshape((3, 3))

    m_ = vtkMatrix4x4()
    m_.SetData(_.flatten())

    m.Multiply4x4(m_, m, m)

    m_ = vtkMatrix4x4()
    m_.SetData((
        1.0, 0.0, 0.0, c0[0][0],
        0.0, 1.0, 0.0, c0[0][1],
        0.0, 0.0, 1.0, c0[0][2],
        0.0, 0.0, 0.0, 1.0,
    ))

    m.Multiply4x4(m_, m, m)

    part = '肾动脉'
    part_matrix[part] = m

    cfg[part]['变换'] = part_matrix[part].data

    part_transform[part] = vtkTransform()
    part_transform[part].SetMatrix(part_matrix[part])

    # 配准肾动脉到髂内动脉
    for part in ('髂内动脉', '肾动脉'):
        _ = pv.vtk_points(active_centerline[part])
        part_transform.get(part, vtkTransform()).TransformPoints(_, _)
        active_centerline[part] = np.array(_.data).reshape(2, -1, 3)[1]

        # c = {'髂内动脉': 'crimson', '肾动脉': 'pink'}[part]
        # _ = pl.add_lines(np.array(active_centerline[part]), c, 10, connected=True)

        for i in range(len(inactive_centerline[part])):
            _ = pv.vtk_points(inactive_centerline[part][i])
            part_transform.get(part, vtkTransform()).TransformPoints(_, _)
            inactive_centerline[part][i] = np.array(_.data).reshape(2, -1, 3)[1]

            # c = {'髂内动脉': 'grey', '肾动脉': 'lightgrey'}[part]
            # _ = pl.add_lines(np.array(inactive_centerline[part][i]), c, 5, connected=True)

    # 根据半径差异计算血管侧面切开长度，作为半径过渡段
    cfg['髂内动脉']['吻合半径'] = ra = active_radius['髂内动脉'][-1]
    cfg['肾动脉']['吻合半径'] = rb = active_radius['肾动脉'][0]
    cfg['肾动脉']['圆周长差'] = l_max = 2 * np.pi * (ra - rb)

    i, l_sum = 0, 0
    while l_sum < l_max:
        l_sum += np.linalg.norm(np.array(active_centerline['肾动脉'][i]) - np.array(active_centerline['肾动脉'][i + 1]))
        _ = min(max(l_sum / l_max, 0), 1)
        active_radius['肾动脉'][i] = ra * (1 - _) + rb * _
        i += 1

    cfg['肾动脉']['侧切长度'] = l_sum

    # 拉直血管中心线，参数化弯折
    centers = []
    c0 = [*active_centerline['髂内动脉'], *active_centerline['肾动脉']]
    c1 = wp.vec3()
    c1[2] = -1.0
    for i in range(len(c0)):
        if len(centers):
            if i < len(active_centerline['髂内动脉']):
                c1 = wp.quat_rotate(wp.quat_rpy(0.0, wp.radians(1.5), 0.0), c1)
            elif i < len(active_centerline['髂内动脉']) + l_sum:
                c1 = wp.quat_rotate(wp.quat_rpy(0.0, wp.radians(-1.0), 0.0), c1)
            else:
                c1 = wp.quat_rotate(wp.quat_rpy(0.0, wp.radians(0.5), 0.0), c1)
            _ = np.array(c1 / wp.length(c1))
            centers.append(centers[-1] + _ * np.linalg.norm(c0[i] - c0[i - 1]))
        else:
            centers.append(np.zeros(3))

    # 重采样血管中心线距离场，重建血管内壁面网格
    centers = np.array(centers)
    radius = np.hstack([active_radius['髂内动脉'], active_radius['肾动脉']])

    b = [np.min(centers, 0), np.max(centers, 0)]
    b[0] -= 2 * active_radius['髂内动脉'][0]
    b[1] += 2 * active_radius['髂内动脉'][0]

    iso_spacing = float(max(cfg.get('模拟', {}).get('血流粒子半径', 0.5), 0))
    cfg['模拟']['血流粒子半径'] = iso_spacing

    size = np.ceil((b[1] - b[0]) / iso_spacing)
    max_dist = float(np.linalg.norm(size * iso_spacing))

    sdf = wp.empty((*size,))

    wp.launch(centerline_sdf, sdf.shape, [
        sdf, wp.vec3(b[0]), iso_spacing, max_dist,
        wp.array(centers, dtype=wp.vec3),
        wp.array(radius, dtype=wp.float32),
        len(centers),
    ])

    _ = wp.empty_like(sdf)
    wp.launch(sdf_smooth, sdf.shape, [sdf, 2, _])
    sdf = _

    vol = np.sum(0 < sdf.numpy()) * iso_spacing ** 3
    n = round(vol / (4 / 3 * np.pi * iso_spacing ** 3))
    cfg['模拟']['血流粒子数量'] = n

    mesh = diff_dmc(sdf, b[0], iso_spacing, 0.0)

    wp.launch(centerline_sdf_end, sdf.shape, [
        sdf, wp.vec3(b[0]), iso_spacing, max_dist,
        wp.array(centers, dtype=wp.vec3),
        len(centers),
    ])

    # 模拟血流
    builder = newton.ModelBuilder('Z', 0.0)
    builder.add_shape_sdf(
        body=-1,
        sdf=newton.SDF(wp.Volume.load_from_numpy(sdf.numpy(), b[0], iso_spacing, max_dist)),
        cfg=newton.ModelBuilder.ShapeConfig(mu=0.0, restitution=0.5),
        key='sdf',
    )

    b_mesh = wp.Mesh(
        wp.array(mesh.vertices, wp.vec3),
        wp.array(mesh.faces.flatten(), wp.int32),
        support_winding_number=True,
    )
    b_shear_qd = wp.zeros((len(mesh.faces),), wp.vec3)

    _ = np.random.random((n, 3)) - 0.5
    _ /= np.linalg.norm(_, axis=1, keepdims=True)
    _ *= np.random.random((n, 1))
    _ = _ * radius[0] + centers[0]
    builder.add_particles(_.tolist(), np.zeros((n, 3)).tolist(), [1.0] * n, [0.0] * n)

    cfg_path.write_text(tomlkit.dumps(cfg), 'utf-8')

    model = builder.finalize()
    model.particle_mu = 0.5
    model.soft_contact_mu = 0.5
    model.particle_adhesion = 0.0
    model.particle_max_velocity = 1e3

    solver = newton.solvers.SolverXPBD(model)
    state_0, state_1 = model.state(), model.state()
    control = model.control()

    fluid = FluidSPH(iso_spacing * 2.5, b[0], b[1])

    fps = int(max(cfg.get('模拟', {}).get('帧率', 60), 0))
    cfg['模拟']['帧率'] = fps

    substeps = int(max(cfg.get('模拟', {}).get('子步', 1), 0))
    cfg['模拟']['子步'] = substeps

    pressure = int(max(cfg.get('模拟', {}).get('血压', 1e2), 0))
    cfg['模拟']['血压'] = pressure

    pressure /= np.pi * radius * radius

    frame_dt = 1.0 / fps
    sim_dt = frame_dt / substeps
    total_seconds = max(int(cfg.get('模拟', {}).get('最大时长', 10)), 0)
    max_frames = fps * total_seconds

    def update(skip_frame=False):
        pl.subplot(0, 0)
        valid = particle_valid.numpy()

        if not np.any(valid):
            return False

        q = state_0.particle_q.numpy()[valid]
        qd = state_0.particle_qd.numpy()[valid]

        qd = np.linalg.norm(qd, axis=1)
        pl.add_points(q, cmap='jet', scalars=qd, clim=[0, 50], show_scalar_bar=False, name='fluid',
                      render_points_as_spheres=True)
        pl.add_mesh(tri_poly(mesh), 'white', opacity=0.5)

        pl.subplot(0, 1)
        qd = b_shear_qd.numpy()
        np.linalg.norm(qd, axis=1)
        pl.add_mesh(tri_poly(mesh), scalars=qd, clim=[0, 1e3], cmap='jet', show_scalar_bar=False, name=f'mesh')

        if not skip_frame:
            pl.write_frame()
        return True

    particle_valid = wp.full((n,), True, wp.bool)

    pl.open_movie(cfg_path.parent / 'movie.mp4', fps)
    update(skip_frame=True)
    pl.reset_camera()
    pl.show(interactive_update=True)

    for fid in range(max_frames):
        for sid in range(substeps):
            state_0.clear_forces()
            state_1.clear_forces()

            contacts = model.collide(state_0)
            contacts.rigid_contact_thickness1.fill_(iso_spacing)

            # 中心线切向血压推力，中心线向心引力，末端流出失效，统计血管内壁受到的剪切作用
            wp.launch(fluid_pressure, state_0.particle_q.shape, [
                state_0.particle_q, state_0.particle_qd, state_0.particle_f,
                wp.array(pressure, dtype=wp.float32), wp.array(centers, dtype=wp.vec3), len(centers), max_dist,
                wp.uint64(b_mesh.id), b_shear_qd, particle_valid,
            ])

            # SPH求解粒子间流体动力学
            fluid.step(sim_dt, state_0.particle_q, state_0.particle_qd)

            # XPBD求解粒子与血管内壁碰撞反弹
            solver.step(state_0, state_1, control, contacts, sim_dt)
            state_0, state_1 = state_1, state_0

        pl.add_text(f'{fid + 1} / {max_frames}', font_size=9, name='text')
        if not update():
            break

    cfg_path.write_text(tomlkit.dumps(cfg), 'utf-8')

    pl.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()

    try:
        main(args.config)
    except KeyboardInterrupt:
        pass
