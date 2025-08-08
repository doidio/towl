# CUDA driver & PyTorch
# pip install git+https://github.com/newton-physics/newton@f701455313df2ee83ec881d6612657882f2472a0
# pip install itk warp-lang diso newton-clips==0.1.5

import argparse
import json
import locale
import warnings
from pathlib import Path

import newton.utils
import newtonclips
import numpy as np
import trimesh
import warp as wp

locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')

newtonclips.SAVE_DIR = '.clips'


def main(cfg_path: str, headless: bool = False, recompute: bool = False):
    cfg_path = Path(cfg_path)
    cfg = json.loads(cfg_path.read_text('utf-8'))

    bone_threshold = cfg['骨阈值']
    prothesis_path = cfg['假体']
    iso_spacing = 1.0

    # 载入术前图像
    image_path = cfg_path.parent / cfg['术前']['原始图像']

    import itk
    image = itk.imread(image_path.as_posix())

    spacing = np.array([*itk.spacing(image)])
    bg_value = float(np.min(image))

    image = itk.array_from_image(image).transpose(2, 1, 0).copy()
    volume = wp.types.Volume.load_from_numpy(image, bg_value=bg_value)

    if (sim_xform := cfg['术前'].get('假体位姿')) is None and not recompute:
        # 载入术前配置
        margin = np.array(cfg['术前']['边距'])
        keypoints = np.array([spacing * cfg['术前'][_] for _ in (
            '股骨颈口外缘', '股骨颈口内缘', '股骨小粗隆髓腔中心', '股骨柄末端髓腔中心',
        )])

        # 术前股骨近端区域，坐标系Z轴 = 股骨近端髓腔中轴
        (
            neck_center, neck_x, neck_y, neck_z,
            canal_x, canal_y, canal_z,
            region_xform, region_size, region_origin,
        ) = subregion(*keypoints, margin, iso_spacing)
        region_height = region_size[2] * iso_spacing

        # 区域重采样，股骨颈截骨，等值面网格重建，构建碰撞体
        from kernel import femur_proximal_region, diff_dmc
        femur_proximal = wp.context.full(shape=(*region_size,), value=bg_value, dtype=wp.types.float32)
        wp.context.launch(femur_proximal_region, femur_proximal.shape, [
            wp.types.uint64(volume.id), wp.types.vec3(spacing),
            femur_proximal, wp.types.vec3(region_origin), iso_spacing, region_xform,
            wp.types.vec3(neck_center), wp.types.vec3(-canal_z),
            wp.types.vec3(neck_center), wp.types.vec3(-neck_z),
            bone_threshold,
        ])

        femur_proximal_mesh = diff_dmc(femur_proximal, iso_spacing, region_origin, bone_threshold)

        # 假体植入碰撞模拟
        builder = newton.ModelBuilder('Z')

        vertices = femur_proximal_mesh.vertices * 1e-2  # mm -> cm
        builder.add_shape_mesh(
            mesh=newton.Mesh(vertices, femur_proximal_mesh.faces.flatten()),
            body=-1,  # 固定刚体
            cfg=builder.ShapeConfig(mu=0.0),  # 零摩擦系数
            key='femur',
        )

        # 载入假体，竖直放置在地面(Z=0)上
        prothesis_mesh = trimesh.load_mesh(f'fs/{prothesis_path}')
        prothesis_mesh.vertices = prothesis_mesh.vertices[:, [0, 2, 1]] * [-1, 1, 1]
        prothesis_mesh.vertices[:, 2] -= np.min(prothesis_mesh.vertices[:, 2])
        prothesis_mesh.fix_normals()

        # 初始位置升高
        vertices = (prothesis_mesh.vertices + [0, 0, region_height]) * 1e-2  # mm -> cm
        builder.add_shape_mesh(
            mesh=newton.Mesh(vertices, prothesis_mesh.faces.flatten()),
            body=builder.add_body(),  # 自由刚体
            cfg=builder.ShapeConfig(mu=0.0),  # 零摩擦系数
            key='prothesis',
        )

        model = builder.finalize()
        solver = newton.solvers.SemiImplicitSolver(model)
        state_0, state_1 = model.state(), model.state()
        control = model.control()

        if headless:
            renderer = newtonclips.SimRendererOpenGL(model)
        else:
            renderer = newton.utils.SimRendererOpenGL(model)

        fps = 60
        frame_dt = 1.0 / fps
        sim_substeps = 50  # 调节时间步长
        sim_dt = frame_dt / sim_substeps
        sim_time = 0.0

        for fid in range(5000):
            with wp.utils.ScopedTimer(f'sim frame {fid}'):
                for _ in range(sim_substeps):
                    contacts = model.collide(state_0)
                    state_0.clear_forces()
                    state_1.clear_forces()

                    solver.step(state_0, state_1, control, contacts, sim_dt)

                    state_0, state_1 = state_1, state_0

            sim_time += frame_dt

            renderer.begin_frame(sim_time)
            renderer.render(state_0)
            renderer.end_frame()

            body_q = state_0.body_q.numpy()
            body_qd = state_0.body_qd.numpy()
            angular = np.max(np.linalg.norm(body_qd[:, :3], axis=1))
            linear = np.max(np.linalg.norm(body_qd[:, 3:], axis=1))

            # 失败，假体速度崩溃或位置过低
            if np.isnan(body_qd[0]).any() or np.min(body_q[:, 2]) < -region_height * 2e-2:
                warnings.warn(f'sim failed {cfg_path.as_posix()}')
                break

            # 静止，角速度 < 1 deg/s，线速度 < 1 mm/s
            elif angular < np.deg2rad(1) and linear < 1e-4:
                print(f'sim completed {cfg_path.as_posix()}')
                sim_xform = body_q[0].copy()
                sim_xform[:3] *= 1e2
                sim_xform[2] += region_height
                cfg['术前']['假体位姿'] = sim_xform.tolist()
                cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=4), 'utf-8')
                break

        renderer.save()

    if sim_xform is None:
        return

    # 假体模拟变换，建模坐标变换到术前股骨近端区域坐标
    sim_xform = wp.types.transform(*sim_xform)

    # 载入术后图像
    image_paths = [image_path, cfg_path.parent / cfg['术后']['原始图像']]
    images = [image, itk.imread(image_paths[1].as_posix())]

    spacings = [spacing, np.array([*itk.spacing(images[1])])]
    bg_values = [bg_value, float(np.min(images[1]))]

    images[1] = itk.array_from_image(images[1]).transpose(2, 1, 0).copy()
    volumes = [volume, wp.types.Volume.load_from_numpy(images[1], bg_value=bg_values[1])]

    # 计算术前术后股骨全长区域，区域重合作为初配准
    region_xforms, region_sizes, region_origins = [], [], []
    for i, side in enumerate(('术前', '术后')):
        margin = np.array(cfg[side]['边距'])
        keypoints = np.array([spacings[i] * cfg[side][_] for _ in (
            '股骨颈口外缘', '股骨颈口内缘', '股骨小粗隆髓腔中心', '股骨髁间窝中心',
        )])

        _, _, _, _, _, _, _, xform, size, origin = subregion(*keypoints, margin, iso_spacing)

        region_xforms.append(xform)
        region_sizes.append(size)
        region_origins.append(origin)

    # 建模假体->术后股骨全长区域->术前股骨全长区域->术前CT->术前股骨近端区域->建模假体
    #
    from kernel import femur_diff_region
    align = wp.types.transform()
    diff = wp.context.full(shape=(*region_sizes[0],), value=0.0, dtype=wp.types.float32)
    wp.context.launch(femur_diff_region, diff.shape, [
        wp.types.uint64(volumes[0].id), wp.types.vec3(spacings[0]),
        region_xforms[0], wp.types.vec3(region_origins[0]),
        wp.types.uint64(volumes[1].id), wp.types.vec3(spacings[1]),
        region_xforms[1], wp.types.vec3(region_origins[1]),
        bone_threshold, align, diff,
    ])


def subregion(neck_lateral, neck_medial, canal_entry, canal_deep, margin, iso_spacing=1.0):
    # 股骨截颈坐标系
    neck_center = 0.5 * (neck_lateral + neck_medial)
    neck_x = neck_lateral - neck_medial
    neck_z = neck_center - canal_entry
    neck_y = np.cross(neck_z, neck_x)
    neck_z = np.cross(neck_x, neck_y)
    neck_x, neck_y, neck_z = [_ / np.linalg.norm(_) for _ in (neck_x, neck_y, neck_z)]

    # 股骨髓腔坐标系
    canal_z = canal_entry - canal_deep
    canal_x = canal_entry - neck_medial
    canal_y = np.cross(canal_z, canal_x)
    canal_x = np.cross(canal_y, canal_z)
    canal_x, canal_y, canal_z = [_ / np.linalg.norm(_) for _ in (canal_x, canal_y, canal_z)]

    # 区域坐标系
    _ = wp.quat_from_matrix(wp.types.mat33(np.array([canal_x, canal_y, canal_z]).T))
    xform = wp.types.transform(canal_deep - margin[2] * canal_z, _)

    _ = [np.array(wp.transform_point(xform, wp.types.vec3(_))) for _ in (
        neck_lateral, neck_medial, canal_entry, canal_deep,
    )]
    box = (
        np.min(_, axis=0) - margin,
        np.max(_, axis=0) + margin,
    )
    size = np.ceil((box[1] - box[0]) / iso_spacing)
    length = size * iso_spacing
    origin = np.array([-0.5 * length[0], -0.5 * length[1], 0.0])

    return neck_center, neck_x, neck_y, neck_z, canal_x, canal_y, canal_z, xform, size, origin


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--headless', default=False)
    parser.add_argument('--recompute', default=False)
    args = parser.parse_args()
    main(args.config, args.headless, args.recompute)
