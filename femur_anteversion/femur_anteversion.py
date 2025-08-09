# CUDA driver & Toolkit & Microsoft Visual C++
# pip install torch --index-url https://download.pytorch.org/whl/cu128
# pip install wheel diso
# pip install git+https://github.com/newton-physics/newton@f701455313df2ee83ec881d6612657882f2472a0
# pip install itk warp-lang pyglet trimesh rtree newton-clips==0.1.5

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
    prothesis_threshold = cfg['假体阈值']
    prothesis_path = cfg['假体']
    margin = np.array(cfg['边距'])
    iso_spacing = 1.0

    # 载入术前图像
    image_path = cfg_path.parent / cfg['术前']['原始图像']

    import itk
    image = itk.imread(image_path.as_posix())

    spacing = np.array([*itk.spacing(image)])
    bg_value = float(np.min(image))

    image = itk.array_from_image(image).transpose(2, 1, 0).copy()
    volume = wp.types.Volume.load_from_numpy(image, bg_value=bg_value)

    # 载入术前配置
    keypoints = np.array([spacing * cfg['术前'][_] for _ in (
        '股骨颈口外缘', '股骨颈口内缘', '股骨小粗隆髓腔中心', '股骨柄末端髓腔中心', '股骨髁间窝中心',
    )])

    # 术前股骨区域，坐标系Z轴 = 股骨近端髓腔中轴
    (
        neck_center, neck_x, neck_y, neck_z,
        canal_x, canal_y, canal_z,
        region_xform, region_size, region_origin,
    ) = subregion(*keypoints, margin, iso_spacing)
    region_height = region_size[2] * iso_spacing

    # 区域重采样，股骨颈截骨，等值面网格重建，构建碰撞体
    from kernel import diff_dmc, femur_sample_region, femur_cut_head
    femur_region = wp.context.full(shape=(*region_size,), value=bg_value, dtype=wp.types.float32)
    wp.context.launch(femur_sample_region, femur_region.shape, [
        wp.types.uint64(volume.id), wp.types.vec3(spacing),
        femur_region, wp.types.vec3(region_origin), iso_spacing, region_xform,
    ])
    wp.context.launch(femur_cut_head, femur_region.shape, [
        femur_region, wp.types.vec3(region_origin), iso_spacing, region_xform,
        wp.types.vec3(neck_center), wp.types.vec3(-canal_z), bone_threshold,
    ])
    wp.context.launch(femur_cut_head, femur_region.shape, [
        femur_region, wp.types.vec3(region_origin), iso_spacing, region_xform,
        wp.types.vec3(neck_center), wp.types.vec3(-neck_z), bone_threshold,
    ])

    femur_mesh = diff_dmc(femur_region, iso_spacing, region_origin, bone_threshold)
    if femur_mesh.is_empty:
        raise RuntimeError('Empty pre-op femur mesh')
    femur_mesh = max(femur_mesh.split(), key=lambda c: c.area)

    _ = image_path.parent / f'{cfg_path.name}_术前股骨.stl'
    femur_mesh.export(_.as_posix())

    # 载入假体，竖直放置在地面(Z=0)上
    prothesis_mesh = trimesh.load_mesh(f'fs/{prothesis_path}')
    prothesis_mesh.vertices = prothesis_mesh.vertices[:, [0, 2, 1]] * [-1, 1, 1]
    prothesis_mesh.vertices[:, 2] += region_height - np.min(prothesis_mesh.vertices[:, 2])
    prothesis_mesh.fix_normals()

    if (sim_xform := cfg.get('模拟假体位姿')) is None and not recompute:
        # 假体植入碰撞模拟
        builder = newton.ModelBuilder('Z')

        vertices = femur_mesh.vertices * 1e-2  # mm -> cm
        builder.add_shape_mesh(
            mesh=newton.Mesh(vertices, femur_mesh.faces.flatten()),
            body=-1,  # 固定刚体
            cfg=builder.ShapeConfig(mu=0.0),  # 零摩擦系数
            key='femur',
        )

        # 初始位置升高
        vertices = prothesis_mesh.vertices * 1e-2  # mm -> cm
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
                cfg['模拟假体位姿'] = sim_xform.tolist()
                cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=4), 'utf-8')
                break

        renderer.save()

    if sim_xform is None:
        return

    # 假体模拟变换，建模坐标变换到术前股骨区域坐标
    sim_xform = wp.types.transform(*sim_xform)
    matrix = np.reshape(wp.transform_to_matrix(sim_xform), (4, 4))
    prothesis_mesh.apply_transform(matrix)

    _ = image_path.parent / f'{cfg_path.name}_术前假体.stl'
    prothesis_mesh.export(_.as_posix())

    # 载入术后图像
    image_paths, images, spacings, bg_values, volumes = [image_path], [image], [spacing], [bg_value], [volume]

    image_paths.append(image_path := cfg_path.parent / cfg['术后']['原始图像'])
    image = itk.imread(image_path.as_posix())

    spacings.append(spacing := np.array([*itk.spacing(image)]))
    bg_values.append(bg_value := float(np.min(image)))

    image = itk.array_from_image(image).transpose(2, 1, 0).copy()
    images.append(image)
    volumes.append(volume := wp.types.Volume.load_from_numpy(image, bg_value=bg_value))

    # 载入术前配置
    keypoints = np.array([spacing * cfg['术后'][_] for _ in (
        '股骨颈口外缘', '股骨颈口内缘', '股骨小粗隆髓腔中心', '股骨柄末端髓腔中心', '股骨髁间窝中心',
    )])

    # 计算术前术后股骨全长区域，区域重合作为初配准
    region_xforms, region_sizes, region_origins = [region_xform], [region_size], [region_origin]
    femur_regions, femur_meshes, prothesis_meshes = [femur_region], [femur_mesh], [prothesis_mesh]

    # 术后股骨区域，坐标系Z轴 = 股骨近端髓腔中轴
    (
        neck_center, neck_x, neck_y, neck_z,
        canal_x, canal_y, canal_z,
        region_xform, region_size, region_origin,
    ) = subregion(*keypoints, margin, iso_spacing)
    region_xforms.append(region_xform)

    # 区域重采样，股骨颈截骨，等值面网格重建，构建碰撞体
    from kernel import femur_sample_region, diff_dmc
    femur_region = wp.context.full(shape=(*region_size,), value=bg_value, dtype=wp.types.float32)
    wp.context.launch(femur_sample_region, femur_region.shape, [
        wp.types.uint64(volume.id), wp.types.vec3(spacing),
        femur_region, wp.types.vec3(region_origin), iso_spacing, region_xform,
    ])
    femur_regions.append(femur_region)

    # 术后重建股骨
    femur_mesh = diff_dmc(femur_region, iso_spacing, region_origin, bone_threshold)
    if femur_mesh.is_empty:
        raise RuntimeError('Empty post-op femur mesh')
    femur_mesh = max(femur_mesh.split(), key=lambda c: c.area)
    femur_meshes.append(femur_mesh)

    _ = image_path.parent / f'{cfg_path.name}_术后股骨.stl'
    femur_mesh.export(_.as_posix())

    # 术后重建假体
    prothesis_mesh = diff_dmc(femur_region, iso_spacing, region_origin, prothesis_threshold)
    if prothesis_mesh.is_empty:
        raise RuntimeError('Empty post-op prothesis mesh')
    # prothesis_mesh = max(prothesis_mesh.split(), key=lambda c: c.area)
    prothesis_meshes.append(prothesis_mesh)

    _ = image_path.parent / f'{cfg_path.name}_术后假体.stl'
    prothesis_mesh.export(_.as_posix())

    # 股骨配准
    wp.context.launch(femur_cut_head, femur_region.shape, [
        femur_region, wp.types.vec3(region_origin), iso_spacing, region_xform,
        wp.types.vec3(keypoints[3]), wp.types.vec3(-canal_z), bone_threshold,
    ])

    femur_mesh = diff_dmc(femur_region, iso_spacing, region_origin, bone_threshold)
    if femur_mesh.is_empty:
        raise RuntimeError('Empty post-op femur mesh')
    femur_mesh = max(femur_mesh.split(), key=lambda c: c.area)

    _ = image_path.parent / f'{cfg_path.name}_术后股骨_远端.stl'
    femur_mesh.export(_.as_posix())

    matrix = None
    mse_last: float | None = None
    for i in range(200):
        with wp.utils.ScopedTimer('', print=False) as t:
            matrix, _, mse = trimesh.registration.icp(femur_mesh.vertices, femur_meshes[0], matrix, max_iterations=1,
                                                      **dict(reflection=False, scale=False))
        print(f'ICP {i} MSE {mse:.2f} mm took {t.elapsed:.2f} ms')

        if i > 19 and mse_last - mse < 1e-3:  # 均方误差优化过少停止
            break

        mse_last = mse

    femur_mesh.apply_transform(matrix)
    femur_meshes[1].apply_transform(matrix)
    prothesis_mesh.apply_transform(matrix)

    _ = image_path.parent / f'{cfg_path.name}_术后股骨_远端_配准.stl'
    femur_mesh.export(_.as_posix())

    _ = image_path.parent / f'{cfg_path.name}_术后股骨_配准.stl'
    femur_meshes[1].export(_.as_posix())

    _ = image_path.parent / f'{cfg_path.name}_术后假体_配准.stl'
    prothesis_mesh.export(_.as_posix())


def subregion(neck_lateral, neck_medial, canal_entry, canal_deep, ic_notch, margin, iso_spacing=1.0):
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
    xform = wp.types.transform(canal_deep - np.dot(canal_deep - ic_notch, canal_z) * canal_z, _)

    _ = [np.array(wp.transform_point(xform, wp.types.vec3(_))) for _ in (
        neck_lateral, neck_medial, canal_entry, canal_deep, ic_notch,
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
