# CUDA driver & Toolkit & Microsoft Visual C++
# pip install itk pyglet trimesh rtree scikit-learn imgui-bundle
# pip install torch --index-url https://download.pytorch.org/whl/cu128
# pip install wheel diso
# pip install -U --pre warp-lang --extra-index-url=https://pypi.nvidia.com/
# pip uninstall newton-physics -y
# pip install -U git+https://github.com/newton-physics/newton.git@ace430de890d344974609443d1bf58a620026968

import argparse
import json
import locale
import warnings
from pathlib import Path

import PIL.Image
import newton.utils
import numpy as np
import tomlkit
import trimesh
import warp as wp
from PIL import Image, ImageDraw
from sklearn.decomposition import PCA
from tomlkit.exceptions import TOMLKitError

locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')


def main(cfg_path: str, headless: bool = False, overwrite: bool = False):
    cfg_path = Path(cfg_path)

    _ = cfg_path.read_text('utf-8')
    try:
        cfg = tomlkit.loads(_)
    except TOMLKitError:
        cfg = json.loads(_)
        cfg_path = cfg_path.parent / (cfg_path.name.removesuffix('.json') + '.toml')

    bone_threshold = [int(_) for _ in cfg['骨阈值']]
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

    if np.any((direction := np.array(image.GetDirection())) != np.eye(3)):
        warnings.warn(f'Abnormal intrinsics {direction.tolist()}')
        cfg['术前']['异常内参'] = direction.tolist()
        cfg_path.write_text(tomlkit.dumps(cfg), 'utf-8')
        spacing *= np.diag(image.GetDirection())

    image = itk.array_from_image(image).transpose(2, 1, 0).copy()
    volume = wp.Volume.load_from_numpy(image, bg_value=bg_value)

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
    op_side = int(np.sign(canal_y[1]))

    from kernel import diff_dmc, region_sample, planar_cut

    # 载入假体，竖直放置在股骨区域上方
    std_prothesis_mesh = trimesh.load_mesh(f'fs/{prothesis_path}')
    std_prothesis_mesh.vertices = std_prothesis_mesh.vertices[:, [0, 2, 1]] * [-1, 1, 1]
    std_prothesis_mesh.vertices[:, 2] -= np.min(std_prothesis_mesh.vertices[:, 2])
    std_prothesis_mesh.fix_normals()

    # 术前股骨近端，小粗隆以上分组骨阈值
    th_list = [float(_) for _ in cfg.get('近端骨阈值', [_ for _ in range(250, 550, 50)])]

    femur_mesh_proximal = []
    for th in th_list:
        femur_region = wp.full(shape=(*region_size,), value=bg_value, dtype=wp.float32)
        wp.launch(region_sample, femur_region.shape, [
            wp.uint64(volume.id), wp.vec3(spacing),
            femur_region, wp.vec3(region_origin), iso_spacing, region_xform,
        ])
        wp.launch(planar_cut, femur_region.shape, [
            femur_region, wp.vec3(region_origin), iso_spacing, region_xform,
            wp.vec3(neck_center), wp.vec3(-canal_z), th,
        ])
        wp.launch(planar_cut, femur_region.shape, [
            femur_region, wp.vec3(region_origin), iso_spacing, region_xform,
            wp.vec3(neck_center), wp.vec3(-neck_z), th,
        ])
        wp.launch(planar_cut, femur_region.shape, [
            femur_region, wp.vec3(region_origin), iso_spacing, region_xform,
            wp.vec3(keypoints[2]), wp.vec3(canal_z), th,
        ])

        mesh = diff_dmc(femur_region, iso_spacing, region_origin, th)
        if mesh.is_empty:
            raise RuntimeError(f'Empty pre-op femur mesh with threshold={th}')
        mesh = max(mesh.split(), key=lambda c: c.area)
        femur_mesh_proximal.append(mesh)

    # 术前股骨远端，小粗隆以下匹配较高骨阈值
    femur_region = wp.full(shape=(*region_size,), value=bg_value, dtype=wp.float32)
    wp.launch(region_sample, femur_region.shape, [
        wp.uint64(volume.id), wp.vec3(spacing),
        femur_region, wp.vec3(region_origin), iso_spacing, region_xform,
    ])
    wp.launch(planar_cut, femur_region.shape, [
        femur_region, wp.vec3(region_origin), iso_spacing, region_xform,
        wp.vec3(keypoints[2]), wp.vec3(-canal_z), bone_threshold[1],
    ])

    femur_mesh_distal = diff_dmc(femur_region, iso_spacing, region_origin, bone_threshold[1])
    if femur_mesh_distal.is_empty:
        raise RuntimeError('Empty pre-op femur mesh')
    femur_mesh_distal = max(femur_mesh_distal.split(), key=lambda c: c.area)

    # 假体植入碰撞模拟
    if (th_results := cfg.get('近端阈值')) and not overwrite:
        th_xforms = {int(th): result['标准假体变换到术前区域'] for th, result in th_results.items()}
        th_xforms = {th: [*[_ * 1e-2 for _ in xform[:3]], *xform[3:]] for th, xform in th_xforms.items()}
    else:
        cfg['近端阈值'] = {f'{th}': {} for th in th_list}
        th_xforms = {}
        n = len(th_list)
        hn = n // 2

        z = float(np.max([np.max(_.vertices[:,2]) for _ in femur_mesh_proximal]))
        init_body_q = [(0, 0, z * 1e-2, 0, 0, 0, 1) for _ in range(n)]  # 初始位姿

        builder = newton.ModelBuilder('Z')

        for _ in range(n):
            vertices = std_prothesis_mesh.vertices * 1e-2  # mm -> cm
            builder.add_shape_mesh(
                mesh=newton.Mesh(vertices, std_prothesis_mesh.faces.flatten()),
                body=builder.add_body(),  # 自由刚体
                xform=((0, _ - hn, 0), (0, 0, 0, 1)),
                cfg=builder.ShapeConfig(mu=0.05, ke=1e3, kd=1e1),  # 零摩擦系数
                key='prothesis',
            )

        for _ in range(n):
            vertices = femur_mesh_proximal[_].vertices * 1e-2  # mm -> cm
            builder.add_shape_mesh(
                mesh=newton.Mesh(vertices, femur_mesh_proximal[_].faces.flatten()),
                body=-1,  # 固定刚体
                xform=((0, _ - hn, 0), (0, 0, 0, 1)),
                cfg=builder.ShapeConfig(mu=0.05, ke=1e3, kd=1e1),  # 零摩擦系数
                key='femur_proximal',
            )

        for _ in range(n):
            vertices = femur_mesh_distal.vertices * 1e-2  # mm -> cm
            builder.add_shape_mesh(
                mesh=newton.Mesh(vertices, femur_mesh_distal.faces.flatten()),
                body=-1,  # 固定刚体
                xform=((0, _ - hn, 0), (0, 0, 0, 1)),
                cfg=builder.ShapeConfig(mu=0.05, ke=1e3, kd=1e1),  # 零摩擦系数
                key='femur_distal',
            )

        model = builder.finalize()
        solver = newton.solvers.SolverSemiImplicit(model)
        state_0, state_1 = model.state(), model.state()
        control = model.control()

        viewer = newton.viewer.ViewerGL(headless=headless)
        viewer.set_model(model)

        fps = 60
        frame_dt = 1.0 / fps
        sim_time = 0.0

        state_0.body_q.assign(init_body_q)  # 初始位姿

        substeps_force = (
                (0, 10, [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * n),
                (1, 10, [[0.0, -50.0, 0.0, 0.0, 0.0, -10.0]] * n),  # 施加力矩(内翻)迫使柄压紧近端前内壁
                (2, 10, [[0.0, -50.0, -50.0 * op_side, 0.0, 0.0, -10.0]] * n),  # 施加力矩(后倾)迫使柄压紧近端后方股骨距
        )

        stage, substeps, force = substeps_force[0]
        for fid in range(5000):
            with wp.ScopedTimer('', print=True):
                sim_dt = frame_dt / substeps
                for _ in range(substeps):
                    contacts = model.collide(state_0)
                    state_0.clear_forces()
                    state_1.clear_forces()

                    state_0.body_f.assign(force)

                    solver.step(state_0, state_1, control, contacts, sim_dt)
                    state_0, state_1 = state_1, state_0

            sim_time += frame_dt

            viewer.begin_frame(sim_time)
            viewer.log_state(state_0)
            viewer.end_frame()

            body_q = state_0.body_q.numpy()
            body_qd = state_0.body_qd.numpy()
            angular = np.rad2deg(np.min(np.linalg.norm(body_qd[:, :3], axis=1)))
            linear = np.min(np.linalg.norm(body_qd[:, 3:], axis=1))

            print(f'sim frame {fid} stage {stage} velocity {angular:.2f} deg/s {linear * 1e2:.2f} mm/s')

            # 失败，假体速度崩溃或位置过低
            if np.isnan(body_qd).any() or np.min(body_q[:, 2]) < -region_height * 1e-2:
                warnings.warn(f'sim failed {cfg_path.as_posix()}')
                break

            # 静止，角速度 < 10 deg/s，线速度 < 1.0 mm/s
            if angular < 10 and linear < 1.0e-2:
                print(f'sim completed {cfg_path.as_posix()}')
                for _ in range(n):
                    xform = body_q[_].copy()
                    xform[:3] *= 1e2
                    xform = np.array(xform).tolist()
                    cfg['近端阈值'][f'{th_list[_]}']['标准假体变换到术前区域'] = xform
                    cfg_path.write_text(tomlkit.dumps(cfg), 'utf-8')
                    th_xforms[th_list[_]] = tuple(float(_) for _ in xform)
                break

            if contacts.rigid_contact_count.numpy()[0] > 0:  # 开始接触时
                stage, substeps, force = substeps_force[1]
                if linear < 1e-1:  # 即将稳定时
                    stage, substeps, force = substeps_force[2]

    # 术前股骨重建，配准松质骨表面
    from kernel import diff_dmc, region_sample, planar_cut
    femur_region = wp.full(shape=(*region_size,), value=bg_value, dtype=wp.float32)
    wp.launch(region_sample, femur_region.shape, [
        wp.uint64(volume.id), wp.vec3(spacing),
        femur_region, wp.vec3(region_origin), iso_spacing, region_xform,
    ])
    wp.launch(planar_cut, femur_region.shape, [
        femur_region, wp.vec3(region_origin), iso_spacing, region_xform,
        wp.vec3(neck_center), wp.vec3(-canal_z), bone_threshold[0],
    ])
    wp.launch(planar_cut, femur_region.shape, [
        femur_region, wp.vec3(region_origin), iso_spacing, region_xform,
        wp.vec3(neck_center), wp.vec3(-neck_z), bone_threshold[0],
    ])

    femur_mesh = diff_dmc(femur_region, iso_spacing, region_origin, bone_threshold[0])
    if femur_mesh.is_empty:
        raise RuntimeError('Empty pre-op femur mesh')
    femur_mesh = max(femur_mesh.split(), key=lambda c: c.area)

    if '术后' in cfg:
        # 载入术后图像
        image_paths, images, spacings, bg_values, volumes = [image_path], [image], [spacing], [bg_value], [volume]

        image_paths.append(image_path := cfg_path.parent / cfg['术后']['原始图像'])
        image = itk.imread(image_path.as_posix())

        spacing = np.array([*itk.spacing(image)])

        if np.any((direction := np.array(image.GetDirection())) != np.eye(3)):
            warnings.warn(f'Abnormal intrinsics {direction.tolist()}')
            cfg['术后']['异常内参'] = direction.tolist()
            cfg_path.write_text(tomlkit.dumps(cfg), 'utf-8')
            spacing *= np.diag(image.GetDirection())

        spacings.append(spacing)
        bg_values.append(bg_value := float(np.min(image)))

        image = itk.array_from_image(image).transpose(2, 1, 0).copy()
        images.append(image)
        volumes.append(volume := wp.Volume.load_from_numpy(image, bg_value=bg_value))

        # 载入术后配置
        keypoints = np.array([spacing * cfg['术后'][_] for _ in (
            '股骨颈口外缘', '股骨颈口内缘', '股骨小粗隆髓腔中心', '股骨柄末端髓腔中心', '股骨髁间窝中心',
        )])

        # 计算术前术后股骨全长区域，区域重合作为初配准
        femur_meshes = [femur_mesh]

        # 术后股骨区域，坐标系Z轴 = 股骨近端髓腔中轴
        (
            neck_center, neck_x, neck_y, neck_z,
            canal_x, canal_y, canal_z,
            region_xform, region_size, region_origin,
        ) = subregion(*keypoints, margin, iso_spacing)

        # 术后股骨重建，配准松质骨表面
        from kernel import region_sample, diff_dmc
        femur_region = wp.full(shape=(*region_size,), value=bg_value, dtype=wp.float32)
        wp.launch(region_sample, femur_region.shape, [
            wp.uint64(volume.id), wp.vec3(spacing),
            femur_region, wp.vec3(region_origin), iso_spacing, region_xform,
        ])

        # 术后重建假体
        post_prothesis_mesh = diff_dmc(femur_region, iso_spacing, region_origin, prothesis_threshold)
        if post_prothesis_mesh.is_empty:
            raise RuntimeError('Empty post-op prothesis mesh')

        # 截骨后重建股骨
        wp.launch(planar_cut, femur_region.shape, [
            femur_region, wp.vec3(region_origin), iso_spacing, region_xform,
            wp.vec3(neck_center), wp.vec3(-canal_z), bone_threshold[0],
        ])
        wp.launch(planar_cut, femur_region.shape, [
            femur_region, wp.vec3(region_origin), iso_spacing, region_xform,
            wp.vec3(neck_center), wp.vec3(-neck_z), bone_threshold[0],
        ])

        femur_mesh = diff_dmc(femur_region, iso_spacing, region_origin, bone_threshold[0])
        if femur_mesh.is_empty:
            raise RuntimeError('Empty post-op femur mesh')
        femur_mesh = max(femur_mesh.split(), key=lambda c: c.area)
        femur_meshes.append(femur_mesh)

        # 术后截取股骨远端无伪影部分
        wp.launch(planar_cut, femur_region.shape, [
            femur_region, wp.vec3(region_origin), iso_spacing, region_xform,
            wp.vec3(keypoints[3]), wp.vec3(-canal_z), bone_threshold[0],
        ])

        femur_distal_mesh = diff_dmc(femur_region, iso_spacing, region_origin, bone_threshold[0])
        if femur_distal_mesh.is_empty:
            raise RuntimeError('Empty post-op femur mesh')
        femur_distal_mesh = max(femur_distal_mesh.split(), key=lambda c: c.area)

        # 利用股骨配准术前术后区域
        if (post_to_pre_region := cfg.get('术后区域变换到术前区域')) and not overwrite:
            post_to_pre_region = wp.transform(*post_to_pre_region)
        else:
            matrix = None
            mse_last: float | None = None
            for i in range(200):
                with wp.utils.ScopedTimer('', print=False) as t:
                    matrix, _, mse = trimesh.registration.icp(
                        femur_distal_mesh.vertices, femur_meshes[0], matrix, max_iterations=1,
                        **dict(reflection=False, scale=False),
                    )
                print(f'Femur ICP {i} MSE {mse:.3f} mm took {t.elapsed:.3f} ms')

                if i > 19 and mse_last - mse < 1e-3:  # 均方误差优化过少停止
                    break

                mse_last = mse

            post_to_pre_region = wp.transform_from_matrix(wp.mat44(matrix))
            cfg['术后区域变换到术前区域'] = np.array(post_to_pre_region).tolist()
            cfg['术前与术后配准误差'] = mse
            cfg_path.write_text(tomlkit.dumps(cfg), 'utf-8')

        assert post_to_pre_region is not None

        for th, xform in th_xforms.items():
            pre_region_to_std = wp.transform_inverse(wp.transform(*xform))

            post_to_pre_region_np = np.reshape(wp.transform_to_matrix(post_to_pre_region), (4, 4))
            pre_region_to_std_np = np.reshape(wp.transform_to_matrix(pre_region_to_std), (4, 4))

            parent = image_path.parent / f'近端阈值_{th}'
            parent.mkdir(parents=True, exist_ok=True)

            # 变换到标准假体坐标系留样检验
            _ = parent / f'{cfg_path.stem}_术前假体.stl'
            std_prothesis_mesh.export(_.as_posix())

            # 不同近端阈值条件
            _ = parent / f'{cfg_path.stem}_术后假体.stl'
            post_prothesis_mesh.apply_transform(post_to_pre_region_np)
            post_prothesis_mesh.apply_transform(pre_region_to_std_np)
            post_prothesis_mesh.export(_.as_posix())

            _ = parent / f'{cfg_path.stem}_术前股骨.stl'
            femur_meshes[0].apply_transform(pre_region_to_std_np)
            femur_meshes[0].export(_.as_posix())

            _ = parent / f'{cfg_path.stem}_术后股骨.stl'
            femur_meshes[1].apply_transform(post_to_pre_region_np)
            femur_meshes[1].apply_transform(pre_region_to_std_np)
            femur_meshes[1].export(_.as_posix())

            # 计算植入深度误差
            delta = float(np.min(std_prothesis_mesh.vertices[:, 2])) - float(np.min(post_prothesis_mesh.vertices[:, 2]))
            cfg['近端阈值'][f'{th}']['模拟与术后深度误差'] = delta
            cfg_path.write_text(tomlkit.dumps(cfg), 'utf-8')

            # 在标准假体坐标系(轴位)投影术后图像和模拟假体
            ray_spacing = 0.1

            box_femur = [_(femur_meshes[1].vertices, 0) for _ in (np.min, np.max)]
            box_metal = [_(post_prothesis_mesh.vertices, 0) for _ in (np.min, np.max)]
            box = [_([*box_femur, *box_metal], 0) for _ in (np.min, np.max)]
            box[0] -= 10
            box[1] += 10
            x, y, z = np.transpose(box)

            size = np.ceil([(x[1] - x[0]) / ray_spacing, (y[1] - y[0]) / ray_spacing])
            origin = np.array([x[0], y[0], z[0]])

            mesh = wp.Mesh(
                wp.array(std_prothesis_mesh.vertices, wp.vec3),
                wp.array(std_prothesis_mesh.faces.flatten(), wp.int32),
            )

            std_to_post_region = wp.transform_inverse(post_to_pre_region) * wp.transform_inverse(pre_region_to_std)

            from kernel import region_raymarching
            img2d = wp.zeros((*size,), wp.vec4ub)
            wp.launch(region_raymarching, img2d.shape, [
                img2d, wp.vec3(origin), wp.vec3(ray_spacing),
                wp.vec3(1, 0, 0), wp.vec3(0, 1, 0), wp.vec3(0, 0, 1), z[1] - z[0], box_femur[1][2] - z[0],
                wp.uint64(volume.id), wp.vec3(spacing), region_xform * std_to_post_region,
                mesh.id, 100, prothesis_threshold, -100, 900,
            ])
            img2d = img2d.numpy()
            rgb, alpha = img2d[:, :, :3], img2d[:, :, 3]

            # 绘制主轴线，计算模拟前倾角误差
            if len(points := np.argwhere(alpha == 255)):
                pca = PCA(n_components=2)
                pca.fit(points)

                axis = np.array(pca.components_[0])  # 第一主成分方向
                if axis[0] < 0:
                    axis *= -1

                cfg['近端阈值'][f'{th}']['模拟与术后前倾角误差'] = np.rad2deg(np.arctan2(axis[1], axis[0])) * -op_side
                cfg_path.write_text(tomlkit.dumps(cfg), 'utf-8')

                rgb = Image.fromarray(np.flipud(np.rot90(rgb, k=op_side)))
                alpha = Image.fromarray(np.flipud(np.rot90(alpha, k=op_side)))
                if op_side > 0:
                    # draw_line(rgb, pca.mean_, axis, size[0] * 0.5, 'white', 8)
                    draw_line(alpha, pca.mean_, axis, size[0] * 0.5, 127, 8)
                else:
                    # draw_line(rgb, size - pca.mean_, -axis, size[0] * 0.5, 'white', 8)
                    draw_line(alpha, size - pca.mean_, -axis, size[0] * 0.5, 127, 8)
            else:
                warnings.warn('No prosthesis cross section found')
                rgb = Image.fromarray(np.flipud(np.rot90(rgb, k=op_side)))
                alpha = Image.fromarray(np.flipud(np.rot90(alpha, k=op_side)))

            _ = parent / f'{cfg_path.stem}_术后轴位.jpg'
            rgb.save(_.as_posix())
            _ = parent / f'{cfg_path.stem}_术后轴位_截面.jpg'
            alpha.save(_.as_posix())

            # 在标准假体坐标系(正位)投影术后图像和模拟假体
            size = np.ceil([(x[1] - x[0]) / ray_spacing, (z[1] - z[0]) / ray_spacing])
            origin = np.array([x[0], y[0], z[0]])

            from kernel import region_raymarching
            img2d = wp.zeros((*size,), wp.vec4ub)
            wp.launch(region_raymarching, img2d.shape, [
                img2d, wp.vec3(origin), wp.vec3(ray_spacing),
                wp.vec3(1, 0, 0), wp.vec3(0, 0, 1), wp.vec3(0, 1, 0), y[1] - y[0], 0.0,
                wp.uint64(volume.id), wp.vec3(spacing), region_xform * std_to_post_region,
                mesh.id, 100, prothesis_threshold, -100, 900,
            ])
            img2d = img2d.numpy()
            rgb, alpha = img2d[:, :, :3], img2d[:, :, 3]

            rgb = np.rot90(rgb, k=op_side)
            if op_side < 0:
                rgb = np.flipud(rgb)
            rgb = Image.fromarray(rgb)
            _ = parent / f'{cfg_path.stem}_术后正位.jpg'
            rgb.save(_.as_posix())

            # 在标准假体坐标系(侧位)投影术后图像和模拟假体
            size = np.ceil([(x[1] - x[0]) / ray_spacing, (z[1] - z[0]) / ray_spacing])
            origin = np.array([x[0], y[0], z[0]])

            from kernel import region_raymarching
            img2d = wp.zeros((*size,), wp.vec4ub)
            wp.launch(region_raymarching, img2d.shape, [
                img2d, wp.vec3(origin), wp.vec3(ray_spacing),
                wp.vec3(0, 1, 0), wp.vec3(0, 0, 1), wp.vec3(1, 0, 0), x[1] - x[0], 0.0,
                wp.uint64(volume.id), wp.vec3(spacing), region_xform * std_to_post_region,
                mesh.id, 100, prothesis_threshold, -100, 900,
            ])
            img2d = img2d.numpy()
            rgb, alpha = img2d[:, :, :3], img2d[:, :, 3]

            rgb = np.rot90(rgb, k=op_side)
            if op_side < 0:
                rgb = np.flipud(rgb)
            rgb = Image.fromarray(rgb)
            _ = parent / f'{cfg_path.stem}_术后侧位.jpg'
            rgb.save(_.as_posix())
    else:
        print('Ignore post-op validation')


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
    _ = wp.quat_from_matrix(wp.mat33(np.array([canal_x, canal_y, canal_z]).T))
    xform = wp.transform(canal_deep - np.dot(canal_deep - ic_notch, canal_z) * canal_z, _)

    _ = [np.array(wp.transform_point(xform, wp.vec3(_))) for _ in (
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


def draw_line(image: PIL.Image.Image, mean, axis, hl, fill, width):
    draw = ImageDraw.Draw(image)
    x1, y1 = int(mean[0] - axis[0] * hl), int(mean[1] - axis[1] * hl)
    x2, y2 = int(mean[0] + axis[0] * hl), int(mean[1] + axis[1] * hl)
    draw.line((x1, y1, x2, y2), fill=fill, width=width)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    with wp.utils.ScopedTimer(args.config):
        main(args.config, args.headless, args.overwrite)
