import os
import shutil
from pathlib import Path
from typing import Optional

import gradio as gr
import imageio.v3 as iio
import itk
import numpy as np
import trimesh
import warp as wp
import warp.sim  # noqa

import towl as tl
import towl.save_pb2 as pb

g_default = set(globals().keys())

init_filename: Optional[str] = None

init_array: Optional[np.ndarray] = None
init_volume: Optional[wp.Volume] = None

init_volume_size: Optional[np.ndarray] = None
init_volume_spacing: Optional[np.ndarray] = None
init_volume_origin: Optional[np.ndarray] = None
init_volume_length: Optional[np.ndarray] = None

iso_spacing = 0.5
main_region_window = [-100.0, 900.0]
bone_threshold = 500.0

main_region_min: Optional[np.ndarray] = None
main_region_max: Optional[np.ndarray] = None
main_region_size: Optional[np.ndarray] = None
main_region_origin: Optional[np.ndarray] = None

main_image_xy: Optional[np.ndarray] = None
main_image_xz: Optional[np.ndarray] = None

xinv = -1

kp_names = ['骨盆髂前上棘', '骨盆耻骨结节']
kp_names = [f'左侧{_}' for _ in kp_names] + [f'右侧{_}' for _ in kp_names]
kp_names += [
    '股骨柄颈锥圆心',
    '股骨颈口上缘', '股骨颈口下缘', '股骨小粗隆髓腔中心', '股骨柄末端髓腔中心',
]

kp_select_rgb = [255, 127, 127]
kp_deselect_rgb = [85, 170, 255]
kp_deselect_radius = 5

kp_name_none = '取消选中'
kp_name: str = kp_name_none
kp_positions = {}

kp_image_xz: Optional[np.ndarray] = None
kp_image_xy: Optional[np.ndarray] = None

taper_center: Optional[np.ndarray] = None
neck_center: Optional[np.ndarray] = None
neck_x: Optional[np.ndarray] = None
neck_y: Optional[np.ndarray] = None
neck_z: Optional[np.ndarray] = None
canal_x: Optional[np.ndarray] = None
canal_y: Optional[np.ndarray] = None
canal_z: Optional[np.ndarray] = None

femur_stem_rgb = [85, 170, 255]

prothesis_v_i: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]] = None

femur_v_i: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]] = None
femur_region: Optional[np.ndarray] = None
femur_region_origin: Optional[np.ndarray] = None
femur_region_size: Optional[np.ndarray] = None

femur_smb: Optional[wp.sim.ModelBuilder] = None
ps_body: Optional[int] = None

g_default = {_: globals()[_] for _ in globals() if _ not in g_default and _ != 'g_default'}


def on_0_save():
    if init_filename is None:
        raise gr.Error('未载入源数据或存档', print_exception=False)

    gr.Info('开始保存')
    save = pb.SaveTotalHip()

    if init_array is not None:
        save.init_volume.CopyFrom(pb.Volume(
            dtype=pb.DataType.INT16,
            data=init_array.tobytes(),
            background=np.min(init_array),
            region=pb.Volume.Region(
                size=init_volume_size,
                spacing=init_volume_spacing,
                origin=init_volume_origin,
            )
        ))

    if main_image_xy is not None:
        save.main_region.CopyFrom(pb.KeyBox(
            min=pb.Floats(values=main_region_min),
            max=pb.Floats(values=main_region_max),
        ))

    if kp_name is not None:
        save.kp_name = kp_name

    save.xinv = xinv < 0

    if len(kp_positions) > 0:
        save.kp_positions.CopyFrom(pb.KeyPoints(
            named_positions={k: pb.Floats(values=v) for k, v in kp_positions.items()},
        ))

    os.makedirs(tl.save_dir, exist_ok=True)
    f = tl.save_dir / f'{init_filename}.save'
    f.write_bytes(save.SerializeToString())
    gr.Success('保存成功')


def on_0_upload(filename):
    filename = Path(filename)
    try:
        itk.imread(filename.as_posix())
    except Exception as e:
        gr.Warning(f'读取失败 {e}')

    os.makedirs(tl.save_dir, exist_ok=True)
    shutil.copy(filename, tl.save_dir / filename.name)
    gr.Success('上传成功')


def on_0_load(filename):
    if filename is None:
        raise gr.Error('未选中', print_exception=False)

    f = Path(filename)
    if not f.is_file() or not f.exists():
        raise gr.Error('选中无效', print_exception=False)

    global init_filename, init_array, init_volume
    global init_volume_size, init_volume_spacing, init_volume_origin, init_volume_length
    global main_region_min, main_region_max
    global xinv, kp_name, kp_positions

    if f.name.lower().endswith(suffix := '.save'):
        gr.Info('正在载入存档')

        try:
            save: pb.SaveTotalHip = pb.SaveTotalHip.FromString(f.read_bytes())
        except Exception as e:
            raise gr.Error(f'载入存档失败 {e}', print_exception=False)

        globals().update(g_default)

        if save.HasField('init_volume'):
            _ = [*save.init_volume.region.size]
            init_array = np.frombuffer(save.init_volume.data, np.int16).reshape(_)
            init_volume = wp.Volume.load_from_numpy(init_array, bg_value=float(np.min(init_array)))

            init_volume_size = np.array(save.init_volume.region.size)
            init_volume_spacing = np.array(save.init_volume.region.spacing)
            init_volume_origin = np.array(save.init_volume.region.origin)
            init_volume_length = init_volume_size * init_volume_spacing

        if save.HasField('main_region'):
            main_region_min = np.array(save.main_region.min.values)
            main_region_max = np.array(save.main_region.max.values)

        xinv = -1 if save.xinv else 1

        if save.HasField('kp_name'):
            kp_name = save.kp_name

        if save.HasField('kp_positions'):
            kp_positions = {k: [*v.values] for k, v in save.kp_positions.named_positions.items()}

        init_filename = f.name.removesuffix(suffix)
        gr.Success(f'载入成功')
    elif f.name.lower().endswith(suffix := '.nii.gz'):
        gr.Info('正在载入源数据')
        with wp.ScopedTimer('', print=False) as t:
            image = itk.imread(f.as_posix(), pixel_type=itk.SS)

            if image.ndim != 3:
                raise gr.Error(f'仅支持3D图像，不支持{image.ndim}D图像', print_exception=False)

            if np.any((direction := np.array(image.GetDirection())) != np.identity(3)):
                gr.Warning(f'忽略已知的图像歪斜\n{direction}')

            globals().update(g_default)

            init_volume_size = np.array(itk.size(image))
            init_volume_spacing = np.array(itk.spacing(image))
            init_volume_origin = np.array(itk.origin(image))
            init_volume_length = init_volume_size * init_volume_spacing

            init_array = itk.array_from_image(image).swapaxes(0, 2).copy()
            init_volume = wp.Volume.load_from_numpy(init_array, bg_value=float(np.min(init_array)))

            main_region_min = np.zeros(3)
            main_region_max = init_volume_length.copy()

        init_filename = f.name.removesuffix(suffix)
        gr.Success(f'载入成功 {t.elapsed} ms')
    else:
        raise gr.Error(f'载入失败，未知文件 {f.name}', print_exception=False)


def on_0_unload():
    print(g_default)
    globals().update(g_default)
    gr.Success('重置成功')


def on_1_main_region(r, a, i, l, p, s):
    global main_region_min, main_region_max
    main_region_min = np.min(np.array([[r, a, i], [l, p, s]]), axis=0)
    main_region_max = np.max(np.array([[r, a, i], [l, p, s]]), axis=0)


def on_2_op_side(index):
    global xinv
    xinv = 1 if index > 0 else -1


def on_2_kp_name(name):
    global kp_name
    kp_name = name


def on_2_image_xz_select(evt: gr.SelectData, image):
    if kp_name not in kp_names:
        raise gr.Error('未选中解剖标志', print_exception=False)

    p0 = main_region_origin[0] + iso_spacing * evt.index[0]
    p2 = main_region_origin[2] + iso_spacing * (image.shape[0] - evt.index[1])
    kp_positions[kp_name] = [p0, p2]


def on_2_image_xy_select(evt: gr.SelectData, _):
    if kp_name not in kp_names:
        raise gr.Error('未选中解剖标志', print_exception=False)

    global kp_positions
    if kp_name not in kp_positions:
        raise gr.Error(f'未先选透视点 {kp_name}', print_exception=False)

    p0 = main_region_origin[0] + iso_spacing * evt.index[0]
    p1 = main_region_origin[1] + iso_spacing * evt.index[1]

    if len(p := kp_positions[kp_name]) == 2:
        kp_positions[kp_name] = [p0, p1, p[1]]
    elif len(p) == 3:
        kp_positions[kp_name] = [p0, p1, p[2]]
    else:
        raise gr.Error(f'数据错误 {kp_name} {p}', print_exception=False)


def on_femur_sim():
    if femur_smb is None:
        raise gr.Error('未创建股骨柄', print_exception=False)

    if any([_ is None for _ in [neck_center, neck_z]]):
        raise gr.Error('缺少必要的解剖标志', print_exception=False)

    model = femur_smb.finalize()
    model.ground = False

    state_0 = model.state()
    state_1 = model.state()

    integrator = wp.sim.XPBDIntegrator(10)
    import warp.sim.render
    renderer = wp.sim.render.SimRenderer(model, 'femur.usd')

    sim_time = 0.0
    sim_total_seconds = 2
    frame_dt = 1.0 / (fps := [15, 5])[0]
    sim_dt = frame_dt / (sim_steps := 20)

    points = wp.array(prothesis_v_i[0], wp.vec3)
    mesh = wp.Mesh(wp.array(prothesis_v_i[0], wp.vec3), wp.array(prothesis_v_i[1].flatten(), wp.int32))

    size = np.array(femur_region.shape)
    center = femur_region_origin + size * iso_spacing * 0.5
    size = (size * 2 + 8) // 16 * 16
    length = size * iso_spacing
    origin = center - length * 0.5
    origin[1] = main_region_origin[1]
    main_region_length = main_region_max - main_region_min

    image = wp.full(shape=(size[0], size[2]), value=wp.vec3(), dtype=wp.vec3)

    frames = []

    for sec in range(sim_total_seconds):
        gr.Info(f'模拟置入 {sec + 1}/{sim_total_seconds} 秒')

        for frame in range(fps[0]):
            for _ in range(sim_steps):
                wp.sim.collide(model, state_0)

                state_0.clear_forces()
                state_1.clear_forces()

                state_0.body_f.assign([[0.0, 0.0, 0.0, *(canal_z * -9.8e6)]])

                integrator.simulate(model, state_0, state_1, sim_dt)
                state_0, state_1 = state_1, state_0

            renderer.begin_frame(sim_time)
            renderer.render(state_0)
            renderer.end_frame()

            sim_time += frame_dt

            wp.launch(kernel=tl.kernel.transform_body, dim=mesh.points.shape, inputs=[
                state_0.body_q, ps_body, points, mesh.points,
            ])
            mesh.refit()

            wp.launch(kernel=tl.kernel.prothesis_render, dim=image.shape, inputs=[
                init_volume.id, wp.vec3(init_volume_origin), init_volume_spacing, bone_threshold,
                image, origin, iso_spacing, main_region_length[1],
                wp.vec3(1, 0, 0), wp.vec3(0, 0, 1), wp.vec3(0, 1, 0),
                main_region_window[0], *main_region_window,
                mesh.id, wp.vec3(np.array(femur_stem_rgb) / 255.0),
                wp.vec3(neck_center), wp.vec3(neck_z),
            ])

            image_ui = np.flipud(image.numpy().astype(np.uint8).swapaxes(0, 1))
            frames.append(image_ui)

            yield {
                _3_femur_sim_output: gr.Image(
                    image_ui, image_mode='RGB', label='股骨柄物理模拟',
                    show_download_button=False, interactive=False, visible=True,
                ),
            }

    iio.imwrite('psf.mp4', frames, fps=fps[1])
    renderer.save()

    gr.Success(f'模拟完成')


def on_0_tab():
    return {
        _0_save: gr.Button(f'保存 ({init_filename if init_filename is not None else str()}.save)', visible=True, ),
        _0_select: gr.FileExplorer(file_count='single', root_dir=tl.save_dir, label='请选择文件', visible=True, ),
        _0_load: gr.Button('载入', visible=True, ),
        _0_upload: gr.UploadButton('上传 (.nii.gz)', visible=True, ),
        _0_unload: gr.Button('重置', visible=True, ),
    }


def on_1_tab():
    if init_array is None:
        return {ui: ui.__class__(visible=False) for ui in all_ui[1]}

    global main_region_min, main_region_max, main_region_size, main_region_origin
    main_region_length = main_region_max - main_region_min
    main_region_size = (main_region_length / iso_spacing).astype(int)
    main_region_size = np.max([main_region_size, [1, 1, 1]], axis=0)
    main_region_origin = init_volume_origin + main_region_min

    image = [wp.full(shape=(main_region_size[0], main_region_size[1]), value=0, dtype=wp.uint8),
             wp.full(shape=(main_region_size[0], main_region_size[2]), value=0, dtype=wp.uint8)]

    with wp.ScopedTimer('on_1_image', print=False) as t:
        wp.launch(kernel=tl.kernel.volume_ray_parallel, dim=image[0].shape, inputs=[
            init_volume.id, wp.vec3(init_volume_origin), wp.vec3(init_volume_spacing),
            image[0], wp.vec3(main_region_origin), iso_spacing,
            wp.vec3(1, 0, 0), wp.vec3(0, 1, 0), wp.vec3(0, 0, 1),
            main_region_length[2], main_region_window[0], *main_region_window,
        ])

        wp.launch(kernel=tl.kernel.volume_ray_parallel, dim=image[1].shape, inputs=[
            init_volume.id, wp.vec3(init_volume_origin), wp.vec3(init_volume_spacing),
            image[1], wp.vec3(main_region_origin), iso_spacing,
            wp.vec3(1, 0, 0), wp.vec3(0, 0, 1), wp.vec3(0, 1, 0),
            main_region_length[1], main_region_window[0], *main_region_window,
        ])

    gr.Success(f'透视成功 {t.elapsed:.1f} ms')

    global main_image_xy, main_image_xz
    main_image_xy = image[0].numpy()
    main_image_xz = image[1].numpy()

    main_image_xy_ui = main_image_xy.swapaxes(0, 1)
    main_image_xz_ui = np.flipud(main_image_xz.swapaxes(0, 1))

    return {
        _1_main_image_xy: gr.Image(
            main_image_xy_ui, image_mode='L',
            show_label=False, show_fullscreen_button=False,
            show_download_button=False, interactive=False, visible=True,
        ),
        _1_main_region_a: gr.Slider(
            0,
            round(float(init_volume_length[1])),
            float(main_region_min[1]),
            step=iso_spacing,
            label='前', visible=True,
        ),
        _1_main_region_p: gr.Slider(
            0,
            round(float(init_volume_length[1])),
            float(main_region_max[1]),
            step=iso_spacing,
            label='后', visible=True,
        ),
        _1_main_image_xz: gr.Image(
            main_image_xz_ui, image_mode='L',
            show_label=False, show_fullscreen_button=False,
            show_download_button=False, interactive=False, visible=True,
        ),
        _1_main_region_r: gr.Slider(
            0,
            round(float(init_volume_length[0])),
            float(main_region_min[0]),
            step=iso_spacing,
            label='右', visible=True,
        ),
        _1_main_region_l: gr.Slider(
            0,
            round(float(init_volume_length[0])),
            float(main_region_max[0]),
            step=iso_spacing,
            label='左', visible=True,
        ),
        _1_main_region_i: gr.Slider(
            0,
            round(float(init_volume_length[2])),
            float(main_region_min[2]),
            step=iso_spacing,
            label='下', visible=True,
        ),
        _1_main_region_s: gr.Slider(
            0,
            round(float(init_volume_length[2])),
            float(main_region_max[2]),
            step=iso_spacing,
            label='上', visible=True,
        ),
    }


def on_2_tab():
    if main_image_xz is None:
        return {ui: ui.__class__(visible=False) for ui in all_ui[2]}

    kp_image_xz_ui = main_image_xz.copy()
    kp_image_xz_ui = np.tile(kp_image_xz_ui[:, :, np.newaxis], (1, 1, 3))

    if (p := kp_positions.get(kp_name)) is not None:
        if len(p) == 3:
            x = round((p[0] - main_region_origin[0]) / iso_spacing)
            y = round((p[1] - main_region_origin[1]) / iso_spacing)
            z = round((p[2] - main_region_origin[2]) / iso_spacing)
        else:
            x = round((p[0] - main_region_origin[0]) / iso_spacing)
            y = None
            z = round((p[1] - main_region_origin[2]) / iso_spacing)

        kp_image_xz_ui[x, :] = kp_select_rgb
        kp_image_xz_ui[:, z] = kp_select_rgb

        origin = main_region_origin + np.array([0, 0, z * iso_spacing])

        with wp.ScopedTimer('', print=False) as t:
            kp_image_xy_ui = wp.full(shape=(main_region_size[0], main_region_size[1]), value=0, dtype=wp.uint8)
            wp.launch(kernel=tl.kernel.volume_slice, dim=kp_image_xy_ui.shape, inputs=[
                init_volume.id, wp.vec3(init_volume_origin), wp.vec3(init_volume_spacing),
                kp_image_xy_ui, wp.vec3(origin), iso_spacing,
                wp.vec3(1, 0, 0), wp.vec3(0, 1, 0),
                *main_region_window,
            ])

        gr.Success(f'切片成功 {t.elapsed:.1f} ms')

        kp_image_xy_ui = kp_image_xy_ui.numpy()
        kp_image_xy_ui = np.tile(kp_image_xy_ui[:, :, np.newaxis], (1, 1, 3))
        kp_image_xy_ui[x, :] = kp_select_rgb

        if y is not None:
            kp_image_xy_ui[:, y] = kp_select_rgb

        kp_image_xy_ui = kp_image_xy_ui.swapaxes(0, 1)
    elif kp_name in kp_names:
        kp_image_xy_ui = None
    else:
        for p in kp_positions.values():
            if len(p) == 3:
                x = round((p[0] - main_region_origin[0]) / iso_spacing)
                z = round((p[2] - main_region_origin[2]) / iso_spacing)
            else:
                x = round((p[0] - main_region_origin[0]) / iso_spacing)
                z = round((p[1] - main_region_origin[2]) / iso_spacing)

            x_min = min(max(x - kp_deselect_radius, 0), kp_image_xz_ui.shape[0])
            x_max = min(max(x + kp_deselect_radius, 0), kp_image_xz_ui.shape[0])
            z_min = min(max(z - kp_deselect_radius, 0), kp_image_xz_ui.shape[1])
            z_max = min(max(z + kp_deselect_radius, 0), kp_image_xz_ui.shape[1])
            kp_image_xz_ui[x_min:x_max + 1, z_min:z_max + 1] = kp_deselect_rgb
            kp_image_xz_ui[x_min:x_max + 1, z_min:z_max + 1] = kp_deselect_rgb
        kp_image_xy_ui = None

    kp_image_xz_ui = np.flipud(kp_image_xz_ui.swapaxes(0, 1))

    return {
        _2_op_side: gr.Radio(
            _ := ['右侧', '左侧'],
            value=_[int(xinv > 0)], label='术侧', type='index', visible=True,
        ),
        _2_kp_name: gr.Radio(
            [kp_name_none, *kp_names],
            value=kp_name, label='解剖标志', visible=True,
        ),
        _2_kp_image_xz: gr.Image(
            kp_image_xz_ui, image_mode='RGB', label='正位透视',
            show_download_button=False, interactive=False, visible=True,
        ),
        _2_kp_image_xy: gr.Image(
            kp_image_xy_ui, image_mode='RGB', label='轴位切片',
            show_download_button=False, interactive=False, visible=True,
        ),
        _2_kp_positions: gr.Json(
            kp_positions, visible=True,
        ),
    }


def on_3_tab():
    global taper_center, neck_center, neck_x, neck_y, neck_z, canal_x, canal_y, canal_z

    taper_center = kp_positions.get('股骨柄颈锥圆心')
    neck = [kp_positions.get(_) for _ in ('股骨颈口上缘', '股骨颈口下缘')]
    canal = [kp_positions.get(_) for _ in ('股骨小粗隆髓腔中心', '股骨柄末端髓腔中心')]

    if any([_ is None for _ in [taper_center] + neck + canal]):
        return {ui: ui.__class__(visible=False) for ui in all_ui[3]}

    taper_center = np.array(taper_center)
    neck = np.array(neck)
    canal = np.array(canal)

    neck_center = 0.5 * (neck[0] + neck[1])
    neck_x = neck[0] - neck[1]
    neck_rx = 0.5 * np.linalg.norm(neck_x)
    neck_ry = 0.5 * neck_rx
    neck_z = neck_center - canal[0]
    neck_y = np.cross(neck_z, neck_x)
    neck_z = np.cross(neck_x, neck_y)
    neck_x, neck_y, neck_z = [_ / np.linalg.norm(_) for _ in (neck_x, neck_y, neck_z)]

    canal_z = canal[0] - canal[1]
    canal_x = canal[0] - neck[1]
    canal_y = np.cross(canal_z, canal_x)
    canal_x = np.cross(canal_y, canal_z)
    canal_x, canal_y, canal_z = [_ / np.linalg.norm(_) for _ in (canal_x, canal_y, canal_z)]

    gr.Info('股骨柄建模')

    canal_x_edges = [
        canal[0], 0.7 * canal[0] + 0.3 * canal[1], 0.4 * canal[0] + 0.6 * canal[1], canal[1],
        canal[0], 0.7 * canal[0] + 0.3 * canal[1], 0.4 * canal[0] + 0.6 * canal[1], canal[1],
    ]
    ray_dirs = [
        canal_x, canal_x, canal_x, canal_x,
        -canal_x, -canal_x, -canal_x, -canal_x,
    ]

    with wp.ScopedTimer('', print=False) as t:
        canal_x_edges = wp.array(canal_x_edges, dtype=wp.vec3)
        wp.launch(kernel=tl.kernel.volume_query_rays, dim=(len(canal_x_edges),), inputs=[
            init_volume.id, wp.vec3(init_volume_origin), wp.vec3(init_volume_spacing),
            canal_x_edges, wp.array(ray_dirs, dtype=wp.vec3),
            iso_spacing, 1e6, bone_threshold,
            canal_x_edges,
        ])
        canal_x_edges = canal_x_edges.numpy() - np.array(ray_dirs) * iso_spacing

        canal_x_edges = canal_x_edges.reshape((2, -1, 3)).astype(float)
        canal_rx = 0.5 * np.linalg.norm(canal_x_edges[0, :] - canal_x_edges[1, :], axis=1)
        canal_ry = 0.8 * canal_rx

        canal_centers = np.mean(canal_x_edges, axis=0)
        canal_y_edges = np.array([
            [canal_centers[_] + canal_y * canal_ry[_] for _ in range(canal_x_edges.shape[1])],
            [canal_centers[_] - canal_y * canal_ry[_] for _ in range(canal_x_edges.shape[1])],
        ])

        splines = np.array([
            [
                _ := neck[0],
                _ * 0.9 + canal_x_edges[0, 0] * 0.1 + canal_x * 2,
                canal_x_edges[0, 0],
                canal_x_edges[0, 1],
                canal_x_edges[0, 2],
                canal_x_edges[0, 3] + canal_z * 2,
                canal_x_edges[0, 3],
            ],
            [
                _ := neck_center + canal_y * neck_ry,
                _ * 0.9 + canal_y_edges[0, 0] * 0.1,
                canal_y_edges[0, 0],
                canal_y_edges[0, 1],
                canal_y_edges[0, 2],
                canal_y_edges[0, 3] + canal_z * 2,
                canal_y_edges[0, 3],
            ],
            [
                _ := neck[1],
                _ * 0.9 + canal_x_edges[1, 0] * 0.1,
                canal_x_edges[1, 0],
                canal_x_edges[1, 1],
                canal_x_edges[1, 2],
                canal_x_edges[1, 3] + canal_z * 2,
                canal_x_edges[1, 3],
            ],
            [
                _ := neck_center - canal_y * neck_ry,
                _ * 0.9 + canal_y_edges[1, 0] * 0.1,
                canal_y_edges[1, 0],
                canal_y_edges[1, 1],
                canal_y_edges[1, 2],
                canal_y_edges[1, 3] + canal_z * 2,
                canal_y_edges[1, 3],
            ],
        ])

        global prothesis_v_i
        prothesis_v_i = tl.mesh.femoral_prothesis(splines, taper_center, neck_x)
        trimesh.Trimesh(*prothesis_v_i[:2]).export('prothesis.stl')

        mesh = wp.Mesh(wp.array(prothesis_v_i[0], wp.vec3), wp.array(prothesis_v_i[1].flatten(), wp.int32),
                       support_winding_number=True)

        gr.Info('股骨柄透视')

        global main_region_min, main_region_max, main_region_size, main_region_origin
        main_region_length = main_region_max - main_region_min

        femur_image_xz_ui = wp.full(shape=(main_region_size[0], main_region_size[2]), value=0, dtype=wp.uint8)
        wp.launch(kernel=tl.kernel.volume_ray_parallel, dim=femur_image_xz_ui.shape, inputs=[
            init_volume.id, wp.vec3(init_volume_origin), wp.vec3(init_volume_spacing),
            femur_image_xz_ui, wp.vec3(main_region_origin), iso_spacing,
            wp.vec3(1, 0, 0), wp.vec3(0, 0, 1), wp.vec3(0, 1, 0),
            main_region_length[1], main_region_window[0], *main_region_window,
        ])

        femur_image_xz_ui = femur_image_xz_ui.numpy()
        femur_image_xz_ui = np.tile(femur_image_xz_ui[:, :, np.newaxis], (1, 1, 3))
        femur_image_xz_ui = wp.array(femur_image_xz_ui, dtype=wp.vec3)

        wp.launch(kernel=tl.kernel.mesh_ray_parallel, dim=femur_image_xz_ui.shape, inputs=[
            mesh.id, wp.vec3(np.array(femur_stem_rgb) / 255.0),
            femur_image_xz_ui, wp.vec3(main_region_origin), iso_spacing,
            wp.vec3(1, 0, 0), wp.vec3(0, 0, 1), wp.vec3(0, 1, 0),
            main_region_length[1],
        ])

        femur_image_xz_ui = np.flipud(femur_image_xz_ui.numpy().astype(np.uint8).swapaxes(0, 1))

        gr.Info('股骨柄截面')

        AP, AB = neck[1] - canal[0], canal[1] - canal[0]
        P = canal[0] + np.dot(AP, AB) / np.dot(AB, AB) * AB

        femur_image_xy_ui = []
        rx, ry = 40.0, 24.0
        for i in range(20):
            i = i / 20.0
            c = P * (1 - i) + canal[1] * i
            o = c - rx * canal_x * xinv - ry * canal_y

            size = tuple(max(round(2 * _ / iso_spacing), 1) for _ in (rx, ry))
            image = wp.full(shape=size, value=0, dtype=wp.uint8)
            wp.launch(kernel=tl.kernel.volume_slice, dim=image.shape, inputs=[
                init_volume.id, wp.vec3(init_volume_origin), wp.vec3(init_volume_spacing),
                image, wp.vec3(o), iso_spacing,
                wp.vec3(canal_x * xinv), wp.vec3(canal_y),
                *main_region_window,
            ])

            image = image.numpy()
            image = np.tile(image[:, :, np.newaxis], (1, 1, 3))
            image = wp.array(image, dtype=wp.vec3)

            wp.launch(kernel=tl.kernel.mesh_slice, dim=image.shape, inputs=[
                mesh.id, 1e6,
                wp.vec3(np.array(femur_stem_rgb) / 255.0), 0.5,
                image, wp.vec3(o), iso_spacing,
                wp.vec3(canal_x * xinv), wp.vec3(canal_y),
            ])

            femur_image_xy_ui.append(image.numpy().astype(np.uint8).swapaxes(0, 1))

        gr.Info('股骨三维重建')

        global femur_region, femur_region_size, femur_region_origin
        box = np.min(prothesis_v_i[0], axis=0), np.max(prothesis_v_i[0], axis=0)
        l = box[1] - box[0]
        l = np.array([l[0] * 2.0, l[1] * 2.0, l[2]])
        femur_region_size = np.round(l / iso_spacing)
        femur_region_origin = np.mean(box, axis=0) - femur_region_size * iso_spacing * 0.5

        bg_value = np.min(init_array)
        femur_region = wp.full(shape=(*femur_region_size,), value=bg_value, dtype=wp.float32)
        wp.launch(kernel=tl.kernel.volume_sample, dim=femur_region.shape, inputs=[
            init_volume.id, wp.vec3(init_volume_origin), wp.vec3(init_volume_spacing),
            femur_region, wp.vec3(femur_region_origin), iso_spacing,
            wp.vec3(1, 0, 0), wp.vec3(0, 1, 0), wp.vec3(0, 0, 1),
            True,
        ])

        wp.launch(kernel=tl.kernel.volume_clip_plane, dim=femur_region.shape, inputs=[
            femur_region, wp.vec3(femur_region_origin), iso_spacing,
            wp.vec3(neck_center), wp.vec3(-neck_z), bone_threshold,
        ])

        wp.launch(kernel=tl.kernel.volume_clip_plane, dim=femur_region.shape, inputs=[
            femur_region, wp.vec3(femur_region_origin), iso_spacing,
            wp.vec3(neck_center), wp.vec3(-canal_z), bone_threshold,
        ])

        global femur_v_i
        femur_v_i = tl.kernel.volume_dual_marching_cubes(femur_region, iso_spacing, femur_region_origin, bone_threshold)
        mesh = trimesh.Trimesh(femur_v_i[0], femur_v_i[1])
        mesh.fix_normals()
        mesh = max(mesh.split(), key=lambda _: _.volume)
        femur_v_i = mesh.vertices, mesh.faces
        trimesh.Trimesh(*prothesis_v_i[:2]).export('femur.stl')

        gr.Info('准备物理模拟器')

        global femur_smb
        femur_smb = wp.sim.ModelBuilder((0.0, 0.0, 1.0), 0.0)

        femur_smb.add_shape_mesh(
            body=-1,
            mesh=wp.sim.Mesh(femur_v_i[0].tolist(), femur_v_i[1].flatten().tolist()),
            density=1.0,
        )

        ps_origin = canal_z * np.linalg.norm(canal[0] - canal[1])
        ps_origin = wp.transform(ps_origin.tolist(), wp.quat_identity())

        global ps_body
        femur_smb.add_shape_mesh(
            body=(ps_body := femur_smb.add_body(ps_origin, name='prothesis')),  # noqa
            mesh=wp.sim.Mesh(prothesis_v_i[0].tolist(), prothesis_v_i[1].flatten().tolist()),
            density=1.0,
        )

    gr.Success(f'生成股骨柄完成 {t.elapsed:.1f} ms')

    return {
        _3_femur_image_xz: gr.Image(
            femur_image_xz_ui, image_mode='RGB', label='股骨柄透视',
            show_download_button=False, interactive=False, visible=True,
        ),
        _3_femur_image_xy: gr.Gallery(
            femur_image_xy_ui, label='股骨柄截面',
            object_fit='contain', selected_index=0, columns=1, interactive=False, visible=True,
        ),
        _3_femur_sim: gr.Button('开始模拟', visible=True, ),
        _3_femur_sim_output: gr.Image(
            None, image_mode='RGB', label='股骨柄物理模拟',
            show_download_button=False, interactive=False, visible=True,
        ),
    }


if __name__ == '__main__':
    with (gr.Blocks(tl.theme, title=(title := f'{tl.__package__}')) as app):
        with gr.Row():
            gr.Markdown(f'# {title}.{Path(__file__).stem}')
            _0_save = gr.Button()
            gr.Column(scale=1)

        with gr.Tab('载入') as _0_tab:
            gr.Markdown('''
            - 上传源数据，使用 [3D Slicer](https://download.slicer.org/) 等软件解析DICOM并保存3D图像为NIFTI(.nii.gz)格式
            - 载入源数据(.nii.gz)，或存档(.save)
            ''')

            _0_select = gr.FileExplorer()

            with gr.Row():
                _0_load = gr.Button()
                _0_upload = gr.UploadButton()
                _0_unload = gr.Button()

        with gr.Tab('识别主区') as _1_tab:
            gr.Markdown('''
            - 调节包围盒边界，使图像上方包含骨盆，下方包含股骨远端，排除四周干扰物体
            ''')

            with gr.Row():
                with gr.Column(scale=1):
                    _1_main_image_xy = gr.Image()
                with gr.Column(scale=2):
                    with gr.Row():
                        _1_main_region_a = gr.Slider()
                        _1_main_region_p = gr.Slider()

            with gr.Row():
                with gr.Column(scale=1):
                    _1_main_image_xz = gr.Image()
                with gr.Column(scale=2):
                    with gr.Row():
                        _1_main_region_r = gr.Slider()
                        _1_main_region_l = gr.Slider()
                    with gr.Row():
                        _1_main_region_i = gr.Slider()
                        _1_main_region_s = gr.Slider()

        with gr.Tab('识别解剖') as _2_tab:
            gr.Markdown('''
            - 在图像中定位解剖标志，正位透视定位左右(X)上下(Z)坐标，轴位切片定位左右(X)前后(Y)坐标
            ''')

            _2_op_side = gr.Radio()
            _2_kp_name = gr.Radio()

            with gr.Row():
                with gr.Column(scale=1):
                    _2_kp_image_xz = gr.Image()

                with gr.Column(scale=1):
                    _2_kp_image_xy = gr.Image()
                    _2_kp_positions = gr.Json()

        with gr.Tab('生成股骨柄') as _3_tab:
            gr.Markdown('''
            - 根据股骨颈口、髓腔形态，参数化自动生成相匹配的股骨柄假体外形
            ''')

            _3_femur_image_xz = gr.Image()

            with gr.Row():
                _3_femur_image_xy = gr.Gallery()

                with gr.Column():
                    _3_femur_sim = gr.Button()
                    _3_femur_sim_output = gr.Image()

        # 控件集合
        all_ui = [[ui for name, ui in globals().items() if name.startswith(f'_{_}_') and 'tab' not in name]
                  for _ in range(4)]

        _0_save.click(  # 保存
            on_0_save,
        ).then(
            lambda: gr.FileExplorer(root_dir=tl.fs_dir), None, _0_select,
        ).then(
            lambda: gr.FileExplorer(root_dir=tl.save_dir), None, _0_select,
        )

        _0_upload.upload(  # 上传
            fn=on_0_upload,
            inputs=_0_upload,
            outputs=None,
            trigger_mode='once',
        ).then(
            lambda: gr.FileExplorer(root_dir=tl.fs_dir), None, _0_select,
        ).then(
            lambda: gr.FileExplorer(root_dir=tl.save_dir), None, _0_select,
        )

        _0_load.click(  # 载入
            on_0_load, _0_select, trigger_mode='once',
        ).success(
            on_0_tab, None, all_ui[0],
        ).success(
            on_1_tab, None, all_ui[1],
        ).success(
            on_2_tab, None, all_ui[2],
        ).success(
            on_3_tab, None, all_ui[3],
        )

        _0_unload.click(  # 重启
            on_0_unload, None, trigger_mode='once',
        ).success(
            on_0_tab, None, all_ui[0],
        ).success(
            on_1_tab, None, all_ui[1],
        ).success(
            on_2_tab, None, all_ui[2],
        ).success(
            on_3_tab, None, all_ui[3],
        )

        for ui in (_ := [_1_main_region_r, _1_main_region_a, _1_main_region_i,
                         _1_main_region_l, _1_main_region_p, _1_main_region_s]):
            ui.input(
                on_1_main_region, _, trigger_mode='once',
            ).success(on_1_tab, None, all_ui[1])

        _2_op_side.select(  # 术侧
            on_2_op_side, _2_op_side, trigger_mode='once',
        ).success(on_2_tab, None, all_ui[2])

        _2_kp_name.select(  # 解剖标志
            on_2_kp_name, _2_kp_name, trigger_mode='once',
        ).success(on_2_tab, None, all_ui[2])

        _2_kp_image_xz.select(  # 正位透视
            on_2_image_xz_select, _2_kp_image_xz, trigger_mode='once',
        ).success(on_2_tab, None, all_ui[2])

        _2_kp_image_xy.select(  # 轴位切片
            on_2_image_xy_select, _2_kp_image_xy, trigger_mode='once',
        ).success(on_2_tab, None, all_ui[2])

        _3_femur_sim.click(on_femur_sim, None, [_3_femur_sim_output])

        # 标签页
        _0_tab.select(on_0_tab, None, all_ui[0])
        _1_tab.select(on_1_tab, None, all_ui[1])
        _2_tab.select(on_2_tab, None, all_ui[2])
        _3_tab.select(on_3_tab, None, all_ui[3])

        # 刷新网页
        app.load(
            on_0_tab, None, all_ui[0],
        ).success(
            on_1_tab, None, all_ui[1],
        ).success(
            on_2_tab, None, all_ui[2],
        ).success(
            on_3_tab, None, all_ui[3],
        )

        # 启动网页
        app.launch(share=True, show_api=False, max_file_size=gr.FileSize.GB, pwa=True)
