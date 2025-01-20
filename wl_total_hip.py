import os
import shutil
from pathlib import Path
from typing import Optional

import gradio as gr
import itk
import numpy as np
import warp as wp

import towl as tl
import towl.save_pb2 as pb

init_filename: Optional[str] = None

init_volume: Optional[np.ndarray] = None
init_volume_wp: Optional[wp.Volume] = None

init_volume_size: Optional[np.ndarray] = None
init_volume_spacing: Optional[np.ndarray] = None
init_volume_origin: Optional[np.ndarray] = None
init_volume_length: Optional[np.ndarray] = None

main_region_spacing = 0.5
main_region_window = [-100.0, 900.0]

main_region_min: Optional[np.ndarray] = None
main_region_max: Optional[np.ndarray] = None
main_region_size: Optional[np.ndarray] = None
main_region_origin: Optional[np.ndarray] = None

main_image_xy: Optional[np.ndarray] = None
main_image_xz: Optional[np.ndarray] = None

kp_names = ['髂前上棘', '股骨头中心']
kp_names = [f'左侧{_}' for _ in kp_names] + [f'右侧{_}' for _ in kp_names]
kp_names += ['骶骨上终板中心']

kp_select_rgb = [255, 127, 127]
kp_deselect_rgb = [0, 127, 255]
kp_deselect_radius = 5

kp_name_none = '取消选中'
kp_name: str = kp_name_none
kp_positions = {}

kp_image_xz: Optional[np.ndarray] = None
kp_image_xy: Optional[np.ndarray] = None


def on_0_save():
    if init_filename is None:
        raise gr.Error('未载入源数据或存档')

    gr.Info('开始保存')
    save = pb.SaveTotalHip()

    if init_volume is not None:
        save.init_volume.CopyFrom(pb.Volume(
            dtype=pb.DataType.INT16,
            data=init_volume.tobytes(),
            background=np.min(init_volume),
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
        raise gr.Error('未选中')

    f = Path(filename)
    if not f.is_file() or not f.exists():
        raise gr.Error('选中无效')

    global init_filename, init_volume, init_volume_wp
    global init_volume_size, init_volume_spacing, init_volume_origin, init_volume_length
    global main_region_min, main_region_max
    global kp_name, kp_positions

    if f.name.lower().endswith(suffix := '.save'):
        gr.Info('正在载入存档')

        try:
            save: pb.SaveTotalHip = pb.SaveTotalHip.FromString(f.read_bytes())
        except Exception as e:
            raise gr.Error(f'载入存档失败 {e}')

        if save.HasField('init_volume'):
            _ = [*save.init_volume.region.size]
            init_volume = np.frombuffer(save.init_volume.data, np.int16).reshape(_)
            init_volume_size = np.array(save.init_volume.region.size)
            init_volume_spacing = np.array(save.init_volume.region.spacing)
            init_volume_origin = np.array(save.init_volume.region.origin)
            init_volume_length = init_volume_size * init_volume_spacing

        if save.HasField('main_region'):
            main_region_min = np.array(save.main_region.min.values)
            main_region_max = np.array(save.main_region.max.values)

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
                raise gr.Error(f'仅支持3D图像，不支持{image.ndim}D图像')

            if np.any((direction := np.array(image.GetDirection())) != np.identity(3)):
                gr.Warning(f'忽略已知的图像歪斜\n{direction}')

            init_volume_size = np.array(itk.size(image))
            init_volume_spacing = np.array(itk.spacing(image))
            init_volume_origin = np.array(itk.origin(image))
            init_volume_length = init_volume_size * init_volume_spacing

            init_volume = itk.array_from_image(image).swapaxes(0, 2).copy()

            main_region_min = np.zeros(3)
            main_region_max = init_volume_length.copy()

        init_filename = f.name.removesuffix(suffix)
        gr.Success(f'载入成功 {t.elapsed} ms')
    else:
        raise gr.Error(f'载入失败，未知文件 {f.name}')


def on_1_main_region(r, a, i, l, p, s):
    global main_region_min, main_region_max
    main_region_min = np.min([[r, a, i], [l, p, s]], axis=0)
    main_region_max = np.max([[r, a, i], [l, p, s]], axis=0)


def on_2_kp_name(name):
    global kp_name
    kp_name = name


def on_2_image_xz_select(evt: gr.SelectData, image):
    if kp_name not in kp_names:
        raise gr.Error('未选中解剖标志')

    p0 = main_region_origin[0] + main_region_spacing * evt.index[0]
    p2 = main_region_origin[2] + main_region_spacing * (image.shape[0] - evt.index[1])
    kp_positions[kp_name] = [p0, p2]


def on_2_image_xy_select(evt: gr.SelectData, _):
    if kp_name not in kp_names:
        raise gr.Error('未选中解剖标志')

    global kp_positions
    if kp_name not in kp_positions:
        raise gr.Error(f'未先选透视点 {kp_name}')

    p1 = main_region_origin[1] + main_region_spacing * evt.index[1]

    if len(p := kp_positions[kp_name]) == 2:
        kp_positions[kp_name] = [p[0], p1, p[1]]
    elif len(p) == 3:
        kp_positions[kp_name] = [p[0], p1, p[2]]
    else:
        raise gr.Error(f'数据错误 {kp_name} {p}')


def on_0_tab():
    return {
        _0_save: gr.Button(f'保存 ({init_filename if init_filename is not None else str()}.save)'),
        _0_save_select: gr.FileExplorer(file_count='single', root_dir=tl.save_dir, label='请选择文件'),
        _0_save_load: gr.Button('载入'),
        _0_save_upload: gr.UploadButton('上传 (.nii.gz)'),
    }


def on_1_tab():
    if init_volume is None:
        return {ui: gr.update() for ui in all_ui[1]}

    global main_region_min, main_region_max, main_region_size, main_region_origin
    main_region_length = main_region_max - main_region_min
    main_region_size = (main_region_length / main_region_spacing).astype(int)
    main_region_size = np.max([main_region_size, [1, 1, 1]], axis=0)
    main_region_origin = init_volume_origin + main_region_min

    image = [wp.full(shape=(main_region_size[0], main_region_size[1]), value=0, dtype=wp.uint8),
             wp.full(shape=(main_region_size[0], main_region_size[2]), value=0, dtype=wp.uint8)]

    global init_volume_wp
    with wp.ScopedTimer('on_1_image', print=False) as t:
        if init_volume_wp is None:
            init_volume_wp = wp.Volume.load_from_numpy(init_volume, bg_value=np.min(init_volume))

        wp.launch(kernel=tl.kernel.volume_xray_parallel, dim=image[0].shape, inputs=[
            init_volume_wp.id, image[0],
            wp.vec3(init_volume_origin), wp.vec3(init_volume_spacing),
            wp.vec3(main_region_origin), main_region_spacing,
            wp.vec3(1, 0, 0), wp.vec3(0, 1, 0), wp.vec3(0, 0, 1),
            main_region_length[2], main_region_window[0], *main_region_window,
        ])

        wp.launch(kernel=tl.kernel.volume_xray_parallel, dim=image[1].shape, inputs=[
            init_volume_wp.id, image[1],
            wp.vec3(init_volume_origin), wp.vec3(init_volume_spacing),
            wp.vec3(main_region_origin), main_region_spacing,
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
            show_download_button=False, interactive=False,
        ),
        _1_main_region_a: gr.Slider(
            0,
            round(float(init_volume_length[1])),
            float(main_region_min[1]),
            step=main_region_spacing,
            label='前',
        ),
        _1_main_region_p: gr.Slider(
            0,
            round(float(init_volume_length[1])),
            float(main_region_max[1]),
            step=main_region_spacing,
            label='后',
        ),
        _1_main_image_xz: gr.Image(
            main_image_xz_ui, image_mode='L',
            show_label=False, show_fullscreen_button=False,
            show_download_button=False, interactive=False,
        ),
        _1_main_region_r: gr.Slider(
            0,
            round(float(init_volume_length[0])),
            float(main_region_min[0]),
            step=main_region_spacing,
            label='右',
        ),
        _1_main_region_l: gr.Slider(
            0,
            round(float(init_volume_length[0])),
            float(main_region_max[0]),
            step=main_region_spacing,
            label='左',
        ),
        _1_main_region_i: gr.Slider(
            0,
            round(float(init_volume_length[2])),
            float(main_region_min[2]),
            step=main_region_spacing,
            label='下',
        ),
        _1_main_region_s: gr.Slider(
            0,
            round(float(init_volume_length[2])),
            float(main_region_max[2]),
            step=main_region_spacing,
            label='上',
        ),
    }


def on_2_tab():
    if main_image_xz is None:
        return {ui: gr.update() for ui in all_ui[2]}

    kp_image_xz_ui = main_image_xz.copy()
    kp_image_xz_ui = np.tile(kp_image_xz_ui[:, :, np.newaxis], (1, 1, 3))

    if (p := kp_positions.get(kp_name)) is not None:
        if len(p) == 3:
            x = round((p[0] - main_region_origin[0]) / main_region_spacing)
            y = round((p[1] - main_region_origin[1]) / main_region_spacing)
            z = round((p[2] - main_region_origin[2]) / main_region_spacing)
        else:
            x = round((p[0] - main_region_origin[0]) / main_region_spacing)
            y = None
            z = round((p[1] - main_region_origin[2]) / main_region_spacing)

        kp_image_xz_ui[x, :] = kp_select_rgb
        kp_image_xz_ui[:, z] = kp_select_rgb

        origin = main_region_origin + np.array([0, 0, z * main_region_spacing])

        global init_volume_wp
        with wp.ScopedTimer('', print=False) as t:
            if init_volume_wp is None:
                init_volume_wp = wp.Volume.load_from_numpy(init_volume, bg_value=np.min(init_volume))

            kp_image_xy_ui = wp.full(shape=(main_region_size[0], main_region_size[1]), value=0, dtype=wp.uint8)
            wp.launch(kernel=tl.kernel.volume_slice, dim=kp_image_xy_ui.shape, inputs=[
                init_volume_wp.id, kp_image_xy_ui,
                wp.vec3(init_volume_origin), wp.vec3(init_volume_spacing),
                wp.vec3(origin), main_region_spacing,
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
                x = round((p[0] - main_region_origin[0]) / main_region_spacing)
                z = round((p[2] - main_region_origin[2]) / main_region_spacing)
            else:
                x = round((p[0] - main_region_origin[0]) / main_region_spacing)
                z = round((p[1] - main_region_origin[2]) / main_region_spacing)

            x_min = min(max(x - kp_deselect_radius, 0), kp_image_xz_ui.shape[0])
            x_max = min(max(x + kp_deselect_radius, 0), kp_image_xz_ui.shape[0])
            z_min = min(max(z - kp_deselect_radius, 0), kp_image_xz_ui.shape[1])
            z_max = min(max(z + kp_deselect_radius, 0), kp_image_xz_ui.shape[1])
            kp_image_xz_ui[x_min:x_max+1, z_min:z_max+1] = kp_deselect_rgb
            kp_image_xz_ui[x_min:x_max+1, z_min:z_max+1] = kp_deselect_rgb
        kp_image_xy_ui = None

    kp_image_xz_ui = np.flipud(kp_image_xz_ui.swapaxes(0, 1))

    return {
        _2_kp_name: gr.Radio(
            [kp_name_none, *kp_names],
            value=kp_name,
            label='解剖标志',
        ),
        _2_kp_image_xz: gr.Image(
            kp_image_xz_ui, image_mode='RGB', label='正位透视',
            show_download_button=False, interactive=False,
        ),
        _2_kp_image_xy: gr.Image(
            kp_image_xy_ui, image_mode='RGB', label='轴位切片',
            show_download_button=False, interactive=False,
        ),
        _2_kp_positions: gr.Json(
            kp_positions,
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

            _0_save_select = gr.FileExplorer()

            with gr.Row():
                _0_save_load = gr.Button()
                _0_save_upload = gr.UploadButton()
                gr.Column(scale=1)

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
            - 在图像中定位解剖标志，正位透视定位左右(X)上下(Z)坐标，轴位切片定位前后(Y)坐标
            ''')

            _2_kp_name = gr.Radio()

            with gr.Row():
                with gr.Column(scale=1):
                    _2_kp_image_xz = gr.Image()

                with gr.Column(scale=1):
                    _2_kp_image_xy = gr.Image()
                    _2_kp_positions = gr.Json()

        # 控件集合
        all_ui = [[ui for name, ui in globals().items() if name.startswith(f'_{_}_')] for _ in range(3)]

        _0_save.click(  # 保存
            on_0_save,
        ).then(
            lambda: gr.FileExplorer(root_dir=tl.fs_dir), None, _0_save_select,
        ).then(
            lambda: gr.FileExplorer(root_dir=tl.save_dir), None, _0_save_select,
        )

        _0_save_upload.upload(  # 上传
            fn=on_0_upload,
            inputs=_0_save_upload,
            outputs=None,
            trigger_mode='once',
        ).then(
            lambda: gr.FileExplorer(root_dir=tl.fs_dir), None, _0_save_select,
        ).then(
            lambda: gr.FileExplorer(root_dir=tl.save_dir), None, _0_save_select,
        )

        _0_save_load.click(
            on_0_load, _0_save_select, trigger_mode='once',
        ).success(
            on_0_tab, None, all_ui[0],
        ).success(
            on_1_tab, None, all_ui[1],
        ).success(
            on_2_tab, None, all_ui[2],
        )

        for ui in (_ := [_1_main_region_r, _1_main_region_a, _1_main_region_i,
                         _1_main_region_l, _1_main_region_p, _1_main_region_s]):
            ui.release(
                on_1_main_region, _, trigger_mode='once',
            ).success(on_1_tab, None, all_ui[1])

        _2_kp_name.select(
            on_2_kp_name, _2_kp_name, trigger_mode='once',
        ).success(on_2_tab, None, all_ui[2])
        _2_kp_image_xz.select(
            on_2_image_xz_select, _2_kp_image_xz, trigger_mode='once',
        ).success(on_2_tab, None, all_ui[2])
        _2_kp_image_xy.select(
            on_2_image_xy_select, _2_kp_image_xy, trigger_mode='once',
        ).success(on_2_tab, None, all_ui[2])

        _0_tab.select(on_0_tab, None, all_ui[0])
        _1_tab.select(on_1_tab, None, all_ui[1])
        _2_tab.select(on_2_tab, None, all_ui[2])

        app.load(on_0_tab, None, all_ui[0])
        app.launch(share=False, show_api=False, max_file_size=gr.FileSize.GB, pwa=True)