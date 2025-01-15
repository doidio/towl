import os
import shutil
from pathlib import Path
from typing import Optional

import gradio as gr
import itk
import numpy as np
import warp as wp

import towl
from towl.kernel import volume_xray_parallel, volume_slice

original_bg: Optional[float] = None
original_volume: Optional[wp.Volume] = None
original_size: Optional[np.ndarray] = None
original_spacing: Optional[np.ndarray] = None
original_length: Optional[np.ndarray] = None
original_origin: Optional[np.ndarray] = None

bbox_iso_spacing = 0.5

bbox_origin: Optional[np.ndarray] = None
bbox_size: Optional[np.ndarray] = None

bbox_image_is: Optional[np.ndarray] = None
bbox_image_ap: Optional[np.ndarray] = None

kp_names = ['髂前上棘', '股骨头中心']
kp_names = [f'左侧{_}' for _ in kp_names] + [f'右侧{_}' for _ in kp_names]
kp_names += ['骶骨上终板中心']

kp_positions = {}


def on_0_save():
    gr.Success('保存成功')


def on_0_save_upload(filename):
    filename = Path(filename)
    try:
        itk.imread(filename.as_posix())
    except Exception as e:
        gr.Warning(f'读取失败 {e}')

    os.makedirs(towl.nifti_dir, exist_ok=True)
    shutil.copy(filename, towl.nifti_dir / filename.name)
    gr.Success('上传成功')


def on_0_save_load(filename):
    if filename is None:
        raise gr.Error('未选中')

    f = Path(filename)
    if not f.is_file() or not f.exists():
        raise gr.Error('选中无效')

    gr.Info('正在载入')
    image = itk.imread(f.as_posix(), pixel_type=itk.SS)

    if image.ndim != 3:
        raise gr.Error(f'仅支持3D图像，不支持{image.ndim}D图像')

    if np.any((direction := np.array(image.GetDirection())) != np.identity(3)):
        gr.Warning(f'忽略已知的图像歪斜\n{direction}')

    global original_bg, original_volume, original_size, original_spacing, original_length, original_origin
    original_size = np.array(itk.size(image))
    original_spacing = np.array(itk.spacing(image))
    original_length = original_size * original_spacing
    original_origin = np.array(itk.origin(image))

    with wp.ScopedTimer('Warp volume from numpy'):
        _ = itk.array_from_image(image).swapaxes(0, 2).copy()
        original_volume = wp.Volume.load_from_numpy(_, bg_value=(original_bg := np.min(_)))

    gr.Success('载入成功')

    return [
        *[gr.Slider(
            maximum=round(float(original_length[_])),
            value=0,
            interactive=True,
        ) for _ in range(3)],
        *[gr.Slider(
            maximum=round(float(original_length[_])),
            value=round(float(original_length[_])),
            interactive=True,
        ) for _ in range(3)],
    ]


def on_1_image(bbox_r, bbox_a, bbox_i, bbox_l, bbox_p, bbox_s):
    if original_volume is None:
        raise gr.Error('未载入原始图像')

    bbox_min = np.min([[bbox_r, bbox_a, bbox_i], [bbox_l, bbox_p, bbox_s]], axis=0)
    bbox_max = np.max([[bbox_r, bbox_a, bbox_i], [bbox_l, bbox_p, bbox_s]], axis=0)

    global bbox_origin, bbox_size
    bbox_origin = original_origin + bbox_min
    bbox_size = (np.max([bbox_max - bbox_min, [1, 1, 1]], axis=0) / bbox_iso_spacing).astype(int)

    image = [wp.full(shape=(bbox_size[0], bbox_size[1]), value=0, dtype=wp.uint8),
             wp.full(shape=(bbox_size[0], bbox_size[2]), value=0, dtype=wp.uint8)]

    with wp.ScopedTimer('on_1_image', print=False) as t:
        wp.launch(kernel=volume_xray_parallel, dim=image[0].shape, inputs=[
            original_volume.id, image[0],
            wp.vec3(original_origin), wp.vec3(original_spacing),
            wp.vec3(bbox_origin), bbox_iso_spacing,
            wp.vec3(1, 0, 0), wp.vec3(0, 1, 0), wp.vec3(0, 0, 1),
            original_length[2], 0.0, -100.0, 900.0,
        ])

        wp.launch(kernel=volume_xray_parallel, dim=image[1].shape, inputs=[
            original_volume.id, image[1],
            wp.vec3(original_origin), wp.vec3(original_spacing),
            wp.vec3(bbox_origin), bbox_iso_spacing,
            wp.vec3(1, 0, 0), wp.vec3(0, 0, 1), wp.vec3(0, 1, 0),
            original_length[1], 0.0, -100.0, 900.0,
        ])

    gr.Success(f'透视成功 {t.elapsed:.1f} ms')

    global bbox_image_is, bbox_image_ap
    bbox_image_is = image[0].numpy()
    bbox_image_ap = image[1].numpy()

    return [
        gr.Image(bbox_image_is.swapaxes(0, 1)),
        gr.Image(np.flipud(bbox_image_ap.swapaxes(0, 1))),
    ]


def on_2_image_ap_select(evt: gr.SelectData, image, name):
    if name not in kp_names:
        raise gr.Error('未选中解剖标志')

    kp_positions[name] = [evt.index[0], image.shape[0] - evt.index[1]]


def on_2_image_ap_update(name):
    image = bbox_image_ap.copy()

    if (p := kp_positions.get(name)) is not None:
        if len(p) == 2:
            p = [round(p[0]), round(p[1])]
        elif len(p) == 3:
            p = [round(p[0]), round(p[2])]
        else:
            raise gr.Error(f'数据错误 {name} {p}')

        image[p[0], :] = 255
        image[:, p[1]] = 255

    return gr.Image(np.flipud(image.swapaxes(0, 1)))


def on_2_image_axial_select(evt: gr.SelectData, _, name):
    if name not in kp_names:
        raise gr.Error('未选中解剖标志')

    global kp_positions
    if name not in kp_positions:
        raise gr.Error(f'未先选透视点 {name}')

    if len(p := kp_positions[name]) == 2:
        kp_positions[name] = [p[0], evt.index[1], p[1]]
    elif len(p) == 3:
        kp_positions[name] = [p[0], evt.index[1], p[2]]
    else:
        raise gr.Error(f'数据错误 {name} {p}')


def on_2_image_axial_update(name):
    image = wp.full(shape=(bbox_size[0], bbox_size[1]), value=0, dtype=wp.uint8)

    if (p := kp_positions.get(name)) is None:
        return gr.Image(image.numpy().swapaxes(0, 1))

    if len(p) == 2:
        z = round(p[1])
    elif len(p) == 3:
        z = round(p[2])
    else:
        raise gr.Error(f'数据错误 {name} {p}')

    origin = bbox_origin + np.array([0, 0, z * bbox_iso_spacing])

    with wp.ScopedTimer('on_2_image_axial', print=False) as t:
        wp.launch(kernel=volume_slice, dim=image.shape, inputs=[
            original_volume.id, image,
            wp.vec3(original_origin), wp.vec3(original_spacing),
            wp.vec3(origin), bbox_iso_spacing,
            wp.vec3(1, 0, 0), wp.vec3(0, 1, 0),
            -100.0, 900.0,
        ])

    gr.Success(f'切片成功 {t.elapsed:.1f} ms')

    image = image.numpy()
    image[p[0], :] = 255

    if len(p) == 3:
        image[:, p[1]] = 255

    return gr.Image(image.swapaxes(0, 1))


def launch():
    with (gr.Blocks(towl.theme, title=(title := f'{towl.__package__}')) as demo):
        with gr.Row():
            gr.Markdown(f'# {title}.{Path(__file__).stem}')
            _0_save = gr.Button('保存')

        with gr.Tab('载入'):
            gr.Markdown('''
            - 使用 [3D Slicer](https://download.slicer.org/) 等软件解析DICOM并保存3D图像为NIFTI(.nii.gz)格式
            - 载入图像(.nii.gz)，或存档(.save)
            ''')

            _0_save_select = gr.FileExplorer(file_count='single', root_dir=towl.save_dir, label='请选择文件')

            with gr.Row():
                _0_save_load = gr.Button('载入')
                _0_save_upload = gr.UploadButton('上传 NIFTI(.nii.gz)')

        with gr.Tab('识别关键区域'):
            with gr.Row():
                with gr.Column(scale=1):
                    _1_image_is = gr.Image(image_mode='L', show_label=False, show_fullscreen_button=False,
                                           show_download_button=False, interactive=False)
                with gr.Column(scale=2):
                    with gr.Row():
                        _1_bbox_a = gr.Slider(0, 0, step=1, label='前')
                        _1_bbox_p = gr.Slider(0, 0, step=1, label='后')
            with gr.Row():
                with gr.Column(scale=1):
                    _1_image_ap = gr.Image(image_mode='L', show_label=False, show_fullscreen_button=False,
                                           show_download_button=False, interactive=False)
                with gr.Column(scale=2):
                    with gr.Row():
                        _1_bbox_r = gr.Slider(0, 0, step=1, label='右')
                        _1_bbox_l = gr.Slider(0, 0, step=1, label='左')
                    with gr.Row():
                        _1_bbox_i = gr.Slider(0, 0, step=1, label='下')
                        _1_bbox_s = gr.Slider(0, 0, step=1, label='上')

        _1_bboxes = [_1_bbox_r, _1_bbox_a, _1_bbox_i, _1_bbox_l, _1_bbox_p, _1_bbox_s]

        with gr.Tab('识别关键点'):
            _2_kp_name = gr.Radio(kp_names, show_label=False)

            with gr.Row():
                with gr.Column(scale=1):
                    _2_image_ap = gr.Image(image_mode='L', show_label=False, show_fullscreen_button=False,
                                           show_download_button=False, interactive=False)

                with gr.Column(scale=1):
                    _2_image_axial = gr.Image(image_mode='L', show_label=False, show_fullscreen_button=False,
                                              show_download_button=False, interactive=False)

        _0_save.click(  # 保存
            on_0_save,
        ).then(
            lambda: gr.FileExplorer(root_dir=towl.fs_dir), None, _0_save_select,
        ).then(
            lambda: gr.FileExplorer(root_dir=towl.save_dir), None, _0_save_select,
        )

        _0_save_upload.upload(  # 上传
            fn=on_0_save_upload,
            inputs=_0_save_upload,
            outputs=None,
            trigger_mode='once',
        ).then(
            lambda: gr.FileExplorer(root_dir=towl.fs_dir), None, _0_save_select,
        ).then(
            lambda: gr.FileExplorer(root_dir=towl.save_dir), None, _0_save_select,
        )

        _0_save_load.click(  # 载入
            fn=on_0_save_load,
            inputs=_0_save_select,
            outputs=_1_bboxes,
            trigger_mode='once',
        ).success(
            on_1_image, _1_bboxes, [_1_image_is, _1_image_ap], trigger_mode='once',
        ).success(
            on_2_image_ap_update, _2_kp_name, _2_image_ap, trigger_mode='once',
        )

        for _ in _1_bboxes:  # 区域
            _.release(
                on_1_image, _1_bboxes, [_1_image_is, _1_image_ap], trigger_mode='once',
            ).success(
                on_2_image_ap_update, _2_kp_name, _2_image_ap, trigger_mode='once',
            ).success(
                on_2_image_axial_update, _2_kp_name, _2_image_axial,
            )

        _2_image_ap.select(  # 选点
            on_2_image_ap_select, [_2_image_ap, _2_kp_name], trigger_mode='once',
        ).success(
            on_2_image_ap_update, _2_kp_name, _2_image_ap, trigger_mode='once',
        ).success(
            on_2_image_axial_update, _2_kp_name, _2_image_axial,
        )

        _2_kp_name.select(  # 解剖
            on_2_image_ap_update, _2_kp_name, _2_image_ap, trigger_mode='once',
        ).success(
            on_2_image_axial_update, _2_kp_name, _2_image_axial,
        )

        _2_image_axial.select(  # 选点
            on_2_image_axial_select, [_2_image_axial, _2_kp_name], trigger_mode='once',
        ).success(
            on_2_image_axial_update, _2_kp_name, _2_image_axial,
        )

    demo.launch(share=False, show_api=False, max_file_size=gr.FileSize.GB, pwa=True)


if __name__ == '__main__':
    launch()
