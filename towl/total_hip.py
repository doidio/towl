import os
import shutil
from pathlib import Path
from typing import Optional

import gradio as gr
import itk
import numpy as np
import warp as wp

import towl
from towl.kernel import volume_parallel_xray


def on_nifti_upload(filename):
    filename = Path(filename)
    try:
        itk.imread(filename.as_posix())
    except Exception as e:
        gr.Warning(f'读取失败 {e}')

    os.makedirs(towl.nifti_dir, exist_ok=True)
    shutil.copy(filename, towl.nifti_dir / filename.name)
    gr.Success('上传成功')


original_bg: Optional[float] = None
original_volume: Optional[wp.Volume] = None
original_size: Optional[np.ndarray] = None
original_spacing: Optional[np.ndarray] = None
original_length: Optional[np.ndarray] = None
original_origin: Optional[np.ndarray] = None


def on_original_load(filename):
    if filename is None:
        raise gr.Error('未选中')

    f = Path(filename)
    if not f.is_file() or not f.exists():
        raise gr.Error('选中无效')

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


image_iso_spacing = 0.5


def on_1_image(bbox_r, bbox_a, bbox_i, bbox_l, bbox_p, bbox_s):
    if original_volume is None:
        raise gr.Error('未载入原始图像')

    bbox_min = np.min([[bbox_r, bbox_a, bbox_i], [bbox_l, bbox_p, bbox_s]], axis=0)
    bbox_max = np.max([[bbox_r, bbox_a, bbox_i], [bbox_l, bbox_p, bbox_s]], axis=0)
    bbox_size = np.max([bbox_max - bbox_min, [1, 1, 1]], axis=0)

    image_origin = original_origin + bbox_min
    image_size = (bbox_size / image_iso_spacing).astype(int)

    image_is = wp.full(shape=(image_size[0], image_size[1]), value=0, dtype=wp.uint8)
    image_ap = wp.full(shape=(image_size[0], image_size[2]), value=0, dtype=wp.uint8)

    with wp.ScopedTimer('on_1_image', print=False) as t:
        wp.launch(kernel=volume_parallel_xray, dim=image_is.shape, inputs=[
            original_volume.id, image_is,
            wp.vec3(original_origin), wp.vec3(original_spacing),
            wp.vec3(image_origin), image_iso_spacing,
            wp.vec3(1, 0, 0), wp.vec3(0, 1, 0), wp.vec3(0, 0, 1),
            original_length[2], 0.0, -100.0, 900.0,
        ])

        wp.launch(kernel=volume_parallel_xray, dim=image_ap.shape, inputs=[
            original_volume.id, image_ap,
            wp.vec3(original_origin), wp.vec3(original_spacing),
            wp.vec3(image_origin), image_iso_spacing,
            wp.vec3(1, 0, 0), wp.vec3(0, 0, 1), wp.vec3(0, 1, 0),
            original_length[1], 0.0, -100.0, 900.0,
        ])

    gr.Success(f'裁剪成功 {t.elapsed:.1f} ms')

    image_is = image_is.numpy().swapaxes(0, 1)
    image_ap = np.flipud(image_ap.numpy().swapaxes(0, 1))

    return [
        gr.Image(image_is),
        gr.Image(image_ap),
    ]


def launch():
    with (gr.Blocks(towl.theme, title=(title := f'{towl.__package__}')) as demo):
        gr.Markdown(f'# {title}.{Path(__file__).stem}')

        with gr.Tab('载入'):
            gr.Markdown('''
            - 使用 [3D Slicer](https://download.slicer.org/) 等软件解析DICOM并保存3D图像为NIFTI(.nii.gz)格式
            ''')

            _0_original_select = gr.FileExplorer(file_count='single', root_dir=towl.nifti_dir, label='请选择文件')

            with gr.Row():
                _0_original_upload = gr.UploadButton('上传 NIFTI(.nii.gz)')
                _0_original_load = gr.Button('载入')

        with gr.Tab('裁剪'):
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

        _0_original_upload.upload(  # 上传
            fn=on_nifti_upload,
            inputs=_0_original_upload,
            outputs=None,
            trigger_mode='once',
        ).then(
            lambda: gr.FileExplorer(root_dir=towl.fs_dir), None, _0_original_select,
        ).then(
            lambda: gr.FileExplorer(root_dir=towl.nifti_dir), None, _0_original_select,
        )

        _0_original_load.click(  # 载入
            fn=on_original_load,
            inputs=_0_original_select,
            outputs=_1_bboxes,
            trigger_mode='once',
        ).success(on_1_image, _1_bboxes, [_1_image_is, _1_image_ap], trigger_mode='once')

        for _ in _1_bboxes:
            _.release(on_1_image, _1_bboxes, [_1_image_is, _1_image_ap], trigger_mode='once')

    demo.launch(share=False, show_api=False, max_file_size=gr.FileSize.GB, pwa=True)


if __name__ == '__main__':
    launch()
