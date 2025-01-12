import os
import shutil
from pathlib import Path
from typing import Optional

import gradio as gr
import itk
import numpy as np

import towl


def on_nifti_upload(filename):
    filename = Path(filename)
    try:
        itk.imread(filename.as_posix())
    except Exception as e:
        gr.Warning(f'读取失败 {e}')

    os.makedirs(towl.nifti_dir, exist_ok=True)
    shutil.copy(filename, towl.nifti_dir / filename.name)
    gr.Success('上传成功')


original_image: Optional[np.ndarray] = None
original_size: Optional[np.ndarray] = None
original_spacing: Optional[np.ndarray] = None
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

    global original_image, original_size, original_spacing, original_origin
    original_image = itk.array_from_image(image).swapaxes(0, 2).copy()
    original_size = np.array(itk.size(image))
    original_spacing = np.array(itk.spacing(image))
    original_origin = np.array(itk.origin(image))

    gr.Success('载入成功')

    return [
        *[gr.Slider(
            maximum=float(original_size[_]),
            value=0,
            interactive=True,
        ) for _ in (0, 2, 1)],
        *[gr.Slider(
            maximum=float(original_size[_]),
            value=float(original_size[_]),
            interactive=True,
        ) for _ in (0, 2, 1)],
    ]


def on_ap_image(bbox_start_0, bbox_start_2, bbox_start_1, bbox_span_0, bbox_span_2, bbox_span_1):
    if original_image is None:
        raise gr.Error('未载入原始图像')

    threshold = [0, 500]
    a = original_image.copy()
    c = (threshold[0] <= a)  # * (a <= threshold[1])
    a = (a * c).sum(axis=1)
    c = np.sum(c, axis=1)
    c[np.where(c <= 0)] = 1
    a = a / c

    window = [-100, 900]
    a = (a - window[0]) / (window[1] - window[0]) * 255
    a = np.flipud(a.astype(np.uint8).swapaxes(0, 1))

    hw = original_spacing[1] * original_size[1] / original_spacing[0] / original_size[0]
    return gr.Image(np.ascontiguousarray(a), height=(_ := 100), width=_ / hw)


def launch():
    with (gr.Blocks(towl.theme, title=(title := f'{towl.__package__}')) as demo):
        gr.Markdown(f'# {title}.{Path(__file__).stem}')

        with gr.Tab('0 原始图像'):
            gr.Markdown(
                '- 请使用 [3D Slicer](https://download.slicer.org/) 等软件解析DICOM并保存3D图像为NIFTI(.nii.gz)格式')

            _0_original_select = gr.FileExplorer(file_count='single', root_dir=towl.nifti_dir, label='请选择文件')

            with gr.Row():
                _0_original_upload = gr.UploadButton('上传 NIFTI(.nii.gz)')
                _0_original_load = gr.Button('载入')

        with gr.Tab('1 正位透视'):
            _1_image = gr.Image(image_mode='L', label='正位透视', interactive=False)

            with gr.Row():
                with gr.Column():
                    _1_bbox_start = [gr.Slider(0, label=['右', '下', '前'][_]) for _ in range(3)]
                with gr.Column():
                    _1_bbox_span = [gr.Slider(1, label=['宽', '高', '厚'][_]) for _ in range(3)]

        _ = _0_original_upload.upload(on_nifti_upload, _0_original_upload, None)
        _ = _.then(lambda: gr.FileExplorer(root_dir=towl.fs_dir), None, _0_original_select)
        _ = _.then(lambda: gr.FileExplorer(root_dir=towl.nifti_dir), None, _0_original_select)

        _ = _0_original_load.click(on_original_load, _0_original_select, [*_1_bbox_start, *_1_bbox_span])
        _ = _.then(on_ap_image, [*_1_bbox_start, *_1_bbox_span], _1_image)

    demo.launch(share=False, show_api=False, max_file_size=gr.FileSize.GB, pwa=True)


if __name__ == '__main__':
    launch()
