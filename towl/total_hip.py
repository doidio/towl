from pathlib import Path

import gradio as gr
import itk

import towl


def on_upload(upload):
    image = itk.imread(upload)

def on_refresh_nifti():
    return {
        'headers': ['Patient Name', 'Shape'],
        'data': [
            ['series 1', '123'],
            ['series 2', '456'],
        ],
    }


def launch():
    with gr.Blocks(towl.theme, title=(title := f'{towl.__package__}')) as demo:
        gr.Markdown(f'# {title}.{Path(__file__).stem}')

        ui_upload = gr.UploadButton('Upload a NIFTI(.nii.gz)', file_types=['.nii.gz'])

        ui_refresh_nifti = gr.Button('Refresh NIFTI')

        ui_refresh_nifti.click(on_refresh_nifti, None, )

        ui_upload.upload(on_upload, ui_upload).then(on_refresh_nifti)

    demo.launch(share=False, show_api=False, max_file_size=gr.FileSize.GB, pwa=True)


if __name__ == '__main__':
    launch()
