import locale
import os
from pathlib import Path

import gradio as gr
from loguru import logger

import towl.kernel

locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')

fs_dir = Path('./fs').absolute()
tmp_dir = fs_dir / 'tmp'
save_dir = fs_dir / 'save'

os.environ['GRADIO_TEMP_DIR'] = tmp_dir.as_posix()

theme = [
    gr.themes.Citrus(),
    gr.themes.Default(),
    gr.themes.Glass(),
    gr.themes.Monochrome(),
    gr.themes.Ocean(),
    gr.themes.Origin(),
    gr.themes.Soft(),
][3]
logger.debug(f'[theme] {theme.__class__.__name__}')
