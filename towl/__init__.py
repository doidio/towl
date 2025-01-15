import locale
import os
from pathlib import Path

import gradio as gr
import numpy as np
from loguru import logger

locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')

fs_dir = Path('./fs').absolute()
tmp_dir = fs_dir / 'tmp'
save_dir = fs_dir / 'save'

os.environ['GRADIO_TEMP_DIR'] = tmp_dir.as_posix()

theme = np.random.choice([
    gr.themes.Citrus(),
    gr.themes.Default(),
    gr.themes.Glass(),
    gr.themes.Monochrome(),
    gr.themes.Ocean(),
    gr.themes.Origin(),
    gr.themes.Soft(),
])
logger.debug(f'[theme] {theme.__class__.__name__}')
