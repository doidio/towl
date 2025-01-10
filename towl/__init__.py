import os
from pathlib import Path

import gradio as gr
import numpy as np
from loguru import logger

fs_dir = Path('fs')
tmp_dir = fs_dir / 'tmp'
nifti_dir = f'{fs_dir}/nifti'

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
