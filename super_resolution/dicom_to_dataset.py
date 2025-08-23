# pip install numpy chainner-ext pillow pydicom tqdm

# https://github.com/neosr-project/neosr/wiki/Installation-Instructions#manual-installation
# cd neosr
# uv run --isolated train.py -opt ../super_resolution/train.toml

import argparse
import warnings
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from random import choice

import cv2
import numpy as np
import pydicom
from PIL import Image
from chainner_ext import resize, ResizeFilter
from pydicom.errors import InvalidDicomError
from tqdm import tqdm

DXAProtocolNames = [
    'Total Body',
    'AP Spine', 'LVA',
    'Left Femur', 'Right Femur',
    'Left Ortho Knee', 'Right Ortho Knee',
]


def main(zip_file: str, dataset_dir: str, is_val: bool):
    dataset_dir = Path(dataset_dir).absolute()
    subfolder = 'val' if is_val else 'train'
    gt_dir = dataset_dir / subfolder / 'gt'
    lq_dir = dataset_dir / subfolder / 'lq'

    with zipfile.ZipFile(zip_file) as zf:
        for file in zf.namelist():
            with zf.open(file) as f:
                try:
                    ds = pydicom.dcmread(f)
                except InvalidDicomError:
                    continue

                if ds.BitsAllocated != 8:  # 仅支持 8-bit，若支持 16-bit 需相应变更超分训练配置
                    warnings.warn(f'{zip_file} {file} 8-bit ≠ {ds.BitsAllocated}')
                    continue

                if ds.PhotometricInterpretation != 'MONOCHROME2':  # 仅支持白色前景黑色背景
                    warnings.warn(f'{zip_file} {file} MONOCHROME2 ≠ {ds.PhotometricInterpretation}')
                    continue

                # 转换髋膝局部图
                if (ProtocolName := ds.ProtocolName.strip().replace(' ', '_')) in [
                    'Left_Femur', 'Right_Femur',
                    'Left_Ortho Knee', 'Right_Ortho_Knee',
                ]:
                    # GT图16倍降采样之后的LQ图，是16的整数倍，避免tile推理
                    base = 16 * 16
                    h, w = ds.pixel_array.shape[:2]

                    pad_h = (base - h % base) % base
                    pad_w = (base - w % base) % base

                    top, bottom = pad_h // 2, pad_h - pad_h // 2
                    left, right = pad_w // 2, pad_w - pad_w // 2

                    gt_image = cv2.copyMakeBorder(ds.pixel_array, top, bottom, left, right, cv2.BORDER_REFLECT101)

                    # 退化
                    lq_image = gt_image.astype(np.float32) / 255.0
                    h, w = lq_image.shape[:2]

                    # 模糊
                    ksize = 7
                    lq_image = cv2.GaussianBlur(lq_image, (ksize, ksize), 0)

                    # 降采样
                    rf = choice([ResizeFilter.Box, ResizeFilter.Linear, ResizeFilter.Lagrange, ResizeFilter.Gauss])
                    lq_image = resize(lq_image, (w // 16, h // 16), rf, gamma_correction=True)
                    lq_image = (np.clip(lq_image, 0.0, 1.0) * 255.0).astype(np.uint8).squeeze()

                    # 保存LQ图
                    _ = lq_dir / f'{Path(zip_file).stem}_{ProtocolName}.png'
                    _.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(lq_image, mode='L').convert('RGB').save(_)

                    # 保存GT图
                    _ = gt_dir / f'{Path(zip_file).stem}_{ProtocolName}.png'
                    _.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(gt_image, mode='L').convert('RGB').save(_)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip_dir', required=True)
    parser.add_argument('--dataset_dir', required=True)
    parser.add_argument('--max_workers', default=16)
    args = parser.parse_args()

    zip_dir = Path(args.zip_dir).absolute()
    zip_files = [_ for _ in zip_dir.rglob('*.zip')]

    bools = np.zeros(len(zip_files), bool)
    bools[::1000] = True

    items = list(zip(zip_files, bools))

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(main, it[0].as_posix(), args.dataset_dir, it[1]) for it in items}

        try:
            for _ in tqdm(as_completed(futures), total=len(futures), unit='it', unit_scale=True, leave=True):
                try:
                    _.result()
                except Exception as e:
                    warnings.warn(str(e))

        except KeyboardInterrupt:
            print('Keyboard interrupted terminating...')
            executor.shutdown(wait=False)
            for future in futures:
                future.cancel()
