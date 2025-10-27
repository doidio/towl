# pip install numpy opencv-python chainner-ext pillow pydicom tqdm

import argparse
import warnings
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from random import choice, random

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
    test_dir = dataset_dir / 'test'

    with zipfile.ZipFile(zip_file) as zf:

        def find(protocols):
            for file in zf.namelist():
                with zf.open(file) as f:
                    try:
                        assert (it := pydicom.dcmread(f)).pixel_array is not None
                    except (InvalidDicomError, ValueError):
                        continue

                    if it.BitsAllocated != 8:  # 仅支持 8-bit，若支持 16-bit 需相应变更超分训练配置
                        warnings.warn(f'{zip_file} {file} 8-bit ≠ {it.BitsAllocated}')
                        continue

                    if it.PhotometricInterpretation != 'MONOCHROME2':  # 仅支持白色前景黑色背景
                        warnings.warn(f'{zip_file} {file} MONOCHROME2 ≠ {it.PhotometricInterpretation}')
                        continue

                    if it.ProtocolName in protocols:
                        yield it, it.ProtocolName

        # test
        max_number = None
        for ds, protocol in find(('Total Body',)):
            if max_number is None:
                max_number = int(ds.InstanceNumber), ds
            elif max_number[0] < int(ds.InstanceNumber):
                max_number = int(ds.InstanceNumber), ds

        ds = max_number[1]  # 经验规律：不透明全身图InstanceNumber + 1 = 透明全身图InstanceNumber

        base = 16
        h, w = ds.pixel_array.shape[:2]

        pad_h = (base - h % base) % base
        pad_w = (base - w % base) % base

        image = cv2.copyMakeBorder(ds.pixel_array, 0, pad_h, 0, pad_w,
                                   cv2.BORDER_CONSTANT, value=[float(np.min(ds.pixel_array))])

        protocol = protocol.replace(' ', '')
        _ = test_dir / protocol / f'{Path(zip_file).stem}_{protocol}_{ds.InstanceNumber}.png'
        _.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image, mode='L').convert('RGB').save(_)

        # train & val
        for ds, protocol in find((
                'AP Spine', 'LVA',
                'Left Femur', 'Right Femur',
                'Left Ortho Knee', 'Right Ortho Knee',
        )):
            wl_enabled = random() < 0.5
            wl = random() * 0.3 + 0.4, random() * 0.2 + 0.4
            rf = choice([ResizeFilter.Box, ResizeFilter.Linear, ResizeFilter.Lagrange, ResizeFilter.Gauss])

            # GT图8倍降采样之后的LQ图，是16的整数倍，避免tile推理
            base = 8 * 16  # TODO: 8 * 16
            h, w = ds.pixel_array.shape[:2]

            pad_h = (base - h % base) % base
            pad_w = (base - w % base) % base

            top, bottom = pad_h // 2, pad_h - pad_h // 2
            left, right = pad_w // 2, pad_w - pad_w // 2

            gt_image = cv2.copyMakeBorder(ds.pixel_array, top, bottom, left, right,
                                          cv2.BORDER_CONSTANT, value=[float(np.min(ds.pixel_array))])

            # 退化
            lq_image = gt_image.astype(np.float32) / 255.0
            h, w = lq_image.shape[:2]

            ksize = 7
            for scaling in (2, 4, 8,):
                # 模糊
                lq_image = cv2.GaussianBlur(lq_image, (ksize, ksize), 0)

                # 降采样
                lq_image = resize(lq_image, (w // scaling, h // scaling), rf, gamma_correction=True)

            # 对比度退化，全身窗与局部窗不同
            if wl_enabled:
                lq_image = ((lq_image - wl[1]) / wl[0] + 1) / 2

            # 噪声
            noise = np.random.normal(loc=0.0, scale=2.0 / 255.0, size=lq_image.shape).astype(np.float32)
            lq_image = lq_image + noise

            lq_image = (np.clip(lq_image, 0.0, 1.0) * 255.0).astype(np.uint8).squeeze()

            protocol = protocol.replace(' ', '')

            # 保存LQ
            _ = lq_dir / '8x' / f'{Path(zip_file).stem}_{protocol}.png'
            _.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(lq_image, mode='L').convert('RGB').save(_)

            # 保存GT
            _ = gt_dir / f'{Path(zip_file).stem}_{protocol}.png'
            _.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(gt_image, mode='L').convert('RGB').save(_)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip_dir', type=str, required=True)
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--val_count', type=int, default=100)
    parser.add_argument('--max_workers', type=int, default=16)
    args = parser.parse_args()

    zip_dir = Path(args.zip_dir).absolute()
    zip_files = [_ for _ in zip_dir.rglob('*.zip')]

    bools = np.zeros(len(zip_files), bool)
    bools[:args.val_count] = True

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(
            main, it[0].as_posix(), args.dataset_dir, it[1],
        ) for it in zip(zip_files, bools)}

        try:
            for _ in tqdm(as_completed(futures), total=len(futures)):
                try:
                    _.result()
                except Exception as e:
                    warnings.warn(str(e))

        except KeyboardInterrupt:
            print('Keyboard interrupted terminating...')
            executor.shutdown(wait=False)
            for future in futures:
                future.cancel()
