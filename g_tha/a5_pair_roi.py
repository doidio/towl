import argparse
import tempfile
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import tomlkit
from minio import Minio, S3Error
from tqdm import tqdm


def main(cfg_path: str, prl: str, data: dict):
    import numpy as np
    import warp as wp
    from kernel import diff_dmc
    import itk
    from define import (
        ct_seg_femur_right, ct_seg_femur_left, ct_seg_hip_right, ct_seg_hip_left, ct_bone_best, ct_metal, ct_min
    )

    cfg_path = Path(cfg_path)
    cfg = tomlkit.loads(cfg_path.read_text('utf-8'))
    client = Minio(**cfg['minio']['client'])

    pid, rl = prl.split('_')

    fem_label = {'R': ct_seg_femur_right, 'L': ct_seg_femur_left}[rl]
    hip_label = {'R': ct_seg_hip_right, 'L': ct_seg_hip_left}[rl]

    for op, object_name in enumerate((data['pre'], data['post'])):
        op_name = ['pre', 'post'][op]

        with tempfile.TemporaryDirectory() as tdir:
            # 下载并读取分割图像
            f = Path(tdir) / 'total.nii.gz'
            try:
                client.fget_object('total', object_name, f.as_posix())
            except (S3Error, Exception):
                raise RuntimeError(f'下载 {op_name} 分割失败 total/{object_name}')

            total = itk.imread(f.as_posix(), itk.UC)
            total = itk.array_from_image(total).transpose(2, 1, 0)

            # 下载并读取原始 CT 图像
            f = Path(tdir) / 'image.nii.gz'
            try:
                client.fget_object('nii', object_name, f.as_posix())
            except (S3Error, Exception):
                raise RuntimeError(f'下载 {op_name} 原图失败 nii/{object_name}')

            image = itk.imread(f.as_posix(), itk.SS)

            # size = np.array(itk.size(image), float)
            spacing = np.array(itk.spacing(image), float)
            origin = np.array(itk.origin(image), float)

            image = itk.array_from_image(image).transpose(2, 1, 0)
            image_bg = ct_min  # 获取背景值（通常是空气的 CT 值）

            for part, label in (('femur', fem_label), ('hip', hip_label)):
                if np.sum(total_roi := (total == label)) == 0:
                    raise RuntimeError(f'{op_name} 自动分割不包含 {part} {label}')

                ijk = np.argwhere(total_roi)
                box = np.array([ijk.min(axis=0), ijk.max(axis=0) + 1])

                # 提取子区域
                roi_image = image[box[0, 0]:box[1, 0], box[0, 1]:box[1, 1], box[0, 2]:box[1, 2]].copy()
                roi_total = total[box[0, 0]:box[1, 0], box[0, 1]:box[1, 1], box[0, 2]:box[1, 2]].copy()

                roi_origin = origin + spacing * box[0]
                roi_spacing = spacing.copy()
                roi_size = box[1] - box[0]

                # 非目标区域的高亮部分（如邻近骨骼）置为背景，避免干扰配准
                roi_image[np.where((roi_total != label) & (roi_image > ct_bone_best))] = image_bg

                # 如果是术后数据，提取金属假体网格
                if op == 1:
                    # 使用 GPU 加速的 Marching Cubes 提取金属等值面
                    metal_mesh = diff_dmc(wp.from_numpy(roi_image, wp.float32), np.zeros(3), spacing, ct_metal)
                    if metal_mesh.is_empty and part in ('femur',):
                        raise RuntimeError(f'{op_name} {part} 不包含金属')

                # 提取骨骼网格
                bone_mesh = diff_dmc(wp.from_numpy(roi_image, wp.float32), np.zeros(3), spacing, ct_bone_best)

                if bone_mesh.is_empty:
                    raise RuntimeError(f'{op_name} {part} 不包含骨骼')

                # 存储
                roi_image = itk.image_from_array(np.ascontiguousarray(roi_image.transpose(2, 1, 0)))
                roi_image.SetOrigin(roi_origin)
                roi_image.SetSpacing(roi_spacing)

                f = Path(tdir) / 'roi.nii.gz'
                itk.imwrite(roi_image, f.as_posix())
                client.fput_object('pair', '/'.join([pid, rl, op_name, part, f.name]), f.as_posix())

                if op == 1 and not metal_mesh.is_empty:
                    f = Path(tdir) / 'metal.stl'
                    metal_mesh.export(f.as_posix())
                    client.fput_object('pair', '/'.join([pid, rl, op_name, part, f.name]), f.as_posix())

                f = Path(tdir) / 'bone.stl'
                bone_mesh.export(f.as_posix())
                client.fput_object('pair', '/'.join([pid, rl, op_name, part, f.name]), f.as_posix())

                roi = {
                    'origin': roi_origin.tolist(),
                    'spacing': roi_spacing.tolist(),
                    'size': roi_size.tolist(),
                }
                roi = tomlkit.dumps(roi).encode('utf-8')
                client.put_object('pair', '/'.join([pid, rl, op_name, part, 'roi.toml']), BytesIO(roi), len(roi))


def launch(cfg_path: str, max_workers: int):
    cfg_path = Path(cfg_path)
    cfg = tomlkit.loads(cfg_path.read_text('utf-8'))
    client = Minio(**cfg['minio']['client'])

    pairs = {}
    for _ in client.list_objects('pair', recursive=True):
        if not _.object_name.endswith('.nii.gz'):
            continue

        if len(paths := _.object_name.split('/')) != 4:
            continue

        pid, rl, op, nii = paths

        if op not in ('pre', 'post'):
            continue

        prl = f'{pid}_{rl}'  # 患者 ID + 左右侧作为唯一标识
        if prl not in pairs:
            pairs[prl] = {'prl': prl}
        pairs[prl][op] = f'{pid}/{nii}'

    valid_pairs = {k: v for k, v in pairs.items() if 'pre' in v and 'post' in v}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(main, cfg_path.as_posix(), prl, pairs[prl]): prl for prl in valid_pairs}

        try:
            for fu in tqdm(as_completed(futures), total=len(futures)):
                try:
                    fu.result()
                except Exception as _:
                    warnings.warn(f'{_} {futures[fu]}', stacklevel=2)

        except KeyboardInterrupt:
            print('Keyboard interrupted terminating...')
            executor.shutdown(wait=False)
            for future in futures:
                future.cancel()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--max_workers', type=int, default=4)
    args = parser.parse_args()

    launch(args.config, args.max_workers)
