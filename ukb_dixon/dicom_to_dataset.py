# pip install numpy pydicom itk tqdm

import argparse
import tempfile
import warnings
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import itk
import numpy as np
import pydicom
import warp as wp
from pydicom.errors import InvalidDicomError
from tqdm import tqdm

from kernel import region_sample


def main(zip_file: str, dataset_dir: str, is_val: bool):
    dataset_dir = Path(dataset_dir).absolute()
    train_val = 'val' if is_val else 'train'

    with zipfile.ZipFile(zip_file) as zf:

        for suffix in ('W', 'F'):
            images_dir = dataset_dir / suffix / train_val / 'images'

            with tempfile.TemporaryDirectory() as tdir:

                subfolders = set()
                for file in zf.namelist():
                    with zf.open(file) as f:
                        try:
                            assert (it := pydicom.dcmread(f)).pixel_array is not None
                        except (InvalidDicomError, ValueError):
                            continue

                        _ = str(it.SeriesDescription)
                        if _.startswith('Dixon') and _.endswith(suffix):
                            _ = Path(tdir) / str(it.SeriesInstanceUID)
                            _.mkdir(parents=True, exist_ok=True)
                            zf.extract(file, _)
                            subfolders.add(_)

                images = []
                for _ in subfolders:
                    images.append(itk.imread(_))

                for image in sorted(images, key=lambda _: itk.origin(_)[2]):
                    _ = images_dir / f'{Path(zip_file).stem}_{suffix}_{itk.origin(image)[2]}.nii.gz'
                    _.parent.mkdir(parents=True, exist_ok=True)
                    itk.imwrite(image, _.as_posix())

                volumes, spacings, box = [], [], None
                for image in images:
                    size = np.array([*itk.size(image)])
                    spacing = np.array([*itk.spacing(image)])
                    origin = np.array([*itk.origin(image)])
                    end = origin + size * spacing

                    spacings.append(float(spacing[0]))
                    spacings.append(float(spacing[1]))

                    if box is None:
                        box = [origin, end]
                    else:
                        box = [np.min([origin, box[0]], 0), np.max([end, box[1]], 0)]

                    _ = 2  # 首尾截断层数
                    image = itk.array_from_image(image).transpose(2, 1, 0).copy()
                    volume = wp.Volume.load_from_numpy(image, bg_value=0.0)
                    volumes.append((volume, spacing, origin))

                re_spacing = float(np.mean(spacings))
                re_spacing = np.array([re_spacing, re_spacing, (re_spacing + 0.25) // 0.5 * 0.5])
                size = np.ceil((box[1] - box[0]) / re_spacing)
                region = wp.full(shape=(*size,), value=0, dtype=wp.float32)

                for volume, spacing, origin in volumes:
                    wp.launch(region_sample, region.shape, [
                        wp.uint64(volume.id), wp.vec3(spacing), wp.vec3(origin),
                        region, wp.vec3(re_spacing), wp.vec3(box[0]),
                    ])

                region = region.numpy().astype(np.uint16)
                image = itk.image_from_array(region.transpose(2, 1, 0).copy())
                image.SetOrigin(box[0])
                image.SetSpacing(re_spacing)

                _ = images_dir / f'{Path(zip_file).stem}_{suffix}.nii.gz'
                _.parent.mkdir(parents=True, exist_ok=True)
                itk.imwrite(image, _.as_posix())


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
                    raise e

        except KeyboardInterrupt:
            print('Keyboard interrupted terminating...')
            executor.shutdown(wait=False)
            for future in futures:
                future.cancel()
