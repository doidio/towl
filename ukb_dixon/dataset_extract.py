import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import itk
import numpy as np
import tomlkit
from tqdm import tqdm

WaterParts = list(range(40, 50))
TotalParts = [18, 19] + list(range(36, 50))


def main(nifti_file: str, done: bool):
    if done: return None

    table = {}

    # 体积
    label = Path(nifti_file).parent.parent.parent / 'labels' / 'total_mr' / Path(nifti_file).name
    if not label.exists():
        return None
    label = itk.imread(label)
    spacing = np.array(list(reversed([*itk.spacing(label)])), float)
    label = itk.array_from_image(label)

    v = spacing[0] * spacing[1] * spacing[2]
    volume = {str(_): np.sum(label == _) * v for _ in TotalParts}
    table['volume'] = volume

    # 读取水脂原图
    fat = Path(nifti_file).parent.parent / 'F' / Path(nifti_file).name
    fat = itk.imread(fat)
    fat = itk.array_from_image(fat).astype(float)
    fat_dtype = fat.dtype

    water = Path(nifti_file).parent.parent / 'W' / Path(nifti_file).name
    water = itk.imread(water)
    water = itk.array_from_image(water).astype(float)
    water_dtype = water.dtype

    # 根据水相肌肉更亮，逐层修正水脂交换错误
    for z in range(water.shape[0]):
        mask = np.isin(label[z], WaterParts)
        if np.sum(water[z][mask]) < np.sum(fat[z][mask]):
            water[z], fat[z] = fat[z].copy(), water[z].copy()

    _ = itk.image_from_array(fat.astype(fat_dtype))
    f = Path(nifti_file).parent.parent / 'F_fix' / Path(nifti_file).name
    f.parent.mkdir(parents=True, exist_ok=True)
    itk.imwrite(_, f.as_posix())

    _ = itk.image_from_array(water.astype(water_dtype))
    f = Path(nifti_file).parent.parent / 'W_fix' / Path(nifti_file).name
    f.parent.mkdir(parents=True, exist_ok=True)
    itk.imwrite(_, f.as_posix())

    # 统计脂肪比例
    fin = (fat + water)
    _ = np.where(fin != 0)
    fin[_] = fat[_] / fin[_]

    table['fat'] = {
        'mean': {str(_): np.mean(fin[np.where(label == _)]) for _ in TotalParts},
        'std': {str(_): np.std(fin[np.where(label == _)]) for _ in TotalParts},
    }

    return table


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output_file', required=True)
    parser.add_argument('--max_workers', type=int, default=5)
    args = parser.parse_args()

    dataset = Path(args.dataset)
    images_dir = dataset / 'images' / 'opp'

    output_file = Path(args.output_file)
    if output_file.exists():
        output = tomlkit.loads(output_file.read_text('utf-8'))
    else:
        output = {}

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(main, _.as_posix(), _.name in output): _.name
                   for _ in images_dir.glob('*.nii.gz')}

        try:
            for fu in tqdm(as_completed(futures), total=len(futures)):
                try:
                    if (result := fu.result()) is not None:
                        output[futures[fu]] = result
                        output_file.parent.mkdir(parents=True, exist_ok=True)
                        output_file.write_text(tomlkit.dumps(output), 'utf-8')
                except Exception as _:
                    warnings.warn(f'{_} {futures[fu]}', stacklevel=2)

        except KeyboardInterrupt:
            print('Keyboard interrupted terminating...')
            executor.shutdown(wait=False)
            for future in futures:
                future.cancel()
