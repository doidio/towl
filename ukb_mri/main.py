import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import itk
import numpy as np
import tomlkit
from tqdm import tqdm

BodyParts = [18, 19] + list(range(36, 50))


def main(nifti_file: str, done: bool):
    if done: return None

    table = {}

    # 体积
    label = Path(nifti_file).parent.parent.parent / 'labels' / 'total_mr' / Path(nifti_file).name
    label = itk.imread(label)
    spacing = np.array(list(reversed([*itk.spacing(label)])), float)
    label = itk.array_from_image(label)

    v = spacing[0] * spacing[1] * spacing[2]
    volume = {str(_): np.sum(label == _) * v for _ in BodyParts}
    table['volume'] = volume

    # 脂肪比例
    fat = Path(nifti_file).parent.parent / 'fat' / Path(nifti_file).name.replace('_p.', '_F.')
    fat = itk.imread(fat)
    fat = itk.array_from_image(fat).astype(float)

    water = Path(nifti_file).parent.parent / 'water' / Path(nifti_file).name.replace('_p.', '_W.')
    water = itk.imread(water)
    water = itk.array_from_image(water).astype(float)

    fin = (fat + water)
    _ = np.where(fin != 0)
    fin[_] = fat[_] / fin[_]

    table['fat'] = {
        'mean': {str(_): np.mean(fin[np.where(label == _)]) for _ in BodyParts},
        'std': {str(_): np.std(fin[np.where(label == _)]) for _ in BodyParts},
    }

    return table


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_file', required=True)
    parser.add_argument('--max_workers', type=int, default=5)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    images_dir = input_dir / 'images' / 'in'

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
                        output_file.write_text(tomlkit.dumps(output), 'utf-8')
                except Exception as _:
                    warnings.warn(f'{_} {futures[fu]}', stacklevel=2)

        except KeyboardInterrupt:
            print('Keyboard interrupted terminating...')
            executor.shutdown(wait=False)
            for future in futures:
                future.cancel()
