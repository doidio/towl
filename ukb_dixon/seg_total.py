import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tqdm import tqdm


def main(dataset: str):
    dataset = Path(dataset)
    images = set((dataset / 'images' / 'opp').glob('*.nii.gz'))

    for image in tqdm(images, total=len(images)):
        for task in ('total_mr', 'tissue_types_mr'):
            label = dataset / 'labels' / task / image.name

            if label.exists(): continue

            with ProcessPoolExecutor(max_workers=1) as executor:
                future = executor.submit(seg, image.as_posix(), label.as_posix(), task)
                try:
                    future.result()
                except Exception as e:
                    warnings.warn(str(e), stacklevel=2)

            if label.exists():
                print(image, task, 'done')
            else:
                print(image, task, 'error')


def seg(image_path: str, label_path: str, task: str):
    from totalsegmentator.python_api import totalsegmentator
    totalsegmentator(image_path, label_path, True, task=task, quiet=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    args = parser.parse_args()

    main(args.dataset)
