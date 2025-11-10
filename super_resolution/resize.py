import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image
from chainner_ext import resize, ResizeFilter
from tqdm import tqdm


def main(dataset_dir: str, total_body_file: str):
    resized_dir = Path(dataset_dir) / 'test' / 'TotalBody_Resized'
    total_body_file = Path(total_body_file).absolute()

    image_0 = np.array(Image.open(total_body_file))

    scaling = 8
    wh = (image_0.shape[1] * scaling, image_0.shape[0] * scaling)
    image_0 = image_0.astype(np.float32) / 255.0
    image_0 = resize(image_0, wh, ResizeFilter.Lanczos, False)
    image_0 = (np.clip(image_0, 0.0, 1.0) * 255.0).astype(np.uint8).squeeze()

    _ = resized_dir / total_body_file.name
    _.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_0).convert('L').save(_)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--max_workers', type=int, default=16)
    args = parser.parse_args()

    total_body_dir = Path(args.dataset_dir) / 'test' / 'TotalBody'

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(
            main, args.dataset_dir, _.as_posix(),
        ) for _ in total_body_dir.rglob('*.png')}

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
