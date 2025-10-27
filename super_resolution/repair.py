import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image
from chainner_ext import resize, ResizeFilter
from tqdm import tqdm


def main(dataset_dir: str, prefix: str, knee_tuples: tuple):
    dataset_dir = Path(dataset_dir).absolute()
    total_body_file = dataset_dir / 'test' / 'TotalBody' / f'{prefix}.png'

    image_0 = np.array(Image.open(total_body_file))

    scaling = 8
    wh = (image_0.shape[1] * scaling, image_0.shape[0] * scaling)
    image_0 = image_0.astype(np.float32) / 255.0
    image_0 = resize(image_0, wh, ResizeFilter.Lanczos, False)
    image_0 = (np.clip(image_0, 0.0, 1.0) * 255.0).astype(np.uint8).squeeze()

    # TODO: repair hip

    # repair knee
    roi, h, crop_h, file = knee_tuples[0], int(knee_tuples[1]), int(knee_tuples[2]), knee_tuples[3]
    image_1 = np.array(Image.open(file))
    image_0[(div_h := h * scaling):(h + crop_h) * scaling] = image_1

    # save
    _ = dataset_dir / 'test' / 'TotalBody_Repaired' / f'{prefix}.png'
    _.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_0).convert('L').save(_)

    _ = dataset_dir / 'test' / 'TotalBody_Repaired_Hip' / f'{prefix}_{0}_{div_h}.png'
    _.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_0[:div_h]).convert('L').save(_)

    _ = dataset_dir / 'test' / 'TotalBody_Repaired_Knee' / f'{prefix}_{div_h}_{image_0.shape[0]}.png'
    _.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_0[div_h:]).convert('L').save(_)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--max_workers', type=int, default=16)
    args = parser.parse_args()

    result_dir = Path(args.dataset_dir) / 'test' / '8x_Result'

    # Femur_results = {_: _.stem.split('_') for _ in result_dir.rglob('*_Femur_*.png')}
    # Femur_results = {'_'.join(_[:-8]): (_[-8], _[-7], _[-6], __) for __, _ in Femur_results.items()}

    OrthoKnee_results = {_: _.stem.split('_') for _ in result_dir.rglob('*_OrthoKnee_*.png')}
    OrthoKnee_results = {'_'.join(_[:-8]): (_[-8], _[-7], _[-6], __) for __, _ in OrthoKnee_results.items()}

    # results = {_: [Femur_results[_], OrthoKnee_results[_]] for _ in set(Femur_results) & set(OrthoKnee_results)}
    results = {_: (OrthoKnee_results[_],) for _ in set(OrthoKnee_results)}

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(
            main, args.dataset_dir, _, __[0],
        ) for _, __ in results.items()}

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
