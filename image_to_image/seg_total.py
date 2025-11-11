import argparse
import tempfile
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import tomlkit
from minio import Minio, S3Error
from totalsegmentator.python_api import totalsegmentator
from tqdm import tqdm


def main(cfg_path: str, object_name: str):
    cfg_path = Path(cfg_path)
    cfg = tomlkit.loads(cfg_path.read_text('utf-8'))
    client = Minio(**cfg['minio']['client'])

    with tempfile.TemporaryDirectory() as tdir:
        image = Path(tdir) / 'image.nii.gz'
        label = Path(tdir) / 'label.nii.gz'

        client.fget_object('nii', object_name, image.as_posix())
        totalsegmentator(image, label, True, task='total', quiet=True)
        client.fput_object('total', object_name, label.as_posix())


def get_objects(cfg_path: str):
    cfg_path = Path(cfg_path)
    cfg = tomlkit.loads(cfg_path.read_text('utf-8'))
    client = Minio(**cfg['minio']['client'])

    if 'hip_implant' not in cfg or 'PatientID' not in cfg['hip_implant']:
        patients = []
    else:
        patients = cfg['hip_implant']['PatientID']
    patients = set(patients)

    for obj in client.list_objects('nii', recursive=True):
        if obj.is_dir:
            continue

        if obj.object_name.split('/')[0] not in patients:
            continue

        try:
            client.stat_object('total', obj.object_name)
        except S3Error:
            yield obj


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--max_workers', type=int, default=4)
    args = parser.parse_args()

    objects = list(get_objects(args.config))

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(
            main, args.config, _.object_name,
        ): (_.object_name,) for _ in objects}

        try:
            for fu in (tq := tqdm(as_completed(futures), total=len(futures))):
                try:
                    fu.result()
                    tq.set_description(f'{futures[fu]}')
                except Exception as _:
                    warnings.warn(f'{futures[fu]} {_}')
                    traceback.print_exc()

        except KeyboardInterrupt:
            print('Keyboard interrupted terminating...')
            executor.shutdown(wait=False)
            for future in futures:
                future.cancel()
