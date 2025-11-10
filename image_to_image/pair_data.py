import argparse
import tempfile
from pathlib import Path

import numpy as np
import tomlkit
from minio import Minio, S3Error
from totalsegmentator.python_api import totalsegmentator
from tqdm import tqdm


def main(cfg_path: str):
    cfg_path = Path(cfg_path)
    cfg = tomlkit.loads(cfg_path.read_text('utf-8'))

    client = Minio(**cfg['minio']['client'])
    bucket = cfg['minio']['bucket']

    objects = [_ for _ in client.list_objects(bucket, recursive=True) if not _.is_dir]

    patient_nii = {}
    for obj in objects:
        if len(obj.object_name.split('/')) != 2:
            continue

        if not obj.object_name.endswith('.nii.gz'):
            continue

        patient_id, filename = obj.object_name.split('/')

        if patient_id not in patient_nii:
            patient_nii[patient_id] = []

        patient_nii[patient_id].append(obj)

    print(len(patient_nii), np.average([len(_) for _ in patient_nii.values()]))

    table = {'femur_left': 75, 'femur_right': 76}

    for obj in tqdm(objects):
        if len(obj.object_name.split('/')) != 2:
            continue

        if not obj.object_name.endswith('.nii.gz'):
            continue

        total_name = obj.object_name.removesuffix('.nii.gz') + '/total.nii.gz'
        try:
            client.stat_object(bucket, total_name)
        except S3Error:
            continue

        with tempfile.TemporaryDirectory() as tdir:
            image = Path(tdir) / 'image.nii.gz'
            client.fget_object(bucket, obj.object_name, image.as_posix())

            total = Path(tdir) / 'total.nii.gz'
            totalsegmentator(image, total, True, task='total', quiet=True)

            client.fput_object(bucket, total_name, total.as_posix())

            import itk
            total = itk.imread(total)
            total = itk.array_from_image(total)
            print(obj.object_name, *[(roi, np.sum(total == value)) for roi, value in table.items()])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--max_workers', type=int, default=16)
    args = parser.parse_args()

    main(args.config)
