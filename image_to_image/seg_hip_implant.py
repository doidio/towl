import argparse
import tempfile
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import tomlkit
from minio import Minio
from totalsegmentator.python_api import totalsegmentator
from tqdm import tqdm


def main(cfg_path: str, object_name: str):
    cfg_path = Path(cfg_path)
    cfg = tomlkit.loads(cfg_path.read_text('utf-8'))
    client = Minio(**cfg['minio']['client'])

    with_implant = False
    with tempfile.TemporaryDirectory() as tdir:
        image = Path(tdir) / 'image.nii.gz'
        label = Path(tdir) / 'label.nii.gz'

        client.fget_object('nii', object_name, image.as_posix())
        totalsegmentator(image, label, True, task='hip_implant', quiet=True)

        import itk
        _ = itk.imread(label)
        _ = itk.array_from_image(_)

        if np.sum(_) > 0:
            with_implant = True
            client.fput_object('hip-implant', object_name, label.as_posix())

    if with_implant:
        return object_name.split('/')[0]

    return None


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
            yield obj


def add_patient(cfg_path: str, patient_id: str):
    cfg_path = Path(cfg_path)
    cfg = tomlkit.loads(cfg_path.read_text('utf-8'))

    if 'hip_implant' not in cfg or 'PatientID' not in cfg['hip_implant']:
        cfg['hip_implant'] = {'PatientID': []}

    ids = set(cfg['hip_implant']['PatientID'])
    ids.add(patient_id)
    cfg['hip_implant']['PatientID'] = list(ids)
    cfg_path.write_text(tomlkit.dumps(cfg), 'utf-8')


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
                    if (_ := fu.result()) is not None:
                        add_patient(args.config, _)
                    tq.set_description(f'{futures[fu]}')
                except Exception as _:
                    warnings.warn(f'{futures[fu]} {_}')
                    traceback.print_exc()

        except KeyboardInterrupt:
            print('Keyboard interrupted terminating...')
            executor.shutdown(wait=False)
            for future in futures:
                future.cancel()
