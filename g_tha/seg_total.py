import argparse
import tempfile
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
from pathlib import Path

import tomlkit
from minio import Minio, S3Error


def main(cfg_path: str):
    cfg_path = Path(cfg_path)
    cfg = tomlkit.loads(cfg_path.read_text('utf-8'))
    client = Minio(**cfg['minio']['client'])

    object_names = set()
    for object_name, detect in cfg['detect'].items():
        if detect[0] == '无效':
            continue
        if detect[1] == '无效':
            continue
        if not detect[2]:
            continue

        object_names.add(object_name)

    total = len([_ for _ in client.list_objects('total', recursive=True)
                 if not _.is_dir and _.object_name.endswith('.nii.gz')])
    print(f'\n[{total}/{len(object_names)}]')

    for object_name in object_names:
        try:
            client.stat_object('total', object_name)
            continue
        except S3Error:
            pass

        total += 1
        print(_ := f'\n[{total}/{len(object_names)}] {object_name}')

        with tempfile.TemporaryDirectory() as tdir:
            image = Path(tdir) / 'image.nii.gz'
            label = Path(tdir) / 'label.nii.gz'

            client.fget_object('nii', object_name, image.as_posix())

            with ProcessPoolExecutor(max_workers=1) as executor:
                future = executor.submit(seg, image.as_posix(), label.as_posix())
                try:
                    future.result()
                except Exception as e:
                    warnings.warn(str(e))

            if label.exists():
                print(_, 'done')
                client.fput_object('total', object_name, label.as_posix())
            else:
                print(_, 'error')
                client.put_object('total', object_name, BytesIO(b''), 0)


def seg(image_path: str, label_path: str):
    from totalsegmentator.python_api import totalsegmentator
    totalsegmentator(image_path, label_path, True, task='total', quiet=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    while True:
        main(args.config)

        for _ in range(60, 0, -1):
            print(f'\rRetry in {_}s...', end='', flush=True)
            time.sleep(1)
