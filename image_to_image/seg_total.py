import argparse
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
from pathlib import Path

import tomlkit
from minio import Minio, S3Error


def main(cfg_path: str):
    cfg_path = Path(cfg_path)
    cfg = tomlkit.loads(cfg_path.read_text('utf-8'))
    client = Minio(**cfg['minio']['client'])

    for object_name, detect in cfg['detect'].items():
        if detect[0] == '无效':
            continue
        if detect[1] == '无效':
            continue
        if not detect[2]:
            continue

        try:
            client.stat_object('total', object_name)
            continue
        except S3Error:
            pass

        print(f'\n{object_name}')
        with tempfile.TemporaryDirectory() as tdir:
            image = Path(tdir) / 'image.nii.gz'
            label = Path(tdir) / 'label.nii.gz'

            client.fget_object('nii', object_name, image.as_posix())

            with ProcessPoolExecutor(max_workers=1) as executor:
                future = executor.submit(seg, image.as_posix(), label.as_posix())
                future.result()

            if label.exists():
                print(object_name, 'done')
                client.fput_object('total', object_name, label.as_posix())
            else:
                print(object_name, 'error')
                client.put_object('total', object_name, BytesIO(b''), 0)


def seg(image_path: str, label_path: str):
    from totalsegmentator.python_api import totalsegmentator
    totalsegmentator(image_path, label_path, True, task='total', quiet=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--max_workers', type=int, default=4)
    args = parser.parse_args()

    while True:
        main(args.config)

        for _ in range(60, 0, -1):
            print(f'\rRetry in {_}s...', end='', flush=True)
            time.sleep(1)
