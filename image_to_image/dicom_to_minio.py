import argparse
import warnings
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import tomlkit
from tqdm import tqdm


class FatalError(Exception):
    pass


def main(zip_file: str, cfg_path: str):
    cfg_path = Path(cfg_path)
    cfg = tomlkit.loads(cfg_path.read_text('utf-8'))

    from minio import Minio
    client = Minio(**cfg['minio']['client'])
    bucket = cfg['minio']['bucket']

    if bucket not in {_.name for _ in client.list_buckets()}:
        client.make_bucket(bucket)

    with zipfile.ZipFile(zip_file, 'r') as zf:
        for file in zf.filelist:
            data = zf.read(file)

            import pydicom
            from pydicom.errors import InvalidDicomError
            try:
                it = pydicom.dcmread(BytesIO(data))
            except (InvalidDicomError, ValueError):
                continue

            key = f'{it.StudyInstanceUID}/{it.SeriesInstanceUID}/{it.SOPInstanceUID}'

            try:
                client.put_object(bucket, key, BytesIO(data), len(data))
            except Exception as _:
                raise FatalError(str(_)) from None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip_dir', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--max_workers', type=int, default=16)
    args = parser.parse_args()

    zip_dir = Path(args.zip_dir).absolute()
    zip_files = [_ for _ in zip_dir.rglob('*.zip')]

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(
            main, _.as_posix(), args.config,
        ): (_.as_posix(), args.config,) for _ in zip_files}

        try:
            for fu in tqdm(as_completed(futures), total=len(futures)):
                try:
                    fu.result()
                except FatalError as _:
                    raise _
                except Exception as _:
                    warnings.warn(f'{_} {futures[fu]}')

        except KeyboardInterrupt:
            print('Keyboard interrupted terminating...')
            executor.shutdown(wait=False)
            for future in futures:
                future.cancel()
