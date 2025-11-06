import argparse
import warnings
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import boto3
import pydicom
import tomlkit
from pydicom.errors import InvalidDicomError
from tqdm import tqdm


def main(zip_file: str, cfg_path: str):
    cfg_path = Path(cfg_path)
    cfg = tomlkit.loads(cfg_path.read_text('utf-8'))

    s3 = boto3.client('s3', **cfg['s3']['client'])
    bucket = cfg['s3']['bucket']

    if bucket not in {_['Name'] for _ in s3.list_buckets()['Buckets']}:
        print(s3.list_buckets()['Buckets'])

    with zipfile.ZipFile(zip_file) as zf:
        for file in zf.namelist():
            with zf.open(file) as f:
                try:
                    it = pydicom.dcmread(f)
                except (InvalidDicomError, ValueError):
                    continue

                key = f'{it.StudyInstanceUID}/{it.SeriesInstanceUID}/{it.SOPInstanceUID}'
                f.seek(0)
                s3.upload_fileobj(f, bucket, key)


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
            for _ in tqdm(as_completed(futures), total=len(futures)):
                try:
                    _.result()
                except Exception as e:
                    warnings.warn(f'{e} {futures[_]}')

        except KeyboardInterrupt:
            print('Keyboard interrupted terminating...')
            executor.shutdown(wait=False)
            for future in futures:
                future.cancel()
