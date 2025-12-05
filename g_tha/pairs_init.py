import argparse
import locale
import tempfile
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from io import BytesIO
from pathlib import Path

import itk
import pydicom
import tomlkit
from minio import Minio
from tqdm import tqdm

locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')


def main(cfg_path: str, series: dict):
    cfg_path = Path(cfg_path)
    cfg = tomlkit.loads(cfg_path.read_text('utf-8'))
    client = Minio(**cfg['minio']['client'])

    collection = {'术前': {}, '术后': {}}

    # 读取时间
    for object_name, pp in series:
        dcm = object_name.removesuffix('.nii.gz') + '.dcm'
        dcm = client.get_object('dcm', dcm).data
        dcm = pydicom.dcmread(BytesIO(dcm))

        dt = datetime(
            year=int(dcm.StudyDate[0:4]),
            month=int(dcm.StudyDate[4:6]),
            day=int(dcm.StudyDate[6:8]),
            hour=int(dcm.StudyTime[0:2]),
            minute=int(dcm.StudyTime[2:4]),
            second=int(dcm.StudyTime[4:6]),
        )

        if dt not in collection[pp]:
            collection[pp][dt] = []

        collection[pp][dt].append(object_name)

    # 筛选相同时间
    for pp in collection:
        for dt in collection[pp]:
            if len(collection[pp][dt]) > 1:
                spacing_min = None
                object_name_best = None

                for object_name in collection[pp][dt]:
                    with tempfile.TemporaryDirectory() as tdir:
                        f = Path(tdir) / '.nii.gz'
                        client.fget_object('nii', object_name, f.as_posix())
                        spacing = float(itk.spacing(itk.imread(f.as_posix()))[2])

                        if spacing_min is None or spacing_min > spacing:
                            spacing_min = spacing
                            object_name_best = object_name

                collection[pp][dt] = object_name_best
            elif len(collection[pp][dt]) > 0:
                collection[pp][dt] = collection[pp][dt][0]
            else:
                raise RuntimeError

    # 同人同侧末次术前早于首次术后
    if len(collection['术前']) > 0 and len(collection['术后']) > 0:
        dt = max(collection['术前']), min(collection['术后'])
        object_names = collection['术前'][dt[0]], collection['术后'][dt[1]]
        if dt[0] < dt[1]:
            return object_names

    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--images', required=True)
    parser.add_argument('--pairs', required=True)
    parser.add_argument('--max_workers', type=int, default=1)
    args = parser.parse_args()

    images_path = Path(args.images)
    images = tomlkit.loads(images_path.read_text('utf-8'))

    # patient/right_left
    table = {}
    for obj, valid in images['images'].items():
        if not valid[2]:
            continue

        patient, _ = obj.split('/')

        if patient not in table:
            table[patient] = [[], []]

        for rl in range(2):
            if valid[rl] != '无效':
                table[patient][rl].append((obj, valid[rl]))

    items = [(patient, rl, table[patient][rl]) for patient in table for rl in range(2)]

    pairs_path = Path(args.pairs)
    assert not pairs_path.exists()

    pairs = {}

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(main, args.config, _[-1]): _ for _ in items}
        count = 0

        try:
            for fu in (bar := tqdm(as_completed(futures), total=len(futures))):
                try:
                    if (result := fu.result()) is not None:
                        patient, rl, _ = futures[fu]
                        if patient not in pairs:
                            pairs[patient] = {}
                        pairs[patient]['RL'[rl]] = [-1, [], *result]
                        pairs_path.write_text(tomlkit.dumps(pairs), 'utf-8')

                        count += 1
                        bar.set_postfix({'pairs': count})
                except Exception as _:
                    warnings.warn(f'{_} {futures[fu]}', stacklevel=2)

        except KeyboardInterrupt:
            print('Keyboard interrupted terminating...')
            executor.shutdown(wait=False)
            for future in futures:
                future.cancel()
