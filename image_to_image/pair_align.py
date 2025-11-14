import argparse
import warnings
from datetime import datetime
from io import BytesIO
from pathlib import Path

import pydicom
import tomlkit
from minio import Minio


def main(cfg_path: str):
    cfg_path = Path(cfg_path)
    cfg = tomlkit.loads(cfg_path.read_text('utf-8'))
    client = Minio(**cfg['minio']['client'])

    # patient/right_left/pre_post_list
    table = {}
    for object_name, valid in cfg['detect'].items():
        if not valid[2]:
            continue

        patient, series = object_name.split('/')

        if patient not in table:
            table[patient] = [[], []]

        for rl in range(2):
            if valid[rl] != '无效':
                table[patient][rl].append((object_name, valid[rl]))

    side = ('右髋', '左髋')

    pairs = []
    for patient in table:
        for rl in range(2):
            pre, post = [], []
            for object_name, pp in table[patient][rl]:
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

                if pp == '术前':
                    pre.append((object_name, pp, dcm.StudyTime, dcm.StudyDate, dt))
                elif pp == '术后':
                    post.append((object_name, pp, dcm.StudyTime, dcm.StudyDate, dt))

            pre = sorted(pre, key=lambda _: _[-1])
            post = sorted(post, key=lambda _: _[-1])

            if len(pre) > 0 and len(post) > 0:
                if pre[-1][-1] < post[0][-1]:  # 同人同侧末次术前早于首次术后
                    pairs.append((patient, rl, pre[-1][0], post[0][0]))
                    print(patient, side[rl], '术前', pre[-1][-1], '术后', post[0][-1])
                else:
                    _ = [_[-2] + _[-3] for _ in pre], [_[-2] + _[-3] for _ in post]
                    warnings.warn(f'{patient} {side[rl]} 术前 {_[0]} 术后 {_[1]}')

    print(len(pairs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--max_workers', type=int, default=4)
    args = parser.parse_args()

    main(args.config)
