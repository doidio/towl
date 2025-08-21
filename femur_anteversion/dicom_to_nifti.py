import argparse
import locale
from pathlib import Path

import itk

# 支持中文路径
locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')


def main(path: str, series_uid: int | str):
    """
    :param path: 指向DICOM目录，可能包含多个Series，不支持子目录递归
    :param series_uid: 指定读取的Series
        当 series_uid 为字符串时，读取与该 UID 匹配的 DICOM 序列
        当 series_uid 为整数时，按 Python 索引方式读取对应顺序的序列（如 -1 表示最后一个）
        若未指定 series_uid，默认读取目录中的第一个序列
    """
    path = Path(path).absolute()
    image = itk.imread(path.as_posix(), series_uid=series_uid)
    print(*itk.size(image))
    itk.imwrite(image, path.parent / f'{path.stem}.nii.gz')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True)
    parser.add_argument('--series_uid', default=0)
    args = parser.parse_args()

    main(args.path, args.series_uid)
