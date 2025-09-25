import argparse
import locale
import warnings
from hashlib import sha1
from pathlib import Path

import itk
import numpy as np
import pydicom
from PIL import Image

locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')


def main(dicom_dir: str):
    dicom_dir = Path(dicom_dir)
    series = {}
    for f in dicom_dir.rglob('*'):
        ds = pydicom.dcmread(f)
        study = f'{ds.PatientName} {ds.StudyInstanceUID}'
        protocol = str(ds.ProtocolName).strip()
        if protocol.startswith('cine') and protocol.endswith('retro_sax'):
            pos = tuple(float(_) for _ in ds.ImagePositionPatient)
            axes = tuple(float(_) for _ in ds.ImageOrientationPatient)

            matrix = np.array([[*axes[:3], 0], [*axes[3:], 0], [*np.cross(axes[:3], axes[3:]), 0], [*pos, 1]])
            origin = tuple(np.linalg.inv(matrix)[3, :3])

            axes = sha1(np.array(axes).tobytes()).hexdigest()

            if study not in series:
                series[study] = {}

            if axes not in series[study]:
                series[study][axes] = {}

            frame = int(ds.InstanceNumber)
            if frame not in series[study][axes]:
                series[study][axes][frame] = {}

            series[study][axes][frame][origin[2]] = f

    cine_dir = dicom_dir.parent / f'{dicom_dir.name}.cine'

    for study in series:
        for axes in series[study]:
            for frame in series[study][axes]:
                images, spacing = [], None

                for k, z in enumerate(sorted(series[study][axes][frame])):
                    f = series[study][axes][frame][z]

                    _ = itk.imread(f.as_posix())
                    spc = np.array([*itk.spacing(_)])

                    if spacing is None:
                        spacing = spc
                    elif np.any(spacing != spc):
                        warnings.warn(f'conflict spacing {study} {axes} {spacing} {spc}')

                    _ = itk.array_from_image(_).squeeze()
                    images.append(_)

                    _ = _.astype(float)
                    mean, std = np.mean(_), np.std(_)
                    _ = np.clip((_ - mean) / std / 3 + 0.5, 0, 1) * 255
                    _ = _.astype(np.uint8)

                    f = cine_dir / f'{study}' / f'{axes}' / f'slice_{k + 1}' / f'frame_{frame}.png'
                    f.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(_).save(f)

                image = np.stack(images, axis=0)
                image = itk.image_from_array(image)
                image.SetSpacing(spacing)

                f = cine_dir / f'{study}' / f'{axes}' / f'frame_{frame}.nii.gz'
                f.parent.mkdir(parents=True, exist_ok=True)
                itk.imwrite(image, f.as_posix())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dicom_dir')
    args = parser.parse_args()

    try:
        main(args.dicom_dir)
    except KeyboardInterrupt:
        pass
