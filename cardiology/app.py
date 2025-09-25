import locale
import shutil
import subprocess
import sys
import warnings
from hashlib import sha1
from pathlib import Path

import itk
import numpy as np
import pydicom
import streamlit as st
from PIL import Image

locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')


def main(dicom_dir, cine_dir):
    series, total = {}, 0

    with st.status('', expanded=True) as s:
        bar = st.progress(0)

        files = list(dicom_dir.rglob('*'))

        for _, f in enumerate(files, 1):
            s.update(label=f'[ {_} / {len(files)} ] 读取 {f.as_posix()}')
            bar.progress(_ / len(files))

            ds = pydicom.dcmread(f)
            study = str(ds.StudyInstanceUID)
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
                total += 1

        processed = 0

        with st.empty():
            for study in series:
                for axes in series[study]:
                    for frame in series[study][axes]:
                        images, pngs, spacing = [], [], None

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
                            pngs.append(_)

                            f = cine_dir / f'{study}' / f'{axes}' / f'slice_{k + 1}' / f'frame_{frame}.png'
                            f.parent.mkdir(parents=True, exist_ok=True)
                            Image.fromarray(_).save(f)

                            processed += 1
                            s.update(label=f'[ {processed} / {total} ] 写入 {f.as_posix()}')
                            bar.progress(processed / total)

                        st.image(np.hstack(pngs))

                        image = np.stack(images, axis=0)
                        image = itk.image_from_array(image)
                        image.SetSpacing(spacing)

                        f = cine_dir / f'{study}' / f'{axes}' / f'{study}_{axes}_frame_{frame}.nii.gz'
                        f.parent.mkdir(parents=True, exist_ok=True)
                        itk.imwrite(image, f.as_posix())


if __name__ == '__main__':
    st.set_page_config('医学影像 AI 平台', layout='wide', initial_sidebar_state='collapsed')
    st.sidebar.markdown(f'## 上海九院心脏科')

    tab1, tab2 = st.tabs(['AI辅助分割', 'cine-MRI转换'])

    with tab1:
        st.markdown('服务器 `127.0.0.1`')
        if st.button('启动服务'):
            subprocess.Popen(
                ['cmd', '/C', sys.executable, '-m', 'itksnap_dls', '||', 'pause'],
                creationflags=subprocess.CREATE_NEW_CONSOLE,
            )

    with tab2:
        if _ := st.text_input('请输入DICOM目录'):
            if len(_) > 0:
                _dicom_dir = Path(_)
                if _dicom_dir.is_dir() and _dicom_dir.parent != _dicom_dir:
                    _cine_dir = _dicom_dir.parent / f'{_dicom_dir.name}.cine'
                    st.write(f'输出目录：{_cine_dir.as_posix()}')

                    with st.empty():
                        if st.button('开始转换'):
                            shutil.rmtree(_cine_dir, ignore_errors=True)
                            main(_dicom_dir, _cine_dir)
                            subprocess.Popen(['explorer', _cine_dir.expanduser().resolve()])
                            st.rerun()
