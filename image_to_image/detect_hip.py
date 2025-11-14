# uv run streamlit run detect_hip.py --server.port 8501 -- --config config.toml

import argparse
import locale
import tempfile
from datetime import datetime
from io import BytesIO
from pathlib import Path

import itk
import numpy as np
import pydicom
import streamlit as st
import tomlkit
from PIL import Image
from matplotlib import cm
from minio import Minio

locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')

th = (0, 800)


def _drr(a, axis):
    a = image.copy()
    c = th[0] <= a
    a = (a * c).sum(axis=axis)
    c = np.sum(c, axis=axis)
    c[np.where(c <= 0)] = 1
    a = a / c

    sm = cm.ScalarMappable(cmap='grey')
    sm.set_clim(th)
    a = sm.to_rgba(a, bytes=True)

    if axis in (1, 2):
        a = np.flipud(a)

    return a


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = tomlkit.loads(cfg_path.read_text('utf-8'))
    client = Minio(**cfg['minio']['client'])

    st.set_page_config('锦瑟医疗数据中心', initial_sidebar_state='collapsed')
    st.markdown('### 全髋关节置换数据分类')

    if 'detect' not in cfg:
        cfg['detect'] = {}

    count, total = len(cfg['detect']), cfg['minio']['nii']['objects']
    st.progress(count / total, text=f'{100 * count / total:.2f}%')
    st.caption(f'{count} / {total}')

    if (it := st.session_state.get('it')) is None:
        with st.empty():
            if st.button('🐋 下一个 🐳'):
                with st.spinner('检索', show_time=True):
                    for it in client.list_objects('nii', recursive=True):
                        if it.is_dir:
                            continue

                        if it.object_name in cfg['detect']:
                            continue

                        st.session_state['it'] = it
                        break

                with tempfile.TemporaryDirectory() as tdir:
                    f = Path(tdir) / 'image.nii.gz'

                    with st.spinner('下载', show_time=True):
                        client.fget_object('nii', it.object_name, f.as_posix())

                        name = it.object_name.removesuffix('.nii.gz') + '.dcm'
                        dcm = client.get_object('dcm', name).data
                        dcm = pydicom.dcmread(BytesIO(dcm))

                    with st.spinner('读取', show_time=True):
                        image = itk.imread(f)

                info = itk.dict_from_image(image)
                del info['name'], info['bufferedRegion'], info['data']

                image = itk.array_from_image(image)
                info['imageType']['range'] = np.array([np.min(image), np.max(image)])
                info['origin'] = np.array(info['origin'])
                info['spacing'] = np.array(info['spacing'])
                info['size'] = np.array(info['size'])
                info['dicom'] = {
                    'ImageType': dcm.get('ImageType'),
                }

                if info['imageType']['dimension'] != 3:
                    drr = None
                elif info['imageType']['componentType'] in ('uint8',):
                    drr = None
                elif info['imageType']['components'] != 1:
                    drr = None
                else:
                    with st.spinner('透视', show_time=True):
                        l = np.array(info['spacing']) * np.array(info['size'])
                        l = tuple(max(round(_) * 2, 1) for _ in l)
                        drr = []
                        for _ in range(2):
                            x = _drr(image.copy(), _)
                            x = Image.fromarray(x).resize([(l[0], l[1]), (l[0], l[2])][_])
                            drr.append(np.array(x))

                st.session_state['info'] = info
                st.session_state['drr'] = drr

                st.rerun()
    else:
        info = st.session_state['info']
        drr = st.session_state['drr']

        st.info(it.object_name)

        st.caption('轴位')
        if drr:
            st.image(drr[0])
        else:
            st.warning('透视失败')

        axial_ok = st.checkbox('(1/3) 上前下后')

        st.caption('正位')
        if drr:
            st.image(drr[1])
        else:
            st.warning('透视失败')

        coronal_l = st.radio('(2/3) 左髋 👉', ['无效', '术前', '术后'])
        coronal_r = st.radio('(3/3) 右髋 👈', ['无效', '术前', '术后'])

        st.write(info)

        info_ok = False
        if info['imageType']['dimension'] != 3:
            st.warning('图像不是三维')
        elif info['imageType']['componentType'] not in ('int16', 'int32'):
            st.warning('图像不是有效值型 {}'.format(info['imageType']['componentType']))
        elif info['imageType']['components'] != 1:
            st.warning('图像不是单通道')
        else:
            info_ok = True

        try:
            tag = info['dicom']['ImageType']
            for _ in ('DERIVED', 'SECONDARY', 'MPR'):
                if _ in tag:
                    info_ok = False
                    st.warning(f'图像不是原始数据 {tag}')
                    break
        except (TypeError, Exception):
            info_ok = False
            st.warning(f'图像缺失 DICOM 属性 ImageType')

        if st.button('提交'):
            cfg['detect'][it.object_name] = [coronal_r, coronal_l, info_ok, datetime.now()]
            cfg_path.write_text(tomlkit.dumps(cfg), 'utf-8')

            for _ in ('it', 'info', 'drr'):
                del st.session_state[_]
            st.rerun()
