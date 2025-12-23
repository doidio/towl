# uv run streamlit run drr_to_pairs.py --server.port 8501 -- --config config.toml --pairs pairs.toml

import argparse
import locale
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
import tomlkit
from PIL import Image
from minio import Minio, S3Error

locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--pairs', required=True)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = tomlkit.loads(cfg_path.read_text('utf-8'))
    client = Minio(**cfg['minio']['client'])

    pairs_path = Path(args.pairs)
    pairs: dict = tomlkit.loads(pairs_path.read_text('utf-8'))

    st.set_page_config('锦瑟医疗数据中心', initial_sidebar_state='collapsed')
    st.markdown('### 全髋关节置换术前术后配对')

    if (it := st.session_state.get('ud')) is None:
        with st.spinner('下一个', show_time=True):  # noqa
            dn, ud = [], []
            for _ in client.list_objects('drr'):
                if not _.is_dir:
                    continue

                object_name = _.object_name[:-1]
                (dn if object_name in pairs else ud).append(object_name)

        if len(ud):
            st.session_state['dn'] = dn
            st.session_state['ud'] = ud
            st.rerun()
        else:
            st.balloons()
            st.success('全部完成')
            st.stop()

    elif (it := st.session_state.get('patient')) is None:
        dn = st.session_state['dn']
        ud = st.session_state['ud']

        with st.spinner('下一个', show_time=True):  # noqa
            patient = ud[0]
            series = []

            for _ in client.list_objects('drr', patient + '/'):
                if not _.is_dir:
                    continue

                object_name = _.object_name[:-1]

                try:
                    client.stat_object('drr', f'{object_name}/invalid')
                    continue
                except S3Error:
                    pass

                with tempfile.TemporaryDirectory() as tdir:
                    f = Path(tdir) / 'info.toml'
                    client.fget_object('drr', f'{object_name}/{f.name}', f.as_posix())
                    info = tomlkit.loads(f.read_text('utf-8'))

                    if any((
                            'DERIVED' in info['dicom']['ImageType'],
                            'SECONDARY' in info['dicom']['ImageType'],
                    )):
                        continue

                    f = Path(tdir) / 'axial.png'
                    client.fget_object('drr', f'{object_name}/{f.name}', f.as_posix())
                    axial = np.asarray(Image.open(f.as_posix()))

                    f = Path(tdir) / 'coronal.png'
                    client.fget_object('drr', f'{object_name}/{f.name}', f.as_posix())
                    coronal = np.asarray(Image.open(f.as_posix()))

                    StudyDate, StudyTime = info['dicom']['StudyDate'], info['dicom']['StudyTime']
                    dt = datetime(
                        year=int(StudyDate[0:4]),
                        month=int(StudyDate[4:6]),
                        day=int(StudyDate[6:8]),
                        hour=int(StudyTime[0:2]),
                        minute=int(StudyTime[2:4]),
                        second=int(StudyTime[4:6]),
                        microsecond=int(StudyTime[7:13]) if len(StudyTime) >= 13 else 0,
                    )
                    series.append([dt, info, axial, coronal, object_name])

            series = sorted(series, key=lambda _: _[0])

        st.session_state['patient'] = patient
        st.session_state['series'] = series
        st.rerun()
    else:
        dn = st.session_state['dn']
        ud = st.session_state['ud']

        patient = st.session_state['patient']
        series = st.session_state['series']

        R0 = st.session_state.get('R0')
        R1 = st.session_state.get('R1')
        L0 = st.session_state.get('L0')
        L1 = st.session_state.get('L1')

        st.progress(_ := len(dn) / (len(dn) + len(ud)), text=f'{100 * _:.2f}%')
        st.metric(f'PatientID {patient}', f'{len(dn)} / {len(dn) + len(ud)}')

        with st.form('submit'):
            options = {f'[{i}]': i for i, t in enumerate(series)}
            rl = st.columns(2)
            for i, c in enumerate(rl):
                rl[i] = c.multiselect(['右侧', '左侧'][i], options.keys())

            n = set(len(_) for _ in rl)

            if st.form_submit_button('提交'):
                if len(n - {0, 2}) > 0:
                    st.error('选择数量错误，只能双选或不选')
                else:
                    pairs[patient] = {}
                    for i in [i for i, c in enumerate(rl) if len(c) == 2]:
                        pairs[patient]['RL'[i]] = [-1, [], *[series[options[rl[i][_]]][-1] for _ in range(2)]]
                    pairs_path.write_text(tomlkit.dumps(pairs), 'utf-8')

                    for _ in ('dn', 'ud', 'patient', 'series'):
                        del st.session_state[_]
                    st.rerun()

        for i in options.values():
            t = series[i]
            with st.expander(f'[{i}] {t[0]}', expanded=True):
                st.caption(t[-1].split('/')[1])
                col1, col2 = st.columns(2)
                with col1:
                    st.image(t[3])
                with col2:
                    st.image(t[2])

                st.code(tomlkit.dumps(t[1]), 'toml')
