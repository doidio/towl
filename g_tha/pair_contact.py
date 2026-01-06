# uv run streamlit run pair_contact.py --server.port 8503 -- --config config.toml

import argparse
import tempfile
from pathlib import Path
from random import choice

import itk
import numpy as np
import streamlit as st
import tomlkit
import warp as wp
from minio import Minio, S3Error

from kernel import compose_op, fast_drr, cv2_line, resize_uint8, contact_drr

st.set_page_config('锦瑟医疗数据中心', initial_sidebar_state='collapsed', layout='wide')
st.markdown('### G-THA 骨与假体接触')

ct_bone, ct_metal = 220, 2700

if (it := st.session_state.get('init')) is None:
    with st.spinner('初始化', show_time=True):  # noqa
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', required=True)
        args = parser.parse_args()

        cfg_path = Path(args.config)
        cfg = tomlkit.loads(cfg_path.read_text('utf-8'))
        client = Minio(**cfg['minio']['client'])

        pairs = {}
        for _ in client.list_objects('pair', recursive=True):
            if not _.object_name.endswith('.nii.gz'):
                continue

            pid, rl, op, nii = _.object_name.split('/')
            prl = f'{pid}_{rl}'
            if prl not in pairs:
                pairs[prl] = {'prl': prl}
            pairs[prl][op] = f'{pid}/{nii}'

        for prl in pairs:
            try:
                data = client.get_object('pair', '/'.join([prl.replace('_', '/'), 'align.toml'])).data
                data = tomlkit.loads(data.decode('utf-8'))
                pairs[prl].update(data)
            except S3Error:
                pass

    st.session_state['init'] = client, pairs
    st.rerun()
elif (it := st.session_state.get('prl')) is None:
    client, pairs = st.session_state['init']

    dn = len([_ for _ in pairs if 'post_xform' in pairs[_]])
    ud = len(pairs) - dn

    st.progress(_ := dn / (dn + ud), text=f'{100 * _:.2f}%')
    st.metric('progress', f'{dn} / {dn + ud}', label_visibility='collapsed')

    if st.button('随机一个'):
        prl = choice([_ for _ in pairs.keys() if 'post_xform' in pairs[_]])
        st.session_state['prl_input'] = prl

    prl = st.text_input('PatientID_RL', key='prl_input')
    if prl in pairs:
        st.code(tomlkit.dumps(pairs[prl]), 'toml')

        if st.button('确定'):
            st.session_state['prl'] = prl
            st.rerun()

elif (it := st.session_state.get('roi')) is None:
    client, pairs = st.session_state['init']
    prl = st.session_state['prl']
    rl = prl.split('_')[1]
    label_femur = {'R': 76, 'L': 75}[rl]

    st.code(f'股骨阈值 = {ct_bone}  金属阈值 = {ct_metal}')
    st.code(tomlkit.dumps(pairs[prl]), 'toml')

    roi_images, sizes, spacings, image_bgs = [], [], [], []

    for op, object_name in enumerate((pairs[prl]['pre'], pairs[prl]['post'])):
        with tempfile.TemporaryDirectory() as tdir:
            with st.spinner(_ := '下载自动分割', show_time=True):  # noqa
                f = Path(tdir) / 'total.nii.gz'
                try:
                    client.fget_object('total', object_name, f.as_posix())
                except S3Error:
                    st.error(_ + '失败')
                    st.stop()

            with st.spinner('读取自动分割', show_time=True):  # noqa
                total = itk.imread(f.as_posix(), itk.UC)
                total = itk.array_from_image(total)

                if np.sum(total == label_femur) == 0:
                    st.error(f'自动分割不包含股骨 {label_femur}')
                    st.stop()

                ijk = np.argwhere(total == label_femur)
                b = np.array([ijk.min(axis=0), ijk.max(axis=0)])

            with st.spinner(_ := '下载原图', show_time=True):  # noqa
                f = Path(tdir) / 'image.nii.gz'
                try:
                    client.fget_object('nii', object_name, f.as_posix())
                except S3Error:
                    st.error(_ + '失败')
                    st.stop()

            with st.spinner(_ := '读取原图', show_time=True):  # noqa
                image = itk.imread(f.as_posix(), itk.SS)

                size = np.array([float(_) for _ in reversed(itk.size(image))])
                spacing = np.array([float(_) for _ in reversed(itk.spacing(image))])

                sizes.append(size)
                spacings.append(spacing)

                image = itk.array_from_image(image)
                image_bg = float(np.min(image))
                image_bgs.append(image_bg)

            with st.spinner(_ := '提取子区', show_time=True):  # noqa
                roi_image = image[b[0, 0]:b[1, 0] + 1, b[0, 1]:b[1, 1] + 1, b[0, 2]:b[1, 2] + 1]
                roi_total = total[b[0, 0]:b[1, 0] + 1, b[0, 1]:b[1, 1] + 1, b[0, 2]:b[1, 2] + 1]

                # 抹除子区中的非股骨高亮体素
                # roi_image[np.where((roi_total != label_femur) & (roi_image > ct_bone))] = image_bg
                roi_images.append(roi_image)

    st.session_state['roi'] = roi_images, sizes, spacings, image_bgs
    st.rerun()
else:
    client, pairs = st.session_state['init']
    prl = st.session_state['prl']
    pid, rl = prl.split('_')
    roi_images, sizes, spacings, image_bgs = st.session_state['roi']

    st.code(f'股骨阈值 = {ct_bone}  金属阈值 = {ct_metal}')
    st.code(tomlkit.dumps(pairs[prl]), 'toml')

    post_xform = wp.transform(*pairs[prl]['post_xform'])
    xforms = (wp.transform_inverse(post_xform), post_xform)
    volumes = [wp.Volume.load_from_numpy(roi_images[_], bg_value=image_bgs[_]) for _ in range(2)]

    # 基于术前/术后空间格式重采样
    for a, b in ((0, 1), (1, 0)):
        with st.expander(['术前空间', '术后空间'][a], False):
            image_a = wp.from_numpy(roi_images[a], float)
            image_b = wp.empty_like(image_a)
            image_c = wp.empty_like(image_a)
            wp.launch(compose_op, image_b.shape, [
                image_a, image_b, image_c,
                xforms[a], wp.uint64(volumes[b].id), spacings[a], spacings[b],
                float(ct_bone), float(ct_metal), a > 0,
            ])
            images = [_.numpy() for _ in (image_a, image_b, image_c)]

            cols = st.columns(3, vertical_alignment='bottom')

            with cols[0]:
                if a == 1:  # 读取配准转换的中轴坐标
                    proximal_ = st.session_state['proximal_']
                    distal_ = st.session_state['distal_']
                    if st.button('一键带入 {} {}'.format(proximal_, distal_)):
                        for i in range(3):
                            if 0 <= proximal_[i] < image_a.shape[i]:
                                st.session_state[f'{a}proximal{i}'] = proximal_[i]
                            if 0 <= distal_[i] < image_a.shape[i]:
                                st.session_state[f'{a}distal{i}'] = distal_[i]

                # 输入中轴坐标
                m = [image_a.shape[i] - 1 for i in range(3)]
                proximal, distal = [[c.number_input(
                    '{} {} (0 ~ {})'.format(['近端', '远端'][k], 'ZYX'[i], m[i]),
                    0, m[i], (m[i] if k == 0 else 0) if i == 0 else round(m[i] / 2), key=f'{a}{key}{i}',
                ) for i, c in enumerate(st.columns(3))] for k, key in enumerate(('proximal', 'distal'))]

                if a == 0:  # 计算配准转换的中轴坐标
                    for key, _ in {'proximal': proximal, 'distal': distal}.items():
                        _ = np.array(_) * spacings[a]
                        _ = wp.transform_point(xforms[a], wp.vec3(_))
                        _ = np.array(_) / spacings[b]
                        _ = [round(_) for _ in _]
                        st.session_state[key + '_'] = _

            # 正位/侧位叠加
            for ax in range(3):
                spacing = spacings[a].tolist().copy()
                del spacing[ax]

                if ax == 0:
                    for i, z in enumerate((proximal[0], distal[0])):
                        p0, p1 = proximal.copy(), distal.copy()
                        del p0[ax], p1[ax]

                        stack = []
                        for _ in images:
                            img = fast_drr(_[z:z + 1], ax)
                            cv2_line(img, p0, p1, (255, 0, 0), 2)
                            stack.append(resize_uint8(img, np.array(spacing) * img.shape[:2] * 5))

                        img = np.hstack(stack)
                        if ax in (1, 2):
                            img = np.flipud(img)
                        _ = ['近端', '远端'][i], ['术前', '术后'][a], ['术前', '术后'][b], '骨与假体融合'
                        cols[ax].image(img, '{} ({}, {}, {})'.format(*_))
                else:
                    p0, p1 = proximal.copy(), distal.copy()
                    del p0[ax], p1[ax]

                    stack = []
                    for _ in images:
                        img = fast_drr(_, ax)
                        cv2_line(img, p0, p1, (255, 0, 0), 2)
                        stack.append(resize_uint8(img, np.array(spacing) * img.shape[:2]))

                    img = np.hstack(stack)
                    if ax in (1, 2):
                        img = np.flipud(img)
                    _ = ['轴位', '正位', '侧位'][ax], ['术前', '术后'][a], ['术前', '术后'][b], '骨与假体融合'
                    cols[ax].image(img, '{} ({}, {}, {})'.format(*_))

            cols = st.columns(4, vertical_alignment='bottom')

            # 正位/侧位接触
            th = 0.0, 900.0
            for i in range(4):
                ax = i % 2 + 1  # 前A后P内M外L

                p0, p1 = proximal.copy(), distal.copy()
                del p0[ax], p1[ax]

                spacing = spacings[a].tolist().copy()
                del spacing[ax]

                stack = []
                for image_3 in (image_a, image_b, image_c):
                    size = list(image_3.shape)
                    del size[ax]

                    image_2 = wp.empty((*size,), wp.vec3ub)
                    wp.launch(contact_drr, image_2.shape, [
                        image_a, image_b, image_3, image_2,
                        float(ct_bone), float(ct_metal), image_bgs[0], th[0], th[1] - th[0], ax, a > 0, i // 2 > 0,
                    ])
                    image_2 = image_2.numpy()
                    # cv2_line(image_2, p0, p1, (255, 0, 0), 2)
                    stack.append(resize_uint8(image_2, np.array(spacing) * image_2.shape[:2]))

                img = np.hstack(stack)
                if ax in (1, 2):
                    img = np.flipud(img)
                _ = ['前方（A）', '内侧（M）', '后方（P）', '外侧（L）']
                if rl == 'R':
                    _[1], _[3] = _[3], _[1]
                cols[i].image(img, _[i] + '接触')
