# uv run streamlit run pair_pak.py --server.port 8503 -- --config config.toml

import argparse
import tempfile
from pathlib import Path
from random import choice

import itk
import numpy as np
import streamlit as st
import tomlkit
import trimesh.primitives
import warp as wp
from minio import Minio, S3Error
from stpyvista import stpyvista
from stpyvista.panel_backend import PanelVTKKwargs

from kernel import fast_drr, diff_dmc, resample_obb

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

            if op not in ('pre', 'post'):
                continue

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

    dn = len([_ for _ in pairs if 'post_xform' in pairs[_] and len(pairs[_].get('excluded', [])) == 0])
    ud = len(pairs) - dn

    st.progress(_ := dn / (dn + ud), text=f'{100 * _:.2f}%')
    st.metric('progress', f'{dn} / {dn + ud} 对配准腿', label_visibility='collapsed')

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

    ct_images, roi_bounds, ct_femurs, sizes, spacings, origins, image_bgs = [], [], [], [], [], [], []

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

                ijk = np.argwhere(_ := (total == label_femur))
                ct_femurs.append(_)

                box = np.array([ijk.min(axis=0), ijk.max(axis=0) + 1])
                roi_bounds.append(box.tolist())

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
                origin = np.array([float(_) for _ in reversed(itk.origin(image))])

                sizes.append(size)
                spacings.append(spacing)
                origins.append(origin)

                image = itk.array_from_image(image)
                ct_images.append(image)

                image_bg = float(np.min(image))
                image_bgs.append(image_bg)

    st.session_state['roi'] = ct_images, roi_bounds, ct_femurs, sizes, spacings, origins, image_bgs
    st.rerun()
else:
    client, pairs = st.session_state['init']
    prl = st.session_state['prl']
    pid, rl = prl.split('_')
    ct_images, roi_bounds, ct_femurs, sizes, spacings, origins, image_bgs = st.session_state['roi']

    st.code(f'股骨阈值 = {ct_bone}  金属阈值 = {ct_metal}')
    st.code(tomlkit.dumps(pairs[prl]), 'toml')

    # 根据术后股骨与金属交集确定采样范围
    _ = ct_femurs[1] & (ct_images[1] >= ct_metal)
    raw_mesh = diff_dmc(wp.from_numpy(_, wp.float32), origins[1], spacings[1], 0.5)

    # 金属可能分离成髋臼杯、球头、股骨柄、膝关节假体，选范围最大的股骨柄以上
    if not raw_mesh.is_empty:
        ls = list(sorted(
            raw_mesh := raw_mesh.split(only_watertight=True),
            key=lambda _: np.linalg.norm(_.bounds[1] - _.bounds[0]), reverse=True,
        ))
        selected = st.multiselect('选择股骨柄（金属部件按尺寸从大到小）', [_ for _ in range(len(ls))], default=0)
        if len(selected) > 1:
            mesh: trimesh.Trimesh = trimesh.util.concatenate([ls[_] for _ in selected])  # noqa
        elif len(selected) > 0:
            mesh: trimesh.Trimesh = ls[selected[0]]
        else:
            st.stop()
    else:
        st.error('ROI 错误')
        st.stop()

    obb = mesh.bounding_box_oriented
    _ = mesh.copy()
    _.apply_transform(np.linalg.inv(obb.transform))
    extents = _.bounds[1] - _.bounds[0]
    x = np.argmax(extents)

    obb_xform = obb.transform.copy()
    indices = [(x + i) % 3 for i in range(3)]
    obb_xform[:3, :3] = obb.transform[:3, :3][:, indices]

    if obb_xform[0, 0] < 0:
        obb_xform[:3, 0] *= -1
        obb_xform[:3, 2] *= -1
    if obb_xform[1, 1] < 0:
        obb_xform[:3, 1] *= -1
        obb_xform[:3, 2] *= -1

    iso = 0.5

    _ = mesh.copy()
    _.apply_transform(np.linalg.inv(obb_xform))
    extents = _.bounds[1] - _.bounds[0]
    extents += np.max(extents * 0.1)
    extents = np.ceil(extents / iso / 16).astype(int) * 16

    st.code(tomlkit.dumps({'extents': extents.tolist(), 'size': (size := [576, 224, 352])}), 'toml')

    if 'post_xform_global' in pairs[prl]:
        post_xform = wp.transform(*pairs[prl]['post_xform_global'])
    elif 'post_xform' in pairs[prl]:
        post_xform = wp.transform(*pairs[prl]['post_xform'])
        post_xform = np.array(wp.transform_to_matrix(post_xform), float).reshape((4, 4))

        offset = [np.array(origins[_]) + np.array(roi_bounds[_][0]) * np.array(spacings[_]) for _ in range(2)]

        pre = np.identity(4)
        pre[:3, 3] = offset[0]

        post_inv = np.identity(4)
        post_inv[:3, 3] = -offset[1]

        post_xform = pre @ post_xform @ post_inv
        post_xform = wp.transform_from_matrix(wp.mat44(post_xform))
    else:
        st.warning('未配准')
        post_xform = None

    with st.spinner(_ := '重采样', show_time=True):  # noqa
        origin = -0.5 * np.array(size) * iso

        obb_xform = wp.transform_from_matrix(wp.mat44(obb_xform))
        volumes = [wp.Volume.load_from_numpy(ct_images[_], bg_value=image_bgs[_]) for _ in range(2)]

        image_obb = wp.full((*size,), wp.vec2(image_bgs[1], image_bgs[0]), wp.vec2)
        wp.launch(resample_obb, image_obb.shape, [
            image_obb, origin, wp.vec3(wp.float32(iso)), obb_xform,
            volumes[1].id, origins[1], spacings[1], post_xform if post_xform is not None else wp.transform_identity(),
            volumes[0].id, origins[0], spacings[0], post_xform is not None,
        ])
        image_obb = image_obb.numpy()
        image_a, image_b = image_obb[:, :, :, 1], image_obb[:, :, :, 0]

    cols = st.columns(3, vertical_alignment='bottom')

    for ax in (1, 2):
        stack = []
        for _ in (image_a, image_b,):
            img = fast_drr(_, ax)
            stack.append(img)

        img = np.hstack(stack)
        if ax in (1, 2):
            img = np.flipud(img)
        cols[ax].image(img, 'OBB {} 术前（左）术后（右）'.format(['轴位', '正位', '侧位'][ax]))

    with cols[0]:
        import pyvista as pv

        pl = pv.Plotter(off_screen=True, window_size=[512, 512])
        pl.enable_parallel_projection()
        pl.enable_depth_peeling()
        pl.enable_anti_aliasing('msaa')
        pl.add_camera_orientation_widget()

        colors = pv.plotting.colors.matplotlib_default_colors
        for i, _ in enumerate(raw_mesh):
            pl.add_mesh(_, color=colors[i % len(colors)])

        pl.camera_position = 'zx'
        pl.reset_camera()

        stpyvista(pl, panel_kwargs=PanelVTKKwargs(orientation_widget=True))
        pl.close()
