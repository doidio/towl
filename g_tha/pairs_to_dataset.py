# uv run streamlit run pairs_to_dataset.py --server.port 8502 -- --config config.toml --pairs pairs.toml

import argparse
import tempfile
from pathlib import Path
from random import choice

import itk
import numpy as np
import pyvista as pv
import streamlit as st
import tomlkit
import trimesh
import warp as wp
from minio import Minio, S3Error
from stpyvista import stpyvista
from stpyvista.panel_backend import PanelVTKKwargs

from kernel import diff_dmc, compute_sdf, icp

st.set_page_config('锦瑟医疗数据中心', initial_sidebar_state='collapsed', layout='centered')
st.markdown('### 全髋关节置换术前术后配准')

hu_bone, hu_metal = 220, 2700

if (it := st.session_state.get('init')) is None:
    with st.spinner('初始化', show_time=True):  # noqa
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', required=True)
        parser.add_argument('--pairs', required=True)
        args = parser.parse_args()

        cfg_path = Path(args.config)
        cfg = tomlkit.loads(cfg_path.read_text('utf-8'))
        client = Minio(**cfg['minio']['client'])

        pairs_path = Path(args.pairs)
        pairs: dict = tomlkit.loads(pairs_path.read_text('utf-8'))

    st.session_state['init'] = client, pairs
    st.rerun()
elif (it := st.session_state.get('prl')) is None:
    client, pairs = st.session_state['init']

    if st.button('随机一个'):
        p = choice([_ for _ in pairs.keys() if len(pairs[_].keys()) > 0])
        rl = choice(list(pairs[p]))
        st.session_state['prl_input'] = f'{p}_{rl}'

    prl = st.text_input('PatientID_RL', key='prl_input')
    if len(prl):
        prl = prl.split('_')
        if len(prl) == 2 and pairs.get(prl[0], {}).get(prl[1]):
            p, rl = prl

            for i, _ in enumerate(('误差', '配准变换', '术前', '术后')):
                st.code(f'{_} = {pairs[p][rl][i]}')

            if st.button('确定'):
                st.session_state['prl'] = p, rl
                st.rerun()

elif (it := st.session_state.get('roi')) is None:
    client, pairs = st.session_state['init']
    p, rl = st.session_state['prl']

    st.code(f'股骨阈值 = {hu_bone}, 金属阈值 = {hu_metal}')
    st.code(f'PatientID_RL = {p}_{rl}')

    label_femur = {'R': 76, 'L': 75}[rl]
    roi_images, sizes, spacings, image_bgs = [], [], [], []
    bone_meshes, metal_meshes = [], []

    for op, object_name in enumerate((pairs[p][rl][2], pairs[p][rl][3])):
        opname = ['术前', '术后'][op]
        st.code(f'{opname} = {object_name}')

        with tempfile.TemporaryDirectory() as tdir:
            with st.spinner(_ := '下载自动分割', show_time=True):  # noqa
                f = Path(tdir) / 'total.nii.gz'
                try:
                    client.fget_object('total', object_name, f.as_posix())
                except S3Error:
                    st.error(_ + '失败')
                    st.stop()

            with st.spinner(_ := '读取自动分割', show_time=True):  # noqa
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
                roi_image[np.where((roi_total != label_femur) & (roi_image > hu_bone))] = image_bg
                roi_images.append(roi_image)

                # 提取术后假体等值面
                if op == 0:
                    metal_meshes.append(None)
                else:
                    mesh = diff_dmc(wp.from_numpy(roi_image, wp.float32), np.zeros(3), spacing, hu_metal)
                    if mesh.is_empty:
                        st.error(f'术后子区不包含金属')
                        st.stop()

                    metal_meshes.append(mesh)

                # 提取股骨等值面
                mesh = diff_dmc(wp.from_numpy(roi_image, wp.float32), np.zeros(3), spacing, hu_bone)
                if mesh.is_empty:
                    st.error(opname + f'子区不包含股骨')
                    st.stop()

                mesh = max(mesh.split(), key=lambda c: c.area)
                mesh = trimesh.smoothing.filter_taubin(mesh)
                bone_meshes.append(mesh)

    st.session_state['roi'] = roi_images, sizes, spacings, image_bgs, bone_meshes, metal_meshes
    st.rerun()
else:
    client, pairs = st.session_state['init']
    p, rl = st.session_state['prl']
    roi_images, sizes, spacings, image_bgs, bone_meshes, metal_meshes = st.session_state['roi']

    st.code(f'股骨阈值 = {hu_bone}  金属阈值 = {hu_metal}')
    st.code(f'PatientID_RL = {p}_{rl}')
    st.code(f'术前 = {pairs[p][rl][2]}')
    st.code(f'术后 = {pairs[p][rl][3]}')

    zl = [_.bounds[1][0] - _.bounds[0][0] for _ in bone_meshes]
    zd = (zl[1] - zl[0])
    z0 = st.number_input(f'近端截除 (0 ~ {int(zl[1]):.0f} mm)', 20, _ := int(zl[1]), step=5)

    with st.spinner(_ := '配准', show_time=True):  # noqa
        post_mesh: trimesh.Trimesh = bone_meshes[1].copy()
        if z0 > 0 or zl[0] < zl[1]:
            z_max = np.max(bone_meshes[1].vertices[:, 0]) - z0
            z_min = z_max - zl[0]

            z = post_mesh.vertices[:, 0]
            mask = (z_min <= z) & (z <= z_max)
            post_mesh.update_faces(np.all(mask[post_mesh.faces], axis=1))
            post_mesh.remove_unreferenced_vertices()

            mask = ~mask
            post_mesh_outlier = bone_meshes[1].copy()
            post_mesh_outlier.update_faces(np.all(mask[post_mesh_outlier.faces], axis=1))
            post_mesh_outlier.remove_unreferenced_vertices()
        else:
            post_mesh_outlier = None

        init_matrix = np.identity(4)
        init_matrix[0, 3] = np.max(bone_meshes[0].vertices[:, 0]) - np.max(post_mesh.vertices[:, 0])

        # 计算术后网格到假体的加权距离
        metal = wp.Mesh(wp.array(metal_meshes[1].vertices, wp.vec3),
                        wp.array(metal_meshes[1].faces.flatten(), wp.int32))
        d = wp.zeros((len(post_mesh.vertices),), float)
        max_dist = np.linalg.norm(sizes[1] * spacings[1])
        wp.launch(compute_sdf, d.shape, [
            wp.uint64(metal.id), wp.array1d(post_mesh.vertices, wp.vec3), d, max_dist,
        ])
        d = d.numpy()

        # 配准术后到术前，配准特征点尽量远离金属，但术后过短则不得不接近金属
        far = st.number_input('远离金属 (mm)', 0, 50, 5, step=5)

        _ = d - far
        _ = np.clip(_, 0, max(far, 1e-6))

        if (n := int(np.sum(_ > 0))) < 100:
            st.error(f'采样点过少 {n}')
            st.stop()

        _ = _ / _.sum()

        _ = np.random.choice(len(post_mesh.vertices), size=min(n, 10000), replace=False, p=_)
        vertices = post_mesh.vertices[_]

        matrix, _, mse, it = icp(
            vertices, bone_meshes[0], init_matrix, 1e-5, 2000,
            **dict(reflection=False, scale=False),
        )
        st.success(f'采样点 {min(n, 10000)} 迭代 {it} 误差 {mse:.3f} mm')

    z, y = [post_mesh.bounds[1][_] - post_mesh.bounds[0][_] for _ in (0, 1)]
    pl = pv.Plotter(off_screen=True, shape=(1, 2), border=False, window_size=[round(2000 * y / z), 1000])
    pl.enable_parallel_projection()
    pl.enable_depth_peeling()
    camera = pl.camera

    if post_mesh_outlier is not None:
        pl.add_mesh(post_mesh_outlier, color='green')
    pl.add_mesh(post_mesh, color='lightgreen')

    pre_mesh: trimesh.Trimesh = bone_meshes[0].copy()
    pre_mesh.apply_transform(np.linalg.inv(matrix))
    pl.add_mesh(pre_mesh, color='lightyellow')
    pl.add_points(vertices, color='crimson', render_points_as_spheres=True, point_size=2)

    pl.subplot(0, 1)
    pl.camera = camera

    pl.add_mesh(metal_meshes[1], color='lightblue')

    pl.camera_position = 'zx'
    pl.reset_camera(bounds=post_mesh.bounds.T.flatten())
    pl.camera.parallel_scale = zl[1] * 0.6
    pl.render()

    if st.radio('plot', _ := ['2D 截图', '3D 场景'], horizontal=True, label_visibility='collapsed') == _[1]:
        with st.spinner(_ := '同步', show_time=True):  # noqa
            stpyvista(pl, panel_kwargs=PanelVTKKwargs(orientation_widget=True))
    else:
        for i, _ in enumerate((0, 90 if rl == 'R' else -90)):
            pl.camera.Azimuth(_)
            pl.render()
            st.image(pl.screenshot(return_img=True), caption=['后视', '内侧视'][i])

    pl.close()
