# uv run streamlit run pairs_to_dataset.py --server.port 8502 -- --config config.toml

import argparse
import tempfile
from io import BytesIO
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

ct_bone, ct_metal = 220, 2700

if (it := st.session_state.get('init')) is None:
    with st.spinner('初始化', show_time=True):  # noqa
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', required=True)
        parser.add_argument('--pairs', required=True)
        args = parser.parse_args()

        cfg_path = Path(args.config)
        cfg = tomlkit.loads(cfg_path.read_text('utf-8'))
        client = Minio(**cfg['minio']['client'])

        pairs = {}
        for pid in client.list_objects('pair', recursive=True):
            if not pid.object_name.endswith('.nii.gz'):
                continue

            pid, rl, op, nii = pid.object_name.split('/')
            prl = f'{pid}_{rl}'
            if prl not in pairs:
                pairs[prl] = {'prl': prl}
            pairs[prl][op] = f'{pid}/{nii}'

    st.session_state['init'] = client, pairs
    st.rerun()
elif (it := st.session_state.get('prl')) is None:
    client, pairs = st.session_state['init']

    if st.button('随机一个'):
        prl = choice(list(pairs.keys()))
        st.session_state['prl_input'] = prl

    prl = st.text_input('PatientID_RL', key='prl_input')
    if len(prl):
        pid, rl = prl.split('_')
        try:
            data = client.get_object('pair', f'{pid}/{rl}/align.toml').data
            data = tomlkit.loads(data.decode('utf-8'))
            pairs[prl].update(data)
        except S3Error:
            pass

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
    bone_meshes, metal_meshes = [], []

    for op, object_name in enumerate((pairs[prl]['pre'], pairs[prl]['post'])):
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
                roi_image[np.where((roi_total != label_femur) & (roi_image > ct_bone))] = image_bg
                roi_images.append(roi_image)

                # 提取术后假体等值面
                if op == 0:
                    metal_meshes.append(None)
                else:
                    mesh = diff_dmc(wp.from_numpy(roi_image, wp.float32), np.zeros(3), spacing, ct_metal)
                    if mesh.is_empty:
                        st.error(f'术后子区不包含金属')
                        st.stop()

                    metal_meshes.append(mesh)

                # 提取股骨等值面
                mesh = diff_dmc(wp.from_numpy(roi_image, wp.float32), np.zeros(3), spacing, ct_bone)
                if mesh.is_empty:
                    st.error(['术前', '术后'][op] + f'子区不包含股骨')
                    st.stop()

                mesh = max(mesh.split(), key=lambda _: _.area)
                mesh = trimesh.smoothing.filter_taubin(mesh)
                bone_meshes.append(mesh)

    st.session_state['roi'] = roi_images, sizes, spacings, image_bgs, bone_meshes, metal_meshes
    st.rerun()
else:
    client, pairs = st.session_state['init']
    prl = st.session_state['prl']
    pid, rl = prl.split('_')
    roi_images, sizes, spacings, image_bgs, bone_meshes, metal_meshes = st.session_state['roi']

    st.code(f'股骨阈值 = {ct_bone}  金属阈值 = {ct_metal}')
    st.code(tomlkit.dumps(pairs[prl]), 'toml')

    st.info('术后（绿）指定区域（浅绿）采样点（深红）配准到术前（浅黄）')

    zl = [round(_.bounds[1][0]) - round(_.bounds[0][0]) for _ in bone_meshes]
    d_proximal = st.number_input(
        f'近端截除 (0 ~ {zl[1]:.0f} mm)', 0, _ := zl[1], pairs[prl].get('d_proximal', 20), step=5, key='d_proximal',
    )

    with st.spinner(_ := '裁剪', show_time=True):  # noqa
        post_mesh: trimesh.Trimesh = bone_meshes[1].copy()

        if d_proximal > 0 or zl[0] < zl[1]:
            z_max = post_mesh.bounds[1][0] - d_proximal
            z_min = z_max - zl[0] + d_proximal

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

    zr_max, zr_min = tuple(zl[1] - round(post_mesh.bounds[_][0]) for _ in range(2))
    d_sample_range = st.slider(
        f'采样点范围 ({zr_min} ~ {zr_max} mm)', zr_min, zr_max,
        pairs[prl].get('d_sample_range', (zr_min, zr_max)), step=1,
        help='近端 ~ 远端', key='d_sample_range',
    )
    zr = [zl[1] - d_sample_range[_] for _ in reversed(range(2))]

    # 配准术后到术前，配准特征点尽量远离金属，但术后过短则不得不接近金属
    d_metal = st.number_input('采样点远离金属 (0 ~ 50 mm)', 0, 50, pairs[prl].get('d_metal', 5), step=5, key='d_metal')

    with st.spinner(_ := '采样', show_time=True):  # noqa
        metal = wp.Mesh(wp.array(metal_meshes[1].vertices, wp.vec3),
                        wp.array(metal_meshes[1].faces.flatten(), wp.int32))
        d = wp.zeros((len(post_mesh.vertices),), float)
        max_dist = np.linalg.norm(sizes[1] * spacings[1])
        wp.launch(compute_sdf, d.shape, [
            wp.uint64(metal.id), wp.array1d(post_mesh.vertices, wp.vec3), d, max_dist,
        ])
        d = d.numpy()

        _ = d - d_metal
        _ = np.clip(_, 0, max(d_metal, 1e-6))
        _ *= (zr[0] <= post_mesh.vertices[:, 0]) & (post_mesh.vertices[:, 0] <= zr[1])

        if (n := min(int(np.sum(_ > 0)), 10000)) < 100:
            st.error(f'采样点过少 {n}')
            st.stop()

        _ = _ / _.sum()

        _ = np.random.choice(len(post_mesh.vertices), size=n, replace=False, p=_)
        vertices = post_mesh.vertices[_]

    with st.spinner(_ := '配准', show_time=True):  # noqa
        matrix = np.identity(4)
        matrix[0, 3] = bone_meshes[0].bounds[1][0] - post_mesh.bounds[1][0]
        matrix, _, mse, it = icp(
            vertices, bone_meshes[0], matrix, 1e-5, 2000,
            **dict(reflection=False, scale=False),
        )

    data = {
        'post_xform': np.array(wp.transform_from_matrix(wp.mat44(matrix)), dtype=float).tolist(),
        'post_points': n,
        'iterations': it,
        'mse': mse,
        'd_proximal': d_proximal,
        'd_sample_range': d_sample_range,
        'd_metal': d_metal,
    }
    st.code(tomlkit.dumps(data), 'toml')

    if st.button('提交'):
        data = {**pairs[prl], **data}
        data = tomlkit.dumps(data).encode('utf-8')
        client.put_object('pair', '/'.join([pid, rl, 'align.toml']), BytesIO(data), len(data))

        for _ in list(st.session_state.keys()):
            del st.session_state[_]

        st.rerun()

    with st.spinner(_ := '场景', show_time=True):  # noqa
        b = np.array(post_mesh.bounds)
        b[1][0] += d_proximal

        z, y, x = [b[1][_] - b[0][_] for _ in (0, 1, 2)]
        h, wy, wx = [round(_ * 5) for _ in (z, y, x)]

        pl = pv.Plotter(
            off_screen=True, border=False, window_size=[max(wy, wx), h],
            line_smoothing=True, point_smoothing=True, polygon_smoothing=True,
        )
        pl.enable_parallel_projection()
        pl.enable_depth_peeling()
        pl.enable_anti_aliasing('msaa')

        metal_actor = pl.add_mesh(metal_meshes[1], color='lightblue')

        if post_mesh_outlier is not None and len(post_mesh_outlier.faces):
            pl.add_mesh(post_mesh_outlier, color='green')
        pl.add_mesh(post_mesh, color='lightgreen')  # noqa

        pre_mesh: trimesh.Trimesh = bone_meshes[0].copy()
        pre_mesh.apply_transform(np.linalg.inv(matrix))
        pl.add_mesh(pre_mesh, color='lightyellow')  # noqa
        pl.add_points(vertices, color='crimson', render_points_as_spheres=True, point_size=3)

        pl.camera_position = 'zx'
        pl.reset_camera(bounds=b.T.flatten())
        pl.camera.parallel_scale = (b[1][0] - b[0][0]) * 0.6
        pl.reset_camera_clipping_range()
        pl.render()

    if st.radio('plot', _ := ['2D 截图', '3D 场景'], horizontal=True, label_visibility='collapsed') == _[1]:
        with st.spinner(_ := '同步', show_time=True):  # noqa
            stpyvista(pl, panel_kwargs=PanelVTKKwargs(orientation_widget=True))
    else:
        sil = pl.add_silhouette(metal_actor.GetMapper().GetInput(), color='lightgray')  # noqa

        cols = []
        for i, deg in enumerate([0, 90 if rl == 'R' else -90]):
            [pl.actors[_].SetVisibility(False) for _ in pl.actors].clear()
            sil.SetVisibility(True)

            pl.window_size = [[wx, wy][i], h]

            pl.camera_position = 'zx'
            pl.reset_camera(bounds=b.T.flatten())
            pl.camera.Azimuth(deg)
            pl.camera.parallel_scale = (b[1][0] - b[0][0]) * 0.6
            pl.reset_camera_clipping_range()
            pl.render()
            a = pl.screenshot(return_img=True).copy()

            [pl.actors[_].SetVisibility(True) for _ in pl.actors].clear()

            pl.reset_camera_clipping_range()
            pl.render()
            c = pl.screenshot(return_img=True).copy()

            mask = (a != pl.background_color.int_rgb).any(axis=-1)
            c[mask] = a[mask]
            cols.append(c)

        st.image(np.hstack(cols))

    pl.close()
