# 启动命令示例：uv run streamlit run b2_align_femur.py --server.port 8502 -- --config config.toml

import argparse
import tempfile
from io import BytesIO
from pathlib import Path

import itk
import numpy as np
import pyvista as pv
import streamlit as st
import tomlkit
import trimesh
import warp as wp
from minio import Minio, S3Error

from define import ct_bone_best, ct_metal, ct_seg_femur_right, ct_seg_femur_left
from kernel import diff_dmc, compute_sdf, icp

# 配置 Streamlit 页面属性
st.set_page_config('锦瑟医疗数据中心', initial_sidebar_state='collapsed', layout='wide')
st.markdown('### G-THA 术前术后配准（股骨）')

# --- 第一阶段：初始化与数据列表加载 ---
if (it := st.session_state.get('init')) is None:
    with st.spinner('初始化', show_time=True):  # noqa
        # 解析命令行参数获取配置文件路径
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', required=True)
        args = parser.parse_args()

        cfg_path = Path(args.config)
        cfg = tomlkit.loads(cfg_path.read_text('utf-8'))
        # 初始化 MinIO 客户端，用于访问云端医学影像数据
        client = Minio(**cfg['minio']['client'])

        # 遍历 MinIO 'pair' 桶，整理术前术后成对的病例数据
        pairs = {}
        for _ in client.list_objects('pair', recursive=True):
            if not _.object_name.endswith('.nii.gz'):
                continue

            # 预期路径结构：PatientID/RL/PreOrPost/image.nii.gz
            pid, rl, op, nii = _.object_name.split('/')

            if op not in ('pre', 'post'):
                continue

            prl = f'{pid}_{rl}'  # 患者 ID + 左右侧作为唯一标识
            if prl not in pairs:
                pairs[prl] = {'prl': prl}
            pairs[prl][op] = f'{pid}/{nii}'

        # 尝试加载已有的配准结果（align.toml），以便断点续作或查看结果
        for prl in pairs:
            try:
                data = client.get_object('pair', '/'.join([prl.replace('_', '/'), 'align.toml'])).data
                data = tomlkit.loads(data.decode('utf-8'))
                pairs[prl].update(data)
            except S3Error:
                pass

    st.session_state['init'] = client, pairs
    st.rerun()

# --- 第二阶段：病例选择界面 ---
elif (it := st.session_state.get('prl')) is None:
    client, pairs = st.session_state['init']

    # 统计配准进度
    dn = len([_ for _ in pairs if 'post_xform' in pairs[_]])
    ud = len(pairs) - dn

    st.progress(_ := dn / (dn + ud), text=f'{100 * _:.2f}%')
    st.metric('progress', f'{dn} / {dn + ud} 个样本', label_visibility='collapsed')

    # 自动跳转到下一个待配准的病例
    if st.button('下一个'):
        for prl in pairs:
            if 'post_xform' not in pairs[prl]:
                st.session_state['prl_input'] = prl
                break

    # 手动输入或选择病例 ID
    prl = st.text_input('PatientID_RL', key='prl_input')
    if prl in pairs:
        st.code(tomlkit.dumps(pairs[prl]), 'toml')

        if st.button('确定'):
            st.session_state['prl'] = prl
            st.rerun()

# --- 第三阶段：ROI 提取与三维重建 ---
elif (it := st.session_state.get('roi')) is None:
    client, pairs = st.session_state['init']
    prl = st.session_state['prl']
    rl = prl.split('_')[1]

    label_femur = {'R': ct_seg_femur_right, 'L': ct_seg_femur_left}[rl]

    st.code(f'股骨阈值 = {ct_bone_best}  金属阈值 = {ct_metal}')
    st.code(tomlkit.dumps(pairs[prl]), 'toml')

    roi_images, roi_bounds, sizes, spacings, origins, image_bgs = [], [], [], [], [], []
    bone_meshes, metal_meshes = [], []

    # 分别处理术前(op=0)和术后(op=1)数据
    for op, object_name in enumerate((pairs[prl]['pre'], pairs[prl]['post'])):
        with tempfile.TemporaryDirectory() as tdir:
            # 1. 下载并读取自动分割掩码
            with st.spinner(_ := '下载自动分割', show_time=True):  # noqa
                f = Path(tdir) / 'total.nii.gz'
                try:
                    client.fget_object('total', object_name, f.as_posix())
                except S3Error:
                    st.error(_ + '失败')
                    st.stop()

            with st.spinner(_ := '读取自动分割', show_time=True):  # noqa
                total = itk.imread(f.as_posix(), itk.UC)
                total = itk.array_from_image(total).transpose(2, 1, 0)

                if np.sum(total == label_femur) == 0:
                    st.error(f'自动分割不包含股骨 {label_femur}')
                    st.stop()

                # 根据分割结果计算股骨的 Bounding Box，用于提取子区域 (ROI)
                ijk = np.argwhere(total == label_femur)
                box = np.array([ijk.min(axis=0), ijk.max(axis=0) + 1])
                roi_bounds.append(box.tolist())

            # 2. 下载并读取原始 CT 图像
            with st.spinner(_ := '下载原图', show_time=True):  # noqa
                f = Path(tdir) / 'image.nii.gz'
                try:
                    client.fget_object('nii', object_name, f.as_posix())
                except S3Error:
                    st.error(_ + '失败')
                    st.stop()

            with st.spinner(_ := '读取原图', show_time=True):  # noqa
                image = itk.imread(f.as_posix(), itk.SS)
                # 获取图像元数据：尺寸、间距、原点
                size = np.array(itk.size(image), float)
                spacing = np.array(itk.spacing(image), float)
                origin = np.array(itk.origin(image), float)

                sizes.append(size)
                spacings.append(spacing)
                origins.append(origin)

                image = itk.array_from_image(image).transpose(2, 1, 0)
                image_bg = float(np.min(image))  # 获取背景值（通常是空气的 CT 值）
                image_bgs.append(image_bg)

            # 3. 提取子区域并进行三维重建
            with st.spinner(_ := '提取子区', show_time=True):  # noqa
                roi_image = image[box[0, 0]:box[1, 0], box[0, 1]:box[1, 1], box[0, 2]:box[1, 2]]
                roi_total = total[box[0, 0]:box[1, 0], box[0, 1]:box[1, 1], box[0, 2]:box[1, 2]]

                # 数据清理：将 ROI 内非股骨区域的高亮部分（如邻近骨骼）置为背景，避免干扰配准
                roi_image[np.where((roi_total != label_femur) & (roi_image > ct_bone_best))] = image_bg
                roi_images.append(roi_image)

                # 如果是术后数据，提取金属假体网格
                if op == 0:
                    metal_meshes.append(None)
                else:
                    # 使用 GPU 加速的 Marching Cubes 提取金属等值面
                    mesh = diff_dmc(wp.from_numpy(roi_image, wp.float32), np.zeros(3), spacing, ct_metal)
                    if mesh.is_empty:
                        st.error(f'术后子区不包含金属')
                        st.stop()
                    metal_meshes.append(mesh)

                # 提取股骨骨骼网格
                mesh = diff_dmc(wp.from_numpy(roi_image, wp.float32), np.zeros(3), spacing, ct_bone_best)

                if mesh.is_empty:
                    st.error(['术前', '术后'][op] + f'子区不包含股骨')
                    st.stop()

                # 后处理：仅保留最大的连通分量（通常是股骨干），并进行 Taubin 平滑处理以降低噪声
                _ = max(mesh.split(), key=lambda _: _.area)
                if len(_.faces) > 0.5 * len(mesh.faces):
                    mesh = trimesh.smoothing.filter_taubin(_)
                bone_meshes.append(mesh)

    # 将 ROI 数据存入 session_state
    st.session_state['roi'] = roi_images, roi_bounds, sizes, spacings, origins, image_bgs, bone_meshes, metal_meshes
    st.rerun()

# --- 第四阶段：交互式配准与结果确认 ---
else:
    client, pairs = st.session_state['init']
    prl = st.session_state['prl']
    pid, rl = prl.split('_')
    roi_images, roi_bounds, sizes, spacings, origins, image_bgs, bone_meshes, metal_meshes = st.session_state['roi']

    col_l, col_r = st.columns(2)

    with col_l:
        st.code(f'股骨阈值 = {ct_bone_best}  金属阈值 = {ct_metal}')
        st.code(tomlkit.dumps(pairs[prl]), 'toml')
        st.info('术后（绿）指定区域（浅绿）采样点（深红）配准到术前（浅黄）')

        zl = [round(_.bounds[1][2] - _.bounds[0][2]) for _ in bone_meshes]

        # 交互参数 1：近端截除高度
        # 理由：术后 CT 可能包含更多的股骨近端结构，或为了避开由于假体植入导致的骨折/截骨区
        d_proximal = st.number_input(
            f'近端截除 (0 ~ {zl[1]:.0f} mm)', 0, zl[1], pairs[prl].get('d_proximal', min(zl[1], 15)), step=5,
            help='截除术后比术前多余的近端特征，或截除术后到大粗隆顶端', key='d_proximal',
        )

        with st.spinner(_ := '裁剪', show_time=True):  # noqa
            post_mesh: trimesh.Trimesh = bone_meshes[1].copy()

            # 根据 Z 轴高度裁剪术后网格，只保留用于配准的稳定骨干部分
            if d_proximal > 0 or zl[0] < zl[1]:
                z_max = post_mesh.bounds[1][2] - d_proximal
                z_min = z_max - zl[0]

                z = post_mesh.vertices[:, 2]
                mask = (z_min <= z) & (z <= z_max)
                post_mesh.update_faces(np.all(mask[post_mesh.faces], axis=1))
                post_mesh.remove_unreferenced_vertices()

                # 记录被裁掉的部分以便在 3D 中对比显示
                mask = ~mask
                post_mesh_outlier = bone_meshes[1].copy()
                post_mesh_outlier.update_faces(np.all(mask[post_mesh_outlier.faces], axis=1))
                post_mesh_outlier.remove_unreferenced_vertices()
            else:
                post_mesh_outlier = None

        if post_mesh.is_empty:
            st.error('近端裁剪过多')
            st.stop()

        # 交互参数 2：采样点纵向范围
        zl.append(round(post_mesh.bounds[1][2] - post_mesh.bounds[0][2]))
        _min, _max = d_proximal, d_proximal + min(zl[0], zl[2])
        d_sample_range = st.slider(
            f'采样点范围 ({_min} ~ {_max} mm)', _min, _max, pairs[prl].get('d_sample_range', (_min, _max)), step=1,
            help='近端 ~ 远端', key='d_sample_range',
        )

        # 交互参数 3：避开金属假体的距离
        # 理由：金属假体周围存在严重的伪影（Metal Artifact），会导致骨骼表面提取不准，配准时需避开这些区域
        d_metal = st.number_input('采样点远离金属 (0 ~ 50 mm)', 0, 50, pairs[prl].get('d_metal', 5), step=5,
                                  key='d_metal')

        with st.spinner(_ := '采样', show_time=True):  # noqa
            # 使用 Warp 计算术后骨骼网格顶点到金属假体网格的距离 (SDF)
            metal = wp.Mesh(wp.array(metal_meshes[1].vertices, wp.vec3),
                            wp.array(metal_meshes[1].faces.flatten(), wp.int32))
            d = wp.zeros((len(post_mesh.vertices),), float)
            max_dist = np.linalg.norm(sizes[1] * spacings[1])
            wp.launch(compute_sdf, d.shape, [
                wp.uint64(metal.id), wp.array1d(post_mesh.vertices, wp.vec3), d, max_dist,
            ])
            d = d.numpy()

            z0 = post_mesh.bounds[1][2] - post_mesh.vertices[:, 2]

            # 权重计算：距离金属越远、在采样范围内的顶点权重越高
            _ = d - d_metal
            _ = np.clip(_, 0, max(d_metal, 1e-6))
            _ *= (d_sample_range[0] - d_proximal <= z0) & (z0 <= d_sample_range[1] - d_proximal)

            if (n := min(int(np.sum(_ > 0)), 10000)) < 100:
                st.error(f'采样点过少 {n}')
                st.stop()
            elif n < 10000:
                st.warning(f'采样点较少 {n}')

            # 随机采样 10000 个顶点用于 ICP 配准
            _ = _ / _.sum()
            _ = np.random.choice(len(post_mesh.vertices), size=n, replace=False, p=_)
            vertices = post_mesh.vertices[_]

        with st.spinner(_ := '配准', show_time=True):  # noqa
            # 初始对齐：将术后网格的顶部对齐到术前网格的顶部（Z 轴平移）
            matrix = np.identity(4)
            matrix[2, 3] = bone_meshes[0].bounds[1][2] - post_mesh.bounds[1][2]
            # 执行 ICP (Iterative Closest Point) 算法精细对齐
            matrix, _, mse, iters = icp(
                vertices, bone_meshes[0], matrix, 1e-5, 2000,
                **dict(reflection=False, scale=False),
            )

        # 核心逻辑：计算从术后“原始图像坐标系”到术前“原始图像坐标系”的全局变换矩阵 (g_matrix)
        # ICP 得到的是 ROI 局部坐标系下的变换，需要结合 ROI 在原图中的偏移量进行还原
        offset = [np.array(origins[_]) + np.array(roi_bounds[_][0]) * np.array(spacings[_]) for _ in range(2)]

        pre = np.identity(4)
        pre[:3, 3] = offset[0]

        post_inv = np.identity(4)
        post_inv[:3, 3] = -offset[1]

        g_matrix = pre @ matrix @ post_inv

        # 准备待保存的数据
        data = {
            'roi_bounds': np.array(roi_bounds).tolist(),
            'post_xform': np.array(wp.transform_from_matrix(wp.mat44(g_matrix)), dtype=float).tolist(),
            'post_points': n,
            'iterations': iters,
            'mse': mse,
            'd_proximal': d_proximal,
            'd_sample_range': d_sample_range,
            'd_metal': d_metal,
        }
        st.code(tomlkit.dumps(data), 'toml')

        # 提交结果界面
        with st.form('submit'):
            if 'excluded' in pairs[prl] and 'excluded' not in st.session_state:
                st.session_state['excluded'] = pairs[prl]['excluded']
            # 特殊情况标记：如配准质量差、存在骨折等可能影响结果的情况
            excluded = st.multiselect('是否排除', ['配准差', '小转子下骨折', '小转子下截骨', '钢板', '髓内钉'],
                                      accept_new_options=True, key='excluded')

            if st.form_submit_button('提交（覆盖）' if 'post_xform' in pairs[prl] else '提交'):
                data = {**pairs[prl], **data}
                if len(excluded):
                    data.update({'excluded': excluded})
                data = tomlkit.dumps(data).encode('utf-8')
                # 将配准参数保存回 MinIO
                client.put_object('pair', '/'.join([pid, rl, 'align.toml']), BytesIO(data), len(data))

                st.session_state.clear()
                st.rerun()

    # 右侧面板：可视化验证
    with col_r:
        with st.spinner(_ := '场景', show_time=True):  # noqa
            # 使用 PyVista 构建三维场景
            b = np.array(post_mesh.bounds)
            b[1][2] += d_proximal

            x, y, z = [b[1][_] - b[0][_] for _ in (0, 1, 2)]
            wx, wy, h = [round(_ * 5) for _ in (x, y, z)]

            pl = pv.Plotter(
                off_screen=True, border=False, window_size=[768, 768],
                line_smoothing=True, point_smoothing=True, polygon_smoothing=True,
            )
            pl.enable_parallel_projection()  # 使用正交投影便于观察几何关系
            pl.enable_depth_peeling()
            pl.enable_anti_aliasing('msaa')

            # 添加各种组件到场景
            metal_actor = pl.add_mesh(metal_meshes[1], color='lightblue')  # 金属假体（浅蓝）

            if post_mesh_outlier is not None and len(post_mesh_outlier.faces):
                pl.add_mesh(post_mesh_outlier, color='green')  # 术后被裁剪掉的部分（深绿）
            pl.add_mesh(post_mesh, color='lightgreen')  # 术后用于配准的部分（浅绿）

            pre_mesh: trimesh.Trimesh = bone_meshes[0].copy()
            pre_mesh.apply_transform(np.linalg.inv(matrix))  # 将术前网格逆变换到术后坐标系对比
            pl.add_mesh(pre_mesh, color='lightyellow')  # 术前参考网格（浅黄）
            pl.add_points(vertices, color='crimson', render_points_as_spheres=True, point_size=3)  # 实际采样点（深红）

            pl.camera_position = 'xz'
            pl.reset_camera(bounds=b.T.flatten())
            pl.camera.parallel_scale = (b[1][2] - b[0][2]) * 0.6
            pl.reset_camera_clipping_range()
            pl.render()

        # 提供 2D 截图预览和 3D 场景切换（3D 目前标记为不可用）
        if st.radio('plot', _ := ['2D 截图', '3D 场景'], horizontal=True, label_visibility='collapsed') == _[1]:
            with st.spinner(_ := '同步', show_time=True):  # noqa
                st.warning('暂不可用')
                # stpyvista(pl, panel_kwargs=PanelVTKKwargs(orientation_widget=True))
        else:
            # 渲染并拼接正侧位（AP & Lateral）视图
            sil = pl.add_silhouette(metal_actor.GetMapper().GetInput(), color='lightgray')  # noqa

            cols = []
            # 循环生成两个视角的截图：正面 (0度) 和侧面 (90或-90度)
            for i, deg in enumerate([0, 90 if rl == 'R' else -90]):
                [pl.actors[_].SetVisibility(False) for _ in pl.actors].clear()
                sil.SetVisibility(True)

                pl.window_size = [[wx, wy][i], h]

                pl.camera_position = 'xz'
                pl.reset_camera(bounds=b.T.flatten())
                pl.camera.Azimuth(deg)
                pl.camera.parallel_scale = (b[1][2] - b[0][2]) * 0.6
                pl.reset_camera_clipping_range()
                pl.render()
                a = pl.screenshot(return_img=True).copy()

                [pl.actors[_].SetVisibility(True) for _ in pl.actors].clear()

                pl.reset_camera_clipping_range()
                pl.render()
                c = pl.screenshot(return_img=True).copy()

                # 将金属假体的剪影合并到截图上，确保假体始终可见
                mask = (a != pl.background_color.int_rgb).any(axis=-1)
                c[mask] = a[mask]
                cols.append(c)

            # 横向拼接图像并显示在 Streamlit
            st.image(np.hstack(cols))

        pl.close()
