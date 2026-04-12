# uv run streamlit run b1_prepare_context.py --server.port 8501 -- --config config.toml

import tempfile
from io import BytesIO
from pathlib import Path

import itk
import numpy as np
import streamlit as st
import tomlkit
import trimesh
import warp as wp
from PIL import Image, ImageDraw, ImageFont
from minio import S3Error

from b0_config import client_pairs
from b0_prothesis import FEMORAL, HEAD_OFFSET
from define import ct_metal, ct_seg_femur_right, ct_seg_femur_left, ct_seg_hip_right, ct_seg_hip_left
from kernel import resample_cup_head, resample_cup_head_3d

save_key = 'cup_center'

st.set_page_config('锦瑟医疗数据中心', initial_sidebar_state='collapsed', layout='wide')
st.markdown('### G-THA 全局标签录入')

# --- 第一阶段：初始化与数据列表加载 ---
if (it := st.session_state.get('init')) is None:
    with st.spinner('初始化', show_time=True):  # noqa
        client, pairs = client_pairs('context')

    st.session_state['init'] = client, pairs
    st.rerun()

# --- 第二阶段：病例选择界面 ---
elif (it := st.session_state.get('prl')) is None:
    client, pairs = st.session_state['init']

    # 统计配准进度
    dn = len([_ for _ in pairs if save_key in pairs[_]])
    ud = len(pairs) - dn

    st.progress(_ := dn / (dn + ud), text=f'{100 * _:.2f}%')
    st.metric('progress', f'{dn} / {dn + ud} 个样本', label_visibility='collapsed')

    # 自动跳转到下一个待配准的病例
    if st.button('下一个'):
        for prl in pairs:
            if save_key not in pairs[prl]:
                st.session_state['prl_input'] = prl
                break

    # 手动输入或选择病例 ID
    prl = st.text_input('PatientID_RL', key='prl_input')
    if prl in pairs:
        st.code(tomlkit.dumps(pairs[prl]), 'toml')

        if st.button('确定'):
            st.session_state['prl'] = prl
            st.rerun()

# --- 第三阶段：下载图像与分割计算 ROI ---
elif (it := st.session_state.get('roi')) is None:
    client, pairs = st.session_state['init']
    prl = st.session_state['prl']
    pid, rl = prl.split('_')

    seg_labels = [
        {'R': ct_seg_hip_right, 'L': ct_seg_hip_left}[rl],
        {'R': ct_seg_femur_right, 'L': ct_seg_femur_left}[rl],
    ]

    st.code(tomlkit.dumps(pairs[prl]), 'toml')

    # 术后
    roi_boxes = []
    for part in ('hip', 'femur'):
        origin = np.array(pairs[prl]['post'][part]['roi']['origin'])
        spacing = np.array(pairs[prl]['post'][part]['roi']['spacing'])
        size = np.array(pairs[prl]['post'][part]['roi']['size'])
        roi_boxes.append([origin, origin + spacing * size])

    object_name = pairs[prl]['post']['nii']

    with tempfile.TemporaryDirectory() as tdir:
        with st.spinner(_ := '下载原图', show_time=True):  # noqa
            f = Path(tdir) / 'image.nii.gz'
            try:
                client.fget_object('nii', object_name, f.as_posix())
            except S3Error:
                st.error(_ + '失败')
                st.stop()

            image = itk.imread(f.as_posix(), itk.SS)

            size = np.array(itk.size(image), float)
            spacing = np.array(itk.spacing(image), float)
            origin = np.array(itk.origin(image), float)

            image = itk.array_from_image(image).transpose(2, 1, 0)
            image_bg = float(np.min(image))

            volume = wp.Volume.load_from_numpy(image, bg_value=image_bg)

        with st.spinner(_ := '下载股骨', show_time=True):  # noqa
            try:
                f = Path(tdir) / 'bone.stl'
                object_name = '/'.join([pid, rl, 'post', 'femur', f.name])
                data = client.fget_object('pair', object_name, f.as_posix())
            except S3Error:
                st.error(f'post/femur/{f.name} 下载失败')
                st.stop()

            _ = trimesh.load_mesh(f.as_posix())
            _ = max(_.split(), key=lambda _: _.area)
            bone_mesh = _

    st.session_state['roi'] = image, volume, size, spacing, origin, image_bg, roi_boxes, bone_mesh
    st.rerun()

# --- 第四阶段：手动标注 ---
else:
    client, pairs = st.session_state['init']
    prl = st.session_state['prl']
    pid, rl = prl.split('_')
    image, volume, size, spacing, origin, image_bg, roi_boxes, bone_mesh = st.session_state['roi']

    with st.expander('存档'):
        st.code(tomlkit.dumps(pairs[prl]), 'toml')

    roi_boxes = np.array(roi_boxes)

    cols = st.columns([2, 3, 3, 3])

    # 股骨柄型号规格
    with cols[0]:
        spec_0, spec_1 = pairs[prl].get('femoral_spec', ['', ''])

        if spec_0 not in FEMORAL:
            spec_0 = ''

        if spec_1 not in FEMORAL[spec_0]:
            spec_1 = ''

        options = list(FEMORAL.keys())
        spec_0 = st.selectbox('股骨柄型号', options, options.index(spec_0) if spec_0 in options else 0)

        if len(spec_0):
            options = FEMORAL[spec_0]
            spec_1 = st.selectbox('股骨柄规格', options, options.index(spec_1) if spec_1 in options else 0)
        else:
            spec_1 = ''

    # 股骨头偏距
    with cols[0]:
        head_offset = pairs[prl].get('head_offset', '')
        if head_offset not in HEAD_OFFSET:
            head_offset = ''
        head_offset = st.selectbox('股骨头偏距/颈长偏移 (mm)', HEAD_OFFSET, HEAD_OFFSET.index(head_offset))

    # 三视图
    view_size = cols[0].radio('视野范围 (mm)', [100, 200, 300], horizontal=True)

    cup_outer = int(pairs[prl].get('cup_outer', 90))
    cup_outer = cols[0].number_input('髋臼杯外径', 10, 90, cup_outer, 2, key='cup_outer')

    head_outer = int(pairs[prl].get('head_outer', 10))
    head_outer = cols[0].number_input('股骨头外径', 10, 90, head_outer, 2, key='head_outer')

    liner_offset = pairs[prl].get('liner_offset', 0.0)
    liner_offset = cols[0].number_input('内衬偏心距', 0.0, 9.0, liner_offset, 0.5, format='%.1f', key='liner_offset')

    step = cols[0].radio('调节步长 (mm/deg)', _ := [5, 2.5, 1, 0.5, 0.25], horizontal=True)
    step_atom = _[0]

    # 初始估计球心
    v_max = bone_mesh.vertices[np.argmax(bone_mesh.vertices[:, 2])]
    cup_center_default = v_max.copy()
    cup_center_default += roi_boxes[1][0]
    cup_center_default[2] -= 20

    cup_center_default = np.array(cup_center_default)
    cup_center_default = np.round(cup_center_default / 0.25) * 0.25
    cup_center_default = cup_center_default.tolist()

    if 'cup_center' not in st.session_state:
        st.session_state['cup_center'] = pairs[prl].get('cup_center', cup_center_default)

    cup_axis_default = np.array([(1 if rl == 'L' else -1) * np.sin(np.deg2rad(40)), 0, -np.cos(np.deg2rad(40))])
    cup_axis_default /= np.linalg.norm(cup_axis_default)

    if 'cup_axis' not in st.session_state:
        st.session_state['cup_axis'] = pairs[prl].get('cup_axis', cup_axis_default)

    img_slots = [cols[1 + _].container() for _ in range(3)]

    for i in range(3):
        col_l, col_m, col_r = cols[1 + i].columns(3, vertical_alignment='bottom')

        axes = [0, 1, 2]
        del axes[i]

        a, d = 'RAI'[axes[0]], 'LPS'[axes[0]]
        s, w = 'RAI'[axes[1]], 'LPS'[axes[1]]

        # +y↓: SSP -> IIP
        if i not in (0, 1):
            s, w = w, s

        if col_l.button(f'↺', key=f'↺_{i}', width='stretch'):
            v = np.array(st.session_state['cup_axis'])
            x, y = v[axes[0]], (-v[axes[1]] if i in (0, 1) else v[axes[1]])
            theta = np.deg2rad(step)
            cos, sin = np.cos(theta), np.sin(theta)
            v[axes[0]], y = x * cos + y * sin, -x * sin + y * cos
            v[axes[1]] = -y if i in (0, 1) else y
            st.session_state['cup_axis'] = v / np.linalg.norm(v)

        if col_r.button(f'↻', key=f'↻_{i}', width='stretch'):
            v = np.array(st.session_state['cup_axis'])
            x, y = v[axes[0]], (-v[axes[1]] if i in (0, 1) else v[axes[1]])
            theta = np.deg2rad(-step)
            cos, sin = np.cos(theta), np.sin(theta)
            v[axes[0]], y = x * cos + y * sin, -x * sin + y * cos
            v[axes[1]] = -y if i in (0, 1) else y
            st.session_state['cup_axis'] = v / np.linalg.norm(v)

        if col_l.button(f'{a}', key=f'{a}_{i}', width='stretch'):
            st.session_state['cup_center'][axes[0]] += step * (1 if a in 'LPS' else -1)
        if col_r.button(f'{d}', key=f'{d}_{i}', width='stretch'):
            st.session_state['cup_center'][axes[0]] += step * (1 if d in 'LPS' else -1)

        if col_m.button(f'{w}', key=f'{w}_{i}', width='stretch'):
            st.session_state['cup_center'][axes[1]] += step * (1 if w in 'LPS' else -1)
        if col_m.button(f'{s}', key=f'{s}_{i}', width='stretch'):
            st.session_state['cup_center'][axes[1]] += step * (1 if s in 'LPS' else -1)

    cup_center = np.array(st.session_state['cup_center'], float)
    cup_axis = np.array(st.session_state['cup_axis'], float)
    head_center = cup_center + liner_offset * cup_axis

    roi_spacing = 0.2
    roi_size = np.ceil((np.ones(3) * view_size) / roi_spacing).astype(int)


    def get_coverage(cc, ca):
        roi = wp.zeros((*roi_size,), dtype=wp.float32)
        metal = wp.zeros((*roi_size,), dtype=wp.float32)
        o = cc - 0.5 * roi_size * roi_spacing
        hc = cc + liner_offset * ca
        wp.launch(resample_cup_head_3d, roi.shape, [
            volume.id, wp.vec3(origin), wp.vec3(spacing), ct_metal,
            roi, wp.vec3(o), roi_spacing, metal,
            wp.vec3(cc), wp.vec3(ca), wp.vec3(hc), head_outer / 2.0, cup_outer / 2.0 - 1.0,  # 修正髋臼杯外层
        ])
        roi, metal = roi.numpy(), metal.numpy()
        if (_ := np.sum(roi)) > 0:
            return float(np.sum(metal) / _)
        return 0.0


    coverage_max = get_coverage(cup_center, cup_axis)
    empty = cols[0].empty()
    empty.info(f'金属占比（3D）{coverage_max * 1e2:.3f}%')

    if cols[0].button('自动微调', width='stretch'):
        # 斐波那契球面均匀采样方向
        samples = 26
        phi = np.pi * (3. - np.sqrt(5.))
        i = np.arange(samples)
        y = 1 - (i / float(samples - 1)) * 2
        radius = np.sqrt(1 - y * y)
        theta = phi * i
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        sphere_directions = np.column_stack((x, y, z))

        better = True
        while better:
            better = False
            for offset in sphere_directions:
                offset = np.array(offset, float)
                offset /= np.linalg.norm(offset)
                cup_center_test = cup_center + offset * step

                coverage = get_coverage(cup_center_test, cup_axis)

                if coverage_max < coverage:
                    coverage_max = coverage
                    cup_center = cup_center_test
                    head_center = cup_center + liner_offset * cup_axis
                    st.session_state['cup_center'] = cup_center.tolist()

                    better = True
                    empty.success(f'金属占比 {coverage_max * 1e2:.3f}%')

        better = True
        while better:
            better = False
            for axis in [
                [-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1],
            ]:
                k = np.array(axis, float)
                v = cup_axis.copy()
                theta = np.deg2rad(step)

                # 罗德里格旋转公式 (Rodrigues' rotation formula)
                v = v * np.cos(theta) + np.cross(k, v) * np.sin(theta) + k * np.dot(k, v) * (1 - np.cos(theta))
                v /= np.linalg.norm(v)

                coverage = get_coverage(cup_center, v)

                if coverage_max < coverage:
                    coverage_max = coverage
                    cup_axis = v.copy()
                    head_center = cup_center + liner_offset * cup_axis
                    st.session_state['cup_axis'] = cup_axis.tolist()

                    better = True
                    empty.success(f'金属占比 {coverage_max * 1e2:.3f}%')

        empty.info(f'金属占比 {coverage_max * 1e2:.3f}%')

    ort = [
        ['S', 'I', 'A', 'P'],
        ['S', 'I', 'R', 'L'],
        ['A', 'P', 'R', 'L'],
    ]

    stack = []
    for i in range(3):
        axes = np.eye(3, dtype=float).tolist()
        del axes[i]
        axes = np.array(axes)

        shape = [*roi_size]
        del shape[i]

        roi_slice = wp.zeros(shape, dtype=wp.vec3ub)
        roi_origin = cup_center - 0.5 * roi_size * roi_spacing * axes[0] - 0.5 * roi_size * roi_spacing * axes[1]

        wp.launch(resample_cup_head, roi_slice.shape, [
            volume.id, wp.vec3(origin), wp.vec3(spacing), -100.0, 1000.0,
            roi_slice, wp.vec3(roi_origin), roi_spacing, wp.vec3(axes[0]), wp.vec3(axes[1]),
            wp.vec3(cup_center), wp.vec3(cup_axis), wp.vec3(head_center), head_outer / 2.0, cup_outer / 2.0,
        ])

        roi_slice = roi_slice.numpy().transpose(1, 0, 2)

        # +y↓: SSP -> IIP
        if i in (0, 1):
            roi_slice = np.flipud(roi_slice)

        img = Image.fromarray(roi_slice)

        # orientation
        draw = ImageDraw.Draw(img)
        cw, ch = [_ / 2 for _ in img.size]

        fill = (255, 255, 255)
        try:
            font = ImageFont.truetype('timesbd.ttf', 20)
        except (OSError, Exception):
            font = ImageFont.load_default()

        draw.text((cw, 20), ort[i][0], fill, font, anchor='mm')
        draw.text((cw, ch * 2 - 20), ort[i][1], fill, font, anchor='mm')
        draw.text((20, ch), ort[i][2], fill, font, anchor='mm')
        draw.text((cw * 2 - 20, ch), ort[i][3], fill, font, anchor='mm')

        stack.append(np.array(img))

    images = []
    captions = ['矢状面 (Sagittal)', '冠状面 (Coronal)', '横断面 (Axial)']
    for _ in range(3):
        img_slots[_].image(stack[_], captions[_], width='stretch')

        _ = Image.fromarray(stack[_])
        buf = BytesIO()
        _.save(buf, format='PNG')
        images.append(buf.getvalue())

    # 提交结果
    with st.form('submit'):
        data = {
            'cup_outer': cup_outer,
            'head_outer': head_outer,
            'liner_offset': liner_offset,
            'cup_center': cup_center.tolist(),
            'cup_axis': cup_axis.tolist(),
            'femoral_spec': [spec_0, spec_1],
            'head_offset': head_offset,
            'coverage': coverage_max,
        }
        st.code(tomlkit.dumps(data), 'toml')

        if 'excluded' in pairs[prl] and 'excluded' not in st.session_state:
            st.session_state['excluded'] = pairs[prl]['excluded']

        excluded = st.multiselect('是否排除', ['半髋置换', ], accept_new_options=True, key='excluded')

        if st.form_submit_button('提交（覆盖）' if save_key in pairs[prl] else '提交'):
            data = {**pairs[prl], **data}
            if len(excluded):
                data.update({'excluded': excluded})

            # 更新内存中的总表
            pairs[prl].update(data)

            data = tomlkit.dumps(data).encode('utf-8')

            client.put_object('pair', '/'.join([pid, rl, 'context.toml']), BytesIO(data), len(data))

            # 保留 init 状态，只清空当前会话的其他状态
            st.session_state.clear()
            st.session_state['init'] = client, pairs
            st.rerun()
