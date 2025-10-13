# CUDA driver & Toolkit & Microsoft Visual C++
# pip install itk pyglet trimesh rtree scikit-learn
# pip install torch --index-url https://download.pytorch.org/whl/cu129
# pip install wheel diso
# pip install -U --pre warp-lang --extra-index-url=https://pypi.nvidia.com/
# pip uninstall newton newton-physics -y
# pip install -U git+https://github.com/newton-physics/newton.git@1bad84b6113c73b620a233198a3c3b77ff25719e

import argparse
import json
import locale
from pathlib import Path

import numpy as np
import pyvista as pv
import tomlkit
import warp as wp
from vtkmodules.vtkCommonMath import vtkMatrix4x4
from vtkmodules.vtkCommonTransforms import vtkTransform

from kernel import centerline_sdf, diff_dmc, tri_poly

locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')

inactive_centerline = {}

active_centerline = {}
active_radius = {}

part_transform = {}
part_matrix = {}


def main(cfg_path: str):
    cfg_path = Path(cfg_path)

    _ = cfg_path.read_text('utf-8')
    cfg = tomlkit.loads(_)

    pl = pv.Plotter(title='血管吻合术模拟', shape=(1, 2), border_width=0)
    pl.enable_parallel_projection()
    pl.add_camera_orientation_widget()
    pl.enable_hidden_line_removal()
    pl.camera_position = 'xz'
    camera = pl.camera

    # 读取SlicerVMTK模块导出的血管中心线数据
    for part in ('髂内动脉', '肾动脉'):
        cfg[part] = cfg.get(part, {})

        inactive_centerline[part] = []

        centerline = Path(cfg[part]['中心线'])

        bounds = []
        for f in centerline.rglob('*.json'):
            segment = json.loads(f.read_text(encoding='utf-8'))
            points = segment['markups'][0]['controlPoints']
            radius_raw = [0] * len(points)
            for _ in segment['markups'][0]['measurements']:
                if _['name'] == 'Radius':
                    radius_raw = _['controlPointValues']

            assert len(points) == len(radius_raw)

            points = [_['position'] for _ in points]

            r_min = radius_raw[0]
            radius = []
            for p, r in zip(points, radius_raw):
                if r_min > r:
                    r_min = r
                radius.append(r_min)
                bounds.append(p)

            # 提取吻合分支和段落
            if f.name == cfg[part]['吻合分支']:
                begin, end = cfg[part]['吻合段落']
                assert 0 <= begin < end <= 1.0

                begin, end = int(np.floor(len(points) * begin)), int(np.ceil(len(points) * end))

                active_centerline[part] = points[begin:end]
                active_radius[part] = radius[begin:end]

            if len(points):
                inactive_centerline[part].append(points)

    # 肾动脉始端连接髂内动脉末端，中心线方向对齐
    c0 = (
        [active_centerline['髂内动脉'][-1][_] for _ in range(3)],
        [active_centerline['肾动脉'][0][_] for _ in range(3)],
    )
    c1 = (
        [active_centerline['髂内动脉'][-1][_] - active_centerline['髂内动脉'][-2][_] for _ in range(3)],
        [active_centerline['肾动脉'][1][_] - active_centerline['肾动脉'][0][_] for _ in range(3)],
    )

    m = vtkMatrix4x4()
    m.SetData((
        1.0, 0.0, 0.0, -c0[1][0],
        0.0, 1.0, 0.0, -c0[1][1],
        0.0, 0.0, 1.0, -c0[1][2],
        0.0, 0.0, 0.0, 1.0,
    ))

    _ = np.identity(4)
    _[:3, :3] = np.array(wp.quat_to_matrix(wp.quat_between_vectors(wp.vec3(c1[1]), wp.vec3(c1[0])))).reshape((3, 3))

    m_ = vtkMatrix4x4()
    m_.SetData(_.flatten())

    m.Multiply4x4(m_, m, m)

    m_ = vtkMatrix4x4()
    m_.SetData((
        1.0, 0.0, 0.0, c0[0][0],
        0.0, 1.0, 0.0, c0[0][1],
        0.0, 0.0, 1.0, c0[0][2],
        0.0, 0.0, 0.0, 1.0,
    ))

    m.Multiply4x4(m_, m, m)

    part = '肾动脉'
    part_matrix[part] = m

    cfg[part]['变换'] = part_matrix[part].data

    part_transform[part] = vtkTransform()
    part_transform[part].SetMatrix(part_matrix[part])

    # 配准肾动脉到髂内动脉
    for part in ('髂内动脉', '肾动脉'):
        _ = pv.vtk_points(active_centerline[part])
        part_transform.get(part, vtkTransform()).TransformPoints(_, _)
        active_centerline[part] = np.array(_.data).reshape(2, -1, 3)[1]

        c = {'髂内动脉': 'crimson', '肾动脉': 'pink'}[part]
        _ = pl.add_lines(np.array(active_centerline[part]), c, 10, connected=True)

        for i in range(len(inactive_centerline[part])):
            _ = pv.vtk_points(inactive_centerline[part][i])
            part_transform.get(part, vtkTransform()).TransformPoints(_, _)
            inactive_centerline[part][i] = np.array(_.data).reshape(2, -1, 3)[1]

            c = {'髂内动脉': 'grey', '肾动脉': 'lightgrey'}[part]
            _ = pl.add_lines(np.array(inactive_centerline[part][i]), c, 5, connected=True)

    pl.subplot(0, 1)
    pl.camera = camera

    # 根据半径差异计算血管侧面切开长度，作为半径过渡段
    cfg['髂内动脉']['吻合半径'] = ra = active_radius['髂内动脉'][-1]
    cfg['肾动脉']['吻合半径'] = rb = active_radius['肾动脉'][0]
    cfg['肾动脉']['圆周长差'] = l_max = 2 * np.pi * (ra - rb)

    i, l_sum = 0, 0
    while l_sum < l_max:
        l_sum += np.linalg.norm(np.array(active_centerline['肾动脉'][i]) - np.array(active_centerline['肾动脉'][i + 1]))
        _ = min(max(l_sum / l_max, 0), 1)
        active_radius['肾动脉'][i] = ra * (1 - _) + rb * _
        i += 1

    cfg['肾动脉']['侧切长度'] = l_sum

    # 重采样血管中心线距离场，重建血管内壁面网格
    points = np.vstack([active_centerline['髂内动脉'], active_centerline['肾动脉']])
    b = [np.min(points, 0), np.max(points, 0)]
    b[0] -= 2 * active_radius['髂内动脉'][0]
    b[1] += 2 * active_radius['髂内动脉'][0]

    iso_spacing = 0.5
    size = np.ceil((b[1] - b[0]) / iso_spacing)
    sdf = wp.zeros((*size,))

    centers = wp.array(points, dtype=wp.vec3)
    radius = wp.array(np.hstack([active_radius['髂内动脉'], active_radius['肾动脉']]), dtype=wp.float32)

    wp.launch(centerline_sdf, sdf.shape, [
        sdf, wp.vec3(b[0]), iso_spacing, centers, radius, len(points),
    ])

    mesh = diff_dmc(sdf, b[0], iso_spacing, -0.1)

    pl.add_mesh(tri_poly(mesh), 'gold')

    cfg_path.write_text(tomlkit.dumps(cfg), 'utf-8')

    pl.reset_camera()
    pl.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()

    try:
        main(args.config)
    except KeyboardInterrupt:
        pass
