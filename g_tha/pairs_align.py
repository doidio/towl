import argparse
import locale
import tempfile
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from io import BytesIO
from pathlib import Path

import itk
import numpy as np
import pydicom
import tomlkit
import trimesh
import warp as wp
from minio import Minio, S3Error
from tqdm import tqdm

from kernel import diff_dmc, compute_sdf

locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')

side = ('右髋', '左髋')
side_femur = (76, 75)
hu_bone = 226
hu_metal = 1600


def get_pairs(cfg_path: str, images_path: str, pairs_done: dict):
    cfg_path = Path(cfg_path)
    cfg = tomlkit.loads(cfg_path.read_text('utf-8'))
    client = Minio(**cfg['minio']['client'])

    images_path = Path(images_path)
    images = tomlkit.loads(images_path.read_text('utf-8'))

    # patient/right_left
    table = {}
    for object_name, valid in images['images'].items():
        if not valid[2]:
            continue

        patient, series = object_name.split('/')

        if patient not in table:
            table[patient] = [[], []]

        for rl in range(2):
            if valid[rl] != '无效':
                table[patient][rl].append((object_name, valid[rl]))

    for patient in table:
        for rl in range(2):
            pre, post = [], []
            for object_name, pp in table[patient][rl]:
                dcm = object_name.removesuffix('.nii.gz') + '.dcm'
                dcm = client.get_object('dcm', dcm).data
                dcm = pydicom.dcmread(BytesIO(dcm))

                dt = datetime(
                    year=int(dcm.StudyDate[0:4]),
                    month=int(dcm.StudyDate[4:6]),
                    day=int(dcm.StudyDate[6:8]),
                    hour=int(dcm.StudyTime[0:2]),
                    minute=int(dcm.StudyTime[2:4]),
                    second=int(dcm.StudyTime[4:6]),
                )

                if pp == '术前':
                    pre.append((object_name, pp, dcm.StudyTime, dcm.StudyDate, dt))
                elif pp == '术后':
                    post.append((object_name, pp, dcm.StudyTime, dcm.StudyDate, dt))

            pre = sorted(pre, key=lambda _: _[-1])
            post = sorted(post, key=lambda _: _[-1])

            if len(pre) > 0 and len(post) > 0:
                if pre[-1][-1] < post[0][-1]:  # 同人同侧末次术前早于首次术后
                    key = '_'.join([patient, ['R', 'L'][rl]])
                    yield key, key in pairs_done, patient, rl, pre[-1][0], post[0][0]
                else:
                    _ = [f'{_[-2]}.{_[-3]}' for _ in pre], [f'{_[-2]}.{_[-3]}' for _ in post]
                    warnings.warn(f'{patient} {side[rl]} 术前 {_[0]} 术后 {_[1]}', stacklevel=3)


def main(cfg_path: str, done: bool, patient: str, rl: int, pre_object_name: str, post_object_name: str):
    if done:
        return None

    cfg_path = Path(cfg_path)
    cfg = tomlkit.loads(cfg_path.read_text('utf-8'))
    client = Minio(**cfg['minio']['client'])

    roi_images, roi_meshes, metal_meshes, sizes, spacings, image_bgs = [], [], [], [], [], []

    for op, object_name in enumerate((pre_object_name, post_object_name)):
        trace = '_'.join([patient, side[rl], ['术前', '术后'][op]])

        # 加载分割图
        with tempfile.TemporaryDirectory() as tdir:
            f = Path(tdir) / 'f.nii.gz'
            try:
                client.fget_object(_ := 'total', object_name, f.as_posix())
            except S3Error:
                return None

            try:
                total = itk.imread(f.as_posix(), itk.UC)
            except RuntimeError:
                return None

        total = itk.array_from_image(total)

        # 检查分割图子区
        ijk = np.argwhere(total == side_femur[rl])
        b = np.array([ijk.min(axis=0), ijk.max(axis=0)])

        if not np.all(b[0] < b[1]):
            warnings.warn(f'femur-empty {trace} {b}', stacklevel=3)
            return None

        # 加载原图
        with tempfile.TemporaryDirectory() as tdir:
            f = Path(tdir) / 'f.nii.gz'
            try:
                client.fget_object('nii', object_name, f.as_posix())
            except S3Error:
                warnings.warn(f'nii-fget {trace}', stacklevel=3)
                return None

            try:
                image = itk.imread(f.as_posix(), itk.SS)
            except RuntimeError as e:
                warnings.warn(f'nii-read {trace} {e}', stacklevel=3)
                return None

        size = np.array([float(_) for _ in reversed(itk.size(image))])
        spacing = np.array([float(_) for _ in reversed(itk.spacing(image))])

        sizes.append(size)
        spacings.append(spacing)

        image = itk.array_from_image(image)
        image_bg = float(np.min(image))
        image_bgs.append(image_bg)

        # 提取子区
        roi_image = image[b[0, 0]:b[1, 0] + 1, b[0, 1]:b[1, 1] + 1, b[0, 2]:b[1, 2] + 1]
        roi_total = total[b[0, 0]:b[1, 0] + 1, b[0, 1]:b[1, 1] + 1, b[0, 2]:b[1, 2] + 1]

        # 抹除子区中的非股骨高亮体素
        roi_image[np.where((roi_total != side_femur[rl]) & (roi_image > hu_bone))] = image_bg
        roi_images.append(roi_image)

        # 提取术后假体等值面
        if op == 0:
            metal_meshes.append(None)
        else:
            mesh = diff_dmc(wp.from_numpy(roi_image, wp.float32), np.zeros(3), spacing, hu_metal)
            if mesh.is_empty:
                warnings.warn(f'metal-empty {trace}', stacklevel=3)
                return None

            mesh = max(mesh.split(), key=lambda c: c.area)
            mesh = trimesh.smoothing.filter_taubin(mesh)
            metal_meshes.append(mesh)

        # 提取骨等值面
        mesh = diff_dmc(wp.from_numpy(roi_image, wp.float32), np.zeros(3), spacing, hu_bone)
        if mesh.is_empty:
            warnings.warn(f'roi-empty {trace}', stacklevel=3)
            return None

        mesh = max(mesh.split(), key=lambda c: c.area)
        mesh = trimesh.smoothing.filter_taubin(mesh)
        roi_meshes.append(mesh)

        # itk.imwrite(itk.image_from_array(image), f'{trace}_image.nii.gz')
        # itk.imwrite(itk.image_from_array(total), f'{trace}_total.nii.gz')
        # mesh.export(f'{trace}.stl')

    # 计算术后网格到假体距离
    metal = wp.Mesh(wp.array(metal_meshes[1].vertices, wp.vec3), wp.array(metal_meshes[1].faces.flatten(), wp.int32))
    sdf = wp.zeros((len(roi_meshes[1].vertices),), float)
    max_dist = np.linalg.norm(sizes[1] * spacings[1])
    wp.launch(compute_sdf, sdf.shape, [
        wp.uint64(metal.id), wp.array1d(roi_meshes[1].vertices, wp.vec3), sdf, max_dist,
    ])
    sdf = sdf.numpy()
    m = sdf.min(), sdf.max()

    # 到假体距离加权采样
    w = (sdf - m[0]) / (m[1] - m[0])
    w = np.exp(-15 * w)
    p = w / w.sum()

    _ = np.random.choice(len(roi_meshes[1].vertices), size=10000, replace=False, p=p)
    vertices = roi_meshes[1].vertices[_]

    # 配准术后远离假体的点云到术前网格
    matrix = np.identity(4)
    matrix[0, 3] = np.max(roi_meshes[0].vertices[:, 0]) - np.max(roi_meshes[1].vertices[:, 0])
    matrix, _, mse = trimesh.registration.icp(
        vertices, roi_meshes[0], matrix, 1e-5, 200,
        **dict(reflection=False, scale=False),
    )

    roi_meshes[1].apply_transform(matrix)
    vertices = trimesh.transform_points(vertices, matrix)

    import pyvista as pv
    pl = pv.Plotter(title=f'{patient} {rl}')
    pl.add_camera_orientation_widget()
    pl.add_text(f'MSE {mse:.3f} mm', font_size=9)
    pl.add_mesh(roi_meshes[0], color='yellow')
    pl.add_mesh(roi_meshes[1], color='green')
    pl.add_points(vertices, color='crimson', render_points_as_spheres=True, point_size=5)
    pl.camera_position = 'zx'
    pl.show()

    return mse, np.array(wp.transform_from_matrix(wp.mat44(matrix)), dtype=float).tolist()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--images', required=True)
    parser.add_argument('--pairs', required=True)
    parser.add_argument('--max_workers', type=int, default=3)
    args = parser.parse_args()

    pairs_path = Path(args.pairs)
    if pairs_path.exists():
        pairs = tomlkit.loads(pairs_path.read_text('utf-8'))
    else:
        pairs = {}

    if 'pairs' not in pairs:
        pairs['pairs'] = {}

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(main, args.config, *_[1:]): _
                   for _ in get_pairs(args.config, args.images, pairs['pairs'])}

        try:
            for fu in tqdm(as_completed(futures), total=len(futures)):
                try:
                    if (result := fu.result()) is not None:
                        _ = list(futures[fu])
                        pairs['format'] = {'patient-id_rl': ['配准误差', '配准变换', '术前图像', '术后图像']}
                        pairs['pairs'][_[0]] = [*result, _[4], _[5]]
                        pairs_path.write_text(tomlkit.dumps(pairs), 'utf-8')
                except Exception as _:
                    warnings.warn(f'{_} {futures[fu]}', stacklevel=2)

        except KeyboardInterrupt:
            print('Keyboard interrupted terminating...')
            executor.shutdown(wait=False)
            for future in futures:
                future.cancel()
