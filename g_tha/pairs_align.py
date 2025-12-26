import argparse
import locale
import tempfile
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import itk
import numpy as np
import tomlkit
import trimesh
import warp as wp
from minio import Minio, S3Error
from tqdm import tqdm

from kernel import diff_dmc, compute_sdf, icp

locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')

wp.config.quiet = True

hu_bone = 250  # 200, 250, 350, 550 骨阈值轮测选优
hu_metal = 2700  # 金属假体阈值
far_from_metal = (30, 15, 10, 7.5, 5.0,)  # 远离金属的距离


class UserCanceled(RuntimeError):
    pass


def main(cfg_path: str, redo_mse: float,
         patient: str, rl: str, save_mse: float, _: list, pre_object_name: str, post_object_name: str):
    if 0 < redo_mse and 0 <= save_mse < redo_mse:
        return None

    print(patient, rl, save_mse)

    cfg_path = Path(cfg_path)
    cfg = tomlkit.loads(cfg_path.read_text('utf-8'))
    client = Minio(**cfg['minio']['client'])

    label_femur = {'R': 76, 'L': 75}[rl]
    roi_images, roi_meshes, metal_meshes, sizes, spacings, image_bgs = [], [], [], [], [], []

    for op, object_name in enumerate((pre_object_name, post_object_name)):
        trace = ' '.join([patient, rl, ['术前', '术后'][op], object_name])

        # 加载分割图
        with tempfile.TemporaryDirectory() as tdir:
            f = Path(tdir) / 'f.nii.gz'
            try:
                client.fget_object(_ := 'total', object_name, f.as_posix())
            except S3Error:
                warnings.warn(f'total-fget {trace}', stacklevel=2)
                return None

            try:
                total = itk.imread(f.as_posix(), itk.UC)
            except RuntimeError:
                warnings.warn(f'itk-read {trace}', stacklevel=2)
                return None

        total = itk.array_from_image(total)

        # 检查分割图子区
        if np.sum(total == label_femur) == 0:
            warnings.warn(f'femur-empty {trace}', stacklevel=2)
            return None

        ijk = np.argwhere(total == label_femur)
        b = np.array([ijk.min(axis=0), ijk.max(axis=0)])

        # 加载原图
        with tempfile.TemporaryDirectory() as tdir:
            f = Path(tdir) / 'f.nii.gz'
            try:
                client.fget_object('nii', object_name, f.as_posix())
            except S3Error:
                warnings.warn(f'nii-fget {trace}', stacklevel=2)
                return None

            try:
                image = itk.imread(f.as_posix(), itk.SS)
            except RuntimeError as mse:
                warnings.warn(f'nii-read {trace} {mse}', stacklevel=2)
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
        roi_image[np.where((roi_total != label_femur) & (roi_image > hu_bone))] = image_bg
        roi_images.append(roi_image)

        # 提取术后假体等值面
        if op == 0:
            metal_meshes.append(None)
        else:
            mesh = diff_dmc(wp.from_numpy(roi_image, wp.float32), np.zeros(3), spacing, hu_metal)
            if mesh.is_empty:
                warnings.warn(f'metal-empty {trace}', stacklevel=2)
                return None

            metal_meshes.append(mesh)

        # 提取骨等值面
        mesh = diff_dmc(wp.from_numpy(roi_image, wp.float32), np.zeros(3), spacing, hu_bone)
        if mesh.is_empty:
            warnings.warn(f'roi-empty {trace}', stacklevel=2)
            return None

        mesh = max(mesh.split(), key=lambda c: c.area)
        mesh = trimesh.smoothing.filter_taubin(mesh)
        roi_meshes.append(mesh)

    # 限制术后配准点采样区域，不太超出术前远端
    post_mesh: trimesh.Trimesh = roi_meshes[1].copy()
    zl = [np.max(_.vertices[:, 0]) - np.min(_.vertices[:, 0]) for _ in roi_meshes]

    # 术后比术前长，术后远端截断
    if 1.05 * zl[0] < zl[1]:
        z_min = np.min(roi_meshes[1].vertices[:, 0]) + zl[1] - zl[0]

        mask = post_mesh.vertices[:, 0] >= z_min
        post_mesh.update_faces(np.all(mask[post_mesh.faces], axis=1))
        post_mesh.remove_unreferenced_vertices()

        mask = ~mask
        post_mesh_outlier = roi_meshes[1].copy()
        post_mesh_outlier.update_faces(np.all(mask[post_mesh_outlier.faces], axis=1))
        post_mesh_outlier.remove_unreferenced_vertices()

    # 术前比术后长或等长
    else:
        post_mesh_outlier = None

    # 计算术后网格到假体的加权距离
    metal = wp.Mesh(wp.array(metal_meshes[1].vertices, wp.vec3), wp.array(metal_meshes[1].faces.flatten(), wp.int32))
    d = wp.zeros((len(post_mesh.vertices),), float)
    max_dist = np.linalg.norm(sizes[1] * spacings[1])
    wp.launch(compute_sdf, d.shape, [
        wp.uint64(metal.id), wp.array1d(post_mesh.vertices, wp.vec3), d, max_dist,
    ])
    d = d.numpy()

    # 初始近端对齐
    init_matrix = np.identity(4)
    init_matrix[0, 3] = np.max(roi_meshes[0].vertices[:, 0]) - np.max(roi_meshes[1].vertices[:, 0])

    # 配准术后到术前，配准特征点尽量远离金属，但术后过短则不得不接近金属
    pak = None
    for far in far_from_metal:
        _ = d - far
        # _[np.where(_ > far * 2)] = 0
        p = np.clip(_, 0, far)

        if (n := int(np.sum(p > 0))) < 100:
            continue

        p = p / p.sum()

        _ = np.random.choice(len(post_mesh.vertices), size=min(n, 10000), replace=False, p=p)
        vertices = post_mesh.vertices[_]

        matrix, _, mse, it = icp(
            vertices, roi_meshes[0], init_matrix, 1e-5, 2000,
            **dict(reflection=False, scale=False),
        )
        print(f'{patient} {rl} METAL-FREE {far}mm SURFACE-POINTS {n} ITERS {it} MSE {mse:.3f}mm')

        if pak is None or pak[0] > mse:
            pak = (mse, far, n, vertices, matrix, mse, it)

    if pak is None:
        raise RuntimeError('ICP failed')

    mse, far, n, vertices, matrix, mse, it = pak

    # post_mesh.apply_transform(matrix)
    # metal_meshes[1].apply_transform(matrix)
    # if post_mesh_outlier is not None:
    #     post_mesh_outlier.apply_transform(matrix)
    # vertices = trimesh.transform_points(vertices, matrix)
    roi_meshes[0].apply_transform(np.linalg.inv(matrix))

    # if mse > save_mse:
    #     raise UserCanceled

    import pyvista as pv
    pl = pv.Plotter(title=f'{patient}.{rl}', window_size=[500, 1000], off_screen=False)
    # pl.add_camera_orientation_widget()
    pl.enable_parallel_projection()
    pl.add_text(f'MSE {mse:.3f} mm', font_size=9)
    if post_mesh_outlier is not None:
        pl.add_mesh(post_mesh_outlier, color='green')
    pl.add_mesh(post_mesh, color='lightgreen')
    pl.add_mesh(roi_meshes[0], color='lightyellow')
    pl.add_mesh(metal_meshes[1], color='blue')
    pl.add_points(vertices, color='crimson', render_points_as_spheres=True, point_size=2)
    pl.camera_position = 'zx'
    pl.reset_camera(bounds=post_mesh.bounds.T.flatten())
    pl.camera.parallel_scale = max(zl) * 0.6
    if not pl.off_screen:
        pl.show(auto_close=False)

    if pl._closed:
        raise UserCanceled

    pl.camera_position = 'zx'
    pl.reset_camera(bounds=post_mesh.bounds.T.flatten())
    pl.camera.parallel_scale = max(zl) * 0.6
    pl.update()
    (f := Path(f'.pairs_align/redo_{redo_mse}/{patient}_{rl}.png')).parent.mkdir(parents=True, exist_ok=True)
    pl.screenshot(f.as_posix())
    pl.close()

    xform = np.array(wp.transform_from_matrix(wp.mat44(matrix)), dtype=float).tolist()
    return mse, xform


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--pairs', required=True)
    parser.add_argument('--redo_mse', type=float, default=0)
    parser.add_argument('--max_workers', type=int, default=1)
    args = parser.parse_args()

    pairs_path = Path(args.pairs)
    pairs: dict = tomlkit.loads(pairs_path.read_text('utf-8'))
    items = [(patient, rl, *pairs[patient][rl]) for patient in pairs for rl in pairs[patient]]
    items = sorted(items, key=lambda _: _[2], reverse=True)

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(main, args.config, args.redo_mse, *_): _ for _ in items}

        try:
            for fu in tqdm(as_completed(futures), total=len(futures)):
                try:
                    if (result := fu.result()) is not None:
                        _ = list(futures[fu])
                        pairs[_[0]][_[1]][0] = result[0]
                        pairs[_[0]][_[1]][1] = result[1]
                        pairs_path.write_text(tomlkit.dumps(pairs), 'utf-8')
                except UserCanceled:
                    print('UserCanceled')
                except Exception as _:
                    traceback.print_exception(_)
                    warnings.warn(f'{_} {futures[fu]}', stacklevel=2)

        except KeyboardInterrupt:
            print('Keyboard interrupted terminating...')
            executor.shutdown(wait=False)
            for future in futures:
                future.cancel()
