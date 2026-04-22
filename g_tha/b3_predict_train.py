import argparse
import tempfile
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import tomlkit
import trimesh
from minio import Minio, S3Error
from tqdm import tqdm

from b0_config import client_pairs
from define import roi_spacing, sdf_t, ct_min, ct_metal
from kernel import resample_roi, diff_dmc, resample_metal

padding = roi_spacing + sdf_t


def preload(cfg: dict, client, it: dict):
    prl = it['prl']
    pid, rl = prl.split('_')

    context, align = it['context'], it['align']

    # 下载数据
    ct_images, roi_bounds, ct_femurs, sizes, spacings, origins = [], [], [], [], [], []

    import itk
    with tempfile.TemporaryDirectory() as tdir:
        for op, object_name in enumerate((it['nii']['pre'], it['nii']['post'])):
            f = Path(tdir) / 'image.nii.gz'
            try:
                client.fget_object('nii', object_name, f.as_posix())
            except S3Error:
                raise RuntimeError('Missing {} data'.format(['pre', 'post'][op]))

            image = itk.imread(f.as_posix(), itk.SS)

            size = np.array(itk.size(image), float)
            spacing = np.array(itk.spacing(image), float)
            origin = np.array(itk.origin(image), float)

            sizes.append(size)
            spacings.append(spacing)
            origins.append(origin)

            image = itk.array_from_image(image).transpose(2, 1, 0)
            ct_images.append(image)

    import warp as wp
    volumes = [wp.Volume.load_from_numpy(ct_images[_], bg_value=ct_min) for _ in range(2)]

    # 配准变换
    xforms = [wp.transform(*align[f'{part}_align']) for part in ('hip', 'femur')]
    xforms_inv = [wp.transform_inverse(_) for _ in xforms]

    cup_radius = int(context['cup_outer']) * 0.5
    cup_center = np.array(context['cup_center'], float)
    cup_center = [np.array(wp.transform_point(xforms[_], wp.vec3(cup_center))) for _ in range(2)]

    cup_axis = np.array(context['cup_axis'], float)
    cup_axis = [np.array(wp.transform_vector(xforms[_], wp.vec3(cup_axis))) for _ in range(2)]

    head_radius = int(context['head_outer']) * 0.5
    head_center = np.array(context['head_center'], float)
    head_center = [np.array(wp.transform_point(xforms[_], wp.vec3(head_center))) for _ in range(2)]

    # 采样区域
    roi_boxes = []
    for i, part in enumerate(('hip', 'femur')):
        cup_box = [cup_center[i] - cup_radius, cup_center[i] + cup_radius]

        origin_pre = np.array(it['roi'][part]['pre']['origin'])
        spacing_pre = np.array(it['roi'][part]['pre']['spacing'])
        size_pre = np.array(it['roi'][part]['pre']['size'])
        box_pre = np.array([origin_pre, origin_pre + spacing_pre * size_pre])

        origin_post = np.array(it['roi'][part]['post']['origin'])
        spacing_post = np.array(it['roi'][part]['post']['spacing'])
        size_post = np.array(it['roi'][part]['post']['size'])
        
        # 计算术后包围盒在其局部坐标系下的中心和半边长
        center_post_local = origin_post + spacing_post * size_post * 0.5
        extents_post_local = spacing_post * size_post * 0.5

        # 变换中心点到术前(世界)坐标系
        center_post_world = np.array(wp.transform_point(xforms[i], wp.vec3(*center_post_local)))

        # 提取旋转矩阵
        axes_world = [np.array(wp.transform_vector(xforms[i], wp.vec3(*axis))) for axis in ([1.0,0.0,0.0], [0.0,1.0,0.0], [0.0,0.0,1.0])]
        R = np.column_stack(axes_world)
        
        # 计算最大内接AABB的半边长：|R^T| * h = extents
        S = np.abs(R).T
        try:
            h_world = np.linalg.solve(S, extents_post_local)
            if np.any(h_world <= 0):
                h_world = extents_post_local / np.sum(S, axis=1) # 降级为保守的安全缩放
        except Exception:
            h_world = extents_post_local / np.sum(S, axis=1)

        box_post = np.array([center_post_world - h_world, center_post_world + h_world])

        box_common = np.array([
            np.maximum(box_pre[0], box_post[0]),
            np.minimum(box_pre[1], box_post[1])
        ])

        box = np.array([box_common, [cup_box[0] - padding, cup_box[1] + padding]])
        box = np.array([box[:, 0].min(0), box[:, 1].max(0)])

        roi_boxes.append(box)
        del origin_pre, spacing_pre, size_pre, box_pre, origin_post, spacing_post, size_post, center_post_local, extents_post_local, center_post_world, axes_world, R, S, h_world, box_post, box_common, box

    roi_boxes = np.array(roi_boxes)
    roi_box = np.array([roi_boxes[:, 0].min(0), roi_boxes[:, 1].max(0)])

    extents = roi_box[1] - roi_box[0]

    roi_size = np.ceil(extents / roi_spacing).astype(int)
    roi_size = np.ceil(roi_size / 64.0).astype(int) * 64
    max_dist = wp.float32(np.linalg.norm(roi_size * roi_spacing))

    roi_origin = (roi_box[0] + roi_box[1]) * 0.5 - 0.5 * roi_spacing * roi_size

    # 采样图像
    roi_images = wp.full((*roi_size,), -1.0, wp.vec3)
    hip_metal = wp.full((*roi_size,), -1.0, wp.float32)
    femur_metal = wp.full((*roi_size,), -1.0, wp.float32)

    wp.launch(resample_roi, (*roi_size,), [
        roi_images, roi_origin, wp.vec3(roi_spacing),
        roi_boxes[0][0], roi_boxes[0][1], roi_boxes[1][0], roi_boxes[1][1],
        xforms_inv[0], xforms_inv[1],
        volumes[0].id, origins[0], spacings[0], volumes[1].id, origins[1], spacings[1],
        hip_metal, femur_metal, ct_metal,
    ])

    roi_images = roi_images.numpy()
    pre_image, part_images = roi_images[..., 0], [roi_images[..., 1], roi_images[..., 2]]

    # 重建假体
    meshes = []
    for i, part in enumerate(('hip', 'femur')):
        mesh = diff_dmc([hip_metal, femur_metal][i], roi_origin, roi_spacing, 0.0)

        if not mesh.is_empty:
            # 允许非水密组件，因为 DMC 结果可能存在微小开孔
            ls = [m for m in sorted(
                mesh.split(only_watertight=False), key=lambda _: np.linalg.norm(_.bounds[1] - _.bounds[0]),
                reverse=True,
            ) if np.linalg.norm(m.vertices - head_center[i], axis=1).min() <= cup_radius]

            if len(ls):
                mesh: trimesh.Trimesh = trimesh.util.concatenate(ls)  # noqa
                mesh.fix_normals()
            else:
                raise RuntimeError(f'{part} metal is far')
        else:
            raise RuntimeError(f'{part} metal is empty')

        wp_mesh = wp.Mesh(wp.array(mesh.vertices, dtype=wp.vec3), wp.array(mesh.faces.flatten(), dtype=wp.int32))
        meshes.append(wp_mesh)

    # 采样假体
    metal_image = wp.full((*roi_size,), -1.0, wp.vec3)

    wp.launch(resample_metal, (*roi_size,), [
        metal_image, roi_origin, wp.vec3(roi_spacing), meshes[0].id, meshes[1].id,
        cup_center[0], cup_center[1], cup_axis[0], cup_axis[1],
        head_center[0], head_center[1], cup_radius, head_radius,
        sdf_t, max_dist, 0.0,
    ])

    metal_image = metal_image.numpy()

    # 快照
    ijk = ((head_center[0] + head_center[1]) * 0.5 - roi_origin) / roi_spacing
    i, j, k = np.round(ijk).astype(int)

    stack = []
    axis = 1
    for image in (pre_image, *part_images, *[metal_image[..., _] for _ in range(3)]):
        if axis == 0:
            img = image[i, :, :].copy()
        elif axis == 1:
            img = image[:, j, :].copy()
        else:
            img = image[:, :, k].copy()

        if axis in (0, 1):
            img = np.flipud(img.T)

        img = np.clip(img * 127 + 127, 0, 255).astype(np.uint8)
        stack.append(img)

    root = Path(cfg['dataset']['root'])
    f = root / 'predict_train' / f'{prl}.png'
    f.parent.mkdir(parents=True, exist_ok=True)

    from PIL import Image
    Image.fromarray(np.hstack(stack)).save(f)

    return cfg, client, it


def submit(cfg: dict, client, it: dict):
    prl = it['prl']
    pid, rl = prl.split('_')


def main(config_file: str, it: dict):
    if it.get('excluded', False):
        return

    cfg_path = Path(config_file)
    cfg = tomlkit.loads(cfg_path.read_text('utf-8')).unwrap()
    client = Minio(**cfg['minio']['client'])

    submit(*preload(cfg, client, it))


def launch(cfg_path: str, max_workers: int):
    client, pairs = client_pairs(cfg_path, ['context', 'align'])

    # for _ in tqdm(['1004333_L', '1066852_R']):
    #     main(cfg_path, pairs[_])
    # return

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(main, cfg_path, it): prl for prl, it in pairs.items()}

        try:
            for fu in tqdm(as_completed(futures), total=len(futures)):
                try:
                    fu.result()
                except Exception as _:
                    warnings.warn(f'{_} {futures[fu]}', stacklevel=1)

        except KeyboardInterrupt:
            print('Keyboard interrupted terminating...')
            executor.shutdown(wait=False)
            for future in futures:
                future.cancel()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--max_workers', type=int, default=4)
    args = parser.parse_args()

    launch(args.config, args.max_workers)
