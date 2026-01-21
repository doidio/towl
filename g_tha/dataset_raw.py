# uv run streamlit run pair_contact.py --server.port 8503 -- --config config.toml

import argparse
import tempfile
from pathlib import Path

import itk
import numpy as np
import tomlkit
import trimesh.primitives
import warp as wp
from PIL import Image
from minio import Minio, S3Error
from tqdm import tqdm

from kernel import diff_dmc, resample_obb, fast_drr

ct_bone, ct_metal = 220, 2700
roi_size = [576, 224, 352]
roi_spacing = 0.5 * np.ones(3)

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

for prl in (progress := tqdm(pairs)):
    if len(pairs[prl].get('excluded', [])) > 0:
        continue

    if 'post_xform_global' not in pairs[prl] and 'post_xform' not in pairs[prl]:
        continue

    rl = prl.split('_')[1]
    label_femur = {'R': 76, 'L': 75}[rl]

    ct_images, roi_bounds, ct_femurs, sizes, spacings, origins, image_bgs = [], [], [], [], [], [], []

    for op, object_name in enumerate((pairs[prl]['pre'], pairs[prl]['post'])):
        with tempfile.TemporaryDirectory() as tdir:
            f = Path(tdir) / 'total.nii.gz'
            try:
                client.fget_object('total', object_name, f.as_posix())
            except S3Error:
                continue

            total = itk.imread(f.as_posix(), itk.UC)
            total = itk.array_from_image(total)

            if np.sum(total == label_femur) == 0:
                continue

            ijk = np.argwhere(_ := (total == label_femur))
            ct_femurs.append(_)

            box = np.array([ijk.min(axis=0), ijk.max(axis=0) + 1])
            roi_bounds.append(box.tolist())

            f = Path(tdir) / 'image.nii.gz'
            try:
                client.fget_object('nii', object_name, f.as_posix())
            except S3Error:
                continue

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

    if len(image_bgs) < 2:
        continue

    # 根据术后股骨与金属交集确定采样范围
    _ = ct_femurs[1] & (ct_images[1] >= ct_metal)
    mesh = diff_dmc(wp.from_numpy(_, wp.float32), origins[1], spacings[1], 0.5)

    # 金属可能分离成髋臼杯、球头、股骨柄、膝关节假体，选范围最大的股骨柄以上
    if not mesh.is_empty:
        ls = list(sorted(
            mesh.split(only_watertight=True),
            key=lambda _: np.linalg.norm(_.bounds[1] - _.bounds[0]), reverse=True,
        ))
        mesh: trimesh.Trimesh = ls[0]
    else:
        continue

    obb = mesh.bounding_box_oriented
    _ = mesh.copy()
    _.apply_transform(np.linalg.inv(obb.transform))
    extents = _.bounds[1] - _.bounds[0]
    x = np.argmax(extents)

    obb_xform = obb.transform.copy()
    indices = [(x + i) % 3 for i in range(3)]
    obb_xform[:3, :3] = obb.transform[:3, :3][:, indices]

    obb_xform[:3, :3] = obb.transform[:3, :3][:, indices]
    if obb_xform[0, 0] < 0:
        obb_xform[:3, 0] *= -1
        obb_xform[:3, 2] *= -1
    if obb_xform[1, 1] < 0:
        obb_xform[:3, 1] *= -1
        obb_xform[:3, 2] *= -1

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
        continue

    origin = -0.5 * roi_spacing * roi_size

    obb_xform = wp.transform_from_matrix(wp.mat44(obb_xform))
    volumes = [wp.Volume.load_from_numpy(ct_images[_], bg_value=image_bgs[_]) for _ in range(2)]

    image_obb = wp.full((*roi_size,), wp.vec2(image_bgs[1], image_bgs[0]), wp.vec2)
    wp.launch(resample_obb, image_obb.shape, [
        image_obb, origin, roi_spacing, obb_xform,
        volumes[1].id, origins[1], spacings[1], post_xform if post_xform is not None else wp.transform_identity(),
        volumes[0].id, origins[0], spacings[0], post_xform is not None,
    ])
    image_obb = image_obb.numpy()
    image_a, image_b = image_obb[:, :, :, 1], image_obb[:, :, :, 0]

    # save
    for op, image in (('pre', image_a), ('post', image_b)):
        f = Path(f'.ds/{op}/{prl}.nii.gz')
        f.parent.mkdir(parents=True, exist_ok=True)

        _ = itk.image_from_array(image)
        _.SetSpacing(roi_spacing)
        itk.imwrite(_, f.as_posix())

    snapshot = []
    for ax in (1, 2):
        stack = [fast_drr(image_a, ax), fast_drr(image_b, ax)]
        img = np.hstack(stack)
        snapshot.append(img)

    snapshot = np.flipud(np.hstack(snapshot))

    f = Path(f'.ds/snapshot/{prl}.png')
    f.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(snapshot, 'RGB').save(f.as_posix())
