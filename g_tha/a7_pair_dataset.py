import argparse
import multiprocessing
import tempfile
import time
from pathlib import Path

import itk
import numpy as np
import tomlkit
import trimesh
import warp as wp
from PIL import Image
from minio import Minio, S3Error
from tqdm import tqdm

from kernel import diff_dmc, resample_obb, fast_drr


def main(config: str, prl: str, pair: dict):
    cfg_path = Path(config)
    cfg = tomlkit.loads(cfg_path.read_text('utf-8'))
    client = Minio(**cfg['minio']['client'])

    ct_metal = cfg['ct']['metal']
    roi_spacing = np.ones(3) * cfg['ct']['roi']['spacing']

    dataset_root = Path(str(cfg['dataset']['root']))

    snapshot_file = dataset_root / 'snapshot' / f'{prl}.png'
    if snapshot_file.exists():
        return

    subdir = 'val' if pair.get('is_val', False) else 'train'

    rl = prl.split('_')[1]
    label_femur = {
        'R': cfg['totalsegmentator']['label']['right_femur'],
        'L': cfg['totalsegmentator']['label']['left_femur'],
    }[rl]

    ct_images, roi_bounds, ct_femurs, sizes, spacings, origins, image_bgs = [], [], [], [], [], [], []

    for op, object_name in enumerate((pair['pre'], pair['post'])):
        with tempfile.TemporaryDirectory() as tdir:
            f = Path(tdir) / 'total.nii.gz'
            try:
                client.fget_object('total', object_name, f.as_posix())
            except S3Error:
                continue

            total = itk.imread(f.as_posix(), itk.UC)
            total = itk.array_from_image(total)

            if not np.any(total == label_femur):
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
        raise RuntimeError('')

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
        raise RuntimeError('')

    if 'post_xform_global' in pair:
        post_xform = wp.transform(*pair['post_xform_global'])
    elif 'post_xform' in pair:
        post_xform = wp.transform(*pair['post_xform'])
        post_xform = np.array(wp.transform_to_matrix(post_xform), float).reshape((4, 4))

        offset = [np.array(origins[_]) + np.array(roi_bounds[_][0]) * np.array(spacings[_]) for _ in range(2)]

        pre = np.identity(4)
        pre[:3, 3] = offset[0]

        post_inv = np.identity(4)
        post_inv[:3, 3] = -offset[1]

        post_xform = pre @ post_xform @ post_inv
        post_xform = wp.transform_from_matrix(wp.mat44(post_xform))
    else:
        raise RuntimeError('')

    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2.0
    extents = bounds[1] - bounds[0]
    extents += 20.0
    
    roi_size = np.ceil(extents / roi_spacing).astype(int)
    roi_size = np.ceil(roi_size / 64.0).astype(int) * 64
    
    obb_xform = np.identity(4)
    obb_xform[:3, 3] = center

    origin = -0.5 * roi_spacing * roi_size

    obb_xform = wp.transform_from_matrix(wp.mat44(obb_xform))
    volumes = [wp.Volume.load_from_numpy(ct_images[_], bg_value=image_bgs[_]) for _ in range(2)]

    image_obb = wp.full((*roi_size,), wp.vec2(image_bgs[1], image_bgs[0]), wp.vec2)
    wp.launch(resample_obb, image_obb.shape, [
        image_obb, origin, roi_spacing, obb_xform,
        volumes[1].id, origins[1], spacings[1], sizes[1],
        volumes[0].id, origins[0], spacings[0], sizes[0],
        post_xform if post_xform is not None else wp.transform_identity(), post_xform is not None,
    ])
    image_obb = image_obb.numpy()
    image_a, image_b = image_obb[:, :, :, 1], image_obb[:, :, :, 0]

    # save
    for op, image in (('pre', image_a), ('post', image_b)):
        f = dataset_root / op / subdir / f'{prl}.nii.gz'
        f.parent.mkdir(parents=True, exist_ok=True)

        _ = itk.image_from_array(image)
        _.SetSpacing(roi_spacing)
        itk.imwrite(_, f.as_posix(), True)

    snapshot = []
    for ax in (1, 2):
        stack = [fast_drr(image_a, ax), fast_drr(image_b, ax)]
        img = np.hstack(stack)
        snapshot.append(img)

    snapshot = np.flipud(np.hstack(snapshot))

    snapshot_file.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(snapshot, 'RGB').save(snapshot_file.as_posix())

    del volumes
    del image_obb
    del ct_images

    import gc
    gc.collect()
    time.sleep(0.5)


def launch():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--max-workers', default=10, type=int)
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

    valid_pairs = {}
    for prl in pairs:
        try:
            data = client.get_object('pair', '/'.join([prl.replace('_', '/'), 'align.toml'])).data
            data = tomlkit.loads(data.decode('utf-8'))

            pairs[prl].update(data)

            if len(pairs[prl].get('excluded', [])) > 0:
                continue

            if 'post_xform_global' not in pairs[prl] and 'post_xform' not in pairs[prl]:
                continue

            pairs[prl]['is_val'] = False
            valid_pairs[prl] = pairs[prl]
        except S3Error:
            pass

    pairs = valid_pairs

    keys = sorted(pairs.keys())
    total = len(keys)
    n_val = min(int(total * 0.1), 100)

    if n_val > 0:
        for i in range(n_val):
            idx = int(i * total / n_val)
            pairs[keys[idx]]['is_val'] = True

    tasks = [(args.config, prl, pair) for prl, pair in pairs.items()]

    # 确保单进程显存回收
    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(processes=args.max_workers, maxtasksperchild=1) as pool:
        for _ in tqdm(pool.imap_unordered(process, tasks), total=len(tasks)):
            pass


def process(args):
    try:
        main(*args)
    except KeyboardInterrupt:
        print('Keyboard interrupted terminating...')


if __name__ == '__main__':
    launch()
