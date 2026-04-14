import argparse
import tempfile
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import tomlkit
from tqdm import tqdm

from b0_config import client_pairs


def main(config_file: str, it: dict):
    if 'head_center' in it:
        return

    if 'post' not in it:
        return

    import numpy as np
    roi_boxes = []
    for part in ('hip', 'femur'):
        if part not in it['post']:
            return

        origin = np.array(it['post'][part]['roi']['origin'])
        spacing = np.array(it['post'][part]['roi']['spacing'])
        size = np.array(it['post'][part]['roi']['size'])
        roi_boxes.append([origin, origin + spacing * size])

    import itk
    import warp as wp
    import trimesh
    from minio import Minio, S3Error
    from kernel import resample_cup_head, count_cup_head_3d
    from define import ct_metal, ct_min
    from PIL import Image, ImageDraw, ImageFont

    cfg_path = Path(config_file)
    cfg = tomlkit.loads(cfg_path.read_text('utf-8')).unwrap()
    client = Minio(**cfg['minio']['client'])

    prl = it['prl']
    pid, rl = prl.split('_')

    object_name = it['post']['nii']

    with tempfile.TemporaryDirectory() as tdir:
        f = Path(tdir) / 'image.nii.gz'
        try:
            client.fget_object('nii', object_name, f.as_posix())
        except S3Error:
            raise RuntimeError(f'下载原图失败 {object_name}')

        image = itk.imread(f.as_posix(), itk.SS)

        spacing = np.array(itk.spacing(image), float)
        origin = np.array(itk.origin(image), float)

        image = np.ascontiguousarray(itk.array_from_image(image).transpose(2, 1, 0))
        volume = wp.Volume.load_from_numpy(image, bg_value=ct_min)

        try:
            f = Path(tdir) / 'bone.stl'
            object_name = '/'.join([pid, rl, 'post', 'femur', f.name])
            client.fget_object('pair', object_name, f.as_posix())
        except S3Error:
            raise RuntimeError(f'下载股骨失败 {object_name}')

        _ = trimesh.load_mesh(f.as_posix())
        _ = max(_.split(), key=lambda _: _.area)
        bone_mesh = _

    counts = wp.zeros(6, dtype=wp.int32)

    def get_occupancy(hc, ca, lo):
        counts.zero_()
        cc = hc - lo * ca
        o = cc - 0.5 * roi_size * roi_spacing

        wp.launch(count_cup_head_3d, tuple(roi_size.tolist()), [
            volume.id, wp.vec3(origin), wp.vec3(spacing), ct_metal,
            wp.vec3(o), roi_spacing,
            wp.vec3(cc), wp.vec3(ca), wp.vec3(hc), head_outer / 2.0, cup_outer / 2.0,
            counts
        ])

        c = counts.numpy()
        head_roi_sum = float(c[0])
        head_metal_sum = float(c[1])
        cup_roi_sum = float(c[2])
        cup_metal_sum = float(c[3])
        liner_roi_sum = float(c[4])
        liner_metal_sum = float(c[5])

        h_occ = head_metal_sum / head_roi_sum if head_roi_sum > 0 else 0.0
        c_occ = cup_metal_sum / cup_roi_sum if cup_roi_sum > 0 else 0.0
        l_occ = liner_metal_sum / liner_roi_sum if liner_roi_sum > 0 else 0.0

        return h_occ, c_occ, l_occ

    roi_boxes = np.array(roi_boxes)

    view_size = 100
    roi_spacing = 0.2
    roi_size = np.ceil((np.ones(3) * view_size) / roi_spacing).astype(int)

    cup_outer = int(it['cup_outer'])
    head_outer = int(it['head_outer'])

    v_max = bone_mesh.vertices[np.argmax(bone_mesh.vertices[:, 2])]
    head_center = v_max.copy()
    head_center += roi_boxes[1][0]
    head_center[2] -= 15

    head_center = np.array(head_center)
    head_center = np.round(head_center / 0.25) * 0.25

    cup_axis = np.array([(1 if rl == 'L' else -1) * np.sin(np.deg2rad(40)), 0, -np.cos(np.deg2rad(40))])
    cup_axis /= np.linalg.norm(cup_axis)

    # 斐波那契球面均匀采样方向
    samples = 26
    phi = np.pi * (3. - np.sqrt(5.))
    i = np.arange(samples)
    y = 1 - (i / float(samples - 1)) * 2
    radius = np.sqrt(1 - y * y)
    theta = phi * i
    x = np.cos(theta) * radius
    z = np.sin(theta) * radius
    sphere_directions = np.column_stack((x, y, z)).tolist()

    liner_offset_best = float(it.get('liner_offset', 0))
    occ_max = get_occupancy(head_center, cup_axis, liner_offset_best)

    # 梯度步长
    for step in (5.0, 0.25, 0.25, 0.25):
        better = True
        while better:  # 位置
            better = False

            for offset in sphere_directions + [-cup_axis, cup_axis]:
                offset = np.array(offset, float)
                offset /= np.linalg.norm(offset)
                head_center_test = head_center + offset * step

                occ = get_occupancy(head_center_test, cup_axis, liner_offset_best)

                if occ_max[0] * 0.8 + occ_max[1] * 0.2 < occ[0] * 0.8 + occ[1] * 0.2:
                    occ_max = occ
                    head_center = head_center_test
                    better = True
                    break

        better = True
        while better:  # 朝向
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

                occ = get_occupancy(head_center, v, liner_offset_best)

                if occ_max[0] * 0.2 + occ_max[1] * 0.8 < occ[0] * 0.2 + occ[1] * 0.8:
                    occ_max = occ
                    cup_axis = v.copy()
                    better = True
                    break

        liner_offset_test = liner_offset_best
        occ_max = get_occupancy(head_center, cup_axis, liner_offset_test)

        th = (cup_outer - head_outer) * 0.25
        for _ in range(-int(th // 0.25), int(5.0 // 0.25)):
            liner_offset_test = liner_offset_best + _ * 0.25

            occ = get_occupancy(head_center, cup_axis, liner_offset_test)

            if occ_max[0] * 0.2 + occ_max[1] * 0.8 < occ[0] * 0.2 + occ[1] * 0.8:
                occ_max = occ
                liner_offset_best = liner_offset_test

    cup_center = head_center - liner_offset_best * cup_axis

    ort = [
        ['S', 'I', 'A', 'P'],
        ['S', 'I', 'R', 'L'],
        ['A', 'P', 'R', 'L'],
    ]

    stack = []
    for m, w in ((-100.0, 1000.0), (2000.0, 1000.0)):
        for i in range(3):
            axes = np.eye(3, dtype=float).tolist()
            del axes[i]
            axes = np.array(axes)

            shape = [*roi_size]
            del shape[i]

            roi_slice = wp.zeros(shape, dtype=wp.vec3ub)
            roi_origin = head_center - 0.5 * roi_size * roi_spacing * axes[0] - 0.5 * roi_size * roi_spacing * axes[1]

            wp.launch(resample_cup_head, roi_slice.shape, [
                volume.id, wp.vec3(origin), wp.vec3(spacing), m, w,
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

    root = cfg['dataset']['root']
    f = Path(root) / 'prothesis_pred' / f'{prl}.png'
    f.parent.mkdir(parents=True, exist_ok=True)

    Image.fromarray(np.vstack([np.hstack(stack[:3]), np.hstack(stack[3:])])).save(f)

    it.update({
        'cup_center': cup_center.tolist(),
        'head_center': head_center.tolist(),
        'cup_axis': cup_axis.tolist(),
        'liner_offset_best': liner_offset_best,
        'liner_material': '陶瓷' if occ_max[2] > 0.5 else '聚乙烯',
        'occupancy': list(occ_max),
    })
    data = tomlkit.dumps(it).encode('utf-8')
    client.put_object('pair', '/'.join([pid, rl, 'context.toml']), BytesIO(data), len(data))


def launch(cfg_path: str, max_workers: int):
    client, pairs = client_pairs(cfg_path, 'context')

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
    parser.add_argument('--max_workers', type=int, default=6)
    args = parser.parse_args()

    launch(args.config, args.max_workers)
