# PyTorch
# pip install numpy PIL opencv-python kornia tqdm

import argparse
import colorsys
import warnings
from concurrent.futures import as_completed, ThreadPoolExecutor
from pathlib import Path
from random import randint

import cv2
import kornia
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def to_tensor(img):
    t = torch.from_numpy(np.array(img)).float() / 255.
    t = t.permute(2, 0, 1).unsqueeze(0)  # [B,C,H,W]
    return t


def feature_match(matcher, conf_sums, image_0, image_1, h, crop_h):
    t1 = to_tensor(image_0[h:h + crop_h]).cuda()
    t2 = to_tensor(image_1).cuda()

    with torch.no_grad():
        out = matcher({
            'image0': kornia.color.rgb_to_grayscale(t1),
            'image1': kornia.color.rgb_to_grayscale(t2),
        })

    kp0 = out['keypoints0'].cpu().numpy()
    kp1 = out['keypoints1'].cpu().numpy()
    conf = out['confidence'].cpu().numpy()
    conf_sums[h] = np.sum(conf)
    return h, kp0, kp1, conf


def draw_matches(im0, im1, kp0, kp1, conf):
    h0, w0 = im0.shape[:2]
    h1, w1 = im1.shape[:2]

    canvas_h = h0 + h1
    canvas_w = max(w0, w1)

    # 创建画布
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # 计算左右偏移量，使两张图都居中
    offset0 = (canvas_w - w0) // 2
    offset1 = (canvas_w - w1) // 2

    # 分别粘贴 im0 和 im1（上下排列，水平居中）
    canvas[:h0, offset0:offset0 + w0] = im0
    canvas[h0:h0 + h1, offset1:offset1 + w1] = im1

    # 绘制匹配
    for (x0, y0), (x1, y1), c in zip(kp0.astype(int), kp1.astype(int), conf.astype(float)):
        r, g, b = colorsys.hsv_to_rgb(randint(0, 12) * 30 / 360, 1, 1)
        # r, g, b = colorsys.hsv_to_rgb(c * 120 / 360, 1, 1)
        color = tuple(int(x * 255) for x in (b, g, r))

        cv2.circle(canvas, (x0 + offset0, y0), 3, color, -1)  # im0 的点
        cv2.circle(canvas, (x1 + offset1, y1 + h0), 3, color, -1)  # im1 的点（再加垂直偏移 h0）
        cv2.line(canvas, (x0 + offset0, y0), (x1 + offset1, y1 + h0), color, 1)
    return canvas


def stack(arr: list):
    shape = max(arr[0].shape[0], arr[1].shape[0]), arr[0].shape[1] + arr[1].shape[1], 3
    stacked = np.zeros(shape, dtype=arr[0].dtype)
    stacked[:arr[0].shape[0], :arr[0].shape[1]] = arr[0]
    stacked[:arr[1].shape[0], arr[0].shape[1]:arr[0].shape[1] + arr[1].shape[1]] = arr[1]
    return stacked


def main(dataset_dir: str, TotalBody_file: str, roi_files: dict, matcher: kornia.feature.LoFTR):
    TotalBody_file = Path(TotalBody_file)
    if (touch := TotalBody_file.parent.parent / 'TotalBody_Matched' / TotalBody_file.name).exists():
        return

    dataset_dir = Path(dataset_dir).absolute()
    test_dir = dataset_dir / 'test'

    _ = TotalBody_file.stem.split('_')
    prefix, TotalBody_i = '_'.join(_[:-2]), _[-1]

    # 全身图
    image_0 = np.array(Image.open(TotalBody_file))

    # 局部图
    roi_files = {roi: (
        roi_files.get(f'{prefix}_Right{roi}'),
        roi_files.get(f'{prefix}_Left{roi}'),
        prefix,
    ) for roi in ('Femur', 'OrthoKnee')}

    if None not in roi_files.values():
        for roi in roi_files:
            image_1 = [roi_files[roi][_] for _ in range(2)]
            if None in image_1:
                warnings.warn(f'Incomplete Left or Right {roi} {roi_files[roi][2]}')
                return

            image_1 = stack([np.array(Image.open(roi_files[roi][_])) for _ in range(2)])

            # 特征匹配
            crop_h = image_1.shape[0]
            end_h = image_0.shape[0] - crop_h

            conf_sums = np.zeros((end_h, 1), float)
            kps = {}

            step = 16
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                futures = {executor.submit(
                    feature_match, matcher, conf_sums, image_0, image_1, h, crop_h,
                ) for h in range(0, end_h, step)}

                for _ in as_completed(futures):
                    h, kp0, kp1, conf = _.result()
                    kps[h] = (kp0, kp1, conf)

            conf_sums = conf_sums[::step]
            for _ in range(5):
                conf_sums = cv2.blur(conf_sums, (5, 5))
            conf_sums = conf_sums.squeeze()

            best_h = int(conf_sums.argmax())

            best_h *= step
            best_kp0, best_kp1, best_conf = kps[best_h]

            roi_image = image_0[best_h:best_h + crop_h]
            canvas = draw_matches(roi_image, image_1, best_kp0, best_kp1, best_conf)

            stem = f'{prefix}_TotalBody_{TotalBody_i}_{roi}_{best_h}_{crop_h}'

            _ = test_dir / '8x_Match' / f'{stem}.png'
            _.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(canvas).save(_)

            _ = test_dir / '8x' / f'{stem}.png'
            _.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(roi_image).save(_)

    touch.parent.mkdir(parents=True, exist_ok=True)
    touch.touch()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--initial', type=int, default=0)
    parser.add_argument('--max_workers', type=int, default=16)
    args = parser.parse_args()

    _matcher = kornia.feature.LoFTR(pretrained='outdoor').cuda()

    _roi_files = {
        Path(_).stem: _
        for __ in ('val', 'train')
        for _ in (Path(args.dataset_dir) / __ / 'lq' / '8x').rglob('*.png')
    }

    TotalBody_files = list((Path(args.dataset_dir) / 'test' / 'TotalBody').rglob('*.png'))

    try:
        for _ in tqdm(TotalBody_files, initial=args.initial):
            main(args.dataset_dir, _.as_posix(), _roi_files, _matcher)

    except KeyboardInterrupt:
        print('Interrupted')
