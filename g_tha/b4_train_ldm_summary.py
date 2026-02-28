import argparse
import os
import re
from pathlib import Path

import SimpleITK as sitk
import cv2
import imageio
import numpy as np
from tqdm import tqdm

from PIL import Image, ImageDraw, ImageFont


def normalize_image(img_arr):
    """Normalize numpy array to 0-255 uint8 using CT Bone Window (W=1500, L=300)"""
    # 骨窗映射：L=300, W=1500 (范围 -450 到 1050)
    l, w = 300, 1500
    img_min = l - w // 2
    img_max = l + w // 2
    img_arr = np.clip(img_arr, img_min, img_max)
    img_arr = (img_arr - img_min) / (img_max - img_min) * 255.0
    return img_arr.astype(np.uint8)


def fuse_sdf_to_ct(sdf_slice, ct_slice):
    """
    参考 v2/infer.py 的融合算法将 SDF 假体融合到 CT 切片中
    """
    metal_min = 2700.0
    ct_max = 3071.0
    delta = 0.2  # 表面柔化范围

    fused = ct_slice.copy().astype(np.float32)

    # 1. 假体核心内部 (SDF >= delta)
    fused[sdf_slice >= delta] = ct_max

    # 2. 假体浅表内部过渡 (0 <= SDF < delta)
    mask_inner = (sdf_slice >= 0.0) & (sdf_slice < delta)
    fused[mask_inner] = metal_min + (sdf_slice[mask_inner] / delta) * (ct_max - metal_min)

    # 3. 假体外部边缘过渡 (-delta <= SDF < 0)
    mask_outer = (sdf_slice >= -delta) & (sdf_slice < 0.0)
    t = (sdf_slice[mask_outer] + delta) / delta
    fused[mask_outer] = ct_slice[mask_outer] * (1.0 - t) + metal_min * t

    return fused


def add_label_with_pil(img_bgr, text):
    """使用 PIL 在图像底部居左添加文字"""
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # 尝试加载中文字体，如果失败则回退到默认
    try:
        font = ImageFont.truetype("msyh.ttc", 10)
    except Exception:
        font = ImageFont.load_default()

    # 获取文字大小
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    img_width, img_height = img_pil.size

    x = 5
    y = img_height - text_height - 10

    # 写白字，不再绘制背景矩形
    draw.text((x, y), text, font=font, fill=(255, 255, 255))

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def main():
    parser = argparse.ArgumentParser(description='Extract validation NIfTI images and create an MP4 movie.')
    parser.add_argument('--log_dir', type=str, required=True, help='Path to the specific tensorboard log directory')
    parser.add_argument('--output', type=str, default='save/training_summary.mp4', help='Output MP4 file path')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second for the video')
    args = parser.parse_args()

    log_path = Path(args.log_dir)
    if not log_path.exists():
        raise FileNotFoundError(f"Log directory not found: {log_path}")

    # 1. Load Condition and GroundTruth (should be from Epoch 0)
    # 术前条件改为使用原图
    cond_path = Path('dataset/pre/val/1004333_L.nii.gz')
    gt_path = log_path / 'val_epoch_000_GroundTruth.nii.gz'

    if not cond_path.exists() or not gt_path.exists():
        raise FileNotFoundError(f"Condition or GroundTruth file missing")

    cond_img = sitk.GetArrayFromImage(sitk.ReadImage(str(cond_path)))
    gt_img_raw = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_path)))

    # 参考 v2/infer.py 进行 CenterCrop，确保尺寸与原图一致
    def center_crop_3d(img, target_shape):
        """Center crop 3D array [Z, Y, X] to target_shape [Z_t, Y_t, X_t]"""
        z, y, x = img.shape
        tz, ty, tx = target_shape

        # 计算起始索引
        sz = max(0, (z - tz) // 2)
        sy = max(0, (y - ty) // 2)
        sx = max(0, (x - tx) // 2)

        # 裁剪并确保不超过边界（以防 target 某维度大于 source）
        return img[sz:sz + tz, sy:sy + ty, sx:sx + tx]

    gt_img = center_crop_3d(gt_img_raw, cond_img.shape)

    # Get middle slice index along the depth axis
    print(f"Original shape: {cond_img.shape}, Patch shape: {gt_img_raw.shape}")

    # 根据用户要求的 [:, idx, :]，索引应该在第 1 个维度 (y轴/冠状面)
    # sitk 读入通常是 (Z, Y, X)。所以 idx 对应 Y 轴。
    # 动态取中间切片，避免 IndexError (例如原图 Y 为 64 时 idx 应为 32)
    idx = cond_img.shape[1] // 2 + 5

    # 获取未归一化的原始切片并上下翻转
    cond_raw_slice = np.flipud(cond_img[:, idx, :])
    gt_raw_slice = np.flipud(gt_img[:, idx, :])

    # 归一化术前条件图
    cond_slice = normalize_image(cond_raw_slice)

    # 生成一个合成的 GroundTruth 切片
    gt_fused_raw = fuse_sdf_to_ct(gt_raw_slice, cond_raw_slice)
    gt_implant_slice = normalize_image(gt_fused_raw)

    # Convert to BGR for adding text and coloring
    cond_bgr = cv2.cvtColor(cond_slice, cv2.COLOR_GRAY2BGR)
    gt_implant_bgr = cv2.cvtColor(gt_implant_slice, cv2.COLOR_GRAY2BGR)

    # 给术前和术后打上中文标签
    cond_bgr_labeled = add_label_with_pil(cond_bgr.copy(), "术前")
    gt_implant_bgr_labeled = add_label_with_pil(gt_implant_bgr.copy(), "术后")

    # 2. Find all Generated NIfTI files and sort them by epoch
    gen_files = list(log_path.glob('val_epoch_*_Generated.nii.gz'))

    def extract_epoch(filepath):
        match = re.search(r'val_epoch_(\d+)_Generated', filepath.name)
        return int(match.group(1)) if match else -1

    gen_files.sort(key=extract_epoch)

    if not gen_files:
        raise ValueError("No generated NIfTI files found.")

    print(f"Found {len(gen_files)} generated images. Creating video...")

    # 计算目标采样 epoch 列表，采样间隔按等差数列递增 (例如: 0, 10, 30, 60, 100...)
    # 间隔从 10 开始，每次递增 10
    max_epoch = extract_epoch(gen_files[-1])
    sample_epochs = {0}
    curr = 0
    step = 10  # 初始间隔
    inc = 10  # 间隔增量
    while curr <= max_epoch:
        sample_epochs.add(curr)
        curr += step
        step += inc

    video_writer_implant = None

    output_implant = args.output.replace('.mp4', '_implant.mp4')
    output_image = args.output.replace('.mp4', '_summary.png')

    # 用于拼接最终摘要长图的帧列表
    # 用户要求：术后紧随术前之后 [术前, 术后, 0, 10, 30...]
    summary_frames = [cond_bgr_labeled, gt_implant_bgr_labeled]

    for gen_file in tqdm(gen_files, desc="Processing frames"):
        epoch = extract_epoch(gen_file)

        gen_img_raw = sitk.GetArrayFromImage(sitk.ReadImage(str(gen_file)))
        gen_img = center_crop_3d(gen_img_raw, cond_img.shape)

        gen_raw_slice = np.flipud(gen_img[:, idx, :])

        # 制作植入效果的生成图 (使用参考 v2/infer.py 的融合函数)
        gen_fused_raw = fuse_sdf_to_ct(gen_raw_slice, cond_raw_slice)
        gen_implant_slice = normalize_image(gen_fused_raw)

        gen_implant_bgr = cv2.cvtColor(gen_implant_slice, cv2.COLOR_GRAY2BGR)
        gen_implant_bgr_labeled = add_label_with_pil(gen_implant_bgr.copy(), f'Epoch {epoch}')

        # 收集用于摘要长图的帧 (按照采样列表选择)
        if epoch in sample_epochs:
            summary_frames.append(gen_implant_bgr_labeled)

        # 拼接帧: 术前 | 生成结果(融合) | 术后真实(融合)
        combined_frame_implant = np.hstack((cond_bgr_labeled, gen_implant_bgr_labeled, gt_implant_bgr_labeled))

        if video_writer_implant is None:
            video_writer_implant = imageio.get_writer(output_implant, format='FFMPEG', fps=args.fps, codec='libx264',
                                                      quality=8)

        # Convert BGR to RGB for imageio
        video_writer_implant.append_data(cv2.cvtColor(combined_frame_implant, cv2.COLOR_BGR2RGB))

    if video_writer_implant is not None:
        video_writer_implant.close()
        print(f"Video saved successfully to:\n{output_implant}")

    # 保存摘要长图
    summary_image = np.hstack(summary_frames)
    cv2.imwrite(output_image, summary_image)
    print(f"Summary image saved to: {output_image}")


if __name__ == '__main__':
    main()
