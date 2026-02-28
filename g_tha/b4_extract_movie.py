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
    """Normalize numpy array to 0-255 uint8"""
    img_min = img_arr.min()
    img_max = img_arr.max()
    if img_max > img_min:
        img_arr = (img_arr - img_min) / (img_max - img_min) * 255.0
    else:
        img_arr = np.zeros_like(img_arr)
    return img_arr.astype(np.uint8)


def add_label_with_pil(img_bgr, text):
    """使用 PIL 在图像底部居左添加中文标签，保留左边距"""
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # 尝试加载中文字体，如果失败则回退到默认
    try:
        # Windows 常用的中文字体
        font = ImageFont.truetype("msyh.ttc", 9)
    except Exception:
        try:
            font = ImageFont.truetype("simhei.ttf", 9)
        except Exception:
            font = ImageFont.load_default()

    # 获取文字大小
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    img_width, img_height = img_pil.size
    
    # 居左并留 10 像素边距
    x = 10
    # 放在底部边缘上方约 10 像素处
    y = img_height - text_height - 10
    
    # 画黑色背景边框以便看清文字
    draw.rectangle([x - 2, y - 2, x + text_width + 2, y + text_height + 2], fill=(0, 0, 0))
    # 写绿字
    draw.text((x, y), text, font=font, fill=(0, 255, 0))
    
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def main():
    parser = argparse.ArgumentParser(description="Extract validation NIfTI images and create an MP4 movie.")
    parser.add_argument('--log_dir', type=str, required=True, help='Path to the specific tensorboard log directory')
    parser.add_argument('--output', type=str, default='training_progress.mp4', help='Output MP4 file path')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second for the video')
    args = parser.parse_args()

    log_path = Path(args.log_dir)
    if not log_path.exists():
        raise FileNotFoundError(f"Log directory not found: {log_path}")

    # 1. Load Condition and GroundTruth (should be from Epoch 0)
    cond_path = log_path / 'val_epoch_000_Condition.nii.gz'
    gt_path = log_path / 'val_epoch_000_GroundTruth.nii.gz'

    if not cond_path.exists() or not gt_path.exists():
        raise FileNotFoundError(f"Condition or GroundTruth file missing in {log_path}")

    cond_img = sitk.GetArrayFromImage(sitk.ReadImage(str(cond_path)))
    gt_img = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_path)))

    # Get middle slice index along the depth axis (usually axis 1 or 0 depending on data shape, assuming [D, H, W] after sitk read)
    # sitk reads as [Z, Y, X]. We assume depth is the first dimension.
    # Note: the prompt says [:, idx, :] which implies the middle axis is depth, but let's check shape
    
    print(f"Volume shape: {cond_img.shape}")
    
    # 根据用户要求的 [:, idx, :]，索引应该在第 1 个维度 (y轴/冠状面)
    # sitk 读入通常是 (Z, Y, X)。所以 idx 对应 Y 轴。
    idx = 67

    # 获取未归一化的原始切片并上下翻转
    cond_raw_slice = np.flipud(cond_img[:, idx, :])
    gt_raw_slice = np.flipud(gt_img[:, idx, :])

    # 归一化用于显示的切片
    cond_slice = normalize_image(cond_raw_slice)
    gt_slice = normalize_image(gt_raw_slice)

    # 生成一个合成的 GroundTruth 切片，营造出一种假体植入到原骨骼背景中的效果
    # 假体区域（>= -0.99）取 gt_slice 和 cond_slice 的最大值（灰度）
    gt_implant_slice = cond_slice.copy()
    fg_mask_gt = gt_raw_slice >= -0.99
    gt_implant_slice[fg_mask_gt] = np.maximum(gt_slice[fg_mask_gt], cond_slice[fg_mask_gt])

    # Convert to BGR for adding text and coloring
    cond_bgr = cv2.cvtColor(cond_slice, cv2.COLOR_GRAY2BGR)
    gt_bgr = cv2.cvtColor(gt_slice, cv2.COLOR_GRAY2BGR)
    gt_implant_bgr = cv2.cvtColor(gt_implant_slice, cv2.COLOR_GRAY2BGR)

    cond_bgr_pure = cond_bgr.copy()
    cond_slice_pure = cond_slice.copy()

    # Add labels
    cond_bgr = add_label_with_pil(cond_bgr, '术前条件')
    gt_bgr = add_label_with_pil(gt_bgr, '术后真实')
    gt_implant_bgr = add_label_with_pil(gt_implant_bgr, '术后真实')

    # 2. Find all Generated NIfTI files and sort them by epoch
    gen_files = list(log_path.glob('val_epoch_*_Generated.nii.gz'))
    
    def extract_epoch(filepath):
        match = re.search(r'val_epoch_(\d+)_Generated', filepath.name)
        return int(match.group(1)) if match else -1

    gen_files.sort(key=extract_epoch)

    if not gen_files:
        raise ValueError("No generated NIfTI files found.")

    print(f"Found {len(gen_files)} generated images. Creating video...")

    video_writer_standard = None
    video_writer_implant = None
    
    output_implant = args.output.replace('.mp4', '_implant.mp4')
    output_image = args.output.replace('.mp4', '_summary.png')

    # 用于拼接最终摘要长图的帧列表
    summary_frames = [cond_bgr]

    for gen_file in tqdm(gen_files, desc="Processing frames"):
        epoch = extract_epoch(gen_file)
        
        gen_img = sitk.GetArrayFromImage(sitk.ReadImage(str(gen_file)))
        gen_raw_slice = np.flipud(gen_img[:, idx, :])
        gen_slice = normalize_image(gen_raw_slice)
        
        gen_bgr = cv2.cvtColor(gen_slice, cv2.COLOR_GRAY2BGR)
        
        # 制作植入效果的生成图 (灰度覆盖)
        gen_implant_slice = cond_slice_pure.copy()
        fg_mask_gen = gen_raw_slice >= -0.99
        gen_implant_slice[fg_mask_gen] = np.maximum(gen_slice[fg_mask_gen], cond_slice_pure[fg_mask_gen])
        
        gen_implant_bgr = cv2.cvtColor(gen_implant_slice, cv2.COLOR_GRAY2BGR)
        
        gen_bgr = add_label_with_pil(gen_bgr, f'生成训练 {epoch}')
        gen_implant_bgr_labeled = add_label_with_pil(gen_implant_bgr.copy(), f'生成训练 {epoch}')

        # 收集用于摘要长图的帧 (每 50 个 epoch)
        if epoch > 0 and epoch % 50 == 0:
            summary_frames.append(gen_implant_bgr_labeled)

        # Concatenate horizontally
        combined_frame_standard = np.hstack((cond_bgr, gen_bgr, gt_bgr))
        combined_frame_implant = np.hstack((cond_bgr, gen_implant_bgr_labeled, gt_implant_bgr))

        if video_writer_standard is None:
            video_writer_standard = imageio.get_writer(args.output, format='FFMPEG', fps=args.fps, codec='libx264', quality=8)
            video_writer_implant = imageio.get_writer(output_implant, format='FFMPEG', fps=args.fps, codec='libx264', quality=8)

        # Convert BGR to RGB for imageio
        video_writer_standard.append_data(cv2.cvtColor(combined_frame_standard, cv2.COLOR_BGR2RGB))
        video_writer_implant.append_data(cv2.cvtColor(combined_frame_implant, cv2.COLOR_BGR2RGB))

    if video_writer_standard is not None:
        video_writer_standard.close()
        video_writer_implant.close()
        print(f"Videos saved successfully to:\n1. {args.output}\n2. {output_implant}")

    # 保存摘要长图
    summary_frames.append(gt_implant_bgr)
    summary_image = np.hstack(summary_frames)
    cv2.imwrite(output_image, summary_image)
    print(f"Summary image saved to: {output_image}")

if __name__ == '__main__':
    main()
