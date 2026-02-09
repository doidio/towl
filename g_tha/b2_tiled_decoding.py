"""
验证 3D VAE 在不同推理模式下的重建质量，确立 Latent Diffusion 的数据预处理方案。

核心目的:
1. 验证 [Latent Space 拼接] 与 [Image Space 拼接] 的数学等价性，确保 LDM 训练数据的可靠性。
2. 通过对比 [整图推理] 与 [滑动窗口]，量化 "归一化层分布漂移 (Distribution Shift)" 对 3D 医学图像重建的破坏性影响。

包含三种推理模式：
--------------------------------------------------------------------------------
1. [Mode A] Image Space Sliding Window (基准 / Baseline):
   - 方式: 常规的 sliding_window_inference。
   - 意义: 代表模型在理想状态下的最佳重建能力。

2. [Mode B] Latent Space Sliding Window (拟采用方案 / Proposed):
   - 方式: 模拟 LDM 流程。先编码存为 Latent -> 读取 -> 缩放 -> 在 Latent 空间做滑动窗口解码。
   - 意义: 如果 Mode B ≈ Mode A，说明我们可以放心地把 TB 级 CT 数据压缩为 GB 级 Latent 供 LDM 训练。

3. [Mode C] Full Image Direct Inference (对照组 / Control):
   - 方式: 强制将整张 3D 卷一次性送入网络 (CPU 运行)。
   - 意义: 用于反证滑动窗口的必要性。

预期结果 (Expected Outcome):
--------------------------------------------------------------------------------
1. Mode A ≈ Mode B: L1 误差差异极小 (例如 < 1e-5)。这证明 Latent 压缩策略是无损的（相对于模型能力而言）。
2. Mode C >> Mode A: 整图推理误差巨大 (例如 > 0.1)。
3. Center L1 Analysis: 即使剔除边缘，Mode C 的中心误差依然很高。
   -> 结论: 误差源于 Patch 训练导致的归一化统计量偏移，而非卷积边界效应 (Padding)。
"""

import argparse
from pathlib import Path

import numpy as np
import tomlkit
import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from monai.transforms import Compose, SaveImage, DivisiblePad
from torch.amp import autocast

import define

try:
    import torch_musa

    device = torch.device('musa' if torch.musa.is_available() else 'cpu')
except ImportError:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def decode_with_sliding_window(z, vae, scale_factor, patch_size, original_shape, overlap=0.25):
    """
    Mode B 的核心：在潜在空间进行滑动窗口解码
    """
    z_unscaled = z / scale_factor
    aligned_shape = [s * 4 for s in z.shape[2:]]
    z_upsampled = F.interpolate(z_unscaled, size=aligned_shape, mode='nearest')

    def decode_predictor_wrapper(z_up_patch):
        z_patch = F.interpolate(z_up_patch, scale_factor=0.25, mode='nearest')
        return vae.decode(z_patch)

    recon_aligned = sliding_window_inference(
        inputs=z_upsampled,
        roi_size=patch_size,
        sw_batch_size=4,
        predictor=decode_predictor_wrapper,
        overlap=overlap,
        mode='gaussian',
        device=device,
        progress=True
    )

    if recon_aligned.shape[2:] != original_shape:
        final_img = F.interpolate(recon_aligned, size=original_shape, mode='trilinear', align_corners=False)
    else:
        final_img = recon_aligned

    return final_img


def compute_metrics_with_margin(pred, target, margin=32):
    """
    同时计算全局误差和中心区域误差（剔除边缘效应）
    Args:
        margin: 剔除边缘的像素数 (32像素足以覆盖深层卷积的感受野边界)
    """
    l1_func = torch.nn.L1Loss()

    # 1. 全局 L1
    global_l1 = l1_func(pred, target).item()

    # 2. 中心 L1 (剔除边界)
    # 检查尺寸是否足够裁剪
    d, h, w = pred.shape[2:]
    if d > 2 * margin and h > 2 * margin and w > 2 * margin:
        pred_center = pred[..., margin:-margin, margin:-margin, margin:-margin]
        target_center = target[..., margin:-margin, margin:-margin, margin:-margin]
        center_l1 = l1_func(pred_center, target_center).item()
        valid_center = True
    else:
        center_l1 = global_l1  # 尺寸太小，无法裁剪，退化为全局
        valid_center = False

    return global_l1, center_l1, valid_center


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    cfg = tomlkit.loads(Path(args.config).read_text('utf-8')).unwrap()
    dataset_root = Path(cfg['dataset']['root'])
    train_root = Path(cfg['train']['root'])
    output_dir = train_root / 'latents_verification'
    ckpt_dir = train_root / 'checkpoints'
    output_dir.mkdir(parents=True, exist_ok=True)

    task = 'vae'
    patch_size = cfg['train'][task]['patch_size']
    use_amp = cfg['train'][task]['use_amp']

    print(f"Main computation device: {device}")
    vae = define.vae().to(device)
    load_pt = ckpt_dir / f'{task}_best.pt'

    try:
        print(f'Loading checkpoint from {load_pt}...')
        checkpoint = torch.load(load_pt, map_location=device)
        vae.load_state_dict(checkpoint['state_dict'])
        scale_factor = checkpoint.get('scale_factor', 1.0)
        print(f'Loaded Scale Factor: {scale_factor}')
    except Exception as e:
        raise SystemExit(f'Failed to load checkpoint: {e}')

    vae.eval()

    # 优先选取之前那个有问题的样本，或者第一个
    pre_files = list((dataset_root / 'post' / 'train').glob('*.nii.gz'))
    if not pre_files:
        print("No files found.")
        return
    sample_file = pre_files[1] if len(pre_files) > 1 else pre_files[0]
    print(f'Target Sample: {sample_file.name}')

    base_transforms = Compose(define.vae_val_transforms())

    # Savers
    saver_a = SaveImage(output_dir=output_dir, output_postfix='mode_a', separate_folder=False, print_log=False)
    saver_b = SaveImage(output_dir=output_dir, output_postfix='mode_b', separate_folder=False, print_log=False)
    saver_c = SaveImage(output_dir=output_dir, output_postfix='mode_c_full', separate_folder=False, print_log=False)

    def encode_decode_predictor(inputs):
        z_mu, _ = vae.encode(inputs)
        return vae.decode(z_mu)

    def encode_predictor(inputs):
        return vae.encode(inputs)[0]

    # 加载数据
    data = base_transforms({'image': sample_file.as_posix()})
    if isinstance(data['image'], np.ndarray):
        data['image'] = torch.from_numpy(data['image'])

    images = data['image'].unsqueeze(0).to(device)
    images_cpu = images.cpu()
    original_shape = images.shape[2:]
    print(f"\nOriginal Shape: {original_shape}")

    # 用于记录结果
    results = {}

    # =========================================================
    # Mode A: Image Space Stitching
    # =========================================================
    print("\n[Mode A] Running Image Space Sliding Window (GPU)...")
    with torch.no_grad():
        amp_ctx = autocast(device.type) if use_amp else torch.no_grad()
        with amp_ctx:
            recon_a = sliding_window_inference(
                inputs=images,
                roi_size=patch_size,
                sw_batch_size=4,
                predictor=encode_decode_predictor,
                overlap=0.25,
                mode='gaussian',
                device=device,
                progress=True
            )

    g_l1, c_l1, _ = compute_metrics_with_margin(recon_a.cpu(), images_cpu, margin=32)
    results['A'] = {'global': g_l1, 'center': c_l1}
    saver_a(recon_a.cpu()[0, 0:1], meta_data=data.get('image_meta_dict'))
    print(f'>> Mode A | Global L1: {g_l1:.6f} | Center L1: {c_l1:.6f}')

    # =========================================================
    # Mode B: Latent Space Stitching
    # =========================================================
    print("\n[Mode B] Running Latent Space Pipeline (GPU)...")
    with torch.no_grad():
        amp_ctx = autocast(device.type) if use_amp else torch.no_grad()
        with amp_ctx:
            z_mu = sliding_window_inference(
                inputs=images,
                roi_size=patch_size,
                sw_batch_size=4,
                predictor=encode_predictor,
                overlap=0.25,
                mode='gaussian',
                device=device,
                progress=True
            )
            z = z_mu * scale_factor
            recon_b = decode_with_sliding_window(z, vae, scale_factor, patch_size, original_shape, overlap=0.25)

    g_l1, c_l1, _ = compute_metrics_with_margin(recon_b.cpu(), images_cpu, margin=32)
    results['B'] = {'global': g_l1, 'center': c_l1}
    saver_b(recon_b.cpu()[0, 0:1], meta_data=data.get('image_meta_dict'))
    print(f'>> Mode B | Global L1: {g_l1:.6f} | Center L1: {c_l1:.6f}')

    # =========================================================
    # Mode C: Full Image Direct Inference
    # =========================================================
    print("\n[Mode C] Running Full Image Direct Inference on CPU...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    vae = vae.cpu()
    images_in_cpu = images.cpu()

    try:
        with torch.no_grad():
            padder = DivisiblePad(k=4)
            images_padded = padder(images_in_cpu[0]).unsqueeze(0)

            z_mu_full, _ = vae.encode(images_padded)
            recon_c_padded = vae.decode(z_mu_full)
            recon_c = padder.inverse(recon_c_padded[0]).unsqueeze(0)

        g_l1, c_l1, valid_c = compute_metrics_with_margin(recon_c, images_cpu, margin=32)
        results['C'] = {'global': g_l1, 'center': c_l1}
        saver_c(recon_c[0, 0:1], meta_data=data.get('image_meta_dict'))
        print(f'>> Mode C | Global L1: {g_l1:.6f} | Center L1: {c_l1:.6f}')

        vae = vae.to(device)  # Put back

    except Exception as e:
        print(f">> Mode C Failed: {e}")
        results['C'] = None

    # =========================================================
    # Comparison Analysis
    # =========================================================
    print("\n" + "=" * 50)
    print("      BOUNDARY EFFECT ANALYSIS (Margin=32px)")
    print("=" * 50)
    print(f"{'Mode':<12} | {'Global L1':<12} | {'Center L1':<12} | {'Edge Impact'}")
    print("-" * 50)

    for key, name in [('A', 'Image Tile'), ('B', 'Latent Tile'), ('C', 'Full Direct')]:
        if results[key]:
            g = results[key]['global']
            c = results[key]['center']
            # Edge Impact: 主要是看中心误差是否比全局误差显著更低
            # 如果 Center << Global，说明边缘误差很大
            impact = "Low" if abs(g - c) < 0.01 else "High"
            print(f"{name:<12} | {g:.6f}       | {c:.6f}       | {impact}")
        else:
            print(f"{name:<12} | {'FAILED':<12} | {'FAILED':<12} | -")

    print("-" * 50)

    if results['C']:
        diff_global = results['C']['global'] - results['A']['global']
        diff_center = results['C']['center'] - results['A']['center']

        print(f"\nAnalysis of Mode C Failure:")
        print(f"1. Global Error Increase: {diff_global:.6f}")
        print(f"2. Center Error Increase: {diff_center:.6f}")

        if diff_center > 0.05:
            print("\n[CONCLUSION]: Center Error is HIGH.")
            print("The model fails even in the center of the image.")
            print("-> Root cause is DISTRIBUTION SHIFT (Normalization), NOT boundary padding.")
        else:
            print("\n[CONCLUSION]: Center Error is LOW.")
            print("The error is concentrated at the edges.")
            print("-> Root cause is PADDING ARTIFACTS.")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
