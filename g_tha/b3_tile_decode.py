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


def decode_with_sliding_window(z, autoencoder, scale_factor, patch_size, original_shape, overlap=0.25):
    """
    Mode B 的核心：在潜在空间进行滑动窗口解码
    """
    # 1. 还原 Scale
    z_unscaled = z / scale_factor

    # 2. 计算对齐后的空间尺寸 (Latent Space Spatial Dim * 4)
    # 模型是4倍下采样，所以 Latent 的每个像素对应 4 个原图像素
    aligned_shape = [s * 4 for s in z.shape[2:]]

    # 3. 上采样 Trick: [B, 4, D, H, W] -> [B, 4, 4D, 4H, 4W]
    z_upsampled = F.interpolate(z_unscaled, size=aligned_shape, mode='nearest')

    # 4. 定义适配器
    def decode_predictor_wrapper(z_up_patch):
        # Input: [B, 4, 128, 128, 128] (伪装成 Image 大小的 Latent Patch)
        # Downsample back to latent size: 128 -> 32 (除以4)
        z_patch = F.interpolate(z_up_patch, scale_factor=0.25, mode='nearest')
        # Decode: [B, 4, 32, 32, 32] -> [B, 1, 128, 128, 128]
        return autoencoder.decode(z_patch)

    # 5. 滑动窗口解码
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

    # 6. 恢复原始尺寸 (处理 padding 或 rounding 带来的微小差异)
    if recon_aligned.shape[2:] != original_shape:
        final_img = F.interpolate(recon_aligned, size=original_shape, mode='trilinear', align_corners=False)
    else:
        final_img = recon_aligned

    return final_img


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

    task = 'autoencoder'
    patch_size = cfg['train'][task]['patch_size']
    use_amp = cfg['train'][task]['use_amp']

    print(f"Main computation device: {device}")
    autoencoder = define.autoencoder().to(device)
    load_pt = ckpt_dir / f'{task}_best.pt'

    try:
        print(f'Loading checkpoint from {load_pt}...')
        checkpoint = torch.load(load_pt, map_location=device)
        autoencoder.load_state_dict(checkpoint['state_dict'])
        scale_factor = checkpoint.get('scale_factor', 1.0)
        print(f'Loaded Scale Factor: {scale_factor}')
    except Exception as e:
        raise SystemExit(f'Failed to load checkpoint: {e}')

    autoencoder.eval()
    l1_loss_func = torch.nn.L1Loss()

    pre_files = list((dataset_root / 'post' / 'train').glob('*.nii.gz'))
    if not pre_files:
        print("No files found.")
        return

    # 选取一个样本进行详细测试
    sample_file = pre_files[1]
    print(f'Target Sample: {sample_file.name}')

    base_transforms = Compose(define.autoencoder_val_transforms())

    # 准备 Saver
    saver_a = SaveImage(output_dir=output_dir, output_postfix='mode_a', separate_folder=False, print_log=False)
    saver_b = SaveImage(output_dir=output_dir, output_postfix='mode_b', separate_folder=False, print_log=False)
    saver_c = SaveImage(output_dir=output_dir, output_postfix='mode_c_full', separate_folder=False, print_log=False)

    # 预测器定义
    def encode_decode_predictor(inputs):
        z_mu, _ = autoencoder.encode(inputs)
        return autoencoder.decode(z_mu)

    def encode_predictor(inputs):
        return autoencoder.encode(inputs)[0]

    # 加载数据
    data = base_transforms({'image': sample_file.as_posix()})

    # 处理 numpy -> tensor
    if isinstance(data['image'], np.ndarray):
        data['image'] = torch.from_numpy(data['image'])

    images = data['image'].unsqueeze(0).to(device)  # [1, 1, D, H, W]
    images_cpu = images.cpu()
    original_shape = images.shape[2:]

    print(f"\nOriginal Shape: {original_shape}")

    # =========================================================
    # Mode A: Image Space Stitching (Baseline) - GPU
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
    l1_a = l1_loss_func(recon_a.cpu(), images_cpu).item()
    saver_a(recon_a.cpu()[0, 0:1], meta_data=data.get('image_meta_dict'))
    print(f'>> Mode A L1: {l1_a:.6f}')

    # =========================================================
    # Mode B: Latent Space Stitching (Simulation of LDM) - GPU
    # =========================================================
    print("\n[Mode B] Running Latent Space Pipeline (GPU)...")
    with torch.no_grad():
        amp_ctx = autocast(device.type) if use_amp else torch.no_grad()
        with amp_ctx:
            # 1. Get Full Latent
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
            # 2. Scale
            z = z_mu * scale_factor

            # 3. Simulate Storage
            z_loaded = z

            # 4. Decode with tiling
            recon_b = decode_with_sliding_window(
                z_loaded,
                autoencoder,
                scale_factor,
                patch_size=patch_size,
                original_shape=original_shape,
                overlap=0.25
            )

    l1_b = l1_loss_func(recon_b.cpu(), images_cpu).item()
    saver_b(recon_b.cpu()[0, 0:1], meta_data=data.get('image_meta_dict'))
    print(f'>> Mode B L1: {l1_b:.6f}')

    # =========================================================
    # Mode C: Full Image Direct Inference (Gold Standard) - CPU
    # =========================================================
    print("\n[Mode C] Running Full Image Direct Inference on CPU (Gold Standard)...")
    print(">> Moving model to CPU and clearing GPU cache...")

    # 1. Move everything to CPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    autoencoder = autoencoder.cpu()
    images_in_cpu = images.cpu()

    try:
        with torch.no_grad():
            # CPU 不需要 AMP，也不支持 CUDA 的 autocast，直接跑 float32
            # 1. Pad to be divisible by 4
            padder = DivisiblePad(k=4)
            images_padded = padder(images_in_cpu[0]).unsqueeze(0)  # Keep on CPU

            # 2. Direct Forward (No tiling)
            # 这里的 autoencoder 已经在 CPU 上了
            z_mu_full, _ = autoencoder.encode(images_padded)
            recon_c_padded = autoencoder.decode(z_mu_full)

            # 3. Inverse Pad (Crop back)
            recon_c = padder.inverse(recon_c_padded[0]).unsqueeze(0)

        l1_c = l1_loss_func(recon_c, images_cpu).item()
        saver_c(recon_c[0, 0:1], meta_data=data.get('image_meta_dict'))
        print(f'>> Mode C L1: {l1_c:.6f}')

        # Move model back to GPU just in case (good practice)
        autoencoder = autoencoder.to(device)

    except Exception as e:
        print(f">> Mode C Failed with error: {e}")
        l1_c = None

    # =========================================================
    # Final Comparison
    # =========================================================
    print("\n--- Final Report ---")
    print(f"Mode A (Image Tile) : {l1_a:.6f}")
    print(f"Mode B (Latent Tile): {l1_b:.6f}")
    if l1_c is not None:
        print(f"Mode C (Full Direct): {l1_c:.6f}")

        diff_ac = abs(l1_a - l1_c)
        diff_bc = abs(l1_b - l1_c)

        print(f"\nGap A vs C (Tiling Error) : {diff_ac:.6f}")
        print(f"Gap B vs C (Latent Error) : {diff_bc:.6f}")

        if diff_ac < 0.01:  # 放宽一点点标准，因为 CPU/GPU 浮点精度可能略有差异
            print(">> Tiling strategy is EXCELLENT (Consistent with full inference).")
        else:
            print(">> Tiling introduces visible divergence.")
    else:
        print(">> Could not compare with Mode C.")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass