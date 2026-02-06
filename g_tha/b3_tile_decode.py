import argparse
from pathlib import Path
import torch.nn.functional as F  # 引入 F 用于插值

import numpy as np
import tomlkit
import torch
from monai.inferers import sliding_window_inference
from monai.transforms import Compose, SaveImage
from torch.amp import autocast

import define

try:
    import torch_musa

    device = torch.device('musa' if torch.musa.is_available() else 'cpu')
except ImportError:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def decode_with_sliding_window(z, autoencoder, scale_factor, patch_size, overlap=0.5):
    """
    使用 MONAI sliding_window_inference 进行解码。
    利用 'Upsample Trick' 解决 Latent->Image 尺寸不匹配问题，
    从而获得 Overlap + Gaussian Blending 的平滑拼接效果。
    """
    # 1. 还原 Scale
    z_unscaled = z / scale_factor

    # 2. 上采样 Trick: 把 Latent 放大 8 倍，伪装成 Image 大小
    # z_unscaled: [B, 4, D, H, W] -> z_upsampled: [B, 4, 8D, 8H, 8W]
    z_upsampled = F.interpolate(z_unscaled, scale_factor=8.0, mode='nearest')

    # 3. 定义适配器 predictor
    def decode_predictor_wrapper(z_up_patch):
        # z_up_patch 是 MONAI 切出来的 [B, 4, 128, 128, 128]
        # 我们把它缩放回 Latent 尺寸 [B, 4, 16, 16, 16]
        z_patch = F.interpolate(z_up_patch, scale_factor=0.125, mode='nearest')

        # 解码: [B, 4, 16, 16, 16] -> [B, 2, 128, 128, 128]
        return autoencoder.decode(z_patch)

    # 4. 让 MONAI 在图像空间做滑动窗口
    recon_img = sliding_window_inference(
        inputs=z_upsampled,  # 喂给它伪装的大图
        roi_size=patch_size,  # [128, 128, 128]
        sw_batch_size=4,
        predictor=decode_predictor_wrapper,
        overlap=overlap,  # 这里可以使用 0.5 或更高
        mode='gaussian',  # 开启高斯混合，消除接缝
        device=device,
        progress=True
    )

    return recon_img


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
    ct_range = cfg['train'][task]['ct_range']
    bone_range = cfg['train'][task]['bone_range']
    use_amp = cfg['train'][task]['use_amp']

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

    sample_files = []
    pre_files = list((dataset_root / 'post' / 'train').glob('*.nii.gz'))
    if pre_files:
        sample_files.append({'image': pre_files[1].as_posix(), 'id': 'pre_sample'})

    base_transforms = Compose(define.autoencoder_base_transforms(ct_range, bone_range))

    saver_a = SaveImage(output_dir=output_dir, output_postfix='mode_a', separate_folder=False, print_log=False)
    saver_b = SaveImage(output_dir=output_dir, output_postfix='mode_b_sliding', separate_folder=False, print_log=False)

    # Mode A Predictor
    def encode_decode_predictor(inputs):
        z_mu, _ = autoencoder.encode(inputs)
        return autoencoder.decode(z_mu)

    # Mode B Predictor
    def encode_predictor(inputs):
        return autoencoder.encode(inputs)[0]

    for item in sample_files:
        print(f'\n--- Processing {item["id"]} ---')
        data = base_transforms(item)
        images = data['image'].unsqueeze(0).to(device)
        images_cpu = images.cpu()

        # =========================================================
        # Mode A: Image Stitching (基准)
        # =========================================================
        print("Running Mode A: Image Space Stitching...")
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
        recon_a = recon_a.cpu()
        l1_a = l1_loss_func(recon_a, images_cpu).item()
        l1_a_ct = l1_loss_func(recon_a[:, 0:1], images_cpu[:, 0:1]).item()
        print(f'>> Mode A Total L1: {l1_a:.6f} | CT-Only L1: {l1_a_ct:.6f}')
        saver_a(recon_a[0, 0:1], meta_data=data.get('image_meta_dict'))

        # =========================================================
        # Mode B: Latent Stitching + Sliding Window Decode (终极测试)
        # =========================================================
        print("\nRunning Mode B: Latent Encode -> Save -> Sliding Window Decode...")

        # 1. 编码 (得到全图 Latent)
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

        # 2. 模拟保存再读取 (确保 NPY 没问题)
        latent_np = z.cpu().numpy()
        np.save(output_dir / 'temp_verify.npy', latent_np)
        z_loaded = torch.from_numpy(np.load(output_dir / 'temp_verify.npy')).to(device)

        # 3. 解码 (使用 Sliding Window Upsample Trick)
        # 这里我们用 overlap=0.5 来获得更好的平滑度
        with torch.no_grad():
            recon_b = decode_with_sliding_window(
                z_loaded,
                autoencoder,
                scale_factor,
                patch_size=patch_size,
                overlap=0.25
            )

        recon_b = recon_b.cpu()
        l1_b = l1_loss_func(recon_b, images_cpu).item()
        l1_b_ct = l1_loss_func(recon_b[:, 0:1], images_cpu[:, 0:1]).item()

        print(f'>> Mode B Total L1: {l1_b:.6f} | CT-Only L1: {l1_b_ct:.6f}')
        saver_b(recon_b[0, 0:1], meta_data=data.get('image_meta_dict'))

        if abs(l1_b - l1_a) < 0.02:
            print("\n[CONCLUSION] Mode B matches Mode A! Latent generation is perfect.")
        else:
            print(f"\n[CONCLUSION] Difference is {abs(l1_b - l1_a):.4f}. Check if overlap/gaussian settings match.")

    print('\nVerification Completed.')


if __name__ == '__main__':
    main()