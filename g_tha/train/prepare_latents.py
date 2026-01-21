from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from monai.networks.nets import AutoencoderKL
from monai.transforms import LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged, Compose
from torch.amp import autocast
from tqdm import tqdm

try:
    import torch_musa

    device = torch.device('musa' if torch.musa.is_available() else 'cpu')
except ImportError:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    use_amp = False
    use_checkpoint = False

    # --- 配置 ---
    root = Path('.ds')
    task = 'post'
    load_pt = root / 'checkpoints' / 'autoencoder_best.pt'

    ct_range = (-200, 2800)

    # --- 1. 加载数据 ---
    files = list((root / task).rglob('*.nii.gz'))
    print(f'Found {len(files)} files to process.')

    transforms = Compose([
        LoadImaged(keys=['image']),
        EnsureChannelFirstd(keys=['image'], channel_dim='no_channel'),
        ScaleIntensityRanged(keys=['image'], a_min=ct_range[0], a_max=ct_range[1], b_min=-1.0, b_max=1.0, clip=True),
    ])

    # --- 2. 加载模型 ---
    model = AutoencoderKL(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        num_res_blocks=(2, 2, 2, 2),
        channels=(32, 64, 64, 64),
        attention_levels=(False, False, False, False),
        latent_channels=3,
        norm_num_groups=32,
        use_checkpoint=use_checkpoint,
    ).to(device)

    try:
        print(f'Loading checkpoint from {load_pt}...')
        checkpoint = torch.load(load_pt, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        print(f'Load from epoch {start_epoch}')
    except Exception:
        raise SystemExit(f'Failed to load checkpoint: {load_pt}') from None

    for _ in ('epoch', 'val_loss', 'best_val_loss', 'val_psnr', 'val_ssim', 'scale_factor'):
        print(_, checkpoint.get(_))

    try:
        scale_factor = checkpoint['scale_factor']
    except KeyError:
        raise SystemExit(f'No scale_factor in {load_pt}') from None

    model.eval()

    # --- 3. 处理并保存 ---
    with torch.no_grad():
        amp_ctx = autocast(device.type) if use_amp else nullcontext()
        with amp_ctx:
            for f in tqdm(files):
                # 加载并预处理
                data = transforms({'image': str(f)})
                img_tensor = data['image'].unsqueeze(0).to(device)  # (1, 1, H, W, D)

                # 编码 (Encode)
                # model.encode 返回的是分布参数 (mu, logvar)
                # 我们只需要均值 (mu) 作为 Latent 表示，这也是 standard practice
                z_mu, _ = model.encode(img_tensor)

                # 重要：乘以缩放系数
                z = z_mu * scale_factor

                # 转移到 CPU 并保存
                z_np = z.squeeze(0).cpu().numpy().astype(np.float16)  # (16, H', W', D')

                # 保存为 .npy 格式，文件名保持一致
                save_path = root / f'{task}_latents' / f.relative_to(root / task)
                save_path = save_path.parent / (save_path.name.removesuffix('.nii.gz') + '.npy')
                save_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(save_path, z_np)

    print(f'Latent shape example: {z_np.shape}')


if __name__ == '__main__':
    main()
