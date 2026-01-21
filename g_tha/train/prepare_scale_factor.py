from contextlib import nullcontext
from pathlib import Path

import torch
from monai.data import DataLoader, PersistentDataset
from monai.networks.nets import AutoencoderKL
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged, RandSpatialCropd
)
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

    batch_size = 1
    num_workers = 8

    root = Path('.ds')
    task = 'post'
    cache_dir = root / 'cache'
    load_pt = root / 'checkpoints' / 'autoencoder_best.pt'

    patch_size = (96,) * 3
    ct_range = (-200, 2800)

    # --- 1. 数据准备 (保持原逻辑) ---
    train_files = [{'image': str(p)} for p in (root / task / 'train').glob('*.nii.gz')]
    print(f'Train: {len(train_files)}')

    base_transforms = [
        LoadImaged(keys=['image']),
        EnsureChannelFirstd(keys=['image'], channel_dim='no_channel'),
        ScaleIntensityRanged(keys=['image'], a_min=ct_range[0], a_max=ct_range[1], b_min=-1.0, b_max=1.0, clip=True),
    ]

    train_transforms = Compose(base_transforms + [
        RandSpatialCropd(keys=['image'], roi_size=patch_size, random_size=False),
    ])

    train_ds = PersistentDataset(train_files, train_transforms, cache_dir)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

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
    except Exception:
        raise SystemExit(f'Failed to load checkpoint: {load_pt}') from None

    for _ in ('epoch', 'val_loss', 'best_val_loss', 'val_psnr', 'val_ssim', 'scale_factor'):
        print(_, checkpoint.get(_))

    model.eval()
    running_std = 0.0
    count = 0
    print('Calculating Latent Scale Factor for LDM...')

    with torch.no_grad():
        amp_ctx = autocast(device.type) if use_amp else nullcontext()
        with amp_ctx:
            for batch in tqdm(train_loader):
                images = batch['image'].to(device)
                # 获取 latent 分布
                z_mu, z_logvar = model.encode(images)
                # 采样得到 z
                z = model.sampling(z_mu, z_logvar)
                running_std += z.std().item()
                count += 1

    scale_factor = 1.0 / (running_std / count)

    # Save scale factor
    checkpoint['scale_factor'] = scale_factor
    torch.save(checkpoint, load_pt)

    print(f'Recommended Scale Factor: {scale_factor}')
    # 后续训练 LDM 时： z = z * scale_factor
    return scale_factor


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Keyboard interrupted terminating...')
