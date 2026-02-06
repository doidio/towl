import argparse
from contextlib import nullcontext
from pathlib import Path

import tomlkit
import torch
from monai.data import DataLoader, PersistentDataset
from monai.transforms import Compose
from torch.amp import autocast
from tqdm import tqdm

import define

try:
    import torch_musa

    device = torch.device('musa' if torch.musa.is_available() else 'cpu')  # noqa
except ImportError:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = tomlkit.loads(cfg_path.read_text('utf-8')).unwrap()

    dataset_root = Path(cfg['dataset']['root'])
    train_root = Path(cfg['train']['root'])
    cache_dir = train_root / 'cache'
    ckpt_dir = train_root / 'checkpoints'

    task = 'autoencoder'
    (
        use_amp, num_workers, patch_size, ct_range, bone_range,
    ) = [cfg['train'][task][_] for _ in (
        'use_amp', 'num_workers', 'patch_size', 'ct_range', 'bone_range',
    )]
    load_pt = ckpt_dir / f'{task}_best.pt'

    train_files = [{'image': p.as_posix()} for p in (dataset_root / 'pre' / 'train').glob('*.nii.gz')]
    train_files += [{'image': p.as_posix()} for p in (dataset_root / 'post' / 'train').glob('*.nii.gz')]
    print(f'Train: {len(train_files)}')

    base_transforms = define.autoencoder_base_transforms(ct_range, bone_range)
    train_transforms = Compose(base_transforms + define.autoencoder_train_transforms(patch_size))

    train_ds = PersistentDataset(train_files, train_transforms, cache_dir)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=num_workers, pin_memory=True)

    autoencoder = define.autoencoder().to(device)

    try:
        print(f'Loading checkpoint from {load_pt}...')
        checkpoint = torch.load(load_pt, map_location=device)
        autoencoder.load_state_dict(checkpoint['state_dict'])
    except Exception:
        raise SystemExit(f'Failed to load checkpoint: {load_pt}') from None

    for _ in ('epoch', 'best_val_ssim', 'val_l1', 'val_psnr', 'val_ssim', 'scale_factor'):
        print(_, checkpoint.get(_))

    autoencoder.eval()

    running_std = 0.0
    count = 0
    print('Calculating Latent Scale Factor for LDM...')

    with torch.no_grad():
        amp_ctx = autocast(device.type) if use_amp else nullcontext()
        with amp_ctx:
            for batch in tqdm(train_loader):
                images = batch['image'].to(device)
                # 获取 latent 分布
                z_mu, _ = autoencoder.encode(images)
                running_std += z_mu.std().item()
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
