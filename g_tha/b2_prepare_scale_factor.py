import argparse
from contextlib import nullcontext
from pathlib import Path

import tomlkit
import torch
from monai.data import DataLoader, Dataset
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
    # cache_dir = train_root / 'cache'
    ckpt_dir = train_root / 'checkpoints'

    task = 'vae'
    (
        use_amp, num_workers, patch_size,
    ) = [cfg['train'][task][_] for _ in (
        'use_amp', 'num_workers', 'patch_size',
    )]
    load_pt = ckpt_dir / f'{task}_best.pt'

    train_files = [{'image': p.as_posix()} for p in (dataset_root / 'pre' / 'train').glob('*.nii.gz')]
    train_files += [{'image': p.as_posix()} for p in (dataset_root / 'post' / 'train').glob('*.nii.gz')]
    print(f'Train: {len(train_files)}')

    train_transforms = Compose(define.vae_train_transforms(patch_size))

    train_ds = Dataset(train_files, train_transforms)
    train_loader = DataLoader(
        train_ds, batch_size=32, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True,
        prefetch_factor=2,
    )

    vae = define.vae_kl().to(device)

    try:
        print(f'Loading checkpoint from {load_pt}...')
        checkpoint = torch.load(load_pt, map_location=device)
        vae.load_state_dict(checkpoint['state_dict'])
    except Exception:
        raise SystemExit(f'Failed to load checkpoint: {load_pt}') from None

    for _ in ('epoch', 'best_val_ssim', 'val_l1', 'val_psnr', 'val_ssim', 'scale_factor'):
        print(_, checkpoint.get(_))

    vae.eval()

    print('Calculating Robust Latent Scale Factor...')

    # 用于计算全局 Mean 和 Std 的统计量
    channel_sum = torch.zeros(4, device=device)
    channel_squared_sum = torch.zeros(4, device=device)
    num_pixels = 0

    with torch.no_grad():
        amp_ctx = autocast(device.type) if use_amp else nullcontext()
        with amp_ctx:
            for batch in tqdm(train_loader):
                images = batch['image'].to(device)

                # 编码获取 Latent Mean
                z_mu, _ = vae.encode(images)

                # [可选] 简单的背景过滤：只统计非背景区域
                # 假设背景在 Latent 空间的值接近 0 或某个常数
                # 这里我们使用简单的阈值策略，或者你可以保留全图计算但使用正确的全局公式
                # mask = (torch.abs(z_mu) > 0.01).any(dim=1, keepdim=True)
                # if mask.sum() == 0: continue
                # z_mu = z_mu * mask # 这不仅是过滤，需要只选择 mask 区域，比较复杂，暂且用全局公式修复

                # 累加统计量 (按通道或全局均可，这里做全局统计)
                # 展平除 Batch/Channel 外的维度
                z_flat = z_mu.detach()

                channel_sum += torch.sum(z_flat, dim=(0, 2, 3, 4))
                channel_squared_sum += torch.sum(z_flat ** 2, dim=(0, 2, 3, 4))
                num_pixels += (z_flat.numel() / z_flat.shape[1])  # 像素总数 (Batch * D * H * W)

    # 计算真实的全局 Mean 和 Std
    # Var[X] = E[X^2] - (E[X])^2
    global_mean = channel_sum / num_pixels
    global_var = (channel_squared_sum / num_pixels) - (global_mean ** 2)
    global_std = torch.sqrt(global_var).mean()  # 对4个通道的 std 求平均

    scale_factor = 1.0 / global_std.item()

    # Save scale factor
    checkpoint['scale_factor'] = scale_factor
    torch.save(checkpoint, load_pt)

    print(f'Global Mean per channel: {global_mean.cpu().numpy()}')
    print(f'Global Std per channel: {torch.sqrt(global_var).cpu().numpy()}')
    print(f'Final Robust Scale Factor: {scale_factor}')

    return scale_factor


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Keyboard interrupted terminating...')
