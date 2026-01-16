import random
from datetime import datetime
from math import ceil
from pathlib import Path

import torch
from monai.data import DataLoader, PersistentDataset
from monai.metrics import PSNRMetric, SSIMMetric
from monai.networks.nets import VQVAE
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged
)
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

if __name__ == '__main__':
    num_workers = 8
    end_epoch = 1000
    batch_size = 1
    val_interval = 1
    model_type = 'small'
    resume = True

    root = Path('.ds')
    task = 'post'
    cache_dir = root / 'cache'
    log_dir = root / 'logs'
    ckpt_dir = root / 'checkpoints'

    ct_range = (-200, 2800)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- 数据准备 ---
    all_files = [{'image': str(p)} for p in (root / task).glob('*.nii.gz')]
    random.shuffle(all_files)

    if len(all_files) == 0:
        raise SystemExit('Dataset is empty.')

    split_idx = ceil(len(all_files) * 0.85)
    train_files, val_files = all_files[:split_idx], all_files[split_idx:]
    print(f'Train: {len(train_files)}, Val: {len(val_files)}')

    # 2. Transforms
    train_transforms = Compose([
        LoadImaged(keys=['image']),
        EnsureChannelFirstd(keys=['image'], channel_dim='no_channel'),
        ScaleIntensityRanged(keys=['image'], a_min=ct_range[0], a_max=ct_range[1], b_min=0.0, b_max=1.0, clip=True),
    ])

    val_transforms = Compose([
        LoadImaged(keys=['image']),
        EnsureChannelFirstd(keys=['image'], channel_dim='no_channel'),
        ScaleIntensityRanged(keys=['image'], a_min=ct_range[0], a_max=ct_range[1], b_min=0.0, b_max=1.0, clip=True),
    ])

    # 3. Dataset & Loader
    train_ds = PersistentDataset(train_files, train_transforms, cache_dir)
    val_ds = PersistentDataset(val_files, val_transforms, cache_dir) if val_files else None

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=0) if val_ds else None

    # --- 模型与优化器 ---
    print('Model type:', model_type)
    if model_type == 'small':
        model = VQVAE(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(32, 64),
            num_res_layers=1,
            num_res_channels=(32, 64),
            downsample_parameters=((2, 4, 1, 1),) * 2,
            upsample_parameters=((2, 4, 1, 1, 0),) * 2,
            num_embeddings=128,
            embedding_dim=16,
        ).to(device)
    elif model_type == 'production':
        model = VQVAE(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(96, 96, 192),
            num_res_layers=3,
            num_res_channels=(96, 96, 192),
            downsample_parameters=((2, 4, 1, 1),) * 3,
            upsample_parameters=((2, 4, 1, 1, 0),) * 3,
            num_embeddings=256,
            embedding_dim=32,
        ).to(device)
    else:
        raise SystemExit('Unknown model type:', model_type)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    l1_loss = torch.nn.L1Loss()

    _ = log_dir / datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(log_dir=_.as_posix())

    resume_pt = (ckpt_dir / task / 'last.pt').resolve().absolute()
    if resume and resume_pt.exists():
        print(f'Loading checkpoint from {resume_pt}...')

        checkpoint = torch.load(resume_pt, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        print(f"Resuming training from epoch {start_epoch + 1}, Best Val Loss: {best_val_loss:.6f}")
    else:
        start_epoch = 0
        best_val_loss = float('inf')

    # --- 训练循环 ---
    try:
        for epoch in range(start_epoch, end_epoch):
            model.train()
            epoch_loss = 0
            step = 0

            pbar = tqdm(train_loader, ncols=100, desc=f'Epoch {epoch + 1}/{end_epoch}')

            for batch in pbar:
                step += 1
                images = batch['image'].to(device)
                optimizer.zero_grad(set_to_none=True)

                reconstruction, quantization_loss = model(images=images)
                reconstruction_loss = l1_loss(reconstruction.float(), images.float())
                loss = reconstruction_loss + quantization_loss

                if torch.isnan(loss):
                    raise SystemExit(f'NaN loss detected at step {step}')

                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

                # TensorBoard 记录
                global_step = epoch * len(train_loader) + step
                writer.add_scalar('train/total_loss', loss.item(), global_step)
                writer.add_scalar('train/recon_loss', reconstruction_loss.item(), global_step)
                writer.add_scalar('train/quant_loss', quantization_loss.item(), global_step)

            epoch_loss /= step
            writer.add_scalar('train/epoch_loss', epoch_loss, epoch)

            # --- 验证循环 ---
            if val_loader and (epoch + 1) % val_interval == 0:
                model.eval()
                val_loss = 0

                val_psnr_metric = PSNRMetric(max_val=1.0)
                val_ssim_metric = SSIMMetric(data_range=1.0, spatial_dims=3)

                with torch.no_grad():
                    for i, batch in enumerate(val_loader):
                        images = batch['image'].to(device)

                        reconstruction, quantization_loss = model(images=images)
                        recon_loss = l1_loss(reconstruction.float(), images.float())
                        val_loss += recon_loss.item()

                        val_psnr_metric(y_pred=reconstruction, y=images)
                        val_ssim_metric(y_pred=reconstruction, y=images)

                        # 保存中间切片到 TensorBoard (只存第一个 batch 的第一张图)
                        if i == 0:
                            img_tensor = images[0, 0]  # Shape: [D, H, W]
                            recon_tensor = reconstruction[0, 0]  # Shape: [D, H, W]

                            x_idx = img_tensor.shape[0] // 2
                            orig_sag = img_tensor[x_idx, :, :]
                            recon_sag = recon_tensor[x_idx, :, :]

                            y_idx = img_tensor.shape[1] // 2
                            orig_cor = img_tensor[:, y_idx, :]
                            recon_cor = recon_tensor[:, y_idx, :]

                            # 写入 TensorBoard，使用不同的 tag 区分
                            writer.add_image('val/Sagittal_Orig', orig_sag, epoch, dataformats='HW')
                            writer.add_image('val/Sagittal_Recon', recon_sag, epoch, dataformats='HW')

                            writer.add_image('val/Coronal_Orig', orig_cor, epoch, dataformats='HW')
                            writer.add_image('val/Coronal_Recon', recon_cor, epoch, dataformats='HW')

                val_loss /= len(val_loader)
                val_psnr = val_psnr_metric.aggregate().item()
                val_ssim = val_ssim_metric.aggregate().item()

                val_psnr_metric.reset()
                val_ssim_metric.reset()

                writer.add_scalar('val/loss', val_loss, epoch)
                writer.add_scalar('val/psnr', val_psnr, epoch)
                writer.add_scalar('val/ssim', val_ssim, epoch)
                print(f'Val Loss: {val_loss:.6f} | PSNR: {val_psnr:.4f} | SSIM: {val_ssim:.4f}')

                # 5. Checkpoint 保存策略优化
                checkpoint = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'best_val_psnr': val_psnr,
                    'best_val_ssim': val_ssim,
                }

                # 始终保存最新的
                _ = ckpt_dir / task / 'last.pt'
                _.parent.mkdir(parents=True, exist_ok=True)
                torch.save(checkpoint, _)

                # 保存最好的
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    _ = ckpt_dir / task / 'best.pt'
                    _.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(checkpoint, _)
                    print(f'New best model saved at epoch {epoch + 1}')

        writer.close()
        print('Training Completed.')
    except KeyboardInterrupt:
        raise SystemExit('Training interrupted')
