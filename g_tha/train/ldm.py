from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import default_collate

try:
    import torch_musa

    device = torch.device('musa' if torch.musa.is_available() else 'cpu')
except ImportError:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from monai.data import DataLoader, PersistentDataset, NumpyReader
from monai.inferers import LatentDiffusionInferer
from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped
)
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def main():
    use_amp = False
    resume = False

    num_workers = 8
    num_epochs = 1000
    batch_size = 2
    val_interval = 5

    lr = 2.5e-5

    # 路径配置
    root = Path('.ds')
    task = 'post'

    cache_dir = root / 'cache'
    log_dir = root / 'logs'
    ckpt_dir = root / 'checkpoints' / 'ldm'

    # Autoencoder 路径
    ae_ckpt_path = root / 'checkpoints' / 'autoencoder_best.pt'
    # Resume 路径 (默认读取最后一次保存的模型)
    resume_ckpt_path = ckpt_dir / 'ldm_last.pt'

    # --- 1. 数据准备 ---
    train_files = [{'image': p.as_posix()} for p in (root / f'{task}_latents' / 'train').glob('*.npy')]
    val_files = [{'image': p.as_posix()} for p in (root / f'{task}_latents' / 'val').glob('*.npy')]
    print(f'Train: {len(train_files)}, Val: {len(val_files)}')

    base_transforms = Compose([
        LoadImaged(keys=['image'], reader=NumpyReader),
        EnsureChannelFirstd(keys=['image'], channel_dim=0),
        EnsureTyped(keys=['image']),
    ])

    train_ds = PersistentDataset(train_files, base_transforms, cache_dir)
    val_ds = PersistentDataset(val_files, base_transforms, cache_dir)

    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, collate_fn=default_collate,
                              shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=num_workers, collate_fn=default_collate)

    # --- 2. AE ---
    try:
        ae_ckpt = torch.load(ae_ckpt_path, map_location=device)

        if 'scale_factor' in ae_ckpt:
            scale_factor = ae_ckpt['scale_factor']
            print(f'Found scale_factor in checkpoint: {scale_factor}')
        else:
            raise SystemExit(f'No scale_factor in checkpoint {ae_ckpt_path}') from None
    except Exception as e:
        raise SystemExit(f'Failed to load Autoencoder: {e}') from None

    # --- 3. Diffusion UNet ---
    unet = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=3,
        out_channels=3,
        num_res_blocks=2,
        channels=(64, 128, 256),
        attention_levels=(False, True, True),
        num_head_channels=32,
    ).to(device)

    # Scheduler
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        schedule='scaled_linear_beta',
        beta_start=0.0015,
        beta_end=0.0195,
    )

    inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)

    # --- 4. 优化器 ---
    optimizer = torch.optim.Adam(unet.parameters(), lr=lr)

    if use_amp:
        scaler = GradScaler()
    else:
        scaler = None

    # --- 5. 初始化状态与 Resume 逻辑 ---
    start_epoch = 0
    best_val_loss = float('inf')

    # [New] Resume 逻辑
    if resume and resume_ckpt_path.exists():
        print(f'Resuming from {resume_ckpt_path}...')
        try:
            ckpt = torch.load(resume_ckpt_path, map_location=device)

            # 加载模型权重
            unet.load_state_dict(ckpt['state_dict'])

            # 加载优化器状态
            if 'optimizer' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer'])

            # 加载 Scaler 状态
            if use_amp and scaler is not None and 'scaler' in ckpt:
                scaler.load_state_dict(ckpt['scaler'])

            # 恢复 Epoch 和 Best Loss
            start_epoch = ckpt['epoch'] + 1
            best_val_loss = ckpt.get('best_val_loss', float('inf'))

            print(f'Successfully resumed from Epoch {start_epoch - 1}. Next epoch: {start_epoch}')
            print(f'Current Best Val Loss: {best_val_loss:.5f}')
        except Exception as e:
            print(f'Resume failed: {e}. Starting from scratch.')

    # 初始化日志
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(log_dir=(log_dir / timestamp).as_posix())
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- 6. 训练循环 ---
    # [New] 使用 range(start_epoch, num_epochs)
    for epoch in range(start_epoch, num_epochs):
        unet.train()
        epoch_loss = 0
        step = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs - 1}', total=len(train_loader))

        for batch in pbar:
            step += 1
            latents = batch['image'].to(device)

            optimizer.zero_grad(set_to_none=True)

            amp_ctx = autocast(device.type) if use_amp else nullcontext()
            with amp_ctx:
                noise = torch.randn_like(latents).to(device)
                timesteps = torch.randint(
                    0, inferer.scheduler.num_train_timesteps,
                    (latents.shape[0],), device=latents.device
                ).long()

                noisy_latents = scheduler.add_noise(original_samples=latents, noise=noise, timesteps=timesteps)
                noise_pred = unet(x=noisy_latents, timesteps=timesteps)

                loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float())

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            # TensorBoard logging
            # global_step 计算方式确保接续之前的 step 计数
            global_step = epoch * len(train_loader) + step
            writer.add_scalar('train/loss_step', loss.item(), global_step)

        avg_epoch_loss = epoch_loss / step
        writer.add_scalar('train/epoch_loss', avg_epoch_loss, epoch)
        print(f'Epoch {epoch} Loss: {avg_epoch_loss:.5f}')

        # --- 验证循环 ---
        if epoch % val_interval == 0:
            unet.eval()
            val_loss = 0
            val_steps = 0

            with torch.no_grad():
                for batch in tqdm(val_loader, desc='Val', leave=False):
                    latents = batch['image'].to(device)
                    noise = torch.randn_like(latents).to(device)
                    timesteps = torch.randint(
                        0, inferer.scheduler.num_train_timesteps,
                        (latents.shape[0],), device=latents.device
                    ).long()

                    noisy_latents = scheduler.add_noise(latents, noise, timesteps)
                    noise_pred = unet(noisy_latents, timesteps=timesteps)

                    loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float())
                    val_loss += loss.item()
                    val_steps += 1

            avg_val_loss = val_loss / val_steps
            writer.add_scalar('val/loss', avg_val_loss, epoch)
            print(f'Val {epoch} Loss: {avg_val_loss:.5f}')

            # 保存 Checkpoint
            checkpoint = {
                'epoch': epoch,
                'state_dict': unet.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'val_loss': avg_val_loss,
            }
            if use_amp:
                checkpoint['scaler'] = scaler.state_dict()

            torch.save(checkpoint, ckpt_dir / 'ldm_last.pt')

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(checkpoint, ckpt_dir / 'ldm_best.pt')
                print('New best saved!')

    writer.close()
    print('LDM Training Completed.')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Keyboard interrupted terminating...')