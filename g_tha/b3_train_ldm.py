import argparse
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import tomlkit
import torch
from monai.data import DataLoader, Dataset
from monai.networks.schedulers import DDIMScheduler
from monai.transforms import Compose, SaveImage
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
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

    train_root = Path(str(cfg['train']['root']))
    # cache_dir = train_root / 'cache'
    log_dir = train_root / 'logs'
    ckpt_dir = train_root / 'checkpoints'

    task = 'ldm'
    (
        use_amp, resume, num_workers, num_epochs, val_interval, val_limit, val_vae,
        batch_size, sw_batch_size, lr, ema_rate, gradient_accumulation_steps,
    ) = [cfg['train'][task][_] for _ in (
        'use_amp', 'resume', 'num_workers', 'num_epochs', 'val_interval', 'val_limit', 'val_vae',
        'batch_size', 'sw_batch_size', 'lr', 'ema_rate', 'gradient_accumulation_steps',
    )]

    patch_size = cfg['train']['vae']['patch_size']

    # 压缩编码后的 latent
    train_files = [{'image': p.as_posix()} for p in sorted((train_root / 'latents' / 'train').glob('*.npy'))]
    val_files = [{'image': p.as_posix()} for p in sorted((train_root / 'latents' / 'val').glob('*.npy'))]
    print(f'Train: {len(train_files)}, Val: {len(val_files)}')

    val_total = min(val_limit, len(val_files))
    val_files = val_files[::len(val_files) // (val_total - 1)]
    print(f'Val limited: {len(val_files)}')

    transforms = Compose(define.ldm_transforms())

    train_ds = Dataset(data=train_files, transform=transforms)
    val_ds = Dataset(data=val_files, transform=transforms)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=num_workers)

    ldm = define.ldm_unet().to(device)
    ema = define.EMA(ldm, decay=ema_rate)

    scheduler = define.scheduler_ddpm()
    num_train_timesteps = scheduler.num_train_timesteps

    optimizer = torch.optim.AdamW(ldm.parameters(), lr=lr, weight_decay=1e-5)
    scaler = GradScaler() if use_amp else None

    vae, scale_factor = None, None
    if val_vae and (vae_ckpt_path := ckpt_dir / 'vae_best.pt').exists():
        print(f'Loading VAE from {vae_ckpt_path.resolve()}')
        vae_ckpt = torch.load(vae_ckpt_path, map_location=device)
        if 'scale_factor' in vae_ckpt:
            vae = define.vae_kl().to(device)
            vae.load_state_dict(vae_ckpt['state_dict'])
            vae.eval()
            scale_factor = vae_ckpt['scale_factor']
            print(f'Scale Factor: {scale_factor}')

    start_epoch = 0
    best_val_loss = float('inf')
    ldm_ckpt_path = ckpt_dir / f'{task}_last.pt'

    if resume and ldm_ckpt_path.exists():
        print(f'Resuming LDM from {ldm_ckpt_path}...')
        ckpt = torch.load(ldm_ckpt_path, map_location=device)
        ldm.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch']
        best_val_loss = ckpt.get('best_val_loss', float('inf'))

        if use_amp and 'scaler' in ckpt:
            scaler.load_state_dict(ckpt['scaler'])

        if 'ema_state_dict' in ckpt:
            ema.load_state_dict(ckpt['ema_state_dict'])
        else:
            ema = define.EMA(ldm, decay=ema_rate)

        print(f'Load from epoch {start_epoch}, best_val_loss {best_val_loss}')

    timestamp = datetime.now().strftime(f'{task}_%Y%m%d_%H%M%S')
    writer = SummaryWriter(log_dir=(log_dir / timestamp).as_posix())

    saver = SaveImage(
        output_dir=log_dir,
        output_postfix='',
        output_ext='.nii.gz',
        separate_folder=False,
        print_log=False,
        resample=False
    )

    amp_ctx = autocast(device.type) if use_amp else nullcontext()

    for epoch in range(start_epoch, num_epochs):
        ldm.train()
        epoch_loss = 0
        step = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs - 1}')

        for batch in pbar:
            step += 1
            image = batch['image'].to(device)
            cond = batch['condition'].to(device)

            # 以角点4x4块作背景参照，增加前景Loss权重
            bg_ref = image[..., 0, 0, 0].detach().view(image.shape[0], image.shape[1], 1, 1, 1)
            dist_to_bg = torch.abs(image - bg_ref).sum(dim=1, keepdim=True)
            mask = (dist_to_bg > 0.1).float().detach()
            mask = torch.nn.functional.max_pool3d(mask, kernel_size=3, stride=1, padding=1)

            optimizer.zero_grad(set_to_none=True)

            with amp_ctx:
                # 采样时间步 t
                timesteps = torch.randint(0, num_train_timesteps, (image.shape[0],), device=device).long()

                # 生成噪声
                noise = torch.randn_like(image)

                # 加噪过程 (Forward)
                noisy_image = scheduler.add_noise(original_samples=image, noise=noise, timesteps=timesteps)

                # 拼接输入
                input_tensor = torch.cat([noisy_image, cond], dim=1)

                # 预测噪声
                noise_pred = ldm(x=input_tensor, timesteps=timesteps)

                # 计算损失
                loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float(), reduction='none')
                loss = 0.2 * loss.mean() + 0.8 * (loss * mask).sum() / (mask.sum() + 1e-6)
                loss = loss / gradient_accumulation_steps

            if use_amp:
                scaler.scale(loss).backward()

                if (step + 1) % gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(ldm.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
            else:
                loss.backward()

                if (step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(ldm.parameters(), 1.0)
                    optimizer.step()

            ema.update(ldm)
            epoch_loss += loss.item()

            if step % 10 == 0:
                global_step = epoch * len(train_loader) + step
                writer.add_scalar('train/loss', loss.item() * gradient_accumulation_steps, global_step)

            pbar.set_postfix({'MSE': f'{loss.item() * gradient_accumulation_steps:.4f}'})

        writer.add_scalar('train/epoch_loss', epoch_loss / step, epoch)

        # 验证与采样
        if epoch % val_interval == 0:
            ema.store(ldm)
            ema.copy_to(ldm)

            ldm.eval()
            val_loss = 0
            val_steps = 0

            with torch.no_grad():
                for i, batch in enumerate(val_bar := tqdm(val_loader, desc='Val', leave=False)):
                    image = batch['image'].to(device)
                    cond = batch['condition'].to(device)

                    # 采样时间步 t
                    timesteps = torch.randint(0, num_train_timesteps, (image.shape[0],), device=device).long()

                    # 生成噪声
                    noise = torch.randn_like(image)

                    # 加噪过程 (Forward)
                    noisy_image = scheduler.add_noise(image, noise, timesteps)

                    # 拼接输入
                    input_tensor = torch.cat([noisy_image, cond], dim=1)

                    # 预测噪声
                    with amp_ctx:
                        noise_pred = ldm(input_tensor, timesteps)
                        loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float())

                    val_loss += loss.item()
                    val_steps += 1

                    # 可视化：仅第一个 Batch 的第一个样本
                    if i == 0 and vae is not None:
                        val_scheduler = DDIMScheduler(
                            num_train_timesteps=1000,
                            schedule='scaled_linear_beta',
                            beta_start=0.00085,
                            beta_end=0.012,
                            prediction_type='epsilon',
                            clip_sample=False,  # Latent 空间通常不 clip
                        )
                        val_scheduler.set_timesteps(num_inference_steps=50, device=device)

                        # 从纯噪声开始采样
                        sample_image = torch.randn_like(image)

                        for t in val_scheduler.timesteps:
                            val_bar.set_postfix({'DDIM': t.item()})

                            # 拼接条件 (Condition)
                            model_input = torch.cat([sample_image, cond], dim=1)

                            with torch.no_grad():
                                # 预测噪声
                                model_output = ldm(model_input, t[None].to(device))

                            # 更新 Latent，注意这里用 val_scheduler.step
                            sample_image, _ = val_scheduler.step(model_output, t, sample_image)

                        # 解码显示 (Latent -> Image)
                        with amp_ctx:
                            decoded = []
                            for idx, img in enumerate((sample_image, image, cond)):
                                img /= scale_factor  # Latent 训练时被 Scale 过了，解码前要除回去
                                decoded.append(define.sliding_window_decode(img, vae, patch_size, sw_batch_size))

                                name = ['Generated', 'GroundTruth', 'Condition'][idx]
                                name = f'val_epoch_{epoch:03d}_{name}.nii.gz'
                                saver(decoded[-1][0], meta_data={'filename_or_obj': name})

                            # recon = vae.decode(sample_image / scale_factor)
                            # gt = vae.decode(image / scale_factor)
                            # cond_vis = vae.decode(cond / scale_factor)

                        recon, gt, cond_vis = decoded

                        def norm_vis(x):
                            return torch.clamp(x * 0.5 + 0.5, 0, 1)

                        z_idx = recon.shape[2] // 2  # [B, C, D, H, W]

                        vis_cond = cond_vis[0, 0, z_idx]
                        vis_recon = recon[0, 0, z_idx]
                        vis_gt = gt[0, 0, z_idx]

                        writer.add_image('val/Condition', norm_vis(vis_cond), epoch, dataformats='HW')
                        writer.add_image('val/Generated', norm_vis(vis_recon), epoch, dataformats='HW')
                        writer.add_image('val/GroundTruth', norm_vis(vis_gt), epoch, dataformats='HW')

            avg_val_loss = val_loss / val_steps
            writer.add_scalar('val/loss', avg_val_loss, epoch)
            print(f'Val Loss (EMA): {avg_val_loss:.5f}')

            ema.restore(ldm)

            # 保存模型
            ckpt = {
                'epoch': epoch,
                'state_dict': ldm.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'ema_state_dict': ema.state_dict(),
            }
            if use_amp:
                ckpt['scaler'] = scaler.state_dict()

            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(ckpt, ckpt_dir / f'{task}_last.pt')

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                ckpt['best_val_loss'] = best_val_loss
                torch.save(ckpt, ckpt_dir / f'{task}_best.pt')
                print('New best model saved!')

    writer.close()
    print('Training Completed.')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Keyboard interrupted terminating...')
