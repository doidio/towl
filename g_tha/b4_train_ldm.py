# 与传统的、依赖于“先验掩码”来强迫空间注意力的 3D Inpainting 不同（而这种掩码在现实临床部署中是无法预知的），显式窄带 TSDF 表达方式本身就是一
# 种固有的注意力机制。通过在远场（值被截断为 -1 或 1 的区域）将梯度归零，我们自然地将神经网络的容量坍缩到最关键的骨-假体界面上。这消除了对“先知
# 掩码”的需求，确保模型的预测能力完全专注于亚像素级别的几何匹配，从而使其成为端到端手术规划中稳健且可直接部署的方案。

import argparse
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import tomlkit
import torch
from monai.data import DataLoader, Dataset
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
        use_amp, resume, num_workers, num_epochs, val_interval, val_limit,
        batch_size, sw_batch_size, lr, gradient_accumulation_steps, ema_decay, cfg,
    ) = [cfg['train'][task][_] for _ in (
        'use_amp', 'resume', 'num_workers', 'num_epochs', 'val_interval', 'val_limit',
        'batch_size', 'sw_batch_size', 'lr', 'gradient_accumulation_steps', 'ema_decay', 'cfg',
    )]

    patch_size = list(cfg['train']['vae']['patch_size'])

    # 压缩编码后的 latent
    train_files = [{'image': p.as_posix()} for p in sorted((train_root / 'latents' / 'train').glob('*.npy'))]
    val_files = [{'image': p.as_posix()} for p in sorted((train_root / 'latents' / 'val').glob('*.npy'))]
    print(f'Train: {len(train_files)}, Val: {len(val_files)}')

    val_total = min(val_limit, len(val_files))
    val_files = val_files[::len(val_files) // (val_total - 1)]
    print(f'Val limited: {len(val_files)}')

    def load_vae(subtask):
        ckpt_path = ckpt_dir / f'vae_{subtask}_best.pt'
        print(f'Loading VAE from {ckpt_path.resolve()}')
        loaded = torch.load(ckpt_path, map_location=device, weights_only=True)
        vae_model = define.vae_kl().to(device)
        vae_model.load_state_dict(loaded['state_dict'])
        vae_model.eval().float()
        sf = loaded['scale_factor']
        mean = loaded.get('global_mean', 0.0)
        print(f'Scale Factor ({subtask}): {sf:.6f}, Mean: {mean:.6f}')
        return vae_model, sf, mean

    vae_metal, sf_metal, mean_metal = load_vae('metal')
    vae_pre, sf_pre, mean_pre = load_vae('pre')

    # Scale Latent with Scale Factor dynamically
    def scale_latent(data):
        data['image'] = (data['image'] - mean_metal) * sf_metal
        data['condition'] = (data['condition'] - mean_pre) * sf_pre
        return data

    transforms = Compose(define.ldm_transforms() + [scale_latent])

    train_ds = Dataset(data=train_files, transform=transforms)
    val_ds = Dataset(data=val_files, transform=transforms)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=num_workers)

    ldm = define.ldm_unet().to(device)
    ema = define.EMA(ldm, decay=ema_decay)

    scheduler = define.scheduler_ddpm()
    num_train_timesteps = scheduler.num_train_timesteps

    optimizer = torch.optim.AdamW(ldm.parameters(), lr=lr, weight_decay=1e-5)
    scaler = GradScaler() if use_amp else None

    start_epoch = 0
    best_val_loss = float('inf')
    ldm_ckpt_path = ckpt_dir / f'{task}_last.pt'

    if resume and ldm_ckpt_path.exists():
        print(f'Resuming LDM from {ldm_ckpt_path}...')
        ckpt = torch.load(ldm_ckpt_path, map_location=device)
        ldm.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if 'ema_state' in ckpt:
            ema.load_state_dict(ckpt['ema_state'])

        start_epoch = ckpt['epoch']
        best_val_loss = ckpt.get('best_val_loss', float('inf'))

        if use_amp and 'scaler' in ckpt:
            scaler.load_state_dict(ckpt['scaler'])

        print(f'Load from epoch {start_epoch}, best_val_loss {best_val_loss}')
        start_epoch += 1

    suffix = datetime.now().strftime(f'{task}_%Y%m%d_%H%M%S')
    if resume:
        suffix += '_resume'
    log_dir = log_dir / suffix
    writer = SummaryWriter(log_dir=log_dir.as_posix())

    saver = SaveImage(
        output_dir=log_dir,
        output_postfix='',
        output_ext='.nii.gz',
        separate_folder=False,
        print_log=False,
        resample=False,
    )

    def decode_gpu(z, name, vae_model, sf, mean):
        # 使用分块 GPU 推理，避免大图解码 OOM
        z = (z / sf + mean).detach().to(device).float()

        with torch.no_grad():
            recon = define.vae_decode_tiled(
                z=z,
                vae=vae_model,
                patch_size=patch_size,
                sw_batch_size=sw_batch_size,
            )

        name = f'val_epoch_{epoch:03d}_{name}.nii.gz'
        saver(recon[0].cpu(), meta_data={'filename_or_obj': name})
        return recon.cpu()

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

            # CFG Condition Dropout
            if torch.rand(1) < 0.15:
                cond = torch.zeros_like(cond)

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
                loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float())
                loss = loss / gradient_accumulation_steps

            if use_amp:
                scaler.scale(loss).backward()

                if step % gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(ldm.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()

                    optimizer.zero_grad(set_to_none=True)
                    ema.update(ldm)
            else:
                loss.backward()

                if step % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(ldm.parameters(), 1.0)
                    optimizer.step()

                    optimizer.zero_grad(set_to_none=True)
                    ema.update(ldm)

            epoch_loss += loss.item()

            if step % 1 == 0:
                global_step = epoch * len(train_loader) + step
                writer.add_scalar('train/loss', loss.item() * gradient_accumulation_steps, global_step)

            pbar.set_postfix({'MSE': f'{loss.item() * gradient_accumulation_steps:.4f}'})

        writer.add_scalar('train/epoch_loss', epoch_loss / step, epoch)

        # 验证与采样
        if epoch % val_interval == 0:
            ldm.eval()
            ema.store(ldm)
            ema.copy_to(ldm)

            val_loss = 0
            val_steps = 0

            with torch.no_grad():
                for i, batch in enumerate(val_bar := tqdm(val_loader, desc='Val')):
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
                    if i == 0:
                        val_scheduler = define.scheduler_ddim()
                        val_scheduler.set_timesteps(num_inference_steps=200, device=device)

                        generator = torch.Generator(device=device).manual_seed(42)  # 固定随机种子
                        generated = torch.randn(image.shape, device=device, generator=generator)
                        # generated = torch.randn_like(image)

                        for t in val_scheduler.timesteps:
                            val_bar.set_postfix({'DDIM': t.item()})

                            # 拼接条件 (Condition)
                            latent_input = torch.cat([generated] * 2)

                            # 构造条件部分：前半部分是 cond，后半部分是 zeros
                            uncond = torch.zeros_like(cond)
                            cond_input = torch.cat([cond, uncond])

                            model_input = torch.cat([latent_input, cond_input], dim=1)

                            with torch.no_grad():
                                t_input = t[None].to(device).repeat(2)
                                noise_pred_batch = ldm(model_input, t_input)

                            noise_cond, noise_uncond = noise_pred_batch.chunk(2)
                            noise_pred = noise_uncond + cfg * (noise_cond - noise_uncond)

                            # 更新 Latent
                            with torch.no_grad():
                                generated, _ = val_scheduler.step(noise_pred, t, generated)

                        # 解码显示 (Latent -> Image)
                        with amp_ctx:
                            vis_generated = decode_gpu(generated, 'Generated', vae_metal, sf_metal, mean_metal)

                            if epoch == 0:
                                vis_gt = decode_gpu(image, 'GroundTruth', vae_metal, sf_metal, mean_metal)
                                vis_cond = decode_gpu(cond, 'Condition', vae_pre, sf_pre, mean_pre)

                        def norm_vis(x):
                            x_min, x_max = x.min(), x.max()
                            return (x - x_min) / (x_max - x_min + 1e-5)

                        z_idx = vis_generated.shape[2] // 2  # [B, C, D, H, W]

                        vis_generated = vis_generated[0, 0, z_idx]
                        writer.add_image('val/Generated', norm_vis(vis_generated), epoch, dataformats='HW')

                        if epoch == 0:
                            vis_cond = vis_cond[0, 0, z_idx]
                            vis_gt = vis_gt[0, 0, z_idx]
                            writer.add_image('val/Condition', norm_vis(vis_cond), epoch, dataformats='HW')
                            writer.add_image('val/GroundTruth', norm_vis(vis_gt), epoch, dataformats='HW')

            ema.restore(ldm)
            avg_val_loss = val_loss / val_steps
            writer.add_scalar('val/loss', avg_val_loss, epoch)
            print(f'Val Loss: {avg_val_loss:.5f}')

            # 保存模型
            ckpt = {
                'epoch': epoch,
                'state_dict': ldm.state_dict(),
                'ema_state': ema.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }
            if use_amp:
                ckpt['scaler'] = scaler.state_dict()

            ckpt_dir.mkdir(parents=True, exist_ok=True)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                ckpt['best_val_loss'] = best_val_loss
                torch.save(ckpt, ckpt_dir / f'{task}_best.pt')
                print('New best model saved!')

            torch.save(ckpt, ckpt_dir / f'{task}_last.pt')
            # torch.save(ckpt, ckpt_dir / f'{task}_epoch_{epoch}.pt')

        torch.cuda.empty_cache()

    writer.close()
    print('Training Completed.')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Keyboard interrupted terminating...')
