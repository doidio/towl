import argparse
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import tomlkit
import torch
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.losses import PatchAdversarialLoss
from monai.metrics import PSNRMetric, SSIMMetric
from monai.transforms import Compose
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

    dataset_root = Path(str(cfg['dataset']['root']))
    train_root = Path(str(cfg['train']['root']))
    # cache_dir = train_root / 'cache'
    log_dir = train_root / 'logs'
    ckpt_dir = train_root / 'checkpoints'

    task = 'autoencoder'
    (
        use_amp, resume,
        num_workers, num_epochs, warmup_epochs, batch_size, val_interval, patch_size, sw_batch_size,
        train_limit, val_limit,
        adv_weight, per_weight, kl_weight,
        ct_range, bone_range, lr_g, lr_d,
    ) = [cfg['train'][task][_] for _ in (
        'use_amp', 'resume',
        'num_workers', 'num_epochs', 'warmup_epochs', 'batch_size', 'val_interval', 'patch_size', 'sw_batch_size',
        'train_limit', 'val_limit',
        'adversarial_weight', 'perceptual_weight', 'kl_weight',
        'ct_range', 'bone_range', 'lr_g', 'lr_d',
    )]

    # 数据集覆盖术前和术后
    train_files = [{'image': p.as_posix()} for p in sorted((dataset_root / 'pre' / 'train').glob('*.nii.gz'))]
    train_files += [{'image': p.as_posix()} for p in sorted((dataset_root / 'post' / 'train').glob('*.nii.gz'))]
    val_files = [{'image': p.as_posix()} for p in sorted((dataset_root / 'pre' / 'val').glob('*.nii.gz'))]
    val_files += [{'image': p.as_posix()} for p in sorted((dataset_root / 'post' / 'val').glob('*.nii.gz'))]
    print(f'Train: {len(train_files)}, Val: {len(val_files)}')

    train_total = min(train_limit, len(train_files))
    val_total = min(val_limit, len(val_files))
    val_files = val_files[::len(val_files) // (val_total - 1)]
    print(f'Train limited: {train_total}, Val limited: {len(val_files)}')

    train_transforms = Compose(define.autoencoder_train_transforms(patch_size, bone_range[0]))
    val_transforms = Compose(define.autoencoder_val_transforms())

    train_ds = Dataset(train_files, train_transforms)
    val_ds = Dataset(val_files, val_transforms) if len(val_files) else None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=num_workers) if val_ds else None

    # 生成器 (AutoencoderKL)
    autoencoder = define.autoencoder().to(device)

    # 判别器 (PatchGAN)
    discriminator = define.discriminator().to(device)

    # L1 损失
    L1Loss = torch.nn.L1Loss()

    # 对抗损失
    AdversarialLoss = PatchAdversarialLoss(criterion='least_squares')

    # 感知损失
    PerceptualLoss = define.perceptual_loss().to(device)

    # 优化器
    optimizer_g = torch.optim.Adam(autoencoder.parameters(), lr=lr_g, betas=(0.5, 0.9))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.9))

    # 混合精度 Scaler
    if use_amp:
        scaler_g = GradScaler()
        scaler_d = GradScaler()
    else:
        scaler_g = None
        scaler_d = None

    # 日志
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(log_dir=(log_dir / timestamp).as_posix())

    start_epoch = 0
    best_val_ssim = -1.0

    # 继续训练
    if resume:
        load_pt = (ckpt_dir / f'{task}_last.pt').resolve()
    else:
        load_pt = None

    if load_pt and load_pt.exists():
        try:
            print(f'Loading checkpoint from {load_pt}...')
            checkpoint = torch.load(load_pt, map_location=device)

            autoencoder.load_state_dict(checkpoint['state_dict'])
            discriminator.load_state_dict(checkpoint['discriminator'])
            optimizer_g.load_state_dict(checkpoint['optimizer_g'])
            optimizer_d.load_state_dict(checkpoint['optimizer_d'])

            if use_amp and 'scaler_g' in checkpoint and 'scaler_d' in checkpoint:
                scaler_g.load_state_dict(checkpoint['scaler_g'])
                scaler_d.load_state_dict(checkpoint['scaler_d'])

            start_epoch = checkpoint['epoch']
            best_val_ssim = checkpoint.get('best_val_ssim', -1.0)
            print(f'Load from epoch {start_epoch}, best_val_ssim {best_val_ssim}')
        except Exception as e:
            print(f'Load failed: {e}. Starting from scratch.')

    # 训练
    for epoch in range(start_epoch, num_epochs):
        warmup = epoch < warmup_epochs

        autoencoder.train()
        discriminator.train()

        epoch_loss_g = 0
        epoch_loss_d = 0
        step = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs - 1}')

        for batch in pbar:
            step += 1
            images = batch['image'].to(device, non_blocking=True)

            optimizer_g.zero_grad(set_to_none=True)

            amp_ctx = autocast(device.type) if use_amp else nullcontext()
            with amp_ctx:
                # 编码获取分布参数
                z_mu, z_sigma = autoencoder.encode(images)

                # 解码
                z = autoencoder.sampling(z_mu, z_sigma)
                reconstruction = autoencoder.decode(z)

                # L1 重建损失
                l1_loss = L1Loss(reconstruction.float(), images.float())

                if torch.isnan(l1_loss):
                    raise SystemExit('NaN in l1_loss')

            # 感知损失，退出AMP避免NaN，加微量噪声避免MedicalNet统计std=0导致NaN
            img_float = images.float()
            noise = torch.randn_like(img_float) * 1e-4
            per_loss = PerceptualLoss(reconstruction.float() + noise, img_float + noise)

            if torch.isnan(per_loss):
                raise SystemExit('NaN in per_loss')

            per_loss *= per_weight

            with amp_ctx:
                # KL 正则化损失
                # kl_loss = 0.5 * torch.mean(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1)
                z_sigma_clamped = torch.clamp(z_sigma, min=1e-6, max=1e3)
                kl_loss = 0.5 * torch.mean(z_mu.pow(2) + z_sigma_clamped.pow(2) - torch.log(z_sigma_clamped.pow(2)) - 1)

                if torch.isnan(kl_loss):
                    raise SystemExit('NaN in kl_loss')

                kl_loss *= kl_weight

                # 生成器损失
                loss_g = l1_loss + per_loss + kl_loss

                # 预热后补充对抗损失
                if not warmup:
                    out = discriminator(reconstruction.float())
                    adv_loss = AdversarialLoss(out, target_is_real=True, for_discriminator=False)

                    if torch.isnan(adv_loss):
                        raise SystemExit('NaN in adv_loss')

                    adv_loss *= adv_weight
                    loss_g += adv_loss

            if use_amp:
                scaler_g.scale(loss_g).backward()
                scaler_g.unscale_(optimizer_g)
                torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=1.0)
                scaler_g.step(optimizer_g)
                scaler_g.update()
            else:
                loss_g.backward()
                torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=1.0)
                optimizer_g.step()

            # 预热后训练判别器
            if not warmup:
                optimizer_d.zero_grad(set_to_none=True)

                amp_ctx = autocast(device.type) if use_amp else nullcontext()
                with amp_ctx:
                    # Real
                    logits_real = discriminator(images.detach())
                    loss_d_real = AdversarialLoss(logits_real, target_is_real=True, for_discriminator=True)

                    # Fake
                    logits_fake = discriminator(reconstruction.detach())
                    loss_d_fake = AdversarialLoss(logits_fake, target_is_real=False, for_discriminator=True)

                    # Total Discriminator Loss
                    loss_d = (loss_d_real + loss_d_fake) * 0.5

                if use_amp:
                    scaler_d.scale(loss_d).backward()
                    scaler_d.unscale_(optimizer_d)
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)

                    scaler_d.step(optimizer_d)
                    scaler_d.update()
                else:
                    loss_d.backward()
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                    optimizer_d.step()
                epoch_loss_d += loss_d.item()

            # 轮次累计
            epoch_loss_g += loss_g.item()

            # 日志
            global_step = epoch * pbar.total + step

            postfix = {'L1': f'{l1_loss.item():.4f}', 'Per.': f'{per_loss.item():.4f}', 'KL': f'{kl_loss.item():.4f}'}
            if step % 10 == 0:
                writer.add_scalar('train/loss_g', loss_g.item(), global_step)
                writer.add_scalar('train/z_mu_mean', z_mu.mean().item(), global_step)
                writer.add_scalar('train/z_sigma_mean', z_sigma.mean().item(), global_step)
                writer.add_scalar('train/l1_loss', l1_loss.item(), global_step)
                writer.add_scalar('train/per_loss', per_loss.item(), global_step)
                writer.add_scalar('train/kl_loss', kl_loss.item(), global_step)

            if not warmup:
                postfix['Adv.'] = f'{adv_loss.item():.4f}'
                if step % 10 == 0:
                    writer.add_scalar('train/adv_loss', adv_loss.item(), global_step)
                    writer.add_scalar('train/loss_d', loss_d.item(), global_step)

            pbar.set_postfix(postfix)

        writer.add_scalar('train/epoch_loss_g', epoch_loss_g / step, epoch)

        # 验证
        if val_loader and epoch % val_interval == 0:
            # Clear cache before validation to provide maximum headroom
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            autoencoder.eval()
            val_l1_loss = 0
            PSNR = PSNRMetric(max_val=2.0)
            SSIM = SSIMMetric(data_range=2.0, spatial_dims=3)

            val_pbar = tqdm(val_loader, desc='Val', leave=False)

            step = 0
            with torch.no_grad():
                amp_ctx = autocast(device.type) if use_amp else nullcontext()
                for i, batch in enumerate(val_pbar):
                    step += 1
                    images = batch['image'].to(device, non_blocking=True)

                    with amp_ctx:
                        reconstruction = sliding_window_inference(
                            inputs=images,
                            roi_size=patch_size,
                            sw_batch_size=sw_batch_size,
                            predictor=define.autoencoder_encode_decode_mu,
                            overlap=0.25,
                            mode='gaussian',
                            device=device,
                            sw_device=device,
                            progress=False,
                        )

                    l1_loss = L1Loss(reconstruction.float(), images.float())
                    val_l1_loss += l1_loss.item()

                    PSNR(y_pred=reconstruction, y=images)
                    SSIM(y_pred=reconstruction, y=images)

                    val_pbar.set_postfix({'L1': f'{l1_loss.item():.4f}'})

                    # 可视化
                    if i == 0:
                        image = images[0] * 0.5 + 0.5  # (-1, 1) -> (0, 1)
                        recon = reconstruction[0] * 0.5 + 0.5
                        diff = torch.abs(image - recon)

                        # 取中间切片
                        z_idx = image.shape[1] // 2  # [2, X, Y, Z]

                        writer.add_image('val/CT_Input', image[0, z_idx].unsqueeze(0), epoch)
                        writer.add_image('val/CT_Recon', recon[0, z_idx].unsqueeze(0), epoch)
                        writer.add_image('val/CT_Diff', diff[0, z_idx].unsqueeze(0), epoch)

            val_l1_loss /= step
            psnr = PSNR.aggregate().item()
            ssim = SSIM.aggregate().item()
            PSNR.reset()
            SSIM.reset()

            writer.add_scalar('val/l1', val_l1_loss, epoch)
            writer.add_scalar('val/psnr', psnr, epoch)
            writer.add_scalar('val/ssim', ssim, epoch)

            print(f'Val Epoch {epoch}: L1={val_l1_loss:.5f} | PSNR={psnr:.4f} | SSIM={ssim:.4f}')

            # 保存
            checkpoint = {
                'epoch': epoch,
                'state_dict': autoencoder.state_dict(),
                'discriminator': discriminator.state_dict(),
                'optimizer_g': optimizer_g.state_dict(),
                'optimizer_d': optimizer_d.state_dict(),
                'best_val_ssim': best_val_ssim,
                'val_l1': val_l1_loss,
                'val_psnr': psnr,
                'val_ssim': ssim,
            }

            if use_amp:
                checkpoint['scaler_g'] = scaler_g.state_dict()
                checkpoint['scaler_d'] = scaler_d.state_dict()

            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, ckpt_dir / f'{task}_last.pt')

            if ssim > best_val_ssim:
                best_val_ssim = ssim
                checkpoint['best_val_ssim'] = best_val_ssim
                torch.save(checkpoint, ckpt_dir / f'{task}_best.pt')
                print('New best model saved!')

    writer.close()
    print('Training Completed.')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Keyboard interrupted terminating...')
