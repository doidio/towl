from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import torch
from monai.data import DataLoader, PersistentDataset
from monai.inferers import sliding_window_inference
from monai.losses import PatchAdversarialLoss, PerceptualLoss
from monai.metrics import PSNRMetric, SSIMMetric
from monai.networks.nets import AutoencoderKL, PatchDiscriminator
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged, RandSpatialCropd
)
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def calculate_scale_factor(model, dataloader, device, use_amp: bool = True):
    model.eval()
    running_std = 0.0
    count = 0
    print('Calculating Latent Scale Factor for LDM...')
    with torch.no_grad():
        amp_ctx = autocast(device.type) if use_amp else nullcontext()
        with amp_ctx:
            for batch in tqdm(dataloader):
                images = batch['image'].to(device)
                # 获取 latent 分布
                z_mu, z_logvar = model.encode(images)
                # 采样得到 z
                z = model.sampling(z_mu, z_logvar)
                running_std += z.std().item()
                count += 1

    scale_factor = 1.0 / (running_std / count)
    print(f'Recommended Scale Factor: {scale_factor}')
    # 后续训练 LDM 时： z = z * scale_factor
    return scale_factor


def main():
    # --- 配置参数 ---
    use_amp = False
    use_checkpoint = False
    num_workers = 8
    num_epochs = 500
    batch_size = 4
    val_interval = 5
    resume = False
    patch_size = (96,) * 3

    num_train_steps = 200
    num_val_steps = 50
    warmup_epochs = 10

    # 权重参数 (参考教程与经验值)
    adv_weight = 0.01  # 对抗损失权重
    perceptual_weight = 0.002  # 感知损失权重
    kl_weight = 1e-6  # KL正则化权重 (非常重要，过大会导致模糊，过小会导致Latent空间不连续)

    # 路径配置
    root = Path('.ds')
    task = 'post'
    cache_dir = root / 'cache'
    log_dir = root / 'logs'
    ckpt_dir = root / 'checkpoints' / 'ae_kl'

    ct_range = (-200, 2800)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- 1. 数据准备 (保持原逻辑) ---
    train_files = [{'image': str(p)} for p in (root / task / 'train').glob('*.nii.gz')]
    val_files = [{'image': str(p)} for p in (root / task / 'val').glob('*.nii.gz')]
    print(f'Train: {len(train_files)}, Val: {len(val_files)}')

    base_transforms = [
        LoadImaged(keys=['image']),
        EnsureChannelFirstd(keys=['image'], channel_dim='no_channel'),
        ScaleIntensityRanged(keys=['image'], a_min=ct_range[0], a_max=ct_range[1], b_min=-1.0, b_max=1.0, clip=True),
    ]

    train_transforms = Compose(base_transforms + [
        RandSpatialCropd(keys=['image'], roi_size=patch_size, random_size=False),
    ])

    val_transforms = Compose(base_transforms)

    train_ds = PersistentDataset(train_files, train_transforms, cache_dir)
    val_ds = PersistentDataset(val_files, val_transforms, cache_dir) if len(val_files) else None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=num_workers) if val_ds else None

    # --- 2. 模型定义 (AutoencoderKL + Discriminator) ---
    print('Initializing AutoencoderKL & Discriminator...')

    # 生成器 (VAE)
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

    # 判别器 (PatchGAN)
    discriminator = PatchDiscriminator(
        spatial_dims=3,
        num_layers_d=3,
        channels=64,
        in_channels=1,
        out_channels=1,
    ).to(device)

    # --- 3. 损失函数 ---
    l1_loss = torch.nn.L1Loss()
    adv_loss = PatchAdversarialLoss(criterion='least_squares')  # LSGAN 更稳定

    # 感知损失 (需要联网下载预训练权重，或者手动指定路径)
    # network_type='squeeze' 显存占用小，适合3D; 'alex' 或 'vgg' 效果可能更好但更重
    perceptual_loss = PerceptualLoss(
        spatial_dims=3,
        network_type='vgg',
        is_fake_3d=True,
        fake_3d_ratio=0.2,
    ).to(device)

    # --- 4. 优化器 (两个) ---
    optimizer_g = torch.optim.Adam(model.parameters(), lr=1e-5)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=5e-5)  # 判别器LR通常略大

    # 混合精度 Scaler (两个) - 可选
    if use_amp:
        scaler_g = GradScaler()
        scaler_d = GradScaler()
    else:
        scaler_g = None
        scaler_d = None

    # --- 5. 初始化日志与恢复训练 ---
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(log_dir=(log_dir / timestamp).as_posix())

    start_epoch = 0
    best_val_loss = float('inf')
    resume_pt = (ckpt_dir / f'{task}_last.pt').resolve()

    if resume and resume_pt.exists():
        try:
            print(f'Loading checkpoint from {resume_pt}...')
            checkpoint = torch.load(resume_pt, map_location=device)
            # 必须检查是否是 KL 模型的权重，避免加载旧 VQVAE 权重报错
            if 'discriminator' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
                discriminator.load_state_dict(checkpoint['discriminator'])
                optimizer_g.load_state_dict(checkpoint['optimizer_g'])
                optimizer_d.load_state_dict(checkpoint['optimizer_d'])
                # 仅在启用 AMP 时恢复 scaler 状态
                if use_amp and 'scaler_g' in checkpoint and 'scaler_d' in checkpoint:
                    scaler_g.load_state_dict(checkpoint['scaler_g'])
                    scaler_d.load_state_dict(checkpoint['scaler_d'])
                start_epoch = checkpoint['epoch']
                best_val_loss = checkpoint.get('best_val_loss', float('inf'))
                print(f'Resumed from epoch {start_epoch + 1}')
            else:
                print('Checkpoint mismatch (likely old VQVAE), starting from scratch.')
        except Exception as e:
            print(f'Resume failed: {e}. Starting from scratch.')

    # --- 6. 训练循环 ---
    try:
        for epoch in range(start_epoch, num_epochs):
            warmup = epoch < warmup_epochs

            model.train()
            discriminator.train()

            epoch_loss_g = 0
            epoch_loss_d = 0
            step = 0

            pbar = tqdm(train_loader, desc=f'Ep {epoch + 1}/{num_epochs}',
                        total=min(len(train_loader), num_train_steps))

            for batch in pbar:
                if step >= num_train_steps:
                    break
                step += 1
                images = batch['image'].to(device)

                # -----------------------
                #  Train Generator (VAE)
                # -----------------------
                optimizer_g.zero_grad(set_to_none=True)

                amp_ctx = autocast(device.type) if use_amp else nullcontext()
                with amp_ctx:
                    # 1. 先编码获取分布参数
                    z_mu, z_logvar = model.encode(images)
                    if warmup:
                        z_mu = torch.clamp(z_mu, -20.0, 20.0)

                    # 2. 根据阶段决定是否采样
                    if warmup:
                        # Warmup阶段：强制确定性 (直接用均值)，类似 VQVAE
                        z = z_mu
                    else:
                        # 正式训练：恢复随机采样 (Standard VAE)
                        z = model.sampling(z_mu, z_logvar)

                    # 3. 解码
                    reconstruction = model.decode(z)

                    z_logvar = torch.clamp(z_logvar, min=-10.0, max=10.0)

                    # 1. Reconstruction Loss (L1)
                    recons_loss = l1_loss(reconstruction.float(), images.float())

                    # 2. Perceptual Loss
                    p_loss = perceptual_loss(reconstruction.float(), images.float())

                    # 3. KL Regularization Loss
                    # 修改为 dim=[1, 2, 3, 4] 以覆盖 Channel, Depth, Height, Width
                    kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp(), dim=[1, 2, 3, 4])
                    kl_loss = torch.mean(kl_loss)  # 在 Batch 维度取平均

                    # 4. Adversarial Loss (Generator part: try to fool D)
                    logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                    gen_adv_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)

                    # Total Generator Loss
                    loss_g = recons_loss
                    if not warmup:
                        loss_g += perceptual_weight * p_loss
                        loss_g += kl_weight * kl_loss
                        loss_g += adv_weight * gen_adv_loss

                if torch.isnan(loss_g):
                    raise SystemExit(f'recon_nan: {torch.isnan(recons_loss)}, '
                                     f'perc_nan: {torch.isnan(p_loss)}, '
                                     f'kl_nan: {torch.isnan(kl_loss)}, '
                                     f'adv_nan: {torch.isnan(gen_adv_loss)} at step {step}')

                if use_amp:
                    scaler_g.scale(loss_g).backward()
                    scaler_g.step(optimizer_g)
                    scaler_g.update()
                else:
                    loss_g.backward()
                    optimizer_g.step()

                # -----------------------
                #  Train Discriminator
                # -----------------------
                if warmup:
                    loss_d = torch.tensor(0.0)
                else:
                    optimizer_d.zero_grad(set_to_none=True)

                    amp_ctx = autocast(device.type) if use_amp else nullcontext()
                    with amp_ctx:
                        # Real
                        logits_real = discriminator(images.contiguous().detach())[-1]
                        loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)

                        # Fake
                        logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                        loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)

                        # Total Discriminator Loss
                        loss_d = (loss_d_real + loss_d_fake) * 0.5 * adv_weight

                    if use_amp:
                        scaler_d.scale(loss_d).backward()

                        scaler_g.unscale_(optimizer_g)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                        scaler_d.step(optimizer_d)
                        scaler_d.update()
                    else:
                        loss_d.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer_d.step()

                # Logging
                epoch_loss_d += loss_d.item()
                epoch_loss_g += loss_g.item()

                # Update Pbar
                if warmup:
                    pbar.set_postfix({
                        'L1': f'{recons_loss.item():.4f}',
                    })
                else:
                    pbar.set_postfix({
                        'L1': f'{recons_loss.item():.4f}',
                        'Adv': f'{gen_adv_loss.item():.4f}',
                        'Perc': f'{p_loss.item():.4f}'
                    })

                # TensorBoard Step
                global_step = epoch * len(train_loader) + step
                if step % 10 == 0:
                    if warmup:
                        writer.add_scalar('train/loss_g', loss_g.item(), global_step)
                        writer.add_scalar('train/recons_loss', recons_loss.item(), global_step)
                    else:
                        writer.add_scalar('train/loss_g', loss_g.item(), global_step)
                        writer.add_scalar('train/loss_d', loss_d.item(), global_step)
                        writer.add_scalar('train/recons_loss', recons_loss.item(), global_step)
                        writer.add_scalar('train/kl_loss', kl_loss.item(), global_step)
                        writer.add_scalar('train/p_loss', p_loss.item(), global_step)

            writer.add_scalar('train/epoch_loss_g', epoch_loss_g / step, epoch)

            # --- 验证循环 ---
            if val_loader and (epoch + 1) % val_interval == 0:
                model.eval()
                val_loss = 0
                val_psnr = PSNRMetric(max_val=2.0)
                val_ssim = SSIMMetric(data_range=2.0, spatial_dims=3)

                val_pbar = tqdm(val_loader, desc='Val', leave=False, total=min(len(val_loader), num_val_steps))

                step = 0
                with torch.no_grad():
                    amp_ctx = autocast(device.type) if use_amp else nullcontext()
                    for i, batch in enumerate(val_pbar):
                        if step >= num_val_steps:
                            break
                        step += 1
                        images = batch['image'].to(device)

                        # Inference (AutoencoderKL returns: recon, mu, sigma)
                        with amp_ctx:
                            reconstruction = sliding_window_inference(
                                inputs=images,
                                roi_size=patch_size,
                                sw_batch_size=4,  # 显存允许的情况下调大，加快推理
                                predictor=lambda _: model.decode(model.encode(_)[0]),
                                overlap=0.25,  # 重叠率，减少拼接缝隙
                                mode='gaussian',  # 拼接模式，推荐 gaussian
                                device=device,
                            )

                        l1 = l1_loss(reconstruction.float(), images.float())
                        val_loss += l1.item()

                        val_psnr(y_pred=reconstruction, y=images)
                        val_ssim(y_pred=reconstruction, y=images)

                        val_pbar.set_postfix({'v_loss': f'{l1.item():.4f}'})

                        if i == 0:
                            # Visualization
                            img_tensor = images[0, 0]
                            recon_tensor = reconstruction[0, 0]

                            if img_tensor.shape != recon_tensor.shape:
                                print(f'Shape mismatch: {img_tensor.shape} vs {recon_tensor.shape}')
                                continue

                            diff = torch.abs(img_tensor - recon_tensor)

                            # Axial
                            z_idx = img_tensor.shape[0] // 2
                            writer.add_image('val/Orig', img_tensor[z_idx], epoch, dataformats='HW')
                            writer.add_image('val/Recon', recon_tensor[z_idx], epoch, dataformats='HW')
                            writer.add_image('val/Diff', diff[z_idx], epoch, dataformats='HW')

                val_loss /= step
                psnr_score = val_psnr.aggregate().item()
                ssim_score = val_ssim.aggregate().item()
                val_psnr.reset()
                val_ssim.reset()

                writer.add_scalar('val/loss', val_loss, epoch)
                writer.add_scalar('val/psnr', psnr_score, epoch)
                writer.add_scalar('val/ssim', ssim_score, epoch)

                print(f'\nVal Epoch {epoch + 1}: Loss={val_loss:.5f} | PSNR={psnr_score:.4f} | SSIM={ssim_score:.4f}')

                # 保存 Checkpoint
                checkpoint = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'discriminator': discriminator.state_dict(),
                    'optimizer_g': optimizer_g.state_dict(),
                    'optimizer_d': optimizer_d.state_dict(),
                    'best_val_loss': best_val_loss,
                }

                if use_amp:
                    checkpoint['scaler_g'] = scaler_g.state_dict()
                    checkpoint['scaler_d'] = scaler_d.state_dict()

                # Save Last
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save(checkpoint, ckpt_dir / f'{task}_last.pt')

                # Save Best
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(checkpoint, ckpt_dir / f'{task}_best.pt')
                    print('New best model saved!')

    except KeyboardInterrupt:
        print('Training interrupted, saving current state...')

    writer.close()
    print('Training Completed.')

    factor = calculate_scale_factor(model, train_loader, device, use_amp=use_amp)
    print(f'Final Latent Scale Factor for LDM: {factor}')


if __name__ == '__main__':
    main()
