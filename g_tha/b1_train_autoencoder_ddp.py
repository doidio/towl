# PYTHONWARNINGS="ignore" uv run torchrun --nproc_per_node=2 b1_train_autoencoder_ddp.py --config config.toml

import argparse
import os
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import tomlkit
import torch
import torch.distributed as dist
import torch.nn as nn
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.losses import PatchAdversarialLoss
from monai.metrics import PSNRMetric, SSIMMetric
from monai.transforms import Compose
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import define


def setup_ddp():
    # 检查是否由 torchrun 启动
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])

        if torch.cuda.is_available():
            # 核心修复：必须先 set_device，再做其他任何 CUDA 操作
            torch.cuda.set_device(local_rank)
            device = torch.device(f'cuda:{local_rank}')
            dist.init_process_group(backend='nccl', init_method='env://')
        else:
            device = torch.device('cpu')
            dist.init_process_group(backend='gloo', init_method='env://')
    else:
        # 单卡/调试模式回退
        print('Not using DDP, falling back to single device.')
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return device, rank, world_size, local_rank


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def main():
    # 1. DDP 环境设置
    device, rank, world_size, local_rank = setup_ddp()

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = tomlkit.loads(cfg_path.read_text('utf-8')).unwrap()

    dataset_root = Path(str(cfg['dataset']['root']))
    train_root = Path(str(cfg['train']['root']))
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

    # 2. 数据准备 (所有 Rank 执行相同的 glob 以保证文件列表一致)
    train_files = [{'image': p.as_posix()} for p in sorted((dataset_root / 'pre' / 'train').glob('*.nii.gz'))]
    train_files += [{'image': p.as_posix()} for p in sorted((dataset_root / 'post' / 'train').glob('*.nii.gz'))]
    val_files = [{'image': p.as_posix()} for p in sorted((dataset_root / 'pre' / 'val').glob('*.nii.gz'))]
    val_files += [{'image': p.as_posix()} for p in sorted((dataset_root / 'post' / 'val').glob('*.nii.gz'))]
    if rank == 0:
        print(f'Train: {len(train_files)}, Val: {len(val_files)}')

    # 截断数据集 (逻辑必须在所有 Rank 上一致)
    train_total = min(train_limit, len(train_files))
    val_total = min(val_limit, len(val_files))

    # 修复之前提到的步长为0的bug，并在所有Rank上统一截断
    train_files = train_files[:train_total]
    if val_total > 1:
        step = max(1, len(val_files) // (val_total - 1))
        val_files = val_files[::step][:val_total]
    else:
        val_files = val_files[:val_total]

    if rank == 0:
        print(f'Train limited: {len(train_files)}, Val limited: {len(val_files)}')

    # Transforms (需要确保 define.py 中的 bug 已修复)
    train_transforms = Compose(define.autoencoder_train_transforms(patch_size, bone_range[0]))
    val_transforms = Compose(define.autoencoder_val_transforms())

    train_ds = Dataset(train_files, train_transforms)
    val_ds = Dataset(val_files, val_transforms) if len(val_files) else None

    # 3. DDP Sampler 设置
    # shuffle=True 移到 Sampler 中
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if val_ds else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,  # DDP 下必须为 False
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers
    ) if val_ds else None

    # 4. 模型初始化与 DDP 封装
    # 生成器
    autoencoder = define.autoencoder().to(device)
    if dist.is_initialized():
        autoencoder = nn.SyncBatchNorm.convert_sync_batchnorm(autoencoder)
        autoencoder = DDP(autoencoder, device_ids=[local_rank], output_device=local_rank)

    # 判别器
    discriminator = define.discriminator().to(device)
    if dist.is_initialized():
        discriminator = nn.SyncBatchNorm.convert_sync_batchnorm(discriminator)
        discriminator = DDP(discriminator, device_ids=[local_rank], output_device=local_rank)

    # 损失函数
    L1Loss = torch.nn.L1Loss()
    AdversarialLoss = PatchAdversarialLoss(criterion='least_squares')
    # 感知损失 (冻结参数，不需要 DDP 封装，只要在正确 device 即可)
    PerceptualLoss = define.perceptual_loss().to(device)

    # 优化器
    optimizer_g = torch.optim.Adam(autoencoder.parameters(), lr=lr_g, betas=(0.5, 0.9))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.9))

    # 混合精度
    if use_amp:
        scaler_g = GradScaler()
        scaler_d = GradScaler()
    else:
        scaler_g = None
        scaler_d = None

    # 日志仅在主进程
    writer = None
    if rank == 0:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        writer = SummaryWriter(log_dir=(log_dir / timestamp).as_posix())

    start_epoch = 0
    best_val_ssim = -1.0

    # Resume 逻辑 (Rank 0 加载，或者所有 Rank 加载同一文件，DDP 会处理权重同步)
    # 建议所有 Rank 都加载，防止不同步
    if resume:
        load_pt = (ckpt_dir / f'{task}_last.pt').resolve()
        if load_pt.exists():
            try:
                # map_location 必须指定到当前 device
                checkpoint = torch.load(load_pt, map_location=device, weights_only=False)

                # 处理 DDP 带来的 'module.' 前缀问题
                # 如果 checkpoint 是 DDP 保存的，key 会有 module.；如果是单卡保存的，没有。
                # DDP 模型加载时期望有 module. 前缀。

                # 这里简单处理：直接加载。如果报错，需要根据 key 进行 strip 或 add module.
                # 假设保存时使用的是 autoencoder.state_dict() (包含或不包含 module 取决于保存方式)
                # 推荐：保存时使用 autoencoder.module.state_dict()，这样加载时通用性更好。
                # 下面的代码假设保存的是原始 state_dict (带不带module视情况而定)

                # 尝试直接加载
                try:
                    autoencoder.load_state_dict(checkpoint['state_dict'])
                except RuntimeError:
                    # 如果 key 不匹配 (多了或少了 module.)
                    new_state_dict = {}
                    for k, v in checkpoint['state_dict'].items():
                        if k.startswith('module.'):
                            new_state_dict[k[7:]] = v  # 去掉 module.
                        else:
                            new_state_dict[f'module.{k}'] = v  # 加上 module.
                    # 再次尝试，根据当前模型是否有 module 前缀决定
                    try:
                        autoencoder.load_state_dict(new_state_dict)
                    except:
                        # 最后的 fallback：如果是 DDP 模型，必须有 module.；如果不是，去掉。
                        if isinstance(autoencoder, DDP):
                            # 确保有 module.
                            final_dict = {f'module.{k}' if not k.startswith('module.') else k: v for k, v in
                                          checkpoint['state_dict'].items()}
                        else:
                            # 确保没有 module.
                            final_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
                        autoencoder.load_state_dict(final_dict)

                # 判别器同理
                try:
                    discriminator.load_state_dict(checkpoint['discriminator'])
                except:
                    if isinstance(discriminator, DDP):
                        final_dict = {f'module.{k}' if not k.startswith('module.') else k: v for k, v in
                                      checkpoint['discriminator'].items()}
                    else:
                        final_dict = {k.replace('module.', ''): v for k, v in checkpoint['discriminator'].items()}
                    discriminator.load_state_dict(final_dict)

                optimizer_g.load_state_dict(checkpoint['optimizer_g'])
                optimizer_d.load_state_dict(checkpoint['optimizer_d'])

                if use_amp and 'scaler_g' in checkpoint and 'scaler_d' in checkpoint:
                    scaler_g.load_state_dict(checkpoint['scaler_g'])
                    scaler_d.load_state_dict(checkpoint['scaler_d'])

                start_epoch = checkpoint['epoch'] + 1  # 从下一轮开始
                best_val_ssim = checkpoint.get('best_val_ssim', -1.0)
                if rank == 0:
                    print(f'Load from epoch {checkpoint['epoch']}, best_val_ssim {best_val_ssim}')
            except Exception as e:
                if rank == 0:
                    print(f'Load failed: {e}. Starting from scratch.')

    # 验证滑动窗口推理包装
    def encode_decode_mu(inputs):
        return autoencoder(inputs)[0]
        # if isinstance(autoencoder, DDP):
        #     return autoencoder.module.decode(autoencoder.module.encode(inputs)[0])
        # return autoencoder.decode(autoencoder.encode(inputs)[0])

    # 训练循环
    for epoch in range(start_epoch, num_epochs):
        # 5. 设置 Sampler Epoch 以保证 Shuffle
        train_sampler.set_epoch(epoch)

        warmup = epoch < warmup_epochs

        autoencoder.train()
        discriminator.train()

        epoch_loss_g = 0
        epoch_loss_d = 0
        step = 0

        # TQDM 仅在 Rank 0 显示
        if rank == 0:
            pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs - 1}', total=len(train_loader))
        else:
            pbar = train_loader

        for batch in pbar:
            step += 1
            images = batch['image'].to(device, non_blocking=True)

            optimizer_g.zero_grad(set_to_none=True)

            amp_ctx = autocast(device.type) if use_amp else nullcontext()
            with amp_ctx:
                # 注意：DDP forward 会自动同步
                if isinstance(autoencoder, DDP):
                    z_mu, z_sigma = autoencoder.module.encode(images)  # 显式调用 encode 避免 DDP forward 的困惑，或者在 forward 中实现
                    z = autoencoder.module.sampling(z_mu, z_sigma)
                    reconstruction = autoencoder.module.decode(z)
                else:
                    z_mu, z_sigma = autoencoder.encode(images)
                    z = autoencoder.sampling(z_mu, z_sigma)
                    reconstruction = autoencoder.decode(z)

                l1_loss = L1Loss(reconstruction.float(), images.float())

                if torch.isnan(l1_loss):
                    raise SystemExit(f'Rank {rank}: NaN in l1_loss')

                per_loss = PerceptualLoss(reconstruction.float(), images.float())

                if torch.isnan(per_loss):
                    raise SystemExit(f'Rank {rank}: NaN in per_loss')

                per_loss *= per_weight

                # kl_loss = 0.5 * torch.mean(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1)
                z_sigma_clamped = torch.clamp(z_sigma, min=1e-6, max=1e3)
                kl_loss = 0.5 * torch.mean(z_mu.pow(2) + z_sigma_clamped.pow(2) - torch.log(z_sigma_clamped.pow(2)) - 1)

                if torch.isnan(kl_loss):
                    raise SystemExit(f'Rank {rank}: NaN in kl_loss')

                kl_loss *= kl_weight

                loss_g = l1_loss + per_loss + kl_loss

                if not warmup:
                    out = discriminator(reconstruction.float())
                    adv_loss = AdversarialLoss(out, target_is_real=True, for_discriminator=False)

                    if torch.isnan(adv_loss):
                        raise SystemExit(f'Rank {rank}: NaN in adv_loss')

                    adv_loss *= adv_weight
                    loss_g += adv_loss
                else:
                    adv_loss = torch.tensor(0.0, device=device)

            if use_amp:
                scaler_g.scale(loss_g).backward()
                scaler_g.unscale_(optimizer_g)
                # DDP 下 clip_grad_norm 依然有效
                torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=1.0)
                scaler_g.step(optimizer_g)
                scaler_g.update()
            else:
                loss_g.backward()
                torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=1.0)
                optimizer_g.step()

            # 判别器训练
            loss_d = torch.tensor(0.0, device=device)
            if not warmup:
                optimizer_d.zero_grad(set_to_none=True)
                with amp_ctx:
                    logits_real = discriminator(images.detach())
                    loss_d_real = AdversarialLoss(logits_real, target_is_real=True, for_discriminator=True)

                    logits_fake = discriminator(reconstruction.detach())
                    loss_d_fake = AdversarialLoss(logits_fake, target_is_real=False, for_discriminator=True)

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

            epoch_loss_g += loss_g.item()

            # 日志 (仅 Rank 0)
            if rank == 0:
                global_step = epoch * len(train_loader) + step

                postfix = {'L1': f'{l1_loss.item():.4f}', 'Per.': f'{per_loss.item():.4f}',
                           'KL': f'{kl_loss.item():.4f}'}

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

        if rank == 0:
            writer.add_scalar('train/epoch_loss_g', epoch_loss_g / len(train_loader), epoch)

        # 6. 分布式验证
        if val_loader and epoch % val_interval == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            autoencoder.eval()

            # 本地累积
            local_l1_loss = torch.tensor(0.0, device=device)
            local_psnr = torch.tensor(0.0, device=device)
            local_ssim = torch.tensor(0.0, device=device)
            local_steps = torch.tensor(0.0, device=device)

            PSNR_Calc = PSNRMetric(max_val=2.0)
            SSIM_Calc = SSIMMetric(data_range=2.0, spatial_dims=3)

            # 仅 Rank 0 显示进度条
            if rank == 0:
                val_pbar = tqdm(val_loader, desc='Val', leave=False)
            else:
                val_pbar = val_loader

            with torch.no_grad():
                for i, batch in enumerate(val_pbar):
                    images = batch['image'].to(device, non_blocking=True)

                    reconstruction = sliding_window_inference(
                        inputs=images,
                        roi_size=patch_size,
                        sw_batch_size=sw_batch_size,
                        predictor=encode_decode_mu,
                        overlap=0.25,
                        mode='gaussian',
                        device=device,
                        sw_device=device,
                        progress=False,
                    )

                    l1 = L1Loss(reconstruction.float(), images.float())
                    local_l1_loss += l1

                    # PSNR/SSIM 需要在每个 batch 后重置或者单独计算
                    # 这里 MONAI Metric 对象如果直接调用不aggregate会返回 batch 结果
                    p_val = PSNR_Calc(y_pred=reconstruction, y=images)
                    s_val = SSIM_Calc(y_pred=reconstruction, y=images)

                    local_psnr += p_val.mean()  # 取 batch 平均
                    local_ssim += s_val.mean()
                    local_steps += 1

                    PSNR_Calc.reset()
                    SSIM_Calc.reset()

                    # 可视化仅在 Rank 0 做一次
                    if rank == 0 and i == 0:
                        image_vis = images[0] * 0.5 + 0.5
                        recon_vis = reconstruction[0] * 0.5 + 0.5
                        diff_vis = torch.abs(image_vis - recon_vis)
                        z_idx = image_vis.shape[1] // 2
                        writer.add_image('val/CT_Input', image_vis[0, z_idx].unsqueeze(0), epoch)
                        writer.add_image('val/CT_Recon', recon_vis[0, z_idx].unsqueeze(0), epoch)
                        writer.add_image('val/CT_Diff', diff_vis[0, z_idx].unsqueeze(0), epoch)

            # 7. 汇总多卡指标
            # 将 local sum 聚合
            global_l1_sum = reduce_tensor(local_l1_loss, 1)  # 先只做 sum
            global_psnr_sum = reduce_tensor(local_psnr, 1)
            global_ssim_sum = reduce_tensor(local_ssim, 1)
            global_steps_sum = reduce_tensor(local_steps, 1)

            # 在所有 Rank 上同步计算平均值
            val_l1_avg = global_l1_sum / global_steps_sum
            val_psnr_avg = global_psnr_sum / global_steps_sum
            val_ssim_avg = global_ssim_sum / global_steps_sum

            # 转回 float python 对象
            val_l1_val = val_l1_avg.item()
            psnr_val = val_psnr_avg.item()
            ssim_val = val_ssim_avg.item()

            if rank == 0:
                writer.add_scalar('val/l1', val_l1_val, epoch)
                writer.add_scalar('val/psnr', psnr_val, epoch)
                writer.add_scalar('val/ssim', ssim_val, epoch)

                print(f'Val Epoch {epoch}: L1={val_l1_val:.5f} | PSNR={psnr_val:.4f} | SSIM={ssim_val:.4f}')

                # 保存模型 (建议保存 autoencoder.module)
                to_save_ae = autoencoder.module if isinstance(autoencoder, DDP) else autoencoder
                to_save_disc = discriminator.module if isinstance(discriminator, DDP) else discriminator

                checkpoint = {
                    'epoch': epoch,
                    'state_dict': to_save_ae.state_dict(),
                    'discriminator': to_save_disc.state_dict(),
                    'optimizer_g': optimizer_g.state_dict(),
                    'optimizer_d': optimizer_d.state_dict(),
                    'best_val_ssim': best_val_ssim,
                    'val_l1': val_l1_val,
                    'val_psnr': psnr_val,
                    'val_ssim': ssim_val,
                }

                if use_amp:
                    checkpoint['scaler_g'] = scaler_g.state_dict()
                    checkpoint['scaler_d'] = scaler_d.state_dict()

                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save(checkpoint, ckpt_dir / f'{task}_last.pt')

                if ssim_val > best_val_ssim:
                    best_val_ssim = ssim_val
                    checkpoint['best_val_ssim'] = best_val_ssim
                    torch.save(checkpoint, ckpt_dir / f'{task}_best.pt')
                    print('New best model saved!')

            # 必须等待 rank 0 保存完，虽然这里不需要 barrier 但为了逻辑严谨可以加
            dist.barrier()

    if rank == 0:
        writer.close()
        print('Training Completed.')

    cleanup_ddp()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        if dist.is_initialized():
            dist.destroy_process_group()
        print('Keyboard interrupted terminating...')
