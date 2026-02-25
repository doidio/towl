import argparse
import os
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import tomlkit
import torch
import torch.distributed as dist
from monai.data import DataLoader, Dataset
from monai.transforms import Compose, SaveImage
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DdP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image as save_image_tensor
from tqdm import tqdm

import define


# --- DDP 工具函数 ---
def setup_ddp():
    """初始化分布式环境 (通过 torchrun 启动会自动设置环境变量)"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        # 初始化进程组，使用 NCCL 后端 (NVIDIA GPU 标准)
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    else:
        print("未检测到 DDP 环境，回退到单卡模式 (不推荐用于多机)")
        return 0, 0, 1


def cleanup_ddp():
    dist.destroy_process_group()


def reduce_tensor(tensor, world_size):
    """将多卡的数值聚合平均，用于 Log 显示"""
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def main():
    # 1. DDP 初始化
    global_rank, local_rank, world_size = setup_ddp()
    is_master = (global_rank == 0)  # 是否为主进程
    device = torch.device(f"cuda:{local_rank}")

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = tomlkit.loads(cfg_path.read_text('utf-8')).unwrap()

    train_root = Path(str(cfg['train']['root']))
    log_dir = train_root / 'logs'
    ckpt_dir = train_root / 'checkpoints'

    task = 'ldm'
    (
        use_amp, resume, num_workers, num_epochs, val_interval, val_limit,
        batch_size, sw_batch_size, lr, gradient_accumulation_steps,
    ) = [cfg['train'][task][_] for _ in (
        'use_amp', 'resume', 'num_workers', 'num_epochs', 'val_interval', 'val_limit',
        'batch_size', 'sw_batch_size', 'lr', 'gradient_accumulation_steps',
    )]

    # patch_size = cfg['train']['vae']['patch_size']

    # 2. 数据准备 (所有节点必须看到相同的文件列表，sorted 很关键)
    train_files = [{'image': p.as_posix()} for p in sorted((train_root / 'latents' / 'train').glob('*.npy'))]
    val_files = [{'image': p.as_posix()} for p in sorted((train_root / 'latents' / 'val').glob('*.npy'))]

    if is_master:
        print(f'Train Total: {len(train_files)}, Val Total: {len(val_files)}')

    # 验证集降采样 (保持确定性)
    val_total = min(val_limit, len(val_files))
    if len(val_files) > 0:
        val_files = val_files[::len(val_files) // (val_total - 1) if (val_total - 1) > 0 else 1]
        val_files = val_files[:val_total]  # 确保不溢出

    if is_master:
        print(f'Val Limited: {len(val_files)}')

    transforms = Compose(define.ldm_transforms())
    train_ds = Dataset(data=train_files, transform=transforms)
    val_ds = Dataset(data=val_files, transform=transforms)

    # 3. DataLoader 与 Sampler
    # shuffle=False 是必须的，因为 Sampler 会负责打乱
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=global_rank, shuffle=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=global_rank, shuffle=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=False,
        sampler=train_sampler,
        num_workers=num_workers, pin_memory=True, persistent_workers=True,
    )
    # 验证集通常不需要多卡并行采样，但为了计算 Val Loss 方便，这里也做并行
    val_loader = DataLoader(val_ds, batch_size=1, sampler=val_sampler, num_workers=num_workers)

    # 4. 模型加载与 DDP 包装
    ldm = define.ldm_unet().to(device)

    # SyncBatchNorm 在 3D U-Net 中通常不是必须的，除非 Batch Size 极小且使用了 BatchNorm (MONAI 默认 GroupNorm)
    ldm = DdP(ldm, device_ids=[local_rank], output_device=local_rank)

    scheduler = define.scheduler_ddpm()
    num_train_timesteps = scheduler.num_train_timesteps

    optimizer = torch.optim.AdamW(ldm.parameters(), lr=lr, weight_decay=1e-5)
    scaler = GradScaler() if use_amp else None

    # 加载 VAE (仅用于验证时的解码，放在 CPU 或 GPU 均可，这里放 GPU 方便解码)
    vae_ckpt_path = ckpt_dir / 'vae_best.pt'
    if is_master:
        print(f'Loading VAE from {vae_ckpt_path.resolve()}')

    # 确保所有节点都能读到文件
    vae_ckpt = torch.load(vae_ckpt_path, map_location=device)
    vae = define.vae_kl().to(device)  # 解码尽量用 GPU
    vae.load_state_dict(vae_ckpt['state_dict'])
    vae.eval().float()
    scale_factor = vae_ckpt['scale_factor']

    start_epoch = 0
    best_val_loss = float('inf')
    ldm_ckpt_path = ckpt_dir / f'{task}_last.pt'

    # 断点续训逻辑
    if resume and ldm_ckpt_path.exists():
        # map_location 确保加载到当前 GPU
        ckpt = torch.load(ldm_ckpt_path, map_location=device)

        # DDP 模型 state_dict 会多一个 'module.' 前缀，需要处理
        # 如果保存的是 DDP 状态，直接加载；如果保存的是 ldm.module，也需要适配
        state_dict = ckpt['state_dict']
        # 简单处理：现在的代码保存的是 ldm.state_dict() (即 module)，DDP 包装后需要 .module
        # 但如果是 DDP wrapper 后直接 .load_state_dict，key 需要 'module.' 前缀
        # 最佳实践：保存时存 ldm.module.state_dict()，加载时 ldm.module.load...
        # 这里兼容旧代码的逻辑 (假设旧代码保存的是单机模型)
        if hasattr(ldm, 'module'):
            # 如果加载的 key 没有 module. 前缀，加上它；或者直接对 ldm.module 加载
            try:
                ldm.module.load_state_dict(state_dict)
            except RuntimeError:
                ldm.load_state_dict(state_dict)  # 尝试直接加载
        else:
            ldm.load_state_dict(state_dict)

        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        if use_amp and 'scaler' in ckpt:
            scaler.load_state_dict(ckpt['scaler'])

        if is_master:
            print(f'Resuming from epoch {start_epoch}, best_val_loss {best_val_loss}')

    # TensorBoard & Saver (仅 Master)
    writer = None
    saver = None
    if is_master:
        timestamp = datetime.now().strftime(f'{task}_%Y%m%d_%H%M%S')
        writer = SummaryWriter(log_dir=(log_dir / timestamp).as_posix())
        saver = SaveImage(
            output_dir=log_dir, output_postfix='', output_ext='.nii.gz',
            separate_folder=False, print_log=False, resample=False,
        )

    def decode_and_save(z, name):
        """辅助函数：解码并保存 NIFTI (仅 Rank 0 调用)"""
        z = (z / scale_factor).float()
        with torch.no_grad():
            # 这里的 vae 已经在 GPU 上
            img_recon = vae.decode(z)

        # 转回 CPU 保存
        img_recon = img_recon.detach().cpu()
        saver(img_recon[0], meta_data={'filename_or_obj': f'val_epoch_{epoch:03d}_{name}.nii.gz'})
        return img_recon

    amp_ctx = autocast(device.type) if use_amp else nullcontext()

    for epoch in range(start_epoch, num_epochs):
        # 重要：每个 Epoch 开始前设置 Sampler 的 epoch，保证 shuffle 随机性
        train_sampler.set_epoch(epoch)

        ldm.train()
        epoch_loss = torch.zeros(1).to(device)  # 用 Tensor 方便 reduce
        step = 0

        # tqdm 仅在 Master 显示
        if is_master:
            pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs - 1}')
        else:
            pbar = train_loader

        for batch in pbar:
            step += 1
            image = batch['image'].to(device) * scale_factor
            cond = batch['condition'].to(device) * scale_factor

            with amp_ctx:
                timesteps = torch.randint(0, num_train_timesteps, (image.shape[0],), device=device).long()
                noise = torch.randn_like(image)
                noisy_image = scheduler.add_noise(original_samples=image, noise=noise, timesteps=timesteps)
                input_tensor = torch.cat([noisy_image, cond], dim=1)

                noise_pred = ldm(x=input_tensor, timesteps=timesteps)

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
            else:
                loss.backward()
                if step % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(ldm.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            # 记录 Loss
            current_loss = loss.detach() * gradient_accumulation_steps
            epoch_loss += current_loss

            if is_master:
                # 这里的 Loss 是单卡的，为了性能不每步都 Reduce，仅本地打印参考
                pbar.set_postfix({'MSE': f'{current_loss.item():.4f}'})

                # TensorBoard 每步记录 (可选：减少频率)
                global_step = epoch * len(train_loader) + step
                if step % 10 == 0:
                    writer.add_scalar('train/loss_local', current_loss.item(), global_step)

        # Epoch 结束，计算全局平均 Loss
        dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
        avg_epoch_loss = epoch_loss.item() / (len(train_loader) * world_size)  # 近似

        if is_master:
            writer.add_scalar('train/epoch_loss', avg_epoch_loss, epoch)
            print(f"Epoch {epoch} Average Loss: {avg_epoch_loss:.5f}")

        # --- 验证与采样 ---
        if epoch % val_interval == 0:
            ldm.eval()
            val_loss = torch.zeros(1).to(device)
            val_steps = torch.zeros(1).to(device)

            # 1. 计算验证集 Loss (多卡并行计算)
            with torch.no_grad():
                # 注意：DDP 验证时，若 Dataset 无法被 BatchSize 整除，DistributedSampler 可能会重复采样补齐
                # 这里的 Metric 计算是近似值
                loader_iter = tqdm(val_loader, desc='Val Loss') if is_master else val_loader
                for batch in loader_iter:
                    image = batch['image'].to(device) * scale_factor
                    cond = batch['condition'].to(device) * scale_factor
                    timesteps = torch.randint(0, num_train_timesteps, (image.shape[0],), device=device).long()
                    noise = torch.randn_like(image)
                    noisy_image = scheduler.add_noise(image, noise, timesteps)
                    input_tensor = torch.cat([noisy_image, cond], dim=1)

                    with amp_ctx:
                        noise_pred = ldm(input_tensor, timesteps)
                        loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float())

                    val_loss += loss
                    val_steps += 1

            # 聚合所有卡的 Loss
            dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_steps, op=dist.ReduceOp.SUM)
            avg_val_loss = val_loss.item() / val_steps.item()

            if is_master:
                writer.add_scalar('val/loss', avg_val_loss, epoch)
                print(f'Val Loss: {avg_val_loss:.5f}')

            # 2. 生成图像 (仅 Master 运行，避免重复计算和保存冲突)
            if is_master:
                # 从验证集中取一个样本 (不使用 Loader，直接取 Dataset[0])
                # 注意：Dataset 每次 getitem 都会做 Transform，可能有随机性，但在 Val Transform 中通常是确定的
                sample_data = val_ds[0]  # 取第一个样本

                # 增加 Batch 维度并送到 GPU
                image = sample_data['image'].unsqueeze(0).to(device) * scale_factor
                cond = sample_data['condition'].unsqueeze(0).to(device) * scale_factor

                val_scheduler = define.scheduler_ddim()
                val_scheduler.set_timesteps(num_inference_steps=50, device=device)

                generator = torch.Generator(device=device).manual_seed(42)  # 固定随机种子
                generated = torch.randn(image.shape, device=device, generator=generator)
                # generated = torch.randn_like(image)

                # 采样进度条
                for t in tqdm(val_scheduler.timesteps, desc='Sampling'):
                    model_input = torch.cat([generated, cond], dim=1)
                    with torch.no_grad():
                        model_output = ldm(model_input, t[None].to(device))
                    generated, _ = val_scheduler.step(model_output, t, generated)

                # 解码与可视化
                vis_generated = decode_and_save(generated, 'Generated')

                # Helper: Normalize for TensorBoard
                def norm_vis(x):
                    x_min, x_max = x.min(), x.max()
                    return (x - x_min) / (x_max - x_min + 1e-5)

                z_idx = vis_generated.shape[2] // 2
                writer.add_image('val/Generated', _ := norm_vis(vis_generated[0, 0, z_idx]), epoch, dataformats='HW')
                save_image_tensor(_, log_dir / f'val_epoch_{epoch:03d}_Generated.png')

                if epoch == 0:
                    vis_gt = decode_and_save(image, 'GroundTruth')
                    writer.add_image('val/GroundTruth', _ := norm_vis(vis_gt[0, 0, z_idx]), epoch, dataformats='HW')
                    save_image_tensor(_, log_dir / f'val_epoch_{epoch:03d}_GroundTruth.png')

                    vis_cond = decode_and_save(cond, 'Condition')
                    writer.add_image('val/Condition', _ := norm_vis(vis_cond[0, 0, z_idx]), epoch, dataformats='HW')
                    save_image_tensor(_, log_dir / f'val_epoch_{epoch:03d}_Condition.png')

            # 3. 保存模型 (仅 Master)
            if is_master:
                # 保存时剥离 DDP 的 module
                model_to_save = ldm.module if hasattr(ldm, 'module') else ldm

                ckpt = {
                    'epoch': epoch,
                    'state_dict': model_to_save.state_dict(),
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

        # 同步等待，确保下一轮开始前所有进程都完成了保存或打印
        dist.barrier()
        torch.cuda.empty_cache()

    if is_master:
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
