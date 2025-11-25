import argparse
from datetime import datetime
from pathlib import Path

import monai
import numpy as np
import tomlkit
import torch
from monai.apps import MedNISTDataset
from monai.data import CacheDataset, DataLoader
from monai.inferers import DiffusionInferer
from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler
from tqdm import tqdm


def train_dataset(root: str, batch_size: int, num_workers: int, spatial_size: list | tuple):

    train_data = MedNISTDataset(root_dir=root, section='training', download=True, progress=True, seed=0)
    train_data = [{'image': item['image']} for item in train_data.data if item['class_name'] == 'Hand']

    train_transforms = monai.transforms.Compose(
        [
            monai.transforms.LoadImaged(keys=['image']),
            monai.transforms.EnsureChannelFirstd(keys=['image']),
            monai.transforms.ScaleIntensityRanged(
                keys=['image'], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True,
            ),
            monai.transforms.RandAffined(
                keys=['image'],
                spatial_size=spatial_size,
                rotate_range=[(-np.pi / 36, np.pi / 36), (-np.pi / 36, np.pi / 36)],
                translate_range=[(-1, 1), (-1, 1)],
                scale_range=[(-0.05, 0.05), (-0.05, 0.05)],
                padding_mode='zeros',
                prob=0.5,
            ),
        ]
    )
    train_ds = CacheDataset(data=train_data, transform=train_transforms)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=True
    )
    return train_loader

def val_dataset(root: str, batch_size: int, num_workers: int):
    val_data = MedNISTDataset(root_dir=root, section='validation', download=True, progress=False, seed=0)
    val_data = [{'image': item['image']} for item in val_data.data if item['class_name'] == 'Hand']
    val_transforms = monai.transforms.Compose(
        [
            monai.transforms.LoadImaged(keys=['image']),
            monai.transforms.EnsureChannelFirstd(keys=['image']),
            monai.transforms.ScaleIntensityRanged(keys=['image'], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
        ]
    )
    val_ds = CacheDataset(data=val_data, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            persistent_workers=True)
    return val_loader


def main(cfg_path: str, num_workers: int, device: str = 'cuda'):
    cfg_path = Path(cfg_path)
    cfg = tomlkit.loads(cfg_path.read_text('utf-8'))

    root = Path(cfg['train']['root'])
    if not root.is_absolute():
        root = cfg_path.parent / root
    root = root.resolve()

    root.mkdir(parents=True, exist_ok=True)

    final_pth = root / 'checkpoints' / 'final.pth'
    batch_size = cfg['train']['batch_size']
    train_epochs = cfg['train']['epochs']
    val_interval = cfg['train']['val_interval']

    # 初始化模型，或载入上次训练存档
    if final_pth.is_file() and final_pth.exists():
        ckpt = torch.load(final_pth, map_location=device)
        print(f'Load checkpoint {final_pth.as_posix()}')
    else:
        ckpt = {}

    if len(ckpt):
        model = DiffusionModelUNet(**ckpt['model']).to(device)
        scheduler = DDPMScheduler(num_train_timesteps=ckpt['scheduler']['num_train_timesteps'])
    else:
        model = DiffusionModelUNet(
            spatial_dims=2,  # 图像维数
            in_channels=1,  # 输入通道
            out_channels=1,  # 输出通道，与输入通道的主通道数一致
            channels=(128, 256, 256),  # 网络复杂度，单调递增，越大显存越高
            attention_levels=(False, True, True),  # 空间自注意力，一般在后两层
            num_res_blocks=1,  # 1 或 2
            num_head_channels=256,  # 注意力头的通道数，越小显存越高
        ).to(device)
        scheduler = DDPMScheduler(num_train_timesteps=1000)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=2.5e-5)
    infer = DiffusionInferer(scheduler)
    scaler = torch.amp.GradScaler(device)

    if len(ckpt):
        model.load_state_dict(ckpt['state']['model'])
        optimizer.load_state_dict(ckpt['state']['optimizer'])
        scheduler.load_state_dict(ckpt['state']['scheduler'])
        scaler.load_state_dict(ckpt['state']['scaler'])
        train_losses = list(ckpt['loss']['train'])
        val_losses = list(ckpt['loss']['val'])
    else:
        train_losses = []
        val_losses = []

    # 数据集
    dataset = ckpt.get('dataset', {})
    spatial_size = dataset.get('spatial_size', [64, 64])

    train_loader = train_dataset(root.as_posix(), batch_size, num_workers, spatial_size)
    val_loader = val_dataset(root.as_posix(), batch_size, num_workers)

    # 训练
    start_epoch = len(train_losses)

    total_start = datetime.now()
    for epoch in range(start_epoch, start_epoch + train_epochs):
        model.train()
        train_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        progress_bar.set_description(f'Epoch {epoch + 1} train')

        loss = None
        batch: dict
        for step, batch in progress_bar:
            images = batch['image'].to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device, enabled=True):
                # Generate random noise
                noise = torch.randn_like(images).to(device)

                # Create timesteps
                timesteps = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],),
                                          device=images.device).long()

                # Get model prediction
                noise_predict = infer(inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps)

                loss = torch.nn.functional.mse_loss(noise_predict.float(), noise.float())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            loss = train_loss / (step + 1)
            progress_bar.set_postfix({'loss': loss})

        progress_bar.close()
        if loss is not None:
            train_losses.append(loss)

        # 保存最后存档
        final_pth.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                'datetime': datetime.now().isoformat(),
                'monai': dict(
                    version=str(monai.__version__),
                    revision_id=str(monai.__revision_id__),
                ),
                'model': dict(
                    spatial_dims=int(model.conv_in.spatial_dims),
                    in_channels=int(model.conv_in.in_channels),
                    out_channels=int(model.out_channels),
                    channels=[int(_) for _ in model.block_out_channels],
                    attention_levels=[bool(_) for _ in model.attention_levels],
                    num_res_blocks=[int(_) for _ in model.num_res_blocks],
                    num_head_channels=[int(_) for _ in model.num_head_channels],
                ),
                'scheduler': dict(
                    num_train_timesteps=int(scheduler.num_train_timesteps),
                ),
                'state': dict(
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),
                    scaler=scaler.state_dict(),
                ),
                'loss': dict(
                    train=[float(_) for _ in train_losses],
                    val=[float(_) for _ in val_losses],
                ),
                'dataset': dict(
                    spatial_size=[int(_) for _ in spatial_size],
                ),
            },
            final_pth,
        )

        if (epoch + 1) % val_interval == 0:
            model.eval()
            val_loss = 0

            val_bar = tqdm(enumerate(val_loader), total=len(val_loader))
            val_bar.set_description(f'Epoch {epoch + 1} val')

            loss = None
            for step, batch in val_bar:
                images = batch['image'].to(device)
                with torch.no_grad(), torch.amp.autocast(device_type=device, enabled=True):
                    noise = torch.randn_like(images).to(device)
                    timesteps = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],),
                                              device=images.device).long()
                    noise_predict = infer(inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps)
                    loss = torch.nn.functional.mse_loss(noise_predict.float(), noise.float())

                val_loss += loss.item()

                loss = val_loss / (step + 1)
                val_bar.set_postfix({'loss': loss})

            val_bar.close()
            if loss is not None:
                val_losses.append(loss)

    total_time = datetime.now() - total_start
    print(f'Train completed, total time: {total_time}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--num_workers', type=int, default=16)
    args = parser.parse_args()

    # monai.config.print_config()
    assert torch.cuda.is_available()

    try:
        main(args.config, args.num_workers)
    except KeyboardInterrupt:
        print('Keyboard interrupted')
