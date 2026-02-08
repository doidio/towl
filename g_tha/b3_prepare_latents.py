import argparse
from pathlib import Path

import numpy as np
import tomlkit
import torch
from monai.inferers import sliding_window_inference
from monai.transforms import Compose
from torch.amp import autocast
from tqdm import tqdm

import define

try:
    import torch_musa

    device = torch.device('musa' if torch.musa.is_available() else 'cpu')  # noqa
except ImportError:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def process_split(files, out_dir, autoencoder, transforms, patch_size, use_amp, scale_factor):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 定义编码预测器
    def encode_predictor(inputs):
        return autoencoder.encode(inputs)[0]

    for file_path in tqdm(files, desc=f'Processing {out_dir.parent.name}/{out_dir.name}'):
        stem = file_path.name.replace('.nii.gz', '').replace('.nii', '')
        save_path = out_dir / f'{stem}.npy'

        # 如果已经存在，跳过 (支持断点续传)
        if save_path.exists():
            continue

        item = {'image': file_path.as_posix()}
        data = transforms(item)

        if isinstance(data['image'], np.ndarray):
            data['image'] = torch.from_numpy(data['image'])

        images = data['image'].unsqueeze(0).to(device)

        with torch.no_grad():
            amp_ctx = autocast(device.type) if use_amp else torch.no_grad()
            with amp_ctx:
                # 使用滑动窗口编码 (Encode)
                # Overlap 0.25 足够用于编码，因为卷积提取特征时的边缘效应较小
                z_mu_stitched = sliding_window_inference(
                    inputs=images,
                    roi_size=patch_size,
                    sw_batch_size=4,
                    predictor=encode_predictor,
                    overlap=0.25,
                    mode='gaussian',
                    device=device,
                    progress=False
                )

                # 乘上 Scale Factor
                z = z_mu_stitched * scale_factor

        latent_np = z.cpu().numpy().astype(np.float32)
        np.save(save_path, latent_np)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    cfg = tomlkit.loads(Path(args.config).read_text('utf-8')).unwrap()
    dataset_root = Path(cfg['dataset']['root'])
    train_root = Path(cfg['train']['root'])
    ckpt_dir = train_root / 'checkpoints'
    latents_root = train_root / 'latents'

    task = 'autoencoder'
    patch_size = cfg['train'][task]['patch_size']
    use_amp = cfg['train'][task]['use_amp']

    # 1. 加载模型
    print(f'Loading model to {device}...')
    autoencoder = define.autoencoder().to(device)
    load_pt = ckpt_dir / f'{task}_best.pt'

    try:
        checkpoint = torch.load(load_pt, map_location=device)
        autoencoder.load_state_dict(checkpoint['state_dict'])
        scale_factor = checkpoint['scale_factor']
        print(f'Model loaded. Scale Factor: {scale_factor}')
    except Exception as e:
        raise SystemExit(f'Failed to load checkpoint: {e}')

    autoencoder.eval()
    base_transforms = Compose(define.autoencoder_val_transforms())

    # 2. 遍历所有数据集划分
    splits = [
        ('pre', 'train'),
        ('pre', 'val'),
        ('post', 'train'),
        ('post', 'val')
    ]

    print('--- Starting Batch Latent Generation ---')
    for phase, split in splits:
        src_dir = dataset_root / phase / split
        dst_dir = latents_root / phase / split

        files = list(src_dir.glob('*.nii.gz'))
        if not files:
            continue

        process_split(files, dst_dir, autoencoder, base_transforms, patch_size, use_amp, scale_factor)

    print('All latents prepared. You are ready for LDM training!')


if __name__ == '__main__':
    main()
