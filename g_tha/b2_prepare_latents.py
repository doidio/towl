import argparse
import warnings
from contextlib import nullcontext
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--sw_batch_size', default=36)
    args = parser.parse_args()

    cfg = tomlkit.loads(Path(args.config).read_text('utf-8')).unwrap()
    dataset_root = Path(cfg['dataset']['root'])
    train_root = Path(cfg['train']['root'])
    ckpt_dir = train_root / 'checkpoints'

    latents_root = train_root / 'latents'

    task = 'vae'
    patch_size = cfg['train'][task]['patch_size']
    use_amp = cfg['train'][task]['use_amp']

    vae = define.vae().to(device)
    load_pt = ckpt_dir / f'{task}_best.pt'

    try:
        print(f'Loading checkpoint from {load_pt}...')
        checkpoint = torch.load(load_pt, map_location=device)
        vae.load_state_dict(checkpoint['state_dict'])

        if 'scale_factor' in checkpoint:
            scale_factor = checkpoint['scale_factor']
            print(f'Scale Factor: {scale_factor}')
        else:
            raise SystemExit(f'Scale factor not prepared')
    except Exception as e:
        raise SystemExit(f'Failed to load checkpoint: {e}')

    vae.eval()
    transforms = Compose(define.vae_val_transforms())

    def encode_predictor(inputs):
        return vae.encode(inputs)[0]

    for subdir in ['train', 'val']:
        post_dir = dataset_root / 'post' / subdir
        pre_dir = dataset_root / 'pre' / subdir
        out_dir = latents_root / subdir

        out_dir.mkdir(parents=True, exist_ok=True)
        post_files = sorted(list(post_dir.glob('*.nii.gz')))

        for post_path in tqdm(post_files, desc=subdir):
            filename = post_path.name
            pre_path = pre_dir / filename

            if not pre_path.exists():
                warnings.warn(f'Pre-op file not found for {filename}. Skipping.')
                continue

            stem = filename.replace('.nii.gz', '').replace('.nii', '')
            save_path = out_dir / f'{stem}.npy'
            if save_path.exists():
                continue

            z_post_pre = []
            for path in (post_path, pre_path):
                batch = transforms({'image': path.as_posix()})['image']

                if isinstance(batch, np.ndarray):
                    batch = torch.from_numpy(batch)

                images = batch.unsqueeze(0).to(device)

                with (torch.no_grad()):
                    with autocast(device.type) if use_amp else nullcontext():
                        z = sliding_window_inference(
                            inputs=images,
                            roi_size=patch_size,
                            sw_batch_size=args.sw_batch_size,
                            predictor=encode_predictor,
                            overlap=0.25,
                            mode='gaussian',
                            device=device,
                            progress=False,
                        )
                        z *= scale_factor

                z = z.cpu().numpy()
                z_post_pre.append(z)

            z_post, z_pre = z_post_pre

            if z_post.shape != z_pre.shape:
                warnings.warn(f'Shape mismatch for {filename} Post {z_post.shape} Pre {z_pre.shape}. Skipping.')
                continue

            # 通道拼接，[0:4]术后[4:8]术前，使用 float16 节省空间 Latent 精度足够
            z_cat = np.concatenate((z_post[0], z_pre[0]), axis=0).astype(np.float16)
            np.save(save_path, z_cat)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Keyboard interrupted terminating...')
