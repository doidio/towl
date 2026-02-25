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
    parser.add_argument('--sw_batch_size', default=12)
    args = parser.parse_args()

    cfg = tomlkit.loads(Path(args.config).read_text('utf-8')).unwrap()
    dataset_root = Path(cfg['dataset']['root'])
    train_root = Path(cfg['train']['root'])
    ckpt_dir = train_root / 'checkpoints'

    latents_root = train_root / 'latents'

    task = 'vae'
    patch_size = cfg['train'][task]['patch_size']
    use_amp = cfg['train'][task]['use_amp']

    def load_vae(subtask):
        vae_model = define.vae_kl().to(device)
        load_pt = ckpt_dir / f'{task}_{subtask}_best.pt'

        print(f'Loading checkpoint from {load_pt}...')
        try:
            checkpoint = torch.load(load_pt, map_location=device, weights_only=True)
            vae_model.load_state_dict(checkpoint['state_dict'])

            if 'scale_factor' in checkpoint:
                scale_factor = checkpoint['scale_factor']
                global_mean = checkpoint.get('global_mean', 0.0)
                print(f'Scale Factor ({subtask}): {scale_factor:.6f}, Mean: {global_mean:.6f}')
            else:
                raise SystemExit(f'Scale factor not prepared for {subtask}')

            vae_model.eval()
            return vae_model, scale_factor, global_mean
        except Exception as e:
            raise SystemExit(f'Failed to load {subtask} checkpoint: {e}')

    vae_metal, sf_metal, mean_metal = load_vae('metal')
    vae_pre, sf_pre, mean_pre = load_vae('pre')

    transforms_metal = Compose(define.vae_val_transforms('metal'))
    transforms_pre = Compose(define.vae_val_transforms('pre'))

    def encode_predictor(model):
        def _predictor(inputs: torch.Tensor) -> torch.Tensor:
            return model.encode(inputs)[0]

        return _predictor

    for subdir in ['train', 'val']:
        metal_dir = dataset_root / 'metal' / subdir
        pre_dir = dataset_root / 'pre' / subdir
        out_dir = latents_root / subdir

        out_dir.mkdir(parents=True, exist_ok=True)
        metal_files = sorted(list(metal_dir.glob('*.nii.gz')))

        for metal_path in tqdm(metal_files, desc=subdir):
            filename = metal_path.name
            pre_path = pre_dir / filename

            if not pre_path.exists():
                warnings.warn(f'Pre-op file not found for {filename}. Skipping.')
                continue

            stem = filename.replace('.nii.gz', '').replace('.nii', '')
            save_path = out_dir / f'{stem}.npy'
            if save_path.exists():
                continue

            z_metal_pre = []
            for path, vae, sf, mean, transforms in [
                (metal_path, vae_metal, sf_metal, mean_metal, transforms_metal),
                (pre_path, vae_pre, sf_pre, mean_pre, transforms_pre)
            ]:
                batch = transforms({'image': path.as_posix()})['image']

                if isinstance(batch, np.ndarray):
                    batch = torch.from_numpy(batch)

                images = batch.unsqueeze(0).to(device)

                with torch.no_grad():
                    amp_ctx = autocast(device.type) if use_amp else nullcontext()
                    with amp_ctx:
                        z = sliding_window_inference(
                            inputs=images,
                            roi_size=patch_size,
                            sw_batch_size=args.sw_batch_size,
                            predictor=encode_predictor(vae),
                            overlap=0.25,
                            mode='gaussian',
                            device=device,
                            sw_device=device,
                            progress=False,
                        )

                # 不应用 scale factor，保持原始 latent 存盘
                z = z.cpu().numpy()
                z_metal_pre.append(z)

            z_metal, z_pre = z_metal_pre

            if z_metal.shape != z_pre.shape:
                warnings.warn(f'Shape mismatch for {filename} Metal {z_metal.shape} Pre {z_pre.shape}. Skipping.')
                continue

            # 通道拼接，[0:4] 目标 metal, [4:8] 条件 pre
            z_cat = np.concatenate((z_metal[0], z_pre[0]), axis=0).astype(np.float32)
            np.save(save_path, z_cat)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Keyboard interrupted terminating...')
