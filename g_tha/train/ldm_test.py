import time
from pathlib import Path

import itk
import numpy as np
import torch

try:
    import torch_musa

    device = torch.device('musa' if torch.musa.is_available() else 'cpu')
except ImportError:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from monai.inferers import LatentDiffusionInferer, DiffusionInferer
from monai.networks.nets import AutoencoderKL, DiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler


def restore_ct(tensor, ct_range):
    '''
    将 [-1, 1] 的模型输出还原回 [min_hu, max_hu]
    公式: x_hu = (x_norm + 1) / 2 * (max - min) + min
    '''
    min_hu, max_hu = ct_range
    # 1. 映射回 [0, 1]
    tensor = (tensor + 1.0) / 2.0
    # 2. 映射回 [min_hu, max_hu]
    tensor = tensor * (max_hu - min_hu) + min_hu
    return tensor


def main():
    decode_cpu = True

    # Latent 形状 (必须与训练一致)
    latent_shape = (1, 3, 48, 32, 72)
    ct_range = (-200, 2800)  # TODO 下次改为 (-1024, 3071)
    spacing = 0.5

    # 路径
    root = Path('.ds')
    ckpt_dir = root / 'checkpoints'

    # 模型路径
    ae_path = ckpt_dir / 'autoencoder_best.pt'
    ldm_path = ckpt_dir / 'ldm' / 'ldm_last.pt'

    output_dir = root / 'ldm_test'
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 2. 加载 Autoencoder (VAE) ---
    print('Loading Autoencoder...')
    autoencoder = AutoencoderKL(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        num_res_blocks=(2, 2, 2, 2),
        channels=(32, 64, 64, 64),
        attention_levels=(False, False, False, False),
        latent_channels=3,
        norm_num_groups=32,
    ).to(device)

    # 加载权重 & Scale Factor
    ae_ckpt = torch.load(ae_path, map_location=device)
    if 'state_dict' in ae_ckpt:
        autoencoder.load_state_dict(ae_ckpt['state_dict'])
    else:
        autoencoder.load_state_dict(ae_ckpt)

    scale_factor = ae_ckpt['scale_factor']
    print(f'Using Scale Factor: {scale_factor}')

    autoencoder.eval()

    # --- 3. 加载 Diffusion UNet ---
    print('Loading LDM UNet...')
    unet = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=3,
        out_channels=3,
        num_res_blocks=2,
        channels=(64, 128, 256, 512),
        attention_levels=(False, False, True, True),
        num_head_channels=32,
    ).to(device)

    unet_ckpt = torch.load(ldm_path, map_location=device)
    if 'ema_state_dict' in unet_ckpt:
        print('Loading UNet EMA weights...')
        unet.load_state_dict(unet_ckpt['ema_state_dict'])
    else:
        print('Loading UNet standard weights...')
        unet.load_state_dict(unet_ckpt['state_dict'])
    unet.eval()

    # --- 4. Scheduler & Inferer ---
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        schedule='scaled_linear_beta',
        beta_start=0.0015,
        beta_end=0.0195,
        clip_sample=False,
    )

    # 注意：这里我们不把 autoencoder 传给 inferer，
    # 这样 inferer.sample 只会返回 Latent，我们可以手动控制解码过程以节省显存。
    inferer = DiffusionInferer(scheduler) if decode_cpu else LatentDiffusionInferer(scheduler, scale_factor)

    # --- 5. 生成循环 ---
    with torch.no_grad():
        # A. 生成随机噪声
        noise = torch.randn(latent_shape).to(device)

        print('Sampling Latents...')
        if decode_cpu:
            latents = inferer.sample(
                input_noise=noise,
                diffusion_model=unet,
                scheduler=scheduler,
                verbose=True,
                save_intermediates=False,
            )
            latents = latents.cpu()
            autoencoder.to('cpu')
            latents = latents / scale_factor
        else:
            latents = inferer.sample(
                input_noise=noise,
                autoencoder_model=autoencoder,
                diffusion_model=unet,
                scheduler=scheduler,
                verbose=True,
                save_intermediates=False,
            )

        print(f'\n--- Generated Latent Stats ---')
        print(f'Mean: {latents.mean().item():.4f}')
        print(f'Std:  {latents.std().item():.4f}')
        print(f'Min:  {latents.min().item():.4f}')
        print(f'Max:  {latents.max().item():.4f}')

        print(f'Decoding on {latents.device}...')
        decoded_img = autoencoder.decode(latents)

        # 还原到 HU 范围
        restored_img = restore_ct(decoded_img, ct_range)
        image = restored_img.squeeze().cpu().numpy().astype(np.int16)
        print(f'Image Stats - Min: {image.min()}, Max: {image.max()}, Mean: {image.mean():.2f}, Std: {image.std():.2f}')
        image = np.clip(image, -1024, 3071)

        image = itk.image_from_array(image.transpose(2, 1, 0).copy())
        image.SetSpacing((spacing, spacing, spacing))
        itk.imwrite(image, output_dir / 'test.nii.gz', compression=True)


if __name__ == '__main__':
    _ = time.perf_counter()
    main()
    print(f'Took {time.perf_counter() - _:.2f} seconds.')
