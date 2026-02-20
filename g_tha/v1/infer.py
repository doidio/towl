import argparse
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from tqdm import tqdm

patch_size = (128,) * 3
bone_range = [150.0, 650.0]
vae_downsample = 4


def main():
    b = argparse.BooleanOptionalAction
    parser = argparse.ArgumentParser()
    parser.add_argument('--cond', type=str, required=True, help='术前条件 (.nii.gz)')
    parser.add_argument('--save', type=str, required=True, help='保存目录')
    parser.add_argument('--vae', type=str, default=None, help='VAE模型')
    parser.add_argument('--ldm', type=str, default=None, help='LDM模型')
    parser.add_argument('--amp', action=b, default=True, help='混合精度')
    parser.add_argument('--sw', type=int, default=4, help='滑动窗口并行数量')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--cfg', type=int, default=1, help='CFG条件生成(1,3,5,7,9)无条件生成(0)')
    parser.add_argument('--ts', type=int, default=50, help='DDIM步数(50,100,200,500,1000)')
    parser.add_argument('--tiled', action=b, default=True, help='分块GPU解码/全局CPU解码')
    args = parser.parse_args()

    sw_batch_size = max(args.sw, 1)
    timesteps = min(max(args.ts, 1), 1000)
    guide_weight = max(args.cfg, 0.0)

    import torch
    from torch import autocast
    from monai.networks.nets import AutoencoderKL, DiffusionModelUNet
    from monai.networks.schedulers import DDIMScheduler
    from monai.transforms import Compose, MapTransform, LoadImaged, SaveImage
    from monai.inferers import sliding_window_inference

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device:\t {device}')

    if args.vae is None:
        vae_path = Path(__file__).parent / 'vae_best.pt'
    else:
        vae_path = Path(args.vae)

    if not vae_path.exists():
        raise SystemError(f'VAE not found:\t {vae_path.resolve()}')
    else:
        print(f'VAE Loading:\t {vae_path.resolve()}')

    vae = AutoencoderKL(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        num_res_blocks=(2, 2, 2),
        channels=(32, 64, 128),
        attention_levels=(False, False, False),
        with_encoder_nonlocal_attn=False,
        with_decoder_nonlocal_attn=False,
        latent_channels=4,
        norm_num_groups=32,
        use_checkpoint=True,
    )

    if args.tiled:
        vae.to(device)
    else:
        vae.to('cpu')

    _ = torch.load(vae_path, map_location=device)
    vae.load_state_dict(_['state_dict'])
    vae.eval().float()
    scale_factor = _['scale_factor']
    print(f'VAE Scale:\t {scale_factor}')

    if args.ldm is None:
        ldm_path = Path(__file__).parent / 'ldm_best.pt'
    else:
        ldm_path = Path(args.ldm)

    if not ldm_path.exists():
        raise SystemError(f'LDM not found:\t {ldm_path.resolve()}')
    else:
        print(f'LDM loading:\t {ldm_path.resolve()}')

    ldm = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=8,
        out_channels=4,
        num_res_blocks=(2, 2, 2),
        channels=(64, 128, 256),
        attention_levels=(False, False, True),
        norm_num_groups=32,
        with_conditioning=False,
        use_flash_attention=True,
    ).to(device)

    _ = torch.load(ldm_path, map_location=device)
    ldm.load_state_dict(_['state_dict'])
    ldm.eval().float()

    class CTBoneNormalized(MapTransform):
        """基于线性分段函数的 CT 值映射变换 (Linear Piecewise Mapping)"""

        def __init__(self, keys, reverse=False, allow_missing_keys=False):
            super().__init__(keys, allow_missing_keys)

            self.src_pts = [*bone_range, 1500.0, 3000.0]
            self.dst_pts = [-1.0, 0.0, 0.5, 1.0]
            if reverse:
                self.src_pts, self.dst_pts = self.dst_pts, self.src_pts

        def __call__(self, data):
            d = dict(data)
            for key in self.key_iterator(d):
                img = d[key]

                if not isinstance(img, torch.Tensor):
                    is_numpy = True
                    img_t = torch.as_tensor(img)
                else:
                    is_numpy = False
                    img_t = img

                xp = torch.tensor(self.src_pts, device=img_t.device, dtype=img_t.dtype)
                fp = torch.tensor(self.dst_pts, device=img_t.device, dtype=img_t.dtype)

                x_clamped = torch.clamp(img_t, min=xp[0], max=xp[-1])

                indices = torch.searchsorted(xp, x_clamped, right=True)
                indices = torch.clamp(indices, 1, len(xp) - 1)

                idx0 = indices - 1
                idx1 = indices

                x0 = xp[idx0]
                x1 = xp[idx1]
                y0 = fp[idx0]
                y1 = fp[idx1]

                res = y0 + (x_clamped - x0) * (y1 - y0) / (x1 - x0)

                if is_numpy:
                    d[key] = res.cpu().numpy().astype(np.float32)
                else:
                    d[key] = res.to(dtype=img_t.dtype)  # 保持原有精度

            return d

    cond_transforms = Compose([
        LoadImaged(keys=['image'], ensure_channel_first=True),
        CTBoneNormalized(keys=['image']),
    ])

    cond_path = Path(args.cond)

    if not cond_path.exists():
        raise SystemError(f'Condition not found:\t {cond_path.resolve()}')
    else:
        print(f'Cond loading:\t {cond_path.resolve()}')

    cond = cond_transforms({'image': cond_path.as_posix()})['image'].unsqueeze(0).to(device)

    print(f'Cond encoding:\t {cond.shape}', 'tiled' if args.tiled else 'no-tiled')

    if args.tiled:
        def encode_predictor(z):
            with autocast(device.type) if args.amp else nullcontext():
                return vae.encode(z)[0]

        with torch.no_grad():
            with autocast(device.type) if args.amp else nullcontext():
                cond = sliding_window_inference(
                    inputs=cond,
                    roi_size=patch_size,
                    sw_batch_size=sw_batch_size,
                    predictor=encode_predictor,
                    overlap=0.25,
                    mode='gaussian',
                    device=device,
                    progress=False,
                )
    else:
        cond = cond.detach().cpu().float()
        with torch.no_grad():
            cond = vae.encode(cond)[0]
        cond = cond.to(device)

    cond = cond * scale_factor
    print(f'Cond encoded:\t {cond.shape}')

    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        schedule='scaled_linear_beta',
        prediction_type='epsilon',
        clip_sample=False,
    )
    scheduler.set_timesteps(num_inference_steps=timesteps, device=device)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    generated = torch.randn(cond.shape, device=device, generator=generator)

    for t in (pbar := tqdm(scheduler.timesteps, desc='LDM generating')):
        pbar.set_postfix({'DDIM': t.item()})

        latent_input = torch.cat([generated] * 2)

        uncond = torch.zeros_like(cond)
        cond_input = torch.cat([cond, uncond])

        model_input = torch.cat([latent_input, cond_input], dim=1)

        with torch.no_grad():
            t_input = t[None].to(device).repeat(2)
            with autocast(device.type) if args.amp else nullcontext():
                noise_pred_batch = ldm(model_input, t_input)

        noise_cond, noise_uncond = noise_pred_batch.chunk(2)
        noise_pred = noise_uncond + guide_weight * (noise_cond - noise_uncond)

        with torch.no_grad():
            generated, _ = scheduler.step(noise_pred, t, generated)

    print(f'Generated decoding:\t {generated.shape}')

    if args.tiled:
        def decode_predictor(latent_patch):
            with autocast(device.type) if args.amp else nullcontext():
                return vae.decode(latent_patch)

        generated = sliding_window_inference(
            inputs=generated,
            roi_size=[_ // vae_downsample for _ in patch_size],
            sw_batch_size=sw_batch_size,
            predictor=decode_predictor,
            overlap=0.25,
            mode='gaussian',
            device=device,
        )
    else:
        generated = (generated / scale_factor).detach().cpu().float()
        with torch.no_grad():
            generated = vae.decode(generated)
        generated = generated.to(device)

    print(f'Generated decoded:\t {generated.shape}')

    denorm = CTBoneNormalized(keys=['image'], reverse=True)
    generated = denorm({'image': generated[0]})

    save = Path(args.save)
    print(f'Generated saving:\t {save.resolve()}')
    saver = SaveImage(
        output_dir=save,
        output_postfix=f'seed_{args.seed}_cfg_{args.cfg}_ts_{args.ts}' + ('_tiled' if args.tiled else '_no-tiled'),
        output_ext='.nii.gz',
        separate_folder=False,
        print_log=False,
    )

    generated = generated['image'].detach().cpu().float()
    saver(generated, meta_data={'filename_or_obj': cond_path.name})


if __name__ == '__main__':
    main()
