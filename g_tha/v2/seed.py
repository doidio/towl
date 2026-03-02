import argparse
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from tqdm import tqdm

# 全局配置
spacing = 1.0
patch_size = (128,) * 3  # VAE 推理时的滑动窗口大小
bone_min = 150.0  # 骨阈值
metal_min = 2700.0  # 假体金属阈值
ct_min, ct_max = -1024.0, 3071.0  # CT最值
vae_downsample = 4  # VAE 的下采样倍率


def main():
    b = argparse.BooleanOptionalAction
    parser = argparse.ArgumentParser(description='Latent Diffusion Model 多种子对比推理脚本')

    # 必须参数
    parser.add_argument('--cond', type=str, required=True, help='术前条件图像路径 (.nii.gz)')
    parser.add_argument('--save', type=str, required=True, help='生成结果保存目录')

    # 模型路径参数
    parser.add_argument('--vae-pre', type=str, default=None, help='VAE模型路径')
    parser.add_argument('--vae-metal', type=str, default=None, help='VAE模型路径')
    parser.add_argument('--ldm', type=str, default=None, help='LDM模型路径')

    # 硬件与性能参数
    parser.add_argument('--amp', action=b, default=True, help='是否启用混合精度')
    parser.add_argument('--sw', type=int, default=4, help='滑动窗口推理时的并行 Batch Size')
    parser.add_argument('--tiled', action=b, default=True, help='是否使用分块推理')

    # 生成控制参数
    parser.add_argument('--n', '--seeds', type=int, default=5, help='对比的随机种子数量 (默认: 5)')
    parser.add_argument('--cfg', type=float, default=3.0, help='CFG 权重 (默认: 3.0)')
    parser.add_argument('--ts', type=int, default=50, help='DDIM 采样步数')

    args = parser.parse_args()

    # 参数后处理
    sw_batch_size = max(args.sw, 1)
    timesteps = min(max(args.ts, 1), 1000)
    cfg = float(max(args.cfg, 0.0))
    # 自动生成指定数量的种子，从 42 开始递增以保证一定的可复现性
    seeds = list(range(42, 42 + max(args.n, 1)))

    import torch
    torch.backends.cudnn.benchmark = True

    from torch import autocast
    from monai.networks.nets import AutoencoderKL, DiffusionModelUNet
    from monai.networks.schedulers import DDIMScheduler
    from monai.transforms import Compose, MapTransform, LoadImaged, SpatialPadd, CenterSpatialCrop
    from monai.inferers import sliding_window_inference

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # 加载 VAE 模型
    vae_dual = []
    for subtask in ('metal', 'pre'):
        print(f'Loading VAE {subtask}...')
        if getattr(args, f'vae_{subtask}') is None:
            vae_path = Path(__file__).parent / f'vae_{subtask}_best.pt'
        else:
            vae_path = Path(getattr(args, f'vae_{subtask}'))

        if not vae_path.exists():
            raise SystemError(f'Not found: {vae_path.resolve()}')

        vae = AutoencoderKL(
            spatial_dims=3, in_channels=1, out_channels=1,
            num_res_blocks=(2, 2, 2), channels=(32, 64, 128),
            attention_levels=(False, False, False),
            with_encoder_nonlocal_attn=False, with_decoder_nonlocal_attn=False,
            latent_channels=4, norm_num_groups=32, use_checkpoint=True,
        )

        _ = torch.load(vae_path, map_location=device)
        vae.load_state_dict(_['state_dict'])
        vae.to(device if args.tiled else 'cpu').eval().float()
        vae_dual += [vae, _['scale_factor'], _['global_mean']]

    vae_image, vae_image_scale, vae_image_mean, vae_cond, vae_cond_scale, vae_cond_mean = vae_dual

    # 加载 LDM 模型
    print('Loading LDM...')
    if args.ldm is None:
        ldm_path = Path(__file__).parent / 'ldm_last.pt'
    else:
        ldm_path = Path(args.ldm)

    ldm = DiffusionModelUNet(
        spatial_dims=3, in_channels=8, out_channels=4,
        num_res_blocks=(2, 2, 2), channels=(64, 128, 256),
        attention_levels=(False, False, True),
        norm_num_groups=32, with_conditioning=False, use_flash_attention=True,
    ).to(device)

    _ = torch.load(ldm_path, map_location=device)
    if 'ema_state' in _: ldm.load_state_dict(_['ema_state'])
    else: ldm.load_state_dict(_['state_dict'])
    ldm.eval().float()

    scheduler = DDIMScheduler(
        num_train_timesteps=1000, schedule='scaled_linear_beta',
        prediction_type='epsilon', clip_sample=False,
    )

    # 准备条件图像
    import itk, cv2
    from PIL import Image, ImageDraw, ImageFont
    cond_path = Path(args.cond)
    itk_img = itk.imread(cond_path.as_posix())
    original_ct = itk.array_from_image(itk_img).astype(np.float32)
    cropper = CenterSpatialCrop(roi_size=original_ct.shape[::-1])
    
    coronal_idx = original_ct.shape[1] // 2 + 5
    save_dir = Path(args.save) / f"{cond_path.stem}_seeds"
    save_dir.mkdir(parents=True, exist_ok=True)

    def hu_to_bgr(hu_slice):
        l, w = 300, 1500
        img = np.clip(hu_slice, l - w // 2, l + w // 2)
        img = (img - (l - w // 2)) / w * 255.0
        return cv2.cvtColor(np.flipud(img.astype(np.uint8)), cv2.COLOR_GRAY2BGR)

    def add_label(img_bgr, text):
        img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        # 尝试加载默认字体
        try: font = ImageFont.load_default()
        except: font = None
        draw.text((10, img_pil.size[1] - 25), text, fill=(255, 255, 255), font=font)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    class CTBoneNormalized(MapTransform):
        def __init__(self, keys):
            super().__init__(keys)
            self.src_pts = [bone_min, 650.0, 1150.0, 3000.0]
            self.dst_pts = [-1.0, 0.0, 0.5, 1.0]
        def __call__(self, data):
            d = dict(data)
            for key in self.key_iterator(d):
                img_t = torch.as_tensor(d[key])
                xp = torch.tensor(self.src_pts, device=img_t.device, dtype=img_t.dtype)
                fp = torch.tensor(self.dst_pts, device=img_t.device, dtype=img_t.dtype)
                x_clamped = torch.clamp(img_t, min=xp[0], max=xp[-1])
                ind = torch.clamp(torch.searchsorted(xp, x_clamped, right=True), 1, len(xp) - 1)
                d[key] = fp[ind-1] + (x_clamped - xp[ind-1]) * (fp[ind] - fp[ind-1]) / (xp[ind] - xp[ind-1])
            return d

    print('Encoding condition...')
    cond_raw = Compose([
        LoadImaged(keys=['image'], ensure_channel_first=True),
        SpatialPadd(keys=['image'], spatial_size=patch_size, constant_values=ct_min),
        CTBoneNormalized(keys=['image']),
    ])({'image': cond_path.as_posix()})['image'].unsqueeze(0).to(device)

    with torch.no_grad():
        if args.tiled:
            cond = sliding_window_inference(cond_raw, patch_size, sw_batch_size, lambda x: vae_cond.encode(x)[0], overlap=0.25)
        else:
            cond = vae_cond.encode(cond_raw.to('cpu'))[0].to(device)
    cond = (cond - vae_cond_mean) * vae_cond_scale

    # 循环生成
    all_sdf_slices = []
    # 颜色列表 (BGR): 红, 绿, 蓝, 黄, 紫, 青, 橙
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 165, 255)]

    for seed in seeds:
        print(f'Generating for seed: {seed}')
        generator = torch.Generator(device=device).manual_seed(seed)
        generated = torch.randn(cond.shape, device=device, generator=generator)
        scheduler.set_timesteps(num_inference_steps=timesteps, device=device)

        for t in tqdm(scheduler.timesteps, desc=f'Seed {seed}'):
            with torch.no_grad():
                t_in = t[None].to(device)
                if cfg > 1.0:
                    latent_input = torch.cat([generated] * 2)
                    cond_input = torch.cat([cond, torch.zeros_like(cond)])
                    model_input = torch.cat([latent_input, cond_input], dim=1)
                    with autocast(device.type) if args.amp else nullcontext():
                        noise_pred_batch = ldm(model_input, t_in.repeat(2))
                    n_cond, n_uncond = noise_pred_batch.chunk(2)
                    noise_pred = n_uncond + cfg * (n_cond - n_uncond)
                else:
                    model_input = torch.cat([generated, cond], dim=1)
                    with autocast(device.type) if args.amp else nullcontext():
                        noise_pred = ldm(model_input, t_in)
                generated, _ = scheduler.step(noise_pred, t, generated)

        # 解码
        print(f'Decoding seed {seed}...')
        z = generated / vae_image_scale + vae_image_mean
        with torch.no_grad():
            if args.tiled:
                decoded = sliding_window_inference(z, [p // vae_downsample for p in patch_size], sw_batch_size, lambda x: vae_image.decode(x), overlap=0.25)
            else:
                decoded = vae_image.decode(z.to('cpu')).to(device)
        
        sdf = cropper(decoded[0]).cpu().numpy()[0].transpose(2, 1, 0)
        sdf_slice = sdf[:, coronal_idx, :]
        # 必须翻转以匹配 hu_to_bgr 中的 np.flipud，否则假体上下颠倒
        all_sdf_slices.append(np.flipud(sdf_slice))

    # 可视化
    print('Creating summary image...')
    preop_bgr = hu_to_bgr(original_ct[:, coronal_idx, :])
    overlay_bgr = preop_bgr.copy()
    individual_results = []

    for i, sdf_slice in enumerate(all_sdf_slices):
        color = colors[i % len(colors)]
        mask = sdf_slice >= 0
        
        # 1. 准备单张 seed 的结果图 (带假体融合)
        seed_fused = preop_bgr.copy()
        # 简单融合：将假体区域设为对应的颜色
        seed_fused[mask] = color
        cv2.addWeighted(seed_fused, 0.4, preop_bgr, 0.6, 0, seed_fused)
        individual_results.append(add_label(seed_fused, f'Seed {seeds[i]}'))

        # 2. 累加到总叠加图
        colored_layer = overlay_bgr.copy()
        colored_layer[mask] = color
        cv2.addWeighted(colored_layer, 0.5, overlay_bgr, 0.5, 0, overlay_bgr)
        # 轮廓线
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_bgr, contours, -1, color, 1)

    # 拼接顺序: Preoperative | Overlay | Seed 1 | Seed 2 | ...
    summary_list = [
        add_label(preop_bgr, 'Preoperative'),
        add_label(overlay_bgr, f'Seeds Overlay ({len(seeds)})')
    ] + individual_results
    
    summary = np.hstack(summary_list)
    save_path = save_dir / f"{cond_path.stem}_seed_compare.png"
    cv2.imwrite(save_path.as_posix(), summary)
    print(f'Comparison saved at: {save_path.resolve()}')

if __name__ == '__main__':
    main()
