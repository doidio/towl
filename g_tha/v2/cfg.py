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
vae_downsample = 4  # VAE 的下采样倍率 (例如: 128 -> 32)


def main():
    b = argparse.BooleanOptionalAction
    parser = argparse.ArgumentParser(description='Latent Diffusion Model CFG 对比推理脚本')

    # 必须参数
    parser.add_argument('--cond', type=str, required=True, help='术前条件图像路径 (.nii.gz)')
    parser.add_argument('--save', type=str, required=True, help='生成结果保存目录')

    # 模型路径参数 (默认使用脚本同级目录下的 .pt 文件)
    parser.add_argument('--vae-pre', type=str, default=None,
                        help='VAE模型路径 (默认: train/checkpoints/vae_pre_best.pt)')
    parser.add_argument('--vae-metal', type=str, default=None,
                        help='VAE模型路径 (默认: train/checkpoints/vae_metal_best.pt)')
    parser.add_argument('--ldm', type=str, default=None, help='LDM模型路径 (默认: train/checkpoints/ldm_last.pt)')

    # 硬件与性能参数
    parser.add_argument('--amp', action=b, default=True, help='是否启用混合精度 (默认: True)')
    parser.add_argument('--sw', type=int, default=4, help='滑动窗口推理时的并行 Batch Size')
    parser.add_argument('--tiled', action=b, default=True, help='是否使用分块推理 (True: 节省显存, False: 全图推理)')

    # 生成控制参数
    parser.add_argument('--seed', type=int, default=None, help='随机种子，固定种子可复现结果 (默认: None/随机)')
    parser.add_argument('--cfgs', type=float, nargs='+', default=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 3.0, 5.0, 10.0],
                        help='CFG 权重列表用于对比横向排列 (默认: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0)')
    parser.add_argument('--ts', type=int, default=50, help='DDIM 采样步数 (通常 50-200)')
    parser.add_argument('--save-all', action=b, default=False, help='是否保存所有CFG对应的 NIfTI 和 STL 文件 (默认: False)')

    args = parser.parse_args()

    # 参数后处理
    sw_batch_size = max(args.sw, 1)
    timesteps = min(max(args.ts, 1), 1000)
    cfgs = [float(max(c, 0.0)) for c in args.cfgs]

    # 延迟导入以加快命令行响应速度
    import torch
    torch.backends.cudnn.benchmark = True

    from torch import autocast
    from monai.networks.nets import AutoencoderKL, DiffusionModelUNet
    from monai.networks.schedulers import DDIMScheduler
    from monai.transforms import Compose, MapTransform, LoadImaged, SpatialPadd, CenterSpatialCrop
    from monai.inferers import sliding_window_inference

    # 设备选择 (优先使用 CUDA)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device:	 {device}')

    # 加载 VAE 模型 (AutoencoderKL)
    vae_dual = []
    for subtask in ('metal', 'pre'):
        print('VAE', subtask)

        if getattr(args, f'vae_{subtask}') is None:
            vae_path = Path(f'train/checkpoints/vae_{subtask}_best.pt')
        else:
            vae_path = Path(getattr(args, f'vae_{subtask}'))

        if not vae_path.exists():
            raise SystemError(f'Not found:	 {vae_path.resolve()}')
        else:
            print(f'Loading: {vae_path.resolve()}')

        # 初始化 VAE 网络结构
        vae = AutoencoderKL(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            num_res_blocks=(2, 2, 2),
            channels=(32, 64, 128),  # 逐层加宽，捕捉高频骨纹理
            attention_levels=(False, False, False),  # 自编码器必须采用纯卷积，Patch Training 与 Attention 之间天然矛盾
            with_encoder_nonlocal_attn=False,  # 关闭非局部注意力
            with_decoder_nonlocal_attn=False,  # 关闭非局部注意力
            latent_channels=4,  # 保持 4 通道，足够编码密度信息
            norm_num_groups=32,  # 归一化层，也会削弱 Patch Training 效果
            use_checkpoint=True,
        )

        # 根据是否分块决定 VAE 放置位置 (分块通常在 GPU，全图可能需要 CPU 内存)
        if args.tiled:
            vae.to(device)
        else:
            vae.to('cpu')

        # 加载权重
        _ = torch.load(vae_path, map_location=device)
        vae.load_state_dict(_['state_dict'])
        vae.eval().float()

        # 获取 latent space 的缩放因子 (用于保持 latent 分布的标准差接近 1)
        scale_factor, global_mean = _['scale_factor'], _['global_mean']
        print(f'	Epoch:	', _['epoch'])
        print(f'	Params:	 {sum(p.numel() for p in vae.parameters()) / 1e9:.2f} B')
        print(f'	Scale:	', scale_factor)

        vae_dual += [vae, scale_factor, global_mean]

    vae_image, vae_image_scale, vae_image_mean, vae_cond, vae_cond_scale, vae_cond_mean = vae_dual

    # 加载 LDM 模型 (DiffusionModelUNet)
    print('LDM')

    if args.ldm is None:
        ldm_path = Path(f'train/checkpoints/ldm_last.pt')
    else:
        ldm_path = Path(args.ldm)

    if not ldm_path.exists():
        raise SystemError(f'Not found:	 {ldm_path.resolve()}')
    else:
        print(f'Loading: {ldm_path.resolve()}')

    ldm = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=8,
        out_channels=4,
        num_res_blocks=(2, 2, 2),
        channels=(64, 128, 256),
        attention_levels=(False, False, True),  # 启用自注意力学习解剖方位关系
        norm_num_groups=32,
        with_conditioning=False,
        use_flash_attention=True,
    ).to(device)

    _ = torch.load(ldm_path, map_location=device)
    print(f'	Epoch:	', _['epoch'])
    print(f'	Params:	 {sum(p.numel() for p in ldm.parameters()) / 1e9:.2f} B')

    if 'ema_state' in _:
        print('	State:	 EMA')
        ldm.load_state_dict(_['ema_state'])
    else:
        print('	State:	 Raw (No EMA found)')
        ldm.load_state_dict(_['state_dict'])

    ldm.eval().float()

    print('	CFGs List:	', cfgs)
    print('	DDIM steps:	', timesteps)

    # 初始化 LDM 采样器
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        schedule='scaled_linear_beta',
        prediction_type='epsilon',
        clip_sample=False,
    )

    # 准备条件图像路径
    cond_path = Path(args.cond)
    if not cond_path.exists():
        raise SystemError(f'Condition not found:	 {cond_path.resolve()}')
    else:
        print(f'Cond loading:	 {cond_path.resolve()}')

    # 准备后处理工具
    import itk
    import cv2
    from PIL import Image, ImageDraw, ImageFont

    itk_img = itk.imread(cond_path.as_posix())
    original_ct = itk.array_from_image(itk_img).astype(np.float32)  # [D, H, W]

    cropper = CenterSpatialCrop(roi_size=original_ct.shape[::-1])
    save_dir = Path(args.save) / '_'.join([
        cond_path.with_suffix('').with_suffix('').name,
        'seed', str(args.seed) if args.seed else 'random',
        'cfgs_compare',
        'ts', str(args.ts),
        'tiled' if args.tiled else 'no-tiled',
    ])
    save_dir.mkdir(parents=True, exist_ok=True)

    # 获取冠状面中心切片索引
    coronal_idx = original_ct.shape[1] // 2 + 5

    def add_label(img_bgr, text):
        """添加白色无背景标签"""
        img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.load_default()
        draw.text((10, img_pil.size[1] - 20), text, font=font, fill=(255, 255, 255))
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def hu_to_bgr(hu_slice):
        """骨窗映射到 BGR"""
        l, w = 300, 1500
        img = np.clip(hu_slice, l - w // 2, l + w // 2)
        img = (img - (l - w // 2)) / w * 255.0
        img = img.astype(np.uint8)
        img = np.flipud(img)  # 保持正位
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    def decode_and_save(latent_tensor, cfg_val):
        """解码 Latent 并保存融合结果，返回用于摘要的 BGR 切片"""
        # 1. 反向缩放 Latent
        z = latent_tensor / vae_image_scale + vae_image_mean

        # 2. VAE 解码
        if args.tiled:
            def predictor(patch):
                with autocast(device.type) if args.amp else nullcontext():
                    return vae_image.decode(patch)

            decoded = sliding_window_inference(
                inputs=z,
                roi_size=[_ // vae_downsample for _ in patch_size],
                sw_batch_size=sw_batch_size,
                predictor=predictor,
                overlap=0.25,
                mode='gaussian',
                device=device,
                progress=False,
            )
        else:
            z_cpu = z.detach().cpu().float()
            with torch.no_grad():
                decoded = vae_image.decode(z_cpu)
            decoded = decoded

        # 3. 裁剪与后处理
        decoded = decoded[0].detach().cpu().float()
        decoded = cropper(decoded)
        sdf_numpy = decoded[0].cpu().numpy().transpose(2, 1, 0)  # [Z, Y, X]

        # 4. 融合
        fused = original_ct.copy()
        delta = 0.1
        sdf_viz = sdf_numpy

        fused[sdf_viz >= delta] = ct_max
        mask_inner = (sdf_viz >= 0.0) & (sdf_viz < delta)
        fused[mask_inner] = metal_min + (sdf_viz[mask_inner] / delta) * (ct_max - metal_min)
        mask_outer = (sdf_viz >= -delta) & (sdf_viz < 0.0)
        t_alpha = (sdf_viz[mask_outer] + delta) / delta
        fused[mask_outer] = original_ct[mask_outer] * (1.0 - t_alpha) + metal_min * t_alpha

        # 5. 保存 NIfTI (若开启 save-all)
        if args.save_all:
            out_name = cond_path.name.replace('.nii.gz', f'_cfg_{cfg_val}_fused.nii.gz')
            out_path = save_dir / out_name

            itk_out = itk.image_from_array(fused)
            itk_out.CopyInformation(itk_img)
            itk.imwrite(itk_out, out_path.as_posix())

        # 6. 提取 BGR 切片用于摘要
        fused_slice = fused[:, coronal_idx, :]
        bgr = hu_to_bgr(fused_slice)
        return sdf_numpy, add_label(bgr, f'CFG {cfg_val}')

    # 数据预处理类定义
    class CTBoneNormalized(MapTransform):
        def __init__(self, keys, reverse=False, allow_missing_keys=False):
            super().__init__(keys, allow_missing_keys)

            self.src_pts = [bone_min, 650.0, 1150.0, 3000.0]
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

                ind = torch.searchsorted(xp, x_clamped, right=True)
                ind = torch.clamp(ind, 1, len(xp) - 1)

                idx0 = ind - 1
                idx1 = ind

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

    # 加载并归一化条件图像 [B, C, H, W, D]
    cond_transforms = Compose([
        LoadImaged(keys=['image'], ensure_channel_first=True),
        SpatialPadd(keys=['image'], spatial_size=patch_size, constant_values=ct_min),
        CTBoneNormalized(keys=['image']),
    ])
    cond_raw = cond_transforms({'image': cond_path.as_posix()})['image']
    cond = cond_raw.unsqueeze(0).to(device)

    print(f'Cond encoding:	 {cond.shape}')

    # VAE 编码 (Encoding)
    if args.tiled:
        # 分块编码以节省显存
        def encode_predictor(z):
            with autocast(device.type) if args.amp else nullcontext():
                return vae_cond.encode(z)[0]

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
                    progress=True,
                )
    else:
        # 全图直接编码
        cond = cond.detach().cpu().float()
        with torch.no_grad():
            cond = vae_cond.encode(cond)[0]
        cond = cond.to(device)

    # 缩放 latent 分布
    cond = (cond - vae_cond_mean) * vae_cond_scale
    print(f'Cond encoded:	 {cond.shape}')

    # 初始化随机噪声
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)
    else:
        generator = None

    # 初始化基准随机噪声，以保证不同 CFG 从相同的初始噪声起点进行去噪
    base_noise = torch.randn(cond.shape, device=device, generator=generator)

    # 准备摘要长图列表
    summary_list = [add_label(hu_to_bgr(original_ct[:, coronal_idx, :]), 'Preop')]

    # 针对不同 CFG 参数循环生成
    for cfg in cfgs:
        print(f'\n================ Generating for CFG: {cfg} ================')
        
        # 复制相同的初始噪声
        generated = base_noise.clone()
        scheduler.set_timesteps(num_inference_steps=timesteps, device=device)

        # 逐步去噪循环
        for i, t in enumerate(bar := tqdm(scheduler.timesteps, desc=f'LDM (CFG={cfg})')):
            bar.set_postfix({'DDIM': t.item()})

            # 预测噪声
            with torch.no_grad():
                t_input = t[None].to(device)

                if cfg > 1.0:
                    # 并行计算条件与无条件预测
                    latent_input = torch.cat([generated] * 2)
                    cond_input = torch.cat([cond, torch.zeros_like(cond)])
                    model_input = torch.cat([latent_input, cond_input], dim=1)

                    with autocast(device.type) if args.amp else nullcontext():
                        noise_pred_batch = ldm(model_input, t_input.repeat(2))

                    # CFG 引导公式: noise = uncond + w * (cond - uncond)
                    noise_cond, noise_uncond = noise_pred_batch.chunk(2)
                    noise_pred = noise_uncond + cfg * (noise_cond - noise_uncond)

                elif cfg == 1.0:
                    # 仅条件路
                    model_input = torch.cat([generated, cond], dim=1)
                    with autocast(device.type) if args.amp else nullcontext():
                        noise_pred = ldm(model_input, t_input)

                else:
                    # 仅无条件路
                    model_input = torch.cat([generated, torch.zeros_like(cond)], dim=1)
                    with autocast(device.type) if args.amp else nullcontext():
                        noise_pred = ldm(model_input, t_input)

            # DDIM 更新步
            with torch.no_grad():
                generated, pred_0 = scheduler.step(noise_pred, t, generated)

        # 最终解码并融合
        print(f'Decoding CFG={cfg} ...')
        sdf_numpy, bgr_final = decode_and_save(generated, cfg)
        summary_list.append(bgr_final)

        # 保存 3D 文件 (如果启用)
        if args.save_all:
            # 提取假体等值面网格体 (STL)
            stl_name = cond_path.name.replace('.nii.gz', f'_cfg_{cfg}_metal.stl')
            stl_path = save_dir / stl_name
            print(f'Isosurface saving:	', stl_path.resolve())

            from diso import DiffDMC
            import trimesh

            # 计算 STL 需要的 SDF 张量 [X, Y, Z]
            final_sdf_x_aligned = torch.from_numpy(sdf_numpy.transpose(2, 1, 0)).to(device)

            vertices, indices = DiffDMC(dtype=torch.float32)(-final_sdf_x_aligned, None, isovalue=0.0)
            vertices, indices = vertices.cpu().numpy(), indices.cpu().numpy()
            vertices = vertices * spacing * (np.array(final_sdf_x_aligned.shape) - 1)

            mesh = trimesh.Trimesh(vertices, indices)
            mesh.export(stl_path.as_posix())

            # 保存假体 SDF 结果 (Metal SDF)
            save_metal = save_dir / cond_path.name.replace('.nii.gz', f'_cfg_{cfg}_metal.nii.gz')
            print('Metal SDF saving:	', save_metal.resolve())

            itk_metal = itk.image_from_array(sdf_numpy)
            itk_metal.CopyInformation(itk_img)
            itk.imwrite(itk_metal, save_metal.as_posix())


    # 保存所有 CFG 的横向对比长图
    summary_img = np.hstack(summary_list)
    summary_path = save_dir / cond_path.name.replace('.nii.gz', f'_cfgs_comparison.png')
    cv2.imwrite(summary_path.as_posix(), summary_img)
    print(f'\n======================================================')
    print(f'CFG comparison summary successfully saved at:\n{summary_path.resolve()}')
    print(f'======================================================')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Keyboard interrupted terminating...')
