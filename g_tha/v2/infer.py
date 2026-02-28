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
    parser = argparse.ArgumentParser(description="Latent Diffusion Model 推理脚本")

    # 必须参数
    parser.add_argument('--cond', type=str, required=True, help='术前条件图像路径 (.nii.gz)')
    parser.add_argument('--save', type=str, required=True, help='生成结果保存目录')

    # 模型路径参数 (默认使用脚本同级目录下的 .pt 文件)
    parser.add_argument('--vae-pre', type=str, default=None, help='VAE模型路径 (默认: ./vae_best.pt)')
    parser.add_argument('--vae-metal', type=str, default=None, help='VAE模型路径 (默认: ./vae_best.pt)')
    parser.add_argument('--ldm', type=str, default=None, help='LDM模型路径 (默认: ./ldm_best.pt)')

    # 硬件与性能参数
    parser.add_argument('--amp', action=b, default=True, help='是否启用混合精度 (默认: True)')
    parser.add_argument('--sw', type=int, default=4, help='滑动窗口推理时的并行 Batch Size')
    parser.add_argument('--tiled', action=b, default=True, help='是否使用分块推理 (True: 节省显存, False: 全图推理)')

    # 生成控制参数
    parser.add_argument('--seed', type=int, default=None, help='随机种子，固定种子可复现结果 (默认: None/随机)')
    parser.add_argument('--cfg', type=int, default=1, help='Classifier-Free Guidance 权重 (0: 无条件, >1: 条件增强)')
    parser.add_argument('--ts', type=int, default=50, help='DDIM 采样步数 (通常 50-200)')

    args = parser.parse_args()

    # 参数后处理
    sw_batch_size = max(args.sw, 1)
    timesteps = min(max(args.ts, 1), 1000)
    cfg = float(max(args.cfg, 0.0))

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
    print(f'Device:\t {device}')

    # 加载 VAE 模型 (AutoencoderKL)
    vae_dual = []
    for subtask in ('metal', 'pre'):
        print('VAE', subtask)

        if getattr(args, f'vae_{subtask}') is None:
            vae_path = Path(f'train/checkpoints/vae_{subtask}_best.pt')
        else:
            vae_path = Path(getattr(args, f'vae_{subtask}'))

        if not vae_path.exists():
            raise SystemError(f'Not found:\t {vae_path.resolve()}')
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
        print(f'\tEpoch:\t', _['epoch'])
        print(f'\tParams:\t {sum(p.numel() for p in vae.parameters()) / 1e9:.2f} B')
        print(f'\tL1:  \t', _['val_l1'])
        print(f'\tPSNR:\t', _['val_psnr'])
        print(f'\tSSIM:\t', _['val_ssim'])
        print(f'\tScale:\t', _['scale_factor'])

        vae_dual += [vae, scale_factor, global_mean]

    vae_image, vae_image_scale, vae_image_mean, vae_cond, vae_cond_scale, vae_cond_mean = vae_dual

    # 加载 LDM 模型 (DiffusionModelUNet)
    print('LDM')

    if args.ldm is None:
        ldm_path = Path(f'train/checkpoints/ldm_last.pt')
    else:
        ldm_path = Path(args.ldm)

    if not ldm_path.exists():
        raise SystemError(f'Not found:\t {ldm_path.resolve()}')
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
    print(f'\tEpoch:\t', _['epoch'])
    print(f'\tParams:\t {sum(p.numel() for p in ldm.parameters()) / 1e9:.2f} B')

    if 'ema_state' in _:
        print('\tState:\t EMA')
        ldm.load_state_dict(_['ema_state'])
    else:
        print('\tState:\t Raw (No EMA found)')
        ldm.load_state_dict(_['state_dict'])

    ldm.eval().float()

    print('\tCFG:\t', cfg, 'Uncond-only' if cfg == 0 else 'Cond-only' if cfg == 1 else 'Guided')
    print('\tDDIM:\t', timesteps)

    # 初始化 LDM 采样器
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        schedule='scaled_linear_beta',
        prediction_type='epsilon',
        clip_sample=False,
    )
    scheduler.set_timesteps(num_inference_steps=timesteps, device=device)

    # 数据预处理类定义
    class CTBoneNormalized(MapTransform):
        """
        基于线性分段函数的 CT 值映射变换 (Linear Piecewise Mapping)

        1. 剔除无效背景: 将低于 bone_min (如 150 HU) 的软组织、脂肪和空气全部截断映射为 -1.0。
           这清除了大部分无用的背景噪声，强制 VAE 将所有网络容量 (Capacity) 集中在骨骼特征上，
           并为后续 LDM 留下大面积“零方差”区域，极大地降低了生成难度。
        2. 放大松质骨纹理 ([-1.0, 0.0]): 将 [150, 650] 的狭窄区间拉伸到了网络输入空间的一半。
           松质骨(骨小梁)是临床最关注且网络最难生成的精细纹理，这样分配算力能让特征学习得最锐利。
        3. 压缩皮质骨阶跃 ([0.0, 1.0]): 致密皮质骨的 CT 值往往在 1150 甚至局部突破 2000。
           如果将 1150 强行截断到 1.0，会导致严重的振铃效应 (Gibbs Artifacts)，MSE 惩罚极高。
           将 1150~3000 非线性压缩到 [0.5, 1.0] 的窄区间内，既保留了硬骨相对密度的层次感，
           又在数值上极大地“宽容”了高密度区域的 L1/MSE 重建误差，防止其在训练中反向吞噬松质骨的纹理细节。
        """

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

    # 准备条件图像 (Condition)
    cond_path = Path(args.cond)

    if not cond_path.exists():
        raise SystemError(f'Condition not found:\t {cond_path.resolve()}')
    else:
        print(f'Cond loading:\t {cond_path.resolve()}')

    # 加载并归一化条件图像 [B, C, H, W, D]
    cond_transforms = Compose([
        LoadImaged(keys=['image'], ensure_channel_first=True),
        SpatialPadd(keys=['image'], spatial_size=patch_size, constant_values=ct_min),
        CTBoneNormalized(keys=['image']),
    ])
    cond_raw = cond_transforms({'image': cond_path.as_posix()})['image']
    cond = cond_raw.unsqueeze(0).to(device)

    print(f'Cond encoding:\t {cond.shape}')

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

    # 缩放 latent 分布 (减去均值，乘以缩放因子)
    cond = (cond - vae_cond_mean) * vae_cond_scale
    print(f'Cond encoded:\t {cond.shape}')

    # 初始化随机噪声
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)
    else:
        generator = None

    generated = torch.randn(cond.shape, device=device, generator=generator)

    # 逐步去噪循环
    for t in (bar := tqdm(scheduler.timesteps, desc='LDM generating')):
        bar.set_postfix({'DDIM': t.item()})

        # 预测噪声
        if cfg > 1.0:
            latent_input = torch.cat([generated] * 2)

            # 构造条件部分: [Cond, Uncond] (Uncond 用全 0 表示)
            uncond = torch.zeros_like(cond)
            cond_input = torch.cat([cond, uncond])

            # 拼接 latent 和 condition: channel 维度 cat -> 8 channels
            model_input = torch.cat([latent_input, cond_input], dim=1)

            # 预测噪声
            with torch.no_grad():
                t_input = t[None].to(device).repeat(2)
                with autocast(device.type) if args.amp else nullcontext():
                    noise_pred_batch = ldm(model_input, t_input)

            # CFG 引导公式: noise = uncond + w * (cond - uncond)
            noise_cond, noise_uncond = noise_pred_batch.chunk(2)
            noise_pred = noise_uncond + cfg * (noise_cond - noise_uncond)
        elif cfg == 1.0:
            model_input = torch.cat([generated, cond], dim=1)
            with torch.no_grad():
                t_input = t[None].to(device)
                with autocast(device.type) if args.amp else nullcontext():
                    noise_pred = ldm(model_input, t_input)
            noise_pred = noise_pred
        else:
            uncond = torch.zeros_like(cond)
            model_input = torch.cat([generated, uncond], dim=1)
            with torch.no_grad():
                t_input = t[None].to(device)
                with autocast(device.type) if args.amp else nullcontext():
                    noise_pred = ldm(model_input, t_input)
            noise_pred = noise_pred

        # DDIM 更新步
        with torch.no_grad():
            generated, _ = scheduler.step(noise_pred, t, generated)

    # 反向缩放 latent (除以缩放因子，加上均值)
    generated = generated / vae_image_scale + vae_image_mean

    print(f'Generated decoding:\t {generated.shape}')

    # VAE 解码 (Decoding)
    if args.tiled:
        # 分块解码
        def decode_predictor(latent_patch):
            with autocast(device.type) if args.amp else nullcontext():
                return vae_image.decode(latent_patch)

        generated = sliding_window_inference(
            inputs=generated,
            roi_size=[_ // vae_downsample for _ in patch_size],
            sw_batch_size=sw_batch_size,
            predictor=decode_predictor,
            overlap=0.25,
            mode='gaussian',
            device=device,
            progress=True,
        )
    else:
        # 全图解码
        generated = generated.detach().cpu().float()
        with torch.no_grad():
            generated = vae_image.decode(generated)
        generated = generated.to(device)

    print(f'Generated decoded:\t {generated.shape}')

    generated = generated[0].detach().cpu().float()

    # 将 Pad 过的 SDF 裁剪回原始图像的尺寸
    import itk
    itk_img = itk.imread(cond_path.as_posix())
    pre = itk.array_from_image(itk_img).astype(np.float32)  # [D, H, W]

    # 使用 CenterSpatialCrop 进行裁剪，因为 SpatialPadd 默认是居中 padding 的
    cropper = CenterSpatialCrop(roi_size=pre.shape[::-1])  # TODO 数据集转置过但真实数据没有
    generated = cropper(generated)

    sdf = generated[0]  # [D, H, W]

    # 提取假体等值面网格体
    postfix = f'seed_{args.seed}_cfg_{args.cfg}_ts_{args.ts}' + ('_tiled' if args.tiled else '_no-tiled')
    save = Path(args.save)

    stl_name = cond_path.name.replace('.nii.gz', f'_{postfix}_metal.stl')
    stl_path = save / stl_name
    print(f'Isosurface saving:\t', stl_path.resolve())

    from diso import DiffDMC
    import trimesh

    vertices, indices = DiffDMC(dtype=torch.float32)(-sdf.to(device), None, isovalue=0.0)
    vertices, indices = vertices.cpu().numpy(), indices.cpu().numpy()
    vertices = vertices * spacing * (np.array(sdf.shape) - 1)

    mesh = trimesh.Trimesh(vertices, indices)
    mesh.export(stl_path.as_posix())

    # 保存假体SDF结果
    save_metal = save / cond_path.name.replace('.nii.gz', f'_{postfix}_metal.nii.gz')
    print('Metal SDF saving:\t', save_metal.resolve())

    sdf = sdf.cpu().numpy().transpose(2, 1, 0)
    fused_itk = itk.image_from_array(sdf)
    fused_itk.CopyInformation(itk_img)

    itk.imwrite(fused_itk, save_metal.as_posix())

    # 将假体 SDF 合成回术前原始图像
    fused_name = cond_path.name.replace('.nii.gz', f'_{postfix}_fused.nii.gz')
    fused_path = save / fused_name
    print('Fused saving:\t', fused_path.resolve())

    # 将 SDF 从 [X, Y, Z] 转置回 [Z, Y, X]
    fused = pre.copy()

    # 假体表面柔化，归一化 0.1 约为 0.5mm
    delta = 0.1
    fused[sdf >= delta] = ct_max

    _ = (sdf >= 0.0) & (sdf < delta)
    fused[_] = metal_min + (sdf[_] / delta) * (ct_max - metal_min)

    _ = (sdf >= -delta) & (sdf < 0.0)
    t = (sdf[_] + delta) / delta
    fused[_] = pre[_] * (1.0 - t) + metal_min * t

    # 保存合成后的 CT
    fused_itk = itk.image_from_array(fused)
    fused_itk.CopyInformation(itk_img)

    itk.imwrite(fused_itk, fused_path.as_posix())


if __name__ == '__main__':
    main()
