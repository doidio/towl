import argparse
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from tqdm import tqdm

# 全局配置
patch_size = (128,) * 3  # VAE 推理时的滑动窗口大小
bone_range = [150.0, 650.0]  # 骨窗范围，用于归一化
vae_downsample = 4  # VAE 的下采样倍率 (例如: 128 -> 32)


def main():
    # --------------------------------------------------------------------------
    # 1. 参数解析
    # --------------------------------------------------------------------------
    b = argparse.BooleanOptionalAction
    parser = argparse.ArgumentParser(description="Latent Diffusion Model 推理脚本")

    # 必须参数
    parser.add_argument('--cond', type=str, required=True, help='术前条件图像路径 (.nii.gz)')
    parser.add_argument('--save', type=str, required=True, help='生成结果保存目录')

    # 模型路径参数 (默认使用脚本同级目录下的 .pt 文件)
    parser.add_argument('--vae', type=str, default=None, help='VAE模型路径 (默认: ./vae_best.pt)')
    parser.add_argument('--ldm', type=str, default=None, help='LDM模型路径 (默认: ./ldm_best.pt)')

    # 硬件与性能参数
    parser.add_argument('--amp', action=b, default=True, help='是否启用混合精度 (默认: True)')
    parser.add_argument('--sw', type=int, default=4, help='滑动窗口推理时的并行 Batch Size')
    parser.add_argument('--tiled', action=b, default=True, help='是否使用分块推理 (True: 节省显存, False: 全图推理)')

    # 生成控制参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子，固定种子可复现结果')
    parser.add_argument('--cfg', type=int, default=1, help='Classifier-Free Guidance 权重 (0: 无条件, >1: 条件增强)')
    parser.add_argument('--ts', type=int, default=50, help='DDIM 采样步数 (通常 50-200)')

    args = parser.parse_args()

    # 参数后处理
    sw_batch_size = max(args.sw, 1)
    timesteps = min(max(args.ts, 1), 1000)
    guide_weight = max(args.cfg, 0.0)

    # 延迟导入以加快命令行响应速度
    import torch
    from torch import autocast
    from monai.networks.nets import AutoencoderKL, DiffusionModelUNet
    from monai.networks.schedulers import DDIMScheduler
    from monai.transforms import Compose, MapTransform, LoadImaged, SaveImage
    from monai.inferers import sliding_window_inference

    # 设备选择 (优先使用 CUDA)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device:\t {device}')

    # --------------------------------------------------------------------------
    # 2. 加载 VAE 模型 (AutoencoderKL)
    # --------------------------------------------------------------------------
    # 确定模型路径
    if args.vae is None:
        vae_path = Path(__file__).parent / 'vae_best.pt'
    else:
        vae_path = Path(args.vae)

    if not vae_path.exists():
        raise SystemError(f'VAE not found:\t {vae_path.resolve()}')
    else:
        print(f'VAE Loading:\t {vae_path.resolve()}')

    # 初始化 VAE 网络结构
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
    scale_factor = _['scale_factor']
    print(f'VAE Scale:\t {scale_factor}')

    # --------------------------------------------------------------------------
    # 3. 加载 LDM 模型 (DiffusionModelUNet)
    # --------------------------------------------------------------------------
    if args.ldm is None:
        ldm_path = Path(__file__).parent / 'ldm_best.pt'
    else:
        ldm_path = Path(args.ldm)

    if not ldm_path.exists():
        raise SystemError(f'LDM not found:\t {ldm_path.resolve()}')
    else:
        print(f'LDM loading:\t {ldm_path.resolve()}')

    # 初始化 LDM 网络结构
    ldm = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=8,  # 4 (Noisy Latent) + 4 (Condition Latent)
        out_channels=4,  # 预测的噪声
        num_res_blocks=(2, 2, 2),
        channels=(64, 128, 256),
        attention_levels=(False, False, True),
        norm_num_groups=32,
        with_conditioning=False,  # 使用 concat 方式作为条件，而非 cross-attention
        use_flash_attention=True,
    ).to(device)

    # 加载权重
    _ = torch.load(ldm_path, map_location=device)
    ldm.load_state_dict(_['state_dict'])
    ldm.eval().float()

    # --------------------------------------------------------------------------
    # 4. 数据预处理类定义
    # --------------------------------------------------------------------------
    class CTBoneNormalized(MapTransform):
        """
        基于线性分段函数的 CT 值映射变换 (Linear Piecewise Mapping)
        将骨窗范围 [150, 650] 映射到归一化空间
        """

        def __init__(self, keys, reverse=False, allow_missing_keys=False):
            super().__init__(keys, allow_missing_keys)

            # 定义源区间 (CT值) 和 目标区间 (归一化值)
            self.src_pts = [*bone_range, 1500.0, 3000.0]
            self.dst_pts = [-1.0, 0.0, 0.5, 1.0]

            self.reverse = reverse
            # 反向变换时交换源和目标
            if self.reverse:
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

                if self.reverse:
                    x_input = img_t
                else:
                    # 正向归一化时，截断超出范围的值
                    x_input = torch.clamp(img_t, min=xp[0], max=xp[-1])

                # 线性插值计算
                indices = torch.searchsorted(xp, x_input, right=True)
                indices = torch.clamp(indices, 1, len(xp) - 1)

                idx0 = indices - 1
                idx1 = indices

                x0 = xp[idx0]
                x1 = xp[idx1]
                y0 = fp[idx0]
                y1 = fp[idx1]

                res = y0 + (x_input - x0) * (y1 - y0) / (x1 - x0)

                if is_numpy:
                    d[key] = res.cpu().numpy().astype(np.float32)
                else:
                    d[key] = res.to(dtype=img_t.dtype)  # 保持原有精度

            return d

    # --------------------------------------------------------------------------
    # 5. 准备条件图像 (Condition)
    # --------------------------------------------------------------------------
    cond_transforms = Compose([
        LoadImaged(keys=['image'], ensure_channel_first=True),
        CTBoneNormalized(keys=['image']),
    ])

    cond_path = Path(args.cond)

    if not cond_path.exists():
        raise SystemError(f'Condition not found:\t {cond_path.resolve()}')
    else:
        print(f'Cond loading:\t {cond_path.resolve()}')

    # 加载并归一化条件图像 [B, C, H, W, D]
    cond = cond_transforms({'image': cond_path.as_posix()})['image'].unsqueeze(0).to(device)

    print(f'Cond encoding:\t {cond.shape}')

    # --------------------------------------------------------------------------
    # 6. VAE 编码 (Encoding)
    # --------------------------------------------------------------------------
    if args.tiled:
        # 分块编码以节省显存
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
        # 全图直接编码
        cond = cond.detach().cpu().float()
        with torch.no_grad():
            cond = vae.encode(cond)[0]
        cond = cond.to(device)

    # 缩放 latent 分布
    cond = cond * scale_factor
    print(f'Cond encoded:\t {cond.shape}')

    # --------------------------------------------------------------------------
    # 7. LDM 采样 (Sampling)
    # --------------------------------------------------------------------------
    # 配置 DDIM 调度器
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        schedule='scaled_linear_beta',
        prediction_type='epsilon',
        clip_sample=False,
    )
    scheduler.set_timesteps(num_inference_steps=timesteps, device=device)

    # 初始化随机噪声
    generator = torch.Generator(device=device).manual_seed(args.seed)
    generated = torch.randn(cond.shape, device=device, generator=generator)

    # 逐步去噪循环
    for t in (pbar := tqdm(scheduler.timesteps, desc='LDM generating')):
        pbar.set_postfix({'DDIM': t.item()})

        # 准备 Classifier-Free Guidance (CFG) 输入
        # 复制一份输入 latent 用于并行计算 (有条件 + 无条件)
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
        noise_pred = noise_uncond + guide_weight * (noise_cond - noise_uncond)

        # DDIM 更新步
        with torch.no_grad():
            generated, _ = scheduler.step(noise_pred, t, generated)

    mi, ma = generated.min(), generated.max()
    print(mi.item(), ma.item())

    # 反向缩放 latent
    generated = generated / scale_factor

    print(f'Generated decoding:\t {generated.shape}')

    # --------------------------------------------------------------------------
    # 8. VAE 解码 (Decoding)
    # --------------------------------------------------------------------------
    if args.tiled:
        # 分块解码
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
        # 全图解码
        generated = generated.detach().cpu().float()
        with torch.no_grad():
            generated = vae.decode(generated)
        generated = generated.to(device)

    print(f'Generated decoded:\t {generated.shape}')

    # --------------------------------------------------------------------------
    # 9. 保存结果
    # --------------------------------------------------------------------------
    # 反归一化: [-1, 1] -> CT 值
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
