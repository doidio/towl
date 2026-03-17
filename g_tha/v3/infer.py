import argparse
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from tqdm import tqdm

spacing = 1.0
patch_size = (128,) * 3  # VAE 推理时的滑动窗口大小
bone_min = 150.0  # 骨阈值
metal_min = 2700.0  # 假体金属阈值
ct_min, ct_max = -1024.0, 3071.0  # CT最值
vae_downsample = 4  # VAE 的下采样倍率 (例如: 128 -> 32)


def _printf(text):
    print(text)


def main(
        cond,
        save,
        vae_pre=None,
        vae_metal=None,
        ldm=None,
        cpu=False,
        amp=True,
        sw=4,
        tiled=True,
        seed=None,
        cfg=3,
        ts=50,
        summary=False,
        printf=_printf,
):
    # 参数后处理
    sw_batch_size = max(sw, 1)
    timesteps = min(max(ts, 1), 1000)
    cfg_val = max(float(cfg), 0.0)

    import time
    start_total = time.time()

    # 延迟导入以加快命令行响应速度
    import torch
    if not cpu:
        torch.backends.cudnn.benchmark = True

    from torch import autocast
    from monai.networks.nets import AutoencoderKL, DiffusionModelUNet
    from monai.networks.schedulers import DDIMScheduler
    from monai.transforms import Compose, MapTransform, LoadImaged, SpatialPadd, CenterSpatialCrop, DivisiblePadd
    from monai.inferers import sliding_window_inference

    # 设备选择
    if cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    printf('Device:\t {0}'.format(device))

    # 加载 VAE 模型 (AutoencoderKL)
    vae_dual = []
    for subtask in ('metal', 'pre'):

        vae_path_arg = vae_metal if subtask == 'metal' else vae_pre
        if vae_path_arg is None:
            # 兼容 v3 路径习惯
            vae_path = Path(__file__).parents[1] / 'train' / 'checkpoints' / f'vae_{subtask}_best.pt'
            if not vae_path.exists():
                vae_path = Path(__file__).parent / f'vae_{subtask}_best.pt'
        else:
            vae_path = Path(vae_path_arg)

        if not vae_path.exists():
            raise SystemError(f'Not found:\t {vae_path.resolve()}')
        else:
            printf('VAE:\t [{0}] {1}'.format(subtask, vae_path.resolve()))

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
        if tiled and not cpu:
            vae.to(device)
        else:
            vae.to('cpu')

        # 加载权重
        _ = torch.load(vae_path, map_location='cpu')
        vae.load_state_dict(_['state_dict'])
        vae.eval().float()
        if not cpu and tiled:
            vae.to(device)

        # 获取 latent space 的缩放因子 (用于保持 latent 分布的标准差接近 1)
        scale_factor, global_mean = _['scale_factor'], _['global_mean']
        printf('Epoch:\t {0}'.format(_['epoch']))
        printf('Param:\t {0:.2f} B'.format(sum(p.numel() for p in vae.parameters()) / 1e9))
        vae_dual += [vae, scale_factor, global_mean]

    vae_image, vae_image_scale, vae_image_mean, vae_cond, vae_cond_scale, vae_cond_mean = vae_dual

    # 加载 LDM 模型 (DiffusionModelUNet)
    if ldm is None:
        ldm_path = Path(__file__).parents[1] / 'train' / 'checkpoints' / 'ldm_last.pt'
        if not ldm_path.exists():
            ldm_path = Path(__file__).parent / 'ldm_last.pt'
    else:
        ldm_path = Path(ldm)

    if not ldm_path.exists():
        raise SystemError(f'Not found:\t {ldm_path.resolve()}')
    else:
        printf('LDM:\t {0}'.format(ldm_path.resolve()))

    ldm_model = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=8,
        out_channels=4,
        num_res_blocks=(2, 2, 2),
        channels=(64, 128, 256),
        attention_levels=(False, False, True),  # 启用自注意力学习解剖方位关系
        norm_num_groups=32,
        with_conditioning=False,
        use_flash_attention=not cpu,
    ).to(device)

    _ = torch.load(ldm_path, map_location=device)
    printf('Epoch:\t {0}'.format(_['epoch']))
    printf('Param:\t {0:.2f} B'.format(sum(p.numel() for p in ldm_model.parameters()) / 1e9))

    if 'ema_state' in _:
        ldm_model.load_state_dict(_['ema_state'])
    else:
        ldm_model.load_state_dict(_['state_dict'])

    ldm_model.eval().float()

    printf(
        'CFG:\t {0} {1}'.format(cfg_val, 'Uncond-only' if cfg_val == 0 else 'Cond-only' if cfg_val == 1 else 'Guided'))
    printf('DDIM:\t {0}'.format(timesteps))

    # 初始化 LDM 采样器
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        schedule='scaled_linear_beta',
        prediction_type='epsilon',
        clip_sample=False,
    )
    scheduler.set_timesteps(num_inference_steps=timesteps, device=device)

    # 准备条件图像路径
    cond_path = Path(cond)
    if not cond_path.exists():
        raise SystemError(f'Condition not found:\t {cond_path.resolve()}')

    # 准备后处理工具
    import itk
    import cv2
    from PIL import Image, ImageDraw, ImageFont

    itk_img = itk.imread(cond_path.as_posix())
    original_ct = itk.array_from_image(itk_img).astype(np.float32)  # [D, H, W] -> [Z, Y, X]

    cropper = CenterSpatialCrop(roi_size=original_ct.shape) # 注意 CenterSpatialCrop 接收 (Z, Y, X)
    save_dir = Path(save) / '_'.join([
        cond_path.with_suffix('').with_suffix('').name,
        'seed', str(seed) if seed else 'random',
        'cfg', str(cfg),
        'ts', str(ts),
        'tiled' if tiled else 'no-tiled',
        'summary' if summary else 'no-summary',
    ])
    save_dir.mkdir(parents=True, exist_ok=True)

    # 获取冠状面中心切片索引 (dim 1 in ZYX)
    coronal_idx = original_ct.shape[1] // 2

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
        img = np.flipud(img)  # 保持正位 (Z 轴向上)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    def decode_and_save(latent_tensor, step_idx=None):
        """解码 Latent 并保存融合结果，返回用于摘要的 BGR 切片"""
        # 1. 反向缩放 Latent
        z = latent_tensor / vae_image_scale + vae_image_mean

        # 2. VAE 解码
        if tiled:
            def predictor(patch):
                with autocast(device.type) if amp else nullcontext():
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
        sdf_numpy = decoded[0].cpu().numpy() # [Z, Y, X]

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

        # 5. 保存 NIfTI
        if step_idx is not None:
            step_dir = save_dir / 'summary'
            step_dir.mkdir(parents=True, exist_ok=True)
            out_path = step_dir / f'{step_idx:03d}.nii.gz'
        else:
            out_name = cond_path.name.replace('.nii.gz', f'_fused.nii.gz')
            out_path = save_dir / out_name

        itk_out = itk.image_from_array(fused)
        itk_out.CopyInformation(itk_img)
        itk.imwrite(itk_out, out_path.as_posix())

        # 6. 提取 BGR 切片用于摘要 (dim 1 is Y)
        fused_slice = fused[:, coronal_idx, :] # [Z, X]
        bgr = hu_to_bgr(fused_slice)
        label_text = f'Step {step_idx}' if step_idx is not None else 'Final'
        return sdf_numpy, add_label(bgr, label_text)

    # 准备摘要长图列表
    summary_list = [add_label(hu_to_bgr(original_ct[:, coronal_idx, :]), '术前')]

    # 计算递增采样点列表 (0, 10, 30, 60, 100, 150...)
    sample_schedule = {0}
    _curr, _step, _inc = 0, 10, 10
    while _curr < timesteps:
        sample_schedule.add(_curr)
        _curr += _step
        _step += _inc

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

    # 加载并归一化条件图像 [B, C, Z, Y, X]
    cond_transforms = Compose([
        LoadImaged(keys=['image'], ensure_channel_first=True),
        SpatialPadd(keys=['image'], spatial_size=patch_size, constant_values=ct_min),
        DivisiblePadd(keys=['image'], k=16, constant_values=ct_min),
        CTBoneNormalized(keys=['image']),
    ])
    cond_raw = cond_transforms({'image': cond_path.as_posix()})['image']
    cond_tensor = cond_raw.unsqueeze(0).to(device)

    # VAE 编码 (Encoding)
    if tiled:
        # 分块编码以节省显存
        def encode_predictor(z):
            with autocast(device.type) if amp and device.type != 'cpu' else nullcontext():
                return vae_cond.encode(z)[0]

        with torch.no_grad():
            cond_encoded = sliding_window_inference(
                inputs=cond_tensor,
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
        cond_encoded = cond_tensor.detach().cpu().float()
        with torch.no_grad():
            cond_encoded = vae_cond.encode(cond_encoded)[0]
        cond_encoded = cond_encoded.to(device)

    # 缩放 latent 分布 (减去均值，乘以缩放因子)
    cond_encoded = (cond_encoded - vae_cond_mean) * vae_cond_scale
    printf('LDM before:\t {0:.2f}s'.format(time.time() - start_total))

    # 初始化随机噪声
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    else:
        generator = None

    generated = torch.randn(cond_encoded.shape, device=device, generator=generator)

    printf('LDM generating...')

    # 逐步去噪循环
    start_gen = time.time()
    for i, t in enumerate(bar := tqdm(scheduler.timesteps, desc='LDM generating')):
        bar.set_postfix({'DDIM': t.item()})

        # 预测噪声
        with torch.no_grad():
            t_input = t[None].to(device)

            if cfg_val > 1.0:
                # 并行计算条件与无条件预测
                latent_input = torch.cat([generated] * 2)
                cond_input = torch.cat([cond_encoded, torch.zeros_like(cond_encoded)])
                model_input = torch.cat([latent_input, cond_input], dim=1)

                with autocast(device.type) if amp and device.type != 'cpu' else nullcontext():
                    noise_pred_batch = ldm_model(model_input, t_input.repeat(2))

                # CFG 引导公式: noise = uncond + w * (cond - uncond)
                noise_cond, noise_uncond = noise_pred_batch.chunk(2)
                noise_pred = noise_uncond + cfg_val * (noise_cond - noise_uncond)

            elif cfg_val == 1.0:
                # 仅条件路
                model_input = torch.cat([generated, cond_encoded], dim=1)
                with autocast(device.type) if amp and device.type != 'cpu' else nullcontext():
                    noise_pred = ldm_model(model_input, t_input)

            else:
                # 仅无条件路
                model_input = torch.cat([generated, torch.zeros_like(cond_encoded)], dim=1)
                with autocast(device.type) if amp and device.type != 'cpu' else nullcontext():
                    noise_pred = ldm_model(model_input, t_input)

        # DDIM 更新步
        with torch.no_grad():
            # scheduler.step 返回 (当前步结果, 预测的原始干净样本)
            generated, pred_0 = scheduler.step(noise_pred, t, generated)

        # 保存中间步骤与摘要图 (如果启用采样计划)
        if summary and (i in sample_schedule):
            with torch.no_grad():
                # 解码预测出的干净样本并添加到摘要长图
                _, bgr_slice = decode_and_save(pred_0, step_idx=i)
                summary_list.append(bgr_slice)

    printf('LDM generation:\t {0:.2f}s ({1:.2f} s/it)'.format(
        time.time() - start_gen, (time.time() - start_gen) / timesteps))

    # 最终解码并融合
    start_dec = time.time()
    generated_np, bgr_final = decode_and_save(generated)
    summary_list.append(bgr_final)

    # 保存摘要长图 (如果启用)
    if summary:
        summary_img = np.hstack(summary_list)
        summary_path = save_dir / cond_path.name.replace('.nii.gz', f'_summary.png')
        cv2.imwrite(summary_path.as_posix(), summary_img)

    # 提取等值面网格体 (STL)
    if device.type != 'cpu':
        stl_name = cond_path.name.replace('.nii.gz', f'_metal.stl')
        stl_path = save_dir / stl_name

        from diso import DiffDMC
        import trimesh

        # 将 [Z, Y, X] 转置为 [X, Y, Z] 后进行重建，确保顶点坐标天然为 (x, y, z) 顺序
        final_sdf_xyz = torch.from_numpy(generated_np.transpose(2, 1, 0)).to(device)

        vertices, indices = DiffDMC(dtype=torch.float32)(-final_sdf_xyz, None, isovalue=0.0)
        vertices, indices = vertices.cpu().numpy(), indices.cpu().numpy()
        
        # 直接应用 spacing
        vertices = vertices * spacing

        mesh = trimesh.Trimesh(vertices, indices)
        mesh.export(stl_path.as_posix())

    # 保存假体 SDF 结果 (Metal SDF)
    save_metal = save_dir / cond_path.name.replace('.nii.gz', f'_metal.nii.gz')

    itk_metal = itk.image_from_array(generated_np)
    itk_metal.CopyInformation(itk_img)
    itk.imwrite(itk_metal, save_metal.as_posix())

    printf('LDM after:\t {0:.2f}s'.format(time.time() - start_dec))
    printf('Total time:\t {0:.2f}s'.format(time.time() - start_total))


if __name__ == '__main__':
    b = argparse.BooleanOptionalAction
    parser = argparse.ArgumentParser(description='Latent Diffusion Model 推理脚本')

    # 必须参数
    parser.add_argument('--cond', type=str, required=True, help='术前条件图像路径 (.nii.gz)')
    parser.add_argument('--save', type=str, required=True, help='生成结果保存目录')

    # 模型路径参数 (默认使用脚本同级目录下的 .pt 文件)
    parser.add_argument('--vae-pre', type=str, default=None,
                        help='VAE模型路径 (默认: vae_pre_best.pt)')
    parser.add_argument('--vae-metal', type=str, default=None,
                        help='VAE模型路径 (默认: vae_metal_best.pt)')
    parser.add_argument('--ldm', type=str, default=None, help='LDM模型路径 (默认: ldm_last.pt)')

    # 硬件与性能参数
    parser.add_argument('--cpu', action='store_true', help='强制使用 CPU 推理 (默认: False)')
    parser.add_argument('--amp', action=b, default=True, help='是否启用混合精度 (默认: True)')
    parser.add_argument('--sw', type=int, default=4, help='滑动窗口推理时的并行 Batch Size')
    parser.add_argument('--tiled', action=b, default=False, help='是否使用分块推理 (True: 节省显存, False: 全图推理)')

    # 生成控制参数
    parser.add_argument('--seed', type=int, default=None, help='随机种子，固定种子可复现结果 (默认: None/随机)')
    parser.add_argument('--cfg', type=int, default=3, help='Classifier-Free Guidance 权重 (0: 无条件, >1: 条件增强)')
    parser.add_argument('--ts', type=int, default=50, help='DDIM 采样步数 (通常 50-200)')
    parser.add_argument('--summary', action=b, default=False, help='是否生成推理过程摘要图并保存中间步骤 (默认: False)')

    args = parser.parse_args()

    try:
        main(
            cond=args.cond,
            save=args.save,
            vae_pre=args.vae_pre,
            vae_metal=args.vae_metal,
            ldm=args.ldm,
            cpu=args.cpu,
            amp=args.amp,
            sw=args.sw,
            tiled=args.tiled,
            seed=args.seed,
            cfg=args.cfg,
            ts=args.ts,
            summary=args.summary,
        )
    except KeyboardInterrupt:
        print('Keyboard interrupted terminating...')
