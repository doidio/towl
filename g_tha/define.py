from copy import deepcopy

import monai
import numpy as np
import torch
from monai.losses import PerceptualLoss
from monai.networks.nets import AutoencoderKL, PatchDiscriminator, DiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler, DDIMScheduler
from monai.transforms import (
    LoadImaged, MapTransform, RandCropByPosNegLabeld, ThresholdIntensityd, CopyItemsd, DeleteItemsd, SpatialPadd,
)

bone_min = 150.0


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

            device = img_t.device
            dtype = img_t.dtype

            xp = torch.tensor(self.src_pts, device=device, dtype=dtype)
            fp = torch.tensor(self.dst_pts, device=device, dtype=dtype)

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
                d[key] = res.to(dtype=dtype)  # 保持原有精度

        return d


vae_downsample = 4


def vae_kl():
    return AutoencoderKL(
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


def discriminator():
    return PatchDiscriminator(
        spatial_dims=3,
        channels=64,  # 起始通道数
        in_channels=1,  # 输入与编码器一致
        out_channels=1,  # 输出必须是单通道 (Real/Fake Score)
        num_layers_d=3,  # 3层下采样，感受野适中，关注局部纹理细节
    )


def perceptual_loss():
    return PerceptualLoss(
        spatial_dims=3,
        network_type='medicalnet_resnet50_23datasets',
        is_fake_3d=False,
        pretrained=True,
    )


def vae_train_transforms(subtask, patch_size):
    if subtask in ('pre',):
        return [
            LoadImaged(keys=['image'], ensure_channel_first=True),
            SpatialPadd(keys=['image'], spatial_size=patch_size, constant_values=-1024),
            CopyItemsd(keys=['image'], times=1, names=['label']),
            ThresholdIntensityd(keys=['label'], threshold=bone_min, above=True, cval=0),
            RandCropByPosNegLabeld(
                keys=['image'],
                label_key='label',
                spatial_size=patch_size,
                pos=2, neg=1,
                num_samples=1,
            ),
            DeleteItemsd(keys=['label']),
            CTBoneNormalized(keys=['image']),
        ]
    elif subtask in ('metal',):
        return [
            LoadImaged(keys=['image'], ensure_channel_first=True),
            SpatialPadd(keys=['image'], spatial_size=patch_size, constant_values=-1.0),
            CopyItemsd(keys=['image'], times=1, names=['label']),
            ThresholdIntensityd(keys=['label'], threshold=-0.95, above=True, cval=0),
            RandCropByPosNegLabeld(
                keys=['image'],
                label_key='label',
                spatial_size=patch_size,
                pos=2, neg=1,
                num_samples=1,
            ),
            DeleteItemsd(keys=['label']),
        ]
    else:
        raise SystemError(f'Unknown VAE subtask {subtask}')


def vae_val_transforms(subtask):
    if subtask in ('pre',):
        return [
            LoadImaged(keys=['image'], ensure_channel_first=True),
            CTBoneNormalized(keys=['image']),
        ]
    elif subtask in ('metal',):
        return [
            LoadImaged(keys=['image'], ensure_channel_first=True),
        ]
    else:
        raise SystemError(f'Unknown VAE subtask {subtask}')


class LoadLatentConditiond(MapTransform):
    """读取 .npy 文件 latent 数据 [8, D, H, W] float16"""

    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        # 加载 npy
        data_npy = np.load(d['image'])

        # 转换为 Tensor
        if isinstance(data_npy, np.ndarray):
            data_tensor = torch.from_numpy(data_npy).float()
        else:
            data_tensor = data_npy.float()

        d['image'] = data_tensor[0:4]  # 术后
        d['condition'] = data_tensor[4:8]  # 术前

        return d


def ldm_transforms():
    return [
        LoadLatentConditiond(keys=['image']),
    ]


def ldm_unet():
    return DiffusionModelUNet(
        spatial_dims=3,
        in_channels=8,
        out_channels=4,
        num_res_blocks=(2, 2, 2),
        channels=(64, 128, 256),
        attention_levels=(False, False, True),  # 启用自注意力学习解剖方位关系
        norm_num_groups=32,
        with_conditioning=False,  # TODO 交叉注意力注入全局条件
        use_flash_attention=True,
    )


def scheduler_ddpm():
    return DDPMScheduler(
        num_train_timesteps=1000,
        schedule='scaled_linear_beta',
        prediction_type='epsilon',
        # beta_start=0.00085,  # LDM 标准参数
        # beta_end=0.012,  # LDM 标准参数
    )


def scheduler_ddim():
    return DDIMScheduler(
        num_train_timesteps=1000,
        schedule='scaled_linear_beta',
        prediction_type='epsilon',
        # beta_start=0.00085,
        # beta_end=0.012,
        clip_sample=False,
    )


def vae_decode_tiled(z, vae, patch_size, sw_batch_size, overlap=0.25):
    """
    针对 LDM 的 Tiled VAE Decoding，直接在 Latent 空间进行滑动窗口。
    z: [B, C, D, H, W] (Latent)
    patch_size: 图像空间的 ROI Size (例如 [128, 128, 128])
    """
    # 计算 Latent 空间的 ROI Size
    latent_roi_size = [s // vae_downsample for s in patch_size]

    def decode_predictor(latent_patch):
        return vae.decode(latent_patch)

    # 5. 滑动窗口推理
    recon = monai.inferers.sliding_window_inference(
        inputs=z,
        roi_size=latent_roi_size,
        sw_batch_size=sw_batch_size,
        predictor=decode_predictor,
        overlap=overlap,
        mode='gaussian',
        device=z.device,
        progress=False,
    )

    return recon


class EMA:
    """指数移动平均 (Exponential Moving Average) 用于稳定扩散模型的生成质量"""

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {}
        self.original = {}

        # 注册模型参数到 shadow 字典
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        """在每个训练 step 后更新 EMA 权重"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def store(self, model):
        """暂存当前模型的真实权重 (验证前调用)"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.original[name] = param.data.clone()

    def copy_to(self, model):
        """将 EMA 权重应用到模型 (验证时调用)"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        """恢复模型的真实权重 (验证后调用，继续训练)"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.original
                param.data.copy_(self.original[name])
        self.original = {}

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = deepcopy(state_dict)
