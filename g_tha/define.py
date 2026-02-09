import numpy as np
import torch
from monai.losses import PerceptualLoss
from monai.networks.nets import AutoencoderKL, PatchDiscriminator, DiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler
from monai.transforms import (
    LoadImaged, MapTransform, RandCropByPosNegLabeld, ThresholdIntensityd, CopyItemsd, DeleteItemsd,
)

bone_range = [150.0, 650.0]


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


def vae():
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


def vae_train_transforms(patch_size):
    return [
        LoadImaged(keys=['image'], ensure_channel_first=True),
        CopyItemsd(keys=['image'], times=1, names=['label']),
        ThresholdIntensityd(keys=['label'], threshold=bone_range[0], above=True, cval=0),
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


def vae_val_transforms():
    return [
        LoadImaged(keys=['image'], ensure_channel_first=True),
        CTBoneNormalized(keys=['image']),
    ]


class LoadLatentConditiond(MapTransform):
    """读取 .npy 文件 latent 数据 [8, D, H, W] float16"""

    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            # 加载 npy
            data_npy = np.load(d[key])  # [8, D, H, W]

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
        attention_levels=(False, True, True),  # 启用自注意力学习解剖方位关系
        norm_num_groups=32,
        with_conditioning=False,  # 通道拼接方式禁用交叉注意力
    )


def scheduler_ddpm():
    return DDPMScheduler(
        num_train_timesteps=1000,
        schedule='linear_beta',
        prediction_type='epsilon',
    )
