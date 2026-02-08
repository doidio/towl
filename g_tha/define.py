import numpy as np
import torch
from monai.losses import PerceptualLoss
from monai.networks.nets import AutoencoderKL, PatchDiscriminator
from monai.transforms import (
    LoadImaged, MapTransform, RandCropByPosNegLabeld, ThresholdIntensityd, CopyItemsd, DeleteItemsd)


class CTBoneNormalized(MapTransform):
    """基于线性分段函数的 CT 值映射变换 (Linear Piecewise Mapping)"""

    def __init__(self, keys, reverse=False, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.src_pts = [150.0, 650.0, 1500.0, 3000.0]  # 源域的关键点，HU 值
        self.dst_pts = [-1.0, 0.0, 0.5, 1.0]  # 目标域的关键点，归一化后的值
        if reverse:
            self.src_pts, self.dst_pts = self.dst_pts, self.src_pts

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            img = d[key]
            if isinstance(img, torch.Tensor):
                img_np = img.detach().cpu().numpy()
                mapped = np.interp(img_np, self.src_pts, self.dst_pts)
                d[key] = torch.from_numpy(mapped).to(img.device) if isinstance(img, torch.Tensor) else mapped
            else:
                mapped = np.interp(img, self.src_pts, self.dst_pts)
                d[key] = mapped.astype(np.float32)
            d[key] = mapped.astype(np.float32)
        return d


def autoencoder():
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


def autoencoder_train_transforms(patch_size, label_threshold):
    return [
        LoadImaged(keys=['image'], ensure_channel_first=True),
        CopyItemsd(keys=['image'], times=1, names=['label']),
        ThresholdIntensityd(keys=['label'], threshold=label_threshold, above=True, cval=0),
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


def autoencoder_val_transforms():
    return [
        LoadImaged(keys=['image'], ensure_channel_first=True),
        CTBoneNormalized(keys=['image']),
    ]


def autoencoder_encode_decode_mu(model, inputs):
    """确定性编解码，用于验证阶段滑动窗口推理"""
    return model.decode(model.encode(inputs)[0])
