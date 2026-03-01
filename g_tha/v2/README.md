# 骨科假体生成 Latent Diffusion Model 推理指南 (`v2/infer.py`)

本文档详细介绍了基于潜在扩散模型 (Latent Diffusion Model, LDM) 的骨科假体生成推理脚本 `v2/infer.py` 的使用方法、注意事项以及完整的内部代码逻辑。

## 🚀 快速开始

### 基础命令

最基础的推理调用，只需提供术前 CT 图像（NIfTI 格式）和保存目录：

```bash
python v2/infer.py --cond dataset/pre/val/1004333_L.nii.gz --save save
```

### 推荐命令 (上大算力环境)

为了结果可复现并生成中间过程的可视化摘要，推荐固定随机种子并开启摘要功能：

```bash
cd g_tha/v2

# 查看帮助
uv run infer.py --help

# 推理
uv run infer.py --cond ../dataset/pre/val/1004333_L.nii.gz --ldm ldm_latest.pt --vae-pre vae_pre_best.pt --vae-metal vae_metal_best.pt --seed 42 --save save

# 推理并保存过程摘要
uv run infer.py --cond ../dataset/pre/val/1004333_L.nii.gz --ldm ldm_latest.pt --vae-pre vae_pre_best.pt --vae-metal vae_metal_best.pt --seed 42 --save save --ts 1000 --summary
```

---

## ⚙️ 详细参数说明

### 必须参数
*   `--cond` `<str>`: **[必须]** 术前条件图像路径，目前支持 `.nii.gz` 格式。
*   `--save` `<str>`: **[必须]** 生成结果保存的根目录，脚本会自动在内按参数配置生成唯一的子文件夹。

### 生成控制参数
*   `--seed` `<int>`: 随机种子。固定种子可复现相同的初始噪声，这对于固定其他变量（调整 CFG 或采样步数）对比时极为重要。(默认: `None`/随机)
*   `--cfg` `<float>`: Classifier-Free Guidance (CFG) 权重。(默认: `1`)
    *   `0`: 完全无条件生成（忽略术前图像）。
    *   `1`: 仅使用条件生成（无引导增强）。
    *   `>1`: 启用 CFG 引导，值越大对条件（解剖结构）的遵循度越高，但过大可能导致失真。推荐范围 `3.0 ~ 7.0`。
*   `--ts` `<int>`: DDIM 采样步数。通常 `50` 步即可获得较好效果，如果需要极高保真度可尝试 `100` 或 `200`。(默认: `50`)
*   `--summary` / `--no-summary`: 是否生成推理过程的长条摘要图并保存每阶段的 NIfTI 演变。(默认: `False`)

### 硬件与性能参数
*   `--tiled` / `--no-tiled`: 是否对 VAE 使用滑动窗口（分块）推理。(默认: `--tiled` 开启)
    *   开启分块 (`--tiled`) 可极大节省显存，支持在显存较小的显卡上运行大尺寸医疗图像的升降维。
    *   关闭分块 (`--no-tiled`) 会进行全图一次性推断，速度更快，但需要极大的 GPU 显存，显存不够时代码会自动将其降级到 CPU 上计算。
*   `--sw` `<int>`: 开启分块推理时（滑动窗口）的并行 Batch Size。(默认: `4`)
*   `--amp` / `--no-amp`: 是否启用自动混合精度 (AMP)，开启能显著加快推理速度并减少显存占用。(默认: `True`)

### 模型权重路径参数
*   `--vae-pre` `<str>`: 术前图像 VAE 模型权重路径 (默认: `vae_pre_best.pt`)
*   `--vae-metal` `<str>`: 假体金属 VAE 模型权重路径 (默认: `vae_metal_best.pt`)
*   `--ldm` `<str>`: 核心扩散网络（UNet）权重路径 (默认: `ldm_last.pt`)

---

## ⚠️ 注意事项

1.  **显存开销限制**：3D 高分辨率医疗图像极其占用显存，遇到 OOM (Out Of Memory) 报错时，请确保使用 `--tiled` 参数，并适当调小 `--sw` (例如设置 `--sw 1` 或 `--sw 2`)。
2.  **数据预处理假设**：脚本内部通过 `CTBoneNormalized` 对骨骼 HU 值进行了特制的分段线性拉伸，默认假设输入影像已经剔除或不包含过多其他杂乱的高密度无关组织。
3.  **CFG 并非越大越好**：当 `cfg > 10.0` 时，模型预测的方差会被极度放大，容易出现像素过饱和、破坏空间连贯性或产生噪点伪影，请合理调节。
4.  **模型权重状态**：加载 LDM 模型时，脚本会自动优先寻找 EMA (Exponential Moving Average) 权重状态（如果保存了的话），这通常能带来更平滑、稳定和质量更高的生成结果。

---

## 🧠 代码执行顺序与核心逻辑深度解读

推理脚本的生命周期遵循 **参数初始化 -> 模型加载 -> 数据预处理降维 -> 潜空间去噪 -> 升维解码融合 -> 保存** 的严格流程。

### 1. 命令行参数解析与环境初始化
*   利用 `argparse` 读取用户传递的所有指令，处理极值和越界保护。
*   初始化 PyTorch，开启 `cudnn.benchmark` 加速，探测 `cuda` 环境。

### 2. 模型加载 (VAE 与 LDM)
*   **双 VAE 模型 (`AutoencoderKL`)**：循环初始化并加载用于预处理 `pre` (术前条件) 和 `metal` (假体目标) 的 VAE。同时从 Checkpoint 中取出极为重要的统计学参数 `scale_factor` 和 `global_mean`，以确保输入数据分布方差缩放至 `~1.0`。
*   **LDM 网络 (`DiffusionModelUNet`)**：构建包含 3D 注意力机制的 UNet，配置 `DDIMScheduler` 去噪步长器。

### 3. 术前条件图像预处理
*   利用 `itk` 读取患者原始三维 CT 数组（支持 `.nii.gz`）。
*   **核心类 `CTBoneNormalized`（分段线性映射）**：
    *   将 $< 150	ext{ HU}$ 的背景（空气、软组织）直接“截断”抛弃，置为 `-1.0`，让网络聚焦核心骨骼结构。
    *   将 $[150, 650]$ 松质骨区域**非线性拉伸**映射到 $[-1.0, 0.0]$，使得精细骨小梁结构的差异被网络放大观察。
    *   将 $[650, 3000]$ 的高密度皮质骨**压缩**到 $[0.0, 1.0]$ 区间。避免致密骨产生极高损失惩罚而吞噬掉松质骨的细节重构，从而彻底解决生成阶段常见的“振铃效应”。

### 4. 图像编码至潜空间 (VAE Encode)
*   根据是否开启 `--tiled`，执行全图 Encode 还是基于滑动窗口 (`sliding_window_inference`) 块级别 Encode，将庞大的像素级 3D 矩阵压缩 4 倍成为通道数为 4 的紧凑潜在特征图 (Latent)。
*   对压缩后的张量使用 `vae_cond_mean` 和 `vae_cond_scale` 进行缩放偏移，为后续的高斯去噪准备好“调料”。

### 5. LDM 反向去噪生成 (Denoising Loop)
*   结合固定的 `--seed`，在显卡上生成一张和 Latent 尺寸完全一致的高斯纯白噪声张量 `generated`。
*   进入按时间步 `t` 倒序迭代的去噪主循环：
    *   **CFG (Classifier-Free Guidance)**：当 `cfg > 1.0` 时，脚本会将 `generated` 在 Batch 维度复制两份，配以“真实的术前潜空间条件”和“全零的空条件”，同时送给 UNet 推断。使用经典外插公式：$noise = uncond\_pred + 	ext{cfg} 	imes (cond\_pred - uncond\_pred)$ 强化模型对术前解剖约束的服从程度。
    *   使用 `DDIMScheduler.step` 公式利用预测出的噪声修正 `generated`，离干净分布更近一步。
    *   *如果开启 `--summary`，还会在此循环中抽样解码阶段性的预测结果，渲染其切片用于拼接动图。*

### 6. 潜空间解码至像素空间 (VAE Decode)
*   代码中的 `decode_and_save` 闭包函数处理。
*   反向缩放：`z = latent_tensor / vae_image_scale + vae_image_mean`。
*   执行 `vae_image.decode` 将 4 通道的潜向量无损放大回原尺寸的单通道三维数组。这个输出的数组实际上并非直接的 CT 密度，而是表征金属假体的 **连续符号距离场 (SDF, Signed Distance Field)**。

### 7. 假体与术前图像融合后处理
*   建立一个 `fused = original_ct.copy()` 画布。
*   设定软融合边界阈值 `delta = 0.1`。
    *   **核心假体** ($SDF \ge \delta$): 粗暴覆盖，直接填满最大金属密度 `ct_max`。
    *   **软性过渡区** ($-\delta \le SDF < \delta$): 计算基于距离的线性透明度权重 `t_alpha`。在原始术前像素和极亮金属像素之间插值融合，确保边界顺滑无伪影。
    *   **非假体区** ($SDF < -\delta$): 保持不干涉，保留解剖学上的原始骨盆结构。

### 8. 3D 网格提取与结果保存
*   保存 NIfTI 医疗格式影像：生成融合了假体的高亮 `.nii.gz`。
*   保存 SDF 的无损结果：名为 `_metal.nii.gz`，利于后续进行高精度的几何重建分析。
*   **Marching Cubes 面片提取**：利用基于 PyTorch 的开源高精度库 `diso` (`DiffDMC`) 计算 $SDF = 0.0$ 等值面的顶点 (Vertices) 和索引 (Indices)，结合物理间距 `spacing` 恢复空间尺度，最终使用 `trimesh` 导出能够用于打印和二次编辑的高光洁度 `.stl` 3D 模型格式。
*   生成 2D 预览：如果启用了 Summary，将截取中间冠状面中心层的图像，映射成 RGB 色块并添加图例标签，横向拼接为大图输出。
