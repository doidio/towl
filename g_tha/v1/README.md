# Latent Diffusion Model 推理脚本使用说明

本目录下的 `infer.py` 脚本用于使用训练好的 Latent Diffusion Model (LDM) 和 VAE 模型，根据术前 CT 图像（Condition）生成对应的目标图像（例如：术后预测、缺失修补等）。

## 1. 环境依赖

确保已安装以下 Python 库：
- `torch`
- `monai`
- `numpy`
- `tqdm`

推荐使用 GPU 进行推理以获得更快的速度。

## 2. 快速开始

最简单的运行方式如下：

```bash
python infer.py --cond path/to/pre_op_ct.nii.gz --save path/to/output_dir
```

该命令将使用默认的 `vae_best.pt` 和 `ldm_best.pt` 模型（需位于同级目录），对 `pre_op_ct.nii.gz` 进行推理，并将结果保存到 `path/to/output_dir`。

## 3. 参数详解

| 参数 | 类型 | 必选 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| `--cond` | `str` | **是** | - | 术前条件图像的路径 (`.nii.gz`)。 |
| `--save` | `str` | **是** | - | 生成结果的保存目录。 |
| `--vae` | `str` | 否 | `./vae_best.pt` | VAE 模型的路径。 |
| `--ldm` | `str` | 否 | `./ldm_best.pt` | LDM 模型的路径。 |
| `--sw` | `int` | 否 | `4` | 滑动窗口推理时的并行 Batch Size。显存越大可设越大。 |
| `--cfg` | `int` | 否 | `1` | Classifier-Free Guidance (CFG) 权重。<br>`0`: 无条件生成 (忽略输入条件)<br>`1`: 标准条件生成<br>`>1`: 增强条件引导 |
| `--ts` | `int` | 否 | `50` | DDIM 采样步数。通常 `50-200` 步能获得较好效果，步数越多越精细但速度越慢。 |
| `--seed` | `int` | 否 | `42` | 随机种子。固定种子可保证每次运行生成相同的结果。 |
| `--tiled` | `flag` | 否 | `True` | 开启分块 (Tiled) 推理。默认为开启。<br>使用 `--no-tiled` 关闭。<br>开启可节省显存，支持处理大尺寸图像；关闭则全图一次性推理，速度可能稍快但极耗显存。 |
| `--amp` | `flag` | 否 | `True` | 开启混合精度推理 (AMP)。默认为开启。<br>使用 `--no-amp` 关闭。 |

## 4. 示例命令

### 4.1. 高质量生成 (增加步数和引导)
```bash
python infer.py --cond patient_001.nii.gz --save ./results --ts 100 --cfg 3
```

### 4.2. 使用指定模型路径
```bash
python infer.py --cond test.nii.gz --save ./out --vae ../models/vae_v2.pt --ldm ../models/ldm_v2.pt
```

### 4.3. 显存充足时关闭分块以加速
```bash
python infer.py --cond input.nii.gz --save ./out --no-tiled
```

## 5. 输出文件说明

生成的文件将保存到 `--save` 指定的目录中。文件名格式通常为：

```
{原始文件名}_seed_{seed}_cfg_{cfg}_ts_{ts}_{tiled_status}.nii.gz
```

例如：`patient_001_seed_42_cfg_1_ts_50_tiled.nii.gz`

## 6. 注意事项

- **输入图像范围**: 脚本内部使用 `CTBoneNormalized` 类将 CT 值在 `[150, 3000]` 范围内的像素映射到 `[-1, 1]`。
    - **重要**: 小于 150 的值（如软组织、空气）会被强制截断并映射为 -1.0。这意味着模型主要关注**骨骼结构**。
- **显存占用**: 如果遇到 CUDA Out of Memory (OOM) 错误：
    1. 确保未关闭 `--tiled` (默认是开启的)。
    2. 减小 `--sw` 参数的值 (例如 `--sw 2` 或 `--sw 1`)。
