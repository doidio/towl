# towl

> [王燎医生](https://m.haodf.com/doctor/2407512001.html)的数字医学开源平台

#### 设计原则

- 体素优于面网格
  - 相比三角面片模型，医学图像体素的解剖理解更直观
- 专用优于通用
  - 相比通用工具箱，专用医工交互界面更易用
  - 相比混合算法集，AI工作流手动即标注更统一
- 多核优于单核
  - 相比ITK/VTK，Python即时编译的GPU算核
    - 运行性能提升100+倍
    - 研发效率提升100+倍
- 分体优于单体
  - 相比Qt/Unity/UE大型引擎，浏览器前端网页更利于医生使用
  - 相比单体应用，计算型后端服务更易于工程实现
    - 部署、分享
    - 持久存储、数据管理
    - AI训练、调优
- 前沿优于广泛
  - 相比Taichi跨硬件兼容，Warp仅限CUDA更极致性能
  - 拒绝兼容某些硬件平台的额外付出

## 运行环境

#### 依赖

- [CUDA Requirements](https://nvidia.github.io/warp/installation.html#cuda-requirements)
- [CUDA Driver](https://www.nvidia.com/en-us/software/nvidia-app)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- [PyTorch](https://pytorch.org/get-started/locally)
- [SAM2](https://github.com/facebookresearch/sam2)

#### Python 3.11

```shell
pip install -U numpy loguru gradio protobuf itk warp-lang cadquery usd-core
```

## 启动应用

```shell
# TotalHip
python total_hip.py
```
