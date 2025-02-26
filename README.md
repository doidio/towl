# towl

> 数字医工开源平台

## Strategic choices

> #### 体素优于面网格
> 相比三维重建之后的三角面片模型，医学专家更擅长阅读原始图像体素，直接洞察解剖细节

> #### 专用优于通用
> 相比通用工具箱，专用医工交互界面更易用
> 相比混合算法集，AI工作流手动即标注更统一

> #### 多核优于单核
> 相比ITK/VTK，Python即时编译的GPU算核运行性能提升100+倍，同时研发效率提升100+倍

> #### 分体优于单体
> 相比Qt/Unity/UE单体应用，浏览器前端网页更利于医生使用，计算型后端服务更易于工程部署、分享，
> 能够持久存储数据，支持直接进行AI训练、调优

> #### 前沿优于广泛
> 相比Taichi跨硬件兼容，Warp仅限CUDA更极致性能，不为兼容特定硬件平台而牺牲先进性

## 运行环境

#### 依赖

- [CUDA Requirements](https://nvidia.github.io/warp/installation.html#cuda-requirements)
- [CUDA Driver](https://www.nvidia.com/en-us/software/nvidia-app)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

#### Python 3.11

```shell
pip install -U numpy loguru gradio protobuf itk warp-lang cadquery gmsh diso
```

## 启动应用

```shell
# TotalHip
python total_hip.py
```
