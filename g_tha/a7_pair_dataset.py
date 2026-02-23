import argparse
import multiprocessing
import tempfile
import time
from pathlib import Path

import itk
import numpy as np
import tomlkit
import trimesh
import warp as wp
from PIL import Image
from minio import Minio, S3Error
from tqdm import tqdm

from kernel import diff_dmc, resample_roi, fast_drr


def main(config: str, prl: str, pair: dict):
    """
    处理单个病例的术前/术后配对数据，提取感兴趣区域(ROI)并进行重采样对齐。
    
    :param config: 配置文件路径
    :param prl: 病例标识符 (例如: patientID_R 或 patientID_L)
    :param pair: 包含术前、术后数据路径及配准变换矩阵的字典
    """
    # 加载 TOML 配置文件
    cfg_path = Path(config)
    cfg = tomlkit.loads(cfg_path.read_text('utf-8'))

    # 初始化 Minio 客户端，用于从对象存储中读取数据
    client = Minio(**cfg['minio']['client'])

    # 获取金属的 CT 阈值和目标 ROI 的体素间距 (spacing)
    ct_metal = cfg['ct']['metal']
    roi_spacing = np.ones(3) * cfg['ct']['roi']['spacing']

    # 设置数据集输出根目录
    dataset_root = Path(str(cfg['dataset']['root']))

    # 检查是否已经处理过（通过快照文件是否存在来判断），避免重复处理
    snapshot_file = dataset_root / 'snapshot' / f'{prl}.png'
    if snapshot_file.exists():
        return

    # 根据 pair 字典中的标记决定输出到验证集(val)还是训练集(train)目录
    subdir = 'val' if pair.get('is_val', False) else 'train'

    # 解析左右侧 (R/L)，并从配置中获取对应的 TotalSegmentator 股骨标签值
    rl = prl.split('_')[1]
    label_femur = {
        'R': cfg['totalsegmentator']['label']['right_femur'],
        'L': cfg['totalsegmentator']['label']['left_femur'],
    }[rl]

    # 初始化存储术前/术后图像属性的列表
    ct_images, roi_bounds, ct_femurs, sizes, spacings, origins, image_bgs = [], [], [], [], [], [], []

    # 遍历术前 (pre) 和术后 (post) 数据
    for op, object_name in enumerate((pair['pre'], pair['post'])):
        with tempfile.TemporaryDirectory() as tdir:
            # 1. 下载并读取 TotalSegmentator 的分割结果 (total.nii.gz)
            f = Path(tdir) / 'total.nii.gz'
            try:
                client.fget_object('total', object_name, f.as_posix())
            except S3Error:
                continue

            total = itk.imread(f.as_posix(), itk.UC)
            total = itk.array_from_image(total)

            # 如果分割结果中没有目标股骨标签，则跳过
            if not np.any(total == label_femur):
                continue

            # 获取股骨掩码的体素坐标，并计算其包围盒边界 (min, max)
            ijk = np.argwhere(_ := (total == label_femur))
            ct_femurs.append(_)

            box = np.array([ijk.min(axis=0), ijk.max(axis=0) + 1])
            roi_bounds.append(box.tolist())

            # 2. 下载并读取原始 CT 图像 (image.nii.gz)
            f = Path(tdir) / 'image.nii.gz'
            try:
                client.fget_object('nii', object_name, f.as_posix())
            except S3Error:
                continue

            image = itk.imread(f.as_posix(), itk.SS)

            # 提取图像的物理属性：尺寸、间距、原点 (注意 ITK 读取的顺序通常需要反转以匹配 numpy 的 ZYX 顺序)
            size = np.array([float(_) for _ in reversed(itk.size(image))])
            spacing = np.array([float(_) for _ in reversed(itk.spacing(image))])
            origin = np.array([float(_) for _ in reversed(itk.origin(image))])

            sizes.append(size)
            spacings.append(spacing)
            origins.append(origin)

            # 转换为 numpy 数组并记录背景值 (通常是图像中的最小值，如 -1024)
            image = itk.array_from_image(image)
            ct_images.append(image)

            image_bg = float(np.min(image))
            image_bgs.append(image_bg)

    # 如果术前或术后数据缺失，则抛出异常
    if len(image_bgs) < 2:
        raise RuntimeError('Missing pre-op or post-op data.')

    # 根据术后股骨与金属交集确定采样范围
    # 提取术后图像中既属于股骨区域，CT值又大于金属阈值的部分
    _ = ct_femurs[1] & (ct_images[1] >= ct_metal)
    # 使用 Dual Marching Cubes (DMC) 算法生成金属假体的 3D 网格
    mesh = diff_dmc(wp.from_numpy(_, wp.float32), origins[1], spacings[1], 0.5)

    # 金属可能分离成髋臼杯、球头、股骨柄、膝关节假体，选范围最大的股骨柄以上
    if not mesh.is_empty:
        # 将网格拆分为独立的连通分量，并按包围盒对角线长度降序排序，取最大的组件（通常是股骨柄）
        ls = list(sorted(
            mesh.split(only_watertight=True),
            key=lambda _: np.linalg.norm(_.bounds[1] - _.bounds[0]), reverse=True,
        ))
        mesh: trimesh.Trimesh = ls[0]
    else:
        raise RuntimeError('No metal implant found in the post-op image.')

    # 解析术后到术前的配准变换矩阵
    if 'post_xform_global' in pair:
        # 如果有全局变换矩阵，直接使用
        post_xform = wp.transform(*pair['post_xform_global'])
    elif 'post_xform' in pair:
        # 如果是局部变换矩阵，需要结合图像的原点和裁剪偏移量计算全局变换
        post_xform = wp.transform(*pair['post_xform'])
        post_xform = np.array(wp.transform_to_matrix(post_xform), float).reshape((4, 4))

        # 计算术前和术后图像裁剪区域的物理坐标偏移
        offset = [np.array(origins[_]) + np.array(roi_bounds[_][0]) * np.array(spacings[_]) for _ in range(2)]

        pre = np.identity(4)
        pre[:3, 3] = offset[0]

        post_inv = np.identity(4)
        post_inv[:3, 3] = -offset[1]

        # 组合变换矩阵：平移到术后局部坐标系 -> 应用局部变换 -> 平移回术前全局坐标系
        post_xform = pre @ post_xform @ post_inv
        post_xform = wp.transform_from_matrix(wp.mat44(post_xform))
    else:
        raise RuntimeError('Missing transformation matrix in pair data.')

    # 计算目标 ROI 的轴向包围盒 (AABB)
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2.0
    extents = bounds[1] - bounds[0]

    # 根据物理范围计算体素尺寸
    roi_size = np.ceil(extents / roi_spacing).astype(int)
    # 向上取整到 64 的倍数，以适配后续深度学习网络（如 U-Net/VAE）的多次下采样要求
    roi_size = np.ceil(roi_size / 64.0).astype(int) * 64

    # 构建 ROI 的变换矩阵（仅包含平移到中心点）
    roi_xform = np.identity(4)
    roi_xform[:3, 3] = center

    # 计算 ROI 的起始物理坐标 (原点)
    origin = -0.5 * roi_spacing * roi_size

    roi_xform = wp.transform_from_matrix(wp.mat44(roi_xform))

    # 将 numpy 数组加载为 Warp 的 Volume 对象，以便在 GPU 上进行高效重采样
    volumes = [wp.Volume.load_from_numpy(ct_images[_], bg_value=image_bgs[_]) for _ in range(2)]

    # 初始化输出的双通道图像张量 (通道0: 术后, 通道1: 术前)
    image_roi = wp.full((*roi_size,), wp.vec2(image_bgs[1], image_bgs[0]), wp.vec2)

    # 调用 Warp kernel 进行 GPU 加速的重采样
    wp.launch(resample_roi, image_roi.shape, [
        image_roi, origin, roi_spacing, roi_xform,
        volumes[1].id, origins[1], spacings[1], sizes[1],
        volumes[0].id, origins[0], spacings[0], sizes[0],
        post_xform,
    ])

    # 将结果转回 numpy 数组并分离术前和术后通道
    image_roi = image_roi.numpy()
    image_a, image_b = image_roi[:, :, :, 1], image_roi[:, :, :, 0]

    # 保存重采样后的术前和术后图像为 NIfTI 格式
    for op, image in (('pre', image_a), ('post', image_b)):
        f = dataset_root / op / subdir / f'{prl}.nii.gz'
        f.parent.mkdir(parents=True, exist_ok=True)

        _ = itk.image_from_array(image)
        _.SetSpacing(roi_spacing)
        itk.imwrite(_, f.as_posix(), True)

    # 生成用于可视化的 2D 投影快照 (DRR - Digitally Reconstructed Radiograph)
    snapshot = []
    for ax in (1, 2):  # 分别在冠状面和矢状面生成投影
        stack = [fast_drr(image_a, ax), fast_drr(image_b, ax)]
        img = np.hstack(stack)  # 水平拼接术前和术后投影
        snapshot.append(img)

    # 垂直翻转并拼接两个视角的投影
    snapshot = np.flipud(np.hstack(snapshot))

    # 保存快照图像
    snapshot_file.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(snapshot, 'RGB').save(snapshot_file.as_posix())

    # 显式释放内存和显存资源
    del volumes
    del image_roi
    del ct_images

    import gc
    gc.collect()
    time.sleep(0.5)  # 短暂休眠以确保资源释放完毕


def launch():
    """
    主控函数：解析参数，扫描对象存储中的数据，划分数据集，并启动多进程处理。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to the TOML configuration file')
    parser.add_argument('--max-workers', default=10, type=int, help='Maximum number of concurrent worker processes')
    args = parser.parse_args()

    # 加载配置并初始化 Minio 客户端
    cfg_path = Path(args.config)
    cfg = tomlkit.loads(cfg_path.read_text('utf-8'))
    client = Minio(**cfg['minio']['client'])

    # 扫描 Minio 中的 'pair' 存储桶，收集所有配对的 NIfTI 文件
    pairs = {}
    for _ in client.list_objects('pair', recursive=True):
        if not _.object_name.endswith('.nii.gz'):
            continue

        # 解析对象路径，例如: patientID/R/pre/image.nii.gz
        pid, rl, op, nii = _.object_name.split('/')
        prl = f'{pid}_{rl}'
        if prl not in pairs:
            pairs[prl] = {'prl': prl}
        pairs[prl][op] = f'{pid}/{nii}'

    # 读取每个配对的配准信息 (align.toml) 并过滤无效数据
    valid_pairs = {}
    for prl in pairs:
        try:
            # 获取配准元数据
            data = client.get_object('pair', '/'.join([prl.replace('_', '/'), 'align.toml'])).data
            data = tomlkit.loads(data.decode('utf-8'))

            pairs[prl].update(data)

            # 如果该病例被标记为排除，则跳过
            if len(pairs[prl].get('excluded', [])) > 0:
                continue

            # 如果缺少配准变换矩阵，则跳过
            if 'post_xform_global' not in pairs[prl] and 'post_xform' not in pairs[prl]:
                continue

            # 默认标记为训练集
            pairs[prl]['is_val'] = False
            valid_pairs[prl] = pairs[prl]
        except S3Error:
            pass

    pairs = valid_pairs

    # 划分验证集 (取总数的 10%，最多 100 个)
    keys = sorted(pairs.keys())
    total = len(keys)
    n_val = min(int(total * 0.1), 100)

    if n_val > 0:
        # 均匀采样作为验证集
        for i in range(n_val):
            idx = int(i * total / n_val)
            pairs[keys[idx]]['is_val'] = True

    # 构建多进程任务列表
    tasks = [(args.config, prl, pair) for prl, pair in pairs.items()]

    # 确保单进程显存回收：使用 'spawn' 模式启动子进程，并在每个任务完成后销毁子进程 (maxtasksperchild=1)
    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(processes=args.max_workers, maxtasksperchild=1) as pool:
        # 使用 tqdm 显示进度条，无序映射任务以最大化并发效率
        for _ in tqdm(pool.imap_unordered(process, tasks), total=len(tasks)):
            pass


def process(args):
    """
    多进程任务的包装函数，用于捕获并处理键盘中断异常。
    """
    try:
        main(*args)
    except KeyboardInterrupt:
        print('Keyboard interrupted terminating...')


if __name__ == '__main__':
    launch()
