import argparse
from pathlib import Path
from typing import Literal

import streamlit as st
import tomlkit
from minio import Minio, S3Error


@st.cache_resource(show_spinner=False)
def cache_client_pairs(cfg_path: str, category: Literal['context', 'align']):
    return client_pairs(cfg_path, category)


def client_pairs(cfg_path: str, category: Literal['context', 'align']):
    cfg_path = Path(cfg_path)
    cfg = tomlkit.loads(cfg_path.read_text('utf-8'))

    client = Minio(**cfg['minio']['client'])

    # 遍历 MinIO 'pair' 桶，整理术前术后成对的病例数据
    pairs = {}
    for _ in client.list_objects('pair', recursive=True):
        if not _.object_name.endswith('.nii.gz'):
            continue

        # 预期路径结构：PatientID/RL/PreOrPost/image.nii.gz
        if len(paths := _.object_name.split('/')) != 4:
            continue

        pid, rl, op, nii = paths

        if op not in ('pre', 'post'):
            continue

        prl = f'{pid}_{rl}'  # 患者 ID + 左右侧作为唯一标识
        if prl not in pairs:
            pairs[prl] = {'prl': prl}
        pairs[prl][op] = {'nii': f'{pid}/{nii}'}

    # 尝试加载已有结果
    for prl in pairs:
        try:
            data = client.get_object('pair', '/'.join([prl.replace('_', '/'), f'{category}.toml'])).data
            data = tomlkit.loads(data.decode('utf-8')).unwrap()
            pairs[prl].update(data)

            pid, rl = prl.split('_')
            for op in ('pre', 'post'):
                for part in ('hip', 'femur'):
                    data = client.get_object('pair', '/'.join([pid, rl, op, part, 'roi.toml'])).data
                    data = tomlkit.loads(data.decode('utf-8')).unwrap()
                    pairs[prl][op][part] = {'roi': data}
        except S3Error:
            pass

    return client, pairs
