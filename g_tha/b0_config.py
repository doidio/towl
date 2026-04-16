from pathlib import Path
from typing import Literal

import streamlit as st
import tomlkit
from minio import Minio, S3Error


@st.cache_resource(show_spinner=False)
def cache_client_pairs(cfg_path: str, categories: list[Literal['context', 'align']]):
    return client_pairs(cfg_path, categories)


def client_pairs(config_file: str, categories: list[Literal['context', 'align']]):
    cfg_path = Path(config_file)
    cfg = tomlkit.loads(cfg_path.read_text('utf-8')).unwrap()

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

        if 'nii' not in pairs[prl]:
            pairs[prl]['nii'] = {}
        pairs[prl]['nii'][op] = f'{pid}/{nii}'

    # 尝试加载已有结果
    for prl in pairs:
        pid, rl = prl.split('_')

        for category in categories:
            pairs[prl][category] = {}

            try:
                with client.get_object('pair', '/'.join([prl.replace('_', '/'), f'{category}.toml'])) as response:
                    data = tomlkit.loads(response.read().decode('utf-8')).unwrap()

                for _ in ('prl', 'pre', 'post',):
                    if _ in data:
                        del data[_]

                pairs[prl][category] = data
            except S3Error:
                pass

        pairs[prl]['roi'] = {}
        for part in ('hip', 'femur'):
            pairs[prl]['roi'][part] = {}
            for op in ('pre', 'post'):
                try:
                    with client.get_object('pair', '/'.join([pid, rl, op, part, 'roi.toml'])) as response:
                        data = tomlkit.loads(response.read().decode('utf-8')).unwrap()
                    pairs[prl]['roi'][part][op] = data
                except S3Error:
                    pass

    return client, pairs
