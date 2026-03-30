import argparse
from pathlib import Path
from typing import Literal

import streamlit as st
import tomlkit
from minio import Minio, S3Error


@st.cache_resource(show_spinner=False)
def client_pairs(category: Literal['context', 'align']):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args, _ = parser.parse_known_args()

    cfg_path = Path(args.config)
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
        pairs[prl][op] = f'{pid}/{nii}'

    # 尝试加载已有结果
    for prl in pairs:
        try:
            data = client.get_object('pair', '/'.join([prl.replace('_', '/'), f'{category}.toml'])).data
            data = tomlkit.loads(data.decode('utf-8')).unwrap()
            pairs[prl].update(data)
        except S3Error:
            pass

    return client, pairs


FEMORAL = {
    '': [''],
    'DePuy Corail': ['', '8', '9', '10', '11', '12', '13', '14', '15', '16', '18', '20'],
    'DePuy Tri-Lock': ['', '0', '1', '2', '3', '4', '5', '6', '7', '8'],
    'DePuy SUMMIT': ['', '1', '2', '3', '4', '5', '6', '8'],
    'Stryker Accolade TMZF': ['', '0', '1', '2', '2.5', '3', '3.5', '4', '5', '6', '9'],
    'Stryker Secur-Fit': ['', '6', '7', '8', '9', '10', '11'],
    'Zimmer Tapered': ['', '4', '5', '6', '7.5', '9', '10', '11', '13'],
    'Zimmer Wagner SL': ['', '190/14', '190/15', '190/16', '190/17', '190/18', '190/19', '190/20'],
    'Wagner Cone': ['', '135/13', '135/14', '135/15', '135/16', '135/17', '135/18', '135/19', '135/20', '135/21'],
}
FEMORAL_OFFSET = [0.0, -5.0, -4.0, -3.5, -3.0, -2.7, -2.5, 1.5, 4.0, 5.0, 8.5, 9.0]

FEMORAL_DESC = {
    'DePuy Corail': {
        'zh': '顶部粗方, 向底部快速变尖细的四棱柱',
        'en': 'Thick square top, rapidly narrowing to a sharp narrow bottom tip'
    },
    'DePuy Tri-Lock': {
        'zh': '宽大扁平的顶部, 向下变薄成片状, 尖端钝圆',
        'en': 'Flat and wide upper part, narrowing down to a thin blade with a round tip'
    },
    'DePuy SUMMIT': {
        'zh': '宽扁楔形, 顶部表面粗糙, 底部平滑变细',
        'en': 'Wide flat wedge, rough surface at the top, smooth and thin at the bottom'
    },
    'Stryker Accolade TMZF': {
        'zh': '顶部侧方加宽的楔形, 颈部较细, 底部逐渐变尖',
        'en': 'Side-flaring wedge top, narrow neck, tapering to a sharp bottom'
    },
    'Stryker Secur-Fit': {
        'zh': '顶部显著膨大填充, 底部平滑过渡',
        'en': 'Bulky flared top for filling space, smooth transition to the bottom'
    },
    'Zimmer Tapered': {
        'zh': '标准扁平楔形, 侧方均匀加宽',
        'en': 'Standard flat wedge, uniform side flaring'
    },
    'Zimmer Wagner SL': {
        'zh': '极长的圆柱体, 表面带有深纵向沟槽',
        'en': 'Very long cylindrical rod with deep vertical ridges and flutes'
    },
    'Wagner Cone': {
        'zh': '带锐利纵向肋条的圆锥体',
        'en': 'Cone shape with sharp vertical ridges running along the surface'
    },
}
