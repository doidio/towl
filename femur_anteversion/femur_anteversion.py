# CUDA driver & PyTorch
# pip install git+https://github.com/newton-physics/newton@f701455313df2ee83ec881d6612657882f2472a0
# pip install -U itk warp-lang newton-clips==0.1.4

import argparse
import json
from pathlib import Path

import itk
import numpy as np
import trimesh
import warp as wp

parser = argparse.ArgumentParser()
parser.add_argument('--name', required=True)
parser.add_argument('--sim_dir')
args = parser.parse_args()

name = args.name

# 读取图像
image = itk.imread(f'fs/{name}.nii.gz')

origin = np.array([*itk.origin(image)])
spacing = np.array([*itk.spacing(image)])
size = np.array([*itk.size(image)])
bg_value = float(np.min(image))

image = itk.array_from_image(image).transpose(2, 1, 0).copy()
volume = wp.types.Volume.load_from_numpy(image, bg_value=bg_value)

# 读取配置
cfg = json.loads(Path(f'fs/{name}.json').read_text('utf-8'))
bone_threshold = cfg['骨阈值']
prothesis_path = cfg['假体']
margin = np.array(cfg['边距'])
keypoints = np.array([spacing * cfg[_] for _ in (
    '股骨柄颈锥圆心', '股骨颈口外缘', '股骨颈口内缘', '股骨小粗隆髓腔中心', '股骨柄末端髓腔中心',
)])
(
    taper_center, neck_lateral, neck_medial, canal_entry, canal_deep,
) = keypoints

# 计算股骨颈截骨面
neck_center = 0.5 * (neck_lateral + neck_medial)
neck_x = neck_lateral - neck_medial
neck_rx = 0.5 * np.linalg.norm(neck_x)
neck_ry = 0.5 * neck_rx
neck_z = neck_center - canal_entry
neck_y = np.cross(neck_z, neck_x)
neck_z = np.cross(neck_x, neck_y)
neck_x, neck_y, neck_z = [_ / np.linalg.norm(_) for _ in (neck_x, neck_y, neck_z)]

canal_z = canal_entry - canal_deep
canal_x = canal_entry - neck_medial
canal_y = np.cross(canal_z, canal_x)
canal_x = np.cross(canal_y, canal_z)
canal_x, canal_y, canal_z = [_ / np.linalg.norm(_) for _ in (canal_x, canal_y, canal_z)]

# 重采样，股骨颈截骨
box = (
    np.min(keypoints, axis=0) - margin,
    np.max(keypoints, axis=0) + margin,
)
region_size = np.round((box[1] - box[0]) / (iso_spacing := 1.0))
region_origin = box[0]

from kernel import femur_proximal_region, diff_dmc

region = wp.context.full(shape=(*region_size,), value=bg_value, dtype=wp.types.float32)
wp.context.launch(femur_proximal_region, region.shape, [
    wp.types.uint64(volume.id), wp.types.vec3(spacing),
    region, wp.types.vec3(region_origin), iso_spacing,
    wp.types.vec3(neck_center), wp.types.vec3(-canal_z),
    wp.types.vec3(neck_center), wp.types.vec3(-neck_z),
    bone_threshold,
])

# 等值面网格重建
femur_mesh = diff_dmc(region, iso_spacing, region_origin, bone_threshold)
femur_mesh.apply_translation([-canal_entry[0], -canal_entry[1], -box[1][2]])
femur_mesh.apply_scale(1e-2)

# 物理模拟

import newtonclips

newtonclips.SAVE_DIR = '.clips'

import newton.utils

builder = newton.ModelBuilder('Z')

builder.add_shape_mesh(
    body=-1,
    mesh=newton.Mesh(femur_mesh.vertices, femur_mesh.faces.flatten()),
    cfg=builder.ShapeConfig(),
    key='femur',
)

mesh = trimesh.load_mesh(f'fs/{prothesis_path}')
mesh.vertices = np.vstack([mesh.vertices[:, 0], mesh.vertices[:, 2], mesh.vertices[:, 1]]).T
mesh.fix_normals()
mesh.apply_translation([0, 0, -np.min(mesh.vertices[:, 2])])
mesh.apply_scale(1e-2)
builder.add_shape_mesh(
    body=builder.add_body(),
    xform=(0, 0, 0, 0, 0, 0, 1),
    mesh=newton.Mesh(mesh.vertices, mesh.faces.flatten()),
    cfg=builder.ShapeConfig(),
    key='femur',
)

model = builder.finalize()
solver = newton.solvers.SemiImplicitSolver(model)
state_0, state_1 = model.state(), model.state()
control = model.control()

renderer = newton.utils.SimRendererOpenGL(model)

fps = 60
frame_dt = 1.0 / fps
sim_substeps = 500
sim_dt = frame_dt / sim_substeps
sim_time = 0.0

for fid in range(num_frames := 500):
    with wp.utils.ScopedTimer(f'frame: {fid}'):
        for _ in range(sim_substeps):
            contacts = model.collide(state_0)
            state_0.clear_forces()
            state_1.clear_forces()

            solver.step(state_0, state_1, control, contacts, sim_dt)

            state_0, state_1 = state_1, state_0

    sim_time += frame_dt

    renderer.begin_frame(sim_time)
    renderer.render(state_0)
    renderer.end_frame()

renderer.save()
