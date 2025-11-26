import numpy as np
import trimesh
import warp as wp



def diff_dmc(volume: wp.array(dtype=wp.float32, ndim=3), origin: np.ndarray, spacing: float | np.ndarray,
             threshold: float, ):
    import torch
    from diso import DiffDMC
    vertices, indices = DiffDMC(dtype=torch.float32)(-wp.to_torch(volume), None, isovalue=-threshold)
    vertices, indices = vertices.cpu().numpy(), indices.cpu().numpy()
    vertices = vertices * spacing * (np.array(volume.shape) - [1, 1, 1]) + origin
    return trimesh.Trimesh(vertices, indices)
