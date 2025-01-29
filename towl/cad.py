import cadquery as cq
import cadquery.func
import numpy as np
from vtkmodules.numpy_interface.dataset_adapter import WrapDataObject


def _shape_to_mesh(shape, tol: float = 1e-3, atol: float = 1e-1, normals: bool = False):
    obj = WrapDataObject(shape.toVtkPolyData(tolerance=tol, angularTolerance=atol, normals=normals))

    if np.all(obj.Polygons.reshape((-1, 4))[:, 0] == 3):
        points = np.array(obj.Points, dtype=float)
        triangles = np.array(obj.Polygons.reshape((-1, 4))[:, 1:], dtype=float)
        point_normals = np.array(obj.PointData['Normals'], dtype=float)
        triangle_normals = np.array(obj.CellData['Normals'], dtype=float)
        return points, triangles, point_normals, triangle_normals
    else:
        raise RuntimeError('Non-triangle mesh')


def femoral_prothesis(splines):
    splines = [cq.func.spline(*splines[_, :, :].tolist()) for _ in range(splines.shape[0])]
    solid = cq.func.loft(*splines, splines[0])
    caps = [cq.func.face(solid.edges(_)) for _ in ('>Z', '<Z')]
    solid = cq.func.solid([*caps, solid.faces()])
    return _shape_to_mesh(solid, 1e-5, 1e-1, True)
