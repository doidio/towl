import tempfile
from pathlib import Path

import cadquery as cq
import cadquery.func
import gmsh
import numpy as np


def _shape_to_mesh(shape, tol: float = 1.0):
    try:
        gmsh.clear()
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / '.step'
            cq.exporters.export(shape, f.as_posix())
            gmsh.merge(f.as_posix())

        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), tol)
        gmsh.model.mesh.generate(3)
        # gmsh.model.mesh.optimize('Netgen')

        nodes = gmsh.model.mesh.get_nodes(-1, -1, False, False)
        node0 = np.min(nodes[0])

        points = nodes[1]
        triangles = gmsh.model.mesh.get_elements(2)[2][0].reshape((3, -1)) - node0
        tetrahedrons = gmsh.model.mesh.get_elements(3)[2][0].reshape((4, -1)) - node0
    except Exception as e:
        raise e
    finally:
        gmsh.clear()

    return points, triangles, tetrahedrons


def femoral_prothesis(splines):
    splines = [cq.func.spline(*splines[_, :, :].tolist()) for _ in range(splines.shape[0])]
    solid = cq.func.loft(*splines, splines[0])
    caps = [cq.func.face(solid.edges(_)) for _ in ('>Z', '<Z')]
    solid = cq.func.solid([*caps, solid.faces()])
    return _shape_to_mesh(solid)
