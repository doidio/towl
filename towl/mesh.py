import tempfile
from pathlib import Path

import cadquery as cq
import cadquery.func
import gmsh
import numpy as np
import trimesh


def _shape_to_mesh(shape, tol: float = 1.0):
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / '.step'
            cq.exporters.export(shape, f.as_posix())

            gmsh.clear()
            gmsh.merge(f.as_posix())

        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), tol)
        gmsh.model.mesh.generate(3)

        nodes = gmsh.model.mesh.get_nodes(-1, -1, False, False)
        node0 = int(np.min(nodes[0]))

        points = nodes[1].astype(float).reshape(-1, 3)
        triangles = gmsh.model.mesh.get_elements(2)[2][0].astype(int).reshape(-1, 3) - node0
        tetrahedrons = gmsh.model.mesh.get_elements(3)[2][0].astype(int).reshape(-1, 4) - node0
    except Exception as e:
        raise e
    finally:
        gmsh.clear()

    return points, triangles, tetrahedrons


def trimesh_to_mesh(vertices, indices, angle_threshold: float = 40.0):
    try:
        mesh = trimesh.Trimesh(vertices, indices)
        mesh.fix_normals()

        if not mesh.is_volume:
            raise RuntimeError(f'Invalid volume ({len(mesh.vertices)} vertices {len(mesh.faces)} faces)')

        all_points = []
        all_triangles = []
        all_tetrahedrons = []
        offset = 0

        for body in mesh.split():
            with tempfile.TemporaryDirectory() as tmpdir:
                f = Path(tmpdir) / '.stl'
                body.export(f.as_posix())

                gmsh.clear()
                gmsh.merge(f.as_posix())

            gmsh.model.mesh.classifySurfaces(np.radians(angle_threshold))
            s = gmsh.model.getEntities(2)
            l = gmsh.model.geo.addSurfaceLoop([e[1] for e in s])
            gmsh.model.geo.addVolume([l])
            gmsh.model.geo.synchronize()
            gmsh.model.mesh.generate(3)

            nodes = gmsh.model.mesh.get_nodes(-1, -1, False, False)
            node0 = int(np.min(nodes[0]))

            points = nodes[1].astype(float).reshape(-1, 3)
            triangles = gmsh.model.mesh.get_elements(2)[2][0].astype(int).reshape(-1, 3) - node0
            tetrahedrons = gmsh.model.mesh.get_elements(3)[2][0].astype(int).reshape(-1, 4) - node0

            all_points.append(points)
            all_triangles.append(triangles + offset)
            all_tetrahedrons.append(tetrahedrons + offset)

            offset += len(points)

        all_points = np.vstack(all_points)
        all_triangles = np.vstack(all_triangles)
        all_tetrahedrons = np.vstack(all_tetrahedrons)
    except Exception as e:
        raise e
    finally:
        gmsh.clear()

    return all_points, all_triangles, all_tetrahedrons

def femoral_prothesis(
        match_points, taper_center,
        taper_x=None, taper_gd=12.50, taper_angle=5 + 2 / 3,
):
    end_x = match_points[0, -1] - match_points[2, -1]
    end_y = match_points[1, -1] - match_points[3, -1]
    end_z = np.cross(end_x, end_y)
    end_x, end_y, end_z = [_ / np.linalg.norm(_) for _ in (end_x, end_y, end_z)]

    splines = [cq.func.spline(*match_points[_, :, :].tolist()) for _ in range(match_points.shape[0])]
    stem = cq.func.loft(*splines, splines[0])
    caps = [cq.func.face(stem.edges(_)) for _ in ('>Z', '<Z')]
    stem = cq.func.solid([*caps, stem.faces()])

    taper_z = taper_center - np.array([*cq.Shape.centerOfMass(stem.faces('>Z'))])
    taper_length = float(np.linalg.norm(taper_z))
    taper_z /= taper_length
    taper_length *= 0.6

    _ = np.cross(taper_z, taper_x)
    taper_x = np.cross(_, taper_z)

    taper = cq.Workplane(cq.Plane(taper_center.tolist(), taper_x.tolist(), taper_z.tolist()))
    taper = taper.circle(taper_gd / 2).extrude(-taper_length, taper=-taper_angle / 2).val()

    neck = cq.func.loft(taper.faces('<Z').wires(), stem.edges('>Z'), cap=True)

    r = match_points[0, -1] - match_points[2, -1]
    r = float(np.linalg.norm(r))
    rr = r * 3
    foot_center = 0.5 * (match_points[0, -1] + match_points[2, -1]) + rr * end_z
    foot = cq.Workplane(cq.Plane(foot_center.tolist(), end_x.tolist(), end_y.tolist()))
    foot = foot.ellipseArc(r, rr, 0, 90, startAtCurrent=False)
    foot = foot.lineTo(r, rr).close()
    foot = foot.revolve().val()

    stem = cq.func.cut(stem, foot, 1e-1)

    solid = cq.func.fuse(taper, neck)
    solid = cq.func.fuse(solid, stem)
    return _shape_to_mesh(solid)
