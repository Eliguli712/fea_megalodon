#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json
import time
import numpy as np
import trimesh
import meshio
import pytetwild
import plotly.graph_objects as go


def boundary_faces_from_tets(tets: np.ndarray) -> np.ndarray:
    f0 = tets[:, [0, 1, 2]]
    f1 = tets[:, [0, 3, 1]]
    f2 = tets[:, [1, 3, 2]]
    f3 = tets[:, [2, 3, 0]]
    faces = np.vstack((f0, f1, f2, f3))
    sfaces = np.sort(faces, axis=1)
    _, idx, counts = np.unique(sfaces, axis=0, return_index=True, return_counts=True)
    return faces[idx[counts == 1]]


def edge_stats(triangles: np.ndarray) -> tuple[int, int]:
    edges = np.vstack((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]))
    edges = np.sort(edges, axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return int(np.sum(counts == 1)), int(np.sum(counts > 2))


def main() -> None:
    root = Path('/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces')
    input_obj = root / 'tetwild_input_smoothed.obj'
    out_msh = root / 'tetwild_input_smoothed.msh'
    out_html = root / 'tetwild_input_smoothed_preview.html'
    out_json = root / 'tetwild_input_smoothed_mesh_report.json'

    m = trimesh.load(input_obj, force='mesh', process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(tuple(m.geometry.values()))

    V = np.asarray(m.vertices, dtype=np.float64)
    F = np.asarray(m.faces, dtype=np.int32)

    edge_length_fac = 0.02
    optimize = False
    t0 = time.time()
    Vt, Tt = pytetwild.tetrahedralize(V, F, optimize=optimize, edge_length_fac=edge_length_fac)
    tet_time = time.time() - t0

    bfaces = boundary_faces_from_tets(Tt)
    bnd_edges, nonman_edges = edge_stats(bfaces)
    surface = trimesh.Trimesh(vertices=Vt, faces=bfaces, process=False)

    mesh = meshio.Mesh(
        points=Vt,
        cells=[('triangle', bfaces.astype(np.int32)), ('tetra', Tt.astype(np.int32))],
    )
    meshio.write(out_msh, mesh, file_format='gmsh22')

    tris = bfaces
    max_preview_faces = 220000
    if len(tris) > max_preview_faces:
        rng = np.random.default_rng(123)
        tris = tris[rng.choice(len(tris), size=max_preview_faces, replace=False)]

    centroids = Vt[Tt].mean(axis=1)
    max_pts = 10000
    if len(centroids) > max_pts:
        rng = np.random.default_rng(123)
        centroids = centroids[rng.choice(len(centroids), size=max_pts, replace=False)]

    fig = go.Figure()
    fig.add_trace(
        go.Mesh3d(
            x=Vt[:, 0], y=Vt[:, 1], z=Vt[:, 2],
            i=tris[:, 0], j=tris[:, 1], k=tris[:, 2],
            color='lightblue', opacity=0.82, name='Watertight boundary',
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=centroids[:, 0], y=centroids[:, 1], z=centroids[:, 2],
            mode='markers', marker=dict(size=1.5, color='crimson', opacity=0.35),
            name='Sample tetra centroids',
        )
    )
    fig.update_layout(
        title=f'tetwild_input_smoothed.msh | nodes={len(Vt):,} tetra={len(Tt):,} boundary_tri={len(bfaces):,}',
        scene=dict(aspectmode='data'),
        margin=dict(l=0, r=0, b=0, t=45),
    )
    fig.write_html(out_html, include_plotlyjs='cdn')

    report = {
        'input_obj': str(input_obj),
        'output_msh': str(out_msh),
        'output_html': str(out_html),
        'method': 'pytetwild',
        'edge_length_fac': edge_length_fac,
        'optimize': optimize,
        'tetrahedralize_seconds': tet_time,
        'input_vertices': int(len(V)),
        'input_faces': int(len(F)),
        'tet_vertices': int(len(Vt)),
        'tetrahedra': int(len(Tt)),
        'boundary_triangles': int(len(bfaces)),
        'surface_watertight': bool(surface.is_watertight),
        'surface_boundary_edges': bnd_edges,
        'surface_nonmanifold_edges': nonman_edges,
        'surface_components': int(len(surface.split(only_watertight=False))),
        'bbox_min': surface.bounds[0].tolist(),
        'bbox_max': surface.bounds[1].tolist(),
    }
    out_json.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
