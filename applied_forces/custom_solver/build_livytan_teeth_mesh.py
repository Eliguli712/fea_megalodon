#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import json
import time

import meshio
import numpy as np
import plotly.graph_objects as go
import pytetwild
import trimesh


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


def load_mesh(path: Path) -> trimesh.Trimesh:
    m = trimesh.load(path, force="mesh", process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(tuple(m.geometry.values()))
    if not isinstance(m, trimesh.Trimesh):
        raise TypeError(f"Expected trimesh.Trimesh, got {type(m)!r}")
    return m


def clean_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    m = mesh.copy()
    m.remove_infinite_values()
    m.merge_vertices()
    m.update_faces(m.unique_faces())
    m.update_faces(m.nondegenerate_faces())
    m.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(m, multibody=True)
    trimesh.repair.fix_winding(m)
    trimesh.repair.fix_inversion(m, multibody=True)
    trimesh.repair.fill_holes(m)
    m.remove_unreferenced_vertices()
    return m


def write_preview_html(
    vertices: np.ndarray,
    boundary_triangles: np.ndarray,
    tetrahedra: np.ndarray,
    out_html: Path,
) -> None:
    tris = boundary_triangles
    max_preview_faces = 250_000
    if len(tris) > max_preview_faces:
        rng = np.random.default_rng(42)
        tris = tris[rng.choice(len(tris), size=max_preview_faces, replace=False)]

    centroids = vertices[tetrahedra].mean(axis=1)
    max_centroids = 20_000
    if len(centroids) > max_centroids:
        rng = np.random.default_rng(42)
        centroids = centroids[rng.choice(len(centroids), size=max_centroids, replace=False)]

    x_mid = float(np.median(centroids[:, 0]))
    x_span = float(vertices[:, 0].max() - vertices[:, 0].min())
    slab = max(0.01 * x_span, 1e-8)
    mask = np.abs(centroids[:, 0] - x_mid) < slab
    centroids_slice = centroids[mask]
    max_slice = 6_000
    if len(centroids_slice) > max_slice:
        rng = np.random.default_rng(42)
        centroids_slice = centroids_slice[rng.choice(len(centroids_slice), size=max_slice, replace=False)]

    fig = go.Figure()
    fig.add_trace(
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=tris[:, 0],
            j=tris[:, 1],
            k=tris[:, 2],
            color="lightskyblue",
            opacity=0.80,
            name="Watertight boundary",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=centroids[:, 0],
            y=centroids[:, 1],
            z=centroids[:, 2],
            mode="markers",
            marker=dict(size=1.2, color="crimson", opacity=0.15),
            name="Sample tetra centroids",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=centroids_slice[:, 0],
            y=centroids_slice[:, 1],
            z=centroids_slice[:, 2],
            mode="markers",
            marker=dict(size=2.0, color="gold", opacity=0.35),
            name="Interior x-slice centroids",
        )
    )
    fig.update_layout(
        title="livytan_melville_teeth | watertight + tet-ready interior preview",
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, b=0, t=45),
    )
    fig.write_html(out_html, include_plotlyjs="cdn")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build livytan teeth obj/msh/html from PLY.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/custom_solver"),
        help="Directory containing Livyatan_melvillei-000461310.ply",
    )
    parser.add_argument("--edge-length-fac", type=float, default=0.015, help="pytetwild edge_length_fac")
    parser.add_argument("--optimize", dest="optimize", action="store_true", help="Enable pytetwild optimization")
    parser.add_argument("--no-optimize", dest="optimize", action="store_false", help="Disable pytetwild optimization")
    parser.set_defaults(optimize=True)
    args = parser.parse_args()

    root = args.root
    input_ply = root / "Livyatan_melvillei-000461310.ply"
    out_obj = root / "livytan_melville_teeth.obj"
    out_msh = root / "livytan_melville_teeth.msh"
    out_html = root / "livytan_dentis.html"
    out_json = root / "livytan_melville_teeth_report.json"

    mesh_in = load_mesh(input_ply)
    mesh = clean_mesh(mesh_in)

    mesh.export(out_obj)

    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int32)

    t0 = time.time()
    vt, tt = pytetwild.tetrahedralize(v, f, optimize=args.optimize, edge_length_fac=args.edge_length_fac)
    tet_seconds = time.time() - t0

    boundary = boundary_faces_from_tets(tt)
    boundary_surface = trimesh.Trimesh(vertices=vt, faces=boundary, process=False)
    boundary_edges, nonmanifold_edges = edge_stats(boundary)

    msh = meshio.Mesh(
        points=vt,
        cells=[("triangle", boundary.astype(np.int32)), ("tetra", tt.astype(np.int32))],
    )
    meshio.write(out_msh, msh, file_format="gmsh22")

    write_preview_html(vt, boundary, tt, out_html)

    report = {
        "input_ply": str(input_ply),
        "output_obj": str(out_obj),
        "output_msh": str(out_msh),
        "output_html": str(out_html),
        "input_vertices": int(len(mesh_in.vertices)),
        "input_faces": int(len(mesh_in.faces)),
        "input_is_watertight": bool(mesh_in.is_watertight),
        "clean_vertices": int(len(mesh.vertices)),
        "clean_faces": int(len(mesh.faces)),
        "clean_is_watertight": bool(mesh.is_watertight),
        "clean_is_volume": bool(mesh.is_volume),
        "tet_method": "pytetwild",
        "edge_length_fac": float(args.edge_length_fac),
        "optimize": bool(args.optimize),
        "tetrahedralize_seconds": tet_seconds,
        "tet_vertices": int(len(vt)),
        "tetrahedra": int(len(tt)),
        "boundary_triangles": int(len(boundary)),
        "boundary_is_watertight": bool(boundary_surface.is_watertight),
        "boundary_edges": int(boundary_edges),
        "nonmanifold_edges": int(nonmanifold_edges),
        "tet_tight_ready": bool(boundary_surface.is_watertight and boundary_edges == 0 and nonmanifold_edges == 0),
        "bbox_min": boundary_surface.bounds[0].tolist(),
        "bbox_max": boundary_surface.bounds[1].tolist(),
    }

    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
