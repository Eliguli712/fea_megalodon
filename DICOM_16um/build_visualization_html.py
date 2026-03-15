#!/usr/bin/env python3
"""Generate HTML previews for the DICOM surface and a TetWild tetrahedralization."""

from __future__ import annotations

import gc
import json
import time
import argparse
from pathlib import Path

import meshio
import numpy as np
import plotly.graph_objects as go
import pytetwild
import trimesh
from plotly.subplots import make_subplots


ROOT = Path(__file__).resolve().parent
EXPORTS = ROOT / "exports"
FULL_SURFACE_OBJ = EXPORTS / "tooth_surface_uncompressed.obj"
VOXEL_SURFACE_OBJ = EXPORTS / "tooth_surface_comsol_watertight.obj"
OUT_HTML = EXPORTS / "tooth_surface_tetwild_visualization.html"
OUT_TET_MSH = EXPORTS / "tooth_surface_tetwild_tet_vol.msh"
OUT_JSON = EXPORTS / "tooth_surface_tetwild_visualization_report.json"


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type: {type(mesh)}")
    return mesh


def boundary_faces_from_tets(tets: np.ndarray) -> np.ndarray:
    f0 = tets[:, [0, 1, 2]]
    f1 = tets[:, [0, 3, 1]]
    f2 = tets[:, [1, 3, 2]]
    f3 = tets[:, [2, 3, 0]]
    faces = np.vstack((f0, f1, f2, f3))
    sfaces = np.sort(faces, axis=1)
    _, idx, counts = np.unique(sfaces, axis=0, return_index=True, return_counts=True)
    return faces[idx[counts == 1]]


def sample_surface_points_edges(
    mesh: trimesh.Trimesh,
    *,
    max_points: int = 120000,
    max_faces: int = 220000,
    max_edges: int = 260000,
    seed: int = 123,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)

    if len(vertices) > max_points:
        point_idx = rng.choice(len(vertices), size=max_points, replace=False)
        points = vertices[point_idx]
    else:
        points = vertices

    if len(faces) > max_faces:
        face_idx = rng.choice(len(faces), size=max_faces, replace=False)
        sampled_faces = faces[face_idx]
    else:
        sampled_faces = faces

    edges = np.vstack(
        (
            sampled_faces[:, [0, 1]],
            sampled_faces[:, [1, 2]],
            sampled_faces[:, [2, 0]],
        )
    )
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    if len(edges) > max_edges:
        edge_idx = rng.choice(len(edges), size=max_edges, replace=False)
        edges = edges[edge_idx]

    return points, edges


def edges_to_lines(vertices: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xyz = vertices[np.asarray(edges, dtype=np.int64)]
    x = np.column_stack((xyz[:, 0, 0], xyz[:, 1, 0], np.full(len(edges), np.nan))).ravel()
    y = np.column_stack((xyz[:, 0, 1], xyz[:, 1, 1], np.full(len(edges), np.nan))).ravel()
    z = np.column_stack((xyz[:, 0, 2], xyz[:, 1, 2], np.full(len(edges), np.nan))).ravel()
    return x, y, z


def edge_stats(triangles: np.ndarray) -> tuple[int, int]:
    edges = np.vstack((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]))
    edges = np.sort(edges, axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return int(np.sum(counts == 1)), int(np.sum(counts > 2))


def sample_tet_edges(
    tets: np.ndarray,
    *,
    max_edges: int = 160000,
    seed: int = 123,
) -> np.ndarray:
    tet_edges = np.vstack(
        (
            tets[:, [0, 1]],
            tets[:, [0, 2]],
            tets[:, [0, 3]],
            tets[:, [1, 2]],
            tets[:, [1, 3]],
            tets[:, [2, 3]],
        )
    )
    tet_edges = np.sort(tet_edges, axis=1)
    tet_edges = np.unique(tet_edges, axis=0)
    if len(tet_edges) > max_edges:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(tet_edges), size=max_edges, replace=False)
        tet_edges = tet_edges[idx]
    return tet_edges


def build_figure(
    surface_points: np.ndarray,
    surface_vertices: np.ndarray,
    surface_edges: np.ndarray,
    enclosure_vertices: np.ndarray,
    enclosure_faces: np.ndarray,
    tet_vertices: np.ndarray,
    tet_centroids: np.ndarray,
    tet_edge_index: np.ndarray,
    *,
    title: str,
) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("Full Watertight Surface Samples", "Watertight Enclosure + TetWild Volume"),
        horizontal_spacing=0.03,
    )

    sx, sy, sz = edges_to_lines(surface_vertices, surface_edges)
    fig.add_trace(
        go.Scatter3d(
            x=surface_points[:, 0],
            y=surface_points[:, 1],
            z=surface_points[:, 2],
            mode="markers",
            marker=dict(size=1.6, color="#ff7f0e", opacity=0.75),
            name="Surface points",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=sx,
            y=sy,
            z=sz,
            mode="lines",
            line=dict(color="#1f77b4", width=2),
            name="Surface edges",
        ),
        row=1,
        col=1,
    )

    tx, ty, tz = edges_to_lines(tet_vertices, tet_edge_index)
    fig.add_trace(
        go.Mesh3d(
            x=enclosure_vertices[:, 0],
            y=enclosure_vertices[:, 1],
            z=enclosure_vertices[:, 2],
            i=enclosure_faces[:, 0],
            j=enclosure_faces[:, 1],
            k=enclosure_faces[:, 2],
            color="#8ecae6",
            opacity=0.22,
            name="Watertight enclosure",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter3d(
            x=tx,
            y=ty,
            z=tz,
            mode="lines",
            line=dict(color="#023047", width=1),
            name="TetWild edges",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter3d(
            x=tet_centroids[:, 0],
            y=tet_centroids[:, 1],
            z=tet_centroids[:, 2],
            mode="markers",
            marker=dict(size=1.5, color="#d62828", opacity=0.35),
            name="Sample tetra centroids",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title=title,
        margin=dict(l=0, r=0, b=0, t=55),
        scene=dict(aspectmode="data", xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)"),
        scene2=dict(aspectmode="data", xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)"),
        legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="left", x=0.01),
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DICOM surface + TetWild HTML preview")
    parser.add_argument("--edge-length-fac", type=float, default=0.10)
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--full-surface-obj", type=Path, default=FULL_SURFACE_OBJ)
    parser.add_argument("--enclosure-obj", type=Path, default=VOXEL_SURFACE_OBJ)
    parser.add_argument("--surface-points", type=int, default=120000)
    parser.add_argument("--surface-faces", type=int, default=220000)
    parser.add_argument("--surface-edges", type=int, default=260000)
    parser.add_argument("--tet-edges", type=int, default=160000)
    parser.add_argument("--tet-centroids", type=int, default=30000)
    args = parser.parse_args()

    full_surface_obj = args.full_surface_obj.resolve()
    enclosure_obj = args.enclosure_obj.resolve()
    if not full_surface_obj.exists():
        fallback = (EXPORTS / "tooth_surface_watertight.obj").resolve()
        if fallback.exists():
            full_surface_obj = fallback
        else:
            raise FileNotFoundError(full_surface_obj)
    if not enclosure_obj.exists():
        raise FileNotFoundError(enclosure_obj)

    full_surface = load_mesh(full_surface_obj)
    surface_points, surface_edges = sample_surface_points_edges(
        full_surface,
        max_points=int(args.surface_points),
        max_faces=int(args.surface_faces),
        max_edges=int(args.surface_edges),
    )
    surface_vertices = np.asarray(full_surface.vertices, dtype=np.float64)
    full_report = {
        "path": str(full_surface_obj),
        "vertices": int(len(full_surface.vertices)),
        "faces": int(len(full_surface.faces)),
        "watertight": bool(full_surface.is_watertight),
        "winding_consistent": bool(full_surface.is_winding_consistent),
        "sampled_points": int(len(surface_points)),
        "sampled_edges": int(len(surface_edges)),
    }

    del full_surface
    gc.collect()

    voxel_surface = load_mesh(enclosure_obj)
    V = np.asarray(voxel_surface.vertices, dtype=np.float64)
    F = np.asarray(voxel_surface.faces, dtype=np.int32)
    enclosure_report = {
        "path": str(enclosure_obj),
        "vertices": int(len(voxel_surface.vertices)),
        "faces": int(len(voxel_surface.faces)),
        "watertight": bool(voxel_surface.is_watertight),
        "winding_consistent": bool(voxel_surface.is_winding_consistent),
    }

    tet_start = time.time()
    V_tet, T_tet = pytetwild.tetrahedralize(
        V,
        F,
        optimize=bool(args.optimize),
        edge_length_fac=float(args.edge_length_fac),
    )
    tet_seconds = time.time() - tet_start

    boundary_faces = boundary_faces_from_tets(T_tet.astype(np.int32))
    boundary_mesh = trimesh.Trimesh(vertices=V_tet, faces=boundary_faces, process=False)
    tet_preview_edges = sample_tet_edges(T_tet.astype(np.int32), max_edges=int(args.tet_edges), seed=123)

    rng = np.random.default_rng(123)
    tet_centroids = V_tet[T_tet].mean(axis=1)
    max_preview_centroids = int(args.tet_centroids)
    if len(tet_centroids) > max_preview_centroids:
        idx = rng.choice(len(tet_centroids), size=max_preview_centroids, replace=False)
        tet_centroids = tet_centroids[idx]

    meshio.write(
        OUT_TET_MSH,
        meshio.Mesh(
            points=np.asarray(V_tet, dtype=np.float64),
            cells=[("triangle", boundary_faces.astype(np.int32)), ("tetra", T_tet.astype(np.int32))],
        ),
        file_format="gmsh22",
    )

    boundary_bnd_edges, boundary_nonman = edge_stats(boundary_faces.astype(np.int32))
    tet_report = {
        "input_surface": str(enclosure_obj),
        "tetwild_edge_length_fac": float(args.edge_length_fac),
        "tetwild_optimize": bool(args.optimize),
        "tetrahedralize_seconds": float(tet_seconds),
        "surface_vertices": int(len(V)),
        "surface_faces": int(len(F)),
        "tet_vertices": int(len(V_tet)),
        "tetrahedra": int(len(T_tet)),
        "boundary_triangles": int(len(boundary_faces)),
        "boundary_watertight": bool(boundary_mesh.is_watertight),
        "boundary_winding_consistent": bool(boundary_mesh.is_winding_consistent),
        "boundary_edges": int(boundary_bnd_edges),
        "boundary_nonmanifold_edges": int(boundary_nonman),
        "display_enclosure_watertight": bool(enclosure_report["watertight"]),
        "display_enclosure_faces": int(len(F)),
        "preview_edges": int(len(tet_preview_edges)),
        "preview_centroids": int(len(tet_centroids)),
        "tetwild_msh": str(OUT_TET_MSH),
    }

    title = (
        f"DICOM surface + TetWild preview | "
        f"surface V={full_report['vertices']:,} F={full_report['faces']:,} | "
        f"tet V={tet_report['tet_vertices']:,} F={tet_report['boundary_triangles']:,} T={tet_report['tetrahedra']:,}"
    )
    fig = build_figure(
        surface_points,
        surface_vertices,
        surface_edges,
        np.asarray(V, dtype=np.float64),
        np.asarray(F, dtype=np.int32),
        np.asarray(V_tet, dtype=np.float64),
        np.asarray(tet_centroids, dtype=np.float64),
        np.asarray(tet_preview_edges, dtype=np.int32),
        title=title,
    )
    fig.write_html(OUT_HTML, include_plotlyjs=True)

    report = {
        "full_surface": full_report,
        "display_enclosure": enclosure_report,
        "tetwild": tet_report,
        "html": str(OUT_HTML),
    }
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
