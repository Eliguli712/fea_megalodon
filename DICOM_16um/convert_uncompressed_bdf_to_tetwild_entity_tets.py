#!/usr/bin/env python3
"""Convert a surface-only Nastran BDF into TetWild tetra entity grids.

This pipeline now treats the TetWild boundary extraction as an intermediate
artifact, not the final surface. It writes both:
- the raw boundary recovered directly from the tetrahedra
- a tightened, watertight boundary rebuilt from the dominant raw components

Outputs are written as plain, uncompressed mesh artifacts:
- Gmsh v2.2 volume mesh (.msh)
- Nastran volume mesh (.bdf)
- Explicit grid/tet/boundary-triangle CSV tables
- Raw and tightened boundary OBJ/STL/CSV exports
- Raw-vs-tightened HTML + PNG comparison
- JSON report with counts, topology, and timing
"""

from __future__ import annotations

import argparse
import csv
import gc
import html
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import meshio
import numpy as np
import plotly.graph_objects as go
import pytetwild
import trimesh
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from plotly.subplots import make_subplots


ROOT = Path(__file__).resolve().parent
EXPORTS = ROOT / "exports"


@dataclass
class SourceCounts:
    grids: int
    triangles: int
    tetrahedra: int
    contiguous_grid_ids: bool


@dataclass
class TopologyStats:
    vertices: int
    faces: int
    edges: int
    watertight: bool
    winding_consistent: bool
    components: int
    boundary_edges: int
    nonmanifold_edges: int
    euler_number: int
    area: float


def parse_bdf_float(field: str) -> float:
    s = field.strip()
    if not s:
        return 0.0
    if "E" in s or "e" in s:
        return float(s)
    for idx in range(1, len(s)):
        c = s[idx]
        if c in "+-" and s[idx - 1].isdigit():
            return float(s[:idx] + "E" + s[idx:])
    return float(s)


def cleanup_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    out = mesh.copy()
    for fn_name in ("remove_infinite_values", "remove_unreferenced_vertices", "merge_vertices"):
        fn = getattr(out, fn_name, None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass
    try:
        out.update_faces(out.unique_faces())
    except Exception:
        pass
    try:
        out.update_faces(out.nondegenerate_faces())
    except Exception:
        pass
    try:
        out.remove_unreferenced_vertices()
    except Exception:
        pass
    return out


def topology_stats(mesh: trimesh.Trimesh) -> TopologyStats:
    edges = np.sort(mesh.edges.reshape(-1, 2), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return TopologyStats(
        vertices=int(len(mesh.vertices)),
        faces=int(len(mesh.faces)),
        edges=int(len(counts)),
        watertight=bool(mesh.is_watertight),
        winding_consistent=bool(mesh.is_winding_consistent),
        components=int(len(mesh.split(only_watertight=False))),
        boundary_edges=int(np.sum(counts == 1)),
        nonmanifold_edges=int(np.sum(counts > 2)),
        euler_number=int(mesh.euler_number),
        area=float(mesh.area),
    )


def inspect_bdf(path: Path) -> SourceCounts:
    grids = 0
    triangles = 0
    tetrahedra = 0
    contiguous = True
    expected_grid_id = 1
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("GRID"):
                grids += 1
                grid_id = int(line[8:16])
                if contiguous and grid_id != expected_grid_id:
                    contiguous = False
                expected_grid_id += 1
            elif line.startswith("CTRIA3"):
                triangles += 1
            elif line.startswith("CTETRA"):
                tetrahedra += 1
    return SourceCounts(
        grids=grids,
        triangles=triangles,
        tetrahedra=tetrahedra,
        contiguous_grid_ids=contiguous,
    )


def load_surface_from_bdf(path: Path) -> tuple[np.ndarray, np.ndarray, SourceCounts]:
    counts = inspect_bdf(path)
    if counts.grids <= 0 or counts.triangles <= 0:
        raise RuntimeError(f"{path} does not contain a usable GRID/CTRIA3 surface")

    vertices = np.empty((counts.grids, 3), dtype=np.float64)
    faces = np.empty((counts.triangles, 3), dtype=np.int32)

    if counts.contiguous_grid_ids:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if not line.startswith("GRID"):
                    continue
                grid_id = int(line[8:16]) - 1
                vertices[grid_id, 0] = parse_bdf_float(line[24:32])
                vertices[grid_id, 1] = parse_bdf_float(line[32:40])
                vertices[grid_id, 2] = parse_bdf_float(line[40:48])

        tri_idx = 0
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if not line.startswith("CTRIA3"):
                    continue
                faces[tri_idx, 0] = int(line[24:32]) - 1
                faces[tri_idx, 1] = int(line[32:40]) - 1
                faces[tri_idx, 2] = int(line[40:48]) - 1
                tri_idx += 1
    else:
        id_to_index: dict[int, int] = {}
        vertex_idx = 0
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if not line.startswith("GRID"):
                    continue
                grid_id = int(line[8:16])
                id_to_index[grid_id] = vertex_idx
                vertices[vertex_idx, 0] = parse_bdf_float(line[24:32])
                vertices[vertex_idx, 1] = parse_bdf_float(line[32:40])
                vertices[vertex_idx, 2] = parse_bdf_float(line[40:48])
                vertex_idx += 1

        tri_idx = 0
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if not line.startswith("CTRIA3"):
                    continue
                faces[tri_idx, 0] = id_to_index[int(line[24:32])]
                faces[tri_idx, 1] = id_to_index[int(line[32:40])]
                faces[tri_idx, 2] = id_to_index[int(line[40:48])]
                tri_idx += 1

    return vertices, faces, counts


def boundary_faces_from_tets(tets: np.ndarray) -> np.ndarray:
    f0 = tets[:, [0, 1, 2]]
    f1 = tets[:, [0, 3, 1]]
    f2 = tets[:, [1, 3, 2]]
    f3 = tets[:, [2, 3, 0]]
    faces = np.vstack((f0, f1, f2, f3))
    sorted_faces = np.sort(faces, axis=1)
    _, idx, counts = np.unique(sorted_faces, axis=0, return_index=True, return_counts=True)
    return faces[idx[counts == 1]]


def write_grids_csv(path: Path, points: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("grid_id", "x_mm", "y_mm", "z_mm"))
        for grid_id, xyz in enumerate(points, start=1):
            writer.writerow((grid_id, f"{xyz[0]:.17g}", f"{xyz[1]:.17g}", f"{xyz[2]:.17g}"))


def write_tris_csv(path: Path, triangles_zero_based: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("tri_id", "grid_1", "grid_2", "grid_3"))
        for tri_id, tri in enumerate(triangles_zero_based, start=1):
            writer.writerow((tri_id, int(tri[0] + 1), int(tri[1] + 1), int(tri[2] + 1)))


def write_tets_csv(path: Path, tets_zero_based: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("tet_id", "grid_1", "grid_2", "grid_3", "grid_4"))
        for tet_id, tet in enumerate(tets_zero_based, start=1):
            writer.writerow((tet_id, int(tet[0] + 1), int(tet[1] + 1), int(tet[2] + 1), int(tet[3] + 1)))


def keep_area_coverage(mesh: trimesh.Trimesh, min_coverage: float) -> tuple[trimesh.Trimesh, dict[str, float | int]]:
    parts = [part for part in mesh.split(only_watertight=False) if len(part.faces) > 0]
    if len(parts) <= 1:
        return cleanup_mesh(mesh), {
            "input_components": int(len(parts)),
            "kept_components": int(len(parts)),
            "kept_area_fraction": 1.0,
        }

    parts = sorted(parts, key=lambda part: float(part.area), reverse=True)
    total_area = sum(float(part.area) for part in parts)
    kept_parts = []
    kept_area = 0.0
    for part in parts:
        kept_parts.append(part)
        kept_area += float(part.area)
        if total_area <= 0.0 or kept_area / total_area >= min_coverage:
            break

    kept = kept_parts[0] if len(kept_parts) == 1 else trimesh.util.concatenate(kept_parts)
    kept = cleanup_mesh(kept)
    return kept, {
        "input_components": int(len(parts)),
        "kept_components": int(len(kept_parts)),
        "kept_area_fraction": float(0.0 if total_area <= 0.0 else kept_area / total_area),
    }


def tighten_boundary(
    raw_boundary_mesh: trimesh.Trimesh,
    *,
    min_coverage: float,
    voxel_n_axis: int,
) -> tuple[trimesh.Trimesh, dict[str, object]]:
    dominant_mesh, coverage_meta = keep_area_coverage(raw_boundary_mesh, min_coverage=min_coverage)
    dominant_stats = topology_stats(dominant_mesh)
    extent = dominant_mesh.bounds[1] - dominant_mesh.bounds[0]
    pitch = float(extent.max() / float(voxel_n_axis))

    shell_voxels = dominant_mesh.voxelized(pitch)
    shell_voxel_count = int(shell_voxels.filled_count)
    filled_voxels = shell_voxels.fill()
    filled_voxel_count = int(filled_voxels.filled_count)
    tightened = filled_voxels.marching_cubes
    tightened.apply_transform(filled_voxels.transform)
    tightened = cleanup_mesh(tightened)
    try:
        trimesh.repair.fix_winding(tightened)
    except Exception:
        pass
    try:
        trimesh.repair.fix_inversion(tightened)
    except Exception:
        pass
    tightened = cleanup_mesh(tightened)

    meta: dict[str, object] = {
        "area_coverage_target": float(min_coverage),
        "voxel_n_axis": int(voxel_n_axis),
        "voxel_pitch_mm": float(pitch),
        "voxel_grid_shape": [int(x) for x in filled_voxels.shape],
        "shell_voxels": shell_voxel_count,
        "filled_voxels": filled_voxel_count,
        "added_internal_voxels": int(filled_voxel_count - shell_voxel_count),
        "dominant_boundary": asdict(dominant_stats),
    }
    meta.update(coverage_meta)
    return tightened, meta


def sample_faces(faces: np.ndarray, max_faces: int, seed: int) -> np.ndarray:
    if len(faces) <= max_faces:
        return faces
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(faces), size=max_faces, replace=False)
    return np.asarray(faces[idx], dtype=np.int32)


def mesh3d_trace(
    mesh: trimesh.Trimesh,
    *,
    max_faces: int,
    color: str,
    name: str,
    opacity: float,
    seed: int,
) -> go.Mesh3d:
    faces = sample_faces(np.asarray(mesh.faces, dtype=np.int32), max_faces=max_faces, seed=seed)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color=color,
        opacity=opacity,
        name=name,
    )


def format_mesh_counts(stats: TopologyStats) -> str:
    return f"V={stats.vertices:,} F={stats.faces:,} E={stats.edges:,}"


def format_trace_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.parent))
    except Exception:
        return str(path.resolve())


def summary_lines(
    *,
    stats: TopologyStats,
    watertight: bool | None = None,
) -> tuple[str, ...]:
    first = format_mesh_counts(stats)
    second = (
        f"components={stats.components:,} "
        f"boundary_edges={stats.boundary_edges:,} "
        f"nonmanifold_edges={stats.nonmanifold_edges:,}"
    )
    if watertight is None:
        return (first, second)
    third = f"watertight={watertight} winding_consistent={stats.winding_consistent}"
    return (first, second, third)


def build_terminal_trace(
    *,
    input_path: Path,
    source_counts: SourceCounts,
    tet_vertices_count: int,
    tetrahedra_count: int,
    boundary_triangles_count: int,
    raw_stats: TopologyStats,
    tightened_stats: TopologyStats,
    repair_meta: dict[str, object],
) -> str:
    dominant_boundary = repair_meta["dominant_boundary"]
    if not isinstance(dominant_boundary, dict):
        raise RuntimeError("repair_meta['dominant_boundary'] is missing")

    kept_components = int(repair_meta["kept_components"])
    input_components = int(repair_meta["input_components"])
    kept_area_fraction = float(repair_meta["kept_area_fraction"])
    voxel_n_axis = int(repair_meta["voxel_n_axis"])
    voxel_pitch_mm = float(repair_meta["voxel_pitch_mm"])
    voxel_grid_shape = repair_meta["voxel_grid_shape"]
    shell_voxels = int(repair_meta["shell_voxels"])
    filled_voxels = int(repair_meta["filled_voxels"])
    added_internal_voxels = int(repair_meta["added_internal_voxels"])

    return "\n".join(
        (
            f"$ ~/.venv313/bin/python {format_trace_path(Path(__file__))}",
            f"[load] input_bdf={format_trace_path(input_path)}",
            (
                "       source_surface "
                f"V={source_counts.grids:,} F={source_counts.triangles:,} "
                f"contiguous_grid_ids={source_counts.contiguous_grid_ids}"
            ),
            (
                "[tetwild] entity_volume "
                f"V={tet_vertices_count:,} T={tetrahedra_count:,} "
                f"boundary_F={boundary_triangles_count:,}"
            ),
            (
                "[recover] raw_boundary "
                f"{format_mesh_counts(raw_stats)} "
                f"components={raw_stats.components:,} "
                f"boundary_edges={raw_stats.boundary_edges:,} "
                f"nonmanifold_edges={raw_stats.nonmanifold_edges:,}"
            ),
            (
                "[restore] dominant_boundary "
                f"V={int(dominant_boundary['vertices']):,} "
                f"F={int(dominant_boundary['faces']):,} "
                f"E={int(dominant_boundary['edges']):,} "
                f"kept_components={kept_components:,}/{input_components:,} "
                f"kept_area_fraction={kept_area_fraction:.15f}"
            ),
            (
                "[voxelize] shell "
                f"voxel_n_axis={voxel_n_axis:,} "
                f"voxel_pitch_mm={voxel_pitch_mm:.16f} "
                f"grid_shape={tuple(int(x) for x in voxel_grid_shape)} "
                f"shell_voxels={shell_voxels:,}"
            ),
            (
                "[fill] internal_voxels "
                f"filled_voxels={filled_voxels:,} "
                f"added_internal_voxels={added_internal_voxels:,}"
            ),
            (
                "[reconstruct] tightened_boundary "
                f"{format_mesh_counts(tightened_stats)} "
                f"components={tightened_stats.components:,} "
                f"boundary_edges={tightened_stats.boundary_edges:,} "
                f"nonmanifold_edges={tightened_stats.nonmanifold_edges:,} "
                f"watertight={tightened_stats.watertight}"
            ),
            (
                "[display] raw_vs_tightened "
                "V/F/E counts are exact full-mesh totals; only the interactive "
                "rendering triangles are sampled for browser responsiveness"
            ),
        )
    )


def write_compare_html(
    path: Path,
    *,
    raw_mesh: trimesh.Trimesh,
    tightened_mesh: trimesh.Trimesh,
    raw_stats: TopologyStats,
    tightened_stats: TopologyStats,
    terminal_trace: str,
) -> None:
    raw_summary_lines = summary_lines(stats=raw_stats)
    tightened_summary_lines = summary_lines(stats=tightened_stats, watertight=tightened_stats.watertight)
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=(
            "Raw TetWild boundary",
            "Tightened boundary",
        ),
        horizontal_spacing=0.03,
    )
    fig.add_trace(
        mesh3d_trace(
            raw_mesh,
            max_faces=90000,
            color="#d62828",
            name="Raw boundary",
            opacity=0.92,
            seed=123,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        mesh3d_trace(
            tightened_mesh,
            max_faces=90000,
            color="#1d3557",
            name="Tightened boundary",
            opacity=0.92,
            seed=456,
        ),
        row=1,
        col=2,
    )
    fig.update_annotations(font=dict(size=15))
    fig.update_layout(
        height=840,
        margin=dict(l=0, r=0, b=0, t=44),
        showlegend=False,
        scene=dict(aspectmode="data", xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)"),
        scene2=dict(aspectmode="data", xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)"),
    )
    figure_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
    document = f"""<html>
<head>
  <meta charset="utf-8" />
  <title>Tooth Surface Uncompressed TetWild Boundary Compare</title>
  <style>
    :root {{
      color-scheme: light;
      --page-bg: #f4f1ea;
      --panel-bg: #ffffff;
      --panel-border: #d8d1c5;
      --terminal-bg: #0d1117;
      --terminal-fg: #e6edf3;
      --terminal-accent: #f4a261;
      --card-bg: #fbfaf6;
      --card-accent: #1f2937;
      --shadow: 0 18px 48px rgba(32, 28, 22, 0.14);
    }}
    body {{
      margin: 0;
      padding: 24px;
      background: linear-gradient(180deg, #efe9dc 0%, var(--page-bg) 100%);
      font-family: "Iosevka Term", "SFMono-Regular", "Menlo", monospace;
      color: #1f2933;
    }}
    .panel {{
      max-width: 1680px;
      margin: 0 auto;
      background: var(--panel-bg);
      border: 1px solid var(--panel-border);
      border-radius: 18px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }}
    .figure {{
      padding: 16px 16px 0 16px;
    }}
    .summary {{
      padding: 24px 24px 0 24px;
    }}
    .summary h1 {{
      margin: 0;
      font-size: 28px;
      line-height: 1.18;
      letter-spacing: 0.01em;
    }}
    .summary p {{
      margin: 10px 0 0 0;
      color: #4b5563;
      font-size: 15px;
      line-height: 1.65;
      max-width: 980px;
    }}
    .stat-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
      padding: 18px 24px 8px 24px;
    }}
    .stat-card {{
      background: var(--card-bg);
      border: 1px solid var(--panel-border);
      border-radius: 16px;
      padding: 18px 20px;
    }}
    .stat-card h2 {{
      margin: 0 0 12px 0;
      font-size: 16px;
      color: var(--card-accent);
      letter-spacing: 0.01em;
    }}
    .stat-card p {{
      margin: 8px 0 0 0;
      font-size: 14px;
      line-height: 1.7;
      color: #374151;
    }}
    .terminal {{
      margin: 16px;
      border-radius: 14px;
      overflow: hidden;
      border: 1px solid #1f2937;
      background: var(--terminal-bg);
    }}
    .terminal-bar {{
      padding: 10px 14px;
      background: #111827;
      color: var(--terminal-accent);
      font-size: 14px;
      letter-spacing: 0.03em;
      text-transform: uppercase;
    }}
    .terminal pre {{
      margin: 0;
      padding: 20px 22px 22px 22px;
      color: var(--terminal-fg);
      font-size: 13px;
      line-height: 1.85;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }}
    @media (max-width: 960px) {{
      .stat-grid {{
        grid-template-columns: 1fr;
      }}
      .summary h1 {{
        font-size: 24px;
      }}
    }}
  </style>
</head>
<body>
  <div class="panel">
    <header class="summary">
      <h1>Tooth Surface Uncompressed TetWild Recovery</h1>
      <p>Raw vs. tightened boundary reconstruction with restored internal voxels. Exact full-mesh counts are shown below and in the terminal trace.</p>
    </header>
    <section class="stat-grid">
      <article class="stat-card">
        <h2>Raw TetWild Boundary</h2>
        <p>{html.escape(raw_summary_lines[0])}</p>
        <p>{html.escape(raw_summary_lines[1])}</p>
      </article>
      <article class="stat-card">
        <h2>Tightened Boundary</h2>
        <p>{html.escape(tightened_summary_lines[0])}</p>
        <p>{html.escape(tightened_summary_lines[1])}</p>
        <p>{html.escape(tightened_summary_lines[2])}</p>
      </article>
    </section>
    <div class="figure">{figure_html}</div>
    <section class="terminal">
      <div class="terminal-bar">Voxel Fill / Restore / Reconstruct Trace</div>
      <pre>{html.escape(terminal_trace)}</pre>
    </section>
  </div>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def add_mesh_subplot(
    ax,
    *,
    mesh: trimesh.Trimesh,
    title: str,
    color: str,
    max_faces: int,
    seed: int,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
) -> None:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = sample_faces(np.asarray(mesh.faces, dtype=np.int32), max_faces=max_faces, seed=seed)
    poly = Poly3DCollection(vertices[faces], linewidths=0.05, alpha=0.95)
    poly.set_facecolor(color)
    poly.set_edgecolor((0.05, 0.05, 0.05, 0.12))
    ax.add_collection3d(poly)

    center = 0.5 * (bounds_min + bounds_max)
    radius = 0.55 * float(np.max(bounds_max - bounds_min))
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    add_background_scale_grid(ax, bounds_min=bounds_min, bounds_max=bounds_max)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=7, azim=180)
    ax.set_proj_type("ortho")
    ax.set_xticks(())
    ax.grid(False)
    ax.set_xlabel("")
    ax.set_ylabel("mm", fontsize=8, color="#475569", labelpad=3)
    ax.set_zlabel("mm", fontsize=8, color="#475569", labelpad=6)
    ax.set_title(title, fontsize=11)


def rounded_ticks(min_value: float, max_value: float, step: float) -> np.ndarray:
    start = step * np.floor(min_value / step)
    end = step * np.ceil(max_value / step)
    return np.arange(start, end + 0.5 * step, step, dtype=np.float64)


def add_background_scale_grid(
    ax,
    *,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
) -> None:
    span = bounds_max - bounds_min
    plane_x = float(bounds_min[0] - 0.035 * max(span[0], 1.0))
    y_minor = rounded_ticks(float(bounds_min[1]), float(bounds_max[1]), 5.0)
    z_minor = rounded_ticks(float(bounds_min[2]), float(bounds_max[2]), 5.0)
    y0 = float(y_minor[0])
    y1 = float(y_minor[-1])
    z0 = float(z_minor[0])
    z1 = float(z_minor[-1])

    for y in y_minor:
        is_major = abs(y / 10.0 - round(y / 10.0)) < 1e-9
        ax.plot(
            (plane_x, plane_x),
            (float(y), float(y)),
            (z0, z1),
            color="#cbd5e1" if not is_major else "#94a3b8",
            linewidth=0.55 if not is_major else 0.9,
            alpha=0.65 if not is_major else 0.82,
            zorder=0,
        )
    for z in z_minor:
        is_major = abs(z / 10.0 - round(z / 10.0)) < 1e-9
        ax.plot(
            (plane_x, plane_x),
            (y0, y1),
            (float(z), float(z)),
            color="#cbd5e1" if not is_major else "#94a3b8",
            linewidth=0.55 if not is_major else 0.9,
            alpha=0.65 if not is_major else 0.82,
            zorder=0,
        )

    ax.plot((plane_x, plane_x), (y0, y1), (z0, z0), color="#64748b", linewidth=1.05, alpha=0.9, zorder=0)
    ax.plot((plane_x, plane_x), (y0, y1), (z1, z1), color="#64748b", linewidth=1.05, alpha=0.9, zorder=0)
    ax.plot((plane_x, plane_x), (y0, y0), (z0, z1), color="#64748b", linewidth=1.05, alpha=0.9, zorder=0)
    ax.plot((plane_x, plane_x), (y1, y1), (z0, z1), color="#64748b", linewidth=1.05, alpha=0.9, zorder=0)

    ax.set_yticks(rounded_ticks(float(bounds_min[1]), float(bounds_max[1]), 10.0))
    ax.set_zticks(rounded_ticks(float(bounds_min[2]), float(bounds_max[2]), 10.0))
    ax.tick_params(axis="y", labelsize=8, pad=1, colors="#475569")
    ax.tick_params(axis="z", labelsize=8, pad=1, colors="#475569")


def crown_is_near_max_z(mesh: trimesh.Trimesh) -> bool:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    z = vertices[:, 2]
    z_min = float(z.min())
    z_max = float(z.max())
    span = z_max - z_min
    if span <= 0.0:
        return True
    band = 0.12 * span
    low = vertices[z <= z_min + band][:, :2]
    high = vertices[z >= z_max - band][:, :2]
    if len(low) == 0 or len(high) == 0:
        return True
    low_extent = np.prod(np.ptp(low, axis=0))
    high_extent = np.prod(np.ptp(high, axis=0))
    return bool(high_extent >= low_extent)


def display_transform(reference_mesh: trimesh.Trimesh) -> np.ndarray:
    oriented = reference_mesh.copy()
    transform = np.asarray(oriented.principal_inertia_transform, dtype=np.float64)
    oriented.apply_transform(transform)

    if not crown_is_near_max_z(oriented):
        flip_z = np.eye(4, dtype=np.float64)
        flip_z[2, 2] = -1.0
        transform = flip_z @ transform
        oriented.apply_transform(flip_z)

    # Candidate inspection showed the principal-frame azim=180 view gives the
    # buccal-facing frontal silhouette instead of the thin proximal profile.
    flip_x = np.eye(4, dtype=np.float64)
    flip_x[0, 0] = -1.0
    transform = flip_x @ transform
    return transform


def write_compare_png(
    path: Path,
    *,
    raw_mesh: trimesh.Trimesh,
    tightened_mesh: trimesh.Trimesh,
    raw_stats: TopologyStats,
    tightened_stats: TopologyStats,
    terminal_trace: str,
) -> None:
    raw_summary_lines = summary_lines(stats=raw_stats)
    tightened_summary_lines = summary_lines(stats=tightened_stats, watertight=tightened_stats.watertight)
    plot_transform = display_transform(tightened_mesh)
    raw_plot_mesh = raw_mesh.copy()
    raw_plot_mesh.apply_transform(plot_transform)
    tightened_plot_mesh = tightened_mesh.copy()
    tightened_plot_mesh.apply_transform(plot_transform)

    raw_bounds = np.asarray(raw_plot_mesh.bounds, dtype=np.float64)
    tight_bounds = np.asarray(tightened_plot_mesh.bounds, dtype=np.float64)
    bounds_min = np.minimum(raw_bounds[0], tight_bounds[0])
    bounds_max = np.maximum(raw_bounds[1], tight_bounds[1])

    fig = plt.figure(figsize=(16, 11.4), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, height_ratios=(0.68, 3.0, 1.48))
    ax_header = fig.add_subplot(grid[0, :])
    ax1 = fig.add_subplot(grid[1, 0], projection="3d")
    ax2 = fig.add_subplot(grid[1, 1], projection="3d")
    ax3 = fig.add_subplot(grid[2, :])

    ax_header.set_axis_off()
    ax_header.text(
        0.5,
        0.95,
        "Tooth Surface Uncompressed TetWild Recovery",
        transform=ax_header.transAxes,
        va="top",
        ha="center",
        fontsize=18,
        color="#111827",
        family="sans-serif",
        fontweight="bold",
    )
    ax_header.text(
        0.02,
        0.62,
        "Raw TetWild boundary",
        transform=ax_header.transAxes,
        va="top",
        ha="left",
        fontsize=12.5,
        color="#9f3a20",
        family="sans-serif",
        fontweight="bold",
    )
    ax_header.text(
        0.02,
        0.40,
        "\n".join(raw_summary_lines),
        transform=ax_header.transAxes,
        va="top",
        ha="left",
        fontsize=10.4,
        color="#374151",
        family="monospace",
        linespacing=1.6,
    )
    ax_header.text(
        0.98,
        0.62,
        "Tightened boundary",
        transform=ax_header.transAxes,
        va="top",
        ha="right",
        fontsize=12.5,
        color="#214e6b",
        family="sans-serif",
        fontweight="bold",
    )
    ax_header.text(
        0.98,
        0.40,
        "\n".join(tightened_summary_lines),
        transform=ax_header.transAxes,
        va="top",
        ha="right",
        fontsize=10.4,
        color="#374151",
        family="monospace",
        linespacing=1.6,
    )

    add_mesh_subplot(
        ax1,
        mesh=raw_plot_mesh,
        title="Raw TetWild boundary",
        color="#e76f51",
        max_faces=30000,
        seed=123,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
    )
    add_mesh_subplot(
        ax2,
        mesh=tightened_plot_mesh,
        title="Tightened boundary",
        color="#457b9d",
        max_faces=30000,
        seed=456,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
    )

    ax3.set_axis_off()
    ax3.set_facecolor("#0d1117")
    ax3.add_patch(
        plt.Rectangle(
            (0.0, 0.0),
            1.0,
            1.0,
            transform=ax3.transAxes,
            facecolor="#0d1117",
            edgecolor="#1f2937",
            linewidth=1.2,
            clip_on=False,
            zorder=0,
        )
    )
    ax3.text(
        0.015,
        0.96,
        "VOXEL FILL / RESTORE / RECONSTRUCT TRACE",
        transform=ax3.transAxes,
        va="top",
        ha="left",
        fontsize=11.5,
        color="#f4a261",
        family="monospace",
        fontweight="bold",
    )
    ax3.text(
        0.015,
        0.86,
        terminal_trace,
        transform=ax3.transAxes,
        va="top",
        ha="left",
        fontsize=9.4,
        color="#e6edf3",
        family="monospace",
        linespacing=1.72,
    )
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="BDF surface -> TetWild tetra entity grids")
    parser.add_argument(
        "--input",
        type=Path,
        default=EXPORTS / "tooth_surface_uncompressed.bdf",
        help="Surface-only Nastran BDF with GRID + CTRIA3 cards",
    )
    parser.add_argument("--output-dir", type=Path, default=EXPORTS)
    parser.add_argument("--prefix", default="tooth_surface_uncompressed_tetwild_entity")
    parser.add_argument(
        "--edge-length-fac",
        type=float,
        default=0.10,
        help="TetWild target edge length factor relative to model scale",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Enable TetWild post-optimization (off by default for speed)",
    )
    parser.add_argument(
        "--repair-area-coverage",
        type=float,
        default=0.999,
        help="Keep largest raw boundary components until this area fraction is covered before tightening",
    )
    parser.add_argument(
        "--repair-voxel-n-axis",
        type=int,
        default=220,
        help="Voxel resolution along the longest axis for the tightened boundary recovery",
    )
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.prefix
    out_msh = output_dir / f"{prefix}_tet_vol.msh"
    out_bdf = output_dir / f"{prefix}_tet_vol.bdf"
    out_grids_csv = output_dir / f"{prefix}_grids.csv"
    out_tets_csv = output_dir / f"{prefix}_tets.csv"
    out_raw_boundary_csv = output_dir / f"{prefix}_boundary_tris.csv"
    out_raw_boundary_obj = output_dir / f"{prefix}_raw_boundary.obj"
    out_raw_boundary_stl = output_dir / f"{prefix}_raw_boundary.stl"
    out_tight_grids_csv = output_dir / f"{prefix}_tightened_boundary_grids.csv"
    out_tight_tris_csv = output_dir / f"{prefix}_tightened_boundary_tris.csv"
    out_tight_obj = output_dir / f"{prefix}_tightened_boundary.obj"
    out_tight_stl = output_dir / f"{prefix}_tightened_boundary.stl"
    out_compare_html = output_dir / f"{prefix}_boundary_compare.html"
    out_compare_png = output_dir / f"{prefix}_boundary_compare.png"
    out_report = output_dir / f"{prefix}_report.json"

    parse_start = time.time()
    surface_vertices, surface_faces, source_counts = load_surface_from_bdf(input_path)
    parse_seconds = time.time() - parse_start

    bbox_min = surface_vertices.min(axis=0)
    bbox_max = surface_vertices.max(axis=0)

    tet_start = time.time()
    tet_vertices, tetrahedra = pytetwild.tetrahedralize(
        np.ascontiguousarray(surface_vertices, dtype=np.float64),
        np.ascontiguousarray(surface_faces, dtype=np.int32),
        optimize=bool(args.optimize),
        edge_length_fac=float(args.edge_length_fac),
    )
    tet_seconds = time.time() - tet_start

    del surface_vertices
    del surface_faces
    gc.collect()

    tet_vertices = np.ascontiguousarray(tet_vertices, dtype=np.float64)
    tetrahedra = np.ascontiguousarray(tetrahedra, dtype=np.int32)
    boundary_tris = np.ascontiguousarray(boundary_faces_from_tets(tetrahedra), dtype=np.int32)

    tet_mesh = meshio.Mesh(
        points=tet_vertices,
        cells=[("triangle", boundary_tris), ("tetra", tetrahedra)],
    )
    meshio.write(out_msh, tet_mesh, file_format="gmsh22")
    meshio.write(out_bdf, tet_mesh, file_format="nastran")

    raw_boundary_mesh = cleanup_mesh(
        trimesh.Trimesh(vertices=tet_vertices, faces=boundary_tris, process=False)
    )
    raw_boundary_stats = topology_stats(raw_boundary_mesh)
    raw_boundary_mesh.export(out_raw_boundary_obj)
    raw_boundary_mesh.export(out_raw_boundary_stl)

    repair_start = time.time()
    tightened_boundary_mesh, repair_meta = tighten_boundary(
        raw_boundary_mesh,
        min_coverage=float(args.repair_area_coverage),
        voxel_n_axis=int(args.repair_voxel_n_axis),
    )
    repair_seconds = time.time() - repair_start
    tightened_boundary_stats = topology_stats(tightened_boundary_mesh)
    terminal_trace = build_terminal_trace(
        input_path=input_path,
        source_counts=source_counts,
        tet_vertices_count=int(len(tet_vertices)),
        tetrahedra_count=int(len(tetrahedra)),
        boundary_triangles_count=int(len(boundary_tris)),
        raw_stats=raw_boundary_stats,
        tightened_stats=tightened_boundary_stats,
        repair_meta=repair_meta,
    )

    tightened_vertices = np.asarray(tightened_boundary_mesh.vertices, dtype=np.float64)
    tightened_tris = np.asarray(tightened_boundary_mesh.faces, dtype=np.int32)
    tightened_boundary_mesh.export(out_tight_obj)
    tightened_boundary_mesh.export(out_tight_stl)

    csv_start = time.time()
    write_grids_csv(out_grids_csv, tet_vertices)
    write_tets_csv(out_tets_csv, tetrahedra)
    write_tris_csv(out_raw_boundary_csv, boundary_tris)
    write_grids_csv(out_tight_grids_csv, tightened_vertices)
    write_tris_csv(out_tight_tris_csv, tightened_tris)
    csv_seconds = time.time() - csv_start

    viz_start = time.time()
    write_compare_html(
        out_compare_html,
        raw_mesh=raw_boundary_mesh,
        tightened_mesh=tightened_boundary_mesh,
        raw_stats=raw_boundary_stats,
        tightened_stats=tightened_boundary_stats,
        terminal_trace=terminal_trace,
    )
    write_compare_png(
        out_compare_png,
        raw_mesh=raw_boundary_mesh,
        tightened_mesh=tightened_boundary_mesh,
        raw_stats=raw_boundary_stats,
        tightened_stats=tightened_boundary_stats,
        terminal_trace=terminal_trace,
    )
    viz_seconds = time.time() - viz_start

    report = {
        "input_bdf": str(input_path),
        "conversion": {
            "method": "pytetwild",
            "edge_length_fac": float(args.edge_length_fac),
            "optimize": bool(args.optimize),
            "parse_seconds": float(parse_seconds),
            "tetrahedralize_seconds": float(tet_seconds),
            "boundary_repair_seconds": float(repair_seconds),
            "csv_write_seconds": float(csv_seconds),
            "visualization_seconds": float(viz_seconds),
        },
        "source_surface": {
            **asdict(source_counts),
            "bbox_min_xyz_mm": [float(x) for x in bbox_min],
            "bbox_max_xyz_mm": [float(x) for x in bbox_max],
        },
        "tetwild_volume": {
            "tet_vertices": int(len(tet_vertices)),
            "tetrahedra": int(len(tetrahedra)),
            "boundary_triangles": int(len(boundary_tris)),
        },
        "boundary_recovery": {
            "raw_boundary": asdict(raw_boundary_stats),
            "dominant_boundary": repair_meta["dominant_boundary"],
            "tightened_boundary": asdict(tightened_boundary_stats),
            "repair": repair_meta,
            "terminal_trace": terminal_trace,
        },
        "outputs": {
            "volume_msh": str(out_msh),
            "volume_bdf": str(out_bdf),
            "entity_grids_csv": str(out_grids_csv),
            "entity_tets_csv": str(out_tets_csv),
            "raw_boundary_tris_csv": str(out_raw_boundary_csv),
            "raw_boundary_obj": str(out_raw_boundary_obj),
            "raw_boundary_stl": str(out_raw_boundary_stl),
            "tightened_boundary_grids_csv": str(out_tight_grids_csv),
            "tightened_boundary_tris_csv": str(out_tight_tris_csv),
            "tightened_boundary_obj": str(out_tight_obj),
            "tightened_boundary_stl": str(out_tight_stl),
            "boundary_compare_html": str(out_compare_html),
            "boundary_compare_png": str(out_compare_png),
            "report_json": str(out_report),
        },
    }
    out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(terminal_trace)
    print("")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
