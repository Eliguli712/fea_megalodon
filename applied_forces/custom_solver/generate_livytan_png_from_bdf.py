#!/usr/bin/env python3
"""Generate non-WebGL PNG previews from volumetric BDF mesh data.

Outputs:
- livytan_pg_geom_preview_comsol.png
- livytan_pg_vms_preview_comsol.png
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


ROOT = Path(__file__).resolve().parent
BDF = ROOT / "livytan_melville_teeth.bdf"
OUT_GEOM = ROOT / "livytan_pg_geom_preview_comsol.png"
OUT_VMS = ROOT / "livytan_pg_vms_preview_comsol.png"


def split_fixed_fields(line: str) -> List[str]:
    text = line.rstrip("\n")
    return [text[i : i + 8].strip() for i in range(0, len(text), 8)]


def parse_float(token: str) -> float:
    s = token.strip().replace("D", "E").replace("d", "E")
    if not s:
        raise ValueError("empty float token")
    if "E" not in s and "e" not in s:
        # Nastran compact exponent forms like 1.2345-3
        pos = max(s.rfind("+"), s.rfind("-"))
        if pos > 0:
            prefix = s[:pos]
            suffix = s[pos:]
            if prefix and suffix[1:].isdigit():
                s = f"{prefix}E{suffix}"
    return float(s)


def parse_int(token: str) -> int:
    s = token.strip()
    if not s:
        raise ValueError("empty int token")
    return int(s)


def load_bdf_mesh(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    node_ids: List[int] = []
    node_xyz: List[Tuple[float, float, float]] = []
    tri_ids: List[Tuple[int, int, int]] = []
    tet_ids: List[Tuple[int, int, int, int]] = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line or line.startswith("$"):
                continue
            card = line[:8].strip()
            if card not in {"GRID", "CTRIA3", "CTETRA"}:
                continue

            fields = split_fixed_fields(line)
            try:
                if card == "GRID":
                    if len(fields) < 6:
                        continue
                    nid = parse_int(fields[1])
                    x = parse_float(fields[3])
                    y = parse_float(fields[4])
                    z = parse_float(fields[5])
                    node_ids.append(nid)
                    node_xyz.append((x, y, z))
                elif card == "CTRIA3":
                    if len(fields) < 6:
                        continue
                    tri_ids.append((parse_int(fields[3]), parse_int(fields[4]), parse_int(fields[5])))
                elif card == "CTETRA":
                    if len(fields) < 7:
                        continue
                    tet_ids.append(
                        (
                            parse_int(fields[3]),
                            parse_int(fields[4]),
                            parse_int(fields[5]),
                            parse_int(fields[6]),
                        )
                    )
            except ValueError:
                continue

    if not node_ids or not tri_ids:
        raise RuntimeError("Failed to parse required GRID/CTRIA3 data from BDF")

    nodes = np.asarray(node_xyz, dtype=np.float64)
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    tris = np.empty((len(tri_ids), 3), dtype=np.int32)
    for i, (a, b, c) in enumerate(tri_ids):
        tris[i, 0] = id_to_idx[a]
        tris[i, 1] = id_to_idx[b]
        tris[i, 2] = id_to_idx[c]

    tets = np.empty((len(tet_ids), 4), dtype=np.int32)
    for i, (a, b, c, d) in enumerate(tet_ids):
        tets[i, 0] = id_to_idx[a]
        tets[i, 1] = id_to_idx[b]
        tets[i, 2] = id_to_idx[c]
        tets[i, 3] = id_to_idx[d]

    return nodes, tris, tets


def orient_vertices(vertices: np.ndarray) -> np.ndarray:
    centered = vertices - vertices.mean(axis=0, keepdims=True)
    sample = centered[::10] if centered.shape[0] > 10000 else centered
    _, _, vh = np.linalg.svd(sample, full_matrices=False)
    rotation = vh.T
    return centered @ rotation


def apply_axes_style(ax, vertices: np.ndarray) -> None:
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    center = 0.5 * (mins + maxs)
    span_xyz = maxs - mins
    span = float(np.max(span_xyz))

    zoom = 0.40
    ax.set_xlim(center[0] - zoom * span, center[0] + zoom * span)
    ax.set_ylim(center[1] - zoom * span, center[1] + zoom * span)
    ax.set_zlim(center[2] - zoom * span, center[2] + zoom * span)

    # Keep aspect proportional while fitting longest axis.
    ax.set_box_aspect(tuple((span_xyz / span).tolist()))
    ax.set_axis_off()
    ax.view_init(elev=35, azim=40)



def triangle_shading(tri_vertices: np.ndarray) -> np.ndarray:
    v1 = tri_vertices[:, 1, :] - tri_vertices[:, 0, :]
    v2 = tri_vertices[:, 2, :] - tri_vertices[:, 0, :]
    normals = np.cross(v1, v2)
    nrm = np.linalg.norm(normals, axis=1)
    normals = normals / (nrm[:, None] + 1e-12)
    light = np.array([0.45, 0.36, 0.82], dtype=np.float64)
    light = light / np.linalg.norm(light)
    return np.clip(normals @ light, 0.0, 1.0)


def render_geometry(vertices: np.ndarray, tris: np.ndarray, out_path: Path) -> None:
    tri_vertices = vertices[tris]
    shade = triangle_shading(tri_vertices)

    base = np.array([0.18, 0.47, 0.72], dtype=np.float64)
    face_rgb = np.clip(base[None, :] * (0.40 + 0.60 * shade[:, None]), 0.0, 1.0)

    fig = plt.figure(figsize=(16, 11), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("#f5f1e8")
    ax.set_facecolor("#f5f1e8")

    poly = Poly3DCollection(
        tri_vertices,
        facecolors=face_rgb,
        linewidths=0.05,
        edgecolors=(0.01, 0.01, 0.01, 0.18),
        antialiased=False,
    )
    ax.add_collection3d(poly)

    apply_axes_style(ax, vertices)
    fig.suptitle("Livytan Melville Teeth: Surface Mesh Preview", fontsize=18, y=0.95)
    fig.text(0.5, 0.03, "Source: livytan_melville_teeth.bdf (CTRIA3 boundary)", ha="center", fontsize=11)

    fig.savefig(out_path, dpi=160, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)



def render_vms_proxy(vertices: np.ndarray, tris: np.ndarray, tets: np.ndarray, out_path: Path) -> None:
    tri_vertices = vertices[tris]
    shade = triangle_shading(tri_vertices)

    # Tetra connectivity-derived scalar gives a stable non-empty volumetric indicator.
    node_tet_count = np.zeros(vertices.shape[0], dtype=np.int32)
    for i in range(4):
        np.add.at(node_tet_count, tets[:, i], 1)

    tri_scalar = np.log1p(node_tet_count[tris].mean(axis=1).astype(np.float64))
    vmin = float(np.percentile(tri_scalar, 2.0))
    vmax = float(np.percentile(tri_scalar, 98.0))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    norm = Normalize(vmin=vmin, vmax=vmax)

    face_rgba = cm.turbo(norm(np.clip(tri_scalar, vmin, vmax)))
    face_rgba[:, :3] = np.clip(face_rgba[:, :3] * (0.50 + 0.50 * shade[:, None]), 0.0, 1.0)

    fig = plt.figure(figsize=(16, 11), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("#eef3f7")
    ax.set_facecolor("#eef3f7")

    poly = Poly3DCollection(
        tri_vertices,
        facecolors=face_rgba,
        linewidths=0.05,
        edgecolors=(0.0, 0.0, 0.0, 0.10),
        antialiased=False,
    )
    ax.add_collection3d(poly)

    apply_axes_style(ax, vertices)

    sm = cm.ScalarMappable(norm=norm, cmap="turbo")
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.026, pad=0.02)
    cbar.set_label("Volumetric connectivity scalar (log scale)", rotation=90)

    fig.suptitle("Livytan Melville Teeth: Volumetric Result Preview", fontsize=18, y=0.95)
    fig.text(
        0.5,
        0.03,
        "Derived from CTETRA connectivity over boundary triangles (static PNG fallback)",
        ha="center",
        fontsize=11,
    )

    fig.savefig(out_path, dpi=160, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)



def main() -> None:
    nodes, tris, tets = load_bdf_mesh(BDF)
    oriented = orient_vertices(nodes)

    render_geometry(oriented, tris, OUT_GEOM)
    render_vms_proxy(oriented, tris, tets, OUT_VMS)

    print(f"WROTE|{OUT_GEOM}")
    print(f"WROTE|{OUT_VMS}")
    print(f"COUNTS|vertices={nodes.shape[0]}|tri={tris.shape[0]}|tet={tets.shape[0]}")


if __name__ == "__main__":
    main()
