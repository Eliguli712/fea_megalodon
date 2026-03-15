#!/usr/bin/env python3
"""Build six von Mises mechanical-stress view sheets from saved exports.

Each output sheet includes:
- frontal view
- left-side view
- right-side view
- enlarged hotspot inset

The hotspot is an inference from the highest von Mises region and is labeled
as a rupture / feeding-failure proxy rather than a physically fractured model.
"""

from __future__ import annotations

import argparse
import tempfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import meshio
import numpy as np
from matplotlib.collections import PolyCollection
from scipy.spatial import cKDTree


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
EXPORTS = SCRIPT_DIR / "exports"
MODEL_PATH = EXPORTS / "static_dynamics_high_resolution_strict3bdf.mph"
OUT_DIR = EXPORTS / "von_mises_mechanical_views"

ENTITY_ORDER = [
    ("surface_mesh_smoothed", "smoothed"),
    ("tooth_surface_uncompressed", "uncompressed"),
    ("tooth_surface_comsol_tet_vol", "rawtet"),
]

ENTITY_META = {
    "surface_mesh_smoothed": {
        "label": "surface_mesh_smoothed",
        "bdf": EXPORTS / "surface_mesh_smoothed.bdf",
        "data": EXPORTS / "von_mises_data_smoothed.txt",
    },
    "tooth_surface_uncompressed": {
        "label": "tooth_surface_uncompressed",
        "bdf": EXPORTS / "tooth_surface_uncompressed.bdf",
        "data": EXPORTS / "von_mises_data_uncompressed.txt",
    },
    "tooth_surface_comsol_tet_vol": {
        "label": "tooth_surface_comsol_tet_vol",
        "bdf": EXPORTS / "tooth_surface_comsol_tet_vol.bdf",
        "data": EXPORTS / "von_mises_data_rawtet.txt",
    },
}


@dataclass
class EntityData:
    name: str
    short: str
    data_xyz: np.ndarray
    data_vm: np.ndarray
    bdf_points: np.ndarray
    tris: np.ndarray
    basis: np.ndarray


def load_comsol_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(path, comments="%")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr[:, :3].astype(np.float64, copy=False), arr[:, 3].astype(np.float64, copy=False)


def read_bdf_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    wrapped = f"CEND\nBEGIN BULK\n{raw}\nENDDATA\n"
    with tempfile.NamedTemporaryFile("w", suffix=".bdf", delete=False) as tf:
        tf.write(wrapped)
        wrapped_path = Path(tf.name)
    mesh = meshio.read(wrapped_path)
    wrapped_path.unlink(missing_ok=True)

    tri_blocks = [cells.data for cells in mesh.cells if cells.type == "triangle"]
    tris = np.vstack(tri_blocks).astype(np.int64, copy=False) if tri_blocks else np.empty((0, 3), dtype=np.int64)
    return mesh.points.astype(np.float64, copy=False), tris


def compute_basis(vertices: np.ndarray) -> np.ndarray:
    centered = vertices - vertices.mean(axis=0, keepdims=True)
    cov = np.cov(centered.T)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    axes = evecs[:, order]

    y_axis = axes[:, 0]
    x_axis = axes[:, 1]
    if y_axis[2] < 0:
        y_axis = -y_axis
    if x_axis[1] < 0:
        x_axis = -x_axis
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)
    if np.linalg.det(np.column_stack((x_axis, y_axis, z_axis))) < 0:
        z_axis = -z_axis
    return np.column_stack((x_axis, y_axis, z_axis))


def orient(vertices: np.ndarray, basis: np.ndarray) -> np.ndarray:
    centered = vertices - vertices.mean(axis=0, keepdims=True)
    return centered @ basis


def sample_points(xyz: np.ndarray, vm: np.ndarray, *, cap: int = 130_000) -> tuple[np.ndarray, np.ndarray]:
    if xyz.shape[0] <= cap:
        return xyz, vm
    idx = np.linspace(0, xyz.shape[0] - 1, cap, dtype=np.int64)
    return xyz[idx], vm[idx]


def sample_tris(tris: np.ndarray, *, cap: int = 220_000) -> np.ndarray:
    if tris.shape[0] <= cap:
        return tris
    idx = np.linspace(0, tris.shape[0] - 1, cap, dtype=np.int64)
    return tris[idx]


def map_vm_to_vertices(vertices: np.ndarray, tris: np.ndarray, data_xyz: np.ndarray, data_vm: np.ndarray) -> np.ndarray:
    uniq = np.unique(tris.reshape(-1))
    tree = cKDTree(data_xyz)
    _, nn = tree.query(vertices[uniq], k=1, workers=-1)

    vm_by_vertex = np.full(vertices.shape[0], np.nan, dtype=np.float64)
    vm_by_vertex[uniq] = data_vm[nn]
    return vm_by_vertex


def make_norm(values: np.ndarray):
    valid = values[np.isfinite(values)]
    vmin = float(np.nanquantile(valid, 0.02))
    vmax = float(np.nanquantile(valid, 0.98))
    return plt.Normalize(vmin=vmin, vmax=vmax)


def project(points: np.ndarray, horizontal: int, vertical: int) -> np.ndarray:
    return points[:, [horizontal, vertical]]


def hotspot_mask(values: np.ndarray, *, q: float = 0.995) -> np.ndarray:
    valid = np.isfinite(values)
    threshold = float(np.nanquantile(values[valid], q))
    return values >= threshold


def draw_point_view(ax, pts3: np.ndarray, values: np.ndarray, norm, *, horizontal: int, vertical: int, title: str):
    pts2 = project(pts3, horizontal, vertical)
    ax.scatter(pts2[:, 0], pts2[:, 1], c=values, s=2.5, cmap="turbo", norm=norm, linewidths=0, alpha=0.9)

    hot = hotspot_mask(values)
    if np.any(hot):
        ax.scatter(
            pts2[hot, 0],
            pts2[hot, 1],
            s=10,
            facecolors="none",
            edgecolors="black",
            linewidths=0.35,
            alpha=0.6,
        )

    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title, fontsize=12, fontweight="bold", color="#173245")


def draw_surface_view(
    ax,
    pts3: np.ndarray,
    tris: np.ndarray,
    tri_values: np.ndarray,
    norm,
    *,
    horizontal: int,
    vertical: int,
    depth_axis: int,
    reverse_depth: bool,
    title: str,
):
    poly2 = pts3[tris][:, :, [horizontal, vertical]]
    depth = pts3[tris][:, :, depth_axis].mean(axis=1)
    order = np.argsort(-depth if reverse_depth else depth)
    poly2 = poly2[order]
    tri_values = tri_values[order]

    colors = matplotlib.colormaps["turbo"](norm(tri_values))
    colors[:, 3] = 0.985

    coll = PolyCollection(
        poly2,
        facecolors=colors,
        edgecolors=(0.02, 0.02, 0.03, 0.04),
        linewidths=0.02,
        antialiased=False,
    )
    ax.add_collection(coll)

    hot = hotspot_mask(tri_values)
    if np.any(hot):
        hot_coll = PolyCollection(
            poly2[hot],
            facecolors=(0, 0, 0, 0),
            edgecolors=(0, 0, 0, 0.28),
            linewidths=0.18,
            antialiased=False,
        )
        ax.add_collection(hot_coll)

    pts2 = project(pts3, horizontal, vertical)
    x_pad = np.ptp(pts2[:, 0]) * 0.08
    y_pad = np.ptp(pts2[:, 1]) * 0.08
    ax.set_xlim(float(pts2[:, 0].min() - x_pad), float(pts2[:, 0].max() + x_pad))
    ax.set_ylim(float(pts2[:, 1].min() - y_pad), float(pts2[:, 1].max() + y_pad))
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title, fontsize=12, fontweight="bold", color="#173245")


def draw_hotspot_inset_point(ax, pts3: np.ndarray, values: np.ndarray, norm):
    pts2 = project(pts3, 0, 1)
    hot = hotspot_mask(values)
    spread = np.ptp(pts2, axis=0)
    if np.any(hot):
        hot_pts = pts2[hot]
        center = hot_pts.mean(axis=0)
        hot_spread = np.ptp(hot_pts, axis=0)
        window = np.maximum(np.maximum(hot_spread * 2.2, spread * 0.08), 1e-6)
    else:
        center = pts2.mean(axis=0)
        window = np.maximum(spread * 0.18, 1e-6)

    ax.scatter(pts2[:, 0], pts2[:, 1], c=values, s=3, cmap="turbo", norm=norm, linewidths=0, alpha=0.45)
    if np.any(hot):
        ax.scatter(pts2[hot, 0], pts2[hot, 1], c=values[hot], s=8, cmap="turbo", norm=norm, linewidths=0, alpha=1.0)

    ax.set_xlim(center[0] - window[0], center[0] + window[0])
    ax.set_ylim(center[1] - window[1], center[1] + window[1])
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title("Rupture / feeding-failure proxy", fontsize=12, fontweight="bold", color="#173245")


def draw_hotspot_inset_surface(ax, pts3: np.ndarray, tris: np.ndarray, tri_values: np.ndarray, norm):
    tri_pts2 = pts3[tris][:, :, [0, 1]]
    hot = hotspot_mask(tri_values)
    centers = tri_pts2.mean(axis=1)
    spread = np.ptp(project(pts3, 0, 1), axis=0)
    if np.any(hot):
        hot_centers = centers[hot]
        center = hot_centers.mean(axis=0)
        hot_spread = np.ptp(hot_centers, axis=0)
        window = np.maximum(np.maximum(hot_spread * 2.6, spread * 0.08), 1e-6)
    else:
        center = centers.mean(axis=0)
        window = np.maximum(spread * 0.18, 1e-6)

    order = np.argsort(pts3[tris][:, :, 2].mean(axis=1))
    tri_pts2 = tri_pts2[order]
    tri_values = tri_values[order]
    hot = hot[order]

    colors = matplotlib.colormaps["turbo"](norm(tri_values))
    colors[:, 3] = 0.97
    base = PolyCollection(tri_pts2, facecolors=colors, edgecolors=(0.02, 0.02, 0.03, 0.03), linewidths=0.02)
    ax.add_collection(base)

    if np.any(hot):
        overlay = PolyCollection(
            tri_pts2[hot],
            facecolors=(0, 0, 0, 0),
            edgecolors=(0, 0, 0, 0.32),
            linewidths=0.2,
            antialiased=False,
        )
        ax.add_collection(overlay)

    ax.set_xlim(center[0] - window[0], center[0] + window[0])
    ax.set_ylim(center[1] - window[1], center[1] + window[1])
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title("Rupture / feeding-failure proxy", fontsize=12, fontweight="bold", color="#173245")


def entity_payload(name: str, short: str) -> EntityData:
    meta = ENTITY_META[name]
    data_xyz, data_vm = load_comsol_data(meta["data"])
    bdf_points, tris = read_bdf_mesh(meta["bdf"])
    basis = compute_basis(data_xyz)
    return EntityData(name=name, short=short, data_xyz=data_xyz, data_vm=data_vm, bdf_points=bdf_points, tris=tris, basis=basis)


def build_point_sheet(ent: EntityData, out_path: Path) -> None:
    pts, values = sample_points(orient(ent.data_xyz, ent.basis), ent.data_vm)
    norm = make_norm(values)

    fig, axs = plt.subplots(2, 2, figsize=(14.6, 13.4), dpi=220, facecolor="#eef4f8")
    for ax in axs.ravel():
        ax.set_facecolor("#f8fbfd")

    draw_point_view(axs[0, 0], pts, values, norm, horizontal=0, vertical=1, title="Frontal")
    draw_point_view(axs[0, 1], pts, values, norm, horizontal=2, vertical=1, title="Left Side")
    draw_point_view(axs[1, 0], pts, values, norm, horizontal=2, vertical=1, title="Right Side")
    axs[1, 0].invert_xaxis()
    draw_hotspot_inset_point(axs[1, 1], pts, values, norm)

    fig.suptitle(f"{ent.name} · Point Cloud", fontsize=24, fontweight="bold", color="#163348", y=0.985)
    fig.text(
        0.05,
        0.955,
        f"Model: {MODEL_PATH.relative_to(PROJECT_ROOT)} | data: {ENTITY_META[ent.name]['data'].relative_to(PROJECT_ROOT)}",
        fontsize=10,
        color="#446578",
    )
    fig.text(
        0.05,
        0.035,
        "Hotspot inset shows the top 0.5% von Mises region as a rupture / feeding-failure proxy inferred from static stress, not a fracture simulation.",
        fontsize=10,
        color="#446578",
    )

    cax = fig.add_axes([0.92, 0.16, 0.022, 0.68])
    sm = plt.cm.ScalarMappable(cmap="turbo", norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Von Mises (Pa)", fontsize=11)
    cb.ax.tick_params(labelsize=9)

    fig.tight_layout(rect=[0.02, 0.06, 0.9, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)


def build_surface_sheet(ent: EntityData, out_path: Path) -> None:
    tris = sample_tris(ent.tris)
    pts3 = orient(ent.bdf_points, ent.basis)
    vm_by_vertex = map_vm_to_vertices(ent.bdf_points, tris, ent.data_xyz, ent.data_vm)
    tri_values = np.nanmean(vm_by_vertex[tris], axis=1)
    norm = make_norm(tri_values)

    fig, axs = plt.subplots(2, 2, figsize=(14.6, 13.4), dpi=220, facecolor="#eef4f8")
    for ax in axs.ravel():
        ax.set_facecolor("#f8fbfd")

    draw_surface_view(axs[0, 0], pts3, tris, tri_values, norm, horizontal=0, vertical=1, depth_axis=2, reverse_depth=False, title="Frontal")
    draw_surface_view(axs[0, 1], pts3, tris, tri_values, norm, horizontal=2, vertical=1, depth_axis=0, reverse_depth=False, title="Left Side")
    draw_surface_view(axs[1, 0], pts3, tris, tri_values, norm, horizontal=2, vertical=1, depth_axis=0, reverse_depth=True, title="Right Side")
    axs[1, 0].invert_xaxis()
    draw_hotspot_inset_surface(axs[1, 1], pts3, tris, tri_values, norm)

    fig.suptitle(f"{ent.name} · Surface", fontsize=24, fontweight="bold", color="#163348", y=0.985)
    fig.text(
        0.05,
        0.955,
        f"Model: {MODEL_PATH.relative_to(PROJECT_ROOT)} | mesh: {ENTITY_META[ent.name]['bdf'].relative_to(PROJECT_ROOT)}",
        fontsize=10,
        color="#446578",
    )
    fig.text(
        0.05,
        0.035,
        "Hotspot inset shows the top 0.5% von Mises region as a rupture / feeding-failure proxy inferred from static stress, not a fracture simulation.",
        fontsize=10,
        color="#446578",
    )

    cax = fig.add_axes([0.92, 0.16, 0.022, 0.68])
    sm = plt.cm.ScalarMappable(cmap="turbo", norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Von Mises (Pa)", fontsize=11)
    cb.ax.tick_params(labelsize=9)

    fig.tight_layout(rect=[0.02, 0.06, 0.9, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render frontal/side/hotspot von Mises sheets for all three entities.")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, short in ENTITY_ORDER:
        ent = entity_payload(name, short)
        build_point_sheet(ent, out_dir / f"{name}_point_cloud_views.png")
        build_surface_sheet(ent, out_dir / f"{name}_surface_views.png")
        print(f"DONE|{name}")


if __name__ == "__main__":
    main()
