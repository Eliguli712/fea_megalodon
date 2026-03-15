#!/usr/bin/env python3
"""Render a frontal von Mises tooth view from saved COMSOL exports."""

from __future__ import annotations

import argparse
import tempfile
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

ENTITIES = {
    "smoothed": {
        "label": "surface_mesh_smoothed",
        "bdf": EXPORTS / "surface_mesh_smoothed.bdf",
        "data": EXPORTS / "von_mises_data_smoothed.txt",
    },
    "uncompressed": {
        "label": "tooth_surface_uncompressed",
        "bdf": EXPORTS / "tooth_surface_uncompressed.bdf",
        "data": EXPORTS / "von_mises_data_uncompressed.txt",
    },
    "rawtet": {
        "label": "tooth_surface_comsol_tet_vol",
        "bdf": EXPORTS / "tooth_surface_comsol_tet_vol.bdf",
        "data": EXPORTS / "von_mises_data_rawtet.txt",
    },
}


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
    if tri_blocks:
        tris = np.vstack(tri_blocks).astype(np.int64, copy=False)
    else:
        tris = np.empty((0, 3), dtype=np.int64)
    return mesh.points.astype(np.float64, copy=False), tris


def oriented_vertices(vertices: np.ndarray) -> np.ndarray:
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

    basis = np.column_stack((x_axis, y_axis, z_axis))
    return centered @ basis


def map_vm_to_vertices(vertices: np.ndarray, tris: np.ndarray, data_xyz: np.ndarray, data_vm: np.ndarray) -> np.ndarray:
    uniq = np.unique(tris.reshape(-1))
    tree = cKDTree(data_xyz)
    _, nn = tree.query(vertices[uniq], k=1, workers=-1)

    vm_by_vertex = np.full(vertices.shape[0], np.nan, dtype=np.float64)
    vm_by_vertex[uniq] = data_vm[nn]
    return vm_by_vertex


def sample_tris(tris: np.ndarray, *, cap: int = 220_000) -> np.ndarray:
    if tris.shape[0] <= cap:
        return tris
    idx = np.linspace(0, tris.shape[0] - 1, cap, dtype=np.int64)
    return tris[idx]


def render_frontal_view(
    out_path: Path,
    label: str,
    vertices: np.ndarray,
    tris: np.ndarray,
    vm_by_vertex: np.ndarray,
) -> None:
    pts = oriented_vertices(vertices)
    polys = pts[tris][:, :, :2]
    depth = pts[tris][:, :, 2].mean(axis=1)
    tri_vm = np.nanmean(vm_by_vertex[tris], axis=1)

    order = np.argsort(depth)
    polys = polys[order]
    tri_vm = tri_vm[order]

    valid = np.isfinite(tri_vm)
    if not np.any(valid):
        raise RuntimeError("No finite von Mises values were mapped onto the frontal surface.")

    vmin = float(np.nanquantile(tri_vm[valid], 0.02))
    vmax = float(np.nanquantile(tri_vm[valid], 0.98))
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    colors = matplotlib.colormaps["turbo"](norm(tri_vm))
    colors[:, 3] = 0.985

    x = pts[:, 0]
    y = pts[:, 1]
    x_pad = np.ptp(x) * 0.08
    y_pad = np.ptp(y) * 0.08

    fig = plt.figure(figsize=(9.4, 11.6), dpi=260, facecolor="#eef4f8")
    ax = fig.add_axes([0.07, 0.08, 0.78, 0.84], facecolor="#f8fbfd")
    coll = PolyCollection(
        polys,
        facecolors=colors,
        edgecolors=(0.02, 0.02, 0.03, 0.04),
        linewidths=0.03,
        antialiased=False,
    )
    ax.add_collection(coll)
    ax.set_xlim(float(x.min() - x_pad), float(x.max() + x_pad))
    ax.set_ylim(float(y.min() - y_pad), float(y.max() + y_pad))
    ax.set_aspect("equal")
    ax.set_axis_off()

    cax = fig.add_axes([0.88, 0.16, 0.035, 0.68])
    sm = plt.cm.ScalarMappable(cmap="turbo", norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Von Mises (Pa)", fontsize=11)
    cb.ax.tick_params(labelsize=9)

    fig.text(0.07, 0.965, "Frontal Von Mises Tooth View", fontsize=22, fontweight="bold", color="#163348")
    fig.text(
        0.07,
        0.936,
        f"Model: {MODEL_PATH.relative_to(PROJECT_ROOT)} | entity: {label}",
        fontsize=10,
        color="#3d5f74",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=260, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a frontal von Mises PNG from saved strict3bdf exports.")
    parser.add_argument("--entity", choices=sorted(ENTITIES), default="smoothed")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "teeth_von_mises_frontal.png"))
    args = parser.parse_args()

    entity = ENTITIES[args.entity]
    data_xyz, data_vm = load_comsol_data(entity["data"])
    vertices, tris = read_bdf_mesh(entity["bdf"])
    if tris.size == 0:
        raise RuntimeError(f"No triangle faces found in {entity['bdf']}")
    tris = sample_tris(tris)

    vm_by_vertex = map_vm_to_vertices(vertices, tris, data_xyz, data_vm)
    render_frontal_view(Path(args.out).resolve(), entity["label"], vertices, tris, vm_by_vertex)
    print(Path(args.out).resolve())


if __name__ == "__main__":
    main()
