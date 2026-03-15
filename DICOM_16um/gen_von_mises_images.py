#!/usr/bin/env python3
"""Generate von Mises point-cloud and surface PNGs from COMSOL data exports."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import meshio
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import cKDTree


ROOT = Path("/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/DICOM_16um")
EXPORTS = ROOT / "exports"

ENTITIES = [
    {
        "name": "surface_mesh_smoothed",
        "short": "smoothed",
        "bdf": EXPORTS / "surface_mesh_smoothed.bdf",
        "data": EXPORTS / "von_mises_data_smoothed.txt",
    },
    {
        "name": "tooth_surface_uncompressed",
        "short": "uncompressed",
        "bdf": EXPORTS / "tooth_surface_uncompressed.bdf",
        "data": EXPORTS / "von_mises_data_uncompressed.txt",
    },
    {
        "name": "tooth_surface_comsol_tet_vol",
        "short": "rawtet",
        "bdf": EXPORTS / "tooth_surface_comsol_tet_vol.bdf",
        "data": EXPORTS / "von_mises_data_rawtet.txt",
    },
]


def load_comsol_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(path, comments="%")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    xyz = arr[:, :3].astype(np.float64, copy=False)
    vm = arr[:, 3].astype(np.float64, copy=False)
    return xyz, vm


def read_bdf_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    wrapped = f"CEND\nBEGIN BULK\n{raw}\nENDDATA\n"
    with tempfile.NamedTemporaryFile("w", suffix=".bdf", delete=False) as tf:
        tf.write(wrapped)
        wrapped_path = Path(tf.name)
    mesh = meshio.read(wrapped_path)
    wrapped_path.unlink(missing_ok=True)

    tri_blocks = [c.data for c in mesh.cells if c.type == "triangle"]
    if tri_blocks:
        tris = np.vstack(tri_blocks).astype(np.int64, copy=False)
    else:
        tris = np.empty((0, 3), dtype=np.int64)
    return mesh.points.astype(np.float64, copy=False), tris


def set_equal_axes(ax: plt.Axes, pts: np.ndarray) -> None:
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    ctr = 0.5 * (mins + maxs)
    half = 0.5 * np.max(maxs - mins)
    ax.set_xlim(ctr[0] - half, ctr[0] + half)
    ax.set_ylim(ctr[1] - half, ctr[1] + half)
    ax.set_zlim(ctr[2] - half, ctr[2] + half)


def render_point_cloud(name: str, short: str, xyz: np.ndarray, vm: np.ndarray) -> Path:
    n = xyz.shape[0]
    cap = 120_000
    if n > cap:
        idx = np.linspace(0, n - 1, cap, dtype=np.int64)
        p = xyz[idx]
        s = vm[idx]
    else:
        p = xyz
        s = vm

    vmin = float(np.quantile(s, 0.02))
    vmax = float(np.quantile(s, 0.98))

    fig = plt.figure(figsize=(10, 8), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=s, s=0.5, cmap="turbo", vmin=vmin, vmax=vmax, linewidths=0)
    ax.set_title(f"Von Mises Point Cloud: {name}", pad=14)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    set_equal_axes(ax, p)
    fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.06, label="Von Mises (Pa)")
    out = EXPORTS / f"von_mises_point_cloud_{short}.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def render_surface(
    name: str,
    short: str,
    bdf_points: np.ndarray,
    tris: np.ndarray,
    data_xyz: np.ndarray,
    data_vm: np.ndarray,
) -> tuple[Path, dict]:
    if tris.size == 0:
        raise RuntimeError(f"No triangle cells found for {name}")

    tri_cap = 180_000
    if tris.shape[0] > tri_cap:
        idx = np.linspace(0, tris.shape[0] - 1, tri_cap, dtype=np.int64)
        tris_view = tris[idx]
    else:
        tris_view = tris

    uniq = np.unique(tris_view.reshape(-1))
    tree = cKDTree(data_xyz)
    dists, nn = tree.query(bdf_points[uniq], k=1, workers=-1)
    mapped_vm = data_vm[nn]

    vm_by_vertex = np.empty(bdf_points.shape[0], dtype=np.float64)
    vm_by_vertex.fill(np.nan)
    vm_by_vertex[uniq] = mapped_vm

    tri_vals = np.nanmean(vm_by_vertex[tris_view], axis=1)
    tri_xyz = bdf_points[tris_view]

    vmin = float(np.nanquantile(tri_vals, 0.02))
    vmax = float(np.nanquantile(tri_vals, 0.98))
    cmap = plt.get_cmap("turbo")
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    face_colors = cmap(norm(tri_vals))

    fig = plt.figure(figsize=(10, 8), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    poly = Poly3DCollection(tri_xyz, linewidths=0.0, antialiased=False)
    poly.set_facecolor(face_colors)
    ax.add_collection3d(poly)

    set_equal_axes(ax, bdf_points[uniq])
    ax.set_title(f"Von Mises Surface: {name}", pad=14)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.06, label="Von Mises (Pa)")

    out = EXPORTS / f"von_mises_surface_{short}.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)

    stats = {
        "triangles_total": int(tris.shape[0]),
        "triangles_rendered": int(tris_view.shape[0]),
        "vertices_mapped": int(uniq.shape[0]),
        "nearest_dist_max": float(np.max(dists)) if dists.size else 0.0,
        "nearest_dist_p99": float(np.quantile(dists, 0.99)) if dists.size else 0.0,
    }
    return out, stats


def main() -> None:
    report: dict[str, dict] = {}

    for ent in ENTITIES:
        name = ent["name"]
        short = ent["short"]

        data_xyz, data_vm = load_comsol_data(ent["data"])
        bdf_points, tris = read_bdf_mesh(ent["bdf"])

        point_png = render_point_cloud(name, short, data_xyz, data_vm)
        surf_png, surf_stats = render_surface(name, short, bdf_points, tris, data_xyz, data_vm)

        report[name] = {
            "data_file": str(ent["data"]),
            "bdf_file": str(ent["bdf"]),
            "point_png": str(point_png),
            "surface_png": str(surf_png),
            "data_points": int(data_xyz.shape[0]),
            "bdf_points": int(bdf_points.shape[0]),
            "bdf_triangles": int(tris.shape[0]),
            "surface_mapping": surf_stats,
        }

    out_report = EXPORTS / "von_mises_image_generation_report.json"
    out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out_report}")


if __name__ == "__main__":
    main()
