#!/usr/bin/env python3
"""Build frontal and ventral von Mises montages for the six tooth properties."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from build_von_mises_mechanical_views import (
    ENTITY_ORDER,
    EntityData,
    entity_payload,
    hotspot_mask,
    make_norm,
    map_vm_to_vertices,
    orient,
    project,
    sample_points,
    sample_tris,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
EXPORTS = SCRIPT_DIR / "exports"
MODEL_PATH = EXPORTS / "static_dynamics_high_resolution_strict3bdf.mph"


def panel_title(label: str) -> str:
    left, right = label.split(" · ", maxsplit=1)
    return f"{left}\n{right}"


def draw_point(ax, pts3: np.ndarray, values: np.ndarray, norm, *, horizontal: int, vertical: int, depth_axis: int, reverse_depth: bool, title: str) -> None:
    pts2 = project(pts3, horizontal, vertical)
    order = np.argsort(-pts3[:, depth_axis] if reverse_depth else pts3[:, depth_axis])
    pts2 = pts2[order]
    values = values[order]

    ax.scatter(pts2[:, 0], pts2[:, 1], c=values, s=1.6, cmap="turbo", norm=norm, linewidths=0, alpha=0.92)

    hot = hotspot_mask(values)
    if np.any(hot):
        ax.scatter(
            pts2[hot, 0],
            pts2[hot, 1],
            s=6,
            facecolors="none",
            edgecolors="black",
            linewidths=0.22,
            alpha=0.45,
        )

    x_pad = np.ptp(pts2[:, 0]) * 0.08
    y_pad = np.ptp(pts2[:, 1]) * 0.08
    ax.set_xlim(float(pts2[:, 0].min() - x_pad), float(pts2[:, 0].max() + x_pad))
    ax.set_ylim(float(pts2[:, 1].min() - y_pad), float(pts2[:, 1].max() + y_pad))
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title, fontsize=10, fontweight="bold", color="#173245", pad=8, linespacing=1.28)


def draw_surface(ax, pts3: np.ndarray, tris: np.ndarray, tri_values: np.ndarray, norm, *, horizontal: int, vertical: int, depth_axis: int, reverse_depth: bool, title: str) -> None:
    from matplotlib.collections import PolyCollection

    polys = pts3[tris][:, :, [horizontal, vertical]]
    depth = pts3[tris][:, :, depth_axis].mean(axis=1)
    order = np.argsort(-depth if reverse_depth else depth)
    polys = polys[order]
    tri_values = tri_values[order]

    colors = matplotlib.colormaps["turbo"](norm(tri_values))
    colors[:, 3] = 0.985
    coll = PolyCollection(
        polys,
        facecolors=colors,
        edgecolors=(0.02, 0.02, 0.03, 0.035),
        linewidths=0.02,
        antialiased=False,
    )
    ax.add_collection(coll)

    hot = hotspot_mask(tri_values)
    if np.any(hot):
        overlay = PolyCollection(
            polys[hot],
            facecolors=(0, 0, 0, 0),
            edgecolors=(0, 0, 0, 0.20),
            linewidths=0.14,
            antialiased=False,
        )
        ax.add_collection(overlay)

    pts2 = project(pts3, horizontal, vertical)
    x_pad = np.ptp(pts2[:, 0]) * 0.08
    y_pad = np.ptp(pts2[:, 1]) * 0.08
    ax.set_xlim(float(pts2[:, 0].min() - x_pad), float(pts2[:, 0].max() + x_pad))
    ax.set_ylim(float(pts2[:, 1].min() - y_pad), float(pts2[:, 1].max() + y_pad))
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title, fontsize=10, fontweight="bold", color="#173245", pad=8, linespacing=1.28)


def draw_point_failure_mask(
    ax,
    pts3: np.ndarray,
    values: np.ndarray,
    norm,
    *,
    horizontal: int,
    vertical: int,
    depth_axis: int,
    reverse_depth: bool,
    title: str,
) -> None:
    pts2 = project(pts3, horizontal, vertical)
    order = np.argsort(-pts3[:, depth_axis] if reverse_depth else pts3[:, depth_axis])
    pts2 = pts2[order]
    values = values[order]

    ax.scatter(pts2[:, 0], pts2[:, 1], c=values, s=1.6, cmap="turbo", norm=norm, linewidths=0, alpha=0.72)
    hot = hotspot_mask(values)
    if np.any(hot):
        ax.scatter(pts2[hot, 0], pts2[hot, 1], s=3.8, c="#ff2ea6", linewidths=0, alpha=0.95)

    x_pad = np.ptp(pts2[:, 0]) * 0.08
    y_pad = np.ptp(pts2[:, 1]) * 0.08
    ax.set_xlim(float(pts2[:, 0].min() - x_pad), float(pts2[:, 0].max() + x_pad))
    ax.set_ylim(float(pts2[:, 1].min() - y_pad), float(pts2[:, 1].max() + y_pad))
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title, fontsize=10, fontweight="bold", color="#173245", pad=8, linespacing=1.28)


def draw_surface_failure_mask(
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
) -> None:
    from matplotlib.collections import PolyCollection

    polys = pts3[tris][:, :, [horizontal, vertical]]
    depth = pts3[tris][:, :, depth_axis].mean(axis=1)
    order = np.argsort(-depth if reverse_depth else depth)
    polys = polys[order]
    tri_values = tri_values[order]

    colors = matplotlib.colormaps["turbo"](norm(tri_values))
    colors[:, 3] = 0.82
    coll = PolyCollection(
        polys,
        facecolors=colors,
        edgecolors=(0.02, 0.02, 0.03, 0.12),
        linewidths=0.05,
        antialiased=False,
    )
    ax.add_collection(coll)

    hot = hotspot_mask(tri_values)
    if np.any(hot):
        overlay = PolyCollection(
            polys[hot],
            facecolors=(1.0, 0.18, 0.65, 0.78),
            edgecolors=(0.45, 0.0, 0.28, 0.85),
            linewidths=0.10,
            antialiased=False,
        )
        ax.add_collection(overlay)

    pts2 = project(pts3, horizontal, vertical)
    x_pad = np.ptp(pts2[:, 0]) * 0.08
    y_pad = np.ptp(pts2[:, 1]) * 0.08
    ax.set_xlim(float(pts2[:, 0].min() - x_pad), float(pts2[:, 0].max() + x_pad))
    ax.set_ylim(float(pts2[:, 1].min() - y_pad), float(pts2[:, 1].max() + y_pad))
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title, fontsize=10, fontweight="bold", color="#173245", pad=8, linespacing=1.28)


def entity_plot_data(ent: EntityData, is_surface: bool):
    if is_surface:
        tris = sample_tris(ent.tris)
        pts3 = orient(ent.bdf_points, ent.basis)
        vm_by_vertex = map_vm_to_vertices(ent.bdf_points, tris, ent.data_xyz, ent.data_vm)
        tri_values = np.nanmean(vm_by_vertex[tris], axis=1)
        return pts3, tris, tri_values

    pts3, values = sample_points(orient(ent.data_xyz, ent.basis), ent.data_vm)
    return pts3, None, values


def shared_norm(payload_cache: dict[tuple[str, bool], tuple]) -> matplotlib.colors.Normalize:
    all_values = np.concatenate(
        [values[np.isfinite(values)] for pts3, tris, values in payload_cache.values()],
        axis=0,
    )
    return make_norm(all_values)


def build_montage(out_path: Path, *, ventral: bool) -> None:
    fig, axs = plt.subplots(2, 3, figsize=(17.2, 10.8), dpi=240, facecolor="#eef4f8")
    for ax in axs.ravel():
        ax.set_facecolor("#f8fbfd")

    if ventral:
        title = "Dentis Side-Edge Failure Views"
        subtitle = "Opposite lateral side projection with a distinct feeding-damage mask over the highest von Mises region"
        horizontal, vertical, depth_axis, reverse_depth = 2, 1, 0, True
    else:
        title = "Dentis Frontal Von Mises Views"
        subtitle = "Frontal projection of the six saved von Mises property sets"
        horizontal, vertical, depth_axis, reverse_depth = 0, 1, 2, False

    panels = [
        ("surface_mesh_smoothed", False, "surface_mesh_smoothed · Point Cloud"),
        ("surface_mesh_smoothed", True, "surface_mesh_smoothed · Surface"),
        ("tooth_surface_uncompressed", False, "tooth_surface_uncompressed · Point Cloud"),
        ("tooth_surface_uncompressed", True, "tooth_surface_uncompressed · Surface"),
        ("tooth_surface_comsol_tet_vol", False, "tooth_surface_comsol_tet_vol · Point Cloud"),
        ("tooth_surface_comsol_tet_vol", True, "tooth_surface_comsol_tet_vol · Surface"),
    ]

    ent_cache: dict[str, EntityData] = {}
    payload_cache: dict[tuple[str, bool], tuple] = {}
    for name, is_surface, label in panels:
        ent = ent_cache.setdefault(name, entity_payload(name, dict(ENTITY_ORDER)[name]))
        key = (name, is_surface)
        if key not in payload_cache:
            payload_cache[key] = entity_plot_data(ent, is_surface)
    norm = shared_norm(payload_cache)

    for ax, (name, is_surface, label) in zip(axs.ravel(), panels):
        formatted_label = panel_title(label)
        pts3, tris, values = payload_cache[(name, is_surface)]
        if is_surface:
            if ventral:
                draw_surface_failure_mask(
                    ax,
                    pts3,
                    tris,
                    values,
                    norm,
                    horizontal=horizontal,
                    vertical=vertical,
                    depth_axis=depth_axis,
                    reverse_depth=reverse_depth,
                    title=formatted_label,
                )
            else:
                draw_surface(
                    ax,
                    pts3,
                    tris,
                    values,
                    norm,
                    horizontal=horizontal,
                    vertical=vertical,
                    depth_axis=depth_axis,
                    reverse_depth=reverse_depth,
                    title=formatted_label,
                )
        else:
            if ventral:
                draw_point_failure_mask(
                    ax,
                    pts3,
                    values,
                    norm,
                    horizontal=horizontal,
                    vertical=vertical,
                    depth_axis=depth_axis,
                    reverse_depth=reverse_depth,
                    title=formatted_label,
                )
            else:
                draw_point(
                    ax,
                    pts3,
                    values,
                    norm,
                    horizontal=horizontal,
                    vertical=vertical,
                    depth_axis=depth_axis,
                    reverse_depth=reverse_depth,
                    title=formatted_label,
                )

    fig.suptitle(title, fontsize=22, fontweight="bold", color="#163348", y=0.975)
    fig.text(0.5, 0.935, subtitle, fontsize=10.5, color="#446578", ha="center")
    fig.text(0.5, 0.912, f"Model: {MODEL_PATH.relative_to(PROJECT_ROOT)}", fontsize=9.5, color="#446578", ha="center")
    fig.text(0.952, 0.145, "Von Mises (Pa)", fontsize=9.5, color="#446578", rotation=90, ha="center", va="bottom")

    cax = fig.add_axes([0.935, 0.25, 0.015, 0.50])
    sm = plt.cm.ScalarMappable(cmap="turbo", norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.ax.tick_params(labelsize=8)

    fig.tight_layout(rect=[0.02, 0.03, 0.91, 0.885], h_pad=2.2, w_pad=1.8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build frontal and ventral dentis von Mises montages.")
    parser.add_argument("--frontal-out", default=str(PROJECT_ROOT / "dentis_frontal.png"))
    parser.add_argument("--ventral-out", default=str(PROJECT_ROOT / "dentis_ventral.png"))
    parser.add_argument(
        "--views",
        choices=("both", "frontal", "ventral"),
        default="both",
        help="Render both montages or only one of them.",
    )
    args = parser.parse_args()

    # These mesh-heavy arrays are almost entirely acyclic, so refcounting is
    # sufficient and avoids multi-minute GC stalls between renders.
    gc.disable()
    try:
        if args.views in {"both", "frontal"}:
            build_montage(Path(args.frontal_out).resolve(), ventral=False)
            gc.collect()
            print(Path(args.frontal_out).resolve())
        if args.views in {"both", "ventral"}:
            build_montage(Path(args.ventral_out).resolve(), ventral=True)
            gc.collect()
            print(Path(args.ventral_out).resolve())
    finally:
        gc.enable()


if __name__ == "__main__":
    main()
