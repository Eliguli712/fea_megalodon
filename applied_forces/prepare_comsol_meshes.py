#!/usr/bin/env python3
"""Make surface meshes COMSOL-friendly (watertight OBJ/STL).

Pipeline uses the same core ideas as teeth_analysis.ipynb:
1) Taubin smoothing
2) Voxel-based remeshing

The voxel step is extended to an explicit watertight reconstruction via
marching cubes over a filled voxel volume.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import trimesh
from trimesh.smoothing import filter_taubin


@dataclass
class MeshStats:
    vertices: int
    faces: int
    watertight: bool
    winding_consistent: bool
    components: int
    boundary_edges: int
    nonmanifold_edges: int
    euler_number: int


def _cleanup_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Clean mesh without deprecated trimesh APIs."""
    m = mesh.copy()
    try:
        m.remove_infinite_values()
    except Exception:
        pass
    try:
        m.remove_unreferenced_vertices()
    except Exception:
        pass
    try:
        m.update_faces(m.unique_faces())
    except Exception:
        pass
    try:
        m.update_faces(m.nondegenerate_faces())
    except Exception:
        pass
    try:
        m.remove_unreferenced_vertices()
    except Exception:
        pass
    try:
        m.merge_vertices()
    except Exception:
        pass
    return m


def _load_mesh(path: Path) -> trimesh.Trimesh:
    m = trimesh.load(path, force="mesh", process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(tuple(m.geometry.values()))
    if not isinstance(m, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type for {path}: {type(m)}")
    return _cleanup_mesh(m)


def _topology_stats(mesh: trimesh.Trimesh) -> MeshStats:
    edges = np.sort(mesh.edges.reshape(-1, 2), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = int(np.sum(counts == 1))
    nonmanifold = int(np.sum(counts > 2))
    return MeshStats(
        vertices=int(len(mesh.vertices)),
        faces=int(len(mesh.faces)),
        watertight=bool(mesh.is_watertight),
        winding_consistent=bool(mesh.is_winding_consistent),
        components=int(len(mesh.split(only_watertight=False))),
        boundary_edges=boundary,
        nonmanifold_edges=nonmanifold,
        euler_number=int(mesh.euler_number),
    )


def _keep_area_coverage(mesh: trimesh.Trimesh, min_coverage: float = 0.95) -> trimesh.Trimesh:
    """Keep connected components that cover at least `min_coverage` of area."""
    comps = [c for c in mesh.split(only_watertight=False) if len(c.faces) > 0]
    if len(comps) <= 1:
        return mesh

    comps = sorted(comps, key=lambda c: float(c.area), reverse=True)
    total_area = sum(float(c.area) for c in comps)
    keep: list[trimesh.Trimesh] = []
    acc = 0.0
    for c in comps:
        keep.append(c)
        acc += float(c.area)
        if total_area <= 0.0 or (acc / total_area) >= min_coverage:
            break

    out = keep[0] if len(keep) == 1 else trimesh.util.concatenate(keep)
    return _cleanup_mesh(out)


def _taubin_like_notebook(mesh: trimesh.Trimesh, iterations: int = 15) -> trimesh.Trimesh:
    """Taubin smoothing similar to teeth_analysis.ipynb."""
    m = mesh.copy()
    filter_taubin(m, lamb=0.45, nu=-0.53, iterations=iterations)
    return _cleanup_mesh(m)


def _voxel_watertight(mesh: trimesh.Trimesh, n_axis: int) -> tuple[trimesh.Trimesh, float]:
    """Voxelize + fill + marching cubes to force watertight topology."""
    extent = mesh.bounds[1] - mesh.bounds[0]
    pitch = float(extent.max() / float(n_axis))
    vox = mesh.voxelized(pitch)
    vox = vox.fill()
    wt = vox.marching_cubes
    wt = _cleanup_mesh(wt)
    try:
        trimesh.repair.fix_winding(wt)
    except Exception:
        pass
    try:
        trimesh.repair.fix_inversion(wt)
    except Exception:
        pass
    return _cleanup_mesh(wt), pitch


def _process_one(input_path: Path, out_dir: Path, n_axis: int) -> dict:
    mesh0 = _load_mesh(input_path)
    stats_raw = _topology_stats(mesh0)

    # Remove tiny disconnected fragments before final reconstruction.
    mesh1 = _keep_area_coverage(mesh0, min_coverage=0.95)
    mesh2 = _taubin_like_notebook(mesh1, iterations=15)
    mesh3, pitch = _voxel_watertight(mesh2, n_axis=n_axis)
    stats_final = _topology_stats(mesh3)

    stem = input_path.stem
    out_obj = out_dir / f"{stem}_comsol_watertight.obj"
    out_stl = out_dir / f"{stem}_comsol_watertight.stl"
    mesh3.export(out_obj)
    mesh3.export(out_stl)

    return {
        "input": str(input_path),
        "n_axis": int(n_axis),
        "pitch": pitch,
        "raw": asdict(stats_raw),
        "final": asdict(stats_final),
        "outputs": {
            "obj": str(out_obj),
            "stl": str(out_stl),
        },
    }


def main() -> None:
    root = Path(__file__).resolve().parent
    tasks: Iterable[tuple[Path, int]] = (
        (root / "tooth_from_dicom_seg_fullres.obj", 220),
        (root / "__tracked_surface.stl", 200),
    )

    results = []
    for path, n_axis in tasks:
        if not path.exists():
            raise FileNotFoundError(f"Missing input mesh: {path}")
        results.append(_process_one(path, root, n_axis=n_axis))

    report = root / "comsol_mesh_prep_report.json"
    report.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote report: {report}")
    for item in results:
        fin = item["final"]
        print(
            f"{Path(item['input']).name}: "
            f"watertight={fin['watertight']} "
            f"boundary_edges={fin['boundary_edges']} "
            f"nonmanifold_edges={fin['nonmanifold_edges']}"
        )


if __name__ == "__main__":
    main()

