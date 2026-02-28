#!/usr/bin/env python3
"""Convert a surface OBJ into COMSOL-importable tetra mesh files.

Outputs:
- <prefix>_watertight.obj
- <prefix>_watertight.stl
- <prefix>_tet.geo
- <prefix>_tet_vol.msh  (Gmsh v2.2, tetra + boundary triangles)
- <prefix>_tet_vol.bdf  (Nastran, CTRIA3 + CTETRA)
- <prefix>_params.json  (all parameters + mesh counts)
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import meshio
import numpy as np
import trimesh
from trimesh.smoothing import filter_taubin


@dataclass
class TopologyStats:
    vertices: int
    faces: int
    watertight: bool
    winding_consistent: bool
    components: int
    boundary_edges: int
    nonmanifold_edges: int
    euler_number: int


@dataclass
class GmshParams:
    characteristic_length_min: float
    characteristic_length_max: float
    algorithm_3d: int
    element_order: int
    optimize: int
    optimize_netgen: int
    num_threads: int
    msh_format: str


def cleanup_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    m = mesh.copy()
    for fn_name in ("remove_infinite_values", "remove_unreferenced_vertices", "merge_vertices"):
        fn = getattr(m, fn_name, None)
        if callable(fn):
            try:
                fn()
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
    return m


def load_mesh(path: Path) -> trimesh.Trimesh:
    m = trimesh.load(path, force="mesh", process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(tuple(m.geometry.values()))
    if not isinstance(m, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type: {type(m)}")
    return cleanup_mesh(m)


def topology_stats(mesh: trimesh.Trimesh) -> TopologyStats:
    edges = np.sort(mesh.edges.reshape(-1, 2), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return TopologyStats(
        vertices=int(len(mesh.vertices)),
        faces=int(len(mesh.faces)),
        watertight=bool(mesh.is_watertight),
        winding_consistent=bool(mesh.is_winding_consistent),
        components=int(len(mesh.split(only_watertight=False))),
        boundary_edges=int(np.sum(counts == 1)),
        nonmanifold_edges=int(np.sum(counts > 2)),
        euler_number=int(mesh.euler_number),
    )


def keep_area_coverage(mesh: trimesh.Trimesh, min_coverage: float) -> trimesh.Trimesh:
    parts = [p for p in mesh.split(only_watertight=False) if len(p.faces) > 0]
    if len(parts) <= 1:
        return mesh
    parts = sorted(parts, key=lambda p: float(p.area), reverse=True)
    total = sum(float(p.area) for p in parts)
    keep = []
    acc = 0.0
    for p in parts:
        keep.append(p)
        acc += float(p.area)
        if total <= 0 or acc / total >= min_coverage:
            break
    out = keep[0] if len(keep) == 1 else trimesh.util.concatenate(keep)
    return cleanup_mesh(out)


def voxel_watertight(mesh: trimesh.Trimesh, n_axis: int, taubin_iters: int) -> tuple[trimesh.Trimesh, float]:
    m = mesh.copy()
    if taubin_iters > 0:
        filter_taubin(m, lamb=0.45, nu=-0.53, iterations=taubin_iters)
    m = cleanup_mesh(m)

    extent = m.bounds[1] - m.bounds[0]
    pitch = float(extent.max() / float(n_axis))
    vox = m.voxelized(pitch).fill()
    wt = vox.marching_cubes
    # marching_cubes vertices are in voxel index coordinates; map back to world coordinates
    # so the exported mesh stays in the original model scale/location.
    wt.apply_transform(vox.transform)
    wt = cleanup_mesh(wt)
    try:
        trimesh.repair.fix_winding(wt)
    except Exception:
        pass
    try:
        trimesh.repair.fix_inversion(wt)
    except Exception:
        pass
    return cleanup_mesh(wt), pitch


def write_geo(path: Path, surface_stl: Path, out_msh: Path, p: GmshParams) -> None:
    geo = f'''Merge "{surface_stl}";
sl[] = Surface "*";
Surface Loop(1) = {{sl[]}};
Volume(1) = {{1}};
Mesh.CharacteristicLengthMin = {p.characteristic_length_min};
Mesh.CharacteristicLengthMax = {p.characteristic_length_max};
Mesh.Algorithm3D = {p.algorithm_3d};
Mesh.ElementOrder = {p.element_order};
Mesh.Optimize = {p.optimize};
Mesh.OptimizeNetgen = {p.optimize_netgen};
General.NumThreads = {p.num_threads};
Mesh {3};
Save "{out_msh}";
'''
    path.write_text(geo, encoding="utf-8")


def run_cmd(args: list[str]) -> None:
    subprocess.run(args, check=True)


def mesh_counts(msh_path: Path) -> dict[str, int]:
    m = meshio.read(msh_path)
    counts: dict[str, int] = {"points": int(len(m.points))}
    for block in m.cells:
        counts[block.type] = counts.get(block.type, 0) + int(len(block.data))
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="OBJ -> COMSOL importable tetra mesh")
    parser.add_argument("--input", type=Path, default=Path("tetwild_input_smoothed.obj"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--prefix", default="tetwild_input_smoothed_comsol")
    parser.add_argument("--n-axis", type=int, default=120, help="Voxel resolution on longest axis")
    parser.add_argument("--taubin-iters", type=int, default=8)
    parser.add_argument("--area-coverage", type=float, default=0.99)
    parser.add_argument("--cl-min", type=float, default=0.9)
    parser.add_argument("--cl-max", type=float, default=2.8)
    parser.add_argument("--algorithm3d", type=int, default=1, choices=[1, 4, 7, 9, 10])
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    input_path = args.input.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.prefix
    out_obj = out_dir / f"{prefix}_watertight.obj"
    out_stl = out_dir / f"{prefix}_watertight.stl"
    out_geo = out_dir / f"{prefix}_tet.geo"
    out_msh = out_dir / f"{prefix}_tet_vol.msh"
    out_bdf = out_dir / f"{prefix}_tet_vol.bdf"
    out_json = out_dir / f"{prefix}_params.json"

    raw = load_mesh(input_path)
    raw_stats = topology_stats(raw)

    cleaned = keep_area_coverage(raw, min_coverage=args.area_coverage)
    wt_mesh, pitch = voxel_watertight(cleaned, n_axis=args.n_axis, taubin_iters=args.taubin_iters)
    wt_stats = topology_stats(wt_mesh)

    wt_mesh.export(out_obj)
    wt_mesh.export(out_stl)

    gmsh_params = GmshParams(
        characteristic_length_min=float(args.cl_min),
        characteristic_length_max=float(args.cl_max),
        algorithm_3d=int(args.algorithm3d),
        element_order=1,
        optimize=1,
        optimize_netgen=1,
        num_threads=int(args.threads),
        msh_format="msh2",
    )

    write_geo(out_geo, out_stl, out_msh, gmsh_params)

    run_cmd([
        "gmsh",
        str(out_geo),
        "-3",
        "-v",
        "2",
        "-format",
        gmsh_params.msh_format,
        "-o",
        str(out_msh),
    ])

    run_cmd([
        "gmsh",
        str(out_msh),
        "-save",
        "-v",
        "2",
        "-format",
        "bdf",
        "-o",
        str(out_bdf),
    ])

    counts = mesh_counts(out_msh)

    report = {
        "input": str(input_path),
        "units_assumed": "mm",
        "preprocess": {
            "area_coverage": float(args.area_coverage),
            "taubin_lambda": 0.45,
            "taubin_nu": -0.53,
            "taubin_iterations": int(args.taubin_iters),
            "voxel_n_axis": int(args.n_axis),
            "voxel_pitch": float(pitch),
        },
        "raw_surface": asdict(raw_stats),
        "watertight_surface": asdict(wt_stats),
        "gmsh": asdict(gmsh_params),
        "outputs": {
            "watertight_obj": str(out_obj),
            "watertight_stl": str(out_stl),
            "geo_recipe": str(out_geo),
            "volume_msh": str(out_msh),
            "volume_bdf": str(out_bdf),
        },
        "mesh_counts": counts,
        "comsol_import": {
            "preferred": {
                "format": "Nastran (.bdf)",
                "file": str(out_bdf),
                "mesh_feature_type": "Import",
            },
            "alternate": {
                "format": "Gmsh v2.2 (.msh)",
                "file": str(out_msh),
                "mesh_feature_type": "Import",
            },
            "notes": [
                "Linear tetrahedra (first-order CTETRA4).",
                "Boundary triangles are preserved for boundary selections.",
                "If model units differ, set unit scale during COMSOL import.",
            ],
        },
    }

    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote: {out_obj}")
    print(f"Wrote: {out_stl}")
    print(f"Wrote: {out_geo}")
    print(f"Wrote: {out_msh}")
    print(f"Wrote: {out_bdf}")
    print(f"Wrote: {out_json}")
    print("Mesh counts:")
    for k in sorted(counts):
        print(f"  {k}: {counts[k]}")


if __name__ == "__main__":
    main()
