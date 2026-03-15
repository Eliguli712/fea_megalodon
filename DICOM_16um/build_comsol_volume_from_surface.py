#!/usr/bin/env python3
"""Build COMSOL-importable volume meshes from the concatenated DICOM surface."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import meshio
import numpy as np
import pydicom
import SimpleITK as sitk
import trimesh


ROOT = Path(__file__).resolve().parents[1]
CONVERT_SCRIPT = ROOT / "applied_forces" / "convert_tetwild_obj_to_comsol.py"


def resolve_source_dir(dicom_dir: Path) -> Path:
    candidates = [
        dicom_dir / "_unzipped_dicom_tmp",
        dicom_dir / "exports" / "_unzipped_dicom_tmp",
    ]
    best_root = None
    best_count = -1
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        count = sum(1 for p in candidate.rglob("*") if p.is_file() and p.stat().st_size > 0)
        if count > best_count:
            best_root = candidate
            best_count = count
    if best_root is not None:
        return best_root.resolve()
    joined = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not find _unzipped_dicom_tmp in:\n{joined}")


def dicom_order_report(source_dir: Path) -> dict:
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(source_dir))
    if not series_ids:
        raise RuntimeError(f"No DICOM series found in {source_dir}")

    files = list(reader.GetGDCMSeriesFileNames(str(source_dir), series_ids[0]))
    instance_numbers: list[int] = []
    z_positions: list[float] = []
    for file_path in files:
        ds = pydicom.dcmread(file_path, stop_before_pixels=True, force=True)
        instance_numbers.append(int(getattr(ds, "InstanceNumber", -1)))
        ipp = getattr(ds, "ImagePositionPatient", None)
        z_positions.append(float(ipp[2]) if ipp is not None else float("nan"))

    inst = np.asarray(instance_numbers, dtype=np.int64)
    z = np.asarray(z_positions, dtype=np.float64)
    inst_d = np.diff(inst)
    z_d = np.diff(z)
    median_z_step = float(np.nanmedian(z_d)) if len(z_d) else float("nan")

    gaps = []
    for i, (di, dz) in enumerate(zip(inst_d, z_d)):
        if int(di) != 1 or abs(float(dz) - median_z_step) > 1e-6:
            gaps.append(
                {
                    "index": int(i),
                    "from_file": Path(files[i]).name,
                    "to_file": Path(files[i + 1]).name,
                    "instance_delta": int(di),
                    "z_delta_mm": float(dz),
                }
            )

    return {
        "source_dir": str(source_dir),
        "series_id": str(series_ids[0]),
        "file_count": int(len(files)),
        "instance_monotonic": bool(np.all(inst_d > 0)) if len(inst_d) else True,
        "z_monotonic": bool(np.all(z_d > 0)) if len(z_d) else True,
        "median_z_step_mm": median_z_step,
        "gap_events": gaps,
        "first_file": Path(files[0]).name if files else None,
        "last_file": Path(files[-1]).name if files else None,
        "first_instance": int(inst[0]) if len(inst) else None,
        "last_instance": int(inst[-1]) if len(inst) else None,
    }


def surface_report(surface_obj: Path) -> tuple[trimesh.Trimesh, dict]:
    mesh = trimesh.load(surface_obj, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type: {type(mesh)}")

    edges = np.sort(mesh.edges.reshape(-1, 2), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    report = {
        "path": str(surface_obj),
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "components": int(len(mesh.split(only_watertight=False))),
        "boundary_edges": int(np.sum(counts == 1)),
        "nonmanifold_edges": int(np.sum(counts > 2)),
        "bounds_min_xyz_mm": [float(x) for x in mesh.bounds[0]],
        "bounds_max_xyz_mm": [float(x) for x in mesh.bounds[1]],
    }
    return mesh, report


def write_surface_msh_and_m(mesh: trimesh.Trimesh, out_msh: Path, out_m: Path) -> None:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)

    meshio.write(
        out_msh,
        meshio.Mesh(points=vertices, cells=[("triangle", faces)]),
        file_format="gmsh",
        binary=False,
    )

    faces1 = faces.astype(np.int64) + 1
    with open(out_m, "w", encoding="utf-8") as f:
        f.write("%% Auto-generated watertight surface mesh\n")
        f.write(f"%% V: {vertices.shape[0]}x3 vertices (double)\n")
        f.write(f"%% F: {faces.shape[0]}x3 faces (1-based)\n\n")
        f.write("V = [\n")
        for x, y, z in vertices:
            f.write(f"  {x:.17g} {y:.17g} {z:.17g};\n")
        f.write("];\n\n")
        f.write("F = [\n")
        for a, b, c in faces1:
            f.write(f"  {int(a)} {int(b)} {int(c)};\n")
        f.write("];\n\n")
        f.write("%% quick view:\n")
        f.write(
            "%% trisurf(F, V(:,1), V(:,2), V(:,3), 'EdgeColor', 'none'); axis equal; camlight; lighting gouraud;\n"
        )


def run_converter(
    input_obj: Path,
    out_dir: Path,
    prefix: str,
    *,
    n_axis: int,
    taubin_iters: int,
    area_coverage: float,
    cl_min: float,
    cl_max: float,
    threads: int,
) -> None:
    cmd = [
        sys.executable,
        str(CONVERT_SCRIPT),
        "--input",
        str(input_obj),
        "--output-dir",
        str(out_dir),
        "--prefix",
        prefix,
        "--n-axis",
        str(n_axis),
        "--taubin-iters",
        str(taubin_iters),
        "--area-coverage",
        str(area_coverage),
        "--cl-min",
        str(cl_min),
        "--cl-max",
        str(cl_max),
        "--threads",
        str(threads),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build COMSOL volume meshes from DICOM surface output")
    parser.add_argument("--dicom-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--surface-obj",
        type=Path,
        default=Path(__file__).resolve().parent / "exports" / "tooth_surface_watertight.obj",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "exports")
    parser.add_argument("--prefix", default="tooth_surface_comsol")
    parser.add_argument("--n-axis", type=int, default=180)
    parser.add_argument("--taubin-iters", type=int, default=4)
    parser.add_argument("--area-coverage", type=float, default=1.0)
    parser.add_argument("--cl-min", type=float, default=0.3)
    parser.add_argument("--cl-max", type=float, default=0.9)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    dicom_dir = args.dicom_dir.resolve()
    surface_obj = args.surface_obj.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix

    if not CONVERT_SCRIPT.exists():
        raise FileNotFoundError(CONVERT_SCRIPT)
    if not surface_obj.exists():
        raise FileNotFoundError(surface_obj)

    source_dir = resolve_source_dir(dicom_dir)
    order = dicom_order_report(source_dir)
    surface_mesh, surface = surface_report(surface_obj)

    run_converter(
        surface_obj,
        out_dir,
        prefix,
        n_axis=args.n_axis,
        taubin_iters=args.taubin_iters,
        area_coverage=args.area_coverage,
        cl_min=args.cl_min,
        cl_max=args.cl_max,
        threads=args.threads,
    )

    voxel_surface_obj = out_dir / f"{prefix}_watertight.obj"
    voxel_surface_msh = out_dir / f"{prefix}_watertight_surface.msh"
    voxel_surface_m = out_dir / f"{prefix}_watertight_surface.m"
    voxel_mesh, voxel_surface = surface_report(voxel_surface_obj)
    write_surface_msh_and_m(voxel_mesh, voxel_surface_msh, voxel_surface_m)

    report = {
        "dicom_order": order,
        "concatenated_surface": surface,
        "voxel_surface": voxel_surface,
        "comsol_volume_outputs": {
            "voxel_surface_obj": str(voxel_surface_obj),
            "voxel_surface_msh": str(voxel_surface_msh),
            "voxel_surface_m": str(voxel_surface_m),
            "volume_msh": str(out_dir / f"{prefix}_tet_vol.msh"),
            "volume_bdf": str(out_dir / f"{prefix}_tet_vol.bdf"),
            "gmsh_geo": str(out_dir / f"{prefix}_tet.geo"),
            "params_json": str(out_dir / f"{prefix}_params.json"),
        },
        "notes": [
            "The concatenated surface is the full merged surface from the ordered DICOM series.",
            "The COMSOL volume mesh is built from a voxel-watertight surface derived from that concatenated surface.",
            "COMSOL import in this repo uses volume .bdf or volume .msh; .db is not an existing mesh format here.",
        ],
    }

    report_path = out_dir / f"{prefix}_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
