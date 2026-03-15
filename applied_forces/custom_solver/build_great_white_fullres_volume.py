#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import time
from pathlib import Path

import meshio
import numpy as np
import pytetwild


def parse_bdf_float(field: str) -> float:
    s = field.strip()
    if not s:
        return 0.0
    if "E" in s or "e" in s:
        return float(s)
    # Nastran fixed-width may encode exponent without E (for example 1.23+4).
    for i in range(1, len(s)):
        c = s[i]
        if c in "+-" and s[i - 1].isdigit():
            return float(s[:i] + "E" + s[i:])
    return float(s)


def boundary_faces_from_tets(tets: np.ndarray) -> np.ndarray:
    f0 = tets[:, [0, 1, 2]]
    f1 = tets[:, [0, 3, 1]]
    f2 = tets[:, [1, 3, 2]]
    f3 = tets[:, [2, 3, 0]]
    faces = np.vstack((f0, f1, f2, f3))
    sfaces = np.sort(faces, axis=1)
    _, idx, counts = np.unique(sfaces, axis=0, return_index=True, return_counts=True)
    return faces[idx[counts == 1]]


def edge_manifold_stats(tri: np.ndarray) -> tuple[int, int]:
    edges = np.vstack((tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]))
    edges = np.sort(edges, axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return int(np.sum(counts == 1)), int(np.sum(counts > 2))


def read_surface_bdf(path: Path) -> tuple[np.ndarray, np.ndarray]:
    grid_count = 0
    tri_count = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("GRID"):
                grid_count += 1
            elif line.startswith("CTRIA3"):
                tri_count += 1

    vertices = np.zeros((grid_count, 3), dtype=np.float64)
    faces = np.zeros((tri_count, 3), dtype=np.int32)
    id_to_index: dict[int, int] = {}

    i = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("GRID"):
                continue
            gid = int(line[8:16])
            x = parse_bdf_float(line[24:32])
            y = parse_bdf_float(line[32:40])
            z = parse_bdf_float(line[40:48])
            id_to_index[gid] = i
            vertices[i, 0] = x
            vertices[i, 1] = y
            vertices[i, 2] = z
            i += 1

    j = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("CTRIA3"):
                continue
            g1 = int(line[24:32])
            g2 = int(line[32:40])
            g3 = int(line[40:48])
            faces[j, 0] = id_to_index[g1]
            faces[j, 1] = id_to_index[g2]
            faces[j, 2] = id_to_index[g3]
            j += 1

    return vertices, faces


def write_bdf_from_msh(out_msh: Path, out_bdf: Path) -> str:
    gmsh = shutil.which("gmsh")
    if gmsh:
        subprocess.run(
            [gmsh, str(out_msh), "-save", "-v", "2", "-format", "bdf", "-o", str(out_bdf)],
            check=True,
        )
        return "gmsh"
    mesh = meshio.read(out_msh)
    meshio.write(out_bdf, mesh, file_format="nastran")
    return "meshio"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build full-resolution volumetric tet mesh from MBIE_White_Shark_HQ.bdf"
    )
    parser.add_argument(
        "--input-bdf",
        type=Path,
        default=Path(
            "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/custom_solver/MBIE_White_Shark_HQ.bdf"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/custom_solver"
        ),
    )
    parser.add_argument("--prefix", default="great_white_jaw_fullres")
    parser.add_argument("--edge-length-fac", type=float, default=0.0013)
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--min-boundary-ratio", type=float, default=0.98)
    args = parser.parse_args()

    t0 = time.time()
    in_bdf = args.input_bdf.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_msh = out_dir / f"{args.prefix}_tet_vol.msh"
    out_bdf = out_dir / f"{args.prefix}_tet_vol.bdf"
    out_json = out_dir / f"{args.prefix}_tet_vol_report.json"

    print(f"INPUT_BDF|{in_bdf}")
    print(f"EDGE_LENGTH_FAC|{args.edge_length_fac}")
    print(f"OPTIMIZE|{args.optimize}")
    print("LOAD_SURFACE|start")
    vertices, faces = read_surface_bdf(in_bdf)
    print(f"LOAD_SURFACE|done|vertices={len(vertices)}|triangles={len(faces)}|sec={time.time()-t0:.3f}")

    bnd_edges_in, nonman_in = edge_manifold_stats(faces)
    print(f"INPUT_MANIFOLD|boundary_edges={bnd_edges_in}|nonmanifold_edges={nonman_in}")

    tet_start = time.time()
    print("TETWILD|start")
    vt, tt = pytetwild.tetrahedralize(
        vertices,
        faces,
        optimize=bool(args.optimize),
        edge_length_fac=float(args.edge_length_fac),
    )
    tet_sec = time.time() - tet_start
    print(f"TETWILD|done|sec={tet_sec:.3f}|vertices={len(vt)}|tetra={len(tt)}")

    boundary = boundary_faces_from_tets(tt.astype(np.int32))
    bnd_edges_out, nonman_out = edge_manifold_stats(boundary)
    ratio = float(len(boundary)) / float(len(faces))
    print(
        f"BOUNDARY|triangles={len(boundary)}|ratio={ratio:.6f}|boundary_edges={bnd_edges_out}|nonmanifold_edges={nonman_out}"
    )

    if ratio < float(args.min_boundary_ratio):
        raise RuntimeError(
            f"Boundary triangle ratio {ratio:.6f} is below min-boundary-ratio {args.min_boundary_ratio:.6f}. "
            "Decrease --edge-length-fac for stricter boundary preservation."
        )

    mesh = meshio.Mesh(
        points=np.asarray(vt, dtype=np.float64),
        cells=[
            ("triangle", boundary.astype(np.int32)),
            ("tetra", tt.astype(np.int32)),
        ],
    )
    meshio.write(out_msh, mesh, file_format="gmsh22")
    bdf_writer = write_bdf_from_msh(out_msh, out_bdf)

    report = {
        "input_bdf": str(in_bdf),
        "edge_length_fac": float(args.edge_length_fac),
        "optimize": bool(args.optimize),
        "input_surface": {
            "vertices": int(len(vertices)),
            "triangles": int(len(faces)),
            "boundary_edges": int(bnd_edges_in),
            "nonmanifold_edges": int(nonman_in),
        },
        "output_volume": {
            "vertices": int(len(vt)),
            "tetrahedra": int(len(tt)),
            "boundary_triangles": int(len(boundary)),
            "boundary_ratio_vs_input": float(ratio),
            "boundary_edges": int(bnd_edges_out),
            "nonmanifold_edges": int(nonman_out),
            "tetrahedralize_seconds": float(tet_sec),
        },
        "outputs": {
            "volume_msh": str(out_msh),
            "volume_bdf": str(out_bdf),
            "bdf_writer": bdf_writer,
        },
        "total_seconds": float(time.time() - t0),
    }
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"WROTE_MSH|{out_msh}")
    print(f"WROTE_BDF|{out_bdf}")
    print(f"WROTE_REPORT|{out_json}")
    print(f"DONE|sec={time.time()-t0:.3f}")


if __name__ == "__main__":
    main()
