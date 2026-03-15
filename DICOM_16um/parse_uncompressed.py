#!/usr/bin/env python3
"""Parse a surface mesh and write uncompressed OBJ/Gmsh/MATLAB exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import meshio
import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parent
EXPORTS = ROOT / "exports"


def boundary_faces_from_tets(tets: np.ndarray) -> np.ndarray:
    f0 = tets[:, [0, 1, 2]]
    f1 = tets[:, [0, 3, 1]]
    f2 = tets[:, [1, 3, 2]]
    f3 = tets[:, [2, 3, 0]]
    faces = np.vstack((f0, f1, f2, f3))
    sorted_faces = np.sort(faces, axis=1)
    _, idx, counts = np.unique(sorted_faces, axis=0, return_index=True, return_counts=True)
    return faces[idx[counts == 1]]


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


def load_surface_mesh(path: Path) -> trimesh.Trimesh:
    if path.suffix.lower() == ".msh":
        mesh = meshio.read(path)
        vertices = np.asarray(mesh.points, dtype=np.float64)
        tri_blocks = [np.asarray(block.data, dtype=np.int32) for block in mesh.cells if block.type == "triangle"]
        if tri_blocks:
            faces = np.vstack(tri_blocks)
        else:
            tet_blocks = [np.asarray(block.data, dtype=np.int32) for block in mesh.cells if block.type == "tetra"]
            if not tet_blocks:
                raise RuntimeError(f"No triangle or tetra cells found in {path}")
            faces = boundary_faces_from_tets(np.vstack(tet_blocks))
        out = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    else:
        out = trimesh.load(path, force="mesh", process=False)
        if isinstance(out, trimesh.Scene):
            out = trimesh.util.concatenate(tuple(out.geometry.values()))
        if not isinstance(out, trimesh.Trimesh):
            raise TypeError(f"Unsupported mesh type: {type(out)}")
    return cleanup_mesh(out)


def boundary_edge_stats(faces: np.ndarray) -> tuple[int, int]:
    if len(faces) == 0:
        return 0, 0
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges = np.sort(edges, axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return int(np.sum(counts == 1)), int(np.sum(counts > 2))


def topology_report(mesh: trimesh.Trimesh) -> dict:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    boundary_edges, nonmanifold_edges = boundary_edge_stats(faces)
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "components": int(len(mesh.split(only_watertight=False))),
        "boundary_edges": int(boundary_edges),
        "nonmanifold_edges": int(nonmanifold_edges),
        "bounds_min_xyz_mm": [float(x) for x in mesh.bounds[0]],
        "bounds_max_xyz_mm": [float(x) for x in mesh.bounds[1]],
    }


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("# uncompressed surface mesh\n")
        for x, y, z in vertices:
            f.write(f"v {x:.17g} {y:.17g} {z:.17g}\n")
        for a, b, c in (faces + 1):
            f.write(f"f {int(a)} {int(b)} {int(c)}\n")


def write_ascii_msh(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    meshio.write(
        path,
        meshio.Mesh(
            points=np.asarray(vertices, dtype=np.float64),
            cells=[("triangle", np.asarray(faces, dtype=np.int32))],
        ),
        file_format="gmsh",
        binary=False,
    )


def write_matlab_surface(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    faces1 = np.asarray(faces, dtype=np.int64) + 1
    with open(path, "w", encoding="utf-8") as f:
        f.write("%% Auto-generated uncompressed surface mesh\n")
        f.write(f"%% V: {vertices.shape[0]}x3 vertices (double)\n")
        f.write(f"%% F: {faces.shape[0]}x3 faces (1-based)\n\n")
        f.write("V = [\n")
        for x, y, z in np.asarray(vertices, dtype=np.float64):
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


def read_gmsh_header(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        line0 = f.readline().strip()
        line1 = f.readline().strip()
    if line0 != "$MeshFormat":
        raise RuntimeError(f"{path} is not a Gmsh file")
    if line1 != "4.1 0 8":
        raise RuntimeError(f"Expected ASCII Gmsh 4.1 header '4.1 0 8', got '{line1}'")
    return line1


def validate_expectations(report: dict, args: argparse.Namespace) -> None:
    checks = [
        ("vertices", args.expected_vertices),
        ("faces", args.expected_faces),
        ("components", args.expected_components),
    ]
    for key, expected in checks:
        if expected is not None and report[key] != expected:
            raise RuntimeError(f"Expected {key}={expected}, got {report[key]}")
    if args.require_watertight and not report["watertight"]:
        raise RuntimeError("Expected a watertight surface mesh")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write uncompressed OBJ/Gmsh/MATLAB surface exports")
    parser.add_argument(
        "--input",
        type=Path,
        default=EXPORTS / "tooth_surface_watertight.obj",
        help="Input surface mesh (.obj or .msh). If .msh contains only tetrahedra, its boundary is extracted.",
    )
    parser.add_argument("--output-dir", type=Path, default=EXPORTS)
    parser.add_argument("--prefix", default="tooth_surface_uncompressed")
    parser.add_argument("--expected-vertices", type=int)
    parser.add_argument("--expected-faces", type=int)
    parser.add_argument("--expected-components", type=int)
    parser.add_argument("--require-watertight", action="store_true")
    args = parser.parse_args()

    input_path = args.input.resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh = load_surface_mesh(input_path)
    try:
        trimesh.repair.fix_normals(mesh, multibody=True)
    except Exception:
        pass
    report = topology_report(mesh)
    validate_expectations(report, args)

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)

    out_obj = out_dir / f"{args.prefix}.obj"
    out_msh = out_dir / f"{args.prefix}.msh"
    out_m = out_dir / f"{args.prefix}.m"
    out_json = out_dir / f"{args.prefix}_report.json"

    write_obj(out_obj, vertices, faces)
    write_ascii_msh(out_msh, vertices, faces)
    write_matlab_surface(out_m, vertices, faces)
    gmsh_header = read_gmsh_header(out_msh)

    summary = {
        "input": str(input_path),
        "mesh": report,
        "outputs": {
            "obj": str(out_obj),
            "msh": str(out_msh),
            "m": str(out_m),
            "report": str(out_json),
        },
        "gmsh_ascii_header": gmsh_header,
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
