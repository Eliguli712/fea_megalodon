#!/usr/bin/env python3
"""Generate raw-vs-smoothed tooth mesh comparison assets and HTML."""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
from pathlib import Path

import meshio
import numpy as np
import plotly.graph_objects as go
import trimesh
from plotly.subplots import make_subplots
from trimesh.smoothing import filter_taubin


ROOT = Path(__file__).resolve().parent
EXPORTS = ROOT / "exports"
CONVERT_SCRIPT = ROOT.parent / "applied_forces" / "convert_tetwild_obj_to_comsol.py"

RAW_SURFACE_OBJ = EXPORTS / "tooth_surface_uncompressed.obj"
RAW_SURFACE_REPORT = EXPORTS / "tooth_surface_uncompressed_report.json"
SMOOTH_PREFIX = "tooth_surface_taubin_smoothed"
OUT_HTML = ROOT / "geometrics.html"
OUT_JSON = EXPORTS / "tooth_surface_raw_vs_smoothed_report.json"
OUT_SETUP_JSON = EXPORTS / "tooth_surface_raw_vs_smoothed_comsol_setup.json"


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
    mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type: {type(mesh)}")
    return cleanup_mesh(mesh)


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
        "area_mm2": float(mesh.area),
        "volume_mm3": float(abs(mesh.volume)),
    }


def geometry_report_from_reference(mesh: trimesh.Trimesh, reference: dict) -> dict:
    return {
        "vertices": int(reference["vertices"]),
        "faces": int(reference["faces"]),
        "watertight": bool(reference["watertight"]),
        "winding_consistent": bool(reference["winding_consistent"]),
        "components": int(reference["components"]),
        "boundary_edges": int(reference["boundary_edges"]),
        "nonmanifold_edges": int(reference["nonmanifold_edges"]),
        "bounds_min_xyz_mm": [float(x) for x in mesh.bounds[0]],
        "bounds_max_xyz_mm": [float(x) for x in mesh.bounds[1]],
        "area_mm2": float(mesh.area),
        "volume_mm3": float(abs(mesh.volume)),
    }


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Taubin-smoothed watertight tooth surface\n")
        for x, y, z in vertices:
            f.write(f"v {x:.17g} {y:.17g} {z:.17g}\n")
        for a, b, c in (faces + 1):
            f.write(f"f {int(a)} {int(b)} {int(c)}\n")


def write_ascii_msh(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    mesh = meshio.Mesh(
        points=np.asarray(vertices, dtype=np.float64),
        cells=[("triangle", np.asarray(faces, dtype=np.int32))],
    )
    meshio.write(path, mesh, file_format="gmsh", binary=False)


def write_matlab_surface(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    faces1 = np.asarray(faces, dtype=np.int64) + 1
    with open(path, "w", encoding="utf-8") as f:
        f.write("%% Auto-generated Taubin-smoothed tooth surface mesh\n")
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


def ensure_smoothed_surface(
    raw_mesh: trimesh.Trimesh,
    *,
    raw_reference: dict,
    output_dir: Path,
    prefix: str,
    iterations: int,
    lamb: float,
    nu: float,
    force: bool,
) -> tuple[Path, dict, np.ndarray]:
    out_obj = output_dir / f"{prefix}.obj"
    out_msh = output_dir / f"{prefix}.msh"
    out_m = output_dir / f"{prefix}.m"
    out_report = output_dir / f"{prefix}_report.json"

    if not force and out_obj.exists():
        smoothed_mesh = load_mesh(out_obj)
        displacement = np.linalg.norm(
            np.asarray(smoothed_mesh.vertices, dtype=np.float64) - np.asarray(raw_mesh.vertices, dtype=np.float64),
            axis=1,
        )
        if out_report.exists():
            report = json.loads(out_report.read_text(encoding="utf-8"))
        else:
            report = {
                "input": str(RAW_SURFACE_OBJ.resolve()),
                "smoothing": {
                    "method": "Taubin",
                    "iterations": int(iterations),
                    "lambda": float(lamb),
                    "nu": float(nu),
                },
                "mesh": geometry_report_from_reference(smoothed_mesh, raw_reference),
                "displacement_mm": summarize_vector(displacement),
                "outputs": {
                    "obj": str(out_obj.resolve()),
                    "msh": str(out_msh.resolve()),
                    "m": str(out_m.resolve()),
                    "report": str(out_report.resolve()),
                },
            }
            out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        del smoothed_mesh
        gc.collect()
        return out_obj, report, displacement

    smoothed = raw_mesh.copy()
    filter_taubin(smoothed, lamb=lamb, nu=nu, iterations=iterations)
    smoothed = cleanup_mesh(smoothed)
    try:
        trimesh.repair.fix_winding(smoothed)
    except Exception:
        pass
    try:
        trimesh.repair.fix_inversion(smoothed)
    except Exception:
        pass
    try:
        trimesh.repair.fix_normals(smoothed, multibody=True)
    except Exception:
        pass
    smoothed = cleanup_mesh(smoothed)

    vertices = np.asarray(smoothed.vertices, dtype=np.float64)
    faces = np.asarray(smoothed.faces, dtype=np.int32)
    write_obj(out_obj, vertices, faces)
    write_ascii_msh(out_msh, vertices, faces)
    write_matlab_surface(out_m, vertices, faces)

    displacement = np.linalg.norm(vertices - np.asarray(raw_mesh.vertices, dtype=np.float64), axis=1)
    report = {
        "input": str(RAW_SURFACE_OBJ.resolve()),
        "smoothing": {
            "method": "Taubin",
            "iterations": int(iterations),
            "lambda": float(lamb),
            "nu": float(nu),
        },
        "mesh": geometry_report_from_reference(smoothed, raw_reference),
        "displacement_mm": summarize_vector(displacement),
        "outputs": {
            "obj": str(out_obj.resolve()),
            "msh": str(out_msh.resolve()),
            "m": str(out_m.resolve()),
            "report": str(out_report.resolve()),
        },
    }
    out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    del smoothed
    gc.collect()
    return out_obj, report, displacement


def summarize_vector(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def sample_surface_points_edges(
    mesh: trimesh.Trimesh,
    *,
    max_points: int,
    max_faces: int,
    max_edges: int,
    seed: int = 123,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)

    if len(vertices) > max_points:
        point_idx = rng.choice(len(vertices), size=max_points, replace=False)
        points = vertices[point_idx]
    else:
        points = vertices

    if len(faces) > max_faces:
        face_idx = rng.choice(len(faces), size=max_faces, replace=False)
        sampled_faces = faces[face_idx]
    else:
        sampled_faces = faces

    edges = np.vstack(
        (
            sampled_faces[:, [0, 1]],
            sampled_faces[:, [1, 2]],
            sampled_faces[:, [2, 0]],
        )
    )
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    if len(edges) > max_edges:
        edge_idx = rng.choice(len(edges), size=max_edges, replace=False)
        edges = edges[edge_idx]
    return points, edges


def sample_mask_points(vertices: np.ndarray, mask: np.ndarray, max_points: int, seed: int = 123) -> np.ndarray:
    points = np.asarray(vertices, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    if len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=max_points, replace=False)
    return points[idx]


def edges_to_lines(vertices: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xyz = vertices[np.asarray(edges, dtype=np.int64)]
    x = np.column_stack((xyz[:, 0, 0], xyz[:, 1, 0], np.full(len(edges), np.nan))).ravel()
    y = np.column_stack((xyz[:, 0, 1], xyz[:, 1, 1], np.full(len(edges), np.nan))).ravel()
    z = np.column_stack((xyz[:, 0, 2], xyz[:, 1, 2], np.full(len(edges), np.nan))).ravel()
    return x, y, z


def sample_tet_edges(tets: np.ndarray, *, max_edges: int, seed: int = 123) -> np.ndarray:
    tet_edges = np.vstack(
        (
            tets[:, [0, 1]],
            tets[:, [0, 2]],
            tets[:, [0, 3]],
            tets[:, [1, 2]],
            tets[:, [1, 3]],
            tets[:, [2, 3]],
        )
    )
    tet_edges = np.sort(tet_edges, axis=1)
    tet_edges = np.unique(tet_edges, axis=0)
    if len(tet_edges) > max_edges:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(tet_edges), size=max_edges, replace=False)
        tet_edges = tet_edges[idx]
    return tet_edges


def compute_cutting_edge_masks(vertices: np.ndarray) -> dict:
    vertices = np.asarray(vertices, dtype=np.float64)
    z = vertices[:, 2]
    top_mask = z >= np.quantile(z, 0.945)
    crown_vertices = vertices[top_mask]
    center_xy = crown_vertices[:, :2].mean(axis=0)
    basis = crown_vertices[:, :2] - center_xy
    _, _, vh = np.linalg.svd(basis, full_matrices=False)
    axis_xy = vh[0]
    proj = (vertices[:, :2] - center_xy) @ axis_xy

    crown_proj = proj[top_mask]
    low_q = float(np.quantile(crown_proj, 0.18))
    high_q = float(np.quantile(crown_proj, 0.82))

    left_mask = top_mask & (proj <= low_q)
    right_mask = top_mask & (proj >= high_q)
    return {
        "top_quantile": 0.945,
        "split_quantiles": [0.18, 0.82],
        "principal_axis_xy": [float(axis_xy[0]), float(axis_xy[1])],
        "center_xy_mm": [float(center_xy[0]), float(center_xy[1])],
        "left_mask": left_mask,
        "right_mask": right_mask,
    }


def patch_report(
    mesh: trimesh.Trimesh,
    vertex_mask: np.ndarray,
    displacement: np.ndarray,
    *,
    label: str,
) -> dict:
    faces = np.asarray(mesh.faces, dtype=np.int32)
    face_hits = np.sum(vertex_mask[faces], axis=1)
    face_mask = face_hits >= 2
    if not np.any(face_mask):
        face_mask = face_hits >= 1
    face_index = np.flatnonzero(face_mask)
    submesh = mesh.submesh([face_index], append=True, repair=False) if len(face_index) else None

    patch_vertices = np.asarray(mesh.vertices, dtype=np.float64)[vertex_mask]
    patch_displacement = np.asarray(displacement, dtype=np.float64)[vertex_mask]

    area = float(submesh.area) if isinstance(submesh, trimesh.Trimesh) else 0.0
    sharpness_deg_mean = None
    sharpness_deg_p95 = None
    if isinstance(submesh, trimesh.Trimesh) and len(submesh.face_adjacency_angles) > 0:
        angles = np.degrees(np.abs(np.asarray(submesh.face_adjacency_angles, dtype=np.float64)))
        sharpness_deg_mean = float(np.mean(angles))
        sharpness_deg_p95 = float(np.quantile(angles, 0.95))

    if patch_vertices.size == 0:
        bounds_min = [0.0, 0.0, 0.0]
        bounds_max = [0.0, 0.0, 0.0]
        centroid = [0.0, 0.0, 0.0]
    else:
        bounds_min = [float(v) for v in patch_vertices.min(axis=0)]
        bounds_max = [float(v) for v in patch_vertices.max(axis=0)]
        centroid = [float(v) for v in patch_vertices.mean(axis=0)]

    return {
        "label": label,
        "vertices": int(np.sum(vertex_mask)),
        "faces": int(np.sum(face_mask)),
        "area_mm2": area,
        "bounds_min_xyz_mm": bounds_min,
        "bounds_max_xyz_mm": bounds_max,
        "centroid_xyz_mm": centroid,
        "displacement_mm": summarize_vector(patch_displacement),
        "sharpness_deg_mean": sharpness_deg_mean,
        "sharpness_deg_p95": sharpness_deg_p95,
    }


def padded_box(bounds_min: np.ndarray, bounds_max: np.ndarray, pad_mm: float) -> dict:
    return {
        "xmin_mm": float(bounds_min[0] - pad_mm),
        "xmax_mm": float(bounds_max[0] + pad_mm),
        "ymin_mm": float(bounds_min[1] - pad_mm),
        "ymax_mm": float(bounds_max[1] + pad_mm),
        "zmin_mm": float(bounds_min[2] - pad_mm),
        "zmax_mm": float(bounds_max[2] + pad_mm),
    }


def run_converter(
    input_obj: Path,
    *,
    prefix: str,
    output_dir: Path,
    n_axis: int,
    taubin_iters: int,
    area_coverage: float,
    cl_min: float,
    cl_max: float,
    threads: int,
) -> dict:
    if not CONVERT_SCRIPT.exists():
        raise FileNotFoundError(CONVERT_SCRIPT)
    params_json = output_dir / f"{prefix}_params.json"
    if not params_json.exists():
        cmd = [
            sys.executable,
            str(CONVERT_SCRIPT),
            "--input",
            str(input_obj),
            "--output-dir",
            str(output_dir),
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
    return json.loads(params_json.read_text(encoding="utf-8"))


def load_volume_mesh(msh_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mesh = meshio.read(msh_path)
    points = np.asarray(mesh.points, dtype=np.float64)
    triangles = []
    tetra = []
    for block in mesh.cells:
        if block.type == "triangle":
            triangles.append(np.asarray(block.data, dtype=np.int32))
        elif block.type == "tetra":
            tetra.append(np.asarray(block.data, dtype=np.int32))
    tri = np.vstack(triangles) if triangles else np.zeros((0, 3), dtype=np.int32)
    tet = np.vstack(tetra) if tetra else np.zeros((0, 4), dtype=np.int32)
    return points, tri, tet


def volume_report(points: np.ndarray, tri: np.ndarray, tet: np.ndarray) -> dict:
    return {
        "vertices": int(len(points)),
        "boundary_triangles": int(len(tri)),
        "tetrahedra": int(len(tet)),
    }


def build_comsol_setup(
    raw_surface: dict,
    smoothed_surface: dict,
    raw_left: dict,
    raw_right: dict,
    smoothed_left: dict,
    smoothed_right: dict,
) -> dict:
    raw_bounds_min = np.asarray(raw_surface["bounds_min_xyz_mm"], dtype=np.float64)
    raw_bounds_max = np.asarray(raw_surface["bounds_max_xyz_mm"], dtype=np.float64)
    smooth_bounds_min = np.asarray(smoothed_surface["bounds_min_xyz_mm"], dtype=np.float64)
    smooth_bounds_max = np.asarray(smoothed_surface["bounds_max_xyz_mm"], dtype=np.float64)
    z_span = float(raw_bounds_max[2] - raw_bounds_min[2])
    smooth_z_span = float(smooth_bounds_max[2] - smooth_bounds_min[2])
    root_zmax = float(raw_bounds_min[2] + 0.12 * z_span)
    smooth_root_zmax = float(smooth_bounds_min[2] + 0.12 * smooth_z_span)

    def edge_box(patch: dict) -> dict:
        bmin = np.asarray(patch["bounds_min_xyz_mm"], dtype=np.float64)
        bmax = np.asarray(patch["bounds_max_xyz_mm"], dtype=np.float64)
        patch_span = float(np.max(bmax - bmin)) if np.any(bmax > bmin) else 0.0
        pad = max(0.25, 0.12 * patch_span)
        return padded_box(bmin, bmax, pad)

    materials = {
        "linear_reference": {
            "E_Pa": 1.5e8,
            "nu": 0.3,
            "rho_kg_m3": 1100.0,
        },
        "mooney_rivlin_mr2": {
            "C10_Pa": 1.6e7,
            "C01_Pa": 4.0e6,
            "kappa_bulk_Pa": 2.5e8,
            "rho_kg_m3": 1100.0,
        },
        "mooney_rivlin_mr5": {
            "C10_Pa": 1.2e7,
            "C01_Pa": 3.0e6,
            "C20_Pa": 2.0e6,
            "C11_Pa": 1.5e6,
            "C02_Pa": 8.0e5,
            "kappa_bulk_Pa": 2.5e8,
            "rho_kg_m3": 1100.0,
        },
    }

    solver = {
        "stationary": {
            "geometric_nonlinearity": True,
            "direct_solver": "PARDISO",
            "relative_tolerance": 1e-3,
            "max_newton_iterations": 50,
            "load_ramp_steps": 8,
            "element_order": 1,
            "notes": [
                "Use auxiliary sweep on a load scale parameter from 0 to 1 for each pressure case.",
                "Keep first-order tetrahedra for robustness on the imported volumetric meshes.",
            ],
        },
        "dynamic": {
            "geometric_nonlinearity": True,
            "time_step_s": 2.5e-4,
            "end_time_s": 5.0e-3,
            "relative_tolerance": 5e-4,
            "rayleigh_alpha": 0.0,
            "rayleigh_beta": 2.0e-6,
            "notes": [
                "Use a smooth ramp function for the pressure load over the first millisecond.",
                "Store only every second or fourth step if result files become too large.",
            ],
        },
    }

    cases = {
        "raw": {
            "static_edge_left_mr5": "Apply follower pressure to the left cutting-edge boundary box only.",
            "static_edge_right_mr5": "Apply follower pressure to the right cutting-edge boundary box only.",
            "static_global_pressure_mr5": "Apply follower pressure to all external boundaries.",
            "dynamic_global_pressure_mr5": "Run a time-dependent pressure ramp on all external boundaries.",
        },
        "smoothed": {
            "static_edge_left_mr5": "Same load case on the smoothed geometry.",
            "static_edge_right_mr5": "Same load case on the smoothed geometry.",
            "static_global_pressure_mr5": "Same global pressure case on the smoothed geometry.",
            "dynamic_global_pressure_mr5": "Same time-dependent pressure ramp on the smoothed geometry.",
        },
    }

    return {
        "materials": materials,
        "loads": {
            "edge_pressure_Pa": 2.0e3,
            "global_pressure_Pa": 2.0e3,
        },
        "boundary_selections_mm": {
            "root_support_raw": {
                "xmin_mm": float(raw_bounds_min[0] - 0.1),
                "xmax_mm": float(raw_bounds_max[0] + 0.1),
                "ymin_mm": float(raw_bounds_min[1] - 0.1),
                "ymax_mm": float(raw_bounds_max[1] + 0.1),
                "zmin_mm": float(raw_bounds_min[2] - 0.1),
                "zmax_mm": root_zmax,
            },
            "root_support_smoothed": {
                "xmin_mm": float(smooth_bounds_min[0] - 0.1),
                "xmax_mm": float(smooth_bounds_max[0] + 0.1),
                "ymin_mm": float(smooth_bounds_min[1] - 0.1),
                "ymax_mm": float(smooth_bounds_max[1] + 0.1),
                "zmin_mm": float(smooth_bounds_min[2] - 0.1),
                "zmax_mm": smooth_root_zmax,
            },
            "left_cutting_edge_raw": edge_box(raw_left),
            "right_cutting_edge_raw": edge_box(raw_right),
            "left_cutting_edge_smoothed": edge_box(smoothed_left),
            "right_cutting_edge_smoothed": edge_box(smoothed_right),
        },
        "solver_recommendations": solver,
        "study_cases": cases,
        "notes": [
            "Material coefficients match the existing COMSOL automation in applied_forces/RunHolocasticFullBody.java.",
            "Expected lower peak stress on the smoothed model is an inference from reduced local edge sharpness, not a solved result in this script.",
        ],
    }


def html_table_rows(rows: list[tuple[str, str, str]]) -> str:
    body = []
    for a, b, c in rows:
        body.append(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td></tr>")
    return "\n".join(body)


def sharpness_text(patch: dict) -> str:
    sharpness = patch["sharpness_deg_mean"]
    sharpness_value = 0.0 if sharpness is None else float(sharpness)
    return "{:.3f} mm^2 / {:.3f} deg".format(float(patch["area_mm2"]), sharpness_value)


def render_html(report: dict, plot_html: str) -> str:
    raw = report["raw_surface"]
    smoothed = report["smoothed_surface"]
    disp = report["vertex_displacement_mm"]
    raw_vol = report["raw_comsol_volume"]
    smoothed_vol = report["smoothed_comsol_volume"]
    edge_rows = [
        (
            "Left cutting edge",
            sharpness_text(report["cutting_edges"]["raw"]["left"]),
            sharpness_text(report["cutting_edges"]["smoothed"]["left"]),
        ),
        (
            "Right cutting edge",
            sharpness_text(report["cutting_edges"]["raw"]["right"]),
            sharpness_text(report["cutting_edges"]["smoothed"]["right"]),
        ),
    ]
    static_cases = "".join(
        f"<li><code>{case}</code>: {desc}</li>"
        for case, desc in report["comsol_setup"]["study_cases"]["raw"].items()
    )
    smoothed_cases = "".join(
        f"<li><code>{case}</code>: {desc}</li>"
        for case, desc in report["comsol_setup"]["study_cases"]["smoothed"].items()
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Tooth Geometry Comparison</title>
  <style>
    :root {{
      --bg: #f4efe6;
      --ink: #1f1d1a;
      --panel: #fff9f0;
      --line: #d8c9b4;
      --accent: #8c3b2a;
      --accent-2: #1d5c63;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top right, rgba(140, 59, 42, 0.13), transparent 28%),
        radial-gradient(circle at bottom left, rgba(29, 92, 99, 0.12), transparent 24%),
        var(--bg);
      color: var(--ink);
    }}
    main {{
      max-width: 1500px;
      margin: 0 auto;
      padding: 28px 22px 40px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: clamp(2rem, 3vw, 3.1rem);
      line-height: 1.04;
      letter-spacing: -0.03em;
    }}
    p {{
      margin: 0;
      line-height: 1.45;
    }}
    .lead {{
      max-width: 1100px;
      margin-bottom: 20px;
      color: rgba(31, 29, 26, 0.82);
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px;
      margin: 18px 0 24px;
    }}
    .card {{
      background: rgba(255, 249, 240, 0.88);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 16px 16px 14px;
      box-shadow: 0 14px 36px rgba(40, 31, 19, 0.06);
      backdrop-filter: blur(8px);
    }}
    .label {{
      display: inline-block;
      margin-bottom: 6px;
      font-size: 0.8rem;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--accent);
    }}
    .big {{
      font-size: 1.6rem;
      font-weight: 700;
      letter-spacing: -0.03em;
    }}
    .sub {{
      margin-top: 6px;
      color: rgba(31, 29, 26, 0.72);
      font-size: 0.92rem;
    }}
    .section {{
      margin-top: 22px;
      padding: 18px 18px 14px;
      border-radius: 22px;
      border: 1px solid var(--line);
      background: rgba(255, 252, 246, 0.9);
    }}
    .section h2 {{
      margin: 0 0 14px;
      font-size: 1.25rem;
      letter-spacing: -0.02em;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.95rem;
    }}
    th, td {{
      text-align: left;
      padding: 10px 8px;
      border-bottom: 1px solid rgba(216, 201, 180, 0.8);
      vertical-align: top;
    }}
    th {{
      font-size: 0.82rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: rgba(31, 29, 26, 0.68);
    }}
    code {{
      font-family: "SFMono-Regular", "Menlo", monospace;
      font-size: 0.92em;
      background: rgba(140, 59, 42, 0.08);
      border-radius: 6px;
      padding: 0.12rem 0.34rem;
    }}
    .actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 14px;
    }}
    .button {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 10px 14px;
      border-radius: 999px;
      text-decoration: none;
      color: #fff7ef;
      background: var(--accent);
      font-weight: 700;
    }}
    .button.alt {{
      background: var(--accent-2);
    }}
    .two-col {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
    }}
    ul {{
      margin: 0;
      padding-left: 18px;
      line-height: 1.55;
    }}
    .plot-wrap {{
      margin-top: 22px;
      border-radius: 24px;
      overflow: hidden;
      border: 1px solid var(--line);
      background: #fff;
      box-shadow: 0 20px 40px rgba(40, 31, 19, 0.08);
    }}
  </style>
</head>
<body>
  <main>
    <h1>3050-slice tooth comparison: raw uncompressed vs Taubin-smoothed</h1>
    <p class="lead">
      The raw mesh remains the original 3050-slice uncompressed watertight surface. The smoothed mesh keeps the same
      vertex and face counts and applies Taubin smoothing only to the vertex positions. Raw and smoothed COMSOL-ready
      tetra packages are included below together with the cutting-edge selections and recommended Mooney-Rivlin setup.
    </p>

    <div class="grid">
      <article class="card">
        <div class="label">Raw Surface</div>
        <div class="big">V={raw['vertices']:,} / F={raw['faces']:,}</div>
        <div class="sub">Watertight: {str(raw['watertight']).lower()} | volume {raw['volume_mm3']:.3f} mm^3</div>
      </article>
      <article class="card">
        <div class="label">Smoothed Surface</div>
        <div class="big">V={smoothed['vertices']:,} / F={smoothed['faces']:,}</div>
        <div class="sub">Watertight: {str(smoothed['watertight']).lower()} | volume {smoothed['volume_mm3']:.3f} mm^3</div>
      </article>
      <article class="card">
        <div class="label">Taubin Displacement</div>
        <div class="big">mean {disp['mean']:.5f} mm</div>
        <div class="sub">median {disp['median']:.5f} mm | p95 {disp['p95']:.5f} mm | max {disp['max']:.5f} mm</div>
      </article>
      <article class="card">
        <div class="label">COMSOL Volumes</div>
        <div class="big">raw T={raw_vol['tetrahedra']:,}</div>
        <div class="sub">smoothed T={smoothed_vol['tetrahedra']:,} | tighter mesh import settings prepared</div>
      </article>
    </div>

    <section class="section">
      <h2>Surface And Volume Counts</h2>
      <table>
        <thead>
          <tr>
            <th>Asset</th>
            <th>Raw</th>
            <th>Smoothed</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Surface mesh</td>
            <td>V={raw['vertices']:,}, F={raw['faces']:,}</td>
            <td>V={smoothed['vertices']:,}, F={smoothed['faces']:,}</td>
          </tr>
          <tr>
            <td>COMSOL tetra volume</td>
            <td>V={raw_vol['vertices']:,}, F={raw_vol['boundary_triangles']:,}, T={raw_vol['tetrahedra']:,}</td>
            <td>V={smoothed_vol['vertices']:,}, F={smoothed_vol['boundary_triangles']:,}, T={smoothed_vol['tetrahedra']:,}</td>
          </tr>
          <tr>
            <td>Area / volume</td>
            <td>{raw['area_mm2']:.3f} mm^2 / {raw['volume_mm3']:.3f} mm^3</td>
            <td>{smoothed['area_mm2']:.3f} mm^2 / {smoothed['volume_mm3']:.3f} mm^3</td>
          </tr>
          {html_table_rows(edge_rows)}
        </tbody>
      </table>
    </section>

    <section class="section two-col">
      <div>
        <h2>Mooney-Rivlin Setup</h2>
        <ul>
          <li><code>MR2</code>: C10 = 1.6e7 Pa, C01 = 4.0e6 Pa, bulk = 2.5e8 Pa.</li>
          <li><code>MR5</code>: C10 = 1.2e7 Pa, C01 = 3.0e6 Pa, C20 = 2.0e6 Pa, C11 = 1.5e6 Pa, C02 = 8.0e5 Pa, bulk = 2.5e8 Pa.</li>
          <li>Edge-pressure and global-pressure studies use 2.0e3 Pa by default in the generated setup JSON.</li>
          <li>The expected reduction in peak stress for the smoothed case is an inference from reduced cutting-edge sharpness, not a solved COMSOL result inside this script.</li>
        </ul>
      </div>
      <div>
        <h2>Recommended Cases</h2>
        <ul>{static_cases}</ul>
        <ul style="margin-top:10px;">{smoothed_cases}</ul>
      </div>
    </section>

    <div class="actions">
      <a class="button" href="geometrics.html">Refresh This HTML</a>
      <a class="button alt" href="exports/tooth_surface_raw_vs_smoothed_report.json">Comparison Report</a>
      <a class="button alt" href="exports/tooth_surface_raw_vs_smoothed_comsol_setup.json">COMSOL Setup</a>
      <a class="button alt" href="exports/{SMOOTH_PREFIX}.obj">Smoothed OBJ</a>
    </div>

    <div class="plot-wrap">{plot_html}</div>
  </main>
</body>
</html>
"""


def build_figure(
    raw_surface_vertices: np.ndarray,
    raw_surface_points: np.ndarray,
    raw_surface_edges: np.ndarray,
    raw_left_points: np.ndarray,
    raw_right_points: np.ndarray,
    smooth_surface_vertices: np.ndarray,
    smooth_surface_points: np.ndarray,
    smooth_surface_edges: np.ndarray,
    smooth_left_points: np.ndarray,
    smooth_right_points: np.ndarray,
    raw_vol_points: np.ndarray,
    raw_vol_boundary: np.ndarray,
    raw_tet_edges: np.ndarray,
    raw_tet_centroids: np.ndarray,
    smooth_vol_points: np.ndarray,
    smooth_vol_boundary: np.ndarray,
    smooth_tet_edges: np.ndarray,
    smooth_tet_centroids: np.ndarray,
) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{"type": "scene"}, {"type": "scene"}],
            [{"type": "scene"}, {"type": "scene"}],
        ],
        subplot_titles=(
            "Raw Uncompressed Surface",
            "Taubin-Smoothed Surface",
            "Raw COMSOL Volume Mesh",
            "Smoothed COMSOL Volume Mesh",
        ),
        horizontal_spacing=0.02,
        vertical_spacing=0.06,
    )

    raw_x, raw_y, raw_z = edges_to_lines(raw_surface_vertices, raw_surface_edges)
    smooth_x, smooth_y, smooth_z = edges_to_lines(smooth_surface_vertices, smooth_surface_edges)
    raw_tx, raw_ty, raw_tz = edges_to_lines(raw_vol_points, raw_tet_edges)
    smooth_tx, smooth_ty, smooth_tz = edges_to_lines(smooth_vol_points, smooth_tet_edges)

    fig.add_trace(
        go.Scatter3d(
            x=raw_surface_points[:, 0],
            y=raw_surface_points[:, 1],
            z=raw_surface_points[:, 2],
            mode="markers",
            marker=dict(size=1.4, color="#d97706", opacity=0.62),
            name="Raw points",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=raw_x,
            y=raw_y,
            z=raw_z,
            mode="lines",
            line=dict(color="#1d4ed8", width=2),
            name="Raw edges",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=raw_left_points[:, 0],
            y=raw_left_points[:, 1],
            z=raw_left_points[:, 2],
            mode="markers",
            marker=dict(size=2.2, color="#b91c1c", opacity=0.9),
            name="Raw left cutting edge",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=raw_right_points[:, 0],
            y=raw_right_points[:, 1],
            z=raw_right_points[:, 2],
            mode="markers",
            marker=dict(size=2.2, color="#047857", opacity=0.9),
            name="Raw right cutting edge",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter3d(
            x=smooth_surface_points[:, 0],
            y=smooth_surface_points[:, 1],
            z=smooth_surface_points[:, 2],
            mode="markers",
            marker=dict(size=1.4, color="#ea580c", opacity=0.62),
            name="Smoothed points",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter3d(
            x=smooth_x,
            y=smooth_y,
            z=smooth_z,
            mode="lines",
            line=dict(color="#0f766e", width=2),
            name="Smoothed edges",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter3d(
            x=smooth_left_points[:, 0],
            y=smooth_left_points[:, 1],
            z=smooth_left_points[:, 2],
            mode="markers",
            marker=dict(size=2.2, color="#b91c1c", opacity=0.9),
            name="Smoothed left cutting edge",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter3d(
            x=smooth_right_points[:, 0],
            y=smooth_right_points[:, 1],
            z=smooth_right_points[:, 2],
            mode="markers",
            marker=dict(size=2.2, color="#047857", opacity=0.9),
            name="Smoothed right cutting edge",
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Mesh3d(
            x=raw_vol_points[:, 0],
            y=raw_vol_points[:, 1],
            z=raw_vol_points[:, 2],
            i=raw_vol_boundary[:, 0],
            j=raw_vol_boundary[:, 1],
            k=raw_vol_boundary[:, 2],
            color="#8ecae6",
            opacity=0.20,
            name="Raw enclosure",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=raw_tx,
            y=raw_ty,
            z=raw_tz,
            mode="lines",
            line=dict(color="#023047", width=1),
            name="Raw tet edges",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=raw_tet_centroids[:, 0],
            y=raw_tet_centroids[:, 1],
            z=raw_tet_centroids[:, 2],
            mode="markers",
            marker=dict(size=1.4, color="#d62828", opacity=0.28),
            name="Raw tet centroids",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Mesh3d(
            x=smooth_vol_points[:, 0],
            y=smooth_vol_points[:, 1],
            z=smooth_vol_points[:, 2],
            i=smooth_vol_boundary[:, 0],
            j=smooth_vol_boundary[:, 1],
            k=smooth_vol_boundary[:, 2],
            color="#9bd3ae",
            opacity=0.20,
            name="Smoothed enclosure",
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter3d(
            x=smooth_tx,
            y=smooth_ty,
            z=smooth_tz,
            mode="lines",
            line=dict(color="#14532d", width=1),
            name="Smoothed tet edges",
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter3d(
            x=smooth_tet_centroids[:, 0],
            y=smooth_tet_centroids[:, 1],
            z=smooth_tet_centroids[:, 2],
            mode="markers",
            marker=dict(size=1.4, color="#16a34a", opacity=0.26),
            name="Smoothed tet centroids",
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        title=(
            "3050-slice tooth comparison | "
            "raw surface and Taubin-smoothed surface with COMSOL-ready tetra volumes"
        ),
        margin=dict(l=0, r=0, b=0, t=56),
        height=1200,
        legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="left", x=0.01),
    )
    for key in ("scene", "scene2", "scene3", "scene4"):
        fig.layout[key].aspectmode = "data"
        fig.layout[key].xaxis.title = "X (mm)"
        fig.layout[key].yaxis.title = "Y (mm)"
        fig.layout[key].zaxis.title = "Z (mm)"
    return fig


def sample_centroids(points: np.ndarray, tets: np.ndarray, max_points: int, seed: int = 123) -> np.ndarray:
    centroids = points[tets].mean(axis=1)
    if len(centroids) <= max_points:
        return centroids
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(centroids), size=max_points, replace=False)
    return centroids[idx]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate raw-vs-smoothed geometry comparison HTML")
    parser.add_argument("--raw-obj", type=Path, default=RAW_SURFACE_OBJ)
    parser.add_argument("--taubin-iters", type=int, default=8)
    parser.add_argument("--taubin-lambda", type=float, default=0.45)
    parser.add_argument("--taubin-nu", type=float, default=-0.53)
    parser.add_argument("--surface-points", type=int, default=140000)
    parser.add_argument("--surface-faces", type=int, default=260000)
    parser.add_argument("--surface-edges", type=int, default=320000)
    parser.add_argument("--edge-highlight-points", type=int, default=12000)
    parser.add_argument("--tet-edges", type=int, default=180000)
    parser.add_argument("--tet-centroids", type=int, default=40000)
    parser.add_argument("--volume-n-axis", type=int, default=240)
    parser.add_argument("--volume-taubin-iters", type=int, default=0)
    parser.add_argument("--cl-min", type=float, default=0.22)
    parser.add_argument("--cl-max", type=float, default=0.55)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--force-smooth", action="store_true")
    parser.add_argument("--force-volumes", action="store_true")
    args = parser.parse_args()

    raw_obj = args.raw_obj.resolve()
    if not raw_obj.exists():
        raise FileNotFoundError(raw_obj)

    raw_mesh = load_mesh(raw_obj)
    if RAW_SURFACE_REPORT.exists():
        raw_reference = json.loads(RAW_SURFACE_REPORT.read_text(encoding="utf-8"))["mesh"]
        raw_report = geometry_report_from_reference(raw_mesh, raw_reference)
    else:
        raw_report = topology_report(raw_mesh)

    smoothed_obj, smoothed_file_report, displacement = ensure_smoothed_surface(
        raw_mesh,
        raw_reference=raw_report,
        output_dir=EXPORTS,
        prefix=SMOOTH_PREFIX,
        iterations=args.taubin_iters,
        lamb=args.taubin_lambda,
        nu=args.taubin_nu,
        force=args.force_smooth,
    )
    smoothed_mesh = load_mesh(smoothed_obj)
    smoothed_report = geometry_report_from_reference(smoothed_mesh, raw_report)

    if raw_report["vertices"] != smoothed_report["vertices"] or raw_report["faces"] != smoothed_report["faces"]:
        raise RuntimeError("Taubin-smoothed mesh must preserve vertex and face counts")

    edge_masks = compute_cutting_edge_masks(np.asarray(raw_mesh.vertices, dtype=np.float64))
    left_mask = edge_masks.pop("left_mask")
    right_mask = edge_masks.pop("right_mask")

    cutting_raw_left = patch_report(raw_mesh, left_mask, displacement, label="raw_left")
    cutting_raw_right = patch_report(raw_mesh, right_mask, displacement, label="raw_right")
    cutting_smooth_left = patch_report(smoothed_mesh, left_mask, displacement, label="smoothed_left")
    cutting_smooth_right = patch_report(smoothed_mesh, right_mask, displacement, label="smoothed_right")

    raw_volume_prefix = "tooth_surface_raw_compare_comsol"
    smooth_volume_prefix = "tooth_surface_smoothed_compare_comsol"
    if args.force_volumes:
        for prefix in (raw_volume_prefix, smooth_volume_prefix):
            for suffix in ("_watertight.obj", "_watertight.stl", "_tet.geo", "_tet_vol.msh", "_tet_vol.bdf", "_params.json"):
                path = EXPORTS / f"{prefix}{suffix}"
                if path.exists():
                    path.unlink()

    raw_params = run_converter(
        raw_obj,
        prefix=raw_volume_prefix,
        output_dir=EXPORTS,
        n_axis=args.volume_n_axis,
        taubin_iters=args.volume_taubin_iters,
        area_coverage=1.0,
        cl_min=args.cl_min,
        cl_max=args.cl_max,
        threads=args.threads,
    )
    smooth_params = run_converter(
        smoothed_obj,
        prefix=smooth_volume_prefix,
        output_dir=EXPORTS,
        n_axis=args.volume_n_axis,
        taubin_iters=args.volume_taubin_iters,
        area_coverage=1.0,
        cl_min=args.cl_min,
        cl_max=args.cl_max,
        threads=args.threads,
    )

    raw_vol_msh = Path(raw_params["outputs"]["volume_msh"]).resolve()
    smooth_vol_msh = Path(smooth_params["outputs"]["volume_msh"]).resolve()
    raw_vol_points, raw_vol_tri, raw_vol_tet = load_volume_mesh(raw_vol_msh)
    smooth_vol_points, smooth_vol_tri, smooth_vol_tet = load_volume_mesh(smooth_vol_msh)

    raw_surface_points, raw_surface_edges = sample_surface_points_edges(
        raw_mesh,
        max_points=args.surface_points,
        max_faces=args.surface_faces,
        max_edges=args.surface_edges,
    )
    smooth_surface_points, smooth_surface_edges = sample_surface_points_edges(
        smoothed_mesh,
        max_points=args.surface_points,
        max_faces=args.surface_faces,
        max_edges=args.surface_edges,
    )

    raw_left_points = sample_mask_points(raw_mesh.vertices, left_mask, args.edge_highlight_points, seed=11)
    raw_right_points = sample_mask_points(raw_mesh.vertices, right_mask, args.edge_highlight_points, seed=19)
    smooth_left_points = sample_mask_points(smoothed_mesh.vertices, left_mask, args.edge_highlight_points, seed=11)
    smooth_right_points = sample_mask_points(smoothed_mesh.vertices, right_mask, args.edge_highlight_points, seed=19)

    raw_tet_edges = sample_tet_edges(raw_vol_tet, max_edges=args.tet_edges, seed=17)
    smooth_tet_edges = sample_tet_edges(smooth_vol_tet, max_edges=args.tet_edges, seed=23)
    raw_tet_centroids = sample_centroids(raw_vol_points, raw_vol_tet, max_points=args.tet_centroids, seed=17)
    smooth_tet_centroids = sample_centroids(smooth_vol_points, smooth_vol_tet, max_points=args.tet_centroids, seed=23)

    fig = build_figure(
        np.asarray(raw_mesh.vertices, dtype=np.float64),
        raw_surface_points,
        raw_surface_edges,
        raw_left_points,
        raw_right_points,
        np.asarray(smoothed_mesh.vertices, dtype=np.float64),
        smooth_surface_points,
        smooth_surface_edges,
        smooth_left_points,
        smooth_right_points,
        raw_vol_points,
        raw_vol_tri,
        raw_tet_edges,
        raw_tet_centroids,
        smooth_vol_points,
        smooth_vol_tri,
        smooth_tet_edges,
        smooth_tet_centroids,
    )
    plot_html = fig.to_html(full_html=False, include_plotlyjs=True)

    comsol_setup = build_comsol_setup(
        raw_report,
        smoothed_report,
        cutting_raw_left,
        cutting_raw_right,
        cutting_smooth_left,
        cutting_smooth_right,
    )

    comparison = {
        "source": {
            "raw_uncompressed_obj": str(raw_obj),
            "raw_uncompressed_report": str(RAW_SURFACE_REPORT.resolve()) if RAW_SURFACE_REPORT.exists() else None,
        },
        "smoothing": smoothed_file_report["smoothing"],
        "raw_surface": raw_report,
        "smoothed_surface": smoothed_report,
        "vertex_displacement_mm": summarize_vector(displacement),
        "cutting_edge_detection": edge_masks,
        "cutting_edges": {
            "raw": {
                "left": cutting_raw_left,
                "right": cutting_raw_right,
            },
            "smoothed": {
                "left": cutting_smooth_left,
                "right": cutting_smooth_right,
            },
        },
        "raw_comsol_volume": {
            **volume_report(raw_vol_points, raw_vol_tri, raw_vol_tet),
            "params_json": str((EXPORTS / f"{raw_volume_prefix}_params.json").resolve()),
            "volume_msh": str(raw_vol_msh),
            "volume_bdf": str(Path(raw_params["outputs"]["volume_bdf"]).resolve()),
            "watertight_obj": str(Path(raw_params["outputs"]["watertight_obj"]).resolve()),
        },
        "smoothed_comsol_volume": {
            **volume_report(smooth_vol_points, smooth_vol_tri, smooth_vol_tet),
            "params_json": str((EXPORTS / f"{smooth_volume_prefix}_params.json").resolve()),
            "volume_msh": str(smooth_vol_msh),
            "volume_bdf": str(Path(smooth_params["outputs"]["volume_bdf"]).resolve()),
            "watertight_obj": str(Path(smooth_params["outputs"]["watertight_obj"]).resolve()),
        },
        "comsol_setup": comsol_setup,
        "html": str(OUT_HTML.resolve()),
    }

    OUT_SETUP_JSON.write_text(json.dumps(comsol_setup, indent=2), encoding="utf-8")
    OUT_JSON.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    OUT_HTML.write_text(render_html(comparison, plot_html), encoding="utf-8")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
