#!/usr/bin/env python3
"""Reconstruct the full DICOM stack in ordered sections and concatenate them."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from pathlib import Path

import meshio
import numpy as np
import pydicom
import trimesh
from scipy import ndimage as ndi
from skimage import filters, morphology
from trimesh.smoothing import filter_taubin


ROOT = Path(__file__).resolve().parent
EXPORTS = ROOT / "exports"
DICOM_MESH_SCRIPT = ROOT / "dicom16um_to_mesh.py"


def load_dicom_mesh_module():
    if not DICOM_MESH_SCRIPT.exists():
        raise FileNotFoundError(DICOM_MESH_SCRIPT)
    spec = importlib.util.spec_from_file_location("dicom16um_to_mesh_sectioned", DICOM_MESH_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"Could not load module from {DICOM_MESH_SCRIPT}")
    spec.loader.exec_module(module)
    return module


dicom_mesh = load_dicom_mesh_module()


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("# sectioned watertight surface extracted from DICOM mask\n")
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
        f.write("%% Auto-generated sectioned watertight surface mesh\n")
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


def keep_largest_component(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, dict]:
    parts = [cleanup_mesh(part) for part in mesh.split(only_watertight=False) if len(part.faces) > 0]
    if not parts:
        return cleanup_mesh(mesh), {"components_before_keep_largest": 0, "removed_components": 0}
    parts.sort(key=lambda part: len(part.faces), reverse=True)
    largest = cleanup_mesh(parts[0])
    return largest, {
        "components_before_keep_largest": int(len(parts)),
        "removed_components": int(max(0, len(parts) - 1)),
        "largest_component_faces": int(len(largest.faces)),
        "largest_component_vertices": int(len(largest.vertices)),
    }


def voxel_watertight(mesh: trimesh.Trimesh, n_axis: int, taubin_iters: int) -> tuple[trimesh.Trimesh, float]:
    m = cleanup_mesh(mesh)
    if taubin_iters > 0:
        filter_taubin(m, lamb=0.45, nu=-0.53, iterations=taubin_iters)
    extent = m.bounds[1] - m.bounds[0]
    pitch = float(extent.max() / float(n_axis))
    vox = m.voxelized(pitch).fill()
    wt = vox.marching_cubes
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
    try:
        trimesh.repair.fix_normals(wt, multibody=True)
    except Exception:
        pass
    return cleanup_mesh(wt), pitch


def resolve_source_dir() -> Path:
    candidates = [
        ROOT / "exports" / "_unzipped_dicom_tmp",
        ROOT / "_unzipped_dicom_tmp",
    ]
    best_root = None
    best_count = -1
    for root in candidates:
        if not root.is_dir():
            continue
        count = sum(1 for p in root.rglob("*") if p.is_file() and p.stat().st_size > 0)
        if count > best_count:
            best_root = root
            best_count = count
    if best_root is None:
        joined = "\n".join(str(p) for p in candidates)
        raise FileNotFoundError(f"Could not find extracted DICOM root in:\n{joined}")
    return best_root.resolve()


def collect_ordered_files(source_dir: Path) -> tuple[list[Path], dict]:
    valid = []
    invalid = []
    for path in sorted(p for p in source_dir.rglob("*") if p.is_file()):
        if path.stat().st_size <= 0:
            invalid.append({"file": path.name, "reason": "empty"})
            continue
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
            inst = getattr(ds, "InstanceNumber", None)
            ipp = getattr(ds, "ImagePositionPatient", None)
            rows = getattr(ds, "Rows", None)
            cols = getattr(ds, "Columns", None)
            if inst is None or ipp is None or rows is None or cols is None:
                invalid.append({"file": path.name, "reason": "missing_required_header"})
                continue
            valid.append(
                {
                    "path": path,
                    "instance": int(inst),
                    "z": float(ipp[2]),
                    "rows": int(rows),
                    "cols": int(cols),
                    "pixel_spacing": tuple(float(v) for v in getattr(ds, "PixelSpacing", [0.018, 0.018])),
                    "slice_thickness": float(getattr(ds, "SliceThickness", 0.016) or 0.016),
                }
            )
        except Exception as exc:
            invalid.append({"file": path.name, "reason": f"{type(exc).__name__}: {exc}"})
    valid.sort(key=lambda item: (item["z"], item["instance"], item["path"].name))
    if not valid:
        raise RuntimeError(f"No valid DICOM files found in {source_dir}")
    first = valid[0]
    spacing_xyz = dicom_mesh.enforce_spacing_16um(
        (first["pixel_spacing"][1], first["pixel_spacing"][0], first["slice_thickness"]),
        z_um=16.0,
    )
    inst = np.asarray([item["instance"] for item in valid], dtype=np.int64)
    z = np.asarray([item["z"] for item in valid], dtype=np.float64)
    report = {
        "source_dir": str(source_dir),
        "valid_files": int(len(valid)),
        "invalid_files": invalid,
        "instance_min": int(inst[0]),
        "instance_max": int(inst[-1]),
        "median_z_step_mm": float(np.median(np.diff(z))) if len(z) > 1 else None,
        "gap_events": [
            {
                "index": int(i),
                "from_file": valid[i]["path"].name,
                "to_file": valid[i + 1]["path"].name,
                "instance_delta": int(inst[i + 1] - inst[i]),
                "z_delta_mm": float(z[i + 1] - z[i]),
            }
            for i in range(len(valid) - 1)
            if (inst[i + 1] - inst[i] != 1) or abs((z[i + 1] - z[i]) - np.median(np.diff(z))) > 1e-6
        ],
        "spacing_xyz_mm": [float(v) for v in spacing_xyz],
        "rows": int(first["rows"]),
        "cols": int(first["cols"]),
    }
    return [item["path"] for item in valid], report


def read_raw_slice(file_path: Path) -> np.ndarray:
    ds = pydicom.dcmread(str(file_path), force=True)
    return np.asarray(ds.pixel_array)


def compute_coarse_bbox_and_threshold(
    files: list[Path],
    *,
    downsample_xy: int = 8,
    margin_zyx: tuple[int, int, int] = (8, 4, 4),
) -> tuple[tuple[slice, slice, slice], float, dict]:
    sample = read_raw_slice(files[0])
    rows, cols = sample.shape
    small_rows = (rows + downsample_xy - 1) // downsample_xy
    small_cols = (cols + downsample_xy - 1) // downsample_xy
    coarse = np.empty((len(files), small_rows, small_cols), dtype=np.float32)
    for i, file_path in enumerate(files):
        coarse[i] = read_raw_slice(file_path)[::downsample_xy, ::downsample_xy][:small_rows, :small_cols]
        if i % 400 == 0:
            print(f"[coarse] loaded {i}/{len(files)}")

    coarse_mask = dicom_mesh.rough_mask(coarse)
    valid_values = coarse[np.isfinite(coarse)]
    lo, hi = np.percentile(valid_values, [5.0, 99.5])
    coarse_clipped = np.clip(coarse, lo, hi)
    threshold = float(filters.threshold_otsu(coarse_clipped.reshape(-1)))
    coords = np.argwhere(coarse_mask)
    if coords.size == 0:
        raise RuntimeError("Coarse segmentation produced an empty mask.")

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    mins = np.maximum(mins - np.asarray(margin_zyx, dtype=np.int64), 0)
    maxs = np.minimum(maxs + np.asarray(margin_zyx, dtype=np.int64), np.asarray(coarse_mask.shape, dtype=np.int64))
    bbox_full = (
        slice(int(mins[0]), int(maxs[0])),
        slice(int(mins[1] * downsample_xy), int(min(rows, maxs[1] * downsample_xy))),
        slice(int(mins[2] * downsample_xy), int(min(cols, maxs[2] * downsample_xy))),
    )

    info = {
        "downsample_xy": int(downsample_xy),
        "clip_percentiles": [float(lo), float(hi)],
        "otsu_threshold_raw_units": float(threshold),
        "bbox_full_zyx": [[int(s.start), int(s.stop)] for s in bbox_full],
        "roi_shape_zyx": [
            int(bbox_full[0].stop - bbox_full[0].start),
            int(bbox_full[1].stop - bbox_full[1].start),
            int(bbox_full[2].stop - bbox_full[2].start),
        ],
    }
    del coarse, coarse_clipped, coarse_mask, coords, valid_values
    gc.collect()
    return bbox_full, threshold, info


def section_ranges(length: int, section_depth: int, overlap: int) -> list[tuple[int, int]]:
    if overlap >= section_depth:
        raise ValueError("overlap must be smaller than section_depth")
    step = section_depth - overlap
    ranges = []
    start = 0
    while start < length:
        stop = min(length, start + section_depth)
        ranges.append((start, stop))
        if stop >= length:
            break
        start += step
    return ranges


def trim_section_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    *,
    abs_start: int,
    abs_stop: int,
    keep_abs_start: int,
    keep_abs_stop: int,
) -> tuple[np.ndarray, np.ndarray]:
    centroids_z = vertices[faces].mean(axis=1)[:, 2]
    low = keep_abs_start * spacing_xyz[2]
    high = keep_abs_stop * spacing_xyz[2]
    mask = (centroids_z >= low) & (centroids_z < high)
    faces = faces[mask]
    if len(faces) == 0:
        return vertices, faces
    used = np.unique(faces.reshape(-1))
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    vertices = vertices[used]
    faces = remap[faces]
    return vertices, faces.astype(np.int32)


def reconstruct_sections(
    files: list[Path],
    bbox_full: tuple[slice, slice, slice],
    threshold: float,
    spacing_xyz: tuple[float, float, float],
    *,
    section_depth: int = 192,
    overlap: int = 24,
    out_dir: Path,
    reuse_sections: bool = False,
) -> tuple[trimesh.Trimesh, list[dict]]:
    z_range = bbox_full[0].stop - bbox_full[0].start
    y_slice = bbox_full[1]
    x_slice = bbox_full[2]
    y_offset_mm = y_slice.start * spacing_xyz[1]
    x_offset_mm = x_slice.start * spacing_xyz[0]

    section_meshes = []
    section_reports: list[dict] = []
    sections_dir = out_dir / "ordered_sections"
    sections_dir.mkdir(parents=True, exist_ok=True)

    ranges = section_ranges(z_range, section_depth=section_depth, overlap=overlap)
    for idx, (start_local, stop_local) in enumerate(ranges):
        abs_start = bbox_full[0].start + start_local
        abs_stop = bbox_full[0].start + stop_local
        keep_abs_start = abs_start if idx == 0 else abs_start + overlap // 2
        keep_abs_stop = abs_stop if idx == len(ranges) - 1 else abs_stop - overlap // 2
        section_obj = sections_dir / f"section_{idx:03d}_{abs_start:05d}_{abs_stop:05d}.obj"
        section_msh = sections_dir / f"section_{idx:03d}_{abs_start:05d}_{abs_stop:05d}.msh"

        if reuse_sections and section_obj.exists():
            mesh = cleanup_mesh(trimesh.load(section_obj, force="mesh", process=False))
            section_meshes.append(mesh)
            section_reports.append(
                {
                    "section_index": int(idx),
                    "abs_slice_range": [int(abs_start), int(abs_stop)],
                    "keep_abs_slice_range": [int(keep_abs_start), int(keep_abs_stop)],
                    "vertices": int(len(mesh.vertices)),
                    "faces": int(len(mesh.faces)),
                    "obj": str(section_obj),
                    "msh": str(section_msh),
                    "reused": True,
                }
            )
            print(f"[section] {idx + 1}/{len(ranges)} reused {section_obj.name} -> faces={len(mesh.faces):,}")
            continue

        vol = np.empty((stop_local - start_local, y_slice.stop - y_slice.start, x_slice.stop - x_slice.start), dtype=np.int16)
        for out_z, in_z in enumerate(range(abs_start, abs_stop)):
            vol[out_z] = read_raw_slice(files[in_z])[y_slice, x_slice]

        mask = vol > threshold
        mask = morphology.remove_small_objects(mask, 5000)
        mask = morphology.binary_closing(mask, morphology.ball(2))
        mask = ndi.binary_fill_holes(mask)

        if not np.any(mask):
            section_reports.append(
                {
                    "section_index": int(idx),
                    "abs_slice_range": [int(abs_start), int(abs_stop)],
                    "kept_faces": 0,
                    "reason": "empty_mask",
                }
            )
            del vol, mask
            gc.collect()
            continue

        padded = np.pad(mask, 1, mode="constant", constant_values=False)
        vertices, faces = dicom_mesh.mask_to_surface(padded, spacing_xyz)
        vertices -= np.asarray(spacing_xyz, dtype=np.float64)
        vertices += np.asarray([x_offset_mm, y_offset_mm, abs_start * spacing_xyz[2]], dtype=np.float64)
        vertices, faces = trim_section_faces(
            vertices,
            faces.astype(np.int32),
            spacing_xyz,
            abs_start=abs_start,
            abs_stop=abs_stop,
            keep_abs_start=keep_abs_start,
            keep_abs_stop=keep_abs_stop,
        )

        if len(faces) == 0:
            section_reports.append(
                {
                    "section_index": int(idx),
                    "abs_slice_range": [int(abs_start), int(abs_stop)],
                    "kept_faces": 0,
                    "reason": "empty_after_trim",
                }
            )
            del vol, mask, padded
            gc.collect()
            continue

        mesh = cleanup_mesh(trimesh.Trimesh(vertices=vertices, faces=faces, process=False))
        section_meshes.append(mesh)

        write_obj(section_obj, np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int32))
        write_ascii_msh(section_msh, np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int32))

        section_reports.append(
            {
                "section_index": int(idx),
                "abs_slice_range": [int(abs_start), int(abs_stop)],
                "keep_abs_slice_range": [int(keep_abs_start), int(keep_abs_stop)],
                "vertices": int(len(mesh.vertices)),
                "faces": int(len(mesh.faces)),
                "obj": str(section_obj),
                "msh": str(section_msh),
                "reused": False,
            }
        )

        del vol, mask, padded
        gc.collect()
        print(f"[section] {idx + 1}/{len(ranges)} slices {abs_start}:{abs_stop} -> faces={len(mesh.faces):,}")

    if not section_meshes:
        raise RuntimeError("No section meshes were produced.")
    merged = cleanup_mesh(trimesh.util.concatenate(section_meshes))
    try:
        trimesh.repair.fix_normals(merged, multibody=True)
    except Exception:
        pass
    return merged, section_reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an ordered full-stack DICOM surface mesh")
    parser.add_argument("--section-depth", type=int, default=192)
    parser.add_argument("--overlap", type=int, default=24)
    parser.add_argument(
        "--voxel-n-axis",
        type=int,
        default=1200,
        help="Max-axis voxel count used only if seam repair needs the voxel watertight fallback.",
    )
    parser.add_argument(
        "--taubin-iters",
        type=int,
        default=0,
        help="Optional Taubin smoothing iterations before voxel watertight fallback.",
    )
    parser.add_argument(
        "--reuse-sections",
        action="store_true",
        help="Reuse existing ordered section OBJ files in exports/ordered_sections if present.",
    )
    args = parser.parse_args()

    source_dir = resolve_source_dir()
    files, source_report = collect_ordered_files(source_dir)
    spacing_xyz = tuple(source_report["spacing_xyz_mm"])
    bbox_full, threshold, coarse_report = compute_coarse_bbox_and_threshold(files)
    merged_raw, section_reports = reconstruct_sections(
        files,
        bbox_full,
        threshold,
        spacing_xyz,
        section_depth=args.section_depth,
        overlap=args.overlap,
        out_dir=EXPORTS,
        reuse_sections=bool(args.reuse_sections),
    )

    raw_report = topology_report(merged_raw)
    final_mesh = merged_raw
    final_method = "section_concat"
    final_extra = {}
    if (not merged_raw.is_watertight) or raw_report["boundary_edges"] != 0 or raw_report["nonmanifold_edges"] != 0:
        final_mesh, pitch = voxel_watertight(merged_raw, n_axis=args.voxel_n_axis, taubin_iters=args.taubin_iters)
        final_method = "section_concat_then_voxel_watertight"
        final_extra = {
            "voxel_pitch_mm": float(pitch),
            "voxel_n_axis": int(args.voxel_n_axis),
            "taubin_iters": int(args.taubin_iters),
        }

    final_mesh = cleanup_mesh(final_mesh)
    final_mesh, largest_info = keep_largest_component(final_mesh)
    final_extra.update(largest_info)
    try:
        trimesh.repair.fix_normals(final_mesh, multibody=True)
    except Exception:
        pass
    final_report = topology_report(final_mesh)

    out_obj = EXPORTS / "tooth_surface_watertight.obj"
    out_msh = EXPORTS / "tooth_surface_watertight.msh"
    out_m = EXPORTS / "tooth_surface_watertight.m"
    write_obj(out_obj, np.asarray(final_mesh.vertices, dtype=np.float64), np.asarray(final_mesh.faces, dtype=np.int32))
    write_ascii_msh(out_msh, np.asarray(final_mesh.vertices, dtype=np.float64), np.asarray(final_mesh.faces, dtype=np.int32))
    write_matlab_surface(out_m, np.asarray(final_mesh.vertices, dtype=np.float64), np.asarray(final_mesh.faces, dtype=np.int32))

    report = {
        "source": source_report,
        "coarse": coarse_report,
        "threshold_raw_units": float(threshold),
        "sectioning": {
            "section_depth": int(args.section_depth),
            "overlap": int(args.overlap),
            "sections": section_reports,
        },
        "raw_concatenated_surface": raw_report,
        "final_surface_method": final_method,
        "final_surface_method_details": final_extra,
        "final_surface": final_report,
        "outputs": {
            "obj": str(out_obj),
            "msh": str(out_msh),
            "m": str(out_m),
        },
    }
    report_path = EXPORTS / "tooth_surface_sectioned_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
