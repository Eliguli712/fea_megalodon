#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


ROOT = Path("/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg")
EXPORTS = ROOT / "DICOM_16um" / "exports"
DEFAULT_SOURCE_BDF = EXPORTS / "tooth_surface_uncompressed.bdf"
DEFAULT_VOLUME_BDF = EXPORTS / "tooth_surface_uncompressed_mesh3_tet_vol.bdf"
DEFAULT_OUT_DIR = EXPORTS / "strict3bdf_mesh_images"
FLOAT_EXP_RE = re.compile(r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+))([+-]\d+)$")

SLOT_CONFIG = {
    "comp1_mesh1": {
        "label": "comp1 / mesh1",
        "prefer": "source",
        "fallback": None,
        "elev": 18,
        "azim": -70,
        "face_color": "#b6d8e5",
        "edge_color": "#17475f",
        "line_width": 0.04,
    },
    "comp1_mesh2": {
        "label": "comp1 / mesh2",
        "prefer": "volume",
        "fallback": None,
        "elev": 20,
        "azim": 20,
        "face_color": "#c5e0d2",
        "edge_color": "#224c3f",
        "line_width": 0.035,
    },
    "comp2_mesh3": {
        "label": "comp2 / mesh3",
        "prefer": "volume",
        "fallback": None,
        "elev": 16,
        "azim": 132,
        "face_color": "#f0d9b8",
        "edge_color": "#5e3b15",
        "line_width": 0.035,
    },
}


def split_small_fields(line: str) -> list[str]:
    raw = line.rstrip("\n")
    return [raw[i : i + 8].strip() for i in range(0, len(raw), 8)]


def parse_bdf_float(text: str) -> float:
    text = text.strip()
    if not text:
        raise ValueError("empty BDF float")
    try:
        return float(text)
    except ValueError:
        match = FLOAT_EXP_RE.match(text)
        if match:
            return float(f"{match.group(1)}e{match.group(2)}")
        raise


def count_bdf(path: Path) -> dict[str, int]:
    grid = 0
    tri = 0
    max_id = 0
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if line.startswith("GRID"):
                fields = split_small_fields(line)
                if len(fields) < 6:
                    continue
                grid += 1
                try:
                    max_id = max(max_id, int(fields[1]))
                except Exception:
                    pass
            elif line.startswith("CTRIA3"):
                tri += 1
    return {"grid": grid, "tri": tri, "max_id": max_id}


def load_sampled_surface(path: Path, max_faces: int) -> dict:
    counts = count_bdf(path)
    if counts["grid"] <= 0:
        raise RuntimeError(f"No GRID cards found in {path}")
    if counts["tri"] <= 0:
        raise RuntimeError(f"No CTRIA3 cards found in {path}")

    nodes = np.full((counts["max_id"] + 1, 3), np.nan, dtype=np.float32)
    stride = max(1, math.ceil(counts["tri"] / max_faces))
    faces: list[tuple[int, int, int]] = []
    tri_seen = 0

    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if line.startswith("GRID"):
                fields = split_small_fields(line)
                if len(fields) < 6:
                    continue
                nid = int(fields[1])
                nodes[nid, 0] = parse_bdf_float(fields[3])
                nodes[nid, 1] = parse_bdf_float(fields[4])
                nodes[nid, 2] = parse_bdf_float(fields[5])
            elif line.startswith("CTRIA3"):
                tri_seen += 1
                if tri_seen % stride != 1 and tri_seen != counts["tri"]:
                    continue
                fields = split_small_fields(line)
                if len(fields) < 6:
                    continue
                faces.append((int(fields[3]), int(fields[4]), int(fields[5])))

    face_ids = np.asarray(faces, dtype=np.int32)
    unique_ids, inverse = np.unique(face_ids.reshape(-1), return_inverse=True)
    coords = nodes[unique_ids]
    if np.isnan(coords).any():
        raise RuntimeError(f"Surface sample from {path} references undefined GRID ids.")
    tris = inverse.reshape(-1, 3)
    return {
        "coords": coords,
        "tris": tris,
        "faces_total": counts["tri"],
        "faces_used": int(tris.shape[0]),
        "grid_total": counts["grid"],
        "source_path": str(path),
        "source_name": path.name,
    }


def set_equal_3d(ax, coords: np.ndarray) -> None:
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0)
    if not math.isfinite(radius) or radius <= 0:
        radius = 1.0
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def render_surface(slot: str, payload: dict, out_path: Path, cfg: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    coords = payload["coords"]
    tris = payload["tris"]
    poly = coords[tris]

    fig = plt.figure(figsize=(10, 8), dpi=200)
    ax = fig.add_subplot(111, projection="3d")
    mesh = Poly3DCollection(
        poly,
        facecolor=cfg["face_color"],
        edgecolor=cfg["edge_color"],
        linewidths=cfg["line_width"],
        alpha=1.0,
    )
    ax.add_collection3d(mesh)
    set_equal_3d(ax, coords)
    ax.view_init(elev=cfg["elev"], azim=cfg["azim"])
    ax.set_axis_off()
    ax.set_title(
        f"{cfg['label']}\n{payload['source_name']} | faces {payload['faces_used']:,d}/{payload['faces_total']:,d}",
        fontsize=10,
        pad=14,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(
        "MESH_PREVIEW|slot={}|file={}|source={}|faces_used={}|faces_total={}".format(
            slot,
            out_path,
            payload["source_name"],
            payload["faces_used"],
            payload["faces_total"],
        )
    )


def build_slot_manifest(slot: str, cfg: dict, payload: dict | None, out_path: Path) -> dict:
    if payload is None:
        return {
            "label": cfg["label"],
            "status": "pending",
            "file": "",
            "source_name": "pending",
            "source_path": "",
            "faces_total": None,
            "faces_used": None,
        }
    return {
        "label": cfg["label"],
        "status": "ready",
        "file": out_path.name,
        "source_name": payload["source_name"],
        "source_path": payload["source_path"],
        "faces_total": payload["faces_total"],
        "faces_used": payload["faces_used"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-bdf", default=str(DEFAULT_SOURCE_BDF))
    ap.add_argument("--volume-bdf", default=str(DEFAULT_VOLUME_BDF))
    ap.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--max-faces", type=int, default=40000)
    args = ap.parse_args()

    source_bdf = Path(args.source_bdf)
    volume_bdf = Path(args.volume_bdf)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets: dict[str, dict] = {}
    if source_bdf.exists() and source_bdf.stat().st_size > 0:
        datasets["source"] = load_sampled_surface(source_bdf, args.max_faces)
    else:
        print(f"MESH_PREVIEW_SKIP|dataset=source|reason=missing|path={source_bdf}")

    if volume_bdf.exists() and volume_bdf.stat().st_size > 0:
        datasets["volume"] = load_sampled_surface(volume_bdf, args.max_faces)
    else:
        print(f"MESH_PREVIEW_SKIP|dataset=volume|reason=missing|path={volume_bdf}")

    manifest = {
        "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_bdf": str(source_bdf),
        "volume_bdf": str(volume_bdf),
        "max_faces": args.max_faces,
        "slots": {},
    }

    for slot, cfg in SLOT_CONFIG.items():
        payload = datasets.get(cfg["prefer"])
        if payload is None and cfg.get("fallback"):
            payload = datasets.get(cfg["fallback"])
        out_path = out_dir / f"{slot}_hr.png"
        if payload is not None:
            render_surface(slot, payload, out_path, cfg)
            manifest["slots"][slot] = build_slot_manifest(slot, cfg, payload, out_path)
        else:
            manifest["slots"][slot] = build_slot_manifest(slot, cfg, None, out_path)

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"MESH_PREVIEW_MANIFEST|{manifest_path}")


if __name__ == "__main__":
    main()
