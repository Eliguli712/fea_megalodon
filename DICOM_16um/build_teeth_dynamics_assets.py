#!/usr/bin/env python3
"""Assemble a frontal tooth dynamics sheet and companion HTML from saved exports."""

from __future__ import annotations

import argparse
import base64
import html
import io
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.collections import PolyCollection
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy import ndimage


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
EXPORTS = SCRIPT_DIR / "exports"
MODEL_PATH = EXPORTS / "static_dynamics_high_resolution_strict3bdf.mph"
TOOTH_OBJ = EXPORTS / "tooth_surface_comsol_watertight.obj"
STUDY_IMAGE_DIR = PROJECT_ROOT / "applied_forces" / "exports" / "holocastic_full_body_images"

STUDY_ORDER = ["std1", "std_nh", "std_og", "std_mr2", "std_mr5", "std_pr"]
STUDY_TO_SOL = {
    "std1": "sol6",
    "std_nh": "sol1",
    "std_og": "sol2",
    "std_mr2": "sol3",
    "std_mr5": "sol4",
    "std_pr": "sol5",
}
STUDY_LABELS = {
    "std1": "Linear Elastic",
    "std_nh": "Neo-Hookean",
    "std_og": "Ogden",
    "std_mr2": "Mooney-Rivlin MR2",
    "std_mr5": "Mooney-Rivlin MR5",
    "std_pr": "Pressure (MR5)",
}
STUDY_IMAGE_CANDIDATES = {
    "std1": ["std1_von_mises_contrast.png", "std1_von_mises.png"],
    "std_nh": ["std_nh_von_mises_contrast.png", "std_nh_von_mises.png"],
    "std_og": ["std_og_von_mises_contrast.png", "std_og_von_mises.png"],
    "std_mr2": ["std_mr2_mooney_rivlin_contrast.png", "std_mr2_von_mises.png"],
    "std_mr5": ["std_mr5_mooney_rivlin_contrast.png", "std_mr5_von_mises.png"],
    "std_pr": ["std_pr_von_mises_contrast.png", "std_pr_von_mises.png"],
}

RUN_DOF_RE = re.compile(r"Number of degrees of freedom solved for:\s*([0-9,]+)")
SOLVER_START_RE = re.compile(r"<---- Stationary Solver 1 in (.+?)\((sol[0-9]+)\)")
STARTED_RE = re.compile(r"Started at (.+)\.")
ENDED_RE = re.compile(r"Ended at (.+)\.")
SOL_TIME_RE = re.compile(r"Solution time:\s*([0-9]+)\s*s")


@dataclass
class StudyRow:
    study: str
    label: str
    dataset: str
    solver: str
    vm: float
    um: float
    finite_nonzero: bool
    solve_time_s: int | None
    started: str
    ended: str
    image_path: Path | None


def parse_kv_line(line: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in line.split("|")[1:]:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        out[key] = value
    return out


def choose_latest_complete_run() -> tuple[Path, Path]:
    candidates = sorted(EXPORTS.glob("run_strict3bdf_contact_active_*.stdout.txt"))
    for stdout_path in reversed(candidates):
        text = stdout_path.read_text(encoding="utf-8", errors="ignore")
        if "SUMMARY|finite_nonzero_studies=6|total_target_studies=6" not in text:
            continue
        log_name = stdout_path.name.replace(".stdout.txt", ".log")
        log_path = stdout_path.with_name(log_name)
        if log_path.exists():
            return stdout_path, log_path
    raise FileNotFoundError("No complete strict3bdf contact-active run stdout/log pair was found.")


def parse_run_stdout(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore")
    by_study: dict[str, dict] = {}
    model = ""
    raw_bdf = ""
    mesh2_tet = None
    mesh3_tet = None
    summary = {"finite_nonzero_studies": 0, "total_target_studies": 0}

    for line in text.splitlines():
        if line.startswith("MODEL|"):
            model = line.split("|", 1)[1].strip()
            continue
        if line.startswith("RAW_BDF_IN_USE|"):
            raw_bdf = line.split("|", 1)[1].strip()
            continue
        if line.startswith("COMP1_MESH2_TET|"):
            try:
                mesh2_tet = int(line.split("|", 1)[1].strip())
            except ValueError:
                pass
            continue
        if line.startswith("COMP2_MESH3_TET|"):
            try:
                mesh3_tet = int(line.split("|", 1)[1].strip())
            except ValueError:
                pass
            continue
        if line.startswith("CHECK|"):
            kv = parse_kv_line(line)
            study = kv.get("study", "")
            if not study:
                continue
            row = by_study.setdefault(study, {})
            row["dataset"] = kv.get("dataset", "")
            row["vm"] = float(kv.get("vm", "nan"))
            row["um"] = float(kv.get("um", "nan"))
            row["finite_nonzero"] = kv.get("finite_nonzero", "").lower() == "true"
            continue
        if line.startswith("SUMMARY|"):
            kv = parse_kv_line(line)
            try:
                summary["finite_nonzero_studies"] = int(kv.get("finite_nonzero_studies", "0"))
                summary["total_target_studies"] = int(kv.get("total_target_studies", "0"))
            except ValueError:
                pass

    return {
        "model": model or str(MODEL_PATH),
        "raw_bdf": raw_bdf,
        "mesh2_tet": mesh2_tet,
        "mesh3_tet": mesh3_tet,
        "summary": summary,
        "studies": by_study,
    }


def parse_solver_log(path: Path) -> tuple[int | None, dict[str, dict]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    dof_vals: list[int] = []
    blocks: dict[str, dict] = {}
    current: str | None = None

    for match in RUN_DOF_RE.finditer(text):
        try:
            dof_vals.append(int(match.group(1).replace(",", "")))
        except ValueError:
            pass

    for line in text.splitlines():
        sm = SOLVER_START_RE.search(line)
        if sm:
            current = sm.group(2)
            blocks[current] = {
                "label": sm.group(1).strip(),
                "started": "",
                "ended": "",
                "solve_time_s": None,
            }
            continue
        if current is None:
            continue
        if not blocks[current]["started"]:
            m = STARTED_RE.search(line)
            if m:
                blocks[current]["started"] = m.group(1).strip()
        if blocks[current]["solve_time_s"] is None:
            m = SOL_TIME_RE.search(line)
            if m:
                try:
                    blocks[current]["solve_time_s"] = int(m.group(1))
                except ValueError:
                    pass
        if not blocks[current]["ended"]:
            m = ENDED_RE.search(line)
            if m:
                blocks[current]["ended"] = m.group(1).strip()

    dof = max(dof_vals) if dof_vals else None
    return dof, blocks


def pick_study_images() -> dict[str, Path | None]:
    out: dict[str, Path | None] = {}
    for study, names in STUDY_IMAGE_CANDIDATES.items():
        chosen = None
        for name in names:
            candidate = STUDY_IMAGE_DIR / name
            if candidate.exists() and candidate.stat().st_size > 0:
                chosen = candidate
                break
        out[study] = chosen
    return out


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type: {type(mesh)!r}")
    return mesh


def oriented_vertices(mesh: trimesh.Trimesh) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    centered = vertices - vertices.mean(axis=0, keepdims=True)
    cov = np.cov(centered.T)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    axes = evecs[:, order]

    y_axis = axes[:, 0]
    x_axis = axes[:, 1]
    if y_axis[2] < 0:
        y_axis = -y_axis
    if x_axis[1] < 0:
        x_axis = -x_axis
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)
    if np.linalg.det(np.column_stack((x_axis, y_axis, z_axis))) < 0:
        z_axis = -z_axis

    basis = np.column_stack((x_axis, y_axis, z_axis))
    return centered @ basis


def render_front_tooth(mesh: trimesh.Trimesh) -> Image.Image:
    pts = oriented_vertices(mesh)
    faces = np.asarray(mesh.faces, dtype=np.int32)

    polys = pts[faces][:, :, :2]
    depth = pts[faces][:, :, 2].mean(axis=1)
    order = np.argsort(depth)
    polys = polys[order]
    depth = depth[order]

    norm = plt.Normalize(float(depth.min()), float(depth.max()))
    colors = matplotlib.colormaps["magma"](norm(depth))
    colors[:, 3] = 0.98

    x = pts[:, 0]
    y = pts[:, 1]
    x_pad = np.ptp(x) * 0.08
    y_pad = np.ptp(y) * 0.08

    fig = plt.figure(figsize=(6.4, 8.4), dpi=240, facecolor="#f7f0e4")
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    coll = PolyCollection(
        polys,
        facecolors=colors,
        edgecolors=(0.03, 0.03, 0.05, 0.05),
        linewidths=0.04,
        antialiased=False,
    )
    ax.add_collection(coll)
    ax.set_xlim(float(x.min() - x_pad), float(x.max() + x_pad))
    ax.set_ylim(float(y.min() - y_pad), float(y.max() + y_pad))
    ax.set_aspect("equal")
    ax.set_axis_off()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), dpi=240)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGBA")


def get_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Avenir Next.ttc" if bold else "/System/Library/Fonts/Supplemental/Avenir.ttc",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def format_sci(value: float) -> str:
    return f"{value:.6e}"


def format_int(value: int | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:,d}"


def format_seconds(seconds: int | None) -> str:
    if seconds is None:
        return "n/a"
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {sec:02d}s"
    return f"{minutes}m {sec:02d}s"


def relative_label(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def build_study_rows(run_meta: dict, solver_meta: dict[str, dict], image_paths: dict[str, Path | None]) -> list[StudyRow]:
    rows: list[StudyRow] = []
    for study in STUDY_ORDER:
        data = run_meta["studies"].get(study, {})
        solver = STUDY_TO_SOL[study]
        block = solver_meta.get(solver, {})
        rows.append(
            StudyRow(
                study=study,
                label=STUDY_LABELS[study],
                dataset=data.get("dataset", ""),
                solver=solver,
                vm=float(data.get("vm", float("nan"))),
                um=float(data.get("um", float("nan"))),
                finite_nonzero=bool(data.get("finite_nonzero", False)),
                solve_time_s=block.get("solve_time_s"),
                started=block.get("started", ""),
                ended=block.get("ended", ""),
                image_path=image_paths.get(study),
            )
        )
    return rows


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font, fill, *, anchor: str | None = None) -> None:
    kwargs = {"fill": fill, "font": font}
    if anchor is not None:
        kwargs["anchor"] = anchor
    draw.text(xy, text, **kwargs)


def line_height(font) -> int:
    bbox = font.getbbox("Ag")
    return bbox[3] - bbox[1]


def fit_cover(path: Path, size: tuple[int, int]) -> Image.Image:
    img = load_cropped_panel(path)
    return ImageOps.contain(img, size, method=Image.Resampling.LANCZOS)


def load_cropped_panel(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img)
    mask = np.any(arr < 245, axis=2)
    if not mask.any():
        return img
    labels, count = ndimage.label(mask)
    if count > 1:
        areas = ndimage.sum(mask, labels, index=np.arange(1, count + 1))
        largest_label = int(np.argmax(areas)) + 1
        mask = labels == largest_label
    ys, xs = np.where(mask)
    pad = 18
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(img.width, int(xs.max()) + 1 + pad)
    y1 = min(img.height, int(ys.max()) + 1 + pad)
    return img.crop((x0, y0, x1, y1))


def build_png_sheet(out_path: Path, front_img: Image.Image, rows: list[StudyRow], run_stdout: Path, run_log: Path, run_meta: dict, dof: int | None) -> None:
    bg = "#efe7d8"
    paper = "#fbf6ee"
    ink = "#1f2e3a"
    muted = "#5d6b74"
    deep = "#184a70"
    accent = "#b8612e"
    line = "#d8ccbc"
    ok = "#1d7a55"
    bad = "#9a2f2f"

    canvas_w, canvas_h = 2600, 2080
    margin = 44
    gutter = 28

    canvas = Image.new("RGB", (canvas_w, canvas_h), bg)
    draw = ImageDraw.Draw(canvas)

    title_font = get_font(54, bold=True)
    h2_font = get_font(28, bold=True)
    body_font = get_font(23)
    small_font = get_font(19)
    mono_font = get_font(20)

    def rounded_card(box: tuple[int, int, int, int], *, radius: int = 28) -> None:
        x0, y0, x1, y1 = box
        shadow_box = (x0 + 6, y0 + 10, x1 + 6, y1 + 10)
        draw.rounded_rectangle(shadow_box, radius=radius, fill="#d7ccbdb8")
        draw.rounded_rectangle(box, radius=radius, fill=paper, outline=line, width=2)

    header_box = (margin, margin, canvas_w - margin, 210)
    rounded_card(header_box, radius=34)
    draw_text(draw, (header_box[0] + 32, header_box[1] + 34), "Shark Teeth Static Dynamics", title_font, ink)
    draw_text(
        draw,
        (header_box[0] + 34, header_box[1] + 114),
        f"Source model: {relative_label(MODEL_PATH)}",
        body_font,
        deep,
    )
    draw_text(
        draw,
        (header_box[0] + 34, header_box[1] + 148),
        "Frontal tooth render synthesized from the exported watertight OBJ; study metrics come from the saved strict3bdf contact-active run.",
        small_font,
        muted,
    )

    hero_box = (margin, 244, 1020, 1236)
    rounded_card(hero_box)
    draw_text(draw, (hero_box[0] + 28, hero_box[1] + 24), "Frontal Tooth View", h2_font, ink)
    hero_img = ImageOps.contain(front_img, (hero_box[2] - hero_box[0] - 46, hero_box[3] - hero_box[1] - 118), method=Image.Resampling.LANCZOS)
    hero_x = hero_box[0] + ((hero_box[2] - hero_box[0]) - hero_img.width) // 2
    hero_y = hero_box[1] + 76
    canvas.paste(hero_img.convert("RGB"), (hero_x, hero_y))
    draw_text(
        draw,
        (hero_box[0] + 28, hero_box[3] - 58),
        f"Geometry source: {relative_label(TOOTH_OBJ)}",
        small_font,
        muted,
    )

    summary_box = (hero_box[2] + gutter, 244, canvas_w - margin, 728)
    rounded_card(summary_box)
    draw_text(draw, (summary_box[0] + 28, summary_box[1] + 24), "Strict3BDF Static Solve Summary", h2_font, ink)

    stat_pairs = [
        ("Finite/non-zero studies", f"{run_meta['summary']['finite_nonzero_studies']} / {run_meta['summary']['total_target_studies']}"),
        ("Observed solver DOF", format_int(dof)),
        ("mesh2 tetrahedra", format_int(run_meta.get("mesh2_tet"))),
        ("mesh3 tetrahedra", format_int(run_meta.get("mesh3_tet"))),
        ("Peak von Mises", format_sci(max(r.vm for r in rows))),
        ("Peak displacement", format_sci(max(r.um for r in rows))),
    ]

    stat_x0 = summary_box[0] + 28
    stat_y0 = summary_box[1] + 86
    stat_w = (summary_box[2] - summary_box[0] - 84) // 2
    stat_h = 110
    for idx, (label, value) in enumerate(stat_pairs):
        col = idx % 2
        row = idx // 2
        x0 = stat_x0 + col * (stat_w + 28)
        y0 = stat_y0 + row * (stat_h + 18)
        box = (x0, y0, x0 + stat_w, y0 + stat_h)
        draw.rounded_rectangle(box, radius=18, fill="#fffdf9", outline=line, width=2)
        draw_text(draw, (x0 + 18, y0 + 18), label, small_font, muted)
        draw_text(draw, (x0 + 18, y0 + 56), value, get_font(30, bold=True), deep)

    info_y = summary_box[1] + 430
    draw_text(draw, (summary_box[0] + 28, info_y), f"Run stdout: {relative_label(run_stdout)}", small_font, muted)
    draw_text(draw, (summary_box[0] + 28, info_y + 32), f"Run log: {relative_label(run_log)}", small_font, muted)
    draw_text(draw, (summary_box[0] + 28, info_y + 64), f"Raw BDF in use: {relative_label(Path(run_meta['raw_bdf'])) if run_meta['raw_bdf'] else 'n/a'}", small_font, muted)

    table_box = (hero_box[2] + gutter, summary_box[3] + gutter, canvas_w - margin, 1236)
    rounded_card(table_box)
    draw_text(draw, (table_box[0] + 28, table_box[1] + 24), "Per-Study Static Checks", h2_font, ink)

    headers = ["Study", "Dataset", "Von Mises (Pa)", "Disp. (m)", "Solve Time", "State"]
    widths = [180, 120, 280, 220, 170, 130]
    table_x = table_box[0] + 28
    table_y = table_box[1] + 78
    row_h = 56

    x = table_x
    for header, width in zip(headers, widths):
        draw.rounded_rectangle((x, table_y, x + width - 8, table_y + row_h), radius=14, fill="#f2ece2", outline=line, width=1)
        draw_text(draw, (x + 14, table_y + 15), header, small_font, ink)
        x += width

    for idx, row in enumerate(rows):
        y = table_y + row_h + 10 + idx * (row_h + 10)
        bg_fill = "#fffdf9" if idx % 2 == 0 else "#f9f4ec"
        draw.rounded_rectangle((table_x, y, table_x + sum(widths) - 8, y + row_h), radius=14, fill=bg_fill, outline=line, width=1)
        state_fill = ok if row.finite_nonzero else bad
        values = [
            row.label,
            row.dataset or "n/a",
            format_sci(row.vm),
            format_sci(row.um),
            format_seconds(row.solve_time_s),
            "finite" if row.finite_nonzero else "failed",
        ]
        x = table_x
        for col_idx, (value, width) in enumerate(zip(values, widths)):
            fill = state_fill if col_idx == len(values) - 1 else ink
            draw_text(draw, (x + 14, y + 16), value, small_font, fill)
            x += width

    gallery_box = (margin, 1270, canvas_w - margin, canvas_h - margin)
    rounded_card(gallery_box)
    draw_text(draw, (gallery_box[0] + 28, gallery_box[1] + 24), "Saved COMSOL Export Panels", h2_font, ink)

    inner_x = gallery_box[0] + 28
    inner_y = gallery_box[1] + 78
    cols = 3
    tile_w = (gallery_box[2] - gallery_box[0] - 56 - gutter * 2) // cols
    tile_h = 328
    caption_h = 86
    for idx, row in enumerate(rows):
        col = idx % cols
        r = idx // cols
        x0 = inner_x + col * (tile_w + gutter)
        y0 = inner_y + r * (tile_h + caption_h + 18)
        frame = (x0, y0, x0 + tile_w, y0 + tile_h)
        draw.rounded_rectangle(frame, radius=20, fill="#fff", outline=line, width=2)
        if row.image_path is not None:
            img = fit_cover(row.image_path, (tile_w - 24, tile_h - 24))
            px = x0 + (tile_w - img.width) // 2
            py = y0 + (tile_h - img.height) // 2
            canvas.paste(img, (px, py))
        else:
            draw_text(draw, (x0 + tile_w // 2, y0 + tile_h // 2), "Image missing", body_font, bad, anchor="mm")

        cap_y = y0 + tile_h + 16
        draw_text(draw, (x0, cap_y), row.label, body_font, deep)
        draw_text(
            draw,
            (x0, cap_y + 32),
            f"{format_sci(row.vm)} Pa | {format_sci(row.um)} m",
            small_font,
            muted,
        )
        img_label = relative_label(row.image_path) if row.image_path else "n/a"
        draw_text(draw, (x0, cap_y + 58), img_label, mono_font, accent)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def build_comprehensive_section_png(out_path: Path, sheet_path: Path) -> None:
    bg = "#efe7d8"
    paper = "#fbf6ee"
    ink = "#1f2e3a"
    muted = "#5d6b74"
    line = "#d8ccbc"

    title_font = get_font(34, bold=True)
    body_font = get_font(24)

    montage = Image.open(sheet_path).convert("RGB")

    margin = 44
    pad = 28
    text_gap = 18
    title_gap = 68
    body_gap = 64
    canvas_w = montage.width + margin * 2 + pad * 2
    canvas_h = montage.height + margin * 2 + pad * 2 + title_gap + body_gap

    canvas = Image.new("RGB", (canvas_w, canvas_h), bg)
    draw = ImageDraw.Draw(canvas)

    card = (margin, margin, canvas_w - margin, canvas_h - margin)
    shadow_box = (card[0] + 6, card[1] + 10, card[2] + 6, card[3] + 10)
    draw.rounded_rectangle(shadow_box, radius=30, fill="#d7ccbdb8")
    draw.rounded_rectangle(card, radius=30, fill=paper, outline=line, width=2)

    text_x = card[0] + pad
    text_y = card[1] + pad
    draw_text(draw, (text_x, text_y), "Comprehensive Sheet", title_font, ink)

    montage_frame = (
        card[0] + pad,
        text_y + title_gap,
        card[2] - pad,
        text_y + title_gap + montage.height,
    )
    draw.rounded_rectangle(montage_frame, radius=18, fill="#fff", outline=line, width=2)
    canvas.paste(montage, (montage_frame[0], montage_frame[1]))

    body_y = montage_frame[3] + text_gap
    draw_text(
        draw,
        (text_x, body_y),
        "Single-sheet export combining the front-facing tooth view, strict3bdf study metrics, and the saved COMSOL result panels.",
        body_font,
        muted,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def image_data_uri(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def pil_image_data_uri(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def render_html(out_path: Path, png_path: Path, front_img: Image.Image, rows: list[StudyRow], run_stdout: Path, run_log: Path, run_meta: dict, dof: int | None) -> None:
    front_uri = pil_image_data_uri(front_img)
    montage_uri = image_data_uri(png_path)

    cards: list[str] = []
    for row in rows:
        img_src = pil_image_data_uri(load_cropped_panel(row.image_path)) if row.image_path is not None else ""
        img_html = (
            f'<img src="{img_src}" alt="{html.escape(row.label)} study export" />'
            if img_src
            else '<div class="missing">No study image located</div>'
        )
        cards.append(
            "<figure class=\"panel-card\">"
            f"{img_html}"
            f"<figcaption><strong>{html.escape(row.label)}</strong><br>"
            f"<code>{html.escape(row.study)}</code> | dataset <code>{html.escape(row.dataset or 'n/a')}</code><br>"
            f"{html.escape(format_sci(row.vm))} Pa | {html.escape(format_sci(row.um))} m</figcaption>"
            "</figure>"
        )

    table_rows = []
    for row in rows:
        table_rows.append(
            "<tr>"
            f"<td><code>{html.escape(row.study)}</code></td>"
            f"<td>{html.escape(row.label)}</td>"
            f"<td><code>{html.escape(row.dataset or 'n/a')}</code></td>"
            f"<td>{html.escape(format_sci(row.vm))}</td>"
            f"<td>{html.escape(format_sci(row.um))}</td>"
            f"<td>{html.escape(format_seconds(row.solve_time_s))}</td>"
            f"<td>{'finite' if row.finite_nonzero else 'failed'}</td>"
            "</tr>"
        )

    raw_bdf_label = relative_label(Path(run_meta["raw_bdf"])) if run_meta["raw_bdf"] else "n/a"
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Shark Teeth Static Dynamics</title>
  <style>
    :root {{
      --bg:#efe7d8;
      --paper:#fbf6ee;
      --ink:#1f2e3a;
      --muted:#5d6b74;
      --line:#d8ccbc;
      --deep:#184a70;
      --warm:#b8612e;
      --ok:#1d7a55;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(255,255,255,.45), transparent 24%),
        linear-gradient(180deg, #f4ecdd 0%, var(--bg) 100%);
      font: 16px/1.5 "Avenir Next","Trebuchet MS","Segoe UI",sans-serif;
    }}
    .wrap {{
      width: min(1520px, 96vw);
      margin: 22px auto 38px;
      display: grid;
      gap: 18px;
    }}
    .card {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 22px;
      box-shadow: 0 12px 28px rgba(40, 34, 24, .08);
    }}
    h1, h2, p {{ margin: 0; }}
    h1 {{ font-size: clamp(1.7rem, 3vw, 2.6rem); }}
    h2 {{ font-size: 1.12rem; margin-bottom: 12px; }}
    .sub {{
      margin-top: 10px;
      color: var(--muted);
      max-width: 82ch;
    }}
    .meta {{
      margin-top: 14px;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .chip {{
      display: inline-flex;
      align-items: center;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #fff;
      padding: 6px 11px;
      font-size: .84rem;
    }}
    code {{
      background: #edf0f1;
      padding: 0 4px;
      border-radius: 4px;
      color: var(--deep);
    }}
    .hero-grid {{
      display: grid;
      grid-template-columns: minmax(320px, 0.9fr) minmax(430px, 1.1fr);
      gap: 16px;
      align-items: stretch;
    }}
    .front-view {{
      width: 100%;
      display: block;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: #fff;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}
    .stat {{
      border: 1px solid var(--line);
      border-radius: 14px;
      background: #fffdf9;
      padding: 12px 14px;
    }}
    .stat .k {{
      color: var(--muted);
      font-size: .86rem;
    }}
    .stat .v {{
      margin-top: 4px;
      font-size: 1.1rem;
      font-weight: 700;
      color: var(--deep);
    }}
    .note {{
      margin-top: 14px;
      padding: 12px 14px;
      border-left: 4px solid var(--warm);
      background: rgba(184, 97, 46, .08);
      color: var(--ink);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: .93rem;
    }}
    th, td {{
      text-align: left;
      padding: 9px 8px;
      border-bottom: 1px solid #e8dfd3;
      vertical-align: top;
    }}
    th {{
      color: #29465d;
      font-weight: 700;
    }}
    .ok {{
      color: var(--ok);
      font-weight: 700;
    }}
    .montage {{
      width: 100%;
      display: block;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: #fff;
    }}
    .panel-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
    }}
    .panel-card {{
      margin: 0;
      display: grid;
      gap: 8px;
      border: 1px solid var(--line);
      border-radius: 14px;
      overflow: hidden;
      background: #fff;
    }}
    .panel-card img {{
      width: 100%;
      height: 260px;
      object-fit: contain;
      display: block;
      background: #f9fafb;
    }}
    .panel-card figcaption {{
      padding: 10px 12px 14px;
      color: var(--muted);
      font-size: .86rem;
      border-top: 1px solid var(--line);
      background: #fffdf9;
    }}
    .missing {{
      height: 260px;
      display: grid;
      place-items: center;
      color: #9a2f2f;
      background: #fff4f2;
    }}
    @media (max-width: 1100px) {{
      .hero-grid,
      .panel-grid {{
        grid-template-columns: 1fr;
      }}
      .stats {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <main class="wrap">
    <section class="card">
      <h1>Shark Teeth Static Dynamics</h1>
      <p class="sub">Front-facing overview assembled from the saved exports of <code>{html.escape(relative_label(MODEL_PATH))}</code>. The tooth silhouette uses the exported watertight geometry, while the metrics and gallery below come from the final strict3bdf contact-active run and its linked surface-image exports.</p>
      <div class="meta">
        <span class="chip">Model: <code>{html.escape(relative_label(MODEL_PATH))}</code></span>
        <span class="chip">Run stdout: <code>{html.escape(relative_label(run_stdout))}</code></span>
        <span class="chip">Run log: <code>{html.escape(relative_label(run_log))}</code></span>
        <span class="chip">Sheet: <code>{html.escape(relative_label(png_path))}</code></span>
      </div>
    </section>

    <section class="card hero-grid">
      <div>
        <h2>Frontal Tooth Geometry</h2>
        <img class="front-view" src="{front_uri}" alt="Frontal shark tooth render" />
        <p class="sub">Orthographic frontal render from <code>{html.escape(relative_label(TOOTH_OBJ))}</code>.</p>
      </div>
      <div>
        <h2>Strict3BDF Summary</h2>
        <div class="stats">
          <div class="stat"><div class="k">Finite/non-zero studies</div><div class="v">{run_meta['summary']['finite_nonzero_studies']} / {run_meta['summary']['total_target_studies']}</div></div>
          <div class="stat"><div class="k">Observed solver DOF</div><div class="v">{html.escape(format_int(dof))}</div></div>
          <div class="stat"><div class="k">mesh2 tetrahedra</div><div class="v">{html.escape(format_int(run_meta.get('mesh2_tet')))}</div></div>
          <div class="stat"><div class="k">mesh3 tetrahedra</div><div class="v">{html.escape(format_int(run_meta.get('mesh3_tet')))}</div></div>
          <div class="stat"><div class="k">Peak von Mises</div><div class="v">{html.escape(format_sci(max(r.vm for r in rows)))}</div></div>
          <div class="stat"><div class="k">Peak displacement</div><div class="v">{html.escape(format_sci(max(r.um for r in rows)))}</div></div>
        </div>
        <div class="note">
          Shared raw BDF input: <code>{html.escape(raw_bdf_label)}</code><br />
          Study gallery source folder: <code>{html.escape(relative_label(STUDY_IMAGE_DIR))}</code>
        </div>
      </div>
    </section>

    <section class="card">
      <h2>Comprehensive Sheet</h2>
      <img class="montage" src="{montage_uri}" alt="Shark teeth dynamics montage" />
      <p class="sub">Single-sheet export combining the front-facing tooth view, strict3bdf study metrics, and the saved COMSOL result panels.</p>
    </section>

    <section class="card">
      <h2>Per-Study Static Checks</h2>
      <table>
        <thead>
          <tr>
            <th>Study</th>
            <th>Label</th>
            <th>Dataset</th>
            <th>Von Mises (Pa)</th>
            <th>Displacement (m)</th>
            <th>Solve Time</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {''.join(table_rows)}
        </tbody>
      </table>
    </section>

    <section class="card">
      <h2>Saved COMSOL Export Panels</h2>
      <div class="panel-grid">
        {''.join(cards)}
      </div>
    </section>
  </main>
</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build frontal teeth dynamics PNG + HTML from saved exports.")
    parser.add_argument("--out-png", default=str(PROJECT_ROOT / "teeth_dynamics.png"))
    parser.add_argument("--out-html", default=str(PROJECT_ROOT / "visualize_teeth_dynamics.html"))
    parser.add_argument(
        "--out-comprehensive-png",
        default=str(PROJECT_ROOT / "teeth_dynamics_comprehensive.png"),
    )
    args = parser.parse_args()

    run_stdout, run_log = choose_latest_complete_run()
    run_meta = parse_run_stdout(run_stdout)
    dof, solver_meta = parse_solver_log(run_log)
    image_paths = pick_study_images()
    rows = build_study_rows(run_meta, solver_meta, image_paths)

    mesh = load_mesh(TOOTH_OBJ)
    front_img = render_front_tooth(mesh)

    out_png = Path(args.out_png).resolve()
    out_html = Path(args.out_html).resolve()
    out_comprehensive_png = Path(args.out_comprehensive_png).resolve()
    build_png_sheet(out_png, front_img, rows, run_stdout, run_log, run_meta, dof)
    build_comprehensive_section_png(out_comprehensive_png, out_png)
    render_html(out_html, out_png, front_img, rows, run_stdout, run_log, run_meta, dof)

    print(f"PNG|{out_png}")
    print(f"COMPREHENSIVE_PNG|{out_comprehensive_png}")
    print(f"HTML|{out_html}")


if __name__ == "__main__":
    main()
