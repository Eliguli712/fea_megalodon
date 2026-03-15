#!/usr/bin/env python3
"""Export the six captioned von Mises panels into a single PNG sheet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


SCRIPT_DIR = Path(__file__).resolve().parent
EXPORTS = SCRIPT_DIR / "exports"
REPORT_PATH = EXPORTS / "von_mises_image_generation_report.json"
HTML_PATH = EXPORTS / "von_mises_visualization.html"

ENTITY_ORDER = [
    "surface_mesh_smoothed",
    "tooth_surface_uncompressed",
    "tooth_surface_comsol_tet_vol",
]

COLORS = {
    "bg": "#f4f8fb",
    "card": "#ffffff",
    "ink": "#12222f",
    "muted": "#4a6378",
    "line": "#d7e2ec",
    "accent": "#0d6b9d",
    "band": "#eff7fc",
}


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


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font, fill, *, anchor: str | None = None) -> None:
    kwargs = {"font": font, "fill": fill}
    if anchor is not None:
        kwargs["anchor"] = anchor
    draw.text(xy, text, **kwargs)


def fmt_count(value: int | None) -> str:
    return "n/a" if value is None else f"{value:,d}"


def fmt_dist(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.10f}"


def crop_panel(img: Image.Image) -> Image.Image:
    arr = np.asarray(img)
    mask = np.any(arr < 245, axis=2)
    if not mask.any():
        return img
    ys, xs = np.where(mask)
    pad = 12
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(img.width, int(xs.max()) + 1 + pad)
    y1 = min(img.height, int(ys.max()) + 1 + pad)
    return img.crop((x0, y0, x1, y1))


def fit_image(path: Path, size: tuple[int, int]) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img = crop_panel(img)
    return ImageOps.contain(img, size, method=Image.Resampling.LANCZOS)


def build_cards(report: dict) -> list[dict]:
    cards: list[dict] = []
    for entity in ENTITY_ORDER:
        item = report[entity]
        stats_line_1 = (
            f"data {fmt_count(item.get('data_points'))} | "
            f"bdf pts {fmt_count(item.get('bdf_points'))} | "
            f"tri {fmt_count(item.get('bdf_triangles'))}"
        )
        mapping = item.get("surface_mapping", {})
        stats_line_2 = (
            f"nearest p99 {fmt_dist(mapping.get('nearest_dist_p99'))} | "
            f"max {fmt_dist(mapping.get('nearest_dist_max'))}"
        )
        cards.append(
            {
                "entity": entity,
                "title": f"{entity} · Point Cloud",
                "filename": Path(item["point_png"]).name,
                "image_path": Path(item["point_png"]),
                "stats_1": stats_line_1,
                "stats_2": stats_line_2,
            }
        )
        cards.append(
            {
                "entity": entity,
                "title": f"{entity} · Surface",
                "filename": Path(item["surface_png"]).name,
                "image_path": Path(item["surface_png"]),
                "stats_1": stats_line_1,
                "stats_2": stats_line_2,
            }
        )
    return cards


def render_sheet(out_path: Path, report: dict) -> None:
    title_font = get_font(56, bold=True)
    subtitle_font = get_font(24)
    card_title_font = get_font(28, bold=True)
    file_font = get_font(22)
    meta_font = get_font(20)

    margin = 64
    gutter = 34
    header_h = 170
    cols = 3
    rows = 2
    card_w = 1330
    card_h = 1220

    canvas_w = margin * 2 + cols * card_w + (cols - 1) * gutter
    canvas_h = margin * 2 + header_h + rows * card_h + (rows - 1) * gutter

    canvas = Image.new("RGB", (canvas_w, canvas_h), COLORS["bg"])
    draw = ImageDraw.Draw(canvas)

    draw_text(draw, (margin, margin), "Von Mises Visualization Export", title_font, COLORS["ink"])
    draw_text(
        draw,
        (margin, margin + 74),
        f"Six captioned panels extracted from {HTML_PATH.relative_to(SCRIPT_DIR.parent)}",
        subtitle_font,
        COLORS["muted"],
    )
    draw_text(
        draw,
        (margin, margin + 110),
        "Top row: point cloud views. Bottom row: surface views. Columns follow the HTML report order.",
        subtitle_font,
        COLORS["muted"],
    )

    cards = build_cards(report)
    image_max = (card_w - 30, 770)

    def draw_card(card: dict, col: int, row: int) -> None:
        x0 = margin + col * (card_w + gutter)
        y0 = margin + header_h + row * (card_h + gutter)
        x1 = x0 + card_w
        y1 = y0 + card_h

        shadow = (x0 + 6, y0 + 10, x1 + 6, y1 + 10)
        draw.rounded_rectangle(shadow, radius=24, fill="#dbe6ef")
        draw.rounded_rectangle((x0, y0, x1, y1), radius=24, fill=COLORS["card"], outline=COLORS["line"], width=2)
        draw.rounded_rectangle((x0, y0, x1, y0 + 64), radius=24, fill=COLORS["band"])
        draw.rectangle((x0, y0 + 32, x1, y0 + 64), fill=COLORS["band"])

        draw_text(draw, (x0 + 20, y0 + 18), card["title"], card_title_font, COLORS["ink"])

        img_box = (x0 + 15, y0 + 82, x1 - 15, y0 + 82 + 810)
        draw.rounded_rectangle(img_box, radius=16, fill="#f7fbff", outline=COLORS["line"], width=2)

        img = fit_image(card["image_path"], image_max)
        px = img_box[0] + (img_box[2] - img_box[0] - img.width) // 2
        py = img_box[1] + (img_box[3] - img_box[1] - img.height) // 2
        canvas.paste(img, (px, py))

        text_y = img_box[3] + 18
        draw_text(draw, (x0 + 20, text_y), card["filename"], file_font, COLORS["accent"])
        draw_text(draw, (x0 + 20, text_y + 38), card["stats_1"], meta_font, COLORS["muted"])
        draw_text(draw, (x0 + 20, text_y + 70), card["stats_2"], meta_font, COLORS["muted"])

    for col, entity in enumerate(ENTITY_ORDER):
        point_card = cards[col * 2]
        surface_card = cards[col * 2 + 1]
        draw_card(point_card, col, 0)
        draw_card(surface_card, col, 1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export captioned von Mises panels to a single PNG.")
    parser.add_argument(
        "--out",
        default=str(EXPORTS / "von_mises_visualization_export.png"),
        help="Output PNG path.",
    )
    args = parser.parse_args()

    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    render_sheet(Path(args.out).resolve(), report)
    print(Path(args.out).resolve())


if __name__ == "__main__":
    main()
