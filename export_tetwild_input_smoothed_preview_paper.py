#!/usr/bin/env python3
"""Export a front-view TetWild preview with a projected major/minor scale grid."""

from __future__ import annotations

import argparse
import base64
import json
import math
import re
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from PIL import Image, ImageChops


PROJECT_ROOT = Path("/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg")
HTML_PATH = PROJECT_ROOT / "applied_forces" / "tetwild_input_smoothed_preview.html"
OUT_PATH = PROJECT_ROOT / "applied_forces" / "tetwild_input_smoothed_preview_paper.png"


def skip_ws(text: str, idx: int) -> int:
    while idx < len(text) and text[idx].isspace():
        idx += 1
    return idx


def parse_string(text: str, idx: int) -> int:
    quote = text[idx]
    idx += 1
    while idx < len(text):
        char = text[idx]
        if char == "\\":
            idx += 2
            continue
        if char == quote:
            return idx + 1
        idx += 1
    raise ValueError("unterminated string")


def parse_balanced(text: str, idx: int) -> tuple[str, int]:
    pairs = {"[": "]", "{": "}"}
    open_ch = text[idx]
    close_ch = pairs[open_ch]
    start = idx
    depth = 0
    in_string: str | None = None
    escaped = False

    while idx < len(text):
        char = text[idx]
        if in_string is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == in_string:
                in_string = None
        else:
            if char in ("'", '"'):
                in_string = char
            elif char == open_ch:
                depth += 1
            elif char == close_ch:
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1], idx + 1
        idx += 1
    raise ValueError("unterminated balanced block")


def extract_plotly_figure(html_path: Path) -> go.Figure:
    text = html_path.read_text(encoding="utf-8")
    match = re.search(r"Plotly\.newPlot\s*\(", text)
    if not match:
        raise ValueError("Plotly.newPlot call not found")

    idx = skip_ws(text, match.end())
    if text[idx] in ("'", '"'):
        idx = parse_string(text, idx)

    idx = skip_ws(text, idx)
    if text[idx] != ",":
        raise ValueError("Could not parse Plotly data argument")
    idx = skip_ws(text, idx + 1)
    data_json, idx = parse_balanced(text, idx)

    idx = skip_ws(text, idx)
    if text[idx] != ",":
        raise ValueError("Could not parse Plotly layout argument")
    idx = skip_ws(text, idx + 1)
    layout_json, idx = parse_balanced(text, idx)

    payload = {"data": json.loads(data_json), "layout": json.loads(layout_json)}
    return pio.from_json(json.dumps(payload))


def decode_array(value) -> np.ndarray:
    if isinstance(value, dict) and "bdata" in value:
        dtype_name = value.get("dtype", "")
        dtype_map = {
            "f8": "<f8",
            "f4": "<f4",
            "i8": "<i8",
            "i4": "<i4",
            "u4": "<u4",
            "u2": "<u2",
            "i2": "<i2",
        }
        if dtype_name not in dtype_map:
            raise ValueError(f"Unsupported encoded dtype: {dtype_name}")
        raw = base64.b64decode(value["bdata"])
        return np.frombuffer(raw, dtype=np.dtype(dtype_map[dtype_name])).astype(np.float64, copy=False)
    return np.asarray(value, dtype=np.float64)


def trace_xyz_arrays(trace) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if not hasattr(trace, "x") or not hasattr(trace, "y") or not hasattr(trace, "z"):
        return None
    x = decode_array(trace.x)
    y = decode_array(trace.y)
    z = decode_array(trace.z)
    if x.size == 0 or y.size == 0 or z.size == 0:
        return None
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if not np.any(mask):
        return None
    return x[mask], y[mask], z[mask]


def data_bounds(fig: go.Figure) -> tuple[float, float, float, float, float, float]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    zs: list[np.ndarray] = []
    for trace in fig.data:
        arrays = trace_xyz_arrays(trace)
        if arrays is None:
            continue
        x, y, z = arrays
        xs.append(x)
        ys.append(y)
        zs.append(z)
    if not xs:
        raise ValueError("No xyz traces found in figure")

    x_all = np.concatenate(xs)
    y_all = np.concatenate(ys)
    z_all = np.concatenate(zs)
    return (
        float(np.min(x_all)),
        float(np.max(x_all)),
        float(np.min(y_all)),
        float(np.max(y_all)),
        float(np.min(z_all)),
        float(np.max(z_all)),
    )


def projected_grid_traces(
    x_min: float,
    x_max: float,
    y_plane: float,
    y_line: float,
    z_min: float,
    z_max: float,
    *,
    major_step: float = 5.0,
    minor_step: float = 1.0,
) -> list[go.BaseTraceType]:
    x0 = math.floor(x_min)
    x1 = math.ceil(x_max)
    z0 = math.floor(z_min)
    z1 = math.ceil(z_max)

    plane_x, plane_z = np.meshgrid([x0, x1], [z0, z1])
    plane_y = np.full_like(plane_x, y_plane, dtype=np.float64)

    traces: list[go.BaseTraceType] = [
        go.Surface(
            x=plane_x,
            y=plane_y,
            z=plane_z,
            showscale=False,
            hoverinfo="skip",
            opacity=1.0,
            colorscale=[[0.0, "#f7d9c4"], [1.0, "#f7d9c4"]],
            lighting={"ambient": 1.0, "diffuse": 0.0, "specular": 0.0, "roughness": 1.0},
        )
    ]

    def grid_trace(values: np.ndarray, *, axis: str, color: str, width: float) -> go.Scatter3d:
        xs: list[float | None] = []
        ys: list[float | None] = []
        zs: list[float | None] = []
        if axis == "x":
            for value in values:
                xs.extend([float(value), float(value), None])
                ys.extend([float(y_line), float(y_line), None])
                zs.extend([float(z0), float(z1), None])
        else:
            for value in values:
                xs.extend([float(x0), float(x1), None])
                ys.extend([float(y_line), float(y_line), None])
                zs.extend([float(value), float(value), None])
        return go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            hoverinfo="skip",
            showlegend=False,
            line={"color": color, "width": width},
        )

    major_x = np.arange(x0, x1 + 0.001, major_step)
    major_z = np.arange(z0, z1 + 0.001, major_step)
    minor_x = np.arange(x0, x1 + 0.001, minor_step)
    minor_z = np.arange(z0, z1 + 0.001, minor_step)

    def remove_major(minor: np.ndarray, major: np.ndarray) -> np.ndarray:
        major_set = {round(v, 6) for v in major.tolist()}
        return np.asarray([v for v in minor.tolist() if round(v, 6) not in major_set], dtype=np.float64)

    minor_x = remove_major(minor_x, major_x)
    minor_z = remove_major(minor_z, major_z)

    # User requested denser minor grid with higher opacity than the major grid.
    traces.extend(
        [
            grid_trace(minor_x, axis="x", color="rgba(88, 66, 49, 0.82)", width=3.0),
            grid_trace(minor_z, axis="z", color="rgba(88, 66, 49, 0.82)", width=3.0),
            grid_trace(major_x, axis="x", color="rgba(92, 113, 136, 0.52)", width=6.0),
            grid_trace(major_z, axis="z", color="rgba(92, 113, 136, 0.52)", width=6.0),
        ]
    )
    return traces


def trim_uniform_background(path: Path, pad: int = 10) -> None:
    image = Image.open(path).convert("RGB")
    background = Image.new("RGB", image.size, image.getpixel((0, 0)))
    diff = ImageChops.difference(image, background)
    bbox = diff.getbbox()
    if bbox is None:
        return
    x0 = max(0, bbox[0] - pad)
    y0 = max(0, bbox[1] - pad)
    x1 = min(image.width, bbox[2] + pad)
    y1 = min(image.height, bbox[3] + pad)
    image.crop((x0, y0, x1, y1)).save(path)


def update_layout(fig: go.Figure) -> go.Figure:
    x_min, x_max, y_min, y_max, z_min, z_max = data_bounds(fig)
    x_pad = 4.0
    z_pad = 4.0
    x0 = math.floor(x_min - x_pad)
    x1 = math.ceil(x_max + x_pad)
    z0 = math.floor(z_min - z_pad)
    z1 = math.ceil(z_max + z_pad)
    y_range = y_max - y_min
    y_back = y_max + max(0.45, 0.08 * y_range)
    y_front = y_min - max(0.45, 0.08 * y_range)

    grid_traces = projected_grid_traces(x0, x1, y_back, y_front, z0, z1)
    fig = go.Figure(data=list(grid_traces) + list(fig.data), layout=fig.layout)

    fig.update_layout(
        width=3800,
        height=3400,
        paper_bgcolor="#fff7f1",
        plot_bgcolor="#fff7f1",
        margin={"l": 34, "r": 18, "t": 28, "b": 18},
        title={
            "text": "TetWild Tooth | buccolingual frontal view",
            "x": 0.5,
            "xanchor": "center",
            "y": 0.945,
            "yanchor": "top",
            "pad": {"t": 0, "b": 0},
            "font": {"size": 42, "color": "#1f2937"},
        },
        scene={
            "domain": {"x": [0.005, 0.995], "y": [0.015, 0.992]},
            "aspectmode": "manual",
            "aspectratio": {"x": (x1 - x0) / (z1 - z0), "y": 0.18, "z": 1.0},
            "camera": {
                "eye": {"x": 0.0, "y": -2.95, "z": 0.08},
                "up": {"x": 0, "y": 0, "z": 1},
                "center": {"x": 0, "y": 0, "z": 0},
                "projection": {"type": "orthographic"},
            },
            "xaxis": {
                "title": {"text": "x", "font": {"size": 30, "color": "#465769"}},
                "range": [x0, x1],
                "tick0": math.ceil(x0 / 5.0) * 5.0,
                "dtick": 5,
                "showbackground": False,
                "showgrid": False,
                "zeroline": False,
                "showticklabels": True,
                "ticks": "",
                "tickfont": {"size": 18, "color": "#627286"},
                "showspikes": False,
            },
            "yaxis": {
                "title": "",
                "range": [y_front, y_back],
                "showbackground": False,
                "showgrid": False,
                "zeroline": False,
                "showticklabels": False,
                "ticks": "",
                "showspikes": False,
            },
            "zaxis": {
                "title": {"text": "z", "font": {"size": 30, "color": "#465769"}},
                "range": [z0, z1],
                "tick0": math.ceil(z0 / 5.0) * 5.0,
                "dtick": 5,
                "showbackground": False,
                "showgrid": False,
                "zeroline": False,
                "showticklabels": True,
                "ticks": "",
                "tickfont": {"size": 18, "color": "#627286"},
                "showspikes": False,
            },
        },
        showlegend=False,
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Export TetWild front-view PNG with projected major/minor scale grid.")
    parser.add_argument("--html", default=str(HTML_PATH))
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()

    fig = extract_plotly_figure(Path(args.html))
    fig = update_layout(fig)
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(out_path), scale=2)
    trim_uniform_background(out_path)
    print(out_path)


if __name__ == "__main__":
    main()
