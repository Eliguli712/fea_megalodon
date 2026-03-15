#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent

results = {
    "timestamp": datetime.now().isoformat(),
    "studies": ["std1", "std2"],
    "std1": {
        "max_mises_pa": 2.256877880331911e7,
        "avg_mises_pa": 1.4560237479750863e6,
        "max_disp_m": 0.6077004139497453,
        "avg_disp_m": 0.23875382535860162,
    },
    "std2": {
        "max_mises_pa": 2.256877880331911e7,
        "avg_mises_pa": 1.4560237479750863e6,
        "max_disp_m": 0.6077004139497453,
        "avg_disp_m": 0.23875382535860162,
    },
}

(ROOT / "livytan_std12_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


def pa_to_mpa(v: float) -> float:
    return v / 1e6


def render_panel(study: str, data: dict[str, float], out_name: str) -> None:
    fig = plt.figure(figsize=(10.5, 7.2), dpi=180)
    fig.patch.set_facecolor("#f6f2ea")

    ax = fig.add_axes([0.08, 0.08, 0.84, 0.84])
    ax.set_facecolor("#fffaf0")
    for spine in ax.spines.values():
        spine.set_color("#d7cfc0")
    ax.set_xticks([])
    ax.set_yticks([])

    ax.text(0.02, 0.95, f"{study.upper()} Study Result Panel", fontsize=16, fontweight="bold", color="#1d2a33", transform=ax.transAxes)
    ax.text(0.02, 0.90, "Source: livytan_melville_teeth_volsolve.mph", fontsize=10, color="#5a6870", transform=ax.transAxes)

    max_mises = pa_to_mpa(data["max_mises_pa"])
    avg_mises = pa_to_mpa(data["avg_mises_pa"])
    max_disp = data["max_disp_m"]
    avg_disp = data["avg_disp_m"]

    # KPI cards
    card_specs = [
        (0.02, 0.62, 0.45, 0.22, "Max Von Mises", f"{max_mises:,.3f} MPa", "#2f5eaa"),
        (0.52, 0.62, 0.45, 0.22, "Avg Von Mises", f"{avg_mises:,.3f} MPa", "#1d7d75"),
        (0.02, 0.34, 0.45, 0.22, "Max Displacement", f"{max_disp:,.6f} m", "#bf5f34"),
        (0.52, 0.34, 0.45, 0.22, "Avg Displacement", f"{avg_disp:,.6f} m", "#8a5a2a"),
    ]

    for x, y, w, h, title, value, color in card_specs:
        rect = plt.Rectangle((x, y), w, h, transform=ax.transAxes, facecolor="#ffffff", edgecolor="#d7cfc0", linewidth=1.1)
        ax.add_patch(rect)
        ax.text(x + 0.03, y + h - 0.07, title, fontsize=10, color="#5a6870", transform=ax.transAxes)
        ax.text(x + 0.03, y + 0.06, value, fontsize=16, fontweight="bold", color=color, transform=ax.transAxes)

    # Mini bars
    bar_labels = ["Max Stress (MPa)", "Avg Stress (MPa)", "Max Disp (m)", "Avg Disp (m)"]
    bar_vals = [max_mises, avg_mises, max_disp, avg_disp]
    bar_colors = ["#2f5eaa", "#1d7d75", "#bf5f34", "#8a5a2a"]

    bx = fig.add_axes([0.12, 0.12, 0.76, 0.17])
    bx.set_facecolor("#fffaf0")
    bx.bar(np.arange(len(bar_vals)), bar_vals, color=bar_colors, width=0.62)
    bx.set_xticks(np.arange(len(bar_vals)))
    bx.set_xticklabels(bar_labels, fontsize=8)
    bx.tick_params(axis="y", labelsize=8)
    bx.grid(axis="y", alpha=0.25)
    bx.set_title("Raw value bars (unit-mixed, for quick check)", fontsize=9, color="#5a6870")

    fig.savefig(ROOT / out_name, dpi=180, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def render_comparison(out_name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=180)
    fig.patch.set_facecolor("#f6f2ea")

    studies = ["std1", "std2"]
    stress_max = [pa_to_mpa(results[s]["max_mises_pa"]) for s in studies]
    stress_avg = [pa_to_mpa(results[s]["avg_mises_pa"]) for s in studies]
    disp_max = [results[s]["max_disp_m"] for s in studies]
    disp_avg = [results[s]["avg_disp_m"] for s in studies]

    x = np.arange(len(studies))
    w = 0.35

    axes[0].bar(x - w / 2, stress_max, w, label="Max", color="#2f5eaa")
    axes[0].bar(x + w / 2, stress_avg, w, label="Avg", color="#1d7d75")
    axes[0].set_title("Von Mises Stress (MPa)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(studies)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].bar(x - w / 2, disp_max, w, label="Max", color="#bf5f34")
    axes[1].bar(x + w / 2, disp_avg, w, label="Avg", color="#8a5a2a")
    axes[1].set_title("Displacement (m)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(studies)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=8)

    fig.suptitle("std1 vs std2 Results Comparison", fontsize=14, color="#1d2a33")
    fig.savefig(ROOT / out_name, dpi=180, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


render_panel("std1", results["std1"], "livytan_std1_results_panel.png")
render_panel("std2", results["std2"], "livytan_std2_results_panel.png")
render_comparison("livytan_std12_comparison.png")

print("WROTE|livytan_std12_results.json")
print("WROTE|livytan_std1_results_panel.png")
print("WROTE|livytan_std2_results_panel.png")
print("WROTE|livytan_std12_comparison.png")
