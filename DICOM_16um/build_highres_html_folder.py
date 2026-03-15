#!/usr/bin/env python3
"""Build md/html folder for static_dynamics_high_resolution verification output."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List


ROOT = Path("/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/DICOM_16um")
EXPORTS = ROOT / "exports"
VERIFY_LOG = EXPORTS / "verify_highres_studies_stdout.log"
OUT_DIR = EXPORTS / "static_dynamics_high_resolution_html"
OUT_MD = OUT_DIR / "report.md"
OUT_HTML = OUT_DIR / "index.html"
OUT_JSON = OUT_DIR / "case_matrix.json"

INPUT_BDFS = [
    EXPORTS / "surface_mesh_smoothed.bdf",
    EXPORTS / "tooth_surface_uncompressed.bdf",
    EXPORTS / "tooth_surface_comsol_tet_vol.bdf",
]


@dataclass
class CaseMetric:
    entity: str
    material: str
    mode: str
    finite: bool
    max_disp: float
    max_von_mises: float
    max_strain: float
    tangent_modulus: float
    expr_vm: str
    non_zero: bool
    converged_non_zero: bool


CASE_RE = re.compile(
    r"CASE_METRIC\|entity=([^|]+)\|material=([^|]+)\|mode=([^|]+)\|finite=([^|]+)"
    r"\|max_disp=([^|]+)\|max_von_mises=([^|]+)\|max_strain=([^|]+)\|tangent_modulus=([^|]+)\|expr_vm=([^\n]+)"
)
SUMMARY_RE = re.compile(r"CASE_SUMMARY\|finite_cases=(\d+)\|total_cases=(\d+)")


def parse_log(text: str) -> tuple[List[CaseMetric], int, int]:
    rows: List[CaseMetric] = []
    for m in CASE_RE.finditer(text):
        entity, material, mode, finite_s, disp_s, vm_s, strain_s, tan_s, expr_vm = m.groups()
        disp = float(disp_s)
        vm = float(vm_s)
        strain = float(strain_s)
        tan = float(tan_s)
        finite = finite_s.lower() == "true"
        non_zero = abs(disp) > 0.0 and abs(vm) > 0.0 and abs(strain) > 0.0 and abs(tan) > 0.0
        rows.append(
            CaseMetric(
                entity=entity,
                material=material,
                mode=mode,
                finite=finite,
                max_disp=disp,
                max_von_mises=vm,
                max_strain=strain,
                tangent_modulus=tan,
                expr_vm=expr_vm.strip(),
                non_zero=non_zero,
                converged_non_zero=finite and non_zero,
            )
        )

    finite_cases = 0
    total_cases = len(rows)
    sm = SUMMARY_RE.search(text)
    if sm:
        finite_cases = int(sm.group(1))
        total_cases = int(sm.group(2))
    else:
        finite_cases = sum(1 for r in rows if r.finite)
    return rows, finite_cases, total_cases


def fmt(v: float) -> str:
    return f"{v:.6e}"


def build_md(rows: List[CaseMetric], finite_cases: int, total_cases: int) -> str:
    lines: List[str] = []
    lines.append("# Static Dynamics High Resolution Result Matrix")
    lines.append("")
    lines.append(f"- Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("- Model: `/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/DICOM_16um/exports/static_dynamics_high_resolution.mph`")
    lines.append("- Inputs preserved (unchanged):")
    for p in INPUT_BDFS:
        lines.append(f"  - `{p}`")
    lines.append("- Materials: `stvkirchhoff`, `mr2`, `mr5`")
    lines.append("- Modes: `linear`, `nonlinear`")
    lines.append(f"- Finite case count: **{finite_cases}/{total_cases}**")
    nz = sum(1 for r in rows if r.converged_non_zero)
    lines.append(f"- Finite + non-zero case count: **{nz}/{len(rows)}**")
    lines.append("")
    lines.append("| Entity | Material | Mode | finite | non_zero | max_disp (m) | max_von_mises (Pa) | max_strain | tangent_modulus (Pa) |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r.entity} | {r.material} | {r.mode} | {str(r.finite).lower()} | {str(r.non_zero).lower()} | "
            f"{fmt(r.max_disp)} | {fmt(r.max_von_mises)} | {fmt(r.max_strain)} | {fmt(r.tangent_modulus)} |"
        )
    return "\n".join(lines)


def build_html(rows: List[CaseMetric], finite_cases: int, total_cases: int) -> str:
    nz = sum(1 for r in rows if r.converged_non_zero)
    table_rows = []
    for r in rows:
        cls = "ok" if r.converged_non_zero else "bad"
        table_rows.append(
            "<tr class='{cls}'>"
            "<td>{entity}</td><td>{material}</td><td>{mode}</td>"
            "<td>{finite}</td><td>{non_zero}</td>"
            "<td>{disp}</td><td>{vm}</td><td>{strain}</td><td>{tan}</td>"
            "</tr>".format(
                cls=cls,
                entity=r.entity,
                material=r.material,
                mode=r.mode,
                finite=str(r.finite).lower(),
                non_zero=str(r.non_zero).lower(),
                disp=fmt(r.max_disp),
                vm=fmt(r.max_von_mises),
                strain=fmt(r.max_strain),
                tan=fmt(r.tangent_modulus),
            )
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>High Resolution Static Dynamics Matrix</title>
  <style>
    body {{
      margin: 0;
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      background: #f3f6fa;
      color: #132433;
    }}
    main {{
      max-width: 1400px;
      margin: 20px auto;
      padding: 0 16px 24px;
    }}
    h1 {{ margin: 0 0 10px; font-size: 1.7rem; }}
    .meta {{
      background: #fff;
      border: 1px solid #d7e2ea;
      border-radius: 10px;
      padding: 12px 14px;
      margin-bottom: 14px;
      line-height: 1.45;
    }}
    .meta code {{ color: #0a648f; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: #fff;
      border: 1px solid #d7e2ea;
      border-radius: 10px;
      overflow: hidden;
    }}
    th, td {{
      border-bottom: 1px solid #e4ecf2;
      padding: 8px 10px;
      font-size: 0.9rem;
      text-align: left;
      white-space: nowrap;
    }}
    th {{
      background: #ebf3f9;
      position: sticky;
      top: 0;
      z-index: 1;
    }}
    tr.ok td {{ background: #f8fcf9; }}
    tr.bad td {{ background: #fff6f6; }}
  </style>
</head>
<body>
  <main>
    <h1>Static Dynamics High Resolution Result Matrix</h1>
    <div class="meta">
      <div><strong>Generated:</strong> {datetime.now().isoformat(timespec='seconds')}</div>
      <div><strong>Model:</strong> <code>/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/DICOM_16um/exports/static_dynamics_high_resolution.mph</code></div>
      <div><strong>Finite cases:</strong> {finite_cases}/{total_cases}</div>
      <div><strong>Finite + non-zero cases:</strong> {nz}/{len(rows)}</div>
      <div><strong>Inputs preserved:</strong></div>
      <div><code>{INPUT_BDFS[0]}</code></div>
      <div><code>{INPUT_BDFS[1]}</code></div>
      <div><code>{INPUT_BDFS[2]}</code></div>
    </div>
    <table>
      <thead>
        <tr>
          <th>Entity</th>
          <th>Material</th>
          <th>Mode</th>
          <th>finite</th>
          <th>non_zero</th>
          <th>max_disp (m)</th>
          <th>max_von_mises (Pa)</th>
          <th>max_strain</th>
          <th>tangent_modulus (Pa)</th>
        </tr>
      </thead>
      <tbody>
        {"".join(table_rows)}
      </tbody>
    </table>
  </main>
</body>
</html>"""


def main() -> None:
    text = VERIFY_LOG.read_text(encoding="utf-8", errors="ignore")
    rows, finite_cases, total_cases = parse_log(text)
    if not rows:
        raise RuntimeError(f"No CASE_METRIC rows parsed from {VERIFY_LOG}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(build_md(rows, finite_cases, total_cases), encoding="utf-8")
    OUT_HTML.write_text(build_html(rows, finite_cases, total_cases), encoding="utf-8")
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "finite_cases": finite_cases,
        "total_cases": total_cases,
        "finite_non_zero_cases": sum(1 for r in rows if r.converged_non_zero),
        "rows": [asdict(r) for r in rows],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_HTML}")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
