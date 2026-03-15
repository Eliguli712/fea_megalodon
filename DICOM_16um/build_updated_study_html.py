#!/usr/bin/env python3
import argparse
import datetime as dt
import re
from pathlib import Path

STUDY_ORDER = ["std1", "std_nh", "std_og", "std_mr2", "std_mr5", "std_pr"]
STUDY_TO_SOL = {
    "std1": "sol6",
    "std_nh": "sol1",
    "std_og": "sol2",
    "std_mr2": "sol3",
    "std_mr5": "sol4",
    "std_pr": "sol5",
}

STUDY_IMAGE_CANDIDATES = {
    "std1": ["std1_von_mises_contrast.png", "std1_von_mises.png"],
    "std_nh": ["std_nh_von_mises_contrast.png", "std_nh_von_mises.png"],
    "std_og": ["std_og_von_mises_contrast.png", "std_og_von_mises.png"],
    "std_mr2": ["std_mr2_mooney_rivlin_contrast.png", "std_mr2_von_mises.png"],
    "std_mr5": ["std_mr5_mooney_rivlin_contrast.png", "std_mr5_von_mises.png"],
    "std_pr": ["std_pr_von_mises_contrast.png", "std_pr_von_mises.png"],
}

CHECK_RE = re.compile(
    r"CHECK\|study=([^|]+)\|dataset=([^|]+)\|vm=([^|]+)\|um=([^|]+)\|finite_nonzero=([^|\s]+)"
)
RUN_RE = re.compile(r"STUDY_RUN\|([^|]+)\|ok=(true|false)(?:\|err=([^\n]+))?")
SUMMARY_RE = re.compile(
    r"SUMMARY\|finite_nonzero_studies=([0-9]+)\|total_target_studies=([0-9]+)"
)
DOF_RE = re.compile(r"Number of degrees of freedom solved for:\s*([0-9,]+)")

SOLVER_START_RE = re.compile(r"<---- Stationary Solver 1 in (.+?)\((sol[0-9]+)\)")
STARTED_RE = re.compile(r"Started at (.+)\.")
ENDED_RE = re.compile(r"Ended at (.+)\.")
SOL_TIME_RE = re.compile(r"Solution time:\s*([0-9]+)\s*s")


def parse_run_stdout(path: Path):
    text = path.read_text(errors="ignore")
    metrics = {}
    run_status = {}
    summary = {}

    for m in CHECK_RE.finditer(text):
        study = m.group(1)
        try:
            vm = float(m.group(3))
            um = float(m.group(4))
        except Exception:
            continue
        metrics[study] = {
            "dataset": m.group(2),
            "vm": vm,
            "um": um,
            "finite_nonzero": m.group(5).lower() == "true",
        }

    for m in RUN_RE.finditer(text):
        run_status[m.group(1)] = {
            "ok": m.group(2) == "true",
            "err": (m.group(3) or "").strip(),
        }

    m = SUMMARY_RE.search(text)
    if m:
        summary["finite_nonzero_studies"] = int(m.group(1))
        summary["total_target_studies"] = int(m.group(2))

    return metrics, run_status, summary


def parse_dof_and_solver_blocks(batchlog: Path):
    text = batchlog.read_text(errors="ignore")
    dof_vals = []
    for m in DOF_RE.finditer(text):
        try:
            dof_vals.append(int(m.group(1).replace(",", "")))
        except Exception:
            pass
    dof = max(dof_vals) if dof_vals else None

    blocks = {}
    current = None
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
                except Exception:
                    pass
        if not blocks[current]["ended"]:
            m = ENDED_RE.search(line)
            if m:
                blocks[current]["ended"] = m.group(1).strip()
    return dof, blocks


def fmt_sci(v):
    return f"{v:.6e}"


def fmt_int(v):
    return f"{v:,d}"


def pick_images(image_dir: Path):
    out = {}
    for study, names in STUDY_IMAGE_CANDIDATES.items():
        picked = None
        for name in names:
            p = image_dir / name
            if p.exists() and p.stat().st_size > 0:
                picked = name
                break
        out[study] = picked
    return out


def build_html(
    out_html: Path,
    model_path: str,
    run_stdout: Path,
    run_batchlog: Path,
    metrics: dict,
    run_status: dict,
    summary: dict,
    dof: int | None,
    blocks: dict,
    images: dict,
):
    generated = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dof_txt = fmt_int(dof) if dof is not None else "n/a"
    summary_txt = (
        f"{summary.get('finite_nonzero_studies', 0)}/{summary.get('total_target_studies', len(STUDY_ORDER))}"
    )

    rows = []
    for st in STUDY_ORDER:
        rs = run_status.get(st, {"ok": False, "err": "missing status"})
        mt = metrics.get(st)
        sol = STUDY_TO_SOL.get(st, "")
        blk = blocks.get(sol, {})
        vm = fmt_sci(mt["vm"]) if mt else "n/a"
        um = fmt_sci(mt["um"]) if mt else "n/a"
        finite = "true" if (mt and mt.get("finite_nonzero")) else "false"
        status = "COMPLETED" if rs.get("ok") else "FAILED"
        err = rs.get("err", "")
        err_cell = err if err else "-"
        tsec = blk.get("solve_time_s")
        ttxt = f"{tsec} s" if isinstance(tsec, int) else "n/a"
        rows.append(
            f"<tr><td><code>{st}</code></td>"
            f"<td><code>{sol}</code></td>"
            f"<td>{status}</td>"
            f"<td>{vm}</td>"
            f"<td>{um}</td>"
            f"<td>{finite}</td>"
            f"<td>{ttxt}</td>"
            f"<td>{blk.get('started','')}</td>"
            f"<td>{blk.get('ended','')}</td>"
            f"<td>{err_cell}</td></tr>"
        )

    cards = []
    rel_prefix = "../../applied_forces/exports/holocastic_full_body_images/"
    for st in STUDY_ORDER:
        img = images.get(st)
        src = rel_prefix + img if img else ""
        if img:
            cards.append(
                f"<figure><img src=\"{src}\" alt=\"{st} von Mises\">"
                f"<figcaption><code>{st}</code> | <code>{img}</code></figcaption></figure>"
            )
        else:
            cards.append(
                f"<figure><div class=\"missing\">No exported image found for <code>{st}</code></div>"
                f"<figcaption><code>{st}</code></figcaption></figure>"
            )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Updated High-Res Study Report</title>
  <style>
    :root {{
      --bg:#f3f7fb; --card:#fff; --line:#d6e2ec; --ink:#102738; --muted:#4d667a;
    }}
    * {{ box-sizing:border-box; }}
    body {{
      margin:0; color:var(--ink); background:radial-gradient(circle at 0% 0%,#e3eef7,var(--bg) 45%);
      font:15px/1.4 "Avenir Next","Segoe UI",sans-serif;
    }}
    .wrap {{ max-width:1420px; margin:24px auto 40px; padding:0 16px; }}
    h1 {{ margin:0 0 8px; font-size:1.9rem; }}
    .sub {{ margin:0 0 14px; color:var(--muted); }}
    .card {{
      background:var(--card); border:1px solid var(--line); border-radius:12px;
      box-shadow:0 4px 14px rgba(16,39,56,.08); margin-top:14px; overflow:hidden;
    }}
    .card h2 {{ margin:0; padding:10px 12px; font-size:1.03rem; border-bottom:1px solid var(--line); background:#f6fbff; }}
    .pad {{ padding:12px; }}
    .stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:10px; }}
    .stat {{ border:1px solid var(--line); border-radius:10px; background:#f9fcff; padding:10px; }}
    .stat .k {{ color:var(--muted); font-size:.86rem; }}
    .stat .v {{ font-weight:700; margin-top:3px; }}
    table {{ width:100%; border-collapse:collapse; font-size:.9rem; }}
    th,td {{ text-align:left; padding:8px 6px; border-bottom:1px solid #e4ecf3; vertical-align:top; }}
    th {{ color:#284a62; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:12px; }}
    figure {{ margin:0; border:1px solid var(--line); border-radius:10px; overflow:hidden; background:#fff; }}
    img {{ display:block; width:100%; height:220px; object-fit:contain; background:#f8fbff; }}
    .missing {{ display:flex; align-items:center; justify-content:center; height:220px; color:#8a1f1f; background:#fff6f6; }}
    figcaption {{ border-top:1px solid var(--line); padding:8px 10px; background:#f8fbff; color:#284a62; font-size:.85rem; }}
    code {{ color:#0a5d8a; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Updated Study Report (High Resolution)</h1>
    <p class="sub">
      Generated: <code>{generated}</code><br>
      Model: <code>{model_path}</code><br>
      Run stdout: <code>{run_stdout}</code><br>
      Run batch log: <code>{run_batchlog}</code>
    </p>

    <section class="card">
      <h2>Run Summary</h2>
      <div class="pad stats">
        <div class="stat"><div class="k">Target studies</div><div class="v">{", ".join(STUDY_ORDER)}</div></div>
        <div class="stat"><div class="k">Finite/nonzero completion</div><div class="v">{summary_txt}</div></div>
        <div class="stat"><div class="k">Solver DOF</div><div class="v">{dof_txt} (+1 internal where reported)</div></div>
      </div>
    </section>

    <section class="card">
      <h2>V/E/F and Boundary Data (BDF Cards + Geometry)</h2>
      <div class="pad">
        <table>
          <tr><th>Item</th><th>Value</th></tr>
          <tr><td><code>V</code> (GRID)</td><td>280,491</td></tr>
          <tr><td><code>E</code> (CBAR/CBEAM edges)</td><td>0</td></tr>
          <tr><td><code>F</code> (CTRIA3 boundary faces)</td><td>190,950</td></tr>
          <tr><td>Volume tets (CTETRA)</td><td>1,366,270</td></tr>
          <tr><td>Geometric boundary sides (<code>geom1</code>)</td><td>1 (single closed body)</td></tr>
          <tr><td>Boundary face elements</td><td>190,950</td></tr>
        </table>
      </div>
    </section>

    <section class="card">
      <h2>Study Results</h2>
      <div class="pad">
        <table>
          <thead>
            <tr>
              <th>Study</th><th>Solver</th><th>Status</th><th>Max von Mises (Pa)</th>
              <th>Max displacement (m)</th><th>Finite/Nonzero</th><th>Solve time</th>
              <th>Started</th><th>Ended</th><th>Error</th>
            </tr>
          </thead>
          <tbody>
            {"".join(rows)}
          </tbody>
        </table>
      </div>
    </section>

    <section class="card">
      <h2>Exported Study Images</h2>
      <div class="pad grid">
        {"".join(cards)}
      </div>
    </section>
  </div>
</body>
</html>
"""
    out_html.write_text(html)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-stdout", required=True)
    ap.add_argument("--run-batchlog", required=True)
    ap.add_argument("--output-html", required=True)
    ap.add_argument("--image-dir", default="applied_forces/exports/holocastic_full_body_images")
    ap.add_argument("--model-path", default="DICOM_16um/exports/static_dynamics_high_resolution_strict3bdf.mph")
    args = ap.parse_args()

    run_stdout = Path(args.run_stdout)
    run_batchlog = Path(args.run_batchlog)
    out_html = Path(args.output_html)
    image_dir = Path(args.image_dir)

    metrics, run_status, summary = parse_run_stdout(run_stdout)
    dof, blocks = parse_dof_and_solver_blocks(run_batchlog)
    images = pick_images(image_dir)

    build_html(
        out_html=out_html,
        model_path=args.model_path,
        run_stdout=run_stdout,
        run_batchlog=run_batchlog,
        metrics=metrics,
        run_status=run_status,
        summary=summary,
        dof=dof,
        blocks=blocks,
        images=images,
    )
    print(
        "UPDATED_STUDY_HTML|"
        f"path={out_html}|metrics={len(metrics)}|dof={dof}|images={sum(1 for v in images.values() if v)}"
    )


if __name__ == "__main__":
    main()
