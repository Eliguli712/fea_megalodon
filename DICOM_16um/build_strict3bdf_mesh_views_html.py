#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import html
import json
import re
from pathlib import Path


ROOT = Path("/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg")
EXPORTS = ROOT / "DICOM_16um" / "exports"
MODEL_PATH = EXPORTS / "static_dynamics_high_resolution_strict3bdf.mph"
VIEWS_STDOUT = EXPORTS / "add_strict3bdf_mesh_views.stdout.txt"
WAIT_LOG = EXPORTS / "continue_mesh3_then_studies.wait.log"
OUT_HTML = EXPORTS / "strict3bdf_mesh_views.html"
LATEST_RUN_TS = EXPORTS / "run_strict3bdf_overnight_updated_latest.ts"
UPDATED_STUDY_HTML = EXPORTS / "updated_study.html"
MESH_IMAGE_DIR = EXPORTS / "strict3bdf_mesh_images"
MESH_IMAGE_MANIFEST = MESH_IMAGE_DIR / "manifest.json"
STRESS_IMAGE_DIR = ROOT / "applied_forces" / "exports" / "holocastic_full_body_images"

VIEW_SETUP_RE = re.compile(r"VIEW_SETUP\|([^/]+)/([^|]+)\|ok=(true|false)(?:\|err=(.+))?")
COMP_VIEWS_RE = re.compile(r"(COMP[12]_VIEWS)\|\[(.*)\]")
SOURCE_COUNTS_RE = re.compile(r"SOURCE_BDF_COUNTS\|GRID=(\d+)\|CTRIA3=(\d+)\|CTETRA=(\d+)")
TARGET_COUNTS_RE = re.compile(r"TARGET_STAGE2_COUNTS\|V=(\d+)\|F=(\d+)")
NOTE_COUNTS_RE = re.compile(
    r"NOTE\|source_bdf_counts_do_not_match_stage2_target\|sourceV=(\d+)\|sourceF=(\d+)\|targetV=(\d+)\|targetF=(\d+)"
)
BDF_TO_OBJ_RE = re.compile(r"BDF_TO_OBJ_DONE\|obj=([^|]+)\|vertices=(\d+)\|faces=(\d+)")
STEP_RE = re.compile(r"STEP\|([^|]+)\|(.+)")
WAIT_RE = re.compile(r"(WATCH_START|WAITING_FOR|FOUND_VOL_BDF)\|(.+)")
CHECK_RE = re.compile(
    r"CHECK\|study=([^|]+)\|dataset=([^|]+)\|vm=([^|]+)\|um=([^|]+)\|finite_nonzero=([^|\s]+)"
)
RUN_RE = re.compile(r"STUDY_RUN\|([^|]+)\|ok=(true|false)(?:\|err=([^\n]+))?")
SUMMARY_RE = re.compile(
    r"SUMMARY\|finite_nonzero_studies=([0-9]+)\|total_target_studies=([0-9]+)"
)
DOF_RE = re.compile(r"Number of degrees of freedom solved for:\s*([0-9,]+)")

STUDY_ORDER = ["std1", "std_nh", "std_og", "std_mr2", "std_mr5", "std_pr"]
STUDY_IMAGE_CANDIDATES = {
    "std1": ["std1_von_mises_contrast.png", "std1_von_mises.png"],
    "std_nh": ["std_nh_von_mises_contrast.png", "std_nh_von_mises.png"],
    "std_og": ["std_og_von_mises_contrast.png", "std_og_von_mises.png"],
    "std_mr2": ["std_mr2_mooney_rivlin_contrast.png", "std_mr2_von_mises.png"],
    "std_mr5": ["std_mr5_mooney_rivlin_contrast.png", "std_mr5_von_mises.png"],
    "std_pr": ["std_pr_von_mises_contrast.png", "std_pr_von_mises.png"],
}
MESH_SLOT_ORDER = ["comp1_mesh1", "comp1_mesh2", "comp2_mesh3"]
MESH_SLOT_LABELS = {
    "comp1_mesh1": "comp1 / mesh1",
    "comp1_mesh2": "comp1 / mesh2",
    "comp2_mesh3": "comp2 / mesh3",
}


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def pick_latest(pattern: str) -> Path | None:
    matches = [p for p in EXPORTS.glob(pattern) if p.is_file()]
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def latest_study_logs() -> tuple[Path | None, Path | None]:
    if LATEST_RUN_TS.exists():
        ts = LATEST_RUN_TS.read_text(encoding="utf-8", errors="ignore").strip()
        if ts:
            stdout = EXPORTS / f"run_strict3bdf_overnight_updated_{ts}.stdout.txt"
            batchlog = EXPORTS / f"run_strict3bdf_overnight_updated_{ts}.log"
            return (stdout if stdout.exists() else None, batchlog if batchlog.exists() else None)
    return (
        pick_latest("run_strict3bdf_overnight_updated_*.stdout.txt"),
        pick_latest("run_strict3bdf_overnight_updated_*.log"),
    )


def latest_mesh_run_log() -> Path | None:
    return pick_latest("run_uncompressed_mesh3_filefirst_*.log")


def parse_views(text: str) -> tuple[list[dict], dict[str, list[str]]]:
    rows: list[dict] = []
    comp_views: dict[str, list[str]] = {"COMP1_VIEWS": [], "COMP2_VIEWS": []}
    for m in VIEW_SETUP_RE.finditer(text):
        comp, view, ok, err = m.groups()
        rows.append(
            {
                "component": comp,
                "view": view,
                "ok": ok == "true",
                "err": (err or "").strip(),
            }
        )
    for m in COMP_VIEWS_RE.finditer(text):
        key, raw = m.groups()
        vals = [v.strip() for v in raw.split(",") if v.strip()]
        comp_views[key] = vals
    return rows, comp_views


def parse_mesh_run(text: str) -> dict:
    out: dict[str, object] = {
        "source_grid": None,
        "source_tri": None,
        "source_tet": None,
        "target_v": None,
        "target_f": None,
        "note": "",
        "obj_path": "",
        "obj_vertices": None,
        "obj_faces": None,
        "steps": [],
    }
    m = TARGET_COUNTS_RE.search(text)
    if m:
        out["target_v"] = int(m.group(1))
        out["target_f"] = int(m.group(2))
    m = SOURCE_COUNTS_RE.search(text)
    if m:
        out["source_grid"] = int(m.group(1))
        out["source_tri"] = int(m.group(2))
        out["source_tet"] = int(m.group(3))
    m = NOTE_COUNTS_RE.search(text)
    if m:
        out["note"] = (
            f"sourceV={m.group(1)}, sourceF={m.group(2)}; "
            f"targetV={m.group(3)}, targetF={m.group(4)}"
        )
    m = BDF_TO_OBJ_RE.search(text)
    if m:
        out["obj_path"] = m.group(1)
        out["obj_vertices"] = int(m.group(2))
        out["obj_faces"] = int(m.group(3))
    for m in STEP_RE.finditer(text):
        out["steps"].append({"step": m.group(1), "time": m.group(2).strip()})
    return out


def parse_wait_log(text: str) -> list[str]:
    rows: list[str] = []
    for m in WAIT_RE.finditer(text):
        rows.append(f"{m.group(1)} | {m.group(2)}")
    return rows[-12:]


def parse_run_stdout(path: Path | None) -> tuple[dict, dict, dict]:
    if path is None or not path.exists():
        return {}, {}, {}
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
        run_status[m.group(1)] = {"ok": m.group(2) == "true", "err": (m.group(3) or "").strip()}
    m = SUMMARY_RE.search(text)
    if m:
        summary["finite_nonzero_studies"] = int(m.group(1))
        summary["total_target_studies"] = int(m.group(2))
    return metrics, run_status, summary


def parse_dof(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    text = path.read_text(errors="ignore")
    vals: list[int] = []
    for m in DOF_RE.finditer(text):
        try:
            vals.append(int(m.group(1).replace(",", "")))
        except Exception:
            pass
    return max(vals) if vals else None


def fmt_int(v: object) -> str:
    if isinstance(v, int):
        return f"{v:,d}"
    return "n/a"


def fmt_sci(v: object) -> str:
    if isinstance(v, float):
        return f"{v:.6e}"
    return "n/a"


def load_mesh_manifest() -> dict:
    if not MESH_IMAGE_MANIFEST.exists():
        return {}
    try:
        return json.loads(MESH_IMAGE_MANIFEST.read_text(encoding="utf-8"))
    except Exception:
        return {}


def pick_stress_images() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for study, names in STUDY_IMAGE_CANDIDATES.items():
        picked = None
        for name in names:
            path = STRESS_IMAGE_DIR / name
            if path.exists() and path.stat().st_size > 0:
                picked = name
                break
        out[study] = picked
    return out


def render_html(
    views: list[dict],
    comp_views: dict[str, list[str]],
    mesh_run: dict,
    wait_lines: list[str],
    mesh_run_log: Path | None,
    run_stdout: Path | None,
    run_batchlog: Path | None,
    run_metrics: dict,
    run_status: dict,
    run_summary: dict,
    dof: int | None,
    mesh_manifest: dict,
    stress_images: dict[str, str | None],
) -> str:
    generated = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model_mtime = dt.datetime.fromtimestamp(MODEL_PATH.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    model_size = f"{MODEL_PATH.stat().st_size:,d} bytes"
    msh_path = EXPORTS / "tooth_surface_uncompressed_mesh3_tet_vol.msh"
    bdf_path = EXPORTS / "tooth_surface_uncompressed_mesh3_tet_vol.bdf"
    msh_exists = msh_path.exists() and msh_path.stat().st_size > 0
    bdf_exists = bdf_path.exists() and bdf_path.stat().st_size > 0

    view_cards = []
    mesh_label = {
        "view_mesh1_hr": "comp1 / mesh1",
        "view_mesh2_hr": "comp1 / mesh2",
        "view_mesh3_hr": "comp2 / mesh3",
    }
    for row in views:
        label = mesh_label.get(row["view"], row["view"])
        status = "READY" if row["ok"] else "ERROR"
        err = html.escape(row["err"]) if row["err"] else "None"
        view_cards.append(
            f"<div class='tile'>"
            f"<div class='eyebrow'>{html.escape(row['component'])}</div>"
            f"<h3><code>{html.escape(row['view'])}</code></h3>"
            f"<p class='target'>{html.escape(label)}</p>"
            f"<p class='status'>{status}</p>"
            f"<p class='meta'>Error: {err}</p>"
            f"</div>"
        )

    view_inventory = (
        f"<tr><td><code>comp1</code></td><td>{html.escape(', '.join(comp_views.get('COMP1_VIEWS', [])) or 'n/a')}</td></tr>"
        f"<tr><td><code>comp2</code></td><td>{html.escape(', '.join(comp_views.get('COMP2_VIEWS', [])) or 'n/a')}</td></tr>"
    )

    steps_html = "".join(
        f"<tr><td><code>{html.escape(s['step'])}</code></td><td>{html.escape(s['time'])}</td></tr>"
        for s in mesh_run["steps"]
    ) or "<tr><td colspan='2'>No mesh-run steps recorded yet.</td></tr>"

    wait_html = "".join(f"<li><code>{html.escape(line)}</code></li>" for line in wait_lines)
    if not wait_html:
        wait_html = "<li><code>No watcher activity recorded.</code></li>"

    study_rows = []
    for study in STUDY_ORDER:
        rs = run_status.get(study)
        mt = run_metrics.get(study)
        status = "COMPLETED" if rs and rs.get("ok") else ("FAILED" if rs else "pending")
        study_rows.append(
            "<tr>"
            f"<td><code>{study}</code></td>"
            f"<td>{status}</td>"
            f"<td>{fmt_sci(mt['vm']) if mt else 'n/a'}</td>"
            f"<td>{fmt_sci(mt['um']) if mt else 'n/a'}</td>"
            f"<td>{'true' if mt and mt.get('finite_nonzero') else 'false'}</td>"
            f"<td>{html.escape((rs or {}).get('err', '') or '-')}</td>"
            "</tr>"
        )

    mesh_cards = []
    slots = mesh_manifest.get("slots", {}) if isinstance(mesh_manifest, dict) else {}
    for slot in MESH_SLOT_ORDER:
        entry = slots.get(slot, {})
        file_name = entry.get("file", "")
        status = entry.get("status", "pending")
        rel_src = f"strict3bdf_mesh_images/{file_name}" if file_name else ""
        source_name = entry.get("source_name", "pending")
        faces_used = entry.get("faces_used")
        faces_total = entry.get("faces_total")
        meta = (
            f"source={source_name}"
            + (
                f" | sampled faces={fmt_int(faces_used)} / {fmt_int(faces_total)}"
                if isinstance(faces_used, int) and isinstance(faces_total, int)
                else ""
            )
        )
        if file_name and (MESH_IMAGE_DIR / file_name).exists():
            mesh_cards.append(
                f"<figure><img src=\"{html.escape(rel_src)}\" alt=\"{html.escape(MESH_SLOT_LABELS[slot])}\">"
                f"<figcaption><code>{html.escape(MESH_SLOT_LABELS[slot])}</code> | {html.escape(meta)}</figcaption></figure>"
            )
        else:
            mesh_cards.append(
                f"<figure><div class=\"missing\">Mesh preview {html.escape(status)}</div>"
                f"<figcaption><code>{html.escape(MESH_SLOT_LABELS[slot])}</code> | {html.escape(meta)}</figcaption></figure>"
            )

    stress_cards = []
    for study in STUDY_ORDER:
        img = stress_images.get(study)
        rel_src = f"../../applied_forces/exports/holocastic_full_body_images/{img}" if img else ""
        if img:
            stress_cards.append(
                f"<figure><img src=\"{html.escape(rel_src)}\" alt=\"{html.escape(study)} stress\">"
                f"<figcaption><code>{html.escape(study)}</code> | <code>{html.escape(img)}</code></figcaption></figure>"
            )
        else:
            stress_cards.append(
                f"<figure><div class=\"missing\">Stress image pending</div>"
                f"<figcaption><code>{html.escape(study)}</code></figcaption></figure>"
            )

    solve_summary = (
        f"{run_summary.get('finite_nonzero_studies', 0)}/{run_summary.get('total_target_studies', len(STUDY_ORDER))}"
        if run_summary
        else "pending"
    )
    updated_href = "updated_study.html" if UPDATED_STUDY_HTML.exists() else ""
    updated_link = (
        '<a href="updated_study.html">Open detailed study report</a>'
        if updated_href
        else "Detailed study report pending."
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Strict3BDF Mesh Views</title>
  <style>
    :root {{
      --bg:#f3f7fb;
      --ink:#102738;
      --muted:#4b6579;
      --line:#d7e1ea;
      --card:#ffffff;
      --accent:#0a6b8f;
      --warn:#8c5c00;
    }}
    * {{ box-sizing:border-box; }}
    body {{
      margin:0;
      color:var(--ink);
      background:
        radial-gradient(circle at 0% 0%, #e4edf6, transparent 38%),
        linear-gradient(180deg, #f6f9fc, #edf3f8);
      font:15px/1.45 "Avenir Next","Segoe UI",sans-serif;
    }}
    a {{ color:#0a628f; }}
    .wrap {{ max-width: 1380px; margin: 28px auto 40px; padding: 0 18px; }}
    h1 {{ margin:0 0 8px; font-size:2rem; letter-spacing:.01em; }}
    .sub {{ margin:0; color:var(--muted); }}
    .card {{
      margin-top:14px;
      background:var(--card);
      border:1px solid var(--line);
      border-radius:14px;
      box-shadow:0 10px 24px rgba(15, 39, 56, 0.07);
      overflow:hidden;
    }}
    .card h2 {{
      margin:0;
      padding:12px 14px;
      font-size:1rem;
      border-bottom:1px solid var(--line);
      background:linear-gradient(90deg, #eef6fb, #f9fcff);
    }}
    .pad {{ padding:14px; }}
    .stats {{
      display:grid;
      grid-template-columns:repeat(auto-fit, minmax(220px, 1fr));
      gap:10px;
    }}
    .stat {{
      border:1px solid var(--line);
      border-radius:12px;
      padding:10px 12px;
      background:#f8fbff;
    }}
    .stat .k {{ color:var(--muted); font-size:.85rem; }}
    .stat .v {{ font-weight:700; margin-top:4px; }}
    .tiles {{
      display:grid;
      grid-template-columns:repeat(auto-fit, minmax(260px, 1fr));
      gap:12px;
    }}
    .tile {{
      border:1px solid var(--line);
      border-radius:12px;
      padding:14px;
      background:linear-gradient(180deg, #ffffff, #f8fbff);
    }}
    .tile h3 {{ margin:4px 0 6px; font-size:1rem; }}
    .eyebrow {{ color:var(--accent); font-size:.78rem; font-weight:700; text-transform:uppercase; letter-spacing:.08em; }}
    .target {{ margin:0 0 6px; font-weight:600; }}
    .status {{ margin:0 0 6px; color:#0e6a4d; font-weight:700; }}
    .meta {{ margin:0; color:var(--muted); font-size:.9rem; }}
    table {{ width:100%; border-collapse:collapse; }}
    th, td {{ text-align:left; padding:8px 6px; border-bottom:1px solid #e5edf4; vertical-align:top; }}
    th {{ color:#284962; font-weight:650; }}
    code {{ color:#0a628f; }}
    .mono-list {{ margin:0; padding-left:18px; }}
    .mono-list li {{ margin:5px 0; }}
    .note {{
      border:1px solid #edd9aa;
      background:#fff9ea;
      color:var(--warn);
      border-radius:12px;
      padding:12px 14px;
    }}
    .gallery {{
      display:grid;
      grid-template-columns:repeat(auto-fit, minmax(320px, 1fr));
      gap:14px;
    }}
    figure {{
      margin:0;
      border:1px solid var(--line);
      border-radius:12px;
      overflow:hidden;
      background:#fff;
    }}
    img {{
      display:block;
      width:100%;
      height:260px;
      object-fit:contain;
      background:#f8fbff;
    }}
    figcaption {{
      border-top:1px solid var(--line);
      padding:8px 10px;
      background:#f8fbff;
      color:#284962;
      font-size:.85rem;
    }}
    .missing {{
      display:flex;
      align-items:center;
      justify-content:center;
      height:260px;
      color:#8a1f1f;
      background:#fff6f6;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Strict3BDF Mesh Views</h1>
    <p class="sub">
      Generated: <code>{html.escape(generated)}</code><br>
      Model: <code>{html.escape(str(MODEL_PATH))}</code><br>
      Mesh run log: <code>{html.escape(str(mesh_run_log) if mesh_run_log else 'pending')}</code><br>
      Study stdout: <code>{html.escape(str(run_stdout) if run_stdout else 'pending')}</code><br>
      Study batch log: <code>{html.escape(str(run_batchlog) if run_batchlog else 'pending')}</code><br>
      {updated_link}
    </p>

    <section class="card">
      <h2>Model Snapshot</h2>
      <div class="pad stats">
        <div class="stat"><div class="k">Model modified</div><div class="v">{html.escape(model_mtime)}</div></div>
        <div class="stat"><div class="k">Model size</div><div class="v">{html.escape(model_size)}</div></div>
        <div class="stat"><div class="k">Mesh `.msh` output</div><div class="v">{'present' if msh_exists else 'pending'}</div></div>
        <div class="stat"><div class="k">Tet `.bdf` output</div><div class="v">{'present' if bdf_exists else 'pending'}</div></div>
        <div class="stat"><div class="k">Finite/nonzero solved studies</div><div class="v">{html.escape(solve_summary)}</div></div>
        <div class="stat"><div class="k">Observed DOF</div><div class="v">{fmt_int(dof)}</div></div>
      </div>
    </section>

    <section class="card">
      <h2>Saved View Tags</h2>
      <div class="pad tiles">
        {''.join(view_cards)}
      </div>
    </section>

    <section class="card">
      <h2>Component View Inventory</h2>
      <div class="pad">
        <table>
          <thead>
            <tr><th>Component</th><th>Views present in model</th></tr>
          </thead>
          <tbody>
            {view_inventory}
          </tbody>
        </table>
      </div>
    </section>

    <section class="card">
      <h2>High-Resolution Uncompressed Mesh Run</h2>
      <div class="pad stats">
        <div class="stat"><div class="k">Source GRID</div><div class="v">{fmt_int(mesh_run['source_grid'])}</div></div>
        <div class="stat"><div class="k">Source CTRIA3</div><div class="v">{fmt_int(mesh_run['source_tri'])}</div></div>
        <div class="stat"><div class="k">Source CTETRA</div><div class="v">{fmt_int(mesh_run['source_tet'])}</div></div>
        <div class="stat"><div class="k">BDF to OBJ</div><div class="v">{fmt_int(mesh_run['obj_vertices'])} / {fmt_int(mesh_run['obj_faces'])}</div></div>
      </div>
      <div class="pad">
        <div class="note">
          Current run is sourced from <code>tooth_surface_uncompressed.bdf</code>. The previous TetWild-reduced target was
          <code>V={fmt_int(mesh_run['target_v'])}</code>, <code>F={fmt_int(mesh_run['target_f'])}</code>. Logged comparison:
          <code>{html.escape(mesh_run['note'] or 'no mismatch note recorded')}</code>.
        </div>
      </div>
      <div class="pad">
        <table>
          <thead>
            <tr><th>Step</th><th>Timestamp</th></tr>
          </thead>
          <tbody>
            {steps_html}
          </tbody>
        </table>
      </div>
    </section>

    <section class="card">
      <h2>Latest Solve Metrics</h2>
      <div class="pad">
        <table>
          <thead>
            <tr>
              <th>Study</th><th>Status</th><th>Max von Mises (Pa)</th>
              <th>Max displacement (m)</th><th>Finite/Nonzero</th><th>Error</th>
            </tr>
          </thead>
          <tbody>
            {''.join(study_rows)}
          </tbody>
        </table>
      </div>
    </section>

    <section class="card">
      <h2>Mesh Preview Gallery</h2>
      <div class="pad gallery">
        {''.join(mesh_cards)}
      </div>
    </section>

    <section class="card">
      <h2>Stress Export Gallery</h2>
      <div class="pad gallery">
        {''.join(stress_cards)}
      </div>
    </section>

    <section class="card">
      <h2>Watcher Status</h2>
      <div class="pad">
        <ul class="mono-list">
          {wait_html}
        </ul>
      </div>
    </section>
  </div>
</body>
</html>
"""


def main() -> None:
    mesh_run_log = latest_mesh_run_log()
    run_stdout, run_batchlog = latest_study_logs()
    views_text = read_text(VIEWS_STDOUT)
    mesh_text = read_text(mesh_run_log) if mesh_run_log else ""
    wait_text = read_text(WAIT_LOG)
    views, comp_views = parse_views(views_text)
    mesh_run = parse_mesh_run(mesh_text)
    wait_lines = parse_wait_log(wait_text)
    run_metrics, run_status, run_summary = parse_run_stdout(run_stdout)
    dof = parse_dof(run_batchlog)
    mesh_manifest = load_mesh_manifest()
    stress_images = pick_stress_images()
    OUT_HTML.write_text(
        render_html(
            views=views,
            comp_views=comp_views,
            mesh_run=mesh_run,
            wait_lines=wait_lines,
            mesh_run_log=mesh_run_log,
            run_stdout=run_stdout,
            run_batchlog=run_batchlog,
            run_metrics=run_metrics,
            run_status=run_status,
            run_summary=run_summary,
            dof=dof,
            mesh_manifest=mesh_manifest,
            stress_images=stress_images,
        ),
        encoding="utf-8",
    )
    print(f"WROTE|{OUT_HTML}")


if __name__ == "__main__":
    main()
