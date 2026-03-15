#!/usr/bin/env python3
import argparse
import datetime as dt
import re
from pathlib import Path

CHECK_RE = re.compile(r"CHECK\|study=([^|]+)\|dataset=([^|]+)\|vm=([^|]+)\|um=([^|]+)\|finite_nonzero=([^|\s]+)")
DOF_RE = re.compile(r"Number of degrees of freedom solved for:\s*([0-9,]+)")


def parse_metrics(run_stdout: Path):
    metrics = {}
    text = run_stdout.read_text(errors="ignore")
    for m in CHECK_RE.finditer(text):
        st = m.group(1)
        try:
            vm = float(m.group(3))
            um = float(m.group(4))
        except Exception:
            continue
        metrics[st] = {"vm": vm, "um": um, "finite": m.group(5).lower() == "true"}
    return metrics


def parse_dof(run_batchlog: Path):
    text = run_batchlog.read_text(errors="ignore")
    vals = []
    for m in DOF_RE.finditer(text):
        raw = m.group(1).replace(",", "")
        try:
            vals.append(int(raw))
        except Exception:
            pass
    return max(vals) if vals else None


def fmt_sci(v):
    return f"{v:.3e}"


def fmt_int(v):
    return f"{v:,d}"


def update_static_html(path: Path, metrics: dict, dof: int | None):
    s = path.read_text(errors="ignore")

    today = dt.date.today().isoformat()
    s = re.sub(
        r"Latest strict3bdf checkpoint refresh: <code>[^<]+</code>",
        f"Latest strict3bdf checkpoint refresh: <code>{today}</code>",
        s,
    )

    order = ["std1", "std_nh", "std_og", "std_mr2", "std_mr5", "std_pr"]
    for st in order:
        if st not in metrics:
            continue
        vm = fmt_sci(metrics[st]["vm"])
        um = fmt_sci(metrics[st]["um"])
        dof_txt = fmt_int(dof) if dof is not None else "n/a"
        row_re = re.compile(
            rf"<tr><td><code>{re.escape(st)}</code></td><td>[^<]+</td><td>[^<]+</td><td>[^<]+</td></tr>"
        )
        new_row = f"<tr><td><code>{st}</code></td><td>{vm}</td><td>{um}</td><td>{dof_txt}</td></tr>"
        s = row_re.sub(new_row, s)

    if dof is not None:
        s = re.sub(
            r"Observed high-res DOF</td><td>[^<]+</td>",
            f"Observed high-res DOF</td><td>{fmt_int(dof)} (+1 internal)</td>",
            s,
        )
        s = re.sub(
            r"Solver DOF \(saved \| high-res std1 run\)</td><td id=\"dofVal\">[^<]+</td>",
            f"Solver DOF (saved | high-res std1 run)</td><td id=\"dofVal\">{fmt_int(dof)} | {fmt_int(dof)}</td>",
            s,
        )

    if "std1" in metrics:
        vm = fmt_sci(metrics["std1"]["vm"])
        um = fmt_sci(metrics["std1"]["um"])
        finite = "true" if metrics["std1"]["finite"] else "false"
        s = re.sub(
            r"Saved dataset check \(<code>dset6</code>\)</td><td><code>vm=[^<]+</code>, <code>um=[^<]+</code>, finite/nonzero=<code>[^<]+</code></td>",
            f"Saved dataset check (<code>dset6</code>)</td><td><code>vm={vm} Pa</code>, <code>um={um} m</code>, finite/nonzero=<code>{finite}</code></td>",
            s,
        )

    path.write_text(s)


def update_vm_html(path: Path, dof: int | None):
    s = path.read_text(errors="ignore")
    if dof is not None:
        s = re.sub(
            r"observed DOF <code>[^<]+</code> for <code>std1</code> solve",
            f"observed DOF <code>{fmt_int(dof)}</code> for <code>std1</code> solve",
            s,
        )
    path.write_text(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-stdout", required=True)
    ap.add_argument("--run-batchlog", required=True)
    ap.add_argument("--static-html", required=True)
    ap.add_argument("--vm-html", required=True)
    args = ap.parse_args()

    run_stdout = Path(args.run_stdout)
    run_batchlog = Path(args.run_batchlog)
    static_html = Path(args.static_html)
    vm_html = Path(args.vm_html)

    metrics = parse_metrics(run_stdout)
    dof = parse_dof(run_batchlog)

    update_static_html(static_html, metrics, dof)
    update_vm_html(vm_html, dof)

    print(f"HTML_UPDATE|metrics_count={len(metrics)}|dof={dof}")


if __name__ == "__main__":
    main()
