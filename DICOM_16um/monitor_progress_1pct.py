#!/usr/bin/env python3
import argparse
import re
import time
from datetime import datetime
from pathlib import Path

PROGRESS_RE = re.compile(r"Current Progress:\s*([0-9]+)\s*%\s*-\s*(.*)$")


def ts_now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_latest_ts(ts_file: Path) -> str:
    return ts_file.read_text(encoding="utf-8").strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", default="")
    ap.add_argument(
        "--ts-file",
        default="DICOM_16um/exports/run_strict3bdf_contact_active_latest.ts",
    )
    ap.add_argument("--poll-seconds", type=float, default=2.0)
    args = ap.parse_args()

    ts = args.ts or read_latest_ts(Path(args.ts_file))
    run_log = Path(f"DICOM_16um/exports/run_strict3bdf_contact_active_{ts}.log")
    out_log = Path(f"DICOM_16um/exports/run_strict3bdf_contact_active_{ts}.progress_1pct.log")

    if not run_log.exists():
        raise SystemExit(f"run log missing: {run_log}")

    out_log.parent.mkdir(parents=True, exist_ok=True)

    last_pct = None
    last_stage = None
    with out_log.open("a", encoding="utf-8") as out:
        out.write(f"PROGRESS_MON_START|{ts_now()}|ts={ts}|run_log={run_log}\n")
        out.flush()

        pos = 0
        while True:
            text = run_log.read_text(encoding="utf-8", errors="ignore")
            if len(text) < pos:
                pos = 0
            chunk = text[pos:]
            pos = len(text)

            if chunk:
                for line in chunk.splitlines():
                    m = PROGRESS_RE.search(line)
                    if m:
                        pct = int(m.group(1))
                        stage = m.group(2).strip()
                        if pct != last_pct or stage != last_stage:
                            out.write(f"PROGRESS|{ts_now()}|{pct:03d}%|{stage}\n")
                            out.flush()
                            last_pct = pct
                            last_stage = stage
                    if "Total time:" in line:
                        out.write(f"PROGRESS_MON_END|{ts_now()}|total_time_seen\n")
                        out.flush()
                        return 0

            time.sleep(max(0.2, args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
