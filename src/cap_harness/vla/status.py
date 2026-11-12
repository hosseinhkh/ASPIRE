"""Status of VLA benchmark runs, from their files (no LeRobot needed).

Usage::

    python -m cap_harness.vla.status        every run under outputs/vla
    python -m cap_harness.vla.status DIR    one run: progress, warnings, table, last log lines
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

from cap_harness.vla import students
from cap_harness.vla.libero_compat import REPOSITORY_ROOT
from cap_harness.vla.monitor import read_progress

VLA_OUTPUTS = REPOSITORY_ROOT / "outputs" / "vla"


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    minutes = seconds / 60
    return f"{minutes:.0f} min" if minutes < 120 else f"{minutes / 60:.1f} h"


def student_of(root: Path) -> str:
    """The student a run used (older runs name none and are SmolVLA's)."""
    try:
        record = json.loads((root / "run.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        record = {}
    return students.label(students.infer(record if isinstance(record, dict) else {})[0])


def run_line(root: Path) -> str:
    progress = read_progress(root) or {}
    done, planned = progress.get("done"), progress.get("planned")
    rate = progress.get("success_rate")
    return (
        f"{root.name:<44} {student_of(root):<8} {progress.get('status', 'no progress file'):<12} "
        f"{(f'{done}/{planned}' if planned else '-'):>11} "
        f"{(f'{100 * rate:.0f}%' if rate is not None else '-'):>7} "
        f"ETA {_duration(progress.get('eta_s') if progress.get('status') == 'running' else None):>8}  "
        f"{progress.get('updated', '')}"
        + (f"  ({len(progress['warnings'])} warning(s))" if progress.get("warnings") else "")
    )


def list_runs(base: Path = VLA_OUTPUTS) -> int:
    runs = sorted((p.parent for p in base.glob("*/run.json")), key=lambda p: p.stat().st_mtime)
    if not runs:
        print(f"no runs under {base}")
        return 0
    print(
        f"{'run':<44} {'student':<8} {'status':<12} {'done':>11} {'success':>7} {'':>12}  updated"
    )
    for root in runs:
        print(run_line(root))
    for parity in sorted(base.glob("parity-*.json")):
        try:
            report = json.loads(parity.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        parts = [f"environment {'PASS' if report['environment']['passed'] else 'FAIL'}"]
        if "policy" in report:
            parts.append(f"policy {'PASS' if report['policy']['passed'] else 'FAIL'}")
        print(f"parity {report.get('suite')}: " + ", ".join(parts))
    return 0


def show_run(root: Path) -> int:
    progress = read_progress(root)
    if progress is None:
        print(f"{root}: no progress.json (not a VLA benchmark run, or not started)")
        return 1
    print(run_line(root))
    print(
        f"started {progress.get('started')}, GPU {progress.get('gpu')}, pid {progress.get('pid')} "
        f"on {progress.get('host')}; {progress.get('s_per_episode') or '-'} s per episode"
    )
    if progress.get("current"):
        print(f"now running: {progress['current']}")
    for warning in progress.get("warnings") or []:
        print(f"WARNING: {warning}")
    if progress.get("error"):
        print(f"ERROR: {progress['error']}")
    results = root / "results.md"
    if results.is_file():
        print()
        print(results.read_text(encoding="utf-8"))
    log = root / "benchmark.log"
    if log.is_file():
        print("last log lines:")
        print("\n".join(log.read_text(encoding="utf-8").splitlines()[-8:]))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args:
        return list_runs()
    root = Path(args[0])
    if not root.is_absolute() and not root.exists():
        root = VLA_OUTPUTS / args[0]
    return show_run(root)


if __name__ == "__main__":
    raise SystemExit(main())
