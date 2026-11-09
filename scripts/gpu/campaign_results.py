"""Held-out success per task for one or more campaigns, side by side with the paper.

Usage: python scripts/gpu/campaign_results.py outputs/runs/<campaign> [outputs/runs/<campaign> ...]

Reads each task's Stage-2 held-out runs (task_<id>/validation/), the campaign's cost
(result.json) and its settings (session.json: memory, library, harness).
"""

from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys

# Claude Opus 4.6, held-out success over 50 seeds (arXiv 2607.00272).
PAPER = {
    "libero_goal_swap": {
        "empty library (Table 19)": [0, 0, 94, 0, 96, 0, 82, 100, 100, 84],
        "library carried (Table 10)": [2, 56, 24, 86, 74, 0, 74, 68, 36, 46],
    },
    "libero_object_swap": {
        "library carried (Table 10)": [100, 100, 96, 100, 92, 98, 100, 92, 100, 100],
    },
}


def read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def held_out(task_dir: Path) -> tuple[int, int]:
    outcomes = [read_json(p) for p in (task_dir / "validation").rglob("outcome.json")]
    outcomes = [o for o in outcomes if o]
    return sum(1 for o in outcomes if o.get("task_success") is True), len(outcomes)


def main() -> int:
    roots = [Path(p) for p in sys.argv[1:]]
    if not roots:
        raise SystemExit(__doc__)
    columns, suites = [], set()
    for root in roots:
        meta = read_json(root / "session.json")
        # A resumed campaign keeps each earlier part as result.partN.json.
        parts = [read_json(p) for p in sorted(root.glob("result*.json"))]
        result = {
            "total_cost_usd": sum(float(r.get("total_cost_usd") or 0) for r in parts),
            "duration_ms": sum(float(r.get("duration_ms") or 0) for r in parts),
            "parts": len(parts),
        }
        tasks = {}
        for task_dir in sorted(root.glob("*/*/task_*")):
            suites.add(task_dir.parent.name)
            tasks[(task_dir.parent.name, int(task_dir.name.split("_")[1]))] = held_out(task_dir)
        columns.append({"name": meta.get("name") or root.name, "meta": meta, "result": result,
                        "tasks": tasks})  # fmt: skip
    keys = sorted({k for c in columns for k in c["tasks"]})
    print("Held-out success per task (successes/seeds)\n")
    header = f"{'task':<24}" + "".join(f"{c['name'][:22]:>24}" for c in columns)
    references = [
        (s, label, values) for s in sorted(suites) for label, values in PAPER.get(s, {}).items()
    ]
    header += "".join(f"{'paper ' + label.split(' (')[0][:16]:>24}" for _, label, _ in references)
    print(header)
    for suite, task_id in keys:
        row = f"{suite + ':' + str(task_id):<24}"
        for c in columns:
            ok, n = c["tasks"].get((suite, task_id), (0, 0))
            row += f"{(f'{ok}/{n} ({100 * ok / n:.0f}%)' if n else '-'):>24}"
        for ref_suite, _, values in references:
            row += f"{(f'{values[task_id]}%' if ref_suite == suite and task_id < len(values) else ''):>24}"
        print(row)
    row = f"{'mean':<24}"
    for c in columns:
        rates = [100 * ok / n for ok, n in c["tasks"].values() if n]
        row += f"{(f'{statistics.mean(rates):.1f}% ({len(rates)} tasks)' if rates else '-'):>24}"
    for _, _, values in references:
        row += f"{statistics.mean(values):>23.1f}%"
    print(row + "\n")
    for c in columns:
        m, r = c["meta"], c["result"]
        print(f"{c['name']}: memory {m.get('memory', '?')}, library {m.get('library_start', '?')} "
              f"({m.get('library_start_words', '?')} -> {m.get('library_end_words', '?')} words), "
              f"harness {m.get('harness', '?')}, cost ${r['total_cost_usd']:.2f}, "
              f"{r['duration_ms'] / 3.6e6:.1f} h of agent time in {r['parts']} part(s)")  # fmt: skip
    print("\n50 held-out seeds give each task a 95% interval of about +/-14 points at 50% success;")
    print("differences of a few points between single campaigns are noise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
