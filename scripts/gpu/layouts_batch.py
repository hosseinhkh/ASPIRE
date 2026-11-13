r"""Run one program on several layouts of a task, one batch per layout, and write one combined
batch.json: what a coding agent reads after each program version when it develops a program that
must work in every layout (LIBERO-Pro's swap suite and its sibling layouts, *_alt1..4).

    layouts_batch.py --layouts libero_object_swap,libero_object_swap_alt1 --task-id 0 \\
        --seeds 51 52 --program P --output-root DIR [--workers 2] [--init-mode seeded]

DIR/<layout>/ holds each layout's runs and its own batch.json (cap-harness run-batch). DIR/batch.json
sums them: ``passes`` and ``seeds`` over all layouts, each layout's own result under ``layouts``,
and the warnings and program errors with their layout named. The layouts run one after another.
Standard library only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def combine(
    batches: dict[str, dict[str, Any] | None], root: Path, seeds: list[str]
) -> dict[str, Any]:
    """One summary from each layout's batch.json (None: that layout's batch wrote none)."""
    combined: dict[str, Any] = {
        "layouts": {}, "passes": 0, "seeds": 0, "runs": 0, "elapsed_s": 0.0,
        "warnings": [], "program_errors": [], "missing": [], "results": [],
    }  # fmt: skip
    for layout, batch in batches.items():
        if not batch:
            combined["layouts"][layout] = {
                "error": f"its batch wrote no batch.json; see {root / layout}/batch-results.jsonl"
            }
            combined["missing"] += [f"{layout}:{seed}" for seed in seeds]
            continue
        combined["layouts"][layout] = {
            "passes": batch.get("passes", 0),
            "seeds": batch.get("seeds", 0),
            "pass_rate": batch.get("pass_rate"),
            "program_errors": batch.get("program_errors", []),
            "output": str(root / layout),
        }
        combined["passes"] += int(batch.get("passes") or 0)
        combined["seeds"] += int(batch.get("seeds") or 0)
        combined["runs"] += int(batch.get("runs") or 0)
        combined["elapsed_s"] += float(batch.get("elapsed_s") or 0.0)
        for warning in batch.get("warnings") or []:
            combined["warnings"].append(
                {**warning, "layout": layout, "text": f"[{layout}] {warning.get('text', '')}"}
            )
        combined["program_errors"] += [f"{layout}:{s}" for s in batch.get("program_errors") or []]
        combined["missing"] += [f"{layout}:{s}" for s in batch.get("missing") or []]
        combined["results"] += [{**r, "layout": layout} for r in batch.get("results") or []]
    combined["pass_rate"] = combined["passes"] / combined["seeds"] if combined["seeds"] else 0.0
    combined["elapsed_s"] = round(combined["elapsed_s"], 1)
    return combined


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--layouts", required=True, help="suites, comma separated")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--seeds", nargs="+", required=True)
    parser.add_argument("--program", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--workers", default="2")
    parser.add_argument("--init-mode", default="seeded")
    parser.add_argument("--benchmark", default="libero-pro")
    parser.add_argument(
        "--cap-harness", default=str(REPOSITORY_ROOT / ".venv-libero" / "bin" / "cap-harness")
    )
    args, extra = parser.parse_known_args(argv)
    layouts = [name.strip() for name in args.layouts.split(",") if name.strip()]
    seeds = [s for part in args.seeds for s in part.replace(",", " ").split()]
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    batches: dict[str, dict[str, Any] | None] = {}
    code = 0
    for layout in layouts:
        print(f"=== layout {layout} ({len(seeds)} seeds)", flush=True)
        command = [
            args.cap_harness, "run-batch", "--benchmark", args.benchmark, "--suite", layout,
            "--task-id", str(args.task_id), "--seeds", *seeds, "--init-mode", args.init_mode,
            "--program", args.program, "--output-root", str(root / layout),
            "--workers", str(args.workers), *extra,
        ]  # fmt: skip
        status = subprocess.call(command)
        code = code or status
        path = root / layout / "batch.json"
        try:
            batches[layout] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            batches[layout] = None
    combined = combine(batches, root, seeds)
    (root / "batch.json").write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
    print(f"=== all layouts: {combined['passes']}/{combined['seeds']} solved", flush=True)
    for layout, result in combined["layouts"].items():
        if "error" in result:
            print(f"  {layout}: {result['error']}")
        else:
            errors = len(result["program_errors"])
            print(
                f"  {layout}: {result['passes']}/{result['seeds']}"
                + (f" ({errors} program errors)" if errors else "")
            )
    print(f"summary: {root / 'batch.json'}")
    return code


if __name__ == "__main__":
    sys.exit(main())
