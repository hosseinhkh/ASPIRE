"""Write the campaign task list for every task of a suite.

Usage (with the benchmark's interpreter):
  .venv-libero/bin/python scripts/gpu/checks/suite_tasks.py libero-pro libero_goal_swap OUT.json
  .venv-robosuite/bin/python scripts/gpu/checks/suite_tasks.py robosuite cube_lifting OUT.json

The format is the one iterative-debugging/scripts/init_run.py reads.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys


def main() -> int:
    benchmark, suite, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    if benchmark == "libero-pro":
        from cap_harness.libero.registry import LiberoSuiteRegistry

        tasks = [
            {"benchmark": benchmark, "suite": suite, "task_id": m.task_id, "task_name": m.task_name}
            for m in LiberoSuiteRegistry().enumerate_tasks([suite])
        ]
    elif benchmark == "robosuite":
        from cap_harness.robosuite.registry import RobosuiteTaskRegistry

        meta = RobosuiteTaskRegistry().resolve(suite, 0)
        tasks = [
            {"benchmark": benchmark, "suite": suite, "task_id": 0, "task_name": meta.task_name}
        ]
    else:
        raise SystemExit(f"unsupported benchmark {benchmark!r} (libero-pro or robosuite)")
    out.write_text(json.dumps(tasks, indent=2) + "\n", encoding="utf-8")
    print(f"{len(tasks)} tasks of {benchmark} {suite} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
