"""Print the benchmark of a campaign task list and its size: "libero-pro 3", or "mixed 4".

Usage: task_list_kind.py TASKS_JSON   (the list format of iterative-debugging/scripts/init_run.py)
"""

from __future__ import annotations

import json
from pathlib import Path
import sys


def main() -> int:
    tasks = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    kinds = {str(task["benchmark"]) for task in tasks}
    print(kinds.pop() if len(kinds) == 1 else "mixed", len(tasks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
