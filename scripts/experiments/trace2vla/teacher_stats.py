"""Teacher episodes: length, outcome, idle share, hand speed, from recorded run folders.

teacher_stats.py DIR [DIR ...]      (any folders holding ASPIRE run directories)
"""

import json
import math
from pathlib import Path
import sys

SUITE_FILTER = "object_swap"


def load(run):
    out = json.loads((run / "outcome.json").read_text()) if (run / "outcome.json").exists() else {}
    meta = json.loads((run / "run.json").read_text()) if (run / "run.json").exists() else {}
    steps = []
    with (run / "episode/steps.jsonl").open() as f:
        for line in f:
            try:
                steps.append(json.loads(line))
            except ValueError:
                pass
    return meta, out, steps


def vec(v):
    if isinstance(v, dict):
        for k in ("$array_inline", "values", "data"):
            if k in v:
                return v[k]
    return v


rows = []
for p in sys.argv[1:]:
    for sj in sorted(Path(p).rglob("episode/steps.jsonl")):
        run = sj.parent.parent
        meta, out, steps = load(run)
        if SUITE_FILTER not in str(meta.get("suite", "")):
            continue
        pos = []
        for s in steps:
            rs = s.get("robot_state") or {}
            ee = (rs.get("end_effector_poses") or {}).get("primary") or {}
            pv = vec(ee.get("position"))
            if pv:
                pos.append([float(x) for x in pv])
        d = [math.dist(pos[i], pos[i - 1]) for i in range(1, len(pos))]
        idle = sum(1 for x in d if x < 2e-4)
        rows.append(
            (
                str(meta.get("suite")),
                meta.get("task_id"),
                meta.get("seed"),
                (meta.get("init_mode") or (meta.get("task_context") or {}).get("init_mode")),
                out.get("task_success"),
                out.get("termination_reason"),
                len(steps),
                idle,
                sum(d),
                max(d) if d else 0,
                str(run)[-60:],
            )
        )
print(f"{'suite':20} task seed mode    ok    steps idle  idle%  path_m  max_mm/step  termination")
for r in rows:
    print(
        f"{r[0]:20} {r[1]!s:>4} {r[2]!s:>4} {r[3]!s:7} {r[4]!s:5} {r[6]:5d} {r[7]:5d} {100 * r[7] / max(r[6], 1):5.0f}% {r[8]:6.2f} {1000 * r[9]:8.1f}     {r[5]}"
    )
