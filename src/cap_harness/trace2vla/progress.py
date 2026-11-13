"""Where an experiment's data collection stands, read from the files the stages write.

Per scene set and task: scenes planned, teacher episodes counted and solved, graded, demos kept
and rejected; what `t2v.sh collect` is doing now (collect.json); which background stages and
claims (locks/) are alive. Used by `t2v.sh status` and the dashboard, so standard library only.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any

from cap_harness.trace2vla.scenes import DEFAULT_SCENE_SETS, seeds_of
from cap_harness.trace2vla.store import read_json

#: Written by t2v.sh collect in the experiment folder.
STATUS_FILE = "collect.json"
_PID = re.compile(r"\(pid (\d+)")


def alive(pid: Any) -> bool:
    try:
        os.kill(int(pid), 0)
    except (TypeError, ValueError, ProcessLookupError):
        return False
    except PermissionError:
        return True
    return True


def demo_set_source(folder: Path) -> str | None:
    """The scene set a demo set was converted from (source.json, else its episodes say)."""
    owner = (read_json(folder / "source.json") or {}).get("scene_set")
    if owner:
        return str(owner)
    for pattern in ("task_*/seed_*/episode.json", "task_*/seed_*.rejected.json"):
        for path in folder.glob(pattern):
            found = (read_json(path) or {}).get("scene_set")
            if found:
                return str(found)
    return None


def collect_status(root: Path) -> dict[str, Any]:
    """collect.json, with ``status`` "died" when it says running but its process is gone."""
    status = read_json(root / STATUS_FILE) or {}
    if status.get("status") in ("running", "waiting") and not alive(status.get("pid")):
        status["status"] = "died"
        status["now"] = None
    return status


def background(root: Path) -> list[dict[str, Any]]:
    """Background stages started with --detach (logs/running.pid) that are still running."""
    found = []
    path = root / "logs" / "running.pid"
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if len(parts) >= 2 and alive(parts[0]):
                found.append({"pid": int(parts[0]), "stage": parts[1]})
    return found


def claims(root: Path) -> list[dict[str, Any]]:
    """Scene sets, demo sets and tasks claimed right now (store.Experiment.hold)."""
    found = []
    for path in sorted((root / "locks").glob("*.lock")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        pid = _PID.search(text)
        if text and pid and alive(pid.group(1)):
            found.append({"key": path.stem, "holder": text})
    return found


def overview(root: Path) -> dict[str, Any]:
    root = Path(root)
    settings = read_json(root / "experiment.json") or {}
    tasks = [int(t) for t in settings.get("tasks") or []]
    specs = {**DEFAULT_SCENE_SETS, **(settings.get("scene_sets") or {})}
    status = collect_status(root)
    cells_status = status.get("cells") or {}
    programs = read_json(root / "teacher" / "programs" / "programs.json") or {}
    demo_sets: dict[str, list[Path]] = {}
    for folder in sorted(p for p in (root / "demos").glob("*") if p.is_dir()):
        source = demo_set_source(folder)
        if source:
            demo_sets.setdefault(source, []).append(folder)
    collected = set(status.get("scene_sets") or [])
    sets = []
    for name, spec in specs.items():
        results = read_json(root / "teacher" / name / "results.json") or {}
        by_task = {int(t["task_id"]): t for t in results.get("tasks") or []}
        if not by_task and name not in collected and name not in demo_sets:
            continue
        planned = len(seeds_of(str(spec.get("seeds", ""))))
        cells = []
        for task in tasks:
            summary = by_task.get(task) or {}
            kept = rejected = 0
            for folder in demo_sets.get(name, []):
                kept += len(list(folder.glob(f"task_{task:02d}/seed_*/episode.json")))
                rejected += len(list(folder.glob(f"task_{task:02d}/seed_*.rejected.json")))
            cell = cells_status.get(f"{name}/{task}") or {}
            cells.append(
                {
                    "task": task,
                    "planned": planned,
                    "recorded": int(summary.get("episodes") or 0),
                    "solved": int(summary.get("successes") or 0),
                    "graded": int(summary.get("graded") or 0),
                    "kept": kept,
                    "rejected": rejected,
                    "state": cell.get("state"),
                    "prepare": cell.get("prepare"),
                    "note": cell.get("note"),
                    "prepare_log": cell.get("prepare_log"),
                }
            )
        sets.append(
            {
                "name": name,
                "role": spec.get("role", "train"),
                "suite": spec.get("suite") or settings.get("suite"),
                "init_mode": spec.get("init_mode"),
                "seeds": spec.get("seeds"),
                "demo_sets": [f.name for f in demo_sets.get(name, [])],
                "in_collection": name in collected,
                "cells": cells,
            }
        )
    # Totals: the scene sets that become demonstrations (test scenes are graded, never converted),
    # of this collection when there is one.
    counted = [
        c
        for s in sets
        if s["role"] != "test" and (s["in_collection"] or not collected)
        for c in s["cells"]
    ]
    totals = {
        "planned": sum(c["planned"] for c in counted),
        "recorded": sum(min(c["recorded"], c["planned"]) for c in counted),
        "solved": sum(c["solved"] for c in counted),
        "graded": sum(c["graded"] for c in counted),
        "kept": sum(c["kept"] for c in counted),
        "rejected": sum(c["rejected"] for c in counted),
    }
    program_states = status.get("programs") or {}
    return {
        "tasks": tasks,
        "collect": status,
        "programs": {
            str(t): {
                "registered": str(t) in programs,
                "sha256": (programs.get(str(t)) or {}).get("sha256"),
                "source": (programs.get(str(t)) or {}).get("source"),
                **(program_states.get(str(t)) or {}),
            }
            for t in tasks
        },
        "scene_sets": sets,
        "totals": totals,
        "background": background(root),
        "claims": claims(root),
    }
