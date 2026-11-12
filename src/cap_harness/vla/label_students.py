"""Label student records written before pi0.5 support with the student they used.

    python -m cap_harness.vla.label_students            what would change (nothing is written)
    python -m cap_harness.vla.label_students --apply    write the labels

Records that get a ``student`` field when they have none:

    outputs/vla/<run>/run.json, progress.json                benchmark runs
    outputs/trace2vla/<exp>/evals/<eval>/run.json, progress.json, eval.json
    outputs/trace2vla/<exp>/training/<run>/run.json          training runs
    outputs/trace2vla/<exp>/datasets/<name>/manifest.json    datasets

The student comes from ``students.infer``: the policy type or checkpoint a record names, else
SmolVLA (the only student before pi0.5 support). Nothing else in a record changes, a file keeps
its modification time (the dashboard and the report order runs by it), nothing is deleted,
episodes and teacher runs are never touched, and a run whose process is still running is skipped
(its own process rewrites those files). Every file is copied to
``outputs/.student-labels/<time>/`` before it is changed. Running it again changes nothing.

Labelling is optional: every reader already treats an unlabelled record this way.
Standard library only.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import socket
import sys
from typing import Any

from cap_harness.vla import students

OUTPUTS = students.REPOSITORY_ROOT / "outputs"


def _read(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _write(path: Path, value: dict[str, Any]) -> None:
    """Replace the file, keeping its access and modification times: the dashboard and the
    report order evaluations, training runs and datasets by them, and show them as "updated".
    """
    times = path.stat()
    temporary = path.with_name(f".{path.name}.labelling")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.utime(temporary, ns=(times.st_atime_ns, times.st_mtime_ns))
    os.replace(temporary, path)


def _running(folder: Path) -> bool:
    """True when the folder's progress.json says running and its process is alive here."""
    progress = _read(folder / "progress.json") or {}
    if progress.get("status") != "running":
        return False
    pid, host = progress.get("pid"), progress.get("host")
    if not pid or host != socket.gethostname():
        return True  # cannot tell: leave it alone
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _student_for(*records: dict[str, Any] | None) -> tuple[str, str]:
    """(key, how) from the first record that names a student; SmolVLA otherwise."""
    for record in records:
        key, how = students.infer(record or {})
        if how != students.LEGACY:
            return key, how
    return students.DEFAULT_STUDENT, students.LEGACY


def plan(outputs: Path = OUTPUTS) -> list[dict[str, Any]]:
    """One entry per record: its path, the student, and what to do."""
    entries: list[dict[str, Any]] = []

    def add(path: Path, kind: str, key: str, how: str, *, skip: str | None = None) -> None:
        record = _read(path)
        if record is None:
            return
        current = students.infer(record)
        if current[1] == "recorded":
            action = "already labelled" if current[0] == key else f"conflict: says {current[0]}"
        elif skip:
            action = f"skip: {skip}"
        else:
            action = "label"
        entries.append({"path": path, "kind": kind, "student": key, "how": how, "action": action})

    benchmark_runs = sorted((outputs / "vla").glob("*/run.json"))
    eval_runs = sorted((outputs / "trace2vla").glob("*/evals/*/run.json"))
    for path in benchmark_runs + eval_runs:
        folder = path.parent
        meta = _read(folder / "eval.json")
        key, how = _student_for(meta, _read(path))
        skip = "the run is still going" if _running(folder) else None
        kind = "evaluation" if meta is not None else "benchmark run"
        add(path, kind, key, how, skip=skip)
        add(folder / "progress.json", f"{kind} progress", key, how, skip=skip)
        if meta is not None:
            add(folder / "eval.json", "evaluation", key, how, skip=skip)
    for path in sorted((outputs / "trace2vla").glob("*/training/*/run.json")):
        key, how = _student_for(_read(path))
        skip = "training is still going" if _running(path.parent) else None
        add(path, "training run", key, how, skip=skip)
    for path in sorted((outputs / "trace2vla").glob("*/datasets/*/manifest.json")):
        key, how = _student_for(_read(path))
        add(path, "dataset", key, how)
    return entries


def label(record: dict[str, Any], path: Path, key: str) -> dict[str, Any]:
    """The record with its student, written the way current code writes it: eval.json holds
    the key, every other record the student's description; a run's settings get the key.
    """
    student = students.get(key)
    changed = dict(record)
    changed["student"] = student.key if path.name == "eval.json" else student.describe()
    if path.name == "run.json" and isinstance(changed.get("settings"), dict):
        changed["settings"] = {**changed["settings"], "student": student.key}
    return changed


def apply(entries: list[dict[str, Any]], outputs: Path = OUTPUTS) -> Path | None:
    """Write the labels; return the folder holding the originals (None when nothing changed)."""
    todo = [e for e in entries if e["action"] == "label"]
    if not todo:
        return None
    backup = outputs / ".student-labels" / datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    for entry in todo:
        path: Path = entry["path"]
        record = _read(path)
        if record is None:
            continue
        copy = backup / path.relative_to(outputs)
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, copy)
        _write(path, label(record, path, entry["student"]))
    return backup


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--outputs", type=Path, default=OUTPUTS)
    parser.add_argument(
        "--apply", action="store_true", help="write the labels (default: only show)"
    )
    args = parser.parse_args(argv)
    entries = plan(args.outputs)
    if not entries:
        print(f"no student records under {args.outputs}")
        return 0
    for entry in entries:
        print(
            f"{entry['action']:<20} {students.label(entry['student']):<8} {entry['kind']:<26} "
            f"{entry['path'].relative_to(args.outputs)}"
            + ("" if entry["how"] == "recorded" else f"  ({entry['how']})")
        )
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry["action"].split(":")[0]] = counts.get(entry["action"].split(":")[0], 0) + 1
    print("; ".join(f"{n} {action}" for action, n in sorted(counts.items())))
    if any(e["action"].startswith("conflict") for e in entries):
        print("conflicts are left as they are: check those records by hand", file=sys.stderr)
    if not args.apply:
        print("nothing written (pass --apply to write the labels)")
        return 0
    backup = apply(entries, args.outputs)
    print(f"written; originals in {backup}" if backup else "nothing to write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
