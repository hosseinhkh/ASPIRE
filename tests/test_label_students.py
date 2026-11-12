"""Labelling old student records: dry run by default, backups, running runs left alone."""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket

from cap_harness.vla import label_students


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _outputs(tmp_path: Path) -> Path:
    out = tmp_path / "outputs"
    smol = {"settings": {"suite": "libero_object", "policy": "lerobot/smolvla_libero"},
            "protocol": {"policy": "lerobot/smolvla_libero"}, "policy": {"policy_type": "smolvla"}}  # fmt: skip
    _write(out / "vla" / "sanity" / "run.json", smol)
    _write(out / "vla" / "sanity" / "progress.json", {"status": "finished", "done": 500})
    # A run that is still going (this very process): left alone.
    _write(out / "vla" / "live" / "run.json", smol)
    _write(out / "vla" / "live" / "progress.json",
           {"status": "running", "pid": os.getpid(), "host": socket.gethostname()})  # fmt: skip
    exp = out / "trace2vla" / "preflight2"
    _write(exp / "evals" / "before-official" / "run.json", smol)
    _write(exp / "evals" / "before-official" / "eval.json",
           {"checkpoint_label": "base (lerobot/smolvla_libero)", "scene_set": "official"})  # fmt: skip
    _write(exp / "training" / "ft-full" / "run.json", {"name": "ft-full", "settings": {"method": "full"},
           "base_checkpoint": "/h/.cache/huggingface/hub/models--lerobot--smolvla_libero/snapshots/a"})  # fmt: skip
    _write(exp / "datasets" / "d50" / "manifest.json", {"name": "d50", "episodes": [], "frames": 0})
    # Already labelled by new code: unchanged.
    _write(exp / "datasets" / "d50-pi05" / "manifest.json",
           {"name": "d50-pi05", "student": {"key": "pi05", "label": "pi0.5"}})  # fmt: skip
    # Teacher runs and episodes are never touched.
    _write(exp / "teacher" / "official" / "task_00" / "0001" / "run-1" / "run.json", {"seed": 1})
    return out


def test_dry_run_writes_nothing_and_shows_every_record(tmp_path, capsys) -> None:
    out = _outputs(tmp_path)
    before = {p: p.read_bytes() for p in out.rglob("*.json")}
    assert label_students.main(["--outputs", str(out)]) == 0
    text = capsys.readouterr().out
    assert {p: p.read_bytes() for p in out.rglob("*.json")} == before
    assert "nothing written" in text and "teacher" not in text
    actions = {str(e["path"].relative_to(out)): e["action"] for e in label_students.plan(out)}
    assert actions["vla/sanity/run.json"] == "label"
    assert actions["vla/live/run.json"].startswith("skip")
    assert actions["trace2vla/preflight2/datasets/d50-pi05/manifest.json"] == "already labelled"
    assert actions["trace2vla/preflight2/training/ft-full/run.json"] == "label"


def test_apply_labels_old_records_as_smolvla_with_backups(tmp_path) -> None:
    out = _outputs(tmp_path)
    teacher = out / "trace2vla/preflight2/teacher/official/task_00/0001/run-1/run.json"
    teacher_bytes = teacher.read_bytes()
    live = (out / "vla/live/run.json").read_bytes()
    assert label_students.main(["--outputs", str(out), "--apply"]) == 0
    run = _read(out / "vla/sanity/run.json")
    assert run["student"]["key"] == "smolvla" and run["settings"]["student"] == "smolvla"
    assert run["protocol"] == {"policy": "lerobot/smolvla_libero"}  # the protocol is untouched
    assert _read(out / "vla/sanity/progress.json")["student"]["key"] == "smolvla"
    assert (
        _read(out / "trace2vla/preflight2/evals/before-official/eval.json")["student"] == "smolvla"
    )
    assert (
        _read(out / "trace2vla/preflight2/training/ft-full/run.json")["student"]["key"] == "smolvla"
    )
    assert (
        _read(out / "trace2vla/preflight2/datasets/d50/manifest.json")["student"]["key"]
        == "smolvla"
    )
    assert (
        _read(out / "trace2vla/preflight2/datasets/d50-pi05/manifest.json")["student"]["key"]
        == "pi05"
    )
    assert (
        teacher.read_bytes() == teacher_bytes and (out / "vla/live/run.json").read_bytes() == live
    )
    (backup,) = (out / ".student-labels").iterdir()
    assert _read(backup / "vla/sanity/run.json")["settings"] == {
        "suite": "libero_object",
        "policy": "lerobot/smolvla_libero",
    }
    # Idempotent: a second run has nothing to label.
    assert not [e for e in label_students.plan(out) if e["action"] == "label"]
    assert label_students.apply(label_students.plan(out), out) is None


def test_apply_keeps_file_times_so_runs_keep_their_order(tmp_path) -> None:
    """The dashboard and the report order evaluations, training and datasets by file time."""
    out = _outputs(tmp_path)
    labelled = [e["path"] for e in label_students.plan(out) if e["action"] == "label"]
    assert labelled
    for number, path in enumerate(labelled):
        os.utime(path, ns=(1_600_000_000_000_000_000, 1_600_000_000_000_000_000 + number))
    times = {p: p.stat().st_mtime_ns for p in labelled}
    label_students.apply(label_students.plan(out), out)
    assert {p: p.stat().st_mtime_ns for p in labelled} == times
    assert all("student" in _read(p) for p in labelled)
    assert not list(out.rglob(".*.labelling"))
