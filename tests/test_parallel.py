"""Parallel grading, conversion and dataset building: the worker pool, safe shared writes, and
the commands that use them. The simulator runs are checked separately (docs).
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path

import pytest

from cap_harness.trace2vla import cli, dataset, parallel
from cap_harness.trace2vla.store import Experiment, read_json, write_json


def square(value: int) -> dict:
    if value == 3:
        raise ValueError("three is bad")
    return {"value": value * value, "pid": __import__("os").getpid()}


def write_many(path: str, worker: int) -> int:
    padding = "x" * 200_000  # a larger file keeps each write open longer: races show up
    for index in range(300):
        write_json(Path(path), {"worker": worker, "index": index, "padding": padding})
    return worker


@pytest.mark.parametrize("workers", [1, 3])
def test_every_job_is_reported_once_and_a_failure_does_not_stop_the_rest(workers) -> None:
    results, errors, cleaned = {}, {}, []
    parallel.run_jobs(
        square,
        [(v,) for v in range(6)],
        workers=workers,
        on_result=lambda job, r: results.__setitem__(job[0], r["value"]),
        on_error=lambda job, e: errors.__setitem__(job[0], str(e)),
        cleanup=lambda: cleaned.append(True),
    )
    assert results == {0: 0, 1: 1, 2: 4, 4: 16, 5: 25}
    assert errors == {3: "three is bad"}
    # In this process the jobs' resources are cleaned up here; workers clean up on exit.
    assert cleaned == ([True] if workers == 1 else [])


def test_jobs_run_in_other_processes_with_several_workers() -> None:
    pids = set()
    parallel.run_jobs(
        square,
        [(v,) for v in (0, 1, 2, 4)],
        workers=2,
        on_result=lambda _job, r: pids.add(r["pid"]),
        on_error=lambda job, e: pytest.fail(f"{job}: {e}"),
    )
    assert pids and __import__("os").getpid() not in pids


def test_processes_writing_one_file_at_once_never_collide(tmp_path) -> None:
    target = tmp_path / "summary.json"
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(4, mp_context=context) as pool:
        done = list(pool.map(write_many, [str(target)] * 4, range(4)))
    assert sorted(done) == [0, 1, 2, 3]
    assert json.loads(target.read_text())["index"] == 299  # whole, valid JSON
    assert not list(tmp_path.glob(".*.tmp"))


def test_chunks_keep_the_order() -> None:
    assert parallel.chunks(range(7), 3) == [[0, 1, 2], [3, 4, 5], [6]]
    assert parallel.chunks([], 3) == []


# ------------------------------------------------------------------------------ commands --
@pytest.fixture
def experiment(tmp_path, monkeypatch) -> Experiment:
    monkeypatch.setattr(cli, "ensure_runtime_env", lambda: None)
    return Experiment.create(
        tmp_path / "trace2vla" / "exp",
        {"suite": "libero_object_swap", "tasks": [0, 1], "instructions": {"0": "a", "1": "b"},
         "scene_sets": {"alt1-train": {"init_mode": "seeded", "seeds": "1001-1003", "role": "train",
                                       "suite": "libero_object_swap_alt1"}}},
    )  # fmt: skip


def test_prepare_grades_then_converts_every_scene_set(experiment, monkeypatch) -> None:
    from cap_harness.trace2vla import convert, teacher

    calls = []
    monkeypatch.setattr(teacher, "results", lambda *_a, **_k: {"scene_set": "x", "successes": 0, "episodes": 0, "tasks": {}})  # fmt: skip
    monkeypatch.setattr(cli, "_teacher_table", lambda summary: "")
    monkeypatch.setattr(
        teacher, "grade_scene_set",
        lambda exp, scenes, tasks, workers: calls.append(("grade", scenes, workers))
        or {"graded": 0, "families": {}, "unreadable": [], "failed": []},
    )  # fmt: skip
    monkeypatch.setattr(
        convert, "convert_scene_set",
        lambda exp, scenes, name, tasks, settings, workers: calls.append(("convert", scenes, name, workers))
        or {"accepted": 1, "rejected": 0, "tasks": [], "failed": []},
    )  # fmt: skip
    root = str(experiment.root)
    assert cli.main(["prepare", root, "--scenes", "train,alt1-train", "--workers", "6"]) == 0
    assert calls == [
        ("grade", "train", 6), ("grade", "alt1-train", 6),
        ("convert", "train", "train-3x", 6), ("convert", "alt1-train", "alt1-train-3x", 6),
    ]  # fmt: skip


def test_an_unknown_scene_set_or_one_name_for_several_sets_is_refused(experiment, capsys) -> None:
    root = str(experiment.root)
    assert cli.main(["convert", root, "--scenes", "train,nope"]) == 1
    assert "unknown scene set 'nope'" in capsys.readouterr().err
    assert cli.main(["convert", root, "--scenes", "train,alt1-train", "--demos", "d"]) == 2


def _demo(root: Path, task: int, seed: int) -> None:
    folder = root / f"task_{task:02d}" / f"seed_{seed}"
    write_json(folder / "episode.json", {"accepted": True, "scene": {"task_id": task}})


def test_a_dataset_from_several_layouts_takes_per_task_from_each(experiment, monkeypatch) -> None:
    for demo_set in ("alt1-train-3x", "alt2-train-3x"):
        for task in (0, 1):
            for seed in (1001, 1002, 1003):
                _demo(experiment.demo_dir(demo_set), task, seed)
    seen = {}

    def fake_build(demos, out_dir, **kwargs):
        seen.update(demos=demos, workers=kwargs["workers"])
        manifest = {"name": kwargs["name"], "episodes": [{}] * len(demos), "frames": 10 * len(demos),
                    "per_task": {}, "rejected": [],
                    "statistics": {"frozen": {"action": {"checkpoint_std": [1.0], "data_std": [1.0]}}}}  # fmt: skip
        write_json(out_dir / "manifest.json", manifest)
        return manifest

    monkeypatch.setattr(dataset, "build_dataset_parallel", fake_build)
    monkeypatch.setattr(cli, "statistics_checkpoint", lambda student: Path("/ckpt"))
    root = str(experiment.root)
    command = ["dataset", root, "--demos", "alt1-train-3x,alt2-train-3x", "--name", "d-alt"]
    assert cli.main([*command, "--per-task", "2", "--workers", "4"]) == 0
    picked = [(d.parents[1].name, d.parent.name, d.name) for d in seen["demos"]]
    assert picked == [
        ("alt1-train-3x", "task_00", "seed_1001"), ("alt1-train-3x", "task_00", "seed_1002"),
        ("alt1-train-3x", "task_01", "seed_1001"), ("alt1-train-3x", "task_01", "seed_1002"),
        ("alt2-train-3x", "task_00", "seed_1001"), ("alt2-train-3x", "task_00", "seed_1002"),
        ("alt2-train-3x", "task_01", "seed_1001"), ("alt2-train-3x", "task_01", "seed_1002"),
    ]  # fmt: skip
    assert seen["workers"] == 4
    manifest = read_json(experiment.dataset_dir("d-alt") / "manifest.json")
    assert manifest["demo_sets"] == ["alt1-train-3x", "alt2-train-3x"]
    # A missing demo set is named.
    assert cli.main(["dataset", root, "--demos", "alt1-train-3x,none", "--name", "d2"]) == 1


def test_one_worker_builds_the_dataset_as_before(monkeypatch, tmp_path) -> None:
    calls = []
    monkeypatch.setattr(
        dataset, "build_dataset", lambda demos, out, **kw: calls.append(kw) or {"ok": 1}
    )
    assert dataset.build_dataset_parallel(
        [tmp_path / "a", tmp_path / "b"], tmp_path / "out", name="d", checkpoint=None, workers=1
    ) == {"ok": 1}
    assert calls and "workers" not in calls[0]
