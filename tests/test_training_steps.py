"""Exact training steps, checkpoints saved during training, and evaluating them (eval-steps)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from cap_harness.trace2vla import cli, report, train
from cap_harness.trace2vla.store import Experiment, read_json, write_json
from cap_harness.vla import students


# ------------------------------------------------------------------------ exact steps --
def test_steps_are_exact_although_lerobot_prints_them_rounded() -> None:
    printed = [train.lerobot_number(s) for s in range(50, 3001, 50)]
    # What LeRobot prints: 1000-1450 as 1K, 1500-2450 as 2K, 2500 still 2K (half to even).
    assert printed[19:21] == ["1K", "1K"] and printed[29] == "2K" and printed[49:51] == ["2K", "3K"]
    previous, steps = 0, []
    for token in printed:
        previous = train.exact_step(token, previous, 50)
        steps.append(previous)
    assert steps == list(range(50, 3001, 50))


def test_metrics_written_with_rounded_steps_are_repaired_on_reading() -> None:
    printed = [train.lerobot_number(s) for s in range(50, 3001, 50)]
    old = [{"step": int(train.big_number(t)), "loss": 0.1} for t in printed]
    assert [r["step"] for r in old[19:22]] == [1000, 1000, 1000]
    repaired = train.repair_steps(old)
    assert [r["step"] for r in repaired] == list(range(50, 3001, 50))
    assert train.repair_steps(repaired) == repaired  # exact rows are left as they are
    assert train.repair_steps([]) == []


def test_parse_metrics_uses_the_previous_step() -> None:
    line = (
        "INFO step:2K smpl:48K ep:263 epch:3.29 loss:0.004 grdn:0.12 lr:2.2e-05 "
        "updt_s:0.500 data_s:0.001"
    )
    parsed = train.parse_metrics(line, 1450, 50, 32)
    assert parsed["step"] == 1500 and parsed["samples"] == 48000
    # Without the context (old callers) the printed value is used.
    assert train.parse_metrics(line)["step"] == 2000


# -------------------------------------------------------------- checkpoints by step --
def _checkpoints(run: Path, steps: list[int], last: int | None) -> None:
    root = run / "lerobot" / "checkpoints"
    for step in steps:
        (root / f"{step:06d}" / "pretrained_model").mkdir(parents=True)
        (root / f"{step:06d}" / "pretrained_model" / "config.json").write_text(
            '{"type": "smolvla"}'
        )
    if last is not None:
        (root / "last").symlink_to(f"{last:06d}")


def test_a_checkpoint_counts_only_once_last_points_at_it(tmp_path) -> None:
    run = tmp_path / "ft"
    # 1500 is being written: its model is there, LeRobot has not moved "last" yet.
    _checkpoints(run, [500, 1000, 1500], last=1000)
    assert train.saved_steps(run) == [500, 1000]
    assert (
        train.checkpoint_at(run, 500)
        == (run / "lerobot/checkpoints/000500/pretrained_model").resolve()
    )
    assert train.checkpoint_at(run, 1500) is None
    assert train.saved_steps(tmp_path / "nothing") == []
    (run / "lerobot/checkpoints/last").unlink()
    assert train.saved_steps(run) == []  # no "last" yet: nothing is complete


@pytest.fixture
def experiment(tmp_path, monkeypatch) -> Experiment:
    monkeypatch.setattr(cli, "ensure_runtime_env", lambda: None)
    monkeypatch.setattr(students, "reexec", lambda *_a, **_k: None)
    exp = Experiment.create(
        tmp_path / "trace2vla" / "exp",
        {
            "suite": "libero_object_swap",
            "tasks": [0, 1],
            "instructions": {"0": "a", "1": "b"},
            "scene_sets": {
                "official": {"init_mode": "saved", "seeds": "1-50", "role": "test"},
                "dev": {"init_mode": "seeded", "seeds": "2001-2010", "role": "dev"},
            },
        },
    )
    run = exp.training_dir("ft")
    start = tmp_path / "base-checkpoint"
    start.mkdir()
    (start / "config.json").write_text('{"type": "smolvla"}')
    record = {"name": "ft", "student": students.SMOLVLA.describe(), "base_checkpoint": str(start)}
    write_json(run / "run.json", record)
    _checkpoints(run, [500, 1000], last=1000)
    return exp


def test_eval_takes_a_step_and_records_it(experiment, monkeypatch) -> None:
    from cap_harness.vla import benchmark

    seen = []
    monkeypatch.setattr(benchmark, "main", lambda argv: seen.append(argv) or 0)
    root = str(experiment.root)
    assert cli.main(["eval", root, "--name", "ft500", "--checkpoint", "ft@500"]) == 0
    meta = read_json(experiment.eval_dir("ft500") / "eval.json")
    assert meta["checkpoint_label"] == "training/ft@500"
    assert meta["training_run"] == "ft" and meta["checkpoint_step"] == 500
    assert seen[0][seen[0].index("--policy") + 1].endswith("000500/pretrained_model")
    # The final checkpoint keeps its old label; the step is not set.
    assert cli.main(["eval", root, "--name", "ft-final", "--checkpoint", "ft"]) == 0
    meta = read_json(experiment.eval_dir("ft-final") / "eval.json")
    assert meta["checkpoint_label"] == "training/ft" and meta["checkpoint_step"] is None
    assert seen[1][seen[1].index("--policy") + 1].endswith("001000/pretrained_model")


def test_eval_of_a_missing_step_names_the_saved_ones(experiment, capsys) -> None:
    root = str(experiment.root)
    assert cli.main(["eval", root, "--name", "x", "--checkpoint", "ft@1500"]) == 1
    assert "saved: 500, 1000" in capsys.readouterr().err
    assert cli.main(["eval", root, "--name", "x", "--checkpoint", "ft@late"]) == 1
    assert "must be a number" in capsys.readouterr().err
    assert not experiment.eval_dir("x").exists()


# ----------------------------------------------------------------------- eval-steps --
def _fake_eval(experiment: Experiment, calls: list[list[str]], after=None):
    def call(command):
        calls.append(command)
        name = command[command.index("--name") + 1]
        out = experiment.eval_dir(name)
        write_json(out / "progress.json", {"status": "finished"})
        write_json(out / "results.json", {"successes": 3, "episodes": 20, "mean": 0.15})
        if after:
            after(command)
        return 0

    return call


def test_eval_steps_evaluates_each_saved_step_once(experiment, monkeypatch, capsys) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(cli.subprocess, "call", _fake_eval(experiment, calls))
    root = str(experiment.root)
    assert cli.main(["eval-steps", root, "--checkpoint", "ft"]) == 0
    names = [c[c.index("--name") + 1] for c in calls]
    assert names == ["ft@0-dev", "ft@500-dev", "ft@1000-dev"]
    assert [c[c.index("--checkpoint") + 1] for c in calls] == ["ft@0", "ft@500", "ft@1000"]
    assert all(c[c.index("--scenes") + 1] == "dev" and "--probe" in c for c in calls)
    assert "step 1000: 3/20 (15%)" in capsys.readouterr().out
    # A rerun skips finished evaluations; another suite gets its own names.
    assert cli.main(["eval-steps", root, "--checkpoint", "ft"]) == 0
    assert len(calls) == 3
    args = ["--suite", "libero_object", "--scenes", "official", "--scene-list", "1-10"]
    assert cli.main(["eval-steps", root, "--checkpoint", "ft", "--steps", "1000", *args]) == 0
    assert calls[-1][calls[-1].index("--name") + 1] == "ft@1000-libero_object-official"
    assert calls[-1][calls[-1].index("--scene-list") + 1] == "1-10"


def test_eval_steps_follows_a_running_training_to_its_last_checkpoint(
    experiment, monkeypatch
) -> None:
    run = experiment.training_dir("ft")
    write_json(run / "progress.json", {"status": "running", "pid": os.getpid()})

    def training_moves_on(command):
        # While step 1000 is evaluated, training saves 1500 and finishes.
        if command[command.index("--checkpoint") + 1] == "ft@1000":
            (run / "lerobot/checkpoints/001500/pretrained_model").mkdir(parents=True)
            (run / "lerobot/checkpoints/last").unlink()
            (run / "lerobot/checkpoints/last").symlink_to("001500")
            write_json(run / "progress.json", {"status": "finished", "pid": os.getpid()})

    calls: list[list[str]] = []
    monkeypatch.setattr(cli.subprocess, "call", _fake_eval(experiment, calls, training_moves_on))
    monkeypatch.setattr(cli.time, "sleep", lambda _s: None)
    root = str(experiment.root)
    command = ["eval-steps", root, "--checkpoint", "ft", "--follow", "--poll", "0", "--no-start"]
    assert cli.main(command) == 0
    assert [c[c.index("--checkpoint") + 1] for c in calls] == ["ft@500", "ft@1000", "ft@1500"]


def test_eval_steps_stops_when_the_filter_leaves_nothing_to_wait_for(
    experiment, monkeypatch
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(cli.subprocess, "call", _fake_eval(experiment, calls))
    root = str(experiment.root)
    # 500 is filtered out by --every 1000 and the training is not running: no endless wait.
    assert cli.main(["eval-steps", root, "--checkpoint", "ft", "--follow", "--every", "1000"]) == 0
    assert [c[c.index("--checkpoint") + 1] for c in calls] == ["ft@0", "ft@1000"]


def test_eval_steps_without_saved_checkpoints_says_how_to_get_them(
    experiment, monkeypatch, capsys
) -> None:
    write_json(experiment.training_dir("bare") / "run.json", {"name": "bare"})
    monkeypatch.setattr(cli.subprocess, "call", lambda _c: pytest.fail("nothing to evaluate"))
    command = ["eval-steps", str(experiment.root), "--checkpoint", "bare", "--no-start"]
    assert cli.main(command) == 1
    assert "--save-freq" in capsys.readouterr().err
    assert cli.main(["eval-steps", str(experiment.root), "--checkpoint", "nope"]) == 2


# --------------------------------------------------------------------------- report --
def test_report_keeps_the_teacher_comparison_to_full_official_evaluations() -> None:
    def evaluation(name, suite, scene_set, step=None, run=None, probe=False):
        return {"name": name, "suite": suite, "scene_set": scene_set, "checkpoint_step": step,
                "training_run": run, "probe": probe}  # fmt: skip

    swap = "libero_object_swap"
    evals = [
        evaluation("before-official", swap, "official"),
        evaluation("legacy", swap, None),
        evaluation("before-standard", "libero_object", "official"),
        evaluation("dev-check", swap, "dev"),
        evaluation("ft@1000-dev", swap, "dev", 1000, "ft", probe=True),
        evaluation("ft@0-dev", swap, "dev", 0, "ft", probe=True),
        evaluation("ft@500-libero_object-official", "libero_object", "official", 500, "ft", True),
        # A step chosen on the curve, then evaluated in full on the official scenes.
        evaluation("chosen-official", swap, "official", 1000, "ft"),
    ]
    main, other, curve = report.split_evaluations(evals, swap)
    assert [e["name"] for e in main] == ["before-official", "legacy", "chosen-official"]
    assert [e["name"] for e in other] == ["before-standard", "dev-check"]
    assert [e["name"] for e in curve] == [
        "ft@500-libero_object-official",
        "ft@0-dev",
        "ft@1000-dev",
        "chosen-official",
    ]


def test_step_zero_is_where_the_run_started(experiment, monkeypatch) -> None:
    from cap_harness.vla import benchmark

    seen = []
    monkeypatch.setattr(benchmark, "main", lambda argv: seen.append(argv) or 0)
    root = str(experiment.root)
    command = ["eval", root, "--name", "start", "--checkpoint", "ft@0", "--probe"]
    assert cli.main(command) == 0
    meta = read_json(experiment.eval_dir("start") / "eval.json")
    assert meta["checkpoint_label"] == "training/ft@0" and meta["checkpoint_step"] == 0
    assert meta["probe"] is True and meta["student"] == "smolvla"
    assert seen[0][seen[0].index("--policy") + 1].endswith("base-checkpoint")
    # A plain evaluation is not a probe.
    assert cli.main(["eval", root, "--name", "full", "--checkpoint", "ft@500"]) == 0
    assert read_json(experiment.eval_dir("full") / "eval.json")["probe"] is False


def test_probes_written_before_the_probe_field_are_recognised_by_name() -> None:
    from cap_harness.trace2vla.store import is_probe

    old = {"training_run": "ft-curve", "checkpoint_step": 500}
    assert is_probe("ft-curve@500-dev", old)
    assert not is_probe("curve500-official", old)  # named by hand: a full evaluation
    assert not is_probe("before-official", {"checkpoint_step": None})
    assert is_probe("anything", {"probe": True}) and not is_probe("ft@500-dev", {"probe": False})
