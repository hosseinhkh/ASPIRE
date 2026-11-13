"""Guards for the full data collection: one process per scene set, task and demo set at a time;
no scene in both a training set and a test or dev set; one scene set per demo set; enough disk.
"""

from __future__ import annotations

import multiprocessing
from pathlib import Path
from typing import NamedTuple

import pytest

from cap_harness.trace2vla import cli, convert, scenes, teacher
from cap_harness.trace2vla.store import Busy, Experiment, read_json, write_json


class Usage(NamedTuple):
    total: int
    used: int
    free: int


@pytest.fixture
def experiment(tmp_path, monkeypatch) -> Experiment:
    monkeypatch.setattr(cli, "ensure_runtime_env", lambda: None)
    return Experiment.create(
        tmp_path / "trace2vla" / "exp",
        {"suite": "libero_object_swap", "tasks": [0, 1], "instructions": {"0": "a", "1": "b"}},
    )


# ---------------------------------------------------------------------------- the claims --
def test_a_claimed_key_is_refused_with_its_holder_and_free_again_after(experiment) -> None:
    with experiment.hold(["teacher-train-task00"], "teacher-run on 'train'"):
        busy = r"teacher-train-task00 is in use by teacher-run on 'train' \(pid"
        with pytest.raises(Busy, match=busy), experiment.hold(["teacher-train-task00"], "x"):
            pass
        with experiment.hold(["teacher-train-task01"], "other task"):  # other keys are free
            pass
    with experiment.hold(["teacher-train-task00"], "again"):  # released
        pass


def _hold_until_killed(root: str, ready) -> None:
    with Experiment(Path(root)).hold(["demos-d-task00"], "a worker"):
        ready.set()
        multiprocessing.Event().wait(60)


def test_a_claim_ends_when_its_process_is_killed(experiment) -> None:
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    holder = context.Process(target=_hold_until_killed, args=(str(experiment.root), ready))
    holder.start()
    try:
        assert ready.wait(30)
        with pytest.raises(Busy, match="a worker"), experiment.hold(["demos-d-task00"], "me"):
            pass
    finally:
        holder.kill()  # no cleanup code runs: the kernel ends the claim
        holder.join(10)
    with experiment.hold(["demos-d-task00"], "me"):
        pass


# --------------------------------------------------------------------------- teacher-run --
def _fake_batch(tmp_path: Path) -> tuple[Path, Path]:
    argv_file = tmp_path / "argv.txt"
    batch = tmp_path / "fake-cap-harness"
    batch.write_text(f'#!/bin/sh\necho "$@" >> "{argv_file}"\n')
    batch.chmod(0o755)
    return batch, argv_file


def test_teacher_run_checks_every_program_before_running_any(experiment, tmp_path) -> None:
    program = tmp_path / "program.py"
    program.write_text("result = 1\n")
    teacher.register_program(experiment, 0, program)  # task 1 has none
    batch, argv_file = _fake_batch(tmp_path)
    train = scenes.scene_set(experiment.settings, "train")
    with pytest.raises(FileNotFoundError, match=r"task\(s\) 1 "):
        teacher.run_scene_set(experiment, train, [0, 1], cap_harness=batch, log=lambda _l: None)
    assert not argv_file.exists()  # task 0 did not start either


def test_a_second_teacher_run_on_the_same_task_is_refused(experiment, tmp_path) -> None:
    program = tmp_path / "program.py"
    program.write_text("result = 1\n")
    teacher.register_program(experiment, 0, program)
    batch, argv_file = _fake_batch(tmp_path)
    train = scenes.scene_set(experiment.settings, "train")
    with experiment.hold(["teacher-train-task00"], "teacher-run on 'train'"), pytest.raises(Busy):
        teacher.run_scene_set(experiment, train, [0], cap_harness=batch, log=lambda _l: None)
    assert not argv_file.exists()
    # Another scene set of the same task runs meanwhile; after the first ends, this one runs.
    teacher.run_scene_set(experiment, train, [0], cap_harness=batch, log=lambda _l: None)
    assert "--seeds" in argv_file.read_text()


def test_teacher_run_takes_several_scene_sets_and_checks_the_disk(
    experiment, monkeypatch, capsys
) -> None:
    ran = []
    monkeypatch.setattr(teacher, "service_ports_down", lambda ports: [])
    monkeypatch.setattr(
        teacher,
        "run_scene_set",
        lambda exp, s, tasks, **kw: ran.append((s.name, tasks, kw["workers"])),
    )
    monkeypatch.setattr(teacher, "results", lambda *_a, **_k: {})
    monkeypatch.setattr(cli, "_teacher_table", lambda summary: "")
    root = str(experiment.root)
    command = ["teacher-run", root, "--scenes", "train,dev", "--workers", "3"]

    # 2 tasks x (50 + 10) seeds = 120 episodes: about 1.2 GB plus 10 GB kept free.
    monkeypatch.setattr(teacher.shutil, "disk_usage", lambda _p: Usage(0, 0, 5 * 2**30))
    assert cli.main(command) == 1
    error = capsys.readouterr().err
    assert "120 teacher episodes" in error and "--no-videos" in error and not ran
    assert cli.main([*command, "--force"]) == 0
    assert ran == [("train", [0, 1], 3), ("dev", [0, 1], 3)]

    ran.clear()
    monkeypatch.setattr(teacher.shutil, "disk_usage", lambda _p: Usage(0, 0, 500 * 2**30))
    assert cli.main(command) == 0
    assert [name for name, *_ in ran] == ["train", "dev"]
    assert cli.main(["teacher-run", root, "--scenes", "train,nope"]) == 1  # unknown set named


# ----------------------------------------------------------------------- grade, convert --
def test_grading_or_converting_what_another_process_holds_is_refused(experiment) -> None:
    held = experiment.hold(["grade-train-task00"], "teacher-grade of 'train'")
    with held, pytest.raises(Busy, match="teacher-grade of 'train'"):
        teacher.grade_scene_set(experiment, "train", [0], log=lambda _l: None)
    held = experiment.hold(["demos-train-3x-task01"], "convert of 'train' into 'train-3x'")
    with held, pytest.raises(Busy):
        convert.convert_scene_set(experiment, "train", "train-3x", [0, 1], log=lambda _l: None)
    # Free again: both run (nothing to do on an empty scene set).
    assert teacher.grade_scene_set(experiment, "train", [0], log=lambda _l: None)["graded"] == 0
    assert (
        convert.convert_scene_set(experiment, "train", "train-3x", [0, 1], log=lambda _l: None)[
            "accepted"
        ]
        == 0
    )


def test_a_demo_set_holds_one_scene_set(experiment) -> None:
    root = experiment.demo_dir("d")
    convert.convert_scene_set(experiment, "train", "d", [0], log=lambda _l: None)
    assert read_json(root / "source.json") == {"scene_set": "train"}
    with pytest.raises(ValueError, match="holds the episodes of scene set 'train'"):
        convert.convert_scene_set(experiment, "dev", "d", [0], log=lambda _l: None)
    convert.convert_scene_set(experiment, "train", "d", [0], log=lambda _l: None)  # same set: fine


def test_a_demo_set_made_before_the_check_is_known_by_its_episodes(experiment) -> None:
    old = experiment.demo_dir("d50")
    write_json(
        old / "task_01" / "seed_1007.rejected.json", {"scene_set": "train", "accepted": False}
    )
    with pytest.raises(ValueError, match="'train'"):
        convert.claim_demo_set(old, "alt1-train")
    convert.claim_demo_set(old, "train")
    assert read_json(old / "source.json") == {"scene_set": "train"}


# ------------------------------------------------------------------------- scene sets --
def test_no_scene_is_both_for_training_and_for_testing(experiment, monkeypatch, capsys) -> None:
    from cap_harness.vla import protocol

    monkeypatch.setattr(protocol, "task_count", lambda suite: 10)
    root = str(experiment.root)
    add = ["scenes", root, "--add"]
    # dev holds seeds 2001-2010 of the experiment's suite (seeded): a train set there leaks.
    assert cli.main([*add, "leak", "--seeds", "2005-2020"]) == 2
    assert "share seeds 2005-2010 of libero_object_swap (seeded)" in capsys.readouterr().err
    # official holds the saved states 1-50.
    assert cli.main([*add, "saved-train", "--init-mode", "saved", "--seeds", "40-60"]) == 2
    # The same seeds of another layout, or in the other init mode, are other scenes.
    assert (
        cli.main(
            [
                *add,
                "alt1-dev",
                "--suite",
                "libero_object_swap_alt1",
                "--seeds",
                "2001-2010",
                "--role",
                "dev",
            ]
        )
        == 0
    )
    assert cli.main([*add, "seeded-low", "--seeds", "1-50"]) == 0
    # Two training sets with the same scenes: allowed, with a warning.
    capsys.readouterr()
    assert cli.main([*add, "more", "--seeds", "1041-1060"]) == 0
    assert "warning: 'train' has the same scenes (seeds 1041-1050" in capsys.readouterr().out
    names = scenes.scene_sets(Experiment.open(root).settings)
    assert {"alt1-dev", "seeded-low", "more"} <= set(names) and "leak" not in names
