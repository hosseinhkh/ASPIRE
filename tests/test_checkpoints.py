"""Checkpoints on another disk: training writes there, existing ones move there, nothing breaks."""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import stat
import sys

import pytest

from cap_harness.trace2vla import checkpoints, cli, train
from cap_harness.trace2vla.store import Experiment, read_json, write_json
from cap_harness.vla import students

# A stand-in for `python -m lerobot.scripts.lerobot_train`, as strict as LeRobot about its output
# folder: it refuses one that exists, saves every --save_freq steps, then points "last" at it.
FAKE_LEROBOT = """\
import pathlib, sys
args = dict(a[2:].split("=", 1) for a in sys.argv[3:] if a.startswith("--") and "=" in a)
out = pathlib.Path(args["output_dir"])
if out.is_dir():
    sys.exit(f"Output directory {out} already exists")
steps, every = int(args["steps"]), int(args["save_freq"])
for step in range(every, steps + 1, every):
    folder = out / "checkpoints" / f"{step:06d}"
    (folder / "pretrained_model").mkdir(parents=True)
    (folder / "pretrained_model" / "config.json").write_text('{"type": "smolvla"}')
    (folder / "pretrained_model" / "model.safetensors").write_bytes(b"w" * 1000)
    (folder / "training_state").mkdir()
    (folder / "training_state" / "optimizer_state.safetensors").write_bytes(b"o" * 2000)
    last = out / "checkpoints" / "last"
    if last.is_symlink():
        last.unlink()
    last.symlink_to(folder.name)
    print(f"INFO step:{step} smpl:{step * 2} ep:1 epch:0.1 loss:0.5 grdn:1.0 lr:1e-4 "
          "updt_s:0.1 data_s:0.0", flush=True)
"""


@pytest.fixture
def experiment(tmp_path, monkeypatch) -> Experiment:
    monkeypatch.delenv(checkpoints.ENV, raising=False)
    monkeypatch.setattr(students, "reexec", lambda *_a, **_k: None)
    monkeypatch.setattr(cli, "ensure_runtime_env", lambda: None)
    return Experiment.create(
        tmp_path / "outputs" / "trace2vla" / "exp",
        {"suite": "libero_object_swap", "tasks": [0], "instructions": {"0": "a"},
         "scene_sets": {"official": {"init_mode": "saved", "seeds": "1-50", "role": "test"}}},
    )  # fmt: skip


def _fake_python(tmp_path: Path) -> str:
    script = tmp_path / "fake_lerobot.py"
    script.write_text(FAKE_LEROBOT)
    wrapper = tmp_path / "python"
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC)
    return str(wrapper)


def _dataset_and_base(tmp_path: Path) -> tuple[Path, Path]:
    data = tmp_path / "d50"
    write_json(data / "manifest.json", {"name": "d50", "episodes": [{}], "frames": 10})
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text(json.dumps({"type": "smolvla"}))
    return data, base


def _train(experiment: Experiment, tmp_path: Path, name: str) -> dict:
    data, base = _dataset_and_base(tmp_path)
    settings = train.TrainSettings(steps=1000, save_freq=500, log_freq=500)
    return train.train(
        experiment.training_dir(name),
        checkpoint=base,
        dataset_root=data,
        settings=settings,
        python=_fake_python(tmp_path),
    )


# ------------------------------------------------------------------------------ root --
def test_the_root_comes_from_the_environment_then_the_setting(tmp_path, monkeypatch) -> None:
    experiments = tmp_path / "trace2vla"
    monkeypatch.delenv(checkpoints.ENV, raising=False)
    assert checkpoints.checkpoint_root(experiments) is None
    assert checkpoints.set_root(experiments, tmp_path / "data") == tmp_path / "data"
    assert checkpoints.checkpoint_root(experiments) == tmp_path / "data"
    monkeypatch.setenv(checkpoints.ENV, str(tmp_path / "elsewhere"))
    assert checkpoints.checkpoint_root(experiments) == tmp_path / "elsewhere"
    monkeypatch.setenv(checkpoints.ENV, "")  # set but empty: none, whatever the setting says
    assert checkpoints.checkpoint_root(experiments) is None
    monkeypatch.delenv(checkpoints.ENV)
    assert checkpoints.set_root(experiments, None) is None
    assert checkpoints.checkpoint_root(experiments) is None
    with pytest.raises(ValueError, match="absolute"):
        checkpoints.set_root(experiments, Path("relative/dir"))


# -------------------------------------------------------------------------- training --
def test_without_a_root_training_is_unchanged(experiment, tmp_path) -> None:
    record = _train(experiment, tmp_path, "ft")
    run = experiment.training_dir("ft")
    assert (run / "lerobot").is_dir() and not (run / "lerobot").is_symlink()
    assert record["checkpoint_dir"] == str(run / "lerobot")
    assert record["status"] == "finished"
    assert train.saved_steps(run) == [500, 1000]


def test_with_a_root_training_writes_there_and_links_it(experiment, tmp_path) -> None:
    data = tmp_path / "data"
    checkpoints.set_root(experiment.root.parent, data)
    record = _train(experiment, tmp_path, "ft")
    run = experiment.training_dir("ft")
    target = data / "exp" / "ft"
    assert (run / "lerobot").is_symlink() and (run / "lerobot").resolve() == target.resolve()
    assert record["checkpoint_dir"] == str(target)
    assert f"--output_dir={target}" in record["command"]
    assert record["status"] == "finished"
    # Everything finds the checkpoints through the link.
    assert train.saved_steps(run) == [500, 1000]
    assert train.checkpoint_of(run) == (target / "checkpoints/001000/pretrained_model").resolve()
    assert checkpoints.location(run)["kind"] == "linked"
    # A second run with the same name in another experiment folder would collide: refused.
    with pytest.raises(FileExistsError, match="choose another run name"):
        checkpoints.prepare_output(tmp_path / "other" / "exp" / "training" / "ft", data)


# ------------------------------------------------------------------------------ move --
def test_moving_keeps_every_recorded_path_working(experiment, tmp_path) -> None:
    record = _train(experiment, tmp_path, "ft")
    run = experiment.training_dir("ft")
    recorded = Path(record["checkpoint"])  # inside the run folder, as written before the move
    before = sorted(str(p.relative_to(run / "lerobot")) for p in (run / "lerobot").rglob("*"))
    size = checkpoints.folder_size(run / "lerobot")
    assert size > 6000  # two checkpoints: weights, optimizer state, config
    data = tmp_path / "data"
    assert checkpoints.move(run, data, dry_run=True)["action"] == "would move"
    assert not (run / "lerobot").is_symlink()
    result = checkpoints.move(run, data, log=lambda _m: None)
    assert result["action"] == "moved" and result["bytes"] == size
    link = run / "lerobot"
    assert link.is_symlink() and link.resolve() == (data / "exp" / "ft").resolve()
    after = sorted(str(p.relative_to(link)) for p in link.rglob("*"))
    assert after == before
    assert os.readlink(link / "checkpoints" / "last") == "001000"  # the relative link survived
    assert (recorded / "config.json").is_file()  # the old path still works, through the link
    assert train.saved_steps(run) == [500, 1000]
    assert not (run / "lerobot.moving").exists()
    assert read_json(run / "checkpoints-moved.json")["bytes"] == size
    # Again: nothing to do.
    assert checkpoints.move(run, data)["action"] == "skipped"


def test_a_run_in_use_is_not_moved(experiment, tmp_path) -> None:
    _train(experiment, tmp_path, "ft")
    run = experiment.training_dir("ft")
    alive = {"status": "running", "pid": os.getpid(), "host": socket.gethostname()}
    write_json(run / "progress.json", alive)
    result = checkpoints.move(run, tmp_path / "data")
    assert result["action"] == "skipped" and "training is running" in result["detail"]
    write_json(run / "progress.json", {"status": "finished"})
    probe = experiment.eval_dir("ft@500-dev")
    write_json(probe / "eval.json", {"training_run": "ft", "checkpoint_step": 500})
    write_json(probe / "progress.json", alive)
    result = checkpoints.move(run, tmp_path / "data")
    assert result["action"] == "skipped" and "ft@500-dev" in result["detail"]
    assert (run / "lerobot").is_dir() and not (run / "lerobot").is_symlink()


def test_a_failed_copy_changes_nothing(experiment, tmp_path, monkeypatch) -> None:
    _train(experiment, tmp_path, "ft")
    run = experiment.training_dir("ft")
    real_copytree = checkpoints.shutil.copytree

    def lossy_copy(source, target, **kwargs):
        real_copytree(source, target, **kwargs)
        next(Path(target).rglob("model.safetensors")).write_bytes(b"short")

    monkeypatch.setattr(checkpoints.shutil, "copytree", lossy_copy)
    result = checkpoints.move(run, tmp_path / "data", log=lambda _m: None)
    assert result["action"] == "error" and "nothing changed" in result["detail"]
    assert (run / "lerobot").is_dir() and not (run / "lerobot").is_symlink()
    assert not (tmp_path / "data" / "exp" / "ft").exists()
    assert not list((tmp_path / "data" / "exp").glob("*.partial"))


def test_the_command_lists_sets_and_moves(experiment, tmp_path, capsys) -> None:
    _train(experiment, tmp_path, "ft")
    root = str(experiment.root)
    assert cli.main(["checkpoints", root]) == 0
    assert "ft" in capsys.readouterr().out
    assert cli.main(["checkpoints", root, "--move"]) == 2  # no root yet
    assert "--set-root" in capsys.readouterr().err
    data = tmp_path / "data"
    assert cli.main(["checkpoints", root, "--set-root", str(data)]) == 0
    assert checkpoints.checkpoint_root(experiment.root.parent) == data
    assert cli.main(["checkpoints", root, "--move"]) == 0
    assert "moved" in capsys.readouterr().out
    assert (experiment.training_dir("ft") / "lerobot").is_symlink()
    assert [e for e in experiment.events() if e["event"] == "checkpoints_moved"]
