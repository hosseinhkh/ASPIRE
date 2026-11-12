"""Trace2VLA with two students: each command uses the right student, environment and data."""

from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np
import pytest

from cap_harness.trace2vla import cli, dataset, train
from cap_harness.trace2vla.store import Experiment, read_json, write_json
from cap_harness.vla import students
from cap_harness.vla.students import PI05, SMOLVLA


class Moved(Exception):
    """Raised by the stand-in for os.execve: the command would continue elsewhere."""


@pytest.fixture
def experiment(tmp_path) -> Experiment:
    return Experiment.create(
        tmp_path / "trace2vla" / "exp",
        {
            "suite": "libero_object_swap",
            "tasks": [0, 1],
            "instructions": {"0": "pick the soup", "1": "pick the cheese"},
            "scene_sets": {"official": {"init_mode": "saved", "seeds": "1-50", "role": "test"}},
        },
    )


@pytest.fixture
def moves(monkeypatch) -> list[tuple[str, list[str]]]:
    """Record where reexec would send a command instead of replacing the test process."""
    moved: list[tuple[str, list[str]]] = []

    def fake(student, module, argv, **_kwargs):
        if student.key != students.DEFAULT_STUDENT:
            moved.append((student.key, list(argv)))
            raise Moved(student.key)

    monkeypatch.setattr(students, "reexec", fake)
    monkeypatch.setattr(cli, "ensure_runtime_env", lambda: None)
    return moved


def _dataset(experiment: Experiment, name: str, student: str | None) -> Path:
    root = experiment.dataset_dir(name)
    manifest = {"name": name, "episodes": [{}], "frames": 10, "per_task": {"0": 1}, "rejected": []}
    if student:
        manifest["student"] = students.get(student).describe()
    write_json(root / "manifest.json", manifest)
    return root


# ------------------------------------------------------------------------------ eval --
def test_eval_of_the_pi05_base_moves_to_its_environment_first(experiment, moves) -> None:
    argv = ["eval", str(experiment.root), "--name", "pi-before", "--student", "pi05"]
    with pytest.raises(Moved):
        cli.main(argv)
    assert moves == [("pi05", argv)]
    # Nothing was written before the move: the run happens entirely in .venv-pi05.
    assert not experiment.eval_dir("pi-before").exists()


def test_eval_labels_and_refuses_mixing_students_in_one_folder(
    experiment, moves, monkeypatch
) -> None:
    from cap_harness.vla import benchmark

    seen = []
    monkeypatch.setattr(benchmark, "main", lambda argv: seen.append(argv) or 0)
    assert cli.main(["eval", str(experiment.root), "--name", "before"]) == 0
    record = read_json(experiment.eval_dir("before") / "eval.json")
    assert record["student"] == "smolvla"
    assert record["checkpoint_label"] == "base (lerobot/smolvla_libero)"  # unchanged label
    assert seen[0][seen[0].index("--student") + 1] == "smolvla" and "--policy" not in seen[0]
    # The same folder for pi0.5 would mix two students' episodes in one results table.
    code = cli.main(["eval", str(experiment.root), "--name", "before", "--student", "pi05"])
    assert code == 2 and moves == []
    event = [e for e in experiment.events() if e["event"] == "student_evaluated"][-1]
    assert event["student"] == "smolvla"


def test_a_training_run_brings_its_own_student(experiment, moves) -> None:
    run = experiment.training_dir("ft-pi")
    checkpoint = run / "lerobot" / "checkpoints" / "last" / "pretrained_model"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text(json.dumps({"type": "pi05"}))
    write_json(run / "run.json", {"name": "ft-pi", "student": PI05.describe()})
    label, policy, student = cli._checkpoint(experiment, "ft-pi")
    assert (label, student) == ("training/ft-pi", PI05) and policy == str(checkpoint.resolve())
    with pytest.raises(ValueError, match=re.escape("is a pi0.5 checkpoint, not SmolVLA")):
        cli._checkpoint(experiment, "ft-pi", "smolvla")
    with pytest.raises(Moved):
        cli.main(["eval", str(experiment.root), "--name", "pi-after", "--checkpoint", "ft-pi"])
    assert moves[0][0] == "pi05"


# ----------------------------------------------------------------------------- train --
def test_train_takes_the_student_from_the_dataset(experiment, moves) -> None:
    _dataset(experiment, "d50-pi05", "pi05")
    argv = ["train", str(experiment.root), "--dataset", "d50-pi05", "--name", "ft-pi"]
    with pytest.raises(Moved):
        cli.main(argv)
    assert moves == [("pi05", argv)]


def test_train_refuses_a_dataset_built_for_another_student(experiment, moves, capsys) -> None:
    _dataset(experiment, "d50", None)  # a dataset from before pi0.5 support: SmolVLA's
    code = cli.main(
        ["train", str(experiment.root), "--dataset", "d50", "--name", "ft", "--student", "pi05"]
    )
    assert code == 2 and moves == []
    assert "--from-dataset d50 --student pi05" in capsys.readouterr().err
    assert not experiment.training_dir("ft").exists()


def test_pi05_training_command(tmp_path) -> None:
    settings = train.TrainSettings(method="full", student="pi05").resolved()
    assert settings["lr"] == 2.5e-5 and settings["student"] == "pi05"
    command = train.build_command(
        python="py", checkpoint=Path("/c"), dataset_root=Path("/d"), repo_id="r",
        output_dir=Path("/o"), name="n", settings=settings,
    )  # fmt: skip
    assert not any(arg.startswith("--rename_map") for arg in command)  # LIBERO names already
    assert command[-1] == "--policy.compile_model=false"
    lora = train.build_command(
        python="py", checkpoint=Path("/c"), dataset_root=Path("/d"), repo_id="r",
        output_dir=Path("/o"), name="n", settings=train.TrainSettings(method="lora", student="pi05").resolved(),
    )  # fmt: skip
    # LeRobot's own default LoRA targets for pi0.5: no --peft.target_modules.
    assert "--peft.method_type=LORA" in lora and not any("target_modules" in a for a in lora)
    with pytest.raises(ValueError, match=re.escape("pi0.5 does not support --method lora-vlm")):
        train.TrainSettings(method="lora-vlm", student="pi05").resolved()


def test_train_checks_dataset_and_checkpoint_before_starting(tmp_path) -> None:
    data = tmp_path / "d50"
    write_json(data / "manifest.json", {"name": "d50"})  # SmolVLA's (legacy)
    smol = tmp_path / "smol"
    smol.mkdir()
    (smol / "config.json").write_text(json.dumps({"type": "smolvla"}))
    train.check_student(data, smol, SMOLVLA)
    with pytest.raises(ValueError, match="carries SmolVLA's normalisation statistics"):
        train.check_student(data, smol, PI05)
    write_json(data / "manifest.json", {"name": "d50", "student": PI05.describe()})
    with pytest.raises(ValueError, match=re.escape("is a SmolVLA checkpoint, not pi0.5")):
        train.check_student(data, smol, PI05)
    with pytest.raises(ValueError, match=re.escape("carries pi0.5's")):
        train.train(tmp_path / "run", checkpoint=smol, dataset_root=data,
                    settings=train.TrainSettings(student="smolvla"))  # fmt: skip
    assert not (tmp_path / "run").exists()


# ---------------------------------------------------------------------------- dataset --
def _checkpoint_with_stats(root: Path, action_mean: float) -> Path:
    # The statistics are read with safetensors: in .venv-libero and .venv-pi05, not .venv-dev.
    save_file = pytest.importorskip("safetensors.numpy").save_file

    root.mkdir(parents=True)
    save_file(
        {
            "action.mean": np.full(7, action_mean, dtype=np.float32),
            "action.std": np.full(7, 2.0, dtype=np.float32),
            "action.q01": np.full(7, -1.0, dtype=np.float32),
            "observation.state.mean": np.zeros(8, dtype=np.float32),
            "observation.state.std": np.ones(8, dtype=np.float32),
        },
        str(root / "policy_preprocessor_step_2_normalizer_processor.safetensors"),
    )
    return root


def test_a_dataset_is_copied_for_pi05_with_its_statistics(experiment, tmp_path) -> None:
    source = _dataset(experiment, "d50", None)
    computed = {
        "action": {"mean": [0.5] * 7, "std": [0.1] * 7},
        "observation.state": {"mean": [0.0] * 8, "std": [1.0] * 8},
        "observation.images.image": {"mean": [[[0.4]]] * 3},
    }
    write_json(source / "meta" / "stats.json", computed)
    (source / "data").mkdir()
    (source / "data" / "frames.parquet").write_bytes(b"frames")
    smolvla_dataset = dataset.freeze_stats(source, _checkpoint_with_stats(tmp_path / "s", 0.25))
    assert smolvla_dataset["action"]["checkpoint_mean"] == [0.25] * 7

    manifest = dataset.derive_dataset(
        source,
        experiment.dataset_dir("d50-pi05"),
        name="d50-pi05",
        checkpoint=_checkpoint_with_stats(tmp_path / "p", 0.75),
        student="pi05",
    )
    copy = experiment.dataset_dir("d50-pi05")
    stats = read_json(copy / "meta" / "stats.json")
    assert stats["action"]["mean"] == [0.75] * 7 and stats["action"]["q01"] == [-1.0] * 7
    assert stats["observation.images.image"] == computed["observation.images.image"]
    # The data's own statistics, not the previous student's, are reported for comparison.
    assert manifest["statistics"]["frozen"]["action"]["data_mean"] == [0.5] * 7
    assert manifest["student"]["key"] == "pi05" and manifest["repo_id"] == "trace2vla/d50-pi05"
    assert (
        manifest["derived_from"]["name"] == "d50"
        and manifest["derived_from"]["student"] == "smolvla"
    )
    assert (copy / "data" / "frames.parquet").read_bytes() == b"frames"
    # The source dataset is untouched.
    assert read_json(source / "meta" / "stats.json")["action"]["mean"] == [0.25] * 7
    with pytest.raises(FileExistsError):
        dataset.derive_dataset(source, copy, name="x", checkpoint=tmp_path / "p", student="pi05")


# ---------------------------------------------------------------- report and dashboard --
def _evaluation(
    experiment: Experiment, name: str, eval_json: dict, run_json: dict, mean: float
) -> None:
    run = experiment.eval_dir(name)
    write_json(run / "eval.json", {"scene_set": "official", **eval_json})
    write_json(run / "run.json", {"settings": {"suite": "libero_object_swap"}, **run_json})
    write_json(run / "progress.json", {"status": "finished", "done": 100, "planned": 100})
    write_json(run / "results.json", {"tasks": [{"task_id": 0, "suite": "libero_object_swap",
               "successes": int(100 * mean), "episodes": 100, "success_rate": mean}],
               "mean": mean, "successes": int(100 * mean), "episodes": 100, "ci95": [0, 1]})  # fmt: skip


def test_report_and_dashboard_tell_the_students_apart(experiment) -> None:
    import sys

    from cap_harness.trace2vla import report

    # SmolVLA's evaluation from before pi0.5 support (no student field) and a pi0.5 one.
    _evaluation(experiment, "before-official", {"checkpoint_label": "base (lerobot/smolvla_libero)"},
                {"policy": {"policy_type": "smolvla"}}, 0.0)  # fmt: skip
    _evaluation(experiment, "pi-before", {"checkpoint_label": f"base ({PI05.checkpoint})", "student": "pi05"},
                {"student": PI05.describe()}, 0.17)  # fmt: skip
    _dataset(experiment, "d50", None)
    _dataset(experiment, "d50-pi05", "pi05")
    write_json(experiment.training_dir("ft-pi") / "run.json",
               {"name": "ft-pi", "student": PI05.describe(), "settings": {"method": "lora"}})  # fmt: skip

    built = report.build(experiment)
    assert [(e["name"], e["student"]) for e in built["evaluations"]] == [
        ("before-official", "smolvla"),
        ("pi-before", "pi05"),
    ]
    assert {d["name"]: d["student"] for d in built["datasets"]} == {
        "d50": "smolvla",
        "d50-pi05": "pi05",
    }
    assert built["training"][0]["student_label"] == "pi0.5"
    text = report.to_markdown(built)
    assert "| SmolVLA: before-official | pi0.5: pi-before |" in text
    assert "| ft-pi | pi0.5 | lora |" in text and "| d50-pi05 | pi0.5 |" in text

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "gpu"))
    from dashboard_store import Store
    from dashboard_t2v import T2vStore

    store = Store(experiment.root.parents[1])
    data = T2vStore(store).experiment(store.resolve("trace2vla/exp"))
    assert [(e["name"], e["student_label"]) for e in data["evals"]] == [
        ("before-official", "SmolVLA"),
        ("pi-before", "pi0.5"),
    ]
    assert {d["name"]: d["student_label"] for d in data["datasets"]} == {
        "d50": "SmolVLA",
        "d50-pi05": "pi0.5",
    }
    assert data["training"][0]["student_label"] == "pi0.5"
    listing = T2vStore(store).experiments()["items"][0]
    assert [e["student_label"] for e in listing["evals"]] == ["SmolVLA", "pi0.5"]


def test_vla_dashboard_and_status_show_the_student(tmp_path, capsys) -> None:
    import sys

    from cap_harness.vla import status

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "gpu"))
    from dashboard_store import Store
    from dashboard_vla import VlaStore

    outputs = tmp_path / "outputs"
    for name, record in (
        ("sanity-libero_object-n10", {"settings": {"policy": "lerobot/smolvla_libero"}}),
        (
            "sanity-pi05-libero_object-n10",
            {"settings": {"policy": PI05.checkpoint, "student": "pi05"}},
        ),
    ):
        write_json(outputs / "vla" / name / "run.json", record)
        write_json(
            outputs / "vla" / name / "progress.json",
            {"status": "finished", "done": 1, "planned": 1},
        )
    runs = {r["name"]: r["student_label"] for r in VlaStore(Store(outputs)).runs()["runs"]}
    assert runs == {"sanity-libero_object-n10": "SmolVLA", "sanity-pi05-libero_object-n10": "pi0.5"}
    status.list_runs(outputs / "vla")
    out = capsys.readouterr().out
    assert "student" in out.splitlines()[0]
    assert any("sanity-pi05" in line and "pi0.5" in line for line in out.splitlines())
