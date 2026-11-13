"""The student registry: which student a record belongs to, and where each one runs."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys

import pytest

from cap_harness.vla import students
from cap_harness.vla.students import PI05, SMOLVLA


def test_the_registry_keeps_smolvla_exactly_as_before() -> None:
    assert students.DEFAULT_STUDENT == "smolvla" and students.get(None) is SMOLVLA
    assert SMOLVLA.checkpoint == "lerobot/smolvla_libero"
    assert SMOLVLA.revision == "a24f1f380461b665158d52acd140152c79e8b36d"
    assert SMOLVLA.venv == ".venv-libero"
    assert dict(SMOLVLA.learning_rates) == {"full": 5e-5, "lora": 1e-4, "lora-vlm": 1e-4}
    assert dict(SMOLVLA.camera_rename) == {
        "observation.images.image": "observation.images.camera1",
        "observation.images.image2": "observation.images.camera2",
    }
    assert not SMOLVLA.eval_config and not SMOLVLA.train_args


def test_pi05_runs_in_its_own_environment_with_libero_camera_names() -> None:
    assert students.get("pi05") is PI05
    assert PI05.checkpoint == "lerobot/pi05_libero_finetuned_v044" and len(PI05.revision) == 40
    assert PI05.venv == ".venv-pi05" and PI05.venv != SMOLVLA.venv
    assert not PI05.camera_rename  # LIBERO's own image/image2
    assert "lora-vlm" not in PI05.learning_rates and PI05.learning_rates["full"] == 2.5e-5
    assert PI05.eval_config == {"compile_model": False}
    assert "--policy.compile_model=false" in PI05.train_args
    with pytest.raises(ValueError, match="unknown student 'pi0'"):
        students.get("pi0")


def test_a_checkpoint_is_matched_by_hub_id_config_or_name(tmp_path) -> None:
    assert students.for_source("lerobot/smolvla_libero") is SMOLVLA
    assert students.for_source("lerobot/pi05_libero_finetuned_v044") is PI05
    assert students.for_source("lerobot/pi05_libero_base") is PI05  # by name
    cache = "/home/u/.cache/huggingface/hub/models--lerobot--smolvla_libero/snapshots/a24f1f"
    assert students.for_source(cache) is SMOLVLA
    trained = tmp_path / "ft-full" / "pretrained_model"  # a name that says nothing
    trained.mkdir(parents=True)
    (trained / "config.json").write_text(json.dumps({"type": "pi05"}))
    assert students.checkpoint_type(trained) == "pi05" and students.for_source(str(trained)) is PI05
    assert students.for_source("some/other_policy") is None and students.for_source(None) is None


def test_infer_reads_new_and_old_records() -> None:
    assert students.infer({"student": "pi05"}) == ("pi05", "recorded")
    assert students.infer({"student": {"key": "pi05"}}) == ("pi05", "recorded")
    assert students.infer({"settings": {"student": "pi05"}}) == ("pi05", "recorded")
    # A benchmark run.json from before the field existed: the policy info names the type.
    assert students.infer({"policy": {"policy_type": "smolvla", "source": "x"}}) == (
        "smolvla",
        "policy type",
    )
    assert students.infer({"settings": {"policy": "lerobot/pi05_libero_finetuned_v044"}}) == (
        "pi05",
        "checkpoint",
    )
    assert students.infer({"checkpoint_label": "base (lerobot/smolvla_libero)"}) == (
        "smolvla",
        "checkpoint",
    )
    # A legacy training record or dataset manifest names nothing: SmolVLA, the only student then.
    assert students.infer({"name": "d50", "frames": 10}) == ("smolvla", students.LEGACY)
    assert students.infer(None) == ("smolvla", students.LEGACY)
    assert students.label("pi05") == "pi0.5" and students.label("other") == "other"


def _venv(root: Path, name: str) -> Path:
    python = root / name / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("")
    return python


def test_reexec_moves_a_command_into_the_students_environment(tmp_path, monkeypatch) -> None:
    calls = []
    monkeypatch.delenv(students.REEXEC_ENV, raising=False)
    monkeypatch.delenv(students.STAY_ENV, raising=False)
    monkeypatch.setattr(sys, "prefix", str(tmp_path / ".venv-libero"))
    python = _venv(tmp_path, ".venv-pi05")
    students.reexec(
        PI05,
        "cap_harness.trace2vla",
        ["eval", "exp", "--student", "pi05"],
        repo_root=tmp_path,
        execve=lambda *a: calls.append(a),
    )
    (path, argv, env) = calls[0]
    assert path == str(python)
    assert argv == [str(python), "-m", "cap_harness.trace2vla", "eval", "exp", "--student", "pi05"]
    assert env[students.REEXEC_ENV] == "pi05"


def test_reexec_stays_put_in_the_right_environment_and_refuses_loops(tmp_path, monkeypatch) -> None:
    calls = []
    monkeypatch.delenv(students.REEXEC_ENV, raising=False)
    monkeypatch.delenv(students.STAY_ENV, raising=False)
    _venv(tmp_path, ".venv-libero")
    monkeypatch.setattr(sys, "prefix", str(tmp_path / ".venv-libero"))
    students.reexec(SMOLVLA, "m", [], repo_root=tmp_path, execve=lambda *a: calls.append(a))
    assert calls == []  # already there
    # Missing environment: a clear instruction, not a confusing import error later.
    with pytest.raises(RuntimeError, match=re.escape("vla.sh setup-pi05")):
        students.reexec(PI05, "m", [], repo_root=tmp_path, execve=lambda *a: calls.append(a))
    _venv(tmp_path, ".venv-pi05")
    monkeypatch.setenv(students.REEXEC_ENV, "pi05")
    with pytest.raises(RuntimeError, match="after switching once"):
        students.reexec(PI05, "m", [], repo_root=tmp_path, execve=lambda *a: calls.append(a))
    monkeypatch.delenv(students.REEXEC_ENV)
    monkeypatch.setenv(students.STAY_ENV, "1")
    students.reexec(PI05, "m", [], repo_root=tmp_path, execve=lambda *a: calls.append(a))
    assert calls == []


def test_the_registry_needs_only_the_standard_library() -> None:
    source = Path(students.__file__).read_text()
    imports = {
        line.split()[1].split(".")[0]
        for line in source.splitlines()
        if line.startswith(("import ", "from "))
    }
    assert imports <= {
        "__future__",
        "collections",
        "dataclasses",
        "json",
        "os",
        "pathlib",
        "sys",
        "typing",
    }


# The answers of protocol.references before pi0.5 support, recorded from the old code.
OLD_REFERENCES = [
    ["libero_object", "lerobot/smolvla_libero", {"expected": 96.0, "notes": ["SmolVLA paper: 96%", "pi0.5 (LIBERO-PRO paper): 98%"]}],
    ["libero_object_swap", "lerobot/smolvla_libero", {"expected": None, "notes": ["pi0.5 (LIBERO-PRO paper): 17%", "no published SmolVLA number on LIBERO-Pro"]}],
    ["libero_object", "/x/training/ft-full/lerobot/checkpoints/last/pretrained_model", {"expected": None, "notes": ["SmolVLA paper: 96%", "pi0.5 (LIBERO-PRO paper): 98%"]}],
    ["libero_object_swap", "/x/training/ft-full/pretrained_model", {"expected": None, "notes": ["pi0.5 (LIBERO-PRO paper): 17%"]}],
    ["libero_goal_swap", "lerobot/smolvla_libero", {"expected": None, "notes": ["pi0.5 (LIBERO-PRO paper): 38%", "no published SmolVLA number on LIBERO-Pro"]}],
    ["libero_object", "/home/u/.cache/huggingface/hub/models--lerobot--smolvla_libero/snapshots/a24f", {"expected": 96.0, "notes": ["SmolVLA paper: 96%", "pi0.5 (LIBERO-PRO paper): 98%"]}],
    ["libero_90", "lerobot/smolvla_libero", {"expected": None, "notes": []}],
    ["other_suite", "p", {"expected": None, "notes": []}],
]  # fmt: skip


@pytest.mark.parametrize(("suite", "policy", "expected"), OLD_REFERENCES)
def test_smolvla_reference_numbers_are_unchanged(suite, policy, expected) -> None:
    from cap_harness.vla.protocol import references

    assert references(suite, policy) == expected
    assert references(suite, policy, "smolvla") == expected


def test_pi05_reference_numbers() -> None:
    from cap_harness.vla.protocol import references

    own = references("libero_object", "lerobot/pi05_libero_finetuned_v044")
    assert own == {
        "expected": 99.0,
        "notes": ["pi0.5 (LeRobot docs): 99%", "pi0.5 (LIBERO-PRO paper): 98%"],
    }
    swap = references("libero_object_swap", "lerobot/pi05_libero_finetuned_v044")
    assert swap["expected"] is None and swap["notes"] == [
        "pi0.5 (LIBERO-PRO paper): 17%",
        "no published number for lerobot/pi05_libero_finetuned_v044 on LIBERO-Pro",
    ]
    tuned = references("libero_object", "/x/training/ft-pi/pretrained_model", "pi05")
    assert tuned["expected"] is None and tuned["notes"][0] == "pi0.5 (LeRobot docs): 99%"


def test_the_benchmark_resolves_student_checkpoint_and_revision(tmp_path) -> None:
    from cap_harness.vla.benchmark import parse_args, resolve_policy

    assert parse_args(["--suite", "libero_object"]).student is None
    # No flags: SmolVLA's own checkpoint at its pinned revision, as before.
    assert resolve_policy(None, None, None) == (SMOLVLA, SMOLVLA.checkpoint, SMOLVLA.revision)
    assert resolve_policy(None, "lerobot/smolvla_libero", None)[2] == SMOLVLA.revision
    assert resolve_policy("pi05", None, None) == (PI05, PI05.checkpoint, PI05.revision)
    assert resolve_policy(None, PI05.checkpoint, None)[0] is PI05
    assert resolve_policy("pi05", None, "abc")[2] == "abc"
    # A fine-tuned checkpoint: its config names the policy type; no revision.
    tuned = tmp_path / "pretrained_model"
    tuned.mkdir()
    (tuned / "config.json").write_text(json.dumps({"type": "pi05"}))
    assert resolve_policy(None, str(tuned), None) == (PI05, str(tuned), None)
    with pytest.raises(ValueError, match=re.escape("is a pi0.5 checkpoint")):
        resolve_policy("smolvla", str(tuned), None)
    # An unknown checkpoint without --student stays with the default, as before.
    assert resolve_policy(None, "someone/policy", None) == (SMOLVLA, "someone/policy", None)


def test_policy_overrides_touch_only_what_the_config_has() -> None:
    from types import SimpleNamespace

    from cap_harness.vla.policy import apply_overrides, rename_map_for

    config = SimpleNamespace(compile_model=True, n_action_steps=50)
    assert apply_overrides(config, PI05.eval_config) == {"compile_model": False}
    assert config.compile_model is False
    assert apply_overrides(SimpleNamespace(), PI05.eval_config) == {}  # SmolVLA has no such field
    smolvla_inputs = {"observation.images.camera1": 1, "observation.images.camera2": 1}
    assert rename_map_for(smolvla_inputs) == dict(SMOLVLA.camera_rename)
    pi05_inputs = {"observation.images.image": 1, "observation.images.image2": 1}
    assert rename_map_for(pi05_inputs, PI05.camera_rename) == {}
    assert rename_map_for(pi05_inputs) == {}  # LIBERO names are never renamed away


def test_benchmark_report_mode_uses_the_runs_own_student(tmp_path, monkeypatch, capsys) -> None:
    from cap_harness.vla import benchmark

    root = tmp_path / "run"
    root.mkdir()
    (root / "run.json").write_text(
        json.dumps(
            {"settings": {"suite": "libero_object", "policy": PI05.checkpoint, "student": "pi05"}}
        )
    )
    monkeypatch.setattr(benchmark, "ensure_runtime_env", lambda: None)
    assert benchmark.main(["--suite", "libero_object", "--out", str(root), "--report"]) == 0
    assert "pi0.5 (LeRobot docs): 99%" in capsys.readouterr().out


def test_a_student_command_starts_with_the_student_environment(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("TOKENIZERS_PARALLELISM", raising=False)
    monkeypatch.setenv(students.STAY_ENV, "1")
    students.reexec(students.SMOLVLA, "m", [], repo_root=tmp_path)
    assert "TOKENIZERS_PARALLELISM" not in os.environ  # SmolVLA unchanged
    students.reexec(students.PI05, "m", [], repo_root=tmp_path)
    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "true")  # a value set by the user wins
    students.reexec(students.PI05, "m", [], repo_root=tmp_path)
    assert os.environ["TOKENIZERS_PARALLELISM"] == "true"
