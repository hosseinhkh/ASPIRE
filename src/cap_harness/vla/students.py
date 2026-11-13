"""The VLA students this repository evaluates and fine-tunes, and where each one runs.

    smolvla  SmolVLA (0.45B), ``lerobot/smolvla_libero``, in ``.venv-libero``
    pi05     pi0.5 (about 3.6B), ``lerobot/pi05_libero_finetuned_v044``, in ``.venv-pi05``

Both take the same inputs (LIBERO's two 256x256 cameras, the 8-value state, the instruction)
and give the same 7-value hand-movement action, so the simulator, the protocol, the converted
demonstrations and the grading are shared. They differ in:

* **environment**: LeRobot v0.4.4's pi0.5 needs ``transformers`` from the branch
  ``fix/lerobot_openpi`` (4.53.3), which conflicts with SmolVLA's (4.57). Each student runs in
  its own virtual environment; ``reexec`` moves a command into the right one.
* **camera names**: the SmolVLA checkpoint renamed LIBERO's cameras to camera1/camera2;
  pi0.5 keeps LIBERO's names.
* **normalisation statistics**: a training dataset carries the statistics of the student it
  is for (``trace2vla.dataset.freeze_stats``).
* **training defaults**: learning rates, LoRA targets, extra trainer arguments.

Standard library only: the dashboard and the status commands import it without LeRobot. A
record written before pi0.5 support has no ``student`` field; ``infer`` reads it as SmolVLA,
the only student until then.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import sys
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
#: Set while a command runs in the environment ``reexec`` moved it to (guards against loops).
REEXEC_ENV = "CAP_HARNESS_STUDENT_REEXEC"
#: Set to 1 to run a student in the current interpreter without moving to its environment
#: (development and tests only: the wrong environment fails in confusing ways).
STAY_ENV = "CAP_HARNESS_STUDENT_STAY"


@dataclass(frozen=True)
class Student:
    key: str
    label: str
    #: The ``type`` in a LeRobot checkpoint's config.json.
    policy_type: str
    checkpoint: str
    revision: str
    venv: str
    setup_command: str
    #: LIBERO camera name -> the name the checkpoint expects, when it renamed them.
    camera_rename: Mapping[str, str] = field(default_factory=dict)
    #: Default learning rate per training method; a method not listed is not supported.
    learning_rates: Mapping[str, float] = field(default_factory=dict)
    #: LoRA target modules for ``lora-vlm`` (adapters on the vision-language backbone too).
    lora_vlm_targets: str | None = None
    #: Published success on the standard LIBERO suites, % (the checkpoint's own reference).
    standard_reference: Mapping[str, float] = field(default_factory=dict)
    reference_source: str = ""
    #: Policy config values set when evaluating (for example no torch.compile).
    eval_config: Mapping[str, Any] = field(default_factory=dict)
    #: Extra lerobot-train arguments.
    train_args: Sequence[str] = ()
    #: Environment variables the student's commands start with, unless already set.
    environment: Mapping[str, str] = field(default_factory=dict)

    def python(self, repo_root: Path = REPOSITORY_ROOT) -> Path:
        return Path(repo_root) / self.venv / "bin" / "python"

    def describe(self) -> dict[str, Any]:
        """What a record stores about the student it used."""
        return {
            "key": self.key,
            "label": self.label,
            "policy_type": self.policy_type,
            "checkpoint": self.checkpoint,
            "revision": self.revision,
        }


SMOLVLA = Student(
    key="smolvla",
    label="SmolVLA",
    policy_type="smolvla",
    checkpoint="lerobot/smolvla_libero",
    # The revision LeRobot's documentation pins.
    revision="a24f1f380461b665158d52acd140152c79e8b36d",
    venv=".venv-libero",
    setup_command="scripts/gpu/vla.sh setup",
    camera_rename={
        "observation.images.image": "observation.images.camera1",
        "observation.images.image2": "observation.images.camera2",
    },
    learning_rates={"full": 5e-5, "lora": 1e-4, "lora-vlm": 1e-4},
    # Attention projections of the action expert, the language model and the vision encoder,
    # plus the projections LeRobot's default LoRA already covers.
    lora_vlm_targets=(
        r"(model\.vlm_with_expert\..*\.(q|v)_proj"
        r"|model\.(state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out))"
    ),
    # SmolVLA (0.45B) on standard LIBERO, % (SmolVLA paper, arXiv 2506.01844, Table 2).
    standard_reference={
        "libero_spatial": 90.0,
        "libero_object": 96.0,
        "libero_goal": 92.0,
        "libero_10": 71.0,
    },
    reference_source="SmolVLA paper",
)

PI05 = Student(
    key="pi05",
    label="pi0.5",
    policy_type="pi05",
    # LeRobot's own pi0.5 LIBERO checkpoint for LeRobot v0.4.4 (fine-tuned from
    # lerobot/pi05_libero_base on HuggingFaceVLA/libero, 6000 steps, batch 32, lr 2.5e-5).
    checkpoint="lerobot/pi05_libero_finetuned_v044",
    revision="7152f13e34c6c115d808aea0e5d08aca74eac14b",
    venv=".venv-pi05",
    setup_command="scripts/gpu/vla.sh setup-pi05",
    camera_rename={},
    # 2.5e-5 is the checkpoint's own fine-tuning rate. LoRA: a first guess, as for SmolVLA.
    # lora-vlm is not offered: its module names are not verified for pi0.5 yet.
    learning_rates={"full": 2.5e-5, "lora": 1e-4},
    # LeRobot's pi0.5 on standard LIBERO with this implementation, % (LeRobot docs, libero.mdx).
    standard_reference={
        "libero_spatial": 97.0,
        "libero_object": 99.0,
        "libero_goal": 98.0,
        "libero_10": 96.0,
    },
    reference_source="LeRobot docs",
    # The checkpoint asks for torch.compile: minutes of compiling in every evaluation process,
    # and a needless risk on a new GPU architecture.
    eval_config={"compile_model": False},
    train_args=("--policy.compile_model=false",),
    # The PaliGemma tokenizer is used before the simulator and the trainer's data workers fork;
    # with its threads on, every fork printed a warning. One instruction per prediction needs
    # no threads.
    environment={"TOKENIZERS_PARALLELISM": "false"},
)

STUDENTS: dict[str, Student] = {student.key: student for student in (SMOLVLA, PI05)}
DEFAULT_STUDENT = SMOLVLA.key
#: What ``infer`` answers for a record that names no student (written before pi0.5 support).
LEGACY = "legacy record: SmolVLA was the only student before pi0.5 support"


def get(key: str | None) -> Student:
    """The student called ``key`` (the default student for None)."""
    if key is None or key == "":
        return STUDENTS[DEFAULT_STUDENT]
    try:
        return STUDENTS[str(key)]
    except KeyError:
        raise ValueError(f"unknown student {key!r}: choose one of {', '.join(STUDENTS)}") from None


def for_policy_type(policy_type: object) -> Student | None:
    return next((s for s in STUDENTS.values() if s.policy_type == policy_type), None)


def checkpoint_type(path: str | Path) -> str | None:
    """The policy ``type`` in a local LeRobot checkpoint's config.json, if there is one."""
    try:
        config = json.loads((Path(path) / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return config.get("type") if isinstance(config, dict) else None


def for_source(source: object) -> Student | None:
    """The student a checkpoint belongs to: a registered Hub id, a local checkpoint's config,
    or the name of the folder or Hub repository. None when nothing says.
    """
    if not isinstance(source, str) or not source:
        return None
    for student in STUDENTS.values():
        if source == student.checkpoint:
            return student
    found = for_policy_type(checkpoint_type(source)) if Path(source).is_dir() else None
    if found:
        return found
    name = source.lower().replace("models--", "").replace("--", "/")
    if "pi05" in name or "pi0.5" in name or "pi0_5" in name:
        return PI05
    if "smolvla" in name:
        return SMOLVLA
    return None


def infer(record: Mapping[str, Any] | None) -> tuple[str, str]:
    """(student key, how it was found) for a run, evaluation, training or dataset record.

    Order: an explicit ``student`` field; the policy type a benchmark recorded; the checkpoint
    it names; otherwise SmolVLA, the only student before this field existed.
    """
    record = record or {}
    settings = record.get("settings") if isinstance(record.get("settings"), Mapping) else {}
    for value in (record.get("student"), settings.get("student")):
        key = value.get("key") if isinstance(value, Mapping) else value
        if isinstance(key, str) and key in STUDENTS:
            return key, "recorded"
    policy = record.get("policy")
    if isinstance(policy, Mapping):
        found = for_policy_type(policy.get("policy_type"))
        if found:
            return found.key, "policy type"
        found = for_source(policy.get("source"))
        if found:
            return found.key, "checkpoint"
    for value in (
        policy if isinstance(policy, str) else None,
        settings.get("policy"),
        record.get("base_checkpoint"),
        record.get("checkpoint"),
        record.get("checkpoint_label"),
    ):
        found = for_source(value) if isinstance(value, str) else None
        if found:
            return found.key, "checkpoint"
    return DEFAULT_STUDENT, LEGACY


def label(key: str | None) -> str:
    """A student's display name; an unknown key is shown as it is."""
    if key is None:
        return get(None).label
    return STUDENTS[key].label if key in STUDENTS else str(key)


# ------------------------------------------------------------------------------ runtime --
def in_runtime(student: Student, repo_root: Path = REPOSITORY_ROOT) -> bool:
    """True when this interpreter is the student's environment."""
    venv = Path(repo_root) / student.venv
    try:
        return Path(sys.prefix).resolve() == venv.resolve()
    except OSError:
        return False


def reexec(
    student: Student,
    module: str,
    argv: Sequence[str],
    *,
    repo_root: Path = REPOSITORY_ROOT,
    execve: Any = os.execve,
) -> None:
    """Continue this command as ``<student venv>/bin/python -m module argv`` when this is not
    the student's environment. Returns only when no move is needed. The process keeps its pid,
    so ``t2v.sh stop`` and ``vla.sh stop`` still find it.
    """
    for name, value in student.environment.items():
        os.environ.setdefault(name, value)
    if in_runtime(student, repo_root) or os.environ.get(STAY_ENV) == "1":
        return
    if os.environ.get(REEXEC_ENV) == student.key:
        raise RuntimeError(
            f"{student.label} should run in {Path(repo_root) / student.venv}, but this "
            f"interpreter is {sys.prefix} after switching once; check that environment"
        )
    python = student.python(repo_root)
    if not python.exists():
        raise RuntimeError(
            f"{student.label} runs in its own environment, {student.venv}, which is missing: "
            f"run {student.setup_command}"
        )
    print(f"{student.label}: continuing in {student.venv}", file=sys.stderr, flush=True)
    sys.stdout.flush()
    environment = {**os.environ, REEXEC_ENV: student.key}
    execve(str(python), [str(python), "-m", module, *argv], environment)
