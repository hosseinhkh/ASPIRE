"""Fine-tune the student on a dataset with LeRobot's own trainer, and keep a readable record.

    <training>/<run>/
        run.json        settings, the exact command, base checkpoint, dataset and its manifest hash
        train.log       everything lerobot-train printed
        metrics.jsonl   one line per logged step: step, loss, grad norm, lr, seconds per update
        progress.json   status, step, steps, latest loss, ETA (live)
        lerobot/        lerobot-train's output; checkpoints/last/pretrained_model is the result

Three methods:

    full       every weight the checkpoint itself trained (vision encoder included)
    lora       LeRobot's default LoRA: the action expert's attention and the state/action
               projections. The vision-language backbone is untouched.
    lora-vlm   LoRA on the attention of the backbone as well (SmolVLA only for now)

The student (``cap_harness.vla.students``) sets the defaults: learning rate per method, the
camera renaming, the LoRA targets and extra trainer arguments. The dataset must carry the base
checkpoint's normalisation statistics (``dataset.freeze_stats``) for the same student, because
lerobot-train copies the dataset's statistics into the fine-tuned checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any

from cap_harness.trace2vla.store import atomic_write, now_iso, read_json, sha256_file, write_json
from cap_harness.vla import students

METHODS = ("full", "lora", "lora-vlm")
#: SmolVLA's (the default student's) LoRA targets and learning rates, kept under their old names.
LORA_VLM_TARGETS = students.SMOLVLA.lora_vlm_targets
DEFAULT_LR = dict(students.SMOLVLA.learning_rates)
METRIC = re.compile(
    r"step:(\S+) smpl:(\S+) ep:(\S+) epch:(\S+) loss:(\S+) grdn:(\S+) lr:(\S+) updt_s:(\S+) data_s:(\S+)"
)
_UNITS = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}


@dataclass
class TrainSettings:
    method: str = "full"
    steps: int = 3000
    batch_size: int = 32
    lr: float | None = None
    lora_rank: int = 32
    seed: int = 1000
    save_freq: int | None = None
    log_freq: int = 50
    num_workers: int = 4
    device: str = "cuda"
    warmup_steps: int | None = None
    student: str = students.DEFAULT_STUDENT

    def resolved(self) -> dict[str, Any]:
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {', '.join(METHODS)}")
        student = students.get(self.student)
        if self.method not in student.learning_rates:
            raise ValueError(
                f"{student.label} does not support --method {self.method} yet: "
                f"use {' or '.join(student.learning_rates)}"
            )
        return {
            **self.__dict__,
            "lr": self.lr if self.lr is not None else student.learning_rates[self.method],
            "save_freq": self.save_freq or self.steps,
            "warmup_steps": self.warmup_steps
            if self.warmup_steps is not None
            else min(200, self.steps // 10),
        }


def big_number(text: str) -> float:
    """Lerobot prints 6K, 1.2M: back to a number."""
    return float(text[:-1]) * _UNITS[text[-1]] if text[-1] in _UNITS else float(text)


def lerobot_number(value: float) -> str:
    """A number the way LeRobot prints it in its log lines (format_big_number, no decimals):
    1000-1499 is 1K, 1500 already 2K (Python rounds half to even).
    """
    for suffix in ("", "K", "M", "B", "T"):
        if abs(value) < 1000:
            return f"{value:.0f}{suffix}"
        value /= 1000
    return f"{value:.0f}Q"


def exact_step(printed: str, previous: int, log_freq: int | None) -> int:
    """The real step of a log line. From step 1000 LeRobot prints it rounded to thousands, so
    the step is the previous one plus ``log_freq`` when that agrees with what is printed; the
    printed value otherwise (a skipped line, or a run that did not start at step 0).
    """
    value = big_number(printed)
    if printed[-1] not in _UNITS or not log_freq:
        return int(value)
    expected = previous + log_freq
    return expected if lerobot_number(expected) == printed else int(value)


def repair_steps(rows: list[dict[str, Any]], log_freq: int | None = None) -> list[dict[str, Any]]:
    """metrics.jsonl written before exact steps: 1000, 1000, ..., 2000 back to 1000, 1050, ...
    Rows already exact are unchanged.
    """
    if not rows:
        return rows
    log_freq = log_freq or int(rows[0].get("step") or 0)
    previous, repaired = 0, []
    for row in rows:
        step = exact_step(lerobot_number(float(row.get("step") or 0)), previous, log_freq)
        repaired.append({**row, "step": step})
        previous = step
    return repaired


def parse_metrics(
    line: str,
    previous_step: int = 0,
    log_freq: int | None = None,
    batch_size: int | None = None,
) -> dict[str, float] | None:
    match = METRIC.search(line)
    if not match:
        return None
    step, samples, episodes, epochs, loss, grad, lr, update_s, data_s = match.groups()
    exact = exact_step(step, previous_step, log_freq)
    return {
        "step": exact,
        "samples": exact * batch_size if batch_size else big_number(samples),
        "epochs": float(epochs),
        "loss": float(loss),
        "grad_norm": float(grad),
        "lr": float(lr),
        "update_s": float(update_s),
        "data_s": float(data_s),
    }


def build_command(
    *,
    python: str,
    checkpoint: Path,
    dataset_root: Path,
    repo_id: str,
    output_dir: Path,
    name: str,
    settings: dict[str, Any],
) -> list[str]:
    command = [
        python, "-m", "lerobot.scripts.lerobot_train",
        f"--policy.path={checkpoint}",
        f"--dataset.repo_id={repo_id}",
        f"--dataset.root={dataset_root}",
        f"--output_dir={output_dir}",
        f"--job_name={name}",
        f"--steps={settings['steps']}",
        f"--batch_size={settings['batch_size']}",
        f"--save_freq={settings['save_freq']}",
        f"--log_freq={settings['log_freq']}",
        f"--num_workers={settings['num_workers']}",
        f"--seed={settings['seed']}",
        "--eval_freq=0",
        "--wandb.enable=false",
        "--policy.push_to_hub=false",
        f"--policy.device={settings['device']}",
        f"--policy.optimizer_lr={settings['lr']}",
        f"--policy.scheduler_warmup_steps={settings['warmup_steps']}",
        f"--policy.scheduler_decay_steps={settings['steps']}",
    ]  # fmt: skip
    student = students.get(settings.get("student"))
    if student.camera_rename:
        # The checkpoint renamed its cameras (SmolVLA: camera1/camera2); the dataset uses
        # LIBERO's names.
        command.append(f"--rename_map={json.dumps(dict(student.camera_rename))}")
    if settings["method"] in ("lora", "lora-vlm"):
        command += ["--peft.method_type=LORA", f"--peft.r={settings['lora_rank']}"]
    if settings["method"] == "lora-vlm":
        command.append(f"--peft.target_modules={student.lora_vlm_targets}")
    return command + list(student.train_args)


def check_student(dataset_root: Path, checkpoint: Path, student: students.Student) -> None:
    """Refuse a dataset built for another student, or a checkpoint of another policy type."""
    manifest = read_json(dataset_root / "manifest.json") or {}
    built_for = students.infer(manifest)[0]
    if built_for != student.key:
        raise ValueError(
            f"dataset {dataset_root.name} carries {students.label(built_for)}'s normalisation "
            f"statistics, not {student.label}'s: build one for {student.label} "
            f"(t2v.sh dataset NAME --from-dataset {dataset_root.name} --student {student.key} "
            "--name NEW)"
        )
    found = students.for_policy_type(students.checkpoint_type(checkpoint))
    if found is not None and found.key != student.key:
        raise ValueError(f"{checkpoint} is a {found.label} checkpoint, not {student.label}")


def saved_steps(run_dir: Path) -> list[int]:
    """Steps whose checkpoint is complete. LeRobot writes the model, then the optimizer state,
    and only then points ``checkpoints/last`` at the new step, so a step is complete when
    ``last`` points at it or at a later step.
    """
    root = run_dir / "lerobot" / "checkpoints"
    try:
        last = int((root / "last").resolve().name)
    except (OSError, ValueError):
        return []
    steps = []
    for folder in root.glob("[0-9]*"):
        if folder.name.isdigit() and (folder / "pretrained_model").is_dir():
            if int(folder.name) <= last:
                steps.append(int(folder.name))
    return sorted(steps)


def checkpoint_at(run_dir: Path, step: int) -> Path | None:
    """The policy a run saved at ``step``, when that checkpoint is complete."""
    if step not in saved_steps(run_dir):
        return None
    root = run_dir / "lerobot" / "checkpoints"
    folder = next(f for f in root.glob("[0-9]*") if f.name.isdigit() and int(f.name) == step)
    return (folder / "pretrained_model").resolve()


def checkpoint_of(run_dir: Path) -> Path | None:
    """The trained policy of a run, if it finished a checkpoint."""
    last = run_dir / "lerobot" / "checkpoints" / "last" / "pretrained_model"
    if last.is_dir():
        return last.resolve()
    numbered = sorted((run_dir / "lerobot" / "checkpoints").glob("[0-9]*/pretrained_model"))
    return numbered[-1].resolve() if numbered else None


def train(
    run_dir: Path,
    *,
    checkpoint: Path,
    dataset_root: Path,
    settings: TrainSettings,
    python: str = sys.executable,
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    """Run lerobot-train to completion; return the run record (also written to run.json)."""
    if (run_dir / "run.json").exists():
        raise FileExistsError(f"{run_dir} already holds a training run (choose another name)")
    manifest = read_json(dataset_root / "manifest.json") or {}
    resolved = settings.resolved()
    student = students.get(resolved["student"])
    check_student(dataset_root, checkpoint, student)
    run_dir.mkdir(parents=True, exist_ok=True)
    command = build_command(
        python=python,
        checkpoint=checkpoint,
        dataset_root=dataset_root,
        repo_id=manifest.get("repo_id", f"trace2vla/{dataset_root.name}"),
        output_dir=run_dir / "lerobot",
        name=run_dir.name,
        settings=resolved,
    ) + list(extra_args or [])
    record: dict[str, Any] = {
        "name": run_dir.name,
        "started": now_iso(),
        "settings": resolved,
        "command": command,
        "student": student.describe(),
        "base_checkpoint": str(checkpoint),
        "dataset": str(dataset_root),
        "dataset_manifest_sha256": sha256_file(dataset_root / "manifest.json")
        if (dataset_root / "manifest.json").is_file()
        else None,
        "dataset_episodes": len(manifest.get("episodes", [])),
        "dataset_frames": manifest.get("frames"),
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    write_json(run_dir / "run.json", record)

    started = time.monotonic()
    latest: dict[str, float] = {}

    def progress(status: str, error: str | None = None) -> None:
        step = int(latest.get("step", 0))
        elapsed = time.monotonic() - started
        eta = (resolved["steps"] - step) * elapsed / step if step else None
        write_json(
            run_dir / "progress.json",
            {
                "status": status,
                "pid": os.getpid(),
                "updated": now_iso(),
                "step": step,
                "steps": resolved["steps"],
                "loss": latest.get("loss"),
                "lr": latest.get("lr"),
                "elapsed_s": round(elapsed, 1),
                "eta_s": round(eta, 0) if eta is not None else None,
                "error": error,
            },
        )

    progress("running")
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    signal.signal(signal.SIGTERM, lambda *_: process.terminate())
    status, error = "finished", None
    try:
        with (
            (run_dir / "train.log").open("a", encoding="utf-8") as log,
            (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as metrics,
        ):
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                parsed = parse_metrics(
                    line,
                    int(latest.get("step", 0)),
                    resolved["log_freq"],
                    resolved["batch_size"],
                )
                if parsed:
                    latest = parsed
                    metrics.write(json.dumps({"time": now_iso(), **parsed}) + "\n")
                    metrics.flush()
                    progress("running")
                    print(
                        f"step {parsed['step']}/{resolved['steps']} loss {parsed['loss']:.3f} lr {parsed['lr']:.1e} ({parsed['update_s']:.2f} s/update)",
                        flush=True,
                    )
                elif "Error" in line or "Traceback" in line:
                    print(line.rstrip(), flush=True)
        code = process.wait()
    except KeyboardInterrupt:
        process.terminate()
        code = process.wait()
        status = "interrupted"
    if status == "finished" and code != 0:
        status, error = "failed", f"lerobot-train exited with status {code}; see train.log"
    result = checkpoint_of(run_dir)
    if status == "finished" and result is None:
        status, error = "failed", "lerobot-train finished without a checkpoint"
    record.update(
        finished=now_iso(),
        status=status,
        error=error,
        checkpoint=str(result) if result else None,
        final_loss=latest.get("loss"),
        wall_s=round(time.monotonic() - started, 1),
    )
    write_json(run_dir / "run.json", record)
    progress(status, error)
    atomic_write(run_dir / "DONE" if status == "finished" else run_dir / "FAILED", now_iso() + "\n")
    return record
