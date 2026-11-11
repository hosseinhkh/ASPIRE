"""The teacher: one frozen ASPIRE program per task, run on named scene sets and graded.

* **Programs.** A task's program is whatever ASPIRE produced for it: ``fix_code.py`` from a
  campaign, ``program_best.py`` from a session's coding agent, or any program file. It is copied
  into the experiment (``teacher/programs/task_<id>.py``) with its hash, and never edited.
* **Running.** ``cap-harness run-batch`` runs the program on every seed of a scene set with
  ASPIRE's full recorder: every control step, both camera videos, every tool call with its
  input and output, SAM3 masks and overlays, grasp candidates, point clouds and keyframes.
  The run folders are ASPIRE's own format, so the dashboard's episode replay opens them.
* **Grading.** Each episode's recorded joint commands are sent again in the simulator
  (``replay.joint_replay``). That checks the recording reproduces, labels what was lifted and
  placed, and stores the simulator state at the start and end. These privileged results go to
  ``grades/``, outside the run folders, so nothing the teacher agent reads contains them.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time
from typing import Any

import numpy as np

from cap_harness.trace2vla.episodes import find_runs, load_teacher_episode
from cap_harness.trace2vla.scenes import SceneSet, compact
from cap_harness.trace2vla.store import Experiment, read_json, sha256_file, write_json
from cap_harness.vla.libero_compat import REPOSITORY_ROOT

TEACHER_MAX_STEPS = 1000  # the harness's own episode limit for LIBERO


# ----------------------------------------------------------------------------- programs --
def discover_programs(source: Path, suite: str) -> dict[int, Path]:
    """Task programs under a campaign folder, a session folder, or a folder of programs."""
    source = Path(source)
    found: dict[int, Path] = {}
    # Campaign: <root>/<benchmark>/<suite>/task_<id>/fix_code.py
    for path in sorted(source.rglob("fix_code.py")):
        if suite in path.parts and path.parent.name.startswith("task_"):
            found[int(path.parent.name.split("_")[1])] = path
    # Session coding agent: agents/llm-<benchmark>-<suite>-<id>/program_best.py
    for path in sorted(source.rglob("program_best.py")):
        name = path.parent.name
        if name.startswith("llm-") and f"-{suite}-" in name:
            task = name.rsplit("-", 1)[1]
            if task.isdigit():
                found.setdefault(int(task), path)
    # A folder of task_<id>.py files (for example another experiment's teacher/programs).
    for path in sorted(source.glob("task_*.py")):
        task = path.stem.split("_")[1]
        if task.isdigit():
            found.setdefault(int(task), path)
    return found


def register_program(
    experiment: Experiment, task_id: int, source: Path, *, force: bool = False
) -> dict[str, Any]:
    target = experiment.program(task_id)
    digest = sha256_file(source)
    if target.is_file() and sha256_file(target) != digest and not force:
        raise FileExistsError(
            f"task {task_id} already has a different program ({target}); pass --force to replace it. "
            "Episodes run with the old program stay labelled with its hash."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    record = {
        "task_id": task_id,
        "source": str(Path(source).resolve()),
        "sha256": digest,
        "program": experiment.rel(target),
    }
    programs = read_json(target.parent / "programs.json", {}) or {}
    programs[str(task_id)] = record
    write_json(target.parent / "programs.json", programs)
    experiment.log("teacher_program_registered", **record)
    return record


def programs(experiment: Experiment) -> dict[int, dict[str, Any]]:
    stored = read_json(experiment.root / "teacher" / "programs" / "programs.json", {}) or {}
    return {
        int(task): record
        for task, record in stored.items()
        if experiment.program(int(task)).is_file()
    }


# ------------------------------------------------------------------------------ running --
def episodes_of(experiment: Experiment, scene_set: str, task_id: int) -> dict[int, Path]:
    """Finalized run folder per seed (the newest when a seed was run more than once)."""
    by_seed: dict[int, Path] = {}
    root = experiment.teacher_dir(scene_set, task_id)
    if not root.is_dir():
        return by_seed
    for run in find_runs(root):
        meta = read_json(run / "run.json") or {}
        if meta.get("status") == "finalized" and (run / "outcome.json").is_file():
            by_seed[int(meta["seed"])] = run
    return by_seed


def service_ports_down(ports: Iterable[int] = (8114, 8115, 8116)) -> list[int]:
    import socket

    down = []
    for port in ports:
        with socket.socket() as probe:
            probe.settimeout(0.5)
            if probe.connect_ex(("127.0.0.1", port)) != 0:
                down.append(port)
    return down


def run_scene_set(
    experiment: Experiment,
    scenes: SceneSet,
    tasks: Iterable[int],
    *,
    workers: int = 2,
    max_steps: int = TEACHER_MAX_STEPS,
    cap_harness: Path | None = None,
    camera: tuple[int, int] = (800, 512),
    videos: bool = True,
    log: Any = print,
) -> dict[str, Any]:
    """Run each task's program on the scene set's seeds that have no finished episode yet."""
    cap_harness = cap_harness or REPOSITORY_ROOT / ".venv-libero" / "bin" / "cap-harness"
    registered = programs(experiment)
    summary: dict[str, Any] = {"scene_set": scenes.name, "tasks": {}}
    for task_id in tasks:
        if task_id not in registered:
            raise FileNotFoundError(
                f"task {task_id} has no registered program (t2v.sh teacher-add)"
            )
        done = episodes_of(experiment, scenes.name, task_id)
        missing = [seed for seed in scenes.seeds if seed not in done]
        out = experiment.teacher_dir(scenes.name, task_id)
        if not missing:
            log(f"task {task_id}: all {len(scenes.seeds)} scenes of '{scenes.name}' already run")
            summary["tasks"][str(task_id)] = {"ran": 0, "skipped": len(done)}
            continue
        out.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        command = [
            str(cap_harness), "run-batch",
            "--benchmark", experiment.benchmark,
            "--suite", experiment.suite,
            "--task-id", str(task_id),
            "--seeds", *compact(missing).split(","),
            "--program", str(experiment.program(task_id)),
            "--output-root", str(out),
            "--flat-run-dir",
            "--init-mode", scenes.init_mode,
            "--max-steps", str(max_steps),
            "--workers", str(workers),
            "--camera-width", str(camera[0]),
            "--camera-height", str(camera[1]),
            "--summary", str(out / f"batch-{stamp}.json"),
            "--results-jsonl", str(out / f"batch-{stamp}.jsonl"),
            "--log-dir", str(out / "logs"),
        ]  # fmt: skip
        if not videos:
            command.append("--no-videos")
        environment = {
            **os.environ,
            "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl"),
            "PYOPENGL_PLATFORM": os.environ.get("PYOPENGL_PLATFORM", "egl"),
            "LIBERO_CONFIG_PATH": os.environ.get(
                "LIBERO_CONFIG_PATH", str(REPOSITORY_ROOT / ".libero")
            ),
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
        }
        log(
            f"task {task_id}: running {len(missing)} scenes of '{scenes.name}' ({scenes.init_mode}) with {workers} workers"
        )
        started = time.monotonic()
        with (out / f"batch-{stamp}.log").open("w", encoding="utf-8") as stream:
            code = subprocess.run(
                command,
                stdout=stream,
                stderr=subprocess.STDOUT,
                env=environment,
                cwd=REPOSITORY_ROOT,
                check=False,
            ).returncode
        after = episodes_of(experiment, scenes.name, task_id)
        ran = sorted(set(after) - set(done))
        result = {
            "ran": len(ran),
            "skipped": len(done),
            "not_finished": sorted(set(missing) - set(after)),
            "exit_status": code,
            "wall_s": round(time.monotonic() - started, 1),
            "program_sha256": registered[task_id]["sha256"],
        }
        summary["tasks"][str(task_id)] = result
        experiment.log(
            "teacher_run",
            scene_set=scenes.name,
            task_id=task_id,
            init_mode=scenes.init_mode,
            seeds=compact(missing),
            **result,
        )
        log(
            f"task {task_id}: {len(ran)} episodes recorded in {result['wall_s']:.0f} s"
            + (
                f"; {len(result['not_finished'])} did not finish (see {out}/logs)"
                if result["not_finished"]
                else ""
            )
        )
    results(experiment, scenes.name, write=True)
    return summary


# ------------------------------------------------------------------------------ results --
def results(experiment: Experiment, scene_set: str, *, write: bool = False) -> dict[str, Any]:
    """Success per task on a scene set, with step counts and how the failures ended."""
    tasks = []
    for task_id in experiment.tasks:
        runs = episodes_of(experiment, scene_set, task_id)
        if not runs:
            continue
        outcomes = {seed: read_json(run / "outcome.json") or {} for seed, run in runs.items()}
        grades = {
            seed: read_json(experiment.grade_dir(scene_set, task_id) / f"seed_{seed}.json")
            for seed in runs
        }
        successes = [seed for seed, o in outcomes.items() if o.get("task_success")]
        steps = [int(outcomes[seed].get("steps_executed") or 0) for seed in successes]
        tasks.append(
            {
                "task_id": task_id,
                "episodes": len(runs),
                "successes": len(successes),
                "success_rate": len(successes) / len(runs),
                "steps_when_successful": {
                    "min": min(steps),
                    "mean": statistics.mean(steps),
                    "max": max(steps),
                }
                if steps
                else None,
                "terminations": dict(
                    Counter(
                        str(o.get("termination_reason"))
                        for seed, o in outcomes.items()
                        if not o.get("task_success")
                    )
                ),
                "failure_families": dict(
                    Counter(
                        g["labels"]["failure_family"]
                        for g in grades.values()
                        if g and not g.get("success")
                    )
                ),
                "graded": sum(1 for g in grades.values() if g),
                "not_reproduced": sorted(
                    seed for seed, g in grades.items() if g and not g.get("reproduced")
                ),
                "seeds": {
                    str(seed): {
                        "run": experiment.rel(run),
                        "success": bool(outcomes[seed].get("task_success")),
                        "steps": outcomes[seed].get("steps_executed"),
                        "termination": outcomes[seed].get("termination_reason"),
                        "family": (grades[seed] or {}).get("labels", {}).get("failure_family"),
                    }
                    for seed, run in sorted(runs.items())
                },
            }
        )
    summary = {
        "scene_set": scene_set,
        "tasks": tasks,
        "episodes": sum(t["episodes"] for t in tasks),
        "successes": sum(t["successes"] for t in tasks),
        "mean": statistics.mean(t["success_rate"] for t in tasks) if tasks else None,
    }
    if write and tasks:
        write_json(experiment.root / "teacher" / scene_set / "results.json", summary)
    return summary


# ------------------------------------------------------------------------------ grading --
def grade_scene_set(
    experiment: Experiment, scene_set: str, tasks: Iterable[int], *, log: Any = print
) -> dict[str, Any]:
    """Joint-replay every ungraded teacher episode; store labels and simulator states."""
    from cap_harness.trace2vla import replay

    adapter = replay.joint_adapter()
    counts: Counter[str] = Counter()
    graded = 0
    try:
        for task_id in tasks:
            out = experiment.grade_dir(scene_set, task_id)
            for seed, run in sorted(episodes_of(experiment, scene_set, task_id).items()):
                target = out / f"seed_{seed}.json"
                if target.is_file():
                    counts[
                        (read_json(target) or {}).get("labels", {}).get("failure_family", "unknown")
                    ] += 1
                    continue
                episode = load_teacher_episode(run)
                rollout = replay.joint_replay(adapter, episode)
                out.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    out / f"seed_{seed}.npz",
                    start_state=rollout.start_state,
                    end_state=rollout.end_state,
                )
                write_json(
                    target,
                    {
                        "run": experiment.rel(run),
                        "scene": episode.scene,
                        "success": rollout.success,
                        "recorded_success": episode.success,
                        "termination": episode.termination,
                        "steps": rollout.steps,
                        "labels": rollout.labels,
                        **rollout.stats,
                    },
                )
                graded += 1
                counts[rollout.labels.get("failure_family", "unknown")] += 1
                log(
                    f"task {task_id} seed {seed}: {rollout.labels.get('failure_family')}"
                    + (
                        ""
                        if rollout.stats["reproduced"]
                        else f" (NOT reproduced: joints differ by {rollout.stats['max_joint_difference_rad']:.4f} rad)"
                    )
                )
    finally:
        adapter.close()
    experiment.log(
        "teacher_graded",
        scene_set=scene_set,
        tasks=list(tasks),
        graded=graded,
        families=dict(counts),
    )
    results(experiment, scene_set, write=True)
    return {"graded": graded, "families": dict(counts)}
