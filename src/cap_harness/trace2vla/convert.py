"""Teacher successes -> demonstrations in the student's control, verified by replay.

For each successful teacher episode:

1. **Re-execute** it in OSC_POSE by following the teacher's hand path at a speed-up
   (``replay.track_in_osc``). Rejected if this no longer solves the task.
2. **Drop idle frames.** A frame is idle when the commanded hand motion is below a threshold
   and the gripper command did not change (the rule the LIBERO training data was filtered
   with). The shortened sequence is then **played open-loop from the scene's reset** and kept
   only if it still solves the task; otherwise a few idle frames are put back after each
   gripper change (``dwell``), and as a last resort all of them.
3. **Record** the final sequence at the student's image size: what the student sees before
   each action, next to that action. Every stored action is therefore one that was really
   executed from the stored observation, and the sequence as stored solves the task.

    <demos>/<name>/task_<id>/seed_<n>/
        episode.json   source run, scene, settings, step counts, dwell, labels, checks
        steps.npz      action, state, joint_pos, eef_pos, gripper_qpos (the stored sequence);
                       tracked_action (before dropping), kept (indices kept);
                       start_state, end_state (simulator state, privileged)
        video.mp4      both cameras as the student sees them
"""

from __future__ import annotations

import atexit
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any

import numpy as np

from cap_harness.trace2vla import replay
from cap_harness.trace2vla.episodes import TeacherEpisode, load_teacher_episode
from cap_harness.trace2vla.store import sha256_file, write_json
from cap_harness.vla.protocol import IMAGE_SIZE, max_steps_for, task_language
from cap_harness.vla.rollout import write_video

#: Hand commands below this (in action units: 0.05 = 2.5 mm, 1.4 degrees) count as standing still.
NOOP_THRESHOLD = 0.05
#: Idle frames put back after each gripper change, tried in this order.
DWELL_OPTIONS: tuple[int, ...] = (0, 4, 8)


@dataclass
class ConvertSettings:
    track: replay.TrackSettings = field(default_factory=replay.TrackSettings)
    noop_threshold: float = NOOP_THRESHOLD
    dwell_options: tuple[int, ...] = DWELL_OPTIONS
    image_size: int = IMAGE_SIZE
    drop_idle: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "track": self.track.to_dict(),
            "noop_threshold": self.noop_threshold,
            "dwell_options": list(self.dwell_options),
            "image_size": self.image_size,
            "drop_idle": self.drop_idle,
        }


def idle_frames(actions: np.ndarray, threshold: float = NOOP_THRESHOLD) -> np.ndarray:
    """True where a frame commands (almost) no hand motion and no gripper change."""
    actions = np.asarray(actions, dtype=np.float64).reshape(-1, 7)
    still = np.abs(actions[:, :6]).max(axis=1) < threshold
    same_gripper = np.ones(len(actions), dtype=bool)
    same_gripper[1:] = actions[1:, 6] == actions[:-1, 6]
    return still & same_gripper


def kept_indices(actions: np.ndarray, threshold: float, dwell: int) -> np.ndarray:
    """Indices to keep: every moving frame, plus ``dwell`` idle frames after a gripper change."""
    actions = np.asarray(actions, dtype=np.float64).reshape(-1, 7)
    idle = idle_frames(actions, threshold)
    keep = ~idle
    if dwell > 0:
        for change in np.flatnonzero(actions[1:, 6] != actions[:-1, 6]) + 1:
            # The idle frames right after a gripper change: the gripper is still moving.
            for index in range(change + 1, min(change + 1 + dwell, len(actions))):
                if not idle[index]:
                    break
                keep[index] = True
    return np.flatnonzero(keep)


def convert_episode(
    episode: TeacherEpisode,
    out_dir: Path,
    *,
    fast: Any,
    recorder: Any,
    settings: ConvertSettings | None = None,
) -> dict[str, Any]:
    """Convert one teacher episode. ``fast`` and ``recorder`` are OSC adapters (small and
    student-size images). Returns the episode summary; writes the demo only when accepted.
    """
    settings = settings or ConvertSettings()
    started = time.monotonic()
    limit = max_steps_for(episode.suite)
    summary: dict[str, Any] = {
        "source_run": str(episode.run_dir),
        "source_steps_sha256": sha256_file(episode.run_dir / "episode" / "steps.jsonl"),
        "scene": episode.scene,
        "instruction": task_language(episode.suite, episode.task_id),
        "settings": settings.to_dict(),
        "teacher_steps": episode.steps,
        "student_step_limit": limit,
        "accepted": False,
    }
    if not episode.success:
        return {**summary, "rejected": "teacher_failed"}
    tracked = replay.track_in_osc(fast, episode, settings.track)
    summary.update(tracked_steps=tracked.steps, tracking=tracked.stats)
    if not tracked.success:
        return {**summary, "rejected": "re-execution_failed", "labels": tracked.labels}

    chosen, dwell, tried = np.arange(tracked.steps), "all", []
    if settings.drop_idle:
        for option in settings.dwell_options:
            indices = kept_indices(tracked.actions, settings.noop_threshold, option)
            if len(indices) == tracked.steps:
                continue
            check = replay.open_loop(fast, episode.scene, tracked.actions[indices])
            tried.append({"dwell": option, "steps": int(len(indices)), "success": check.success})
            if check.success:
                chosen, dwell = indices, option
                break
    # Replay exactly what steps.npz stores (float32): the recorded states then match any later
    # replay of the stored actions bit for bit. Replaying the float64 actions instead gave states
    # that a replay of their float32 copy misses by up to 5e-5 after contact.
    stored = tracked.actions[chosen].astype(np.float32)
    final = replay.open_loop(recorder, episode.scene, stored, record=True)
    summary.update(
        idle_frames=int(idle_frames(tracked.actions, settings.noop_threshold).sum()),
        idle_tried=tried,
        dwell=dwell,
        steps=final.steps,
        within_student_limit=final.steps <= limit,
        labels=final.labels,
    )
    if not final.success:
        return {**summary, "rejected": "stored_sequence_failed"}

    out_dir.mkdir(parents=True, exist_ok=True)
    frames = final.frames
    np.savez_compressed(
        out_dir / "steps.npz",
        action=final.actions.astype(np.float32),
        state=np.stack(frames["state"]),
        joint_pos=np.stack(frames["joint_pos"]),
        eef_pos=np.stack(frames["eef_pos"]),
        gripper_qpos=np.stack(frames["gripper_qpos"]),
        tracked_action=tracked.actions.astype(np.float32),
        kept=chosen[: final.steps],
        start_state=final.start_state,
        end_state=final.end_state,
    )
    write_video(
        out_dir / "video.mp4",
        [
            np.concatenate([a, b], axis=1)
            for a, b in zip(frames["image"], frames["image2"], strict=True)
        ],
    )
    summary.update(
        accepted=True,
        actions_sha256=sha256_file(out_dir / "steps.npz"),
        command_abs_mean=round(float(np.abs(final.actions[:, :3]).mean()), 4),
        wall_s=round(time.monotonic() - started, 2),
    )
    write_json(out_dir / "episode.json", summary)  # last: its presence marks a finished demo
    return summary


def convert_run(
    run_dir: Path,
    out_dir: Path,
    *,
    fast: Any,
    recorder: Any,
    settings: ConvertSettings | None = None,
) -> dict[str, Any]:
    return convert_episode(
        load_teacher_episode(run_dir), out_dir, fast=fast, recorder=recorder, settings=settings
    )


#: This process's simulator adapters for conversion jobs (made on first use, kept between jobs).
_ADAPTERS: dict[str, Any] = {}


def _adapters(image_size: int) -> tuple[Any, Any]:
    """Two OSC adapters: small images for the checks, student-size images for the recording."""
    if "fast" not in _ADAPTERS:
        _ADAPTERS["fast"] = replay.osc_adapter(64)
        _ADAPTERS["recorder"] = replay.osc_adapter(image_size)
        atexit.register(close_adapters)
    return _ADAPTERS["fast"], _ADAPTERS["recorder"]


def close_adapters() -> None:
    for key in ("fast", "recorder"):
        adapter = _ADAPTERS.pop(key, None)
        if adapter is not None:
            adapter.close()


def convert_job(
    task_id: int, seed: int, run: str, out: str, scene_set: str, settings: ConvertSettings
) -> dict[str, Any]:
    """Convert one teacher episode into ``out``: one unit of work, in this process or a worker.

    Returns the summary; {"unreadable": why} when the recording cannot be read, {"skipped":
    "teacher_failed"} when the teacher did not succeed (nothing is written then).
    """
    try:
        episode = load_teacher_episode(Path(run))
    # One unreadable recording must not stop the conversion of the others.
    except (OSError, ValueError, KeyError) as exc:
        return {"unreadable": str(exc)}
    if not episode.success:
        return {"skipped": "teacher_failed"}
    fast, recorder = _adapters(settings.image_size)
    target = Path(out)
    summary = convert_episode(episode, target, fast=fast, recorder=recorder, settings=settings)
    summary["scene_set"] = scene_set
    if summary["accepted"]:
        write_json(target / "episode.json", summary)
    else:
        write_json(target.with_name(target.name + ".rejected.json"), summary)
    return summary


def convert_scene_set(
    experiment: Any,
    scene_set: str,
    name: str,
    tasks: list[int],
    settings: ConvertSettings | None = None,
    *,
    workers: int = 1,
    log: Any = print,
) -> dict[str, Any]:
    """Convert every successful teacher episode of a scene set; skip the ones already done.

    ``workers`` > 1 converts that many episodes at once, each worker with its own simulator.
    A second conversion into the same demo set and task, while this one runs, is refused
    (store.Busy).
    """
    from cap_harness.trace2vla.store import read_json

    settings = settings or ConvertSettings()
    root = experiment.demo_dir(name)
    root.mkdir(parents=True, exist_ok=True)
    stored = read_json(root / "settings.json")
    if stored and stored != settings.to_dict():
        raise ValueError(f"demo set {name!r} was made with other settings; choose another --name")
    keys = [f"demos-{name}-task{task_id:02d}" for task_id in tasks]
    with experiment.hold(keys, f"convert of '{scene_set}' into '{name}'"):
        claim_demo_set(root, scene_set)
        write_json(root / "settings.json", settings.to_dict())
        return _convert_scene_set(experiment, scene_set, name, root, tasks, settings, workers, log)


def claim_demo_set(root: Path, scene_set: str) -> None:
    """A demo set holds the episodes of one scene set: its folders are named by task and seed
    only, so a second scene set's seed 1001 would be taken for the first one's. Refuse that.
    """
    from cap_harness.trace2vla.progress import demo_set_source

    # Made before source.json existed: the episodes say where they came from.
    owner = demo_set_source(root)
    if owner is not None and owner != scene_set:
        raise ValueError(
            f"demo set {root.name!r} holds the episodes of scene set {owner!r}; convert "
            f"'{scene_set}' into another demo set (--demos NAME, or leave --demos out)"
        )
    write_json(root / "source.json", {"scene_set": scene_set})


def _convert_scene_set(
    experiment: Any,
    scene_set: str,
    name: str,
    root: Path,
    tasks: list[int],
    settings: ConvertSettings,
    workers: int,
    log: Any,
) -> dict[str, Any]:
    from cap_harness.trace2vla import parallel
    from cap_harness.trace2vla.store import read_json
    from cap_harness.trace2vla.teacher import episodes_of

    jobs: list[tuple[Any, ...]] = []
    stale: list[tuple[int, int]] = []  # demos made from a teacher episode that was replaced
    for task_id in tasks:
        for seed, run in sorted(episodes_of(experiment, scene_set, task_id).items()):
            out = root / f"task_{task_id:02d}" / f"seed_{seed}"
            rejected = out.with_name(out.name + ".rejected.json")
            made = read_json(out / "episode.json") or read_json(rejected)
            if made:
                if made.get("source_run") and Path(made["source_run"]).name != run.name:
                    stale.append((task_id, seed))
                continue
            jobs.append((task_id, seed, str(run), str(out), scene_set, settings))
    if jobs and workers > 1:
        log(f"converting {len(jobs)} episodes of '{scene_set}' with {workers} workers")
    counts = {"new": 0}
    failed: list[str] = []

    def done(job: tuple[Any, ...], summary: dict[str, Any]) -> None:
        task_id, seed = job[0], job[1]
        if "unreadable" in summary:
            log(
                f"task {task_id} seed {seed}: SKIPPED, the recording cannot be read ({summary['unreadable']})"
            )
            return
        if summary.get("skipped"):
            return
        counts["new"] += 1
        if summary["accepted"]:
            log(
                f"task {task_id} seed {seed}: {summary['teacher_steps']} teacher steps -> "
                f"{summary['steps']} (idle frames dropped: {summary['tracked_steps'] - summary['steps']}, "
                f"dwell {summary['dwell']})"
            )
        else:
            log(f"task {task_id} seed {seed}: REJECTED ({summary['rejected']})")

    def broke(job: tuple[Any, ...], error: BaseException) -> None:
        failed.append(f"task {job[0]} seed {job[1]}")
        log(
            f"task {job[0]} seed {job[1]}: FAILED ({type(error).__name__}: {error}); rerun to retry"
        )

    parallel.run_jobs(
        convert_job, jobs, workers=workers, on_result=done, on_error=broke, cleanup=close_adapters
    )
    if stale:
        log(
            f"WARNING: {len(stale)} demos of '{name}' were made from teacher episodes that have "
            "since been replaced (a new program, or a rerun). They were left as they are; "
            "convert into a new demo set (--demos NEW_NAME) to use the current episodes."
        )
    summary = demo_summary(experiment, name)
    experiment.log(
        "demos_converted", name=name, scene_set=scene_set, tasks=tasks, new=counts["new"],
        accepted=summary["accepted"], rejected=summary["rejected"], settings=settings.to_dict(),
        failed=failed, workers=workers,
    )  # fmt: skip
    return {**summary, "failed": failed}


def demo_summary(experiment: Any, name: str) -> dict[str, Any]:
    """Yield and step counts of a demo set, per task."""
    from cap_harness.trace2vla.store import read_json

    root = experiment.demo_dir(name)
    tasks = []
    for task_dir in sorted(root.glob("task_*")):
        accepted = [read_json(p) or {} for p in sorted(task_dir.glob("seed_*/episode.json"))]
        rejected = [read_json(p) or {} for p in sorted(task_dir.glob("seed_*.rejected.json"))]
        steps = [int(e["steps"]) for e in accepted]
        tasks.append(
            {
                "task_id": int(task_dir.name.split("_")[1]),
                "accepted": len(accepted),
                "rejected": len(rejected),
                "rejected_reasons": sorted({str(e.get("rejected")) for e in rejected}),
                "steps": {"min": min(steps), "mean": float(np.mean(steps)), "max": max(steps)}
                if steps
                else None,
                "teacher_steps_mean": float(np.mean([e["teacher_steps"] for e in accepted]))
                if accepted
                else None,
                "idle_dropped_mean": float(
                    np.mean([e["tracked_steps"] - e["steps"] for e in accepted])
                )
                if accepted
                else None,
                "kept_all_idle": sum(1 for e in accepted if e.get("dwell") == "all"),
                "over_student_limit": sum(1 for e in accepted if not e.get("within_student_limit")),
            }
        )
    summary = {
        "name": name,
        "settings": read_json(root / "settings.json"),
        "tasks": tasks,
        "accepted": sum(t["accepted"] for t in tasks),
        "rejected": sum(t["rejected"] for t in tasks),
    }
    write_json(root / "summary.json", summary)
    return summary
