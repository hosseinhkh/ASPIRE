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
    final = replay.open_loop(recorder, episode.scene, tracked.actions[chosen], record=True)
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


def convert_scene_set(
    experiment: Any,
    scene_set: str,
    name: str,
    tasks: list[int],
    settings: ConvertSettings | None = None,
    *,
    log: Any = print,
) -> dict[str, Any]:
    """Convert every successful teacher episode of a scene set; skip the ones already done."""
    from cap_harness.trace2vla.store import read_json
    from cap_harness.trace2vla.teacher import episodes_of

    settings = settings or ConvertSettings()
    root = experiment.demo_dir(name)
    root.mkdir(parents=True, exist_ok=True)
    stored = read_json(root / "settings.json")
    if stored and stored != settings.to_dict():
        raise ValueError(f"demo set {name!r} was made with other settings; choose another --name")
    write_json(root / "settings.json", settings.to_dict())
    fast = replay.osc_adapter(64)
    recorder = replay.osc_adapter(settings.image_size)
    new = 0
    try:
        for task_id in tasks:
            for seed, run in sorted(episodes_of(experiment, scene_set, task_id).items()):
                out = root / f"task_{task_id:02d}" / f"seed_{seed}"
                rejected = out.with_name(out.name + ".rejected.json")
                if (out / "episode.json").is_file() or rejected.is_file():
                    continue
                episode = load_teacher_episode(run)
                if not episode.success:
                    continue
                summary = convert_episode(
                    episode, out, fast=fast, recorder=recorder, settings=settings
                )
                summary["scene_set"] = scene_set
                new += 1
                if summary["accepted"]:
                    write_json(out / "episode.json", summary)
                    log(
                        f"task {task_id} seed {seed}: {summary['teacher_steps']} teacher steps -> "
                        f"{summary['steps']} (idle frames dropped: {summary['tracked_steps'] - summary['steps']}, "
                        f"dwell {summary['dwell']})"
                    )
                else:
                    write_json(rejected, summary)
                    log(f"task {task_id} seed {seed}: REJECTED ({summary['rejected']})")
    finally:
        fast.close()
        recorder.close()
    summary = demo_summary(experiment, name)
    experiment.log(
        "demos_converted", name=name, scene_set=scene_set, tasks=tasks, new=new,
        accepted=summary["accepted"], rejected=summary["rejected"], settings=settings.to_dict(),
    )  # fmt: skip
    return summary


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
