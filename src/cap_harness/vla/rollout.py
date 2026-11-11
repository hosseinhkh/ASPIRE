"""One student episode through ``LiberoAdapter`` (OSC_POSE), recorded step by step.

Every episode keeps the robot's own trajectory, so it can be inspected, compared with the
teacher and reused as data later:

    episode.json   task, scene, seeds, success, steps, timing, protocol
    steps.npz      start_state / end_state: the simulator state, when the episode was graded
                   (privileged: for replay and grading, never a policy input);
                   per step t (the observation the action was chosen from, then the action):
                   action (T,7), state (T,8) the policy input, joint_pos (T,7),
                   eef_pos (T,3), eef_quat_xyzw (T,4), gripper_qpos (T,2);
                   final_* hold the observation after the last action
    video.mp4      both cameras as the policy saw them (agentview | wrist), 20 fps
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any, Protocol

import numpy as np

from cap_harness.vla.grading import sim_state
from cap_harness.vla.observation import student_images, student_state


class StudentPolicy(Protocol):
    def reset(self) -> None: ...

    def act(self, raw: Mapping[str, Any], instruction: str) -> np.ndarray: ...


@dataclass
class EpisodeSpec:
    suite: str
    task_id: int
    scene: int  # the adapter seed: saved mode uses init state (scene - 1) % 50
    policy_seed: int
    max_steps: int
    instruction: str
    seed_index: int = 0  # which of the benchmark's policy seeds this episode belongs to

    @property
    def task_ref(self) -> str:
        return f"{self.suite}:{self.task_id}"


@dataclass
class EpisodeRecorder:
    """Collects one episode in memory and writes it to ``directory``."""

    directory: Path
    video: str = "all"  # all | failures | none
    _arrays: dict[str, list[np.ndarray]] = field(default_factory=dict)
    _extras: dict[str, np.ndarray] = field(default_factory=dict)
    _frames: list[np.ndarray] = field(default_factory=list)

    def _push(self, key: str, value: Any) -> None:
        self._arrays.setdefault(key, []).append(np.asarray(value))

    def _robot(self, raw: Mapping[str, Any], prefix: str = "") -> None:
        self._push(prefix + "state", student_state(raw))
        self._push(prefix + "joint_pos", np.asarray(raw["robot0_joint_pos"], dtype=np.float64))
        self._push(prefix + "eef_pos", np.asarray(raw["robot0_eef_pos"], dtype=np.float64))
        self._push(prefix + "eef_quat_xyzw", np.asarray(raw["robot0_eef_quat"], dtype=np.float64))
        self._push(
            prefix + "gripper_qpos", np.asarray(raw["robot0_gripper_qpos"], dtype=np.float64)
        )

    def _frame(self, raw: Mapping[str, Any]) -> None:
        if self.video != "none":
            images = student_images(raw)
            self._frames.append(np.concatenate([images["image"], images["image2"]], axis=1))

    def record_step(self, raw: Mapping[str, Any], action: np.ndarray) -> None:
        self._robot(raw)
        self._push("action", np.asarray(action, dtype=np.float64))
        self._frame(raw)

    def extra(self, key: str, value: Any) -> None:
        """A whole-episode array stored beside the per-step ones (e.g. the simulator state)."""
        self._extras[key] = np.asarray(value)

    def finish(self, raw: Mapping[str, Any], summary: Mapping[str, Any]) -> None:
        self._robot(raw, prefix="final_")
        self._frame(raw)
        self.directory.mkdir(parents=True, exist_ok=True)
        arrays = {
            key: (np.stack(values) if not key.startswith("final_") else values[-1])
            for key, values in self._arrays.items()
        }
        arrays.update(self._extras)
        if "action" not in arrays:
            arrays["action"] = np.zeros((0, 7))
        np.savez_compressed(self.directory / "steps.npz", **arrays)
        keep = self.video == "all" or (self.video == "failures" and not summary.get("success"))
        if keep and self._frames:
            write_video(self.directory / "video.mp4", self._frames)
        # episode.json last: its presence marks a finished episode (resume skips it).
        (self.directory / "episode.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def write_video(path: Path, frames: list[np.ndarray], fps: int = 20) -> None:
    import imageio.v2 as imageio

    with imageio.get_writer(
        path, fps=fps, codec="libx264", quality=7, macro_block_size=1
    ) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))


def episode_diagnostics(
    actions: np.ndarray, hand_positions: np.ndarray, policy_s: float
) -> dict[str, Any]:
    """Signs of a policy or pipeline problem, per episode (all plain numbers).

    action_saturated_fraction: share of hand-motion values at the controller limit (|a| >= 0.999);
    a high share means the policy is pushing as hard as it can (a scale or frame problem).
    hand_path_m / hand_net_m: how far the hand travelled, and how far it ended from its start;
    a hand that barely moves points at wrong inputs.  gripper_switches: open/close changes.
    """
    steps = int(actions.shape[0]) if actions.ndim == 2 else 0
    if steps == 0:
        return {"policy_ms_per_step": None}
    motion = np.abs(actions[:, :6])
    gripper = actions[:, 6] > 0
    moves = (
        np.linalg.norm(np.diff(hand_positions, axis=0), axis=1) if len(hand_positions) > 1 else []
    )
    return {
        "action_abs_mean": round(float(motion.mean()), 4),
        "action_saturated_fraction": round(float((motion >= 0.999).mean()), 4),
        "gripper_closed_fraction": round(float(gripper.mean()), 4),
        "gripper_switches": int(np.count_nonzero(gripper[1:] != gripper[:-1])),
        "hand_path_m": round(float(np.sum(moves)), 4),
        "hand_net_m": round(float(np.linalg.norm(hand_positions[-1] - hand_positions[0])), 4),
        "policy_ms_per_step": round(1000.0 * policy_s / steps, 2),
    }


def run_episode(
    adapter: Any,
    policy: StudentPolicy,
    spec: EpisodeSpec,
    *,
    recorder: EpisodeRecorder | None = None,
    seed_fn: Callable[[int], None] | None = None,
    grader: Any | None = None,
) -> dict[str, Any]:
    """Run one episode; return its summary (and write it when a recorder is given).

    ``grader`` (``cap_harness.vla.grading.LiberoGrader``) labels the episode from simulator
    state for the record: which object was lifted and placed. It only observes; nothing it
    reads goes to the policy.
    """
    started = time.monotonic()
    adapter.reset(spec.task_ref, seed=spec.scene)
    raw = adapter.raw_observation
    if grader is not None:
        grader.start()
        if recorder is not None:
            recorder.extra("start_state", sim_state(adapter))
    policy.reset()
    if seed_fn is not None:
        seed_fn(spec.policy_seed)
    success = False
    terminated = False
    steps = 0
    policy_s = 0.0
    actions: list[np.ndarray] = []
    hand = [np.asarray(raw["robot0_eef_pos"], dtype=np.float64).reshape(3)]
    while steps < spec.max_steps:
        tick = time.monotonic()
        action = np.asarray(policy.act(raw, spec.instruction), dtype=np.float64).reshape(-1)
        policy_s += time.monotonic() - tick
        if recorder is not None:
            recorder.record_step(raw, action)
        raw = adapter.step_native_action(action)
        if grader is not None:
            grader.step(raw, bool(action[6] > 0))
        actions.append(action)
        hand.append(np.asarray(raw["robot0_eef_pos"], dtype=np.float64).reshape(3))
        steps += 1
        success = bool(adapter.check_success())
        terminated = bool(adapter.last_terminated)
        if success or terminated:
            break
    summary = {
        "suite": spec.suite,
        "task_id": spec.task_id,
        "task_ref": spec.task_ref,
        "instruction": spec.instruction,
        "scene": spec.scene,
        "init_state_index": adapter.selected_init_state_index,
        "policy_seed": spec.policy_seed,
        "policy_seed_index": spec.seed_index,
        "max_steps": spec.max_steps,
        "success": success,
        "steps": steps,
        "ended": "success" if success else ("terminated" if terminated else "step_limit"),
        "wall_s": round(time.monotonic() - started, 3),
        "policy_s": round(policy_s, 3),
        "diagnostics": episode_diagnostics(
            np.stack(actions) if actions else np.zeros((0, 7)), np.stack(hand), policy_s
        ),
    }
    if grader is not None:
        summary["labels"] = grader.finish(success)
        if recorder is not None:
            recorder.extra("end_state", sim_state(adapter))
    if recorder is not None:
        recorder.finish(raw, summary)
    return summary
