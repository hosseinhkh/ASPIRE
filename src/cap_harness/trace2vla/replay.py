"""Replays of an episode in the simulator (no model services, no LLM).

    joint_replay      send a teacher episode's recorded joint commands again: checks that the
                      recording reproduces, and grades it from simulator state
    track_in_osc      re-execute a teacher episode in OSC_POSE, the student's control, by
                      following the teacher's hand path at a speed-up
    open_loop         play a stored OSC action sequence; optionally record what the student
                      would see at every step

All three reset the scene from (suite, task, init mode, seed), which reproduces it exactly.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from cap_harness.contracts import ArmCommand, RobotAction
from cap_harness.libero.adapter import PRIMARY_ARM, LiberoAdapter
from cap_harness.trace2vla.episodes import TeacherEpisode
from cap_harness.vla.grading import LiberoGrader, sim_state
from cap_harness.vla.observation import student_images, student_state
from cap_harness.vla.protocol import CONTROLLER, IMAGE_SIZE, SETTLE_STEPS

POS_MAX, ROT_MAX = 0.05, 0.5  # what one action unit means in OSC_POSE (metres, radians)


@dataclass
class TrackSettings:
    """How the OSC tracker follows the teacher's hand path (docs/trace2vla-review.md, 2.2)."""

    speed: float = 3.0  # teacher steps consumed per control step in free motion
    lookahead: float = 3.0  # teacher steps ahead of the playhead the hand aims at
    hold_error_m: float = 0.02  # the playhead waits while the hand is this far behind it
    idle_step_m: float = 2e-4  # teacher hand motion per step below this is "standing still"
    slow_window: int = 12  # teacher steps around a gripper change that play at 1x
    tail_steps: int = 40  # steps to hold the last pose while waiting for success
    max_steps: int = 1500

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class Rollout:
    """The result of one replay. ``frames`` is filled only when recording was requested."""

    success: bool
    steps: int
    actions: np.ndarray
    labels: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
    start_state: np.ndarray | None = None
    end_state: np.ndarray | None = None
    frames: dict[str, list[np.ndarray]] = field(default_factory=dict)


def osc_adapter(image_size: int = IMAGE_SIZE, init_mode: str = "saved") -> LiberoAdapter:
    return LiberoAdapter(
        controller=CONTROLLER,
        camera_height=image_size,
        camera_width=image_size,
        init_mode=init_mode,
        post_settle_steps=SETTLE_STEPS,
    )


def joint_adapter(init_mode: str = "saved", image_size: int = 64) -> LiberoAdapter:
    """The teacher's controller, with the harness's default reset (10 settle steps)."""
    return LiberoAdapter(camera_height=image_size, camera_width=image_size, init_mode=init_mode)


def _reset(adapter: LiberoAdapter, scene: dict[str, Any]) -> None:
    """Reset to the scene in a freshly built environment.

    ASPIRE runs every teacher episode in a new environment. A reused one starts from the same
    state to within 1e-21 but can drift once contacts begin: a recorded close-the-microwave
    episode replayed exactly in a fresh environment and ended 0.18 rad off, unsolved, in a
    reused one. Rebuilding makes every replay of the same actions give the same physics.
    """
    adapter._close_environment()
    if adapter.init_mode != scene["init_mode"]:
        adapter.init_mode = scene["init_mode"]
    adapter.reset(f"{scene['suite']}:{scene['task_id']}", seed=int(scene["seed"]))


# ------------------------------------------------------------------------- joint replay --
def joint_replay(adapter: LiberoAdapter, episode: TeacherEpisode) -> Rollout:
    """Send the recorded joint commands again and compare the robot with the recording."""
    _reset(adapter, episode.scene)
    grader = LiberoGrader(adapter)
    grader.start()
    start = sim_state(adapter)
    worst = float(
        np.abs(
            adapter.get_robot_state().joint_positions[PRIMARY_ARM] - episode.joint_positions[0]
        ).max()
    )
    success, steps = False, 0
    for index in range(episode.steps):
        gripper = float(episode.gripper_commands[index])
        action = RobotAction(
            arms={
                PRIMARY_ARM: ArmCommand(
                    mode="joint_position",
                    target=episode.joint_targets[index],
                    gripper_position=gripper,
                    embodiment=adapter.embodiment,
                )
            }
        )
        # The harness's own encoding of the command, stepped without building the typed
        # camera observation (which a replay does not need).
        native = adapter._codec.encode(
            action, adapter.get_robot_state(), action_spec=adapter._action_bounds
        )
        adapter.step_native_action(native)
        steps += 1
        joints = adapter.get_robot_state().joint_positions[PRIMARY_ARM]
        worst = max(worst, float(np.abs(joints - episode.joint_positions[index + 1]).max()))
        grader.step(adapter.raw_observation, gripper < 0.5)
        if adapter.check_success():
            success = True
            break
    return Rollout(
        success=success,
        steps=steps,
        actions=episode.joint_targets[:steps],
        labels=grader.finish(success),
        stats={
            "recorded_success": episode.success,
            "recorded_steps": episode.steps,
            "reproduced": success == episode.success and worst < 1e-3,
            "max_joint_difference_rad": round(worst, 8),
        },
        start_state=start,
        end_state=sim_state(adapter),
    )


# ----------------------------------------------------------------- OSC re-execution --
def track_in_osc(
    adapter: LiberoAdapter, episode: TeacherEpisode, settings: TrackSettings | None = None
) -> Rollout:
    """Follow the teacher's hand path in OSC_POSE; return the commands that were sent."""
    settings = settings or TrackSettings()
    _reset(adapter, episode.scene)
    grader = LiberoGrader(adapter)
    grader.start()
    start = sim_state(adapter)
    n = episode.steps
    world_from_base = np.linalg.inv(adapter._base_from_world())
    positions = episode.hand_positions @ world_from_base[:3, :3].T + world_from_base[:3, 3]
    wxyz = episode.hand_quaternions_wxyz
    rotations = Rotation.from_matrix(world_from_base[:3, :3]) * Rotation.from_quat(
        wxyz[:, [1, 2, 3, 0]]
    )
    slerp = Slerp(np.arange(n + 1), rotations)
    grippers = episode.gripper_commands
    step_length = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    slow = np.zeros(n + 1, dtype=bool)
    for change in np.flatnonzero(np.diff(grippers) != 0) + 1:
        slow[max(change - settings.slow_window, 0) : change + settings.slow_window] = True
    idle = np.concatenate([[False], step_length < settings.idle_step_m])

    raw = adapter.raw_observation
    start_gap = float(np.linalg.norm(np.asarray(raw["robot0_eef_pos"]) - positions[0]))
    playhead, steps, tail, success = 0.0, 0, 0, False
    actions: list[np.ndarray] = []
    lags: list[float] = []
    while steps < settings.max_steps:
        hand = np.asarray(raw["robot0_eef_pos"], dtype=np.float64)
        hand_rotation = Rotation.from_quat(np.asarray(raw["robot0_eef_quat"], dtype=np.float64))
        index = min(int(round(playhead)), n)
        aim = min(playhead + settings.lookahead, float(n))
        low = int(math.floor(aim))
        fraction = aim - low
        target = (
            positions[low]
            if low >= n
            else (1 - fraction) * positions[low] + fraction * positions[low + 1]
        )
        action = np.zeros(7)
        action[:3] = np.clip((target - hand) / POS_MAX, -1.0, 1.0)
        action[3:6] = np.clip(
            (slerp([aim])[0] * hand_rotation.inv()).as_rotvec() / ROT_MAX, -1.0, 1.0
        )
        closed = grippers[min(index, n - 1)] < 0.5
        action[6] = 1.0 if closed else -1.0
        try:
            raw = adapter.step_native_action(action)
        except ValueError as error:
            # The simulator's own step limit (the adapter's horizon, 1,000) ended the episode
            # before the hand finished the path: a failed re-execution, not an error.
            if "terminated episode" not in str(error):
                raise
            break
        actions.append(action)
        steps += 1
        grader.step(raw, closed)
        if adapter.check_success():
            success = True
            break
        lag = float(np.linalg.norm(np.asarray(raw["robot0_eef_pos"]) - positions[index]))
        lags.append(lag)
        if playhead >= n:
            tail += 1
            if tail > settings.tail_steps:
                break
        elif lag < settings.hold_error_m:
            playhead = min(
                playhead + (1.0 if (slow[index] or idle[index]) else settings.speed), float(n)
            )
    sent = np.asarray(actions).reshape(-1, 7)
    return Rollout(
        success=success,
        steps=steps,
        actions=sent,
        labels=grader.finish(success),
        stats={
            "teacher_steps": n,
            "start_gap_mm": round(1000 * start_gap, 2),
            "lag_mean_mm": round(1000 * float(np.mean(lags)), 2) if lags else None,
            "lag_max_mm": round(1000 * float(np.max(lags)), 2) if lags else None,
            "saturated_fraction": round(
                float((np.abs(sent[:, :6]) >= 0.999).any(axis=1).mean()), 4
            ),
            "command_abs_mean": round(float(np.abs(sent[:, :3]).mean()), 4),
            "gripper_changes": int(np.count_nonzero(np.diff(grippers) != 0)),
        },
        start_state=start,
        end_state=sim_state(adapter),
    )


# ------------------------------------------------------------------------- open loop --
def open_loop(
    adapter: LiberoAdapter,
    scene: dict[str, Any],
    actions: np.ndarray,
    *,
    record: bool = False,
    on_frame: Callable[[dict[str, Any]], None] | None = None,
    extra_steps: int = 0,
) -> Rollout:
    """Play stored OSC actions from the scene's reset. Stops at the first success.

    With ``record`` every step keeps what the student sees *before* the action (both cameras
    rotated as LeRobot does, the 8-value state) next to the action: exactly an imitation
    sample. ``on_frame`` receives the same samples one by one instead of keeping them.
    ``extra_steps`` repeats the last action's gripper command with no motion, for sequences
    whose final frames were dropped.
    """
    _reset(adapter, scene)
    grader = LiberoGrader(adapter)
    grader.start()
    start = sim_state(adapter)
    raw = adapter.raw_observation
    frames: dict[str, list[np.ndarray]] = {
        key: [] for key in ("image", "image2", "state", "joint_pos", "eef_pos", "gripper_qpos")
    }
    sequence = [np.asarray(a, dtype=np.float64) for a in actions]
    if extra_steps and sequence:
        hold = np.zeros(7)
        hold[6] = sequence[-1][6]
        sequence += [hold] * extra_steps
    success, steps = False, 0
    for action in sequence:
        if record or on_frame is not None:
            images = student_images(raw)
            sample = {
                "image": images["image"],
                "image2": images["image2"],
                "state": student_state(raw),
                "joint_pos": np.asarray(raw["robot0_joint_pos"], dtype=np.float64),
                "eef_pos": np.asarray(raw["robot0_eef_pos"], dtype=np.float64),
                "gripper_qpos": np.asarray(raw["robot0_gripper_qpos"], dtype=np.float64),
                "action": action.astype(np.float32),
                "index": steps,
            }
            if on_frame is not None:
                on_frame(sample)
            if record:
                for key, values in frames.items():
                    values.append(sample[key])
        raw = adapter.step_native_action(action)
        steps += 1
        grader.step(raw, action[6] > 0)
        if adapter.check_success():
            success = True
            break
    return Rollout(
        success=success,
        steps=steps,
        actions=np.asarray(sequence[:steps]).reshape(-1, 7),
        labels=grader.finish(success),
        start_state=start,
        end_state=sim_state(adapter),
        frames=frames if record else {},
    )
