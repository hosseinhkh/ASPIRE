"""Read a recorded ASPIRE teacher episode (a run folder) without changing it."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


def _array(value: Any) -> np.ndarray:
    if isinstance(value, dict):
        value = value.get("$array_inline", value.get("values"))
    return np.asarray(value, dtype=np.float64)


@dataclass
class TeacherEpisode:
    """What the recorder stored for one teacher run. Step k's state is the state after step k."""

    run_dir: Path
    suite: str
    task_id: int
    seed: int
    init_mode: str
    success: bool
    termination: str | None
    joint_targets: np.ndarray  # (T, 7) the joint-position command sent at each step
    gripper_commands: np.ndarray  # (T,) 1 = open, 0 = closed
    joint_positions: np.ndarray  # (T + 1, 7) row 0 is the reset state
    hand_positions: np.ndarray  # (T + 1, 3) in the robot base frame
    hand_quaternions_wxyz: np.ndarray  # (T + 1, 4) in the robot base frame

    @property
    def steps(self) -> int:
        return int(self.joint_targets.shape[0])

    @property
    def scene(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "task_id": self.task_id,
            "init_mode": self.init_mode,
            "seed": self.seed,
        }


def is_run_dir(path: Path) -> bool:
    return (path / "run.json").is_file() and (path / "episode" / "steps.jsonl").is_file()


def find_runs(root: Path) -> list[Path]:
    """Every finalized run folder under ``root``, oldest first."""
    return sorted(
        (p.parent for p in root.rglob("run.json") if is_run_dir(p.parent)),
        key=lambda p: p.name,
    )


def load_teacher_episode(run_dir: Path) -> TeacherEpisode:
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    outcome_path = run_dir / "outcome.json"
    outcome = json.loads(outcome_path.read_text(encoding="utf-8")) if outcome_path.is_file() else {}
    reset = json.loads((run_dir / "episode" / "reset.json").read_text(encoding="utf-8"))

    def pose(state: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        hand = state["end_effector_poses"]["primary"]
        return (
            _array(state["joint_positions"]["primary"]),
            _array(hand["position"]),
            _array(hand["quaternion_wxyz"]),
        )

    joints, positions, quaternions = (
        [value] for value in pose(reset["observation"]["robot_state"])
    )
    targets, grippers = [], []
    with (run_dir / "episode" / "steps.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            try:
                step = json.loads(line)
            except ValueError:
                continue
            if not step.get("ok") or not step.get("robot_state"):
                continue
            command = step["action"]["arms"]["primary"]
            targets.append(_array(command["target"]))
            gripper = command.get("gripper_position")
            if gripper is None:  # no gripper command this step: the previous one still holds
                gripper = grippers[-1] if grippers else 1.0
            grippers.append(float(gripper))
            joint, position, quaternion = pose(step["robot_state"])
            joints.append(joint)
            positions.append(position)
            quaternions.append(quaternion)
    return TeacherEpisode(
        run_dir=run_dir,
        suite=str(meta["suite"]),
        task_id=int(meta["task_id"]),
        seed=int(meta["seed"]),
        init_mode=str(meta.get("init_mode", "saved")),
        success=bool(outcome.get("task_success")),
        termination=outcome.get("termination_reason"),
        joint_targets=np.asarray(targets, dtype=np.float64).reshape(-1, 7),
        gripper_commands=np.asarray(grippers, dtype=np.float64),
        joint_positions=np.asarray(joints, dtype=np.float64),
        hand_positions=np.asarray(positions, dtype=np.float64),
        hand_quaternions_wxyz=np.asarray(quaternions, dtype=np.float64),
    )
