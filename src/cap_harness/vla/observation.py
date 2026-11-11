"""What the student sees, built from the adapter's raw LIBERO observation.

LeRobot's LIBERO pipeline (``LiberoProcessorStep``) rotates both camera images by 180 degrees
and builds an 8-value state: hand position (3), hand rotation as axis-angle (3) and the two
gripper finger positions (2).  ``lerobot_env_observation`` hands the raw observation to that
pipeline in the shape LeRobot's own LIBERO env produces; ``student_images`` and
``student_state`` compute the same values in numpy, for recording and for datasets.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

AGENTVIEW = "agentview_image"
WRIST = "robot0_eye_in_hand_image"
#: LeRobot's names for the two cameras (observation.images.image / .image2).
CAMERA_KEYS: dict[str, str] = {AGENTVIEW: "image", WRIST: "image2"}


def rotate_180(image: Any) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(image)[::-1, ::-1])


def quat_xyzw_to_axis_angle(quaternion: Any) -> np.ndarray:
    """Axis-angle of an XYZW quaternion, computed as LeRobot's ``_quat2axisangle`` (float32)."""
    quat = np.asarray(quaternion, dtype=np.float32).reshape(4)
    w = np.float32(np.clip(quat[3], -1.0, 1.0))
    den = np.sqrt(np.maximum(np.float32(1.0) - w * w, np.float32(0.0)))
    if den <= 1e-10:
        return np.zeros(3, dtype=np.float32)
    angle = np.float32(2.0) * np.arccos(w)
    return (quat[:3] / den * angle).astype(np.float32)


def student_state(raw: Mapping[str, Any]) -> np.ndarray:
    """The 8-value ``observation.state``: hand position, hand axis-angle, gripper fingers."""
    return np.concatenate(
        [
            np.asarray(raw["robot0_eef_pos"], dtype=np.float32).reshape(3),
            quat_xyzw_to_axis_angle(raw["robot0_eef_quat"]),
            np.asarray(raw["robot0_gripper_qpos"], dtype=np.float32).reshape(2),
        ]
    ).astype(np.float32)


def student_images(raw: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Both cameras as the policy sees them (rotated 180 degrees), keyed image / image2."""
    return {key: rotate_180(raw[name]) for name, key in CAMERA_KEYS.items()}


def lerobot_env_observation(raw: Mapping[str, Any]) -> dict[str, Any]:
    """A batch of one, shaped like LeRobot's ``LiberoEnv`` (obs_type pixels_agent_pos) output.

    Images stay as rendered: LeRobot's ``LiberoProcessorStep`` rotates them, exactly as it
    does for its own env.
    """

    def batch(name: str, size: int) -> np.ndarray:
        return np.asarray(raw[name], dtype=np.float64).reshape(1, size)

    return {
        "pixels": {key: np.asarray(raw[name])[None] for name, key in CAMERA_KEYS.items()},
        "robot_state": {
            "eef": {"pos": batch("robot0_eef_pos", 3), "quat": batch("robot0_eef_quat", 4)},
            "gripper": {"qpos": batch("robot0_gripper_qpos", 2)},
        },
    }
