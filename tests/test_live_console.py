"""The live console's argument handling and robot-thread routing (no simulator needed)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading
import time

import numpy as np

from cap_harness.contracts import MotionStrategy, Pose
from cap_harness.live import RobotThreadSession, _convert_arguments, _tool_schema
from cap_harness.registry import ToolRegistry


class _Spec:
    def __init__(self, function):
        import inspect

        self.function = function
        self.signature = inspect.signature(function)
        self.name = function.__name__


def move(target_pose: Pose, tolerance: float = 0.01, *, strategy: MotionStrategy | None = None):
    return target_pose, strategy


def joints(target: np.ndarray):
    return target


def needs_object(point_cloud: object, lower_xyz: object):  # not expressible as JSON
    return point_cloud


def test_pose_arguments_accept_rounded_quaternions_and_strategies() -> None:
    converted = _convert_arguments(
        _Spec(move),
        {
            "target_pose": {
                "position": [0.4, 0.0, 0.2],
                "quaternion_wxyz": [0.045, 0.7055, 0.7058, 0.0449],  # rounded: not unit length
                "frame": "robot0_base",
            },
            "strategy": {"ik_solver": "pyroki"},
        },
    )
    pose = converted["target_pose"]
    assert isinstance(pose, Pose)
    assert abs(float(np.linalg.norm(pose.quaternion_wxyz)) - 1.0) < 1e-9
    assert isinstance(converted["strategy"], MotionStrategy)
    assert isinstance(
        _convert_arguments(_Spec(joints), {"target": [0, 1, 2]})["target"], np.ndarray
    )


def test_tool_schema_marks_json_callable_tools() -> None:
    schema, callable_as_json = _tool_schema(_Spec(move))
    assert callable_as_json and "position" in schema["properties"]["target_pose"]["properties"]
    assert _tool_schema(_Spec(needs_object))[1] is False


class _Api:
    default_camera = "agentview"
    configured_providers = {"segmentation": "", "ik": (), "pose_planning": ()}

    def get_robot_state(self) -> None:
        return None


def test_jobs_run_on_the_robot_thread() -> None:
    robot = ThreadPoolExecutor(max_workers=1, thread_name_prefix="robot")
    owner = robot.submit(threading.current_thread).result()
    seen: list[threading.Thread] = []
    session = RobotThreadSession(
        _Api(),
        ToolRegistry(),
        on_robot_thread=lambda fn, *a: robot.submit(fn, *a).result(),
    )
    try:
        job = session._start_job(
            kind="tool",
            label="probe",
            phase="act",
            operation=lambda: seen.append(threading.current_thread()) or "done",
        )
        deadline = time.time() + 10
        while (session.job(job) or {}).get("active", True) and time.time() < deadline:
            time.sleep(0.02)
        assert seen == [owner]  # the simulator's GL context lives on this thread only
    finally:
        session.close()
        robot.shutdown()
