"""Make planned joint trajectories trackable by the simulators' joint-position controllers.

LIBERO and Robosuite run robosuite's JOINT_POSITION controller: each control step it may move
the goal at most ``output_max = 0.05`` rad, and with ``kp = 50`` the arm itself advances only
about 0.008 rad per step (measured at saturation). The motion planners interpolate at up to
0.05 rad per step, so on long moves the arm fell far behind the plan. Each joint moved at the
same capped rate, so joints with further to go lagged, and the hand swept a different, lower
path than the one planned (on LIBERO it dropped from 0.26 m to 0.03 m above the table on a
move to 0.14 m). Then the fixed 120 settle steps ran out while the arm was still converging,
and the move was reported as "final trajectory waypoint did not converge".

Two remedies, used by both adapters:

* ``retime``: insert intermediate waypoints so no step asks a joint to move more than
  ``TRACKABLE_JOINT_STEP_RAD``. The arm then follows the planned path with a few hundredths of
  a radian of lag, and the trajectory takes as long as the robot needs.
* ``Settler``: after the last waypoint, keep commanding it while the error is still falling.
  It never stops before the original 120-step budget, so nothing that converged before fails
  now. It stops early only on a stall (an arm blocked by contact) or at ``SETTLE_MAX_STEPS``.

Both are **off by default**, so the harness executes plans exactly as the published ASPIRE
release does and results stay comparable with the paper (whose agents, and skill library,
learned to stage long moves around this behaviour). Set ``CAP_HARNESS_TRAJECTORY_RETIMING=1``
to turn both on, and compare the two with the dashboard's Compare page.
"""

from __future__ import annotations

from collections.abc import Mapping
import os

import numpy as np

#: Largest per-step joint change the JOINT_POSITION controllers follow closely.
TRACKABLE_JOINT_STEP_RAD = 0.006
#: The original settle budget; settling never ends before it.
SETTLE_MIN_STEPS = 119
#: Hard cap on settle steps after the last waypoint (30 s of simulated time at 20 Hz).
SETTLE_MAX_STEPS = 600
#: Past SETTLE_MIN_STEPS, stop when the error has not fallen by SETTLE_PROGRESS_RAD
#: within SETTLE_STALL_STEPS steps.
SETTLE_STALL_STEPS = 25
SETTLE_PROGRESS_RAD = 0.002


def retiming_enabled() -> bool:
    """Whether ``CAP_HARNESS_TRAJECTORY_RETIMING`` turns re-timing and extended settling on."""
    value = os.environ.get("CAP_HARNESS_TRAJECTORY_RETIMING", "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


def retime(
    positions: Mapping[str, np.ndarray],
    grippers: Mapping[str, np.ndarray] | None = None,
    max_step_rad: float = TRACKABLE_JOINT_STEP_RAD,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray] | None]:
    """Subdivide every segment so no joint of any arm moves more than ``max_step_rad``.

    All arms are subdivided together, so synchronized trajectories stay synchronized. Gripper
    targets are interpolated on the same schedule. Waypoints already close enough are kept as
    they are; the first and last waypoints are unchanged.
    """
    if max_step_rad <= 0:
        raise ValueError("max_step_rad must be positive")
    arms = list(positions)
    arrays = {arm: np.asarray(positions[arm], dtype=np.float64) for arm in arms}
    count = len(arrays[arms[0]])
    if count < 2:
        return arrays, None if grippers is None else {a: np.asarray(g) for a, g in grippers.items()}
    gripper_arrays = (
        None
        if grippers is None
        else {arm: np.asarray(grippers[arm], dtype=np.float64) for arm in grippers}
    )
    out: dict[str, list[np.ndarray]] = {arm: [arrays[arm][0]] for arm in arms}
    gout: dict[str, list[float]] | None = (
        None
        if gripper_arrays is None
        else {arm: [float(g[0])] for arm, g in gripper_arrays.items()}
    )
    for index in range(1, count):
        delta = max(
            float(np.max(np.abs(arrays[arm][index] - arrays[arm][index - 1]))) for arm in arms
        )
        parts = max(1, int(np.ceil(delta / max_step_rad - 1e-9)))
        for part in range(1, parts + 1):
            fraction = part / parts
            for arm in arms:
                start, end = arrays[arm][index - 1], arrays[arm][index]
                out[arm].append(start + fraction * (end - start))
            if gout is not None and gripper_arrays is not None:
                for arm, values in gripper_arrays.items():
                    start_g, end_g = float(values[index - 1]), float(values[index])
                    gout[arm].append(start_g + fraction * (end_g - start_g))
    retimed = {arm: np.asarray(values) for arm, values in out.items()}
    retimed_grippers = (
        None if gout is None else {arm: np.asarray(values) for arm, values in gout.items()}
    )
    return retimed, retimed_grippers


class Settler:
    """Decide whether to keep commanding the final waypoint after a trajectory ends.

    ``extended=False`` is the published harness: at most ``SETTLE_MIN_STEPS`` settle steps,
    whatever the progress.
    """

    def __init__(self, tolerance_rad: float, *, extended: bool = True) -> None:
        self.tolerance_rad = float(tolerance_rad)
        self.extended = bool(extended)
        self.steps = 0
        self.best = float("inf")
        self.since_progress = 0
        self.stalled = False

    def keep_going(self, error_rad: float) -> bool:
        """Call with the current error before each settle step; False means stop."""
        if error_rad <= self.tolerance_rad:
            return False
        if not self.extended:
            if self.steps >= SETTLE_MIN_STEPS:
                return False
            self.steps += 1
            return True
        if error_rad < self.best - SETTLE_PROGRESS_RAD:
            self.best = error_rad
            self.since_progress = 0
        else:
            self.since_progress += 1
        if self.steps >= SETTLE_MAX_STEPS:
            return False
        if self.steps >= SETTLE_MIN_STEPS and self.since_progress >= SETTLE_STALL_STEPS:
            self.stalled = True
            return False
        self.steps += 1
        return True

    def details(self) -> dict[str, object]:
        return {
            "settle_steps": self.steps,
            "stalled": self.stalled,
            "settle_max_steps": SETTLE_MAX_STEPS,
        }


__all__ = [
    "SETTLE_MAX_STEPS",
    "SETTLE_MIN_STEPS",
    "TRACKABLE_JOINT_STEP_RAD",
    "Settler",
    "retime",
    "retiming_enabled",
]
