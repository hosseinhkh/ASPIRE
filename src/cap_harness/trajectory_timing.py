"""Execute planned joint trajectories at a speed the joint controller can actually follow.

LIBERO and Robosuite run robosuite's JOINT_POSITION controller (kp = 50, critically damped,
goal moved at most ``output_max = 0.05`` rad per 20 Hz control step). The planners time their
trajectories by what the controller *accepts* per step (0.05 rad), but what the arm *achieves*
is set by the PD gains: with the goal clipped 0.05 rad ahead, a joint settles at

    v = (kp / kd) * output_max / (1 + (kp / kd) * dt / 2)  ~ 0.16 rad/s ~ 0.008 rad per step,

six times slower than planned (measured in recorded episodes: 0.0079 rad per step). On a long
move the arm falls up to a radian behind the plan, every joint moving at the same capped speed,
so the hand leaves the planned path; then the fixed 120-step settle runs out while the arm is
still moving, and the move fails with "final trajectory waypoint did not converge" even though
nothing blocks it.

The fix is the one real robot drivers use, time-scaling to the actuator's limits:

* ``trackable_step_rad``: the per-step joint change the controller follows closely, derived
  from its own gains (85 % of the saturated speed above), so other robots and gains get their
  own limit;
* ``retime``: insert intermediate waypoints so no step asks a joint to move more than that;
  the arm follows the planned path with a few hundredths of a radian of lag;
* ``Settler``: after the last waypoint keep commanding it while the error is still falling.
  It never stops before the original 120-step budget, and stops early only on a stall (an arm
  blocked by contact) or at ``SETTLE_MAX_STEPS``.

On by default. ``CAP_HARNESS_COMPAT=published`` (or ``CAP_HARNESS_TRAJECTORY_RETIMING=0``)
executes plans exactly as the published release does, for comparisons with the paper.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import os
from typing import Any

import numpy as np

from cap_harness.compat import published_behavior

#: Per-step joint change used when a controller's gains are not available.
TRACKABLE_JOINT_STEP_RAD = 0.006
#: Fraction of the controller's saturated speed that trajectories are timed to.
TRACKING_MARGIN = 0.85
#: The original settle budget; settling never ends before it.
SETTLE_MIN_STEPS = 119
#: Hard cap on settle steps after the last waypoint (30 s of simulated time at 20 Hz).
SETTLE_MAX_STEPS = 600
#: Past SETTLE_MIN_STEPS, stop when the error has not fallen by SETTLE_PROGRESS_RAD
#: within SETTLE_STALL_STEPS steps.
SETTLE_STALL_STEPS = 25
SETTLE_PROGRESS_RAD = 0.002


def retiming_enabled() -> bool:
    """On unless CAP_HARNESS_COMPAT=published; CAP_HARNESS_TRAJECTORY_RETIMING=0/1 overrides."""
    value = os.environ.get("CAP_HARNESS_TRAJECTORY_RETIMING", "").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return not published_behavior()


def trackable_step_rad(
    kp: Any, kd: Any, output_max: Any, period_s: float, margin: float = TRACKING_MARGIN
) -> float:
    """Per-step joint change a goal-clipped PD joint controller follows (smallest joint)."""
    kp, kd, limit = (np.atleast_1d(np.asarray(v, dtype=float)) for v in (kp, kd, output_max))
    ratio = kp / np.maximum(kd, 1e-9)
    speed = ratio * limit / (1.0 + ratio * period_s / 2.0)  # rad/s at saturation
    step = float(np.min(speed)) * period_s * margin
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("controller gains give no positive trackable speed")
    return step


def controller_step_rad(controllers: Iterable[Any], period_s: float) -> float:
    """``trackable_step_rad`` for robosuite controllers (kp, kd, output_max), or the default."""
    steps = []
    for controller in controllers:
        try:
            steps.append(
                trackable_step_rad(controller.kp, controller.kd, controller.output_max, period_s)
            )
        except (AttributeError, TypeError, ValueError):
            continue
    return min(steps) if steps else TRACKABLE_JOINT_STEP_RAD


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
    "controller_step_rad",
    "retime",
    "retiming_enabled",
    "trackable_step_rad",
]
