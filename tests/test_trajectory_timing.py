"""Trajectory retiming and progress-based settling (cap_harness.trajectory_timing)."""

from __future__ import annotations

import numpy as np
import pytest

from cap_harness.trajectory_timing import (
    SETTLE_MAX_STEPS,
    SETTLE_MIN_STEPS,
    TRACKABLE_JOINT_STEP_RAD,
    Settler,
    retime,
    retiming_enabled,
)


def test_retime_bounds_every_step_and_keeps_the_endpoints() -> None:
    # A 1.4 rad move in 30 waypoints (0.049 rad per step), as the interpolation planner makes.
    plan = np.linspace([0.0] * 7, [0.0, 1.4, 0.0, 1.4, 0.0, -0.3, 0.0], 30)
    retimed, grippers = retime({"primary": plan}, {"primary": np.linspace(1.0, 0.0, 30)})
    path = retimed["primary"]
    assert np.abs(np.diff(path, axis=0)).max() <= TRACKABLE_JOINT_STEP_RAD + 1e-12
    np.testing.assert_allclose(path[0], plan[0])
    np.testing.assert_allclose(path[-1], plan[-1])
    assert len(path) >= 1.4 / TRACKABLE_JOINT_STEP_RAD
    assert grippers is not None and len(grippers["primary"]) == len(path)
    assert grippers["primary"][0] == 1.0 and grippers["primary"][-1] == 0.0


def test_retime_keeps_arms_synchronized_and_short_plans_untouched() -> None:
    plan = {
        "primary": np.array([[0.0] * 7, [0.1] * 7]),
        "secondary": np.array([[0.0] * 7, [0.01] * 7]),
    }
    retimed, _ = retime(plan)
    assert len(retimed["primary"]) == len(retimed["secondary"])  # the slower arm sets the pace
    small = {"primary": np.array([[0.0] * 7, [0.004] * 7])}
    assert len(retime(small)[0]["primary"]) == 2


def test_settler_keeps_going_while_progressing_and_stops_on_stall() -> None:
    converging = Settler(tolerance_rad=0.01)
    error, steps = 0.6, 0
    while converging.keep_going(error):  # 0.004 rad per step: needs ~150 steps
        error -= 0.004
        steps += 1
    assert error <= 0.01 and steps > SETTLE_MIN_STEPS and not converging.stalled

    blocked = Settler(tolerance_rad=0.01)
    stuck_steps = 0
    while blocked.keep_going(0.3):  # an arm pressed against something
        stuck_steps += 1
    assert blocked.stalled and stuck_steps == SETTLE_MIN_STEPS  # never shorter than before


def test_settler_has_a_hard_cap() -> None:
    settler = Settler(tolerance_rad=0.0)
    error, steps = 10.0, 0
    while settler.keep_going(error):
        error -= 0.003  # always progressing, never arriving
        steps += 1
    assert steps == SETTLE_MAX_STEPS


@pytest.mark.parametrize(("value", "enabled"), [("1", True), ("on", True), ("0", False)])
def test_retiming_switch(monkeypatch: pytest.MonkeyPatch, value: str, enabled: bool) -> None:
    monkeypatch.setenv("CAP_HARNESS_TRAJECTORY_RETIMING", value)
    assert retiming_enabled() is enabled


def test_retiming_is_off_by_default_so_runs_match_the_published_harness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CAP_HARNESS_TRAJECTORY_RETIMING", raising=False)
    assert retiming_enabled() is False


def test_published_settle_stops_after_119_steps_even_while_converging() -> None:
    settler = Settler(0.01, extended=False)
    error, steps = 1.0, 0
    while settler.keep_going(error):
        error *= 0.99  # still converging, never within tolerance in time
        steps += 1
    assert steps == SETTLE_MIN_STEPS == 119
    assert Settler(0.01, extended=False).keep_going(0.005) is False


def test_robosuite_adapter_retimes_both_arms_only_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("robosuite")
    from cap_harness.contracts import SynchronizedTrajectory
    from tests.test_robosuite import _adapter

    def run() -> object:
        adapter, _env, observation = _adapter()
        trajectory = SynchronizedTrajectory(
            joint_positions={
                "primary": np.array([[0.01] * 7, [0.06] * 7]),
                "secondary": np.array([[-0.01] * 7, [-0.06] * 7]),
            },
            dt_s=0.05,
            joint_names=observation.robot_state.joint_names,
            planner="test",
            collision_aware=False,
            expected_start=observation.robot_state,
        )
        return adapter.execute_trajectory(trajectory)

    monkeypatch.delenv("CAP_HARNESS_TRAJECTORY_RETIMING", raising=False)
    published = run()
    assert published.ok and published.steps_executed == 2
    assert "executed_waypoints" not in published.diagnostics

    monkeypatch.setenv("CAP_HARNESS_TRAJECTORY_RETIMING", "1")
    retimed = run()
    assert retimed.ok
    # 0.05 rad in one step becomes nine steps of at most 0.006 rad, for both arms together.
    assert retimed.diagnostics["planned_waypoints"] == 2
    assert retimed.diagnostics["executed_waypoints"] == 10
    for arm, value in (("primary", 0.06), ("secondary", -0.06)):
        np.testing.assert_allclose(
            retimed.final_observation.robot_state.joint_positions[arm], value
        )
