"""Stored demo actions are float32: the converter records the replay of exactly those actions,
and the dataset step accepts a successful replay whose states differ only by rounding.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from cap_harness.trace2vla import convert, dataset, replay


def test_replay_problem_accepts_rounding_and_rejects_real_differences() -> None:
    # The rejections seen when building d50: solved, same length, drift 1e-5 .. 5.5e-5.
    for drift in (0.0, 2.98e-08, 1.24e-05, 5.47e-05, 1e-3):
        assert dataset.replay_problem(True, 172, 172, drift) is None
    assert "did not solve" in dataset.replay_problem(False, 172, 172, 0.0)
    assert "after 150 of 172" in dataset.replay_problem(True, 150, 172, 0.0)
    assert "state drift" in dataset.replay_problem(True, 172, 172, 0.02)
    assert "state drift" in dataset.replay_problem(True, 172, 172, float("inf"))
    assert "state drift" in dataset.replay_problem(True, 172, 172, float("nan"))


def test_converter_records_the_replay_of_the_float32_actions_it_stores(
    tmp_path, monkeypatch
) -> None:
    steps = 6
    # float64 actions that float32 cannot hold exactly.
    tracked = np.full((steps, 7), 0.1, dtype=np.float64) + np.arange(steps)[:, None] * 1e-9
    assert not np.array_equal(tracked, tracked.astype(np.float32))
    played: list[tuple[np.ndarray, bool]] = []

    def open_loop(adapter, scene, actions, *, record=False, **_kwargs):
        actions = np.asarray(actions)
        played.append((actions, record))
        frames = {}
        if record:
            frames = {
                "image": [np.zeros((4, 4, 3), np.uint8)] * steps,
                "image2": [np.zeros((4, 4, 3), np.uint8)] * steps,
                "state": [np.zeros(8, np.float32)] * steps,
                "joint_pos": [np.zeros(7)] * steps,
                "eef_pos": [np.zeros(3)] * steps,
                "gripper_qpos": [np.zeros(2)] * steps,
            }
        return replay.Rollout(
            success=True,
            steps=len(actions),
            actions=actions.astype(np.float64),
            frames=frames,
            start_state=np.zeros(3),
            end_state=np.zeros(3),
        )

    monkeypatch.setattr(
        replay,
        "track_in_osc",
        lambda *_a, **_k: replay.Rollout(success=True, steps=steps, actions=tracked),
    )
    monkeypatch.setattr(replay, "open_loop", open_loop)
    monkeypatch.setattr(convert, "write_video", lambda *_a, **_k: None)
    monkeypatch.setattr(convert, "task_language", lambda *_a: "pick the soup")
    monkeypatch.setattr(convert, "max_steps_for", lambda *_a: 280)
    run_dir = tmp_path / "teacher-run"
    (run_dir / "episode").mkdir(parents=True)
    (run_dir / "episode" / "steps.jsonl").write_text("{}\n")
    episode = SimpleNamespace(
        success=True,
        run_dir=run_dir,
        scene={"suite": "libero_object_swap", "task_id": 0, "seed": 1001, "init_mode": "seeded"},
        suite="libero_object_swap",
        task_id=0,
        steps=steps,
    )
    out = tmp_path / "demo"
    summary = convert.convert_episode(
        episode,
        out,
        fast=None,
        recorder=None,
        settings=convert.ConvertSettings(drop_idle=False),
    )
    assert summary["accepted"] is True
    recorded = [actions for actions, record in played if record]
    assert len(recorded) == 1
    stored = np.load(out / "steps.npz")["action"]
    assert stored.dtype == np.float32
    # The recorded replay played exactly the stored values (no float64 extra digits).
    assert np.array_equal(recorded[0].astype(np.float64), stored.astype(np.float64))
