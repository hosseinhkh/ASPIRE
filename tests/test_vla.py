"""The VLA student path: OSC_POSE adapter mode, observations, rollout recording, results."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cap_harness.libero.adapter import OSC_SETTLE_ACTION, LiberoAdapter
from cap_harness.libero.registry import LiberoTaskMetadata
from cap_harness.vla import benchmark, observation, protocol
from cap_harness.vla.rollout import EpisodeRecorder, EpisodeSpec, run_episode


class OscEnv:
    """A LIBERO env with LIBERO's default OSC_POSE controller: 7 action values in [-1, 1]."""

    def __init__(self, success_after: int | None = None) -> None:
        self.sim = SimpleNamespace(model=None, data=None)
        self.action_spec = (-np.ones(7), np.ones(7))
        self.actions: list[np.ndarray] = []
        self.success_after = success_after
        self.position = np.zeros(3)
        self.seeded: int | None = None
        self.init_state: float | None = None

    def _observation(self) -> dict[str, np.ndarray]:
        image = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
        return {
            "agentview_image": image,
            "robot0_eye_in_hand_image": image + 100,
            "robot0_eef_pos": self.position.copy(),
            "robot0_eef_quat": np.array([0.0, 0.0, np.sin(0.25), np.cos(0.25)]),
            "robot0_gripper_qpos": np.array([0.04, -0.04]),
            "robot0_joint_pos": np.arange(7, dtype=np.float64),
        }

    def seed(self, seed: int) -> None:
        self.seeded = seed

    def reset(self) -> dict[str, np.ndarray]:
        self.position = np.zeros(3)
        return self._observation()

    def set_init_state(self, state: float) -> dict[str, np.ndarray]:
        self.init_state = state
        return self._observation()

    def step(self, action: np.ndarray):
        self.actions.append(np.asarray(action, dtype=np.float64).copy())
        self.position = self.position + np.asarray(action[:3]) * 0.01
        return self._observation(), 0.0, False, {}

    def check_success(self) -> bool:
        stepped = len(self.actions) - protocol.SETTLE_STEPS
        return self.success_after is not None and stepped >= self.success_after

    def close(self) -> None:
        pass


class Registry:
    def __init__(self, metadata: LiberoTaskMetadata) -> None:
        self.metadata = metadata

    def resolve(self, task_ref):
        return self.metadata

    def get_init_states(self, metadata):
        return [0.1, 0.2, 0.3]


def _adapter(tmp_path, env: OscEnv, controller: str = "OSC_POSE") -> tuple[LiberoAdapter, dict]:
    bddl = tmp_path / "task.bddl"
    bddl.write_text("(define (problem p) (:language pick up the soup))", encoding="utf-8")
    metadata = LiberoTaskMetadata(
        suite_name="libero_object_swap",
        task_id=3,
        task_name="pick_soup",
        family="object",
        bddl_path=bddl,
        language="pick up the soup",
        init_state_count=3,
    )
    kwargs: dict = {}

    def factory(**values):
        kwargs.update(values)
        return env

    adapter = LiberoAdapter(
        registry=Registry(metadata),
        env_factory=factory,
        controller=controller,
        camera_height=2,
        camera_width=3,
        post_settle_steps=protocol.SETTLE_STEPS,
        depth_converter=lambda sim, depth: depth,
    )
    return adapter, kwargs


def test_adapter_rejects_unknown_controllers() -> None:
    with pytest.raises(ValueError, match="JOINT_POSITION and OSC_POSE"):
        LiberoAdapter(controller="OSC_POSITION")


def test_osc_mode_settles_like_lerobot_and_steps_raw_actions(tmp_path) -> None:
    env = OscEnv()
    adapter, kwargs = _adapter(tmp_path, env)
    adapter.get_observation = lambda: None  # the typed observation is not under test here
    adapter.reset("libero_object_swap:3", seed=2)

    assert kwargs["controller"] == "OSC_POSE"
    assert env.init_state == pytest.approx(0.2)  # scene 2 is saved init state 1
    assert len(env.actions) == protocol.SETTLE_STEPS
    for action in env.actions:
        np.testing.assert_array_equal(action, OSC_SETTLE_ACTION)
    assert adapter.elapsed_steps == 0

    raw = adapter.step_native_action([2.0, 0.5, 0, 0, 0, 0, 1.0])
    np.testing.assert_array_equal(env.actions[-1], [1.0, 0.5, 0, 0, 0, 0, 1.0])  # clipped
    assert adapter.elapsed_steps == 1
    np.testing.assert_allclose(raw["robot0_eef_pos"], [0.01, 0.005, 0.0])
    assert adapter.raw_observation is raw
    with pytest.raises(ValueError, match="expects 7 action values"):
        adapter.step_native_action(np.zeros(8))
    with pytest.raises(ValueError, match="finite"):
        adapter.step_native_action([np.nan] * 7)


def test_osc_mode_refuses_the_joint_space_harness_tools(tmp_path) -> None:
    adapter, _ = _adapter(tmp_path, OscEnv())
    adapter.get_observation = lambda: None
    adapter.reset("libero_object_swap:3", seed=1)
    for call in (
        lambda: adapter.native_step(None),
        lambda: adapter.execute_trajectory(None),
        lambda: adapter.set_grippers({"primary": 1.0}),
        lambda: adapter.close_gripper(),
    ):
        with pytest.raises(RuntimeError, match="needs the JOINT_POSITION controller"):
            call()


def test_osc_mode_requires_a_seven_value_action_spec(tmp_path) -> None:
    env = OscEnv()
    env.action_spec = (-np.ones(8), np.ones(8))
    adapter, _ = _adapter(tmp_path, env)
    with pytest.raises(ValueError, match="7-D action spec"):
        adapter.reset("libero_object_swap:3", seed=1)


def test_suite_limits_follow_lerobot_including_libero_pro_suites() -> None:
    assert protocol.max_steps_for("libero_object") == 280
    assert protocol.max_steps_for("libero_object_swap") == 280
    assert protocol.max_steps_for("libero_goal_lan") == 300
    assert protocol.max_steps_for("libero_10_swap") == 520
    assert protocol.max_steps_for("libero_spatial_task") == 280
    assert protocol.max_steps_for("libero_mine") == protocol.DEFAULT_MAX_STEPS
    assert protocol.suite_family("libero_90") == "libero_90"


def test_parse_int_list() -> None:
    assert protocol.parse_int_list("0,3,5-7") == [0, 3, 5, 6, 7]
    assert protocol.parse_int_list("all", default=range(3)) == [0, 1, 2]
    assert protocol.parse_int_list("1-50")[-1] == 50
    with pytest.raises(ValueError):
        protocol.parse_int_list("5-3")


@pytest.mark.parametrize("angle", [0.0, 0.3, 1.7, 3.0])
def test_axis_angle_matches_a_reference_rotation(angle: float) -> None:
    axis = np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
    quat_xyzw = Rotation.from_rotvec(axis * angle).as_quat()
    if quat_xyzw[3] < 0:
        quat_xyzw = -quat_xyzw
    np.testing.assert_allclose(
        observation.quat_xyzw_to_axis_angle(quat_xyzw), axis * angle, atol=1e-5
    )


def test_student_state_and_images() -> None:
    raw = OscEnv()._observation()
    state = observation.student_state(raw)
    assert state.dtype == np.float32 and state.shape == (8,)
    np.testing.assert_allclose(state[3:6], [0.0, 0.0, 0.5], atol=1e-6)
    np.testing.assert_allclose(state[6:], [0.04, -0.04], atol=1e-7)
    images = observation.student_images(raw)
    np.testing.assert_array_equal(images["image"], raw["agentview_image"][::-1, ::-1])
    np.testing.assert_array_equal(images["image2"], raw["robot0_eye_in_hand_image"][::-1, ::-1])
    batch = observation.lerobot_env_observation(raw)
    assert batch["pixels"]["image"].shape == (1, 2, 3, 3)
    assert batch["robot_state"]["eef"]["quat"].shape == (1, 4)


def test_student_observation_matches_lerobots_libero_processor() -> None:
    """Our numpy state and image rotation equal what LeRobot feeds the policy."""
    pytest.importorskip("lerobot")
    from lerobot.envs.utils import preprocess_observation
    from lerobot.processor.env_processor import LiberoProcessorStep

    raw = OscEnv()._observation()
    rng = np.random.default_rng(0)
    raw["agentview_image"] = rng.integers(0, 256, (6, 8, 3), dtype=np.uint8)
    raw["robot0_eye_in_hand_image"] = rng.integers(0, 256, (6, 8, 3), dtype=np.uint8)
    raw["robot0_eef_quat"] = Rotation.from_rotvec([0.4, -1.1, 2.0]).as_quat()
    processed = LiberoProcessorStep()._process_observation(
        preprocess_observation(observation.lerobot_env_observation(raw))
    )
    np.testing.assert_allclose(
        processed["observation.state"][0].numpy(), observation.student_state(raw), atol=1e-6
    )
    image = (processed["observation.images.image"][0].numpy().transpose(1, 2, 0) * 255).round()
    np.testing.assert_array_equal(image.astype(np.uint8), observation.student_images(raw)["image"])


class ScriptedPolicy:
    def __init__(self) -> None:
        self.resets = 0
        self.calls: list[str] = []

    def reset(self) -> None:
        self.resets += 1

    def act(self, raw, instruction: str) -> np.ndarray:
        self.calls.append(instruction)
        return np.array([1.0, 0, 0, 0, 0, 0, -1.0])


def test_run_episode_stops_at_success_and_records_the_trajectory(tmp_path) -> None:
    env = OscEnv(success_after=4)
    adapter, _ = _adapter(tmp_path, env)
    adapter.get_observation = lambda: None
    policy = ScriptedPolicy()
    seeds: list[int] = []
    spec = EpisodeSpec(
        "libero_object_swap",
        3,
        scene=2,
        policy_seed=77,
        max_steps=10,
        instruction="pick up the soup",
    )
    summary = run_episode(
        adapter,
        policy,
        spec,
        recorder=EpisodeRecorder(tmp_path / "episode", video="none"),
        seed_fn=seeds.append,
    )

    assert summary["success"] is True and summary["steps"] == 4 and summary["ended"] == "success"
    assert summary["init_state_index"] == 1 and seeds == [77] and policy.resets == 1
    stored = json.loads((tmp_path / "episode" / "episode.json").read_text())
    assert stored["steps"] == 4
    arrays = np.load(tmp_path / "episode" / "steps.npz")
    assert arrays["action"].shape == (4, 7) and arrays["state"].shape == (4, 8)
    assert arrays["joint_pos"].shape == (4, 7) and arrays["final_eef_pos"].shape == (3,)
    np.testing.assert_allclose(arrays["eef_pos"][:, 0], [0.0, 0.01, 0.02, 0.03])
    assert not (tmp_path / "episode" / "video.mp4").exists()


def test_run_episode_stops_at_the_step_limit(tmp_path) -> None:
    adapter, _ = _adapter(tmp_path, OscEnv(success_after=None))
    adapter.get_observation = lambda: None
    spec = EpisodeSpec(
        "libero_object_swap", 3, scene=1, policy_seed=0, max_steps=6, instruction="x"
    )
    summary = run_episode(adapter, ScriptedPolicy(), spec)
    assert summary == summary | {"success": False, "steps": 6, "ended": "step_limit"}


def test_summarize_averages_tasks_then_reports_spread_across_seeds() -> None:
    episodes = []
    for seed, successes in ((0, (1, 1, 0, 0)), (1, (1, 0, 0, 0))):
        for task_id in (0, 1):
            for scene, ok in enumerate(successes, start=1):
                episodes.append(
                    {"suite": "s", "task_id": task_id, "policy_seed_index": seed, "scene": scene,
                     "success": bool(ok) if task_id == 0 else True, "steps": 10}
                )  # fmt: skip
    summary = benchmark.summarize(episodes)
    task0, task1 = summary["tasks"]
    assert task0["successes"] == 3 and task0["episodes"] == 8
    assert task0["per_seed"] == {"0": 0.5, "1": 0.25}
    assert task1["success_rate"] == 1.0
    assert summary["per_seed_mean"] == {"0": 0.75, "1": 0.625}
    assert summary["mean"] == pytest.approx((3 / 8 + 1.0) / 2)
    assert summary["std_across_seeds"] == pytest.approx(np.std([0.75, 0.625], ddof=1))
    table = benchmark.format_table(summary, "t")
    assert "| s:0 | 3/8 (37.5%)" in table and "**mean**" in table


def test_policy_seeds_are_distinct_per_episode() -> None:
    seeds = {
        benchmark.policy_seed_for(s, t, k)
        for s in range(3)
        for t in range(10)
        for k in range(1, 51)
    }
    assert len(seeds) == 3 * 10 * 50
