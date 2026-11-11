"""Check that ASPIRE's adapter and LeRobot's LIBERO env are the same simulator for the student.

    python -m cap_harness.vla.parity --suite libero_object --task 0 --scene 1
    python -m cap_harness.vla.parity --suite libero_object_swap --task 0 --scene 1 --policy

1. Environment: build LeRobot's ``LiberoEnv`` and ``LiberoAdapter`` (OSC_POSE) on the same task
   and saved scene, reset both, then send both the same action sequence.  After the reset and
   after every step the camera images, the robot state and the full MuJoCo state must match.
2. Policy (``--policy``): run one SmolVLA episode with LeRobot's own ``rollout`` on its env, and
   one with ``run_episode`` on the adapter, from the same noise seed.  The actions, the step
   count and the outcome must match.

Exit status 0 means every check passed; the report is printed as JSON.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from cap_harness.vla.libero_compat import ensure_libero_alias
from cap_harness.vla.observation import CAMERA_KEYS
from cap_harness.vla.protocol import (
    CONTROLLER,
    IMAGE_SIZE,
    SETTLE_STEPS,
    max_steps_for,
    task_language,
)

STATE_TOLERANCE = 1e-9
ACTION_TOLERANCE = 1e-5


def scripted_actions(steps: int, seed: int = 0) -> np.ndarray:
    """Smooth random hand motion, gripper open for the first half and closed for the second."""
    rng = np.random.default_rng(seed)
    actions = np.zeros((steps, 7))
    motion = np.zeros(6)
    for index in range(steps):
        motion = np.clip(0.85 * motion + 0.25 * rng.standard_normal(6), -1.0, 1.0)
        actions[index, :6] = 0.5 * motion
        actions[index, 6] = -1.0 if index < steps // 2 else 1.0
    return actions


def lerobot_env(suite: str, task_id: int, scene: int, image_size: int, episode_length: int):
    ensure_libero_alias()
    from lerobot.envs.libero import LiberoEnv, _get_suite

    return LiberoEnv(
        task_suite=_get_suite(suite),
        task_id=task_id,
        task_suite_name=suite,
        obs_type="pixels_agent_pos",
        observation_width=image_size,
        observation_height=image_size,
        init_states=True,
        episode_index=scene - 1,
        episode_length=episode_length,
        num_steps_wait=SETTLE_STEPS,
    )


def aspire_adapter(image_size: int):
    from cap_harness.libero.adapter import LiberoAdapter

    return LiberoAdapter(
        controller=CONTROLLER,
        camera_height=image_size,
        camera_width=image_size,
        init_mode="saved",
        post_settle_steps=SETTLE_STEPS,
    )


def _sim_state(env: Any) -> np.ndarray:
    sim = getattr(env, "sim", None) or env.env.sim
    return np.asarray(sim.get_state().flatten(), dtype=np.float64)


def compare(
    lerobot_obs: Mapping[str, Any],
    raw: Mapping[str, Any],
    lerobot_state: np.ndarray,
    aspire_state: np.ndarray,
) -> dict[str, float]:
    """Largest differences between the two observations (0 everywhere means identical)."""
    differences = {}
    for name, key in CAMERA_KEYS.items():
        a = np.asarray(lerobot_obs["pixels"][key], dtype=np.int16)
        b = np.asarray(raw[name], dtype=np.int16)
        differences[f"image:{key}"] = (
            float(np.abs(a - b).max()) if a.shape == b.shape else float("inf")
        )
    robot = lerobot_obs["robot_state"]
    pairs = {
        "eef_pos": (robot["eef"]["pos"], raw["robot0_eef_pos"]),
        "eef_quat": (robot["eef"]["quat"], raw["robot0_eef_quat"]),
        "gripper_qpos": (robot["gripper"]["qpos"], raw["robot0_gripper_qpos"]),
        "joint_pos": (robot["joints"]["pos"], raw["robot0_joint_pos"]),
    }
    for name, (a, b) in pairs.items():
        differences[name] = float(
            np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)).max()
        )
    differences["mujoco_state"] = (
        float(np.abs(lerobot_state - aspire_state).max())
        if lerobot_state.shape == aspire_state.shape
        else float("inf")
    )
    return differences


def _passes(differences: Mapping[str, float]) -> bool:
    return all(
        value == 0.0 if name.startswith("image:") else value <= STATE_TOLERANCE
        for name, value in differences.items()
    )


def env_parity(suite: str, task_id: int, scene: int, steps: int, image_size: int) -> dict[str, Any]:
    episode_length = max_steps_for(suite)
    reference = lerobot_env(suite, task_id, scene, image_size, episode_length)
    adapter = aspire_adapter(image_size)
    try:
        lerobot_obs, _ = reference.reset(seed=scene)
        adapter.reset(f"{suite}:{task_id}", seed=scene)
        worst: dict[str, float] = compare(
            lerobot_obs,
            adapter.raw_observation,
            _sim_state(reference._env),
            _sim_state(adapter.native_env),
        )
        report: dict[str, Any] = {
            "after_reset": dict(worst),
            "first_mismatch_step": None if _passes(worst) else 0,
        }
        for index, action in enumerate(scripted_actions(steps), start=1):
            lerobot_obs, _, terminated, _, info = reference.step(action.astype(np.float32))
            raw = adapter.step_native_action(action.astype(np.float32))
            if terminated:
                # LeRobot's env resets itself on success; the comparison ends there.
                report["ended_early_at_step"] = index
                break
            differences = compare(
                lerobot_obs, raw, _sim_state(reference._env), _sim_state(adapter.native_env)
            )
            if report["first_mismatch_step"] is None and not _passes(differences):
                report["first_mismatch_step"] = index
                report["first_mismatch"] = differences
            worst = {name: max(worst[name], value) for name, value in differences.items()}
        report["steps"] = steps
        report["worst"] = worst
        report["passed"] = report["first_mismatch_step"] is None
        return report
    finally:
        reference.close()
        adapter.close()


class _Recording:
    """Wraps the student policy and keeps every action it returns."""

    def __init__(self, policy: Any) -> None:
        self.policy = policy
        self.actions: list[np.ndarray] = []

    def reset(self) -> None:
        self.policy.reset()

    def act(self, raw: Mapping[str, Any], instruction: str) -> np.ndarray:
        action = self.policy.act(raw, instruction)
        self.actions.append(np.asarray(action, dtype=np.float64))
        return action


def policy_parity(
    suite: str,
    task_id: int,
    scene: int,
    image_size: int,
    *,
    source: str | None,
    device: str | None,
    seed: int,
) -> dict[str, Any]:
    import gymnasium as gym
    from lerobot.scripts.lerobot_eval import rollout

    from cap_harness.vla.policy import (
        DEFAULT_POLICY,
        DEFAULT_REVISION,
        LeRobotLiberoPolicy,
        seed_everything,
    )
    from cap_harness.vla.rollout import EpisodeSpec, run_episode

    source = source or DEFAULT_POLICY
    student = LeRobotLiberoPolicy(
        source,
        revision=DEFAULT_REVISION if source == DEFAULT_POLICY else None,
        device=device,
        image_size=image_size,
        suite=suite,
    )
    episode_length = max_steps_for(suite)
    reference_env = gym.vector.SyncVectorEnv(
        [lambda: lerobot_env(suite, task_id, scene, image_size, episode_length)]
    )
    try:
        student.reset()
        seed_everything(seed)
        data = rollout(
            env=reference_env,
            policy=student.policy,
            env_preprocessor=student.env_preprocessor,
            env_postprocessor=student.env_postprocessor,
            preprocessor=student.preprocessor,
            postprocessor=student.postprocessor,
            seeds=[scene],
        )
    finally:
        reference_env.close()
    done = data["done"][0].numpy()
    reference_steps = int(np.argmax(done)) + 1 if done.any() else int(done.shape[0])
    reference_actions = data["action"][0].numpy()[:reference_steps].astype(np.float64)
    reference_success = bool(data["success"][0].numpy()[:reference_steps].any())

    recording = _Recording(student)
    adapter = aspire_adapter(image_size)
    try:
        summary = run_episode(
            adapter,
            recording,
            EpisodeSpec(suite, task_id, scene, seed, episode_length, task_language(suite, task_id)),
            seed_fn=seed_everything,
        )
    finally:
        adapter.close()
    ours = np.stack(recording.actions) if recording.actions else np.zeros((0, 7))
    common = min(len(ours), len(reference_actions))
    difference = float(np.abs(ours[:common] - reference_actions[:common]).max()) if common else 0.0
    passed = (
        common == len(ours) == len(reference_actions)
        and difference <= ACTION_TOLERANCE
        and summary["success"] == reference_success
    )
    return {
        "lerobot": {"steps": reference_steps, "success": reference_success},
        "aspire": {"steps": summary["steps"], "success": summary["success"]},
        "max_action_difference": difference,
        "policy": student.info.to_dict(),
        "passed": passed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--scene", type=int, default=1)
    parser.add_argument(
        "--steps", type=int, default=120, help="scripted steps for the environment check"
    )
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument(
        "--policy",
        nargs="?",
        const="",
        default=None,
        help="also compare a policy episode (default checkpoint when no value)",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None, help="also write the report here")
    args = parser.parse_args(argv)

    report: dict[str, Any] = {
        "suite": args.suite,
        "task": args.task,
        "scene": args.scene,
        "image_size": args.image_size,
        "environment": env_parity(args.suite, args.task, args.scene, args.steps, args.image_size),
    }
    if args.policy is not None:
        report["policy"] = policy_parity(
            args.suite, args.task, args.scene, args.image_size,
            source=args.policy or None, device=args.device, seed=args.seed,
        )  # fmt: skip
    report["passed"] = all(
        part["passed"] for part in (report["environment"], report.get("policy", {"passed": True}))
    )
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
