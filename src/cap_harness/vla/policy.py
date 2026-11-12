"""A LeRobot policy acting on LIBERO, with LeRobot's own pre- and post-processing.

The processing chain is the one ``lerobot-eval`` builds:

    preprocess_observation -> task -> LiberoProcessorStep (rotate images, 8-value state)
    -> policy preprocessor (rename cameras, tokenize, normalize, device)
    -> select_action -> policy postprocessor (unnormalize) -> env postprocessor

so a checkpoint sees exactly what it would see under lerobot-eval.  Hub checkpoints are
resolved to a local snapshot at a pinned revision first, so every part of a run (weights,
config, normalization statistics) comes from the same commit.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
import random
from typing import Any

import numpy as np

from cap_harness.vla import students
from cap_harness.vla.libero_compat import ensure_runtime_env
from cap_harness.vla.observation import lerobot_env_observation
from cap_harness.vla.protocol import IMAGE_SIZE

#: lerobot/smolvla_libero names its cameras camera1/camera2; LIBERO's are image/image2.
CAMERA_RENAME: dict[str, str] = dict(students.SMOLVLA.camera_rename)
#: The SmolVLA LIBERO checkpoint and the revision LeRobot's documentation pins (the default
#: student; ``cap_harness.vla.students`` has the others).
DEFAULT_POLICY = students.SMOLVLA.checkpoint
DEFAULT_REVISION = students.SMOLVLA.revision


@dataclass(frozen=True)
class PolicyInfo:
    source: str
    revision: str | None
    local_path: str
    policy_type: str
    device: str
    n_action_steps: int | None
    chunk_size: int | None
    rename_map: dict[str, str]
    student: str = students.DEFAULT_STUDENT
    #: Policy config values this evaluation set (``Student.eval_config``).
    config_overrides: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def resolve_checkpoint(source: str, revision: str | None) -> str:
    """A local directory for ``source`` (a path, or a Hub repo id at ``revision``)."""
    path = Path(source).expanduser()
    if path.is_dir():
        return str(path.resolve())
    from huggingface_hub import snapshot_download

    return snapshot_download(source, revision=revision)


def rename_map_for(
    input_features: Mapping[str, Any], camera_rename: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Map LIBERO's camera names to the checkpoint's, when the checkpoint renamed them."""
    return {
        source: target
        for source, target in (CAMERA_RENAME if camera_rename is None else camera_rename).items()
        if source not in input_features and target in input_features
    }


def student_for(source: str, student: str | None) -> students.Student:
    """The student a policy belongs to: the one named, else the one its checkpoint points to,
    else the default. Naming a student its checkpoint contradicts is an error.
    """
    detected = students.for_source(source)
    if student is None:
        return detected or students.get(None)
    chosen = students.get(student)
    if detected is not None and detected.key != chosen.key:
        raise ValueError(f"--student {chosen.key}, but {source} is a {detected.label} checkpoint")
    return chosen


def apply_overrides(config: Any, overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Set the policy config values a student's evaluation needs; return those applied."""
    applied = {}
    for key, value in overrides.items():
        if hasattr(config, key) and getattr(config, key) != value:
            setattr(config, key, value)
            applied[key] = value
    return applied


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LeRobotLiberoPolicy:
    """Load once, then ``reset()`` per episode and ``act(raw_observation, instruction)`` per step."""

    def __init__(
        self,
        source: str = DEFAULT_POLICY,
        *,
        revision: str | None = DEFAULT_REVISION,
        device: str | None = None,
        n_action_steps: int | None = None,
        image_size: int = IMAGE_SIZE,
        suite: str = "libero_object",
        student: str | None = None,
    ) -> None:
        ensure_runtime_env()
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
        from lerobot.envs.factory import make_env_pre_post_processors
        from lerobot.policies.factory import make_policy, make_pre_post_processors
        import torch

        chosen = student_for(source, student)
        local_path = resolve_checkpoint(source, revision if not Path(source).is_dir() else None)
        config = PreTrainedConfig.from_pretrained(local_path)
        if str(config.type) != chosen.policy_type and students.for_policy_type(config.type):
            raise ValueError(
                f"{source} is a {config.type} policy, not {chosen.label} ({chosen.policy_type})"
            )
        config.pretrained_path = local_path
        config.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if n_action_steps is not None:
            config.n_action_steps = int(n_action_steps)
        overrides = apply_overrides(config, chosen.eval_config)
        rename = rename_map_for(config.input_features or {}, chosen.camera_rename)
        # Only the environment's feature list is read from this config (to name the action);
        # the suite chosen here does not affect the policy.
        env_config = LiberoEnvConfig(
            task=suite, observation_height=image_size, observation_width=image_size
        )
        # lerobot-eval's numeric settings.
        if config.device == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
        self.policy = make_policy(config, env_cfg=env_config, rename_map=rename)
        self.policy.eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=config,
            pretrained_path=local_path,
            preprocessor_overrides={
                "device_processor": {"device": str(config.device)},
                "rename_observations_processor": {"rename_map": rename},
            },
        )
        self.env_preprocessor, self.env_postprocessor = make_env_pre_post_processors(
            env_cfg=env_config, policy_cfg=config
        )
        self.info = PolicyInfo(
            source=source,
            revision=revision,
            local_path=local_path,
            policy_type=str(config.type),
            device=str(config.device),
            n_action_steps=getattr(config, "n_action_steps", None),
            chunk_size=getattr(config, "chunk_size", None),
            rename_map=rename,
            student=chosen.key,
            config_overrides=overrides,
        )
        self._torch = torch

    def reset(self) -> None:
        """Drop any queued action chunk; call at the start of every episode."""
        self.policy.reset()

    def act(self, raw: Mapping[str, Any], instruction: str) -> np.ndarray:
        """The 7 OSC_POSE action values for the adapter's raw observation."""
        from lerobot.envs.utils import preprocess_observation
        from lerobot.utils.constants import ACTION

        observation = preprocess_observation(lerobot_env_observation(raw))
        observation["task"] = [instruction]
        observation = self.env_preprocessor(observation)
        observation = self.preprocessor(observation)
        with self._torch.inference_mode():
            action = self.policy.select_action(observation)
        action = self.postprocessor(action)
        action = self.env_postprocessor({ACTION: action})[ACTION]
        values = action.to("cpu").numpy()
        if values.ndim != 2 or values.shape[0] != 1:
            raise ValueError(f"expected one action, got shape {values.shape}")
        return values[0].astype(np.float64)
