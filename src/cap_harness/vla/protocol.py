"""The evaluation protocol, matched to LeRobot's LIBERO evaluation (lerobot v0.4.4).

* Controller: OSC_POSE with LIBERO's default configuration, 20 Hz.
* Reset: the task's saved init state for the scene, then 10 still steps with the gripper open
  (``OSC_SETTLE_ACTION``) that are not counted in the episode.
* Episode limit: by suite family, as ``TASK_SUITE_MAX_STEPS`` in ``lerobot/envs/libero.py``.
  LIBERO-Pro suites inherit the limit of the suite they perturb (libero_object_swap: 280).
* Success: the task's own predicate (``check_success``) after every step; the episode ends at
  the first success.
* Instruction: the benchmark task's ``language`` (what lerobot-eval passes as the task).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from cap_harness.libero.adapter import OSC_SETTLE_ACTION

#: lerobot/envs/libero.py TASK_SUITE_MAX_STEPS (v0.4.4), keyed by suite family.
SUITE_MAX_STEPS: dict[str, int] = {
    "libero_spatial": 280,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
DEFAULT_MAX_STEPS = 500
SETTLE_STEPS = 10
IMAGE_SIZE = 256
CONTROLLER = "OSC_POSE"

__all__ = [
    "CONTROLLER",
    "DEFAULT_MAX_STEPS",
    "IMAGE_SIZE",
    "OSC_SETTLE_ACTION",
    "SETTLE_STEPS",
    "SUITE_MAX_STEPS",
    "max_steps_for",
    "parse_int_list",
    "suite_family",
    "task_count",
    "task_language",
]


def suite_family(suite: str) -> str | None:
    """The base LIBERO suite a (possibly LIBERO-Pro) suite belongs to."""
    matches = [
        family for family in SUITE_MAX_STEPS if suite == family or suite.startswith(family + "_")
    ]
    return max(matches, key=len) if matches else None


def max_steps_for(suite: str) -> int:
    family = suite_family(suite)
    return SUITE_MAX_STEPS[family] if family else DEFAULT_MAX_STEPS


def _benchmark(suite: str):
    from cap_harness.vla.libero_compat import ensure_libero_alias

    ensure_libero_alias()
    from libero.libero import benchmark

    suites = benchmark.get_benchmark_dict()
    if suite not in suites:
        raise ValueError(f"unknown LIBERO suite {suite!r}")
    return suites[suite]()


def task_language(suite: str, task_id: int) -> str:
    """The instruction lerobot-eval gives the policy for this task."""
    return str(_benchmark(suite).get_task(task_id).language)


def task_count(suite: str) -> int:
    return int(_benchmark(suite).n_tasks)


def parse_int_list(text: str | Iterable[int], *, default: Sequence[int] | None = None) -> list[int]:
    """Parse "0,3,5-9" (inclusive ranges) or "all" (returns ``default``)."""
    if not isinstance(text, str):
        return [int(value) for value in text]
    text = text.strip()
    if text in ("", "all"):
        if default is None:
            raise ValueError("a list is required here")
        return list(default)
    values: list[int] = []
    for raw_part in text.split(","):
        part = raw_part.strip()
        if not part:
            continue
        low, separator, high = part.partition("-")
        if separator:
            if int(high) < int(low):
                raise ValueError(f"empty range {part!r}")
            values.extend(range(int(low), int(high) + 1))
        else:
            values.append(int(part))
    return values
