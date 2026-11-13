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
from cap_harness.vla import students

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
#: Actions executed from each predicted chunk before the policy looks again.  Measured with
#: lerobot/smolvla_libero on LIBERO-Object (10 tasks x 10 scenes, one seed, on cali):
#: 50 (the checkpoint's stored value) 78%, 10 93% (95% interval 86-97%), 1 83%; the SmolVLA
#: paper reports 96%.  Fifty steps (2.5 s) run blind cannot correct a grasp; re-planning every
#: step draws fresh flow-matching noise each step and the hand switches between plans.
N_ACTION_STEPS = 10
CONTROLLER = "OSC_POSE"

#: SmolVLA (0.45B) success on standard LIBERO, % (SmolVLA paper, arXiv 2506.01844, Table 2).
SMOLVLA_PAPER: dict[str, float] = dict(students.SMOLVLA.standard_reference)
#: pi0.5 success per LIBERO-Pro perturbation, % (LIBERO-PRO, arXiv 2510.03827, Tables 2-5).
#: No source reports SmolVLA on LIBERO-Pro.
PI05_LIBERO_PRO: dict[str, dict[str, float]] = {
    #               original  _object  _swap  _lan  _task  _env
    "libero_goal": dict(
        zip(("", "_object", "_swap", "_lan", "_task", "_env"), (97, 97, 38, 97, 0, 46))
    ),
    "libero_spatial": dict(
        zip(("", "_object", "_swap", "_lan", "_task", "_env"), (98, 97, 20, 97, 1, 46))
    ),
    "libero_10": dict(
        zip(("", "_object", "_swap", "_lan", "_task", "_env"), (93, 92, 8, 93, 1, 46))
    ),
    "libero_object": dict(
        zip(("", "_object", "_swap", "_lan", "_task", "_env"), (98, 98, 17, 96, 1, 73))
    ),
}
#: Warn when a run with a published SmolVLA number trails it by more than this many points.
WARN_BELOW_REFERENCE = 25.0
WARN_AFTER_EPISODES = 20

__all__ = [
    "CONTROLLER",
    "DEFAULT_MAX_STEPS",
    "IMAGE_SIZE",
    "N_ACTION_STEPS",
    "OSC_SETTLE_ACTION",
    "PI05_LIBERO_PRO",
    "SETTLE_STEPS",
    "SMOLVLA_PAPER",
    "SUITE_MAX_STEPS",
    "max_steps_for",
    "parse_int_list",
    "references",
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


def references(suite: str, policy: str, student: str | None = None) -> dict[str, object]:
    """Published numbers to read a run against: the expected value and context.

    ``expected`` is set only for a student's own published checkpoint on a standard suite (a
    fine-tuned checkpoint has no published number). SmolVLA's answers are unchanged from
    before pi0.5 support; ``student`` defaults to the one the policy name points to.
    """
    chosen = students.get(student) if student else students.for_source(policy)
    if chosen is None or chosen.key == students.SMOLVLA.key:
        return _smolvla_references(suite, policy)
    family = suite_family(suite)
    out: dict[str, object] = {"expected": None, "notes": []}
    notes: list[str] = out["notes"]  # type: ignore[assignment]
    own = policy == chosen.checkpoint or chosen.checkpoint.replace("/", "--") in policy
    if family and suite == family and family in chosen.standard_reference:
        value = chosen.standard_reference[family]
        if own:
            out["expected"] = value
        notes.append(f"{chosen.label} ({chosen.reference_source}): {value:.0f}%")
    if family in PI05_LIBERO_PRO:
        perturbation = suite[len(family) :]
        if perturbation in PI05_LIBERO_PRO[family]:
            notes.append(f"pi0.5 (LIBERO-PRO paper): {PI05_LIBERO_PRO[family][perturbation]:.0f}%")
        if perturbation:
            notes.append(f"no published number for {chosen.checkpoint} on LIBERO-Pro")
    return out


def _smolvla_references(suite: str, policy: str) -> dict[str, object]:
    family = suite_family(suite)
    out: dict[str, object] = {"expected": None, "notes": []}
    notes: list[str] = out["notes"]  # type: ignore[assignment]
    smolvla = "smolvla" in policy.lower()
    if family and suite == family and family in SMOLVLA_PAPER:
        if smolvla:
            out["expected"] = SMOLVLA_PAPER[family]
        notes.append(f"SmolVLA paper: {SMOLVLA_PAPER[family]:.0f}%")
    if family in PI05_LIBERO_PRO:
        perturbation = suite[len(family) :]
        if perturbation in PI05_LIBERO_PRO[family]:
            notes.append(f"pi0.5 (LIBERO-PRO paper): {PI05_LIBERO_PRO[family][perturbation]:.0f}%")
        if perturbation and smolvla:
            notes.append("no published SmolVLA number on LIBERO-Pro")
    return out


def _benchmark(suite: str):
    from cap_harness.vla.libero_compat import ensure_libero_alias

    ensure_libero_alias()
    from libero.libero import benchmark

    suites = benchmark.get_benchmark_dict()
    if suite not in suites:
        from cap_harness.libero import layouts

        layouts.register(suites)  # the generated sibling layouts
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
