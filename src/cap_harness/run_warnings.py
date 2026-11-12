"""Plain-language warnings read from a finished run's recorded calls.

A program can finish, or fail, for a reason that is visible in the calls it made but in nothing
a debugger reads first: SAM3 did not recognise the prompt and the best mask was a guess, two
objects matched almost equally, the gripper closed on nothing, a model service was down. This
module reads `trace/calls/*` of one run and says so, so `run-batch` can put it next to each
seed's result. It reads recorded files only and changes nothing in the run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

#: Read at most this many call folders of one run (a run has tens to a few hundred).
MAX_CALLS = 2000


def _read(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _tool(folder: Path) -> str:
    """`call-000004-localize-object` -> `localize-object`."""
    parts = folder.name.split("-", 2)
    return parts[2] if len(parts) == 3 else folder.name


def _query(inputs: Any) -> str:
    args = (inputs or {}).get("args") if isinstance(inputs, dict) else None
    if isinstance(args, list) and args and isinstance(args[0], str):
        return args[0]
    return "?"


def _localization(query: str, diagnostics: dict[str, Any]) -> str | None:
    score = diagnostics.get("segmentation_score")
    runner_up = diagnostics.get("runner_up_score")
    if not isinstance(score, (int, float)):
        return None
    if diagnostics.get("low_confidence"):
        return (
            f'localize_object("{query}"): best mask scores {score:.2f}. SAM3 does not recognise '
            "this prompt, so the object it returned is a guess and may be a different object. "
            "Look at media/overlays/sam3-*.png and describe the object by what is visible "
            "(colour, shape, material)."
        )
    if diagnostics.get("ambiguous") and isinstance(runner_up, (int, float)):
        return (
            f'localize_object("{query}"): two objects match almost equally ({score:.2f} and '
            f"{runner_up:.2f}). The program took the first; check media/overlays/sam3-*.png "
            "that it is the right one, and tell them apart in the program if it is not."
        )
    return None


def run_warnings(run_dir: Path | str) -> list[str]:
    """Warnings for one run directory, in the order the calls happened, without repeats."""
    calls = Path(run_dir) / "trace" / "calls"
    if not calls.is_dir():
        return []
    notes: list[str] = []
    counts: dict[str, int] = {}

    def add(text: str) -> None:
        if text in counts:
            counts[text] += 1
        else:
            counts[text] = 1
            notes.append(text)

    services_down: dict[str, int] = {}
    localized: set[tuple[str, bool]] = set()  # one note per prompt and kind, not per retry
    for folder in sorted(calls.iterdir())[:MAX_CALLS]:
        if not folder.is_dir():
            continue
        tool = _tool(folder)
        output = _read(folder / "output.json")
        if not isinstance(output, dict):
            continue
        error = output.get("error")
        if isinstance(error, dict) and error.get("code") == "provider_unavailable":
            services_down[tool] = services_down.get(tool, 0) + 1
            continue
        details = error.get("details") if isinstance(error, dict) else None
        if tool == "move-to-pose" and isinstance(details, dict) and details.get("stalled"):
            short = details.get("joint_error_rad")
            add(
                "move_to_pose stopped short of its target with the arm no longer moving"
                + (f" ({short:.3f} rad away)" if isinstance(short, (int, float)) else "")
                + ": something is in the way, usually the gripper or the held object touching "
                "the scene. Check the target position against the keyframe after the move."
            )
            continue
        diagnostics = output.get("diagnostics")
        if not isinstance(diagnostics, dict):
            continue
        if tool == "localize-object":
            query = _query(_read(folder / "input.json"))
            key = (query, bool(diagnostics.get("low_confidence")))
            text = _localization(query, diagnostics)
            if text and key not in localized:
                localized.add(key)
                add(text)
        elif tool == "close-gripper" and diagnostics.get("closed_empty"):
            add(
                "close_gripper closed on nothing: the grasp missed, so the moves after it "
                "carried nothing."
            )
    if services_down:
        listed = ", ".join(f"{tool} x{count}" for tool, count in sorted(services_down.items()))
        notes.insert(
            0,
            f"a model service did not answer ({listed}). This is not a program fault: check "
            "the services and the GPU's free memory, then run this seed again.",
        )
    return [text if counts.get(text, 1) == 1 else f"{text} (x{counts[text]})" for text in notes]


def service_failure(run_dir: Path | str) -> bool:
    """True when a call of this run failed because a model service did not answer."""
    return any(text.startswith("a model service did not answer") for text in run_warnings(run_dir))
