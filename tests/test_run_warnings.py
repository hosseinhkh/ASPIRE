"""Warnings read from a run's recorded calls, and how run-batch reports them."""

from __future__ import annotations

import json
from pathlib import Path

from cap_harness.batch import SeedResult, summarize
from cap_harness.run_warnings import run_warnings, service_failure


def _call(run: Path, index: int, tool: str, output: dict, args: list | None = None) -> None:
    folder = run / "trace" / "calls" / f"call-{index:06d}-{tool}"
    folder.mkdir(parents=True)
    (folder / "input.json").write_text(json.dumps({"args": args or [], "kwargs": {}}))
    (folder / "output.json").write_text(json.dumps(output))


def _localized(score: float, runner_up: float) -> dict:
    return {
        "ok": True,
        "diagnostics": {
            "segmentation_score": score,
            "runner_up_score": runner_up,
            "low_confidence": score < 0.3,
            "ambiguous": score - runner_up < 0.1,
        },
    }


def test_a_guessed_object_is_reported_once_per_prompt(tmp_path) -> None:
    """The alphabet-soup session: SAM3 scored every object near 0.1 and the milk was used."""
    _call(tmp_path, 4, "localize-object", _localized(0.11, 0.09), ["alphabet soup"])
    _call(tmp_path, 10, "localize-object", _localized(0.91, 0.12), ["basket"])
    _call(tmp_path, 36, "localize-object", _localized(0.14, 0.09), ["alphabet soup"])
    notes = run_warnings(tmp_path)
    assert len(notes) == 1
    assert 'localize_object("alphabet soup")' in notes[0] and "0.11" in notes[0]
    assert "is a guess" in notes[0] and "media/overlays" in notes[0]


def test_two_similar_objects_are_reported_as_ambiguous(tmp_path) -> None:
    _call(tmp_path, 4, "localize-object", _localized(0.82, 0.78), ["soup can"])
    (note,) = run_warnings(tmp_path)
    assert "two objects match almost equally (0.82 and 0.78)" in note


def test_an_empty_grasp_and_a_dead_service_are_reported(tmp_path) -> None:
    down = {"ok": False, "error": {"code": "provider_unavailable", "message": "CUDA out of memory"}}
    _call(tmp_path, 5, "segment-text", down)
    _call(tmp_path, 6, "sam3-segment-text", down)
    _call(tmp_path, 30, "close-gripper", {"ok": True, "diagnostics": {"closed_empty": True}})
    _call(tmp_path, 31, "close-gripper", {"ok": True, "diagnostics": {"closed_empty": True}})
    notes = run_warnings(tmp_path)
    assert notes[0].startswith("a model service did not answer")
    assert "not a program fault" in notes[0]
    assert notes[1].startswith("close_gripper closed on nothing") and notes[1].endswith("(x2)")
    assert service_failure(tmp_path)


def test_a_clean_run_and_a_run_without_calls_have_no_warnings(tmp_path) -> None:
    assert run_warnings(tmp_path) == []
    _call(tmp_path, 4, "localize-object", _localized(0.91, 0.12), ["basket"])
    _call(tmp_path, 30, "close-gripper", {"ok": True, "diagnostics": {"closed_empty": False}})
    assert run_warnings(tmp_path) == [] and not service_failure(tmp_path)


def test_the_batch_summary_groups_warnings_by_seed() -> None:
    guessed = 'localize_object("alphabet soup"): best mask scores 0.11.'
    results = [
        SeedResult(
            51, True, "a", True, False, "program_completed", 1.0, 1, 0, 1.0, None, (guessed,)
        ),
        SeedResult(
            52, True, "b", True, False, "program_completed", 1.0, 1, 1, 1.0, None, (guessed,)
        ),
        SeedResult(53, True, "c", True, True, "task_succeeded", 1.0, 1, 2, 1.0, None),
    ]
    summary = summarize(results)
    assert summary["warnings"] == [{"text": guessed, "seeds": [51, 52]}]
    assert summary["results"][2]["warnings"] == ()


def test_a_move_blocked_by_contact_is_reported(tmp_path) -> None:
    """A held can touching the basket wall: the arm stalls short of the release pose."""
    blocked = {
        "ok": False,
        "error": {
            "code": "timeout",
            "message": "final trajectory waypoint did not converge",
            "details": {"stalled": True, "joint_error_rad": 0.0188},
        },
    }
    _call(tmp_path, 37, "move-to-pose", blocked)
    (note,) = run_warnings(tmp_path)
    assert "stopped short of its target" in note and "0.019 rad" in note
    assert "something is in the way" in note
