"""One program run on several layouts (scripts/gpu/layouts_batch.py), and the converter's slower
retry. The simulator runs are checked on the machine (docs/trace2vla-full-run.md).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from cap_harness.trace2vla import convert, replay

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "gpu" / "layouts_batch.py"
spec = importlib.util.spec_from_file_location("layouts_batch", SCRIPT)
layouts_batch = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(layouts_batch)


def batch(passes: int, seeds: int, warnings=(), errors=()) -> dict:
    return {
        "passes": passes, "seeds": seeds, "runs": seeds, "pass_rate": passes / seeds,
        "elapsed_s": 10.0, "warnings": [{"seeds": [51], "text": w} for w in warnings],
        "program_errors": list(errors), "missing": [],
        "results": [{"seed": 51, "task_success": passes > 0}],
    }  # fmt: skip


def test_the_layouts_add_up_in_one_summary_that_names_each(tmp_path) -> None:
    combined = layouts_batch.combine(
        {"official": batch(2, 2), "alt1": batch(0, 2, ["grasp missed"], [51]), "alt2": None},
        tmp_path, ["51", "52"],
    )  # fmt: skip
    assert (combined["passes"], combined["seeds"], combined["runs"]) == (2, 4, 4)
    assert combined["pass_rate"] == 0.5
    assert combined["layouts"]["alt1"]["passes"] == 0
    assert "no batch.json" in combined["layouts"]["alt2"]["error"]
    assert combined["missing"] == ["alt2:51", "alt2:52"]
    assert combined["warnings"][0]["text"] == "[alt1] grasp missed"
    assert combined["program_errors"] == ["alt1:51"]
    assert {r["layout"] for r in combined["results"]} == {"official", "alt1"}


def test_it_runs_each_layout_and_writes_the_combined_file(tmp_path) -> None:
    """A stand-in cap-harness writes each layout's batch.json; the helper sums them."""
    fake = tmp_path / "fake-cap-harness"
    fake.write_text(
        "#!" + sys.executable + "\n"
        "import json, sys, pathlib\n"
        "args = sys.argv[1:]\n"
        "out = pathlib.Path(args[args.index('--output-root') + 1]); out.mkdir(parents=True)\n"
        "suite = args[args.index('--suite') + 1]\n"
        "seeds = args[args.index('--seeds') + 1: args.index('--init-mode')]\n"
        "ok = 0 if suite.endswith('alt1') else len(seeds)\n"
        "(out / 'batch.json').write_text(json.dumps({'passes': ok, 'seeds': len(seeds), 'runs': len(seeds), 'warnings': [], 'results': []}))\n"
    )
    fake.chmod(0o755)
    out = tmp_path / "dev_v01"
    code = layouts_batch.main([
        "--layouts", "libero_object_swap,libero_object_swap_alt1", "--task-id", "0",
        "--seeds", "51", "52", "--program", "p.py", "--output-root", str(out),
        "--cap-harness", str(fake),
    ])  # fmt: skip
    assert code == 0
    combined = json.loads((out / "batch.json").read_text())
    assert (combined["passes"], combined["seeds"]) == (2, 4)
    assert combined["layouts"]["libero_object_swap_alt1"]["passes"] == 0
    assert (out / "libero_object_swap" / "batch.json").is_file()


# ----------------------------------------------------------------------- converter speeds --
def test_slower_speeds_are_tried_only_when_asked_and_do_not_change_the_demo_set() -> None:
    assert convert.ConvertSettings().speeds() == [3.0]  # off by default
    asked = convert.ConvertSettings(fallback_speeds=(2.0, 4.0, 1.0))
    assert asked.speeds() == [3.0, 2.0, 1.0]  # slower ones only, in order
    old = convert.ConvertSettings().to_dict()
    del old["fallback_speeds"]  # a demo set made before the option existed
    assert convert.same_demo_set(old, asked.to_dict())
    other = convert.ConvertSettings(track=replay.TrackSettings(speed=2.0)).to_dict()
    assert not convert.same_demo_set(old, other)


def test_each_speed_tried_is_recorded_when_all_lose_the_object(monkeypatch, tmp_path) -> None:
    tried = []

    def track(adapter, episode, settings):
        tried.append(settings.speed)
        return replay.Rollout(success=False, steps=10, actions=np.zeros((10, 7)), stats={})

    monkeypatch.setattr(replay, "track_in_osc", track)
    monkeypatch.setattr(convert, "sha256_file", lambda path: "x")
    monkeypatch.setattr(convert, "task_language", lambda suite, task: "pick it up")

    class Episode:
        run_dir, scene, steps, success, suite, task_id = (
            tmp_path,
            {},
            100,
            True,
            "libero_object_swap",
            0,
        )

    settings = convert.ConvertSettings(fallback_speeds=(2.0,))
    summary = convert.convert_episode(
        Episode(), tmp_path / "out", fast=None, recorder=None, settings=settings
    )
    assert tried == [3.0, 2.0]
    assert summary["rejected"] == "re-execution_failed"
    assert [t["speed"] for t in summary["speed_tries"]] == [3.0, 2.0]
