"""Sibling layouts: LIBERO-Pro's swap with each task's other partners, and scene sets that use
them. The simulator check (positions in the scene) is in docs; these need no simulator.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cap_harness.libero import layouts
from cap_harness.trace2vla import cli, scenes, teacher
from cap_harness.trace2vla.store import Experiment, read_json

BDDL = layouts.LIBERO_PRO / "libero" / "libero" / "bddl_files"
pytestmark = pytest.mark.skipif(
    not (BDDL / "libero_object").is_dir(), reason="LIBERO-PRO is not checked out"
)


def test_the_rule_reproduces_every_official_swap_file() -> None:
    """Our swap of the original task with the official partner is LIBERO-Pro's own file."""
    for spec in layouts.plan().values():
        for task in spec:
            original = (BDDL / "libero_object" / f"{task.name}.bddl").read_text()
            official = (BDDL / "libero_object_swap" / f"{task.name}.bddl").read_text()
            assert layouts.swap(original, task.target, task.official_partner) == official


def test_four_new_partners_per_task_each_moving_exactly_two_objects() -> None:
    planned = layouts.plan()
    assert sorted(planned) == [f"libero_object_swap_alt{k}" for k in range(1, 5)]
    for task_id in range(10):
        partners = {planned[alt][task_id].partner for alt in planned}
        official = planned["libero_object_swap_alt1"][task_id].official_partner
        assert len(partners) == 4 and official not in partners
    for alt, tasks in planned.items():
        assert [t.task_id for t in tasks] == list(range(10))
        for task in tasks:
            original = layouts.init_regions(
                (BDDL / "libero_object" / f"{task.name}.bddl").read_text()
            )
            mine = layouts.init_regions((layouts.LAYOUTS_DIR / task.bddl).read_text())
            moved = {o for o in original if original[o] != mine[o]}
            assert moved == {task.target, task.partner}, (alt, task.name)
            assert mine[task.target] == original[task.partner]


def test_the_committed_files_are_current() -> None:
    assert layouts.main(["--check"]) == 0


def test_registered_suites_look_like_libero_suites() -> None:
    mapping: dict = {"libero_object_swap": object()}
    added = layouts.register(mapping)
    assert added == [f"libero_object_swap_alt{k}" for k in range(1, 5)]
    assert layouts.register(mapping) == []  # idempotent
    suite = mapping["libero_object_swap_alt2"]()
    assert suite.n_tasks == 10 and suite.seeded_only
    task = suite.get_task(1)
    # LIBERO's rule: the instruction comes from the task name, the same for every layout.
    assert task.language == "pick up the cream cheese and place it in the basket"
    assert Path(task.bddl_path).is_file() and suite.get_task_init_states(1) == []


# ------------------------------------------------------------------------ scene sets --
@pytest.fixture
def experiment(tmp_path, monkeypatch) -> Experiment:
    monkeypatch.setattr(cli, "ensure_runtime_env", lambda: None)
    from cap_harness.vla import protocol, students

    monkeypatch.setattr(students, "reexec", lambda *_a, **_k: None)
    monkeypatch.setattr(protocol, "task_count", lambda suite: 10)
    return Experiment.create(
        tmp_path / "trace2vla" / "exp",
        {"suite": "libero_object_swap", "tasks": [0, 1], "instructions": {"0": "a", "1": "b"},
         "scene_sets": {"official": {"init_mode": "saved", "seeds": "1-50", "role": "test"}}},
    )  # fmt: skip


def test_a_scene_set_can_carry_its_own_suite(experiment, capsys) -> None:
    root = str(experiment.root)
    command = ["scenes", root, "--add", "alt1-train", "--suite", "libero_object_swap_alt1"]
    assert cli.main([*command, "--seeds", "1001-1020"]) == 0
    sets = scenes.scene_sets(Experiment.open(root).settings)
    assert sets["alt1-train"].suite_or("x") == "libero_object_swap_alt1"
    assert sets["official"].suite_or("libero_object_swap") == "libero_object_swap"
    assert "libero_object_swap_alt1" in capsys.readouterr().out
    # A sibling layout has no saved layouts: refused.
    saved = ["scenes", root, "--add", "bad", "--suite", "libero_object_swap_alt2"]
    assert cli.main([*saved, "--init-mode", "saved"]) == 2
    assert cli.main(command) == 2  # exists already


def test_the_teacher_runs_the_scene_sets_suite(experiment, tmp_path) -> None:
    program = tmp_path / "program.py"
    program.write_text("result = 1\n")
    teacher.register_program(experiment, 0, program)
    argv_file = tmp_path / "argv.txt"
    batch = tmp_path / "fake-cap-harness"
    batch.write_text(f'#!/bin/sh\necho "$@" > "{argv_file}"\n')
    batch.chmod(0o755)
    alt = scenes.SceneSet("alt1-train", "seeded", (1001,), "train", "libero_object_swap_alt1")
    teacher.run_scene_set(experiment, alt, [0], cap_harness=batch, log=lambda _l: None)
    assert "--suite libero_object_swap_alt1" in argv_file.read_text()


def test_an_evaluation_on_a_sibling_scene_set_uses_its_suite(experiment, monkeypatch) -> None:
    from cap_harness.vla import benchmark

    seen = []
    monkeypatch.setattr(benchmark, "main", lambda argv: seen.append(argv) or 0)
    root = str(experiment.root)
    cli.main(["scenes", root, "--add", "alt4-dev", "--suite", "libero_object_swap_alt4",
              "--seeds", "2001-2010", "--role", "dev"])  # fmt: skip
    assert cli.main(["eval", root, "--name", "alt4-before", "--scenes", "alt4-dev"]) == 0
    assert seen[0][seen[0].index("--suite") + 1] == "libero_object_swap_alt4"
    assert seen[0][seen[0].index("--init-mode") + 1] == "seeded"
    meta = read_json(experiment.eval_dir("alt4-before") / "eval.json")
    assert meta["suite"] == "libero_object_swap_alt4" and meta["scene_set"] == "alt4-dev"
