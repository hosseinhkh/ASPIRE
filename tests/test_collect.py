"""t2v.sh collect: programs as their agents finish, the teacher on every (scene set, task), and
grading and conversion of each finished one in the background. The teacher and the background
jobs are stand-ins here; the real commands are checked on the machine (docs/trace2vla-full-run.md).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pytest

from cap_harness.trace2vla import cli, collect, progress, scenes, teacher
from cap_harness.trace2vla.store import Busy, Experiment, read_json, write_json

SUITE = "libero_object_swap"


@pytest.fixture
def experiment(tmp_path, monkeypatch) -> Experiment:
    monkeypatch.setattr(cli, "ensure_runtime_env", lambda: None)
    exp = Experiment.create(
        tmp_path / "trace2vla" / "exp",
        {"suite": SUITE, "tasks": [0, 1, 2], "instructions": {},
         "scene_sets": {"a-train": {"init_mode": "seeded", "seeds": "1001-1003", "role": "train"},
                        "b-train": {"init_mode": "seeded", "seeds": "1001-1002", "role": "train"}}},
    )  # fmt: skip
    for task in (0, 1):
        program = tmp_path / f"p{task}.py"
        program.write_text(f"result = {task}\n")
        teacher.register_program(exp, task, program)
    return exp


class FakeTeacher:
    """Records every scene asked for, unless told otherwise per (scene set, task)."""

    def __init__(self, monkeypatch) -> None:
        self.recorded: dict[tuple[str, int], set[int]] = {}
        self.calls: list[tuple[str, int]] = []
        self.skip: set[tuple[str, int]] = set()  # these record nothing
        self.raise_for: dict[tuple[str, int], list[BaseException]] = {}
        monkeypatch.setattr(teacher, "service_ports_down", lambda ports=None: [])
        monkeypatch.setattr(teacher, "missing_seeds", self.missing)
        monkeypatch.setattr(teacher, "run_scene_set", self.run)

    def missing(self, exp, s, task):
        return [seed for seed in s.seeds if seed not in self.recorded.get((s.name, task), set())]

    def run(self, exp, s, tasks, *, workers, videos, log):
        (task,) = tasks
        self.calls.append((s.name, task))
        pending = self.raise_for.get((s.name, task))
        if pending:
            raise pending.pop(0)
        if (s.name, task) in self.skip:
            return {}
        todo = self.missing(exp, s, task)
        for index, seed in enumerate(todo, start=1):
            self.recorded.setdefault((s.name, task), set()).add(seed)
            log(f"  task {task} seed {seed}: success, 20 s  [{index}/{len(todo)}]")
        return {}


def collector(exp, tmp_path, monkeypatch, sets=("a-train", "b-train"), **kwargs):
    """A collector whose background jobs append their scene set and tasks to prepared.txt."""
    record = tmp_path / "prepared.txt"

    def command(self, scene_set, tasks):
        line = f"{scene_set}:{','.join(map(str, tasks))}"
        return [sys.executable, "-c", f"open({str(record)!r}, 'a').write({line!r} + '\\n')"]

    monkeypatch.setattr(collect.Collector, "prepare_command", command)
    chosen = [scenes.scene_set(exp.settings, name) for name in sets]
    kwargs.setdefault("sleep", lambda _s: None)
    kwargs.setdefault("log", lambda _l: None)
    return collect.Collector(exp, chosen, exp.tasks, **kwargs), record


def prepared(record: Path) -> list[str]:
    return record.read_text().splitlines() if record.is_file() else []


def agent(session: Path, task: int, *, finished=True, program=True, heldout=(9, 10)) -> Path:
    workspace = session / "agents" / f"llm-libero-pro-{SUITE}-{task}"
    workspace.mkdir(parents=True, exist_ok=True)
    if program:
        (workspace / "program_best.py").write_text(f"result = 'agent {task}'\n")
    if heldout:
        write_json(
            workspace / "heldout" / "batch.json", {"passes": heldout[0], "seeds": heldout[1]}
        )
    if finished:
        write_json(workspace / "result.json", {"subtype": "success"})
    return workspace


# ------------------------------------------------------------------------ agent programs --
def test_an_agents_program_is_used_once_it_finished_and_solved_enough(tmp_path) -> None:
    session = tmp_path / "session"
    agent(session, 2)
    agent(session, 3, finished=False)
    agent(session, 4, program=False)
    agent(session, 5, heldout=(2, 10))
    agent(session, 6, heldout=None)
    (session / "agents" / f"llm-libero-pro-{SUITE}_alt1-7").mkdir()  # another suite
    (session / "agents" / "llm-robosuite-Lift-0").mkdir()
    found = collect.agent_programs(session, SUITE)
    assert sorted(found) == [2, 3, 4, 5, 6]
    states = {t: collect.judge(e, 0.5)[0] for t, e in found.items()}
    assert states == {2: "ready", 3: "agent_running", 4: "no_program", 5: "weak", 6: "unchecked"}


# ---------------------------------------------------------------------------- the loop --
def test_task_by_task_every_scene_set_then_prepared_in_the_background(
    experiment, tmp_path, monkeypatch
) -> None:
    fake = FakeTeacher(monkeypatch)
    run, record = collector(experiment, tmp_path, monkeypatch)
    assert run.run() == 1  # task 2 has no program: not collected, and said so
    assert fake.calls == [("a-train", 0), ("b-train", 0), ("a-train", 1), ("b-train", 1)]
    assert sorted(prepared(record)) == sorted(
        ["a-train:0", "b-train:0", "a-train:1", "b-train:1", "a-train:0,1", "b-train:0,1"]
    )  # one job per finished cell, then one final pass per scene set
    status = read_json(experiment.root / "collect.json")
    assert status["status"] == "finished" and status["summary"]["without_program"] == [2]
    assert status["cells"]["a-train/0"]["state"] == "recorded"
    assert status["cells"]["a-train/0"]["prepare"] == "finished"
    assert status["programs"]["2"]["state"] == "waiting"
    assert status["seconds_per_episode"] is not None and status["episodes_left"] == 0
    events = [e["event"] for e in experiment.events()]
    assert "collect_started" in events and "collect_finished" in events
    # A second run finds everything done: no new episode, only the cheap final pass.
    fake.calls.clear()
    assert run.run() == 1 and fake.calls == []


def test_programs_arrive_from_the_session_while_it_follows(
    experiment, tmp_path, monkeypatch
) -> None:
    fake = FakeTeacher(monkeypatch)
    session = tmp_path / "session"
    agent(session, 2, finished=False)
    waits = []

    def sleep(seconds):  # the agent of task 2 finishes while the collector waits
        waits.append(seconds)
        agent(session, 2)
        write_json(session / "session.json", {"status": "complete"})

    run, _ = collector(
        experiment, tmp_path, monkeypatch, session=session, follow=True, sleep=sleep, poll=7
    )
    assert run.run() == 0
    assert 7 in waits
    assert ("a-train", 2) in fake.calls and ("b-train", 2) in fake.calls
    registered = teacher.programs(experiment)[2]
    assert registered["source"].endswith(f"llm-libero-pro-{SUITE}-2/program_best.py")
    assert read_json(experiment.root / "collect.json")["programs"]["2"]["state"] == "taken"


def test_a_weak_program_is_not_used_and_a_finished_session_ends_the_wait(
    experiment, tmp_path, monkeypatch
) -> None:
    FakeTeacher(monkeypatch)
    session = tmp_path / "session"
    agent(session, 2, heldout=(1, 10))
    write_json(session / "session.json", {"status": "complete"})
    run, _ = collector(experiment, tmp_path, monkeypatch, session=session, follow=True)
    assert run.run() == 1
    assert 2 not in teacher.programs(experiment)
    state = read_json(experiment.root / "collect.json")["programs"]["2"]
    assert state["state"] == "weak" and "1/10" in state["note"]


def test_a_cell_that_keeps_missing_scenes_is_left_incomplete(
    experiment, tmp_path, monkeypatch
) -> None:
    fake = FakeTeacher(monkeypatch)
    fake.skip.add(("a-train", 1))
    run, record = collector(experiment, tmp_path, monkeypatch, attempts=2)
    assert run.run() == 1
    assert fake.calls.count(("a-train", 1)) == 2
    status = read_json(experiment.root / "collect.json")
    cell = status["cells"]["a-train/1"]
    assert cell["state"] == "incomplete" and "3 of 3 scenes" in cell["note"]
    assert status["summary"]["incomplete"] == ["a-train/1"]
    assert "a-train:1" in prepared(record)  # what exists is still prepared
    assert status["cells"]["b-train/1"]["state"] == "recorded"


def test_a_scene_set_another_process_records_is_taken_up_later(
    experiment, tmp_path, monkeypatch
) -> None:
    fake = FakeTeacher(monkeypatch)
    fake.raise_for[("a-train", 0)] = [Busy("teacher-a-train-task00 is in use by someone")]
    run, _ = collector(experiment, tmp_path, monkeypatch)
    assert run.run() == 1  # (task 2 has no program)
    assert fake.calls.count(("a-train", 0)) == 2
    assert read_json(experiment.root / "collect.json")["cells"]["a-train/0"]["state"] == "recorded"


def test_model_services_failing_three_times_in_a_row_stop_it(
    experiment, tmp_path, monkeypatch
) -> None:
    fake = FakeTeacher(monkeypatch)
    failure = RuntimeError("every episode of task 0 failed on a model service")
    fake.raise_for[("a-train", 0)] = [failure, failure, failure]
    run, _ = collector(experiment, tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="three runs in a row"):
        run.run()
    status = read_json(experiment.root / "collect.json")
    assert status["status"] == "failed" and "three runs" in status["error"]
    assert status["cells"]["a-train/0"]["attempts"] == 0  # not counted against the scenes


def test_it_waits_for_services_that_went_down_then_gives_up(
    experiment, tmp_path, monkeypatch
) -> None:
    FakeTeacher(monkeypatch)
    monkeypatch.setattr(teacher, "service_ports_down", lambda ports=None: [8114])
    waits = []
    run, _ = collector(experiment, tmp_path, monkeypatch, service_wait=120, sleep=waits.append)
    with pytest.raises(RuntimeError, match="have not answered for 2 min"):
        run.run()
    assert waits == [60.0, 60.0]


def test_one_collection_per_experiment(experiment, tmp_path, monkeypatch) -> None:
    FakeTeacher(monkeypatch)
    run, _ = collector(experiment, tmp_path, monkeypatch)
    with experiment.hold(["collect"], "t2v.sh collect"), pytest.raises(Busy):
        run.run()


def test_test_scenes_are_graded_never_converted(experiment) -> None:
    sets = [scenes.scene_set(experiment.settings, n) for n in ("official", "a-train")]
    run = collect.Collector(experiment, sets, [0], log=lambda _l: None)
    assert run.prepare_command("official", [0])[3] == "teacher-grade"
    assert run.prepare_command("a-train", [0, 1])[3:] == [
        "prepare", str(experiment.root), "--scenes", "a-train", "--tasks", "0,1", "--workers", "4",
    ]  # fmt: skip


def test_collect_checks_before_starting(experiment, tmp_path, monkeypatch, capsys) -> None:
    root = str(experiment.root)
    monkeypatch.setattr(teacher, "service_ports_down", lambda ports=None: [8114])
    assert cli.main(["collect", root, "--scenes", "a-train"]) == 1
    assert "8114" in capsys.readouterr().err
    assert cli.main(["collect", root, "--scenes", "a-train", "--programs-from", str(tmp_path)]) == 2


# ------------------------------------------------------------------------------ progress --
def test_the_overview_counts_every_cell_and_sees_a_dead_collection(
    experiment, tmp_path, monkeypatch
) -> None:
    FakeTeacher(monkeypatch)
    run, _ = collector(experiment, tmp_path, monkeypatch)
    run.run()
    write_json(experiment.root / "teacher" / "a-train" / "results.json",
               {"tasks": [{"task_id": 0, "episodes": 3, "successes": 2, "graded": 3}]})  # fmt: skip
    demos = experiment.demo_dir("a-train-3x")
    write_json(demos / "source.json", {"scene_set": "a-train"})
    for seed in (1001, 1002):
        write_json(demos / "task_00" / f"seed_{seed}" / "episode.json", {"accepted": True})
    write_json(demos / "task_00" / "seed_1003.rejected.json", {"accepted": False})
    view = progress.overview(experiment.root)
    a = next(s for s in view["scene_sets"] if s["name"] == "a-train")
    assert a["demo_sets"] == ["a-train-3x"] and a["in_collection"]
    cell = a["cells"][0]
    assert (cell["recorded"], cell["planned"], cell["solved"], cell["kept"], cell["rejected"]) == (
        3,
        3,
        2,
        2,
        1,
    )
    assert cell["state"] == "recorded"
    assert [s["name"] for s in view["scene_sets"]] == ["a-train", "b-train"]  # nothing else ran
    assert view["totals"]["kept"] == 2 and view["totals"]["planned"] == 15
    assert view["programs"]["0"]["registered"] and not view["programs"]["2"]["registered"]
    # It said running, but its process is gone.
    status = read_json(experiment.root / "collect.json")
    write_json(experiment.root / "collect.json", {**status, "status": "running", "pid": 2**22 + 7})
    assert progress.overview(experiment.root)["collect"]["status"] == "died"


def test_claims_and_background_stages_that_are_alive(experiment) -> None:
    with experiment.hold(["teacher-a-train-task00"], "teacher-run on 'a-train'"):
        held = progress.claims(experiment.root)
        assert [c["key"] for c in held] == ["teacher-a-train-task00"]
    assert progress.claims(experiment.root) == []  # released: the file is emptied
    assert (experiment.root / "locks" / "teacher-a-train-task00.lock").read_text() == ""
    (experiment.root / "logs").mkdir(exist_ok=True)
    (experiment.root / "logs" / "running.pid").write_text(
        f"{os.getpid()} collect\n{2**22 + 7} train\n"
    )
    assert progress.background(experiment.root) == [{"pid": os.getpid(), "stage": "collect"}]


def test_status_shows_the_collection(experiment, tmp_path, monkeypatch, capsys) -> None:
    FakeTeacher(monkeypatch)
    run, _ = collector(experiment, tmp_path, monkeypatch)
    run.run()
    monkeypatch.setattr(
        teacher, "results", lambda *_a, **_k: {"successes": 0, "episodes": 0, "tasks": []}
    )
    assert cli.main(["status", str(experiment.root)]) == 0
    out = capsys.readouterr().out
    assert "collect: finished" in out and "a-train" in out and "task 2: no program" in out


def test_a_dataset_takes_training_scenes_only(experiment, capsys) -> None:
    demos = experiment.demo_dir("dev-3x")
    write_json(demos / "source.json", {"scene_set": "dev"})
    write_json(demos / "task_00" / "seed_2001" / "episode.json", {"accepted": True})
    code = cli.main(["dataset", str(experiment.root), "--demos", "dev-3x", "--name", "d"])
    assert code == 2 and "choosing checkpoints (dev)" in capsys.readouterr().err
    assert json.loads((demos / "source.json").read_text()) == {"scene_set": "dev"}
