"""Trace2VLA pipeline pieces that need no simulator: store, scene sets, grading logic,
teacher-episode reading, idle-frame dropping, the trainer wrapper, the report, the dashboard.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from cap_harness.trace2vla import convert, report, scenes, teacher, train
from cap_harness.trace2vla.episodes import load_teacher_episode
from cap_harness.trace2vla.store import Experiment, read_json, write_json
from cap_harness.vla.grading import LiberoGrader


def _experiment(tmp_path: Path, tasks=(0, 1)) -> Experiment:
    return Experiment.create(
        tmp_path / "trace2vla" / "exp",
        {
            "suite": "libero_object_swap",
            "tasks": list(tasks),
            "instructions": {"0": "pick the soup", "1": "pick the cheese"},
        },
    )


# ------------------------------------------------------------------------- store, scenes --
def test_experiment_ledger_records_every_event(tmp_path) -> None:
    experiment = _experiment(tmp_path)
    experiment.log("teacher_run", scene_set="official", task_id=0, ran=50)
    reopened = Experiment.open(experiment.root)
    assert reopened.suite == "libero_object_swap" and reopened.tasks == [0, 1]
    events = reopened.events()
    assert [e["event"] for e in events] == ["experiment_created", "teacher_run"]
    assert events[1]["ran"] == 50 and "time" in events[1]
    assert reopened.events(["teacher_run"])[0]["scene_set"] == "official"
    with pytest.raises(FileExistsError):
        Experiment.create(experiment.root, {"suite": "x"})
    with pytest.raises(FileNotFoundError, match=r"t2v\.sh init"):
        Experiment.open(tmp_path / "missing")


def test_scene_sets_are_disjoint_from_aspires_own_seeds() -> None:
    sets = scenes.scene_sets({})
    assert sets["official"].init_mode == "saved" and sets["official"].seeds == tuple(range(1, 51))
    assert sets["official"].role == "test" and sets["train"].role == "train"
    aspire = set(range(1, 81))  # held-out 1-50, development 51-65, unseen 66-80 (seeded mode)
    for name in ("train", "dev"):
        assert sets[name].init_mode == "seeded" and not aspire & set(sets[name].seeds)
    assert not set(sets["train"].seeds) & set(sets["dev"].seeds)
    custom = scenes.scene_sets(
        {"scene_sets": {"train": {"init_mode": "seeded", "seeds": "1001-1003,1010"}}}
    )
    assert custom["train"].seeds == (1001, 1002, 1003, 1010)
    assert scenes.compact([1, 2, 3, 7, 9, 10]) == "1-3,7,9-10"
    with pytest.raises(KeyError, match="unknown scene set"):
        scenes.scene_set({}, "nope")


# ---------------------------------------------------------------------------- programs --
def test_programs_are_found_in_sessions_and_campaigns_and_frozen(tmp_path) -> None:
    session = tmp_path / "session" / "agents"
    (session / "llm-libero-pro-libero_object_swap-0").mkdir(parents=True)
    (session / "llm-libero-pro-libero_object_swap-0" / "program_best.py").write_text(
        "# session 0\n"
    )
    (session / "llm-libero-pro-libero_goal_swap-1").mkdir()
    (session / "llm-libero-pro-libero_goal_swap-1" / "program_best.py").write_text(
        "# other suite\n"
    )
    campaign = tmp_path / "campaign" / "libero-pro" / "libero_object_swap" / "task_1"
    campaign.mkdir(parents=True)
    (campaign / "fix_code.py").write_text("# campaign 1\n")

    assert set(teacher.discover_programs(tmp_path / "session", "libero_object_swap")) == {0}
    assert set(teacher.discover_programs(tmp_path / "campaign", "libero_object_swap")) == {1}

    experiment = _experiment(tmp_path)
    record = teacher.register_program(
        experiment, 0, session / "llm-libero-pro-libero_object_swap-0" / "program_best.py"
    )
    assert experiment.program(0).read_text() == "# session 0\n" and len(record["sha256"]) == 64
    assert set(teacher.programs(experiment)) == {0}
    with pytest.raises(FileExistsError, match="different program"):
        teacher.register_program(experiment, 0, campaign / "fix_code.py")
    teacher.register_program(experiment, 0, campaign / "fix_code.py", force=True)
    assert experiment.program(0).read_text() == "# campaign 1\n"
    assert [e["event"] for e in experiment.events(["teacher_program_registered"])] == [
        "teacher_program_registered"
    ] * 2


# ---------------------------------------------------------------------- teacher episodes --
def _run_folder(
    root: Path, *, seed: int = 51, success: bool = True, grippers=(1.0, 1.0, 0.0, 0.0)
) -> Path:
    run = root / f"{seed:04d}" / f"run-{seed}"
    (run / "episode").mkdir(parents=True)

    def state(k: int) -> dict:
        return {
            "joint_positions": {"primary": {"$array_inline": [0.1 * k] * 7}},
            "end_effector_poses": {
                "primary": {
                    "position": {"$array_inline": [0.4, 0.0, 0.1 * k]},
                    "quaternion_wxyz": {"$array_inline": [0, 1, 0, 0]},
                }
            },
            "gripper_positions": {"primary": 1.0},
        }

    write_json(
        run / "run.json",
        {
            "suite": "libero_object_swap",
            "task_id": 0,
            "seed": seed,
            "init_mode": "seeded",
            "status": "finalized",
        },
    )
    write_json(
        run / "outcome.json",
        {
            "task_success": success,
            "termination_reason": "task_succeeded" if success else "program_completed",
            "steps_executed": len(grippers),
        },
    )
    write_json(run / "episode" / "reset.json", {"observation": {"robot_state": state(0)}})
    with (run / "episode" / "steps.jsonl").open("w") as stream:
        for k, gripper in enumerate(grippers, start=1):
            command = {"target": {"$array_inline": [0.1 * k] * 7}, "gripper_position": gripper}
            stream.write(
                json.dumps(
                    {"ok": True, "action": {"arms": {"primary": command}}, "robot_state": state(k)}
                )
                + "\n"
            )
    return run


def test_teacher_episode_keeps_closed_gripper_commands(tmp_path) -> None:
    """A closed-gripper command is 0.0, which is falsy: it must not be read as "open"."""
    episode = load_teacher_episode(_run_folder(tmp_path, grippers=(1.0, 0.0, None, 1.0)))
    assert episode.success and episode.steps == 4 and episode.init_mode == "seeded"
    np.testing.assert_array_equal(
        episode.gripper_commands, [1.0, 0.0, 0.0, 1.0]
    )  # None repeats the last
    assert episode.joint_positions.shape == (5, 7) and episode.hand_positions.shape == (5, 3)
    np.testing.assert_allclose(episode.hand_positions[:, 2], [0.0, 0.1, 0.2, 0.3, 0.4])
    assert episode.scene == {
        "suite": "libero_object_swap",
        "task_id": 0,
        "init_mode": "seeded",
        "seed": 51,
    }


def test_teacher_results_count_outcomes_and_families(tmp_path) -> None:
    experiment = _experiment(tmp_path, tasks=(0,))
    root = experiment.teacher_dir("official", 0)
    _run_folder(root, seed=1, success=True)
    failed = _run_folder(root, seed=2, success=False)
    write_json(
        experiment.grade_dir("official", 0) / "seed_2.json",
        {
            "run": experiment.rel(failed),
            "success": False,
            "reproduced": True,
            "labels": {"failure_family": "physical"},
        },
    )
    summary = teacher.results(experiment, "official", write=True)
    task = summary["tasks"][0]
    assert (task["successes"], task["episodes"], task["graded"]) == (1, 2, 1)
    assert task["terminations"] == {"program_completed": 1} and task["failure_families"] == {
        "physical": 1
    }
    assert task["seeds"]["2"]["family"] == "physical" and task["seeds"]["1"]["success"] is True
    assert read_json(experiment.root / "teacher" / "official" / "results.json")["successes"] == 1


def test_episodes_of_a_dead_service_or_a_replaced_program_are_not_counted(tmp_path) -> None:
    """They stay on disk, teacher-run records those seeds again, and results leave them out."""
    experiment = _experiment(tmp_path, tasks=(0,))
    program = tmp_path / "first.py"
    program.write_text("result = 1\n")
    registered = teacher.register_program(experiment, 0, program)
    root = experiment.teacher_dir("official", 0)

    def provenance(run: Path, sha: str) -> None:
        write_json(
            run / "source" / "provenance.json",
            {"input_path": str(experiment.program(0).resolve()), "sha256": sha},
        )

    good = _run_folder(root, seed=1, success=True)
    provenance(good, registered["sha256"])
    oom = _run_folder(root, seed=2, success=False)
    provenance(oom, registered["sha256"])
    write_json(
        oom / "source" / "program-result.json",
        {"calls": [{"tool": "localize_object", "failure_code": "provider_unavailable"}]},
    )
    assert teacher.service_failed(oom) and not teacher.service_failed(good)
    assert sorted(teacher.episodes_of(experiment, "official", 0)) == [1]
    assert teacher.set_aside(experiment, "official", 0) == {"service_failed": [2]}
    task = teacher.results(experiment, "official")["tasks"][0]
    assert (task["successes"], task["episodes"]) == (1, 1)
    assert task["set_aside"] == {"service_failed": [2]}

    # A grade made from another run of the seed is not this episode's grade.
    write_json(
        experiment.grade_dir("official", 0) / "seed_1.json",
        {"run": "teacher/official/task_00/0001/run-older", "labels": {"failure_family": "x"}},
    )
    assert teacher.grade_of(experiment, "official", 0, 1, good) is None

    # Replacing the program sets its episodes aside, so the new program is run on every seed.
    replacement = tmp_path / "second.py"
    replacement.write_text("result = 2\n")
    teacher.register_program(experiment, 0, replacement, force=True)
    assert teacher.episodes_of(experiment, "official", 0) == {}
    assert teacher.set_aside(experiment, "official", 0) == {
        "service_failed": [2],
        "old_program": [1],
    }


def test_an_episode_stopped_by_ctrl_c_is_not_counted_and_is_recorded_again(tmp_path) -> None:
    """Ctrl-C during a reset still finalizes the run (user_interrupt), with no reset state."""
    experiment = _experiment(tmp_path, tasks=(0,))
    root = experiment.teacher_dir("train", 0)
    _run_folder(root, seed=1010, success=True)
    stopped = _run_folder(root, seed=1011, success=False)
    write_json(
        stopped / "outcome.json",
        {"task_success": False, "termination_reason": "user_interrupt", "steps_executed": 0},
    )
    (stopped / "episode" / "reset.json").unlink()
    assert teacher.interrupted(stopped)
    assert sorted(teacher.episodes_of(experiment, "train", 0)) == [1010]
    assert teacher.set_aside(experiment, "train", 0) == {"interrupted": [1011]}
    task = teacher.results(experiment, "train")["tasks"][0]
    assert (task["successes"], task["episodes"], task["terminations"]) == (1, 1, {})

    # A finalized run whose recording is incomplete is set aside the same way.
    broken = _run_folder(root, seed=1012, success=True)
    (broken / "episode" / "steps.jsonl").unlink()
    assert teacher.interrupted(broken)
    assert sorted(teacher.episodes_of(experiment, "train", 0)) == [1010]

    # Run again, the seed gets a complete episode and that one counts.
    _run_folder(root / "again", seed=1011, success=True)
    assert sorted(teacher.episodes_of(experiment, "train", 0)) == [1010, 1011]


def test_teacher_run_reports_each_episode_as_it_ends(tmp_path) -> None:
    """The batch's output goes to its log file, and every finished seed is reported live."""
    experiment = _experiment(tmp_path, tasks=(0,))
    program = tmp_path / "program.py"
    program.write_text("result = 1\n")
    teacher.register_program(experiment, 0, program)
    batch = tmp_path / "fake-cap-harness"
    batch.write_text(
        "#!/bin/sh\n"
        "echo 'simulator chatter'\n"
        "echo 'seed 1001: task_success=True program_ok=True 36.3s pid=1 rss=10MB'\n"
        "echo 'seed 1002: task_success=False program_ok=False 12.0s pid=1 rss=10MB'\n"
        "echo 'warning (seeds 1002): close_gripper closed on nothing'\n"
    )
    batch.chmod(0o755)
    lines: list[str] = []
    summary = teacher.run_scene_set(
        experiment,
        scenes.SceneSet("train", "seeded", (1001, 1002), "train"),
        [0],
        cap_harness=batch,
        log=lines.append,
    )
    assert "  task 0 seed 1001: success, 36 s  [1/2]" in lines
    assert "  task 0 seed 1002: not solved (program error), 12 s  [2/2]" in lines
    assert any(line.strip().startswith("warning (seeds 1002)") for line in lines)
    assert not any("chatter" in line for line in lines)
    (log_file,) = experiment.teacher_dir("train", 0).glob("batch-*.log")
    assert "simulator chatter" in log_file.read_text()
    assert summary["tasks"]["0"]["exit_status"] == 0


def test_a_session_program_carries_its_held_out_record(tmp_path) -> None:
    workspace = tmp_path / "agents" / "llm-libero-pro-libero_object_swap-0"
    (workspace / "dev_v04").mkdir(parents=True)
    (workspace / "heldout").mkdir()
    program = workspace / "program_best.py"
    program.write_text("result = 1\n")
    write_json(workspace / "dev_v04" / "batch.json", {"passes": 0, "seeds": 3})
    write_json(
        workspace / "heldout" / "batch.json",
        {"passes": 0, "seeds": 3, "warnings": [{"text": "guessed object", "seeds": [1, 2, 3]}]},
    )
    record = teacher.agent_record(program)
    assert record == {
        "dev": "0/3",
        "heldout": "0/3",
        "heldout_passes": 0,
        "warnings": ["guessed object"],
    }
    assert teacher.agent_record(tmp_path / "plain.py") is None


# ------------------------------------------------------------------------------ grading --
class GradedEnv:
    """Objects on a table with a basket; the goal is (in soup basket_region)."""

    def __init__(self) -> None:
        self.positions = {
            "soup": np.array([0.0, 0.0, 0.0]),
            "butter": np.array([0.2, 0.0, 0.0]),
            "basket": np.array([0.0, 0.5, 0.0]),
        }
        self.objects_dict = dict.fromkeys(self.positions)
        self.obj_body_id = {name: index for index, name in enumerate(self.positions)}
        self.parsed_problem = {"goal_state": [["in", "soup", "basket_contain_region"]]}
        self.sim = SimpleNamespace(data=SimpleNamespace(body_xpos=self))

    def __getitem__(self, index: int) -> np.ndarray:
        return list(self.positions.values())[index]

    def _eval_predicate(self, state) -> bool:
        return bool(
            np.linalg.norm(self.positions[state[1]][:2] - self.positions["basket"][:2]) < 0.05
        )


def _grade(moves, success: bool = False) -> dict:
    """moves: (object or None, gripper closed, lift, to basket) per step."""
    env = GradedEnv()
    adapter = SimpleNamespace(native_env=SimpleNamespace(env=env, sim=env.sim))
    grader = LiberoGrader(adapter)
    grader.start()
    for near, closed, lift, place in moves:
        hand = env.positions[near] + [0.0, 0.0, 0.01] if near else np.array([0.5, 0.5, 0.5])
        grader.step({"robot0_eef_pos": hand}, closed)
        if near and lift:
            env.positions[near] = env.positions[near] + [0.0, 0.0, 0.1]
            grader.step({"robot0_eef_pos": hand}, closed)
        if near and place:
            env.positions[near] = np.array([0.0, 0.5, 0.0])
            grader.step({"robot0_eef_pos": hand}, False)
    return grader.finish(success)


def test_grader_separates_semantic_from_physical_failures() -> None:
    wrong_placed = _grade([("butter", True, True, True)])
    assert wrong_placed["failure_family"] == "semantic" and wrong_placed["chosen"] == "butter"
    assert wrong_placed["semantic_correct"] is False and wrong_placed["physical_success"] is True
    assert wrong_placed["placed"] == ["butter"] and wrong_placed["target"] == "soup"

    right_dropped = _grade([("soup", True, True, False)])
    assert right_dropped["failure_family"] == "physical" and right_dropped["target_lifted"] is True

    wrong_dropped = _grade([("butter", True, True, False)])
    assert wrong_dropped["failure_family"] == "both"

    closed_on_wrong_never_lifted = _grade([("butter", True, False, False)])
    assert (
        closed_on_wrong_never_lifted["failure_family"] == "both"
        and closed_on_wrong_never_lifted["chosen"] == "butter"
    )

    assert _grade([(None, False, False, False)])["failure_family"] == "no_attempt"
    solved = _grade([("soup", True, True, True)], success=True)
    assert solved["failure_family"] == "success" and solved["target_placed"] is True


def test_grader_reports_unknown_for_goals_it_cannot_read() -> None:
    env = GradedEnv()
    env.parsed_problem = {"goal_state": [["open", "drawer"]]}
    grader = LiberoGrader(SimpleNamespace(native_env=SimpleNamespace(env=env, sim=env.sim)))
    grader.start()
    grader.step({"robot0_eef_pos": np.zeros(3)}, True)
    assert grader.finish(False) == {"supported": False, "failure_family": "unknown"}
    assert grader.finish(True)["failure_family"] == "success"


# -------------------------------------------------------------------------- idle frames --
def test_idle_frames_follow_the_libero_rule_and_dwell_puts_some_back() -> None:
    move, still = [0.5, 0, 0, 0, 0, 0], [0.01, 0, 0, 0, 0.02, 0]
    actions = np.array(
        [
            move + [-1],
            still + [-1],
            move + [-1],
            still + [1],
            still + [1],
            still + [1],
            still + [1],
            move + [1],
        ],
        dtype=float,
    )
    np.testing.assert_array_equal(
        convert.idle_frames(actions), [False, True, False, False, True, True, True, False]
    )  # frame 3 changes the gripper, so it is not idle
    np.testing.assert_array_equal(convert.kept_indices(actions, 0.05, dwell=0), [0, 2, 3, 7])
    np.testing.assert_array_equal(convert.kept_indices(actions, 0.05, dwell=2), [0, 2, 3, 4, 5, 7])
    np.testing.assert_array_equal(
        convert.kept_indices(actions, 0.05, dwell=8), [0, 2, 3, 4, 5, 6, 7]
    )
    assert convert.ConvertSettings().to_dict()["track"]["speed"] == 3.0


# ------------------------------------------------------------------------------ training --
def test_train_command_and_metrics() -> None:
    settings = train.TrainSettings(method="lora-vlm", steps=3000, batch_size=32).resolved()
    assert (
        settings["lr"] == 1e-4 and settings["save_freq"] == 3000 and settings["warmup_steps"] == 200
    )
    command = train.build_command(
        python="python", checkpoint=Path("/ckpt"), dataset_root=Path("/data/d50"), repo_id="trace2vla/d50",
        output_dir=Path("/out/lerobot"), name="ft", settings=settings,
    )  # fmt: skip
    assert command[:3] == ["python", "-m", "lerobot.scripts.lerobot_train"]
    for expected in (
        "--policy.path=/ckpt",
        "--dataset.root=/data/d50",
        "--eval_freq=0",
        "--wandb.enable=false",
        "--peft.method_type=LORA",
        "--peft.r=32",
        "--policy.scheduler_decay_steps=3000",
    ):
        assert expected in command
    assert any(
        arg.startswith("--peft.target_modules=") and "vlm_with_expert" in arg for arg in command
    )
    rename = json.loads(
        next(arg for arg in command if arg.startswith("--rename_map=")).split("=", 1)[1]
    )
    assert rename == {
        "observation.images.image": "observation.images.camera1",
        "observation.images.image2": "observation.images.camera2",
    }

    full = train.build_command(
        python="python", checkpoint=Path("/c"), dataset_root=Path("/d"), repo_id="r", output_dir=Path("/o"), name="n",
        settings=train.TrainSettings(method="full").resolved(),
    )  # fmt: skip
    assert not any("peft" in arg for arg in full) and "--policy.optimizer_lr=5e-05" in full
    with pytest.raises(ValueError, match="method must be"):
        train.TrainSettings(method="qlora").resolved()

    line = "INFO 2026-11-11 ot_train.py:400 step:1K smpl:32K ep:164 epch:1.64 loss:0.087 grdn:0.912 lr:8.5e-05 updt_s:0.312 data_s:0.004"
    assert train.parse_metrics(line) == {
        "step": 1000, "samples": 32000.0, "epochs": 1.64, "loss": 0.087, "grad_norm": 0.912, "lr": 8.5e-05,
        "update_s": 0.312, "data_s": 0.004,
    }  # fmt: skip
    assert train.parse_metrics("INFO Creating policy") is None
    assert train.checkpoint_of(Path("/nonexistent")) is None


# -------------------------------------------------------------------- report, dashboard --
def _filled_experiment(tmp_path: Path) -> Experiment:
    experiment = _experiment(tmp_path)
    for task, successes in ((0, 48), (1, 45)):
        root = experiment.teacher_dir("official", task)
        for seed in range(1, 4):
            run = _run_folder(root, seed=seed, success=seed <= 2)
            write_json(run / "run.json", {**read_json(run / "run.json"), "task_id": task})
        del successes
    teacher.results(experiment, "official", write=True)
    for name, checkpoint, rates in (
        ("before-official", "base (lerobot/smolvla_libero)", (0, 0)),
        ("after-official", "training/ft-full", (40, 30)),
    ):
        run = experiment.eval_dir(name)
        run.mkdir(parents=True)
        write_json(run / "run.json", {"settings": {"suite": "libero_object_swap", "policy": "p"}})
        write_json(run / "eval.json", {"checkpoint_label": checkpoint, "scene_set": "official"})
        write_json(run / "progress.json", {"status": "finished", "done": 100, "planned": 100})
        tasks = [
            {
                "task_id": t,
                "suite": "libero_object_swap",
                "successes": r,
                "episodes": 50,
                "success_rate": r / 50,
                "failure_families": {"semantic": 50 - r},
            }
            for t, r in enumerate(rates)
        ]
        write_json(
            run / "results.json",
            {
                "tasks": tasks,
                "mean": sum(rates) / 100,
                "successes": sum(rates),
                "episodes": 100,
                "ci95": [0.1, 0.2],
                "failure_families": {"semantic": 100 - sum(rates)},
            },
        )
    demo = experiment.demo_dir("train-3x") / "task_00" / "seed_1001"
    write_json(
        demo / "episode.json",
        {
            "accepted": True,
            "steps": 195,
            "teacher_steps": 525,
            "tracked_steps": 204,
            "dwell": 0,
            "within_student_limit": True,
            "scene": {"task_id": 0, "seed": 1001, "init_mode": "seeded"},
            "instruction": "pick the soup",
        },
    )
    write_json(
        demo.with_name("seed_1002.rejected.json"),
        {
            "accepted": False,
            "rejected": "re-execution_failed",
            "scene": {"task_id": 0, "seed": 1002},
        },
    )
    write_json(
        experiment.dataset_dir("d50") / "manifest.json",
        {"name": "d50", "frames": 195, "per_task": {"0": 1}, "episodes": [{}], "rejected": []},
    )
    training = experiment.training_dir("ft-full")
    write_json(
        training / "run.json",
        {
            "name": "ft-full",
            "settings": {
                "method": "full",
                "steps": 3000,
                "batch_size": 32,
                "lr": 5e-5,
                "log_freq": 50,
            },
            "dataset": "/x/d50",
            "dataset_episodes": 1,
            "status": "finished",
            "final_loss": 0.05,
            "wall_s": 900,
        },
    )
    write_json(
        training / "progress.json",
        {"status": "finished", "step": 3000, "steps": 3000, "loss": 0.05},
    )
    (training / "metrics.jsonl").write_text(
        "".join(json.dumps({"step": s, "loss": 1 / s, "lr": 1e-5}) + "\n" for s in (50, 100))
    )
    return experiment


def test_report_compares_teacher_and_student_per_task(tmp_path) -> None:
    experiment = _filled_experiment(tmp_path)
    result = report.write(experiment)
    text = result["markdown"]
    assert (
        "| task | teacher (official scenes) | student: before-official | student: after-official |"
        in text
    )
    assert "| libero_object_swap:0 | 2/3 (66.7%) | 0/50 (0.0%) | 40/50 (80.0%) |" in text
    assert "| **mean** | 66.7% | **0.0%** | **70.0%** |" in text
    assert (
        "| train-3x | 0 | 1 | 1 (re-execution_failed) | 195 / 195 / 195 | 525 | 9.0 | 0 |" in text
    )
    assert (
        "| ft-full | full | 3000/3000 | 32 | 5e-05 | d50 (1 episodes) | finished | 0.05 | 15 min |"
        in text
    )
    assert (experiment.root / "reports" / "summary.md").read_text() == text
    assert result["report"]["evaluations"][1]["failure_families"] == {"semantic": 30}


def test_dashboard_reads_an_experiment_and_stays_inside_trace2vla(tmp_path) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "gpu"))
    from dashboard_store import Store
    from dashboard_t2v import T2vStore
    from dashboard_vla import VlaStore

    (tmp_path / "outputs").mkdir()
    experiment = _filled_experiment(tmp_path / "outputs")
    store = Store(tmp_path / "outputs")
    t2v = T2vStore(store)

    listing = t2v.experiments()["items"]
    assert listing[0]["name"] == "exp" and listing[0]["teacher_official"] == {
        "successes": 4,
        "episodes": 6,
    }
    assert [e["name"] for e in listing[0]["evals"]] == ["before-official", "after-official"]

    data = t2v.experiment(store.resolve("trace2vla/exp"))
    assert data["teacher"]["official"]["tasks"][0]["seeds"]["1"]["run"].startswith(
        "trace2vla/exp/teacher/official/task_00/"
    )
    assert data["training"][0]["loss"] == [1 / 50, 1 / 100] and data["demos"][0]["accepted"] == 1
    assert data["ledger"][0]["event"] == "experiment_created"

    demos = t2v.demos(store.resolve("trace2vla/exp/demos/train-3x"))
    assert [(e["accepted"], e["rejected"]) for e in demos["episodes"]] == [
        (True, None),
        (False, "re-execution_failed"),
    ]

    # A student evaluation inside an experiment opens in the student pages; other folders do not.
    assert (
        VlaStore(store).run(store.resolve("trace2vla/exp/evals/after-official"))["results"][
            "successes"
        ]
        == 70
    )
    (tmp_path / "outputs" / "sessions" / "s1").mkdir(parents=True)
    with pytest.raises(ValueError, match="not a VLA student run"):
        VlaStore(store).run(store.resolve("sessions/s1"))
    with pytest.raises(ValueError, match="not a Trace2VLA experiment"):
        t2v.experiment(store.resolve("sessions/s1"))
    del experiment
