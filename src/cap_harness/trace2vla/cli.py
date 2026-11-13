"""Command line for the Trace2VLA pipeline (``python -m cap_harness.trace2vla``, t2v.sh).

init NAME --suite S --tasks 0,1          create the experiment folder
check NAME                               sanity checks before spending time or money
teacher-add NAME --from DIR              register task programs from a session or campaign
teacher-import NAME --from DIR --scenes SET   reuse recorded teacher episodes
teacher-run NAME --scenes SET            run the programs on a scene set (full recording)
teacher-grade NAME --scenes SET          replay and grade the teacher's episodes
convert NAME --scenes SET                successes -> demos in the student's control
dataset NAME --demos D --name N          demos -> LeRobot dataset (checkpoint statistics)
  [--student pi05] [--from-dataset N]    for another student; copy an existing dataset's frames
train NAME --dataset N --name RUN        fine-tune the student the dataset is for
eval NAME --name E --checkpoint C        benchmark a checkpoint on a scene set
  [--student pi05]                       which student's base checkpoint ('base')

Students (cap_harness.vla.students): smolvla (default) and pi05. Each runs in its own
environment; eval and train move themselves into the right one.
grade-student RUN_DIR                    add failure labels to an existing student run
report NAME                              teacher vs student before vs after
status [NAME]                            what exists, what is running, last events
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np

from cap_harness.trace2vla.scenes import DEFAULT_SCENE_SETS, compact, scene_set, scene_sets
from cap_harness.trace2vla.store import OUTPUTS, Experiment, read_json, write_json
from cap_harness.vla import students
from cap_harness.vla.libero_compat import ensure_runtime_env
from cap_harness.vla.protocol import parse_int_list

SERVICE_PORTS = {"sam3": 8114, "contact_graspnet": 8115, "pyroki": 8116, "curobo": 8118}


def _tasks(experiment: Experiment, text: str | None) -> list[int]:
    return experiment.tasks if not text or text == "all" else parse_int_list(text)


# --------------------------------------------------------------------------------- init --
def cmd_init(args: argparse.Namespace) -> int:
    ensure_runtime_env()
    from cap_harness.vla.protocol import task_count, task_language

    count = task_count(args.suite)
    tasks = parse_int_list(args.tasks, default=range(count))
    if any(t < 0 or t >= count for t in tasks):
        print(f"error: {args.suite} has tasks 0-{count - 1}", file=sys.stderr)
        return 2
    sets = {name: dict(spec) for name, spec in DEFAULT_SCENE_SETS.items()}
    if args.train_seeds:
        sets["train"]["seeds"] = args.train_seeds
    if args.dev_seeds:
        sets["dev"]["seeds"] = args.dev_seeds
    experiment = Experiment.create(
        args.name,
        {
            "benchmark": "libero-pro",
            "suite": args.suite,
            "tasks": tasks,
            "instructions": {str(t): task_language(args.suite, t) for t in tasks},
            "scene_sets": sets,
        },
    )
    print(f"created {experiment.root}")
    for task in tasks:
        print(f"  task {task}: {experiment.settings['instructions'][str(task)]}")
    for name, spec in sets.items():
        print(f"  scene set {name}: {spec['init_mode']} seeds {spec['seeds']} ({spec['role']})")
    return 0


# -------------------------------------------------------------------------------- check --
def _student_ready(student: students.Student) -> tuple[bool | None, str]:
    """(PASS/WARN, detail) for a student's own environment and checkpoint."""
    import subprocess

    python = student.python()
    if not python.exists():
        return None, f"not set up: {student.setup_command}"
    probe = subprocess.run(
        [str(python), "-c", f"import lerobot.policies.{student.policy_type}"],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    if probe.returncode != 0:
        last = (probe.stderr.strip().splitlines() or ["?"])[-1]
        return False, f"{student.venv} cannot import its policy ({last}); {student.setup_command}"
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(
            student.checkpoint, "model.safetensors", revision=student.revision
        )
    except Exception:
        cached = None
    weights = (
        "checkpoint downloaded" if isinstance(cached, str) else "checkpoint downloads on first use"
    )
    return True, f"{student.venv} ready; {student.checkpoint}: {weights}"


def cmd_check(args: argparse.Namespace) -> int:
    """Sanity checks. Exit status 1 when something a stage needs is missing."""
    ensure_runtime_env()
    experiment = Experiment.open(args.name)
    failed = 0

    def report(ok: bool | None, what: str, detail: str = "") -> None:
        nonlocal failed
        failed += ok is False
        print(
            f"  [{'PASS' if ok else ('WARN' if ok is None else 'FAIL')}] {what}"
            + (f": {detail}" if detail else "")
        )

    print(f"experiment {experiment.root.name}: {experiment.suite}, tasks {experiment.tasks}")
    try:
        import lerobot  # noqa: F401

        report(True, "LeRobot is installed in this environment")
    except ImportError:
        report(False, "LeRobot is installed", "run scripts/gpu/vla.sh setup")
    try:
        import peft  # noqa: F401

        report(True, "peft is installed (LoRA training)")
    except ImportError:
        report(
            None,
            "peft is installed (LoRA training)",
            "missing: run scripts/gpu/vla.sh setup (only --method lora needs it)",
        )
    try:
        from cap_harness.vla.policy import DEFAULT_POLICY, DEFAULT_REVISION, resolve_checkpoint

        report(True, "base checkpoint", resolve_checkpoint(DEFAULT_POLICY, DEFAULT_REVISION))
    except Exception as exc:
        report(False, "base checkpoint", f"{type(exc).__name__}: {exc}")
    for other in students.STUDENTS.values():
        if other.key == students.DEFAULT_STUDENT:
            continue
        ok, detail = _student_ready(other)
        report(ok, f"{other.label} student (optional)", detail)
    try:
        import torch

        if torch.cuda.is_available():
            free = [torch.cuda.mem_get_info(i)[0] / 2**30 for i in range(torch.cuda.device_count())]
            report(
                True,
                "GPU",
                ", ".join(f"visible GPU {i}: {f:.0f} GiB free" for i, f in enumerate(free)),
            )
        else:
            report(None, "GPU", "none visible: training and evaluation will run on CPU (very slow)")
    except ImportError:
        report(False, "PyTorch")
    free_gb = shutil.disk_usage(experiment.root).free / 2**30
    report(
        free_gb > 20,
        "disk space",
        f"{free_gb:.0f} GiB free (a teacher episode with videos is about 10 MB)",
    )

    from cap_harness.trace2vla import teacher

    registered = teacher.programs(experiment)
    missing = [t for t in experiment.tasks if t not in registered]
    report(
        not missing,
        "teacher programs",
        f"registered for tasks {sorted(registered)}"
        + (f"; missing {missing} (t2v.sh teacher-add)" if missing else ""),
    )
    down = [name for name, port in SERVICE_PORTS.items() if teacher.service_ports_down([port])]
    needed = [name for name in down if name != "curobo"]
    report(
        not needed if not args.no_services else None,
        "model services (needed by teacher-run only)",
        "all up" if not down else f"down: {', '.join(down)} (scripts/gpu/services.sh start)",
    )
    try:
        from cap_harness.trace2vla import replay
        from cap_harness.vla.grading import LiberoGrader

        adapter = replay.osc_adapter(64)
        for task in experiment.tasks:
            adapter.reset(f"{experiment.suite}:{task}", seed=1)
            grader = LiberoGrader(adapter)
            grader.start()
            report(
                True if grader.supported else None,
                f"simulator: task {task} resets",
                f"grading target {grader.target}"
                if grader.supported
                else "this task's goal is not a pick-and-place: failure labels will be 'unknown'",
            )
        adapter.close()
    except Exception as exc:
        report(False, "simulator", f"{type(exc).__name__}: {exc}")
    print("all checks passed" if not failed else f"{failed} check(s) failed")
    return 1 if failed else 0


# ------------------------------------------------------------------------------ teacher --
def cmd_teacher_add(args: argparse.Namespace) -> int:
    from cap_harness.trace2vla import teacher

    experiment = Experiment.open(args.name)
    if args.program:
        if args.task is None:
            print("error: --program needs --task", file=sys.stderr)
            return 2
        found = {args.task: Path(args.program)}
    else:
        found = teacher.discover_programs(Path(args.source), experiment.suite)
    wanted = {t: p for t, p in found.items() if t in experiment.tasks}
    if not wanted:
        print(
            f"error: no program for tasks {experiment.tasks} of {experiment.suite} under {args.source or args.program}",
            file=sys.stderr,
        )
        return 1
    for task, path in sorted(wanted.items()):
        record = teacher.register_program(experiment, task, path, force=args.force)
        print(f"task {task}: {path} (sha256 {record['sha256'][:12]})")
        measured = teacher.agent_record(path)
        if measured:
            print(
                f"    its agent measured: development {measured['dev']}, held-out {measured['heldout']}"
            )
            if not measured["heldout_passes"]:
                print(
                    "    WARNING: this program solved none of its held-out scenes. Its episodes "
                    "will be failures, not demonstrations: fix the program before teacher-run."
                )
            for text in measured["warnings"]:
                print(f"    warning: {text}")
    missing = [t for t in experiment.tasks if t not in teacher.programs(experiment)]
    if missing:
        print(f"still without a program: tasks {missing}")
    return 0


def cmd_teacher_import(args: argparse.Namespace) -> int:
    """Copy recorded teacher episodes (ASPIRE run folders) into the experiment as a scene set."""
    from cap_harness.trace2vla.episodes import find_runs

    experiment = Experiment.open(args.name)
    chosen: dict[tuple[int, int], Path] = {}
    modes = set()
    for run in find_runs(Path(args.source)):
        meta = read_json(run / "run.json") or {}
        if (
            meta.get("suite") != experiment.suite
            or int(meta.get("task_id", -1)) not in experiment.tasks
        ):
            continue
        if meta.get("status") != "finalized" or not (run / "outcome.json").is_file():
            continue
        if args.only_successes and not (read_json(run / "outcome.json") or {}).get("task_success"):
            continue
        if args.init_mode and meta.get("init_mode") != args.init_mode:
            continue
        modes.add(meta.get("init_mode"))
        chosen[(int(meta["task_id"]), int(meta["seed"]))] = run  # the newest run of a seed wins
    if not chosen:
        print(
            f"error: no finalized {experiment.suite} episodes for tasks {experiment.tasks} under {args.source}",
            file=sys.stderr,
        )
        return 1
    if len(modes) != 1:
        print(
            f"error: the episodes mix init modes {sorted(modes)}; pass --init-mode saved or seeded",
            file=sys.stderr,
        )
        return 2
    for (task, seed), run in sorted(chosen.items()):
        target = experiment.teacher_dir(args.scenes, task) / f"{seed:04d}" / run.name
        if not target.exists():
            shutil.copytree(run, target)
    seeds = sorted({seed for _, seed in chosen})
    experiment.settings.setdefault("scene_sets", {})[args.scenes] = {
        "init_mode": modes.pop(),
        "seeds": compact(seeds),
        "role": args.role,
    }
    experiment.save_settings()
    experiment.log(
        "teacher_imported",
        scene_set=args.scenes,
        source=str(Path(args.source).resolve()),
        episodes=len(chosen),
    )
    from cap_harness.trace2vla import teacher

    summary = teacher.results(experiment, args.scenes, write=True)
    print(
        f"imported {len(chosen)} episodes as scene set '{args.scenes}': {summary['successes']}/{summary['episodes']} solved"
    )
    return 0


def cmd_teacher_run(args: argparse.Namespace) -> int:
    from cap_harness.trace2vla import teacher

    experiment = Experiment.open(args.name)
    down = teacher.service_ports_down(
        [SERVICE_PORTS["sam3"], SERVICE_PORTS["contact_graspnet"], SERVICE_PORTS["pyroki"]]
    )
    if down and not args.force:
        print(
            f"error: model services are not answering on ports {down}; start them (scripts/gpu/services.sh start) or pass --force",
            file=sys.stderr,
        )
        return 1
    scenes = scene_set(experiment.settings, args.scenes)
    teacher.run_scene_set(
        experiment,
        scenes,
        _tasks(experiment, args.tasks),
        workers=args.workers,
        videos=not args.no_videos,
    )
    print(_teacher_table(teacher.results(experiment, args.scenes)))
    return 0


def _teacher_table(summary: dict[str, Any]) -> str:
    lines = [
        f"teacher on scene set '{summary['scene_set']}': {summary['successes']}/{summary['episodes']} solved"
    ]
    for t in summary["tasks"]:
        steps = t["steps_when_successful"]
        lines.append(
            f"  task {t['task_id']}: {t['successes']}/{t['episodes']}"
            + (f", steps {steps['min']}-{steps['max']} (mean {steps['mean']:.0f})" if steps else "")
            + (f", failures ended as {t['terminations']}" if t["terminations"] else "")
            + (f", families {t['failure_families']}" if t["failure_families"] else "")
        )
        aside = t.get("set_aside") or {}
        if aside.get("interrupted"):
            lines.append(
                f"    not counted: {len(aside['interrupted'])} episodes that were stopped before "
                "they ended (Ctrl-C, or a simulator failure); teacher-run records them again"
            )
        if aside.get("service_failed"):
            lines.append(
                f"    not counted: {len(aside['service_failed'])} episodes in which a model service "
                "did not answer (teacher-run records them again)"
            )
        if aside.get("old_program"):
            lines.append(
                f"    not counted: {len(aside['old_program'])} episodes of a replaced program "
                "(teacher-run records them again)"
            )
    return "\n".join(lines)


def cmd_teacher_grade(args: argparse.Namespace) -> int:
    ensure_runtime_env()
    from cap_harness.trace2vla import teacher

    experiment = Experiment.open(args.name)
    summary = teacher.grade_scene_set(experiment, args.scenes, _tasks(experiment, args.tasks))
    print(f"graded {summary['graded']} new episodes; families: {summary['families']}")
    if summary["unreadable"]:
        print(f"could not read, not graded: {', '.join(summary['unreadable'])}")
    print(_teacher_table(teacher.results(experiment, args.scenes)))
    return 1 if summary["unreadable"] else 0


# ------------------------------------------------------------------ convert and dataset --
def cmd_convert(args: argparse.Namespace) -> int:
    ensure_runtime_env()
    from cap_harness.trace2vla import convert, replay

    experiment = Experiment.open(args.name)
    settings = convert.ConvertSettings(
        track=replay.TrackSettings(speed=args.speed), drop_idle=not args.keep_idle
    )
    name = args.demos or f"{args.scenes}-{args.speed:g}x"
    summary = convert.convert_scene_set(
        experiment, args.scenes, name, _tasks(experiment, args.tasks), settings
    )
    print(f"demo set '{name}': {summary['accepted']} kept, {summary['rejected']} rejected")
    for t in summary["tasks"]:
        steps = t["steps"]
        print(
            f"  task {t['task_id']}: {t['accepted']} kept, {t['rejected']} rejected {t['rejected_reasons'] or ''}"
            + (
                f"; steps {steps['min']}-{steps['max']} (teacher mean {t['teacher_steps_mean']:.0f}); idle frames dropped {t['idle_dropped_mean']:.1f} on average"
                if steps
                else ""
            )
            + (
                f"; {t['over_student_limit']} over the student's step limit"
                if t["over_student_limit"]
                else ""
            )
        )
    return 0 if summary["accepted"] else 1


def statistics_checkpoint(student: students.Student) -> Path:
    """A local copy of the student's checkpoint with at least its config and statistics
    (the weights are not needed to build a dataset, and pi0.5's are 7.5 GB).
    """
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            student.checkpoint,
            revision=student.revision,
            allow_patterns=["config.json", "policy_*"],
        )
    )


def cmd_dataset(args: argparse.Namespace) -> int:
    ensure_runtime_env()
    from cap_harness.trace2vla import dataset

    experiment = Experiment.open(args.name)
    student = students.get(args.student)
    if args.from_dataset:
        source = experiment.dataset_dir(args.from_dataset)
        print(
            f"dataset '{args.dataset}': the frames of '{args.from_dataset}' with {student.label}'s "
            f"statistics ({student.checkpoint})"
        )
        manifest = dataset.derive_dataset(
            source,
            experiment.dataset_dir(args.dataset),
            name=args.dataset,
            checkpoint=statistics_checkpoint(student),
            student=student.key,
        )
        experiment.log(
            "dataset_derived", name=args.dataset, source=args.from_dataset, student=student.key,
            episodes=len(manifest["episodes"]), frames=manifest["frames"],
        )  # fmt: skip
    else:
        if args.demos is None:
            print("error: --demos is required (or --from-dataset NAME)", file=sys.stderr)
            return 2
        demos = dataset.demo_dirs(
            experiment.demo_dir(args.demos), _tasks(experiment, args.tasks), args.per_task
        )
        if not demos:
            print(
                f"error: demo set '{args.demos}' has no accepted demos for these tasks (t2v.sh convert)",
                file=sys.stderr,
            )
            return 1
        print(
            f"building dataset '{args.dataset}' for {student.label} from {len(demos)} demos (statistics from {student.checkpoint})"
        )
        manifest = dataset.build_dataset(
            demos,
            experiment.dataset_dir(args.dataset),
            name=args.dataset,
            checkpoint=statistics_checkpoint(student),
            student=student.key,
        )
        experiment.log(
            "dataset_built", name=args.dataset, demos=args.demos, student=student.key,
            episodes=len(manifest["episodes"]), frames=manifest["frames"], per_task=manifest["per_task"],
            rejected=len(manifest["rejected"]),
        )  # fmt: skip
    print(
        f"dataset '{args.dataset}' ({student.label}): {len(manifest['episodes'])} episodes, {manifest['frames']} frames, per task {manifest['per_task']}"
    )
    frozen = manifest["statistics"]["frozen"]["action"]
    print(
        "  action std, checkpoint vs this data: "
        + ", ".join(
            f"{a:.2f}/{b:.2f}"
            for a, b in zip(frozen["checkpoint_std"], frozen["data_std"], strict=True)
        )
    )
    return 0


# ------------------------------------------------------------------------- train / eval --
def _train_student(args: argparse.Namespace, experiment: Experiment) -> students.Student:
    """--student, else the student of the run trained from, else the one the dataset is for."""
    if args.student:
        return students.get(args.student)
    if args.base != "base":
        return _checkpoint(experiment, args.base)[2]
    manifest = read_json(experiment.dataset_dir(args.dataset) / "manifest.json") or {}
    return students.get(students.infer(manifest)[0])


def cmd_train(args: argparse.Namespace) -> int:
    from cap_harness.trace2vla import train
    from cap_harness.vla.policy import resolve_checkpoint

    experiment = Experiment.open(args.name)
    dataset_root = experiment.dataset_dir(args.dataset)
    if not (dataset_root / "manifest.json").is_file():
        print(f"error: no dataset '{args.dataset}' (t2v.sh dataset)", file=sys.stderr)
        return 1
    student = _train_student(args, experiment)
    built_for = students.infer(read_json(dataset_root / "manifest.json") or {})[0]
    if built_for != student.key:
        print(
            f"error: dataset '{args.dataset}' is for {students.label(built_for)}, not {student.label}: "
            f"t2v.sh dataset {experiment.root.name} --from-dataset {args.dataset} --student {student.key} --name NEW",
            file=sys.stderr,
        )
        return 2
    # Before anything heavy: each student trains in its own environment.
    students.reexec(student, "cap_harness.trace2vla", args.argv)
    base = (
        _checkpoint(experiment, args.base)[1]
        if args.base != "base"
        else Path(resolve_checkpoint(student.checkpoint, student.revision))
    )
    settings = train.TrainSettings(
        method=args.method, steps=args.steps, batch_size=args.batch_size, lr=args.lr, lora_rank=args.lora_rank,
        seed=args.seed, save_freq=args.save_freq, log_freq=args.log_freq, num_workers=args.num_workers, device=args.device,
        student=student.key,
    )  # fmt: skip
    print(
        f"training '{args.run}' ({student.label}): {args.method}, {args.steps} steps, batch {args.batch_size}, from {args.base}, on dataset '{args.dataset}'"
    )
    record = train.train(
        experiment.training_dir(args.run),
        checkpoint=Path(base),
        dataset_root=dataset_root,
        settings=settings,
    )
    experiment.log(
        "training_finished", name=args.run, dataset=args.dataset, base=args.base, student=student.key, status=record["status"],
        checkpoint=record.get("checkpoint"), final_loss=record.get("final_loss"), wall_s=record.get("wall_s"), settings=record["settings"],
    )  # fmt: skip
    print(
        f"training {record['status']}: final loss {record.get('final_loss')}, checkpoint {record.get('checkpoint')}"
    )
    if record.get("error"):
        print(
            f"error: {record['error']} ({experiment.training_dir(args.run) / 'train.log'})",
            file=sys.stderr,
        )
    return 0 if record["status"] == "finished" else 1


def _training_ref(experiment: Experiment, name: str) -> tuple[str | None, int | None]:
    """(training run, step) for ``RUN`` or ``RUN@STEP`` when RUN is a training run of this
    experiment; (None, None) otherwise. A malformed step is an error.
    """
    run_name, at, step = name.partition("@")
    if not run_name or not experiment.training_dir(run_name).is_dir():
        return None, None
    if not at:
        return run_name, None
    if not step.isdigit():
        raise ValueError(f"'{name}': the step after @ must be a number, e.g. {run_name}@500")
    return run_name, int(step)


def _checkpoint(
    experiment: Experiment, name: str, student: str | None = None
) -> tuple[str, str | None, students.Student]:
    """(label, policy path or None for the student's pinned base checkpoint, student).

    'base' is the base checkpoint of ``student`` (default SmolVLA); a training run or a
    checkpoint folder brings its own student, and naming a different one is an error.
    """
    from cap_harness.trace2vla.train import checkpoint_at, checkpoint_of, saved_steps

    if name == "base":
        chosen = students.get(student)
        return f"base ({chosen.checkpoint})", None, chosen
    run_name, step = _training_ref(experiment, name)
    run = experiment.training_dir(run_name) if run_name else None
    if run is not None:
        if step is None:
            path = checkpoint_of(run)
            if path is None:
                raise FileNotFoundError(f"training run '{run_name}' has no checkpoint yet")
            label = f"training/{run_name}"
        elif step == 0:
            # Step 0 is where the run started: the base checkpoint, or the run it continued.
            start = (read_json(run / "run.json") or {}).get("base_checkpoint")
            if not start or not Path(start).is_dir():
                raise FileNotFoundError(
                    f"training run '{run_name}' does not record the checkpoint it started from"
                )
            path = Path(start)
            label = f"training/{run_name}@0"
        else:
            path = checkpoint_at(run, step)
            if path is None:
                saved = saved_steps(run)
                raise FileNotFoundError(
                    f"training run '{run_name}' has no complete checkpoint at step {step} "
                    + (
                        f"(saved: {', '.join(str(s) for s in saved)})"
                        if saved
                        else "(none yet: train with --save-freq N to keep checkpoints during training)"
                    )
                )
            label = f"training/{run_name}@{step}"
        found = students.get(students.infer(read_json(run / "run.json") or {})[0])
        policy = str(path)
    elif Path(name).is_dir():
        label, policy = name, str(Path(name).resolve())
        found = students.for_source(policy) or students.get(student)
    else:
        raise FileNotFoundError(
            f"'{name}' is neither 'base', a training run of this experiment, nor a checkpoint folder"
        )
    if student is not None and students.get(student).key != found.key:
        raise ValueError(f"'{name}' is a {found.label} checkpoint, not {students.label(student)}")
    return label, policy, found


def cmd_eval(args: argparse.Namespace) -> int:
    ensure_runtime_env()
    from cap_harness.vla import benchmark

    experiment = Experiment.open(args.name)
    label, policy, student = _checkpoint(experiment, args.checkpoint, args.student)
    scenes = scene_set(experiment.settings, args.scenes)
    suite = args.suite or experiment.suite
    out = experiment.eval_dir(args.eval)
    previous = read_json(out / "eval.json") or {}
    if previous and students.infer(previous)[0] != student.key:
        print(
            f"error: evaluation '{args.eval}' holds {students.label(students.infer(previous)[0])} "
            f"episodes; choose another --name for {student.label}",
            file=sys.stderr,
        )
        return 2
    # Before anything heavy: each student runs in its own environment.
    students.reexec(student, "cap_harness.trace2vla", args.argv)
    out.mkdir(parents=True, exist_ok=True)
    training_run, checkpoint_step = _training_ref(experiment, args.checkpoint)
    write_json(
        out / "eval.json",
        {
            "checkpoint_label": label,
            "checkpoint": policy,
            "training_run": training_run,
            "checkpoint_step": checkpoint_step,
            # A short evaluation of one checkpoint among many (eval-steps), not a result to
            # put next to the teacher.
            "probe": bool(args.probe),
            "scene_set": scenes.name,
            "suite": suite,
            "student": student.key,
        },
    )
    argv = [
        "--suite", suite,
        "--tasks", args.tasks or ",".join(str(t) for t in experiment.tasks),
        "--scenes", args.scene_list or compact(scenes.seeds),
        "--seeds", args.seeds,
        "--init-mode", scenes.init_mode,
        "--video", args.video,
        "--out", str(out),
        "--student", student.key,
    ]  # fmt: skip
    if policy:
        argv += ["--policy", policy, "--label", label]
    if args.device:
        argv += ["--device", args.device]
    code = benchmark.main(argv)
    results = read_json(out / "results.json") or {}
    experiment.log(
        "student_evaluated", name=args.eval, checkpoint=label, student=student.key, suite=suite, scene_set=scenes.name,
        successes=results.get("successes"), episodes=results.get("episodes"), mean=results.get("mean"),
        failure_families=results.get("failure_families"), exit_status=code,
    )  # fmt: skip
    return code


def _alive(pid: object) -> bool:
    try:
        os.kill(int(pid), 0)  # type: ignore[arg-type]
    except (TypeError, ValueError, ProcessLookupError):
        return False
    except PermissionError:
        return True
    return True


def cmd_eval_steps(args: argparse.Namespace) -> int:
    """Evaluate every saved checkpoint of a training run, one after another; with --follow,
    keep waiting for new ones while the training runs. Each becomes an evaluation of its own,
    RUN@STEP-TAG, which a rerun skips once finished.
    """
    from cap_harness.trace2vla.train import saved_steps

    ensure_runtime_env()
    experiment = Experiment.open(args.name)
    run = experiment.training_dir(args.checkpoint)
    record = read_json(run / "run.json")
    if not record:
        print(f"error: no training run '{args.checkpoint}' in {experiment.root}", file=sys.stderr)
        return 2
    student = students.get(students.infer(record)[0])
    scenes = scene_set(experiment.settings, args.scenes)
    suite = args.suite or experiment.suite
    tag = args.tag or (scenes.name if suite == experiment.suite else f"{suite}-{scenes.name}")
    # Every evaluation below runs in this student's environment.
    students.reexec(student, "cap_harness.trace2vla", args.argv)
    wanted = set(parse_int_list(args.steps)) if args.steps else None
    done: set[int] = set()
    failed = 0
    print(
        f"evaluating checkpoints of '{run.name}' ({student.label}) on {suite}, scene set "
        f"{scenes.name}{' (following the training)' if args.follow else ''}",
        flush=True,
    )

    def pending() -> list[int]:
        # Step 0, the checkpoint the run started from, gives the curve its starting point on
        # the same scenes.
        steps = ([0] if args.start else []) + saved_steps(run)
        return [
            s
            for s in steps
            if s not in done
            and (wanted is None or s in wanted)
            and (not args.every or s % args.every == 0)
        ]

    while True:
        for step in pending():
            name = f"{run.name}@{step}-{tag}"
            out = experiment.eval_dir(name)
            progress = read_json(out / "progress.json") or {}
            if progress.get("status") != "finished":
                command = [
                    sys.executable, "-m", "cap_harness.trace2vla", "eval", str(experiment.root),
                    "--name", name, "--checkpoint", f"{run.name}@{step}",
                    "--scenes", scenes.name, "--seeds", args.seeds, "--video", args.video,
                    "--probe",
                ]  # fmt: skip
                for flag, value in (
                    ("--suite", args.suite),
                    ("--scene-list", args.scene_list),
                    ("--tasks", args.tasks),
                    ("--device", args.device),
                ):
                    if value:
                        command += [flag, value]
                print(f"step {step}: evaluating as '{name}'", flush=True)
                code = subprocess.call(command)
                if code != 0:
                    failed += 1
                    print(f"step {step}: evaluation exited with status {code}", flush=True)
            results = read_json(out / "results.json") or {}
            if results.get("episodes"):
                print(
                    f"step {step}: {results.get('successes')}/{results.get('episodes')} "
                    f"({100 * float(results.get('mean') or 0):.0f}%)  [{name}]",
                    flush=True,
                )
            done.add(step)
        training = read_json(run / "progress.json") or {}
        still_training = training.get("status") == "running" and _alive(training.get("pid"))
        if not args.follow or not still_training:
            # The last checkpoint may have landed after this round's listing.
            if args.follow and pending():
                continue
            break
        time.sleep(args.poll)
    if not done:
        print(
            f"no saved checkpoint of '{run.name}' matched (saved: {saved_steps(run) or 'none'}; "
            "train with --save-freq N to keep checkpoints during training)",
            file=sys.stderr,
        )
        return 1
    return 1 if failed else 0


def cmd_checkpoints(args: argparse.Namespace) -> int:
    """Where training checkpoints are kept; set that place; move existing ones there."""
    from cap_harness.trace2vla import checkpoints

    experiments = Experiment.open(args.name).root.parent if args.name else OUTPUTS
    if args.set_root is not None:
        root = None if args.set_root.lower() in ("", "none") else Path(args.set_root)
        chosen = checkpoints.set_root(experiments, root)
        if chosen is None:
            print("checkpoint root cleared: new training runs keep checkpoints in their run folder")
        else:
            free = shutil.disk_usage(chosen).free
            print(f"checkpoint root: {chosen} ({checkpoints.human(free)} free)")
            print("new training runs write their checkpoints there; their run folder links to them")
    root = Path(args.to) if args.to else checkpoints.checkpoint_root(experiments)
    source = (
        f"{checkpoints.ENV} environment variable"
        if checkpoints.ENV in os.environ
        else str(experiments / checkpoints.SETTING)
    )
    if args.set_root is None:
        print(
            f"checkpoint root: {root or 'none (checkpoints stay in each run folder)'}  [{source}]"
        )
    if not args.name:
        return 0
    experiment = Experiment.open(args.name)
    runs = sorted(p.parent for p in experiment.root.glob("training/*/run.json"))
    if not runs:
        print(f"no training runs in {experiment.root}")
        return 0
    if args.move and root is None:
        print(
            "error: no checkpoint root: set one with --set-root DIR (or pass --to DIR)",
            file=sys.stderr,
        )
        return 2
    failed = 0
    for run in runs:
        where = checkpoints.location(run)
        line = f"  {run.name:<24} {where['kind']:<8} {checkpoints.human(where['bytes']):>9}  {where['path'] or '-'}"
        if not args.move:
            print(line)
            continue
        result = checkpoints.move(run, root, dry_run=args.dry_run)  # type: ignore[arg-type]
        print(f"  {run.name:<24} {result['action']}: {result['detail']}", flush=True)
        if result["action"] == "error":
            failed += 1
        if result["action"] == "moved":
            experiment.log(
                "checkpoints_moved", run=run.name, to=result["to"], bytes=result["bytes"]
            )
    if args.move and args.dry_run:
        print("nothing moved (without --dry-run they would be)")
    return 1 if failed else 0


def cmd_grade_student(args: argparse.Namespace) -> int:
    """Label an existing student run: replay each episode's stored actions and grade it."""
    ensure_runtime_env()
    from cap_harness.trace2vla import replay
    from cap_harness.vla import benchmark
    from cap_harness.vla.protocol import references

    root = Path(args.run).resolve()
    run = read_json(root / "run.json")
    if not run:
        print(f"error: {root} is not a student benchmark run", file=sys.stderr)
        return 1
    settings = run["settings"]
    seeds = None if not args.seeds else set(parse_int_list(args.seeds))
    tasks = None if not args.tasks else set(parse_int_list(args.tasks))
    adapter = replay.osc_adapter(64, settings.get("init_mode", "saved"))
    graded = mismatched = 0
    try:
        for path in sorted((root / "episodes").glob("*/task_*/seed_*/scene_*/episode.json")):
            episode = read_json(path) or {}
            if episode.get("labels") and not args.again:
                continue
            if (seeds is not None and episode["policy_seed_index"] not in seeds) or (
                tasks is not None and episode["task_id"] not in tasks
            ):
                continue
            actions = np.load(path.parent / "steps.npz")["action"]
            scene = {
                "suite": episode["suite"],
                "task_id": episode["task_id"],
                "init_mode": settings.get("init_mode", "saved"),
                "seed": episode["scene"],
            }
            rollout = replay.open_loop(adapter, scene, actions)
            same = rollout.success == bool(episode["success"])
            mismatched += not same
            episode["labels"] = (
                rollout.labels if same else {"supported": False, "failure_family": "unknown"}
            )
            episode["regraded"] = {
                "replay_success": rollout.success,
                "replay_steps": rollout.steps,
                "matches_recording": same,
            }
            write_json(path, episode)
            graded += 1
            if graded % 25 == 0 or (args.limit and graded >= args.limit):
                print(
                    f"graded {graded} episodes ({mismatched} replays did not match their recording)",
                    flush=True,
                )
            if args.limit and graded >= args.limit:
                break
    finally:
        adapter.close()
    from cap_harness.vla import students

    title = f"{settings.get('policy')} on {settings['suite']}"
    student = students.infer(run)[0]
    result = benchmark.write_results(
        root,
        title,
        notes=references(settings["suite"], str(settings.get("policy")), student)["notes"],
    )
    print(f"graded {graded} episodes; {mismatched} replays did not match their recording")
    print(result["table"])
    return 0


# ---------------------------------------------------------------------- report / status --
def cmd_report(args: argparse.Namespace) -> int:
    from cap_harness.trace2vla import report

    experiment = Experiment.open(args.name)
    result = report.write(experiment)
    print(result["markdown"])
    print(f"written to {experiment.root / 'reports' / 'summary.md'}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    from cap_harness.trace2vla import report, teacher
    from cap_harness.vla.monitor import read_progress

    if not args.name:
        found = sorted(p.parent for p in OUTPUTS.glob("*/experiment.json"))
        if not found:
            print(f"no experiments under {OUTPUTS} (t2v.sh init NAME --suite SUITE --tasks 0,1)")
        for root in found:
            experiment = Experiment(root)
            last = (experiment.events() or [{}])[-1]
            print(
                f"{root.name:<32} {experiment.suite:<24} tasks {experiment.tasks}  last: {last.get('event', '-')} {last.get('time', '')}"
            )
        return 0
    experiment = Experiment.open(args.name)
    print(f"experiment {experiment.root}\n  suite {experiment.suite}, tasks {experiment.tasks}")
    registered = teacher.programs(experiment)
    print(
        f"teacher programs: tasks {sorted(registered) or 'none'}"
        + (
            ""
            if len(registered) == len(experiment.tasks)
            else f" (missing {[t for t in experiment.tasks if t not in registered]})"
        )
    )
    for name, scenes in scene_sets(experiment.settings).items():
        summary = teacher.results(experiment, name)
        planned = len(scenes.seeds) * len(experiment.tasks)
        graded = sum(t["graded"] for t in summary["tasks"])
        print(
            f"scene set {name:<10} ({scenes.role}, {scenes.init_mode} {compact(scenes.seeds)}): "
            + (
                f"teacher {summary['successes']}/{summary['episodes']} solved of {planned} planned, {graded} graded"
                if summary["episodes"]
                else "teacher not run"
            )
        )
    for demo in report.demo_sets(experiment):
        print(f"demo set {demo['name']}: {demo['accepted']} kept, {demo['rejected']} rejected")
    for dataset in report.datasets(experiment):
        print(
            f"dataset {dataset['name']} ({dataset['student_label']}): {dataset['episodes']} episodes, {dataset['frames']} frames, per task {dataset['per_task']}"
        )
    for run in report.trainings(experiment):
        progress = run["progress"]
        eta = progress.get("eta_s")
        print(
            f"training {run.get('name')} ({run['student_label']}): {progress.get('status')} step {progress.get('step')}/{progress.get('steps')} loss {progress.get('loss')}"
            + (f" ETA {eta / 60:.0f} min" if eta and progress.get("status") == "running" else "")
        )
    for path in sorted((experiment.root / "evals").glob("*/run.json")):
        progress = read_progress(path.parent) or {}
        rate = progress.get("success_rate")
        student = report.student_of(
            read_json(path.parent / "eval.json") or {}, read_json(path) or {}
        )
        print(
            f"eval {path.parent.name} ({students.label(student)}): {progress.get('status')} {progress.get('done')}/{progress.get('planned')}"
            + (f" success {100 * rate:.1f}%" if rate is not None else "")
            + (f" ({len(progress['warnings'])} warning(s))" if progress.get("warnings") else "")
        )
    events = experiment.events()
    print(f"last events ({len(events)} in the ledger):")
    for event in events[-6:]:
        detail = {
            k: v for k, v in event.items() if k not in ("time", "event", "host", "pid", "settings")
        }
        print(f"  {event['time']} {event['event']} {str(detail)[:150]}")
    return 0


# ------------------------------------------------------------------------------- parser --
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="t2v", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, handler: Any, needs_name: bool = True) -> argparse.ArgumentParser:
        command = sub.add_parser(name)
        if needs_name:
            command.add_argument(
                "name", help="experiment name (a folder under outputs/trace2vla) or path"
            )
        command.set_defaults(handler=handler)
        return command

    c = add("init", cmd_init)
    c.add_argument("--suite", required=True)
    c.add_argument("--tasks", default="all")
    c.add_argument("--train-seeds", help="seeded layouts for demonstrations (default 1001-1050)")
    c.add_argument("--dev-seeds", help="seeded layouts for the gate (default 2001-2010)")

    c = add("check", cmd_check)
    c.add_argument(
        "--no-services", action="store_true", help="do not fail when model services are down"
    )

    c = add("teacher-add", cmd_teacher_add)
    c.add_argument(
        "--from",
        dest="source",
        help="a session folder, a campaign folder, or a folder of task_<id>.py",
    )
    c.add_argument("--task", type=int)
    c.add_argument("--program")
    c.add_argument("--force", action="store_true", help="replace a task's program")

    c = add("teacher-import", cmd_teacher_import)
    c.add_argument(
        "--from", dest="source", required=True, help="a folder holding recorded ASPIRE run folders"
    )
    c.add_argument("--scenes", required=True, help="name of the scene set the episodes become")
    c.add_argument("--init-mode", choices=("saved", "seeded"))
    c.add_argument("--role", default="train", choices=("train", "dev", "test"))
    c.add_argument("--only-successes", action="store_true")

    c = add("teacher-run", cmd_teacher_run)
    c.add_argument("--scenes", required=True)
    c.add_argument("--tasks")
    c.add_argument("--workers", type=int, default=2)
    c.add_argument("--no-videos", action="store_true")
    c.add_argument("--force", action="store_true", help="run even when model services look down")

    c = add("teacher-grade", cmd_teacher_grade)
    c.add_argument("--scenes", required=True)
    c.add_argument("--tasks")

    c = add("convert", cmd_convert)
    c.add_argument("--scenes", required=True)
    c.add_argument("--demos", help="name of the demo set (default <scenes>-<speed>x)")
    c.add_argument("--speed", type=float, default=3.0, help="speed-up over the teacher (default 3)")
    c.add_argument(
        "--keep-idle", action="store_true", help="keep frames where the robot stands still"
    )
    c.add_argument("--tasks")

    c = add("dataset", cmd_dataset)
    c.add_argument("--demos")
    c.add_argument("--dataset", "--name", dest="dataset", required=True)
    c.add_argument(
        "--student",
        choices=sorted(students.STUDENTS),
        default=students.DEFAULT_STUDENT,
        help="whose normalisation statistics the dataset carries (default smolvla)",
    )
    c.add_argument(
        "--from-dataset",
        help="copy this dataset's frames instead of rendering the demos again (with --student)",
    )
    c.add_argument("--per-task", type=int, help="at most this many demos per task")
    c.add_argument("--tasks")

    c = add("train", cmd_train)
    c.add_argument("--dataset", required=True)
    c.add_argument("--run", "--name", dest="run", required=True)
    c.add_argument("--base", default="base", help="'base' or a training run to continue from")
    c.add_argument(
        "--student",
        choices=sorted(students.STUDENTS),
        help="default: the student the base run, else the dataset, is for",
    )
    c.add_argument("--method", default="full", choices=("full", "lora", "lora-vlm"))
    c.add_argument("--steps", type=int, default=3000)
    c.add_argument("--batch-size", type=int, default=32)
    c.add_argument("--lr", type=float)
    c.add_argument("--lora-rank", type=int, default=32)
    c.add_argument("--seed", type=int, default=1000)
    c.add_argument("--save-freq", type=int)
    c.add_argument("--log-freq", type=int, default=50)
    c.add_argument("--num-workers", type=int, default=4)
    c.add_argument("--device", default="cuda")

    c = add("eval", cmd_eval)
    c.add_argument("--eval", "--name", dest="eval", required=True)
    c.add_argument(
        "--checkpoint", default="base", help="'base', a training run name, or a checkpoint folder"
    )
    c.add_argument(
        "--student",
        choices=sorted(students.STUDENTS),
        help="whose base checkpoint 'base' means (default smolvla); a training run knows its own",
    )
    c.add_argument("--scenes", default="official")
    c.add_argument("--scene-list", help="override the scene numbers, e.g. 1-10")
    c.add_argument("--suite", help="another suite, e.g. libero_object for retention")
    c.add_argument("--seeds", default="0")
    c.add_argument("--tasks")
    c.add_argument("--video", default="all", choices=("all", "failures", "none"))
    c.add_argument("--device")
    c.add_argument("--probe", action="store_true", help=argparse.SUPPRESS)

    c = add("eval-steps", cmd_eval_steps)
    c.add_argument("--checkpoint", "--run", dest="checkpoint", required=True, help="a training run")
    c.add_argument("--follow", action="store_true", help="wait for new checkpoints while it trains")
    c.add_argument("--every", type=int, help="only steps that are a multiple of this")
    c.add_argument("--steps", help="only these steps, e.g. 500,1000")
    c.add_argument("--scenes", default="dev", help="scene set (default dev: kept for choosing)")
    c.add_argument("--scene-list", help="override the scene numbers, e.g. 1-10")
    c.add_argument("--suite", help="another suite, e.g. libero_object for retention")
    c.add_argument("--seeds", default="0")
    c.add_argument("--tasks")
    c.add_argument("--video", default="failures", choices=("all", "failures", "none"))
    c.add_argument("--device")
    c.add_argument("--tag", help="end of each evaluation's name (default: the scene set)")
    c.add_argument("--poll", type=float, default=30.0, help="seconds between looks (--follow)")
    c.add_argument(
        "--no-start",
        dest="start",
        action="store_false",
        help="skip step 0 (the checkpoint the run started from)",
    )

    c = add("checkpoints", cmd_checkpoints, needs_name=False)
    c.add_argument("name", nargs="?", help="an experiment: list (or --move) its runs' checkpoints")
    c.add_argument(
        "--set-root", help="where checkpoints go from now on (an absolute path; none clears)"
    )
    c.add_argument("--move", action="store_true", help="move existing checkpoints to the root")
    c.add_argument("--to", help="move to this folder instead of the root")
    c.add_argument("--dry-run", action="store_true", help="with --move: show, change nothing")

    c = add("grade-student", cmd_grade_student, needs_name=False)
    c.add_argument("run", help="a student benchmark run folder")
    c.add_argument("--seeds")
    c.add_argument("--tasks")
    c.add_argument("--limit", type=int)
    c.add_argument("--again", action="store_true", help="regrade episodes that already have labels")

    add("report", cmd_report)
    c = add("status", cmd_status, needs_name=False)
    c.add_argument("name", nargs="?")
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    args.argv = argv  # to continue in a student's own environment (students.reexec)
    try:
        return int(args.handler(args) or 0)
    except (FileNotFoundError, FileExistsError, KeyError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(
            "\nstopped. What finished is kept; run the same command again to continue.",
            file=sys.stderr,
        )
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
