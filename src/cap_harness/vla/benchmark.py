"""Benchmark a LeRobot policy on a LIBERO / LIBERO-Pro suite through ASPIRE's simulator.

    python -m cap_harness.vla.benchmark --suite libero_object_swap
    python -m cap_harness.vla.benchmark --suite libero_object --scenes 1-10 --seeds 0   # quick

Each (task, policy seed, scene) is one recorded episode (see ``rollout``).  Scenes are the
adapter's seeds: in the default ``saved`` init mode scene k is the task's saved init state
k-1, so ``--scenes 1-50`` is the official 50-scene test of LIBERO and LIBERO-Pro.  Policy
seeds only change the policy's own sampling noise (SmolVLA's flow matching); the scene and the
physics are deterministic.  The run directory is resumable: finished episodes are kept.

    <out>/run.json         settings, policy, versions
    <out>/episodes/<suite>/task_<id>/seed_<s>/scene_<k>/   one episode each
    <out>/benchmark.log    one timestamped line per episode, warnings, start and stop
    <out>/progress.json    status, done/planned, success so far, ETA, warnings (live)
    <out>/results.json     success per task and seed, means, spread across seeds (live)
    <out>/results.md       the same as a table, with the published numbers to compare against
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from datetime import datetime
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import signal
import statistics
import subprocess
import sys
from typing import Any

from cap_harness.vla import students
from cap_harness.vla.libero_compat import REPOSITORY_ROOT, ensure_runtime_env
from cap_harness.vla.monitor import RunMonitor, atomic_write, read_progress
from cap_harness.vla.protocol import (
    CONTROLLER,
    IMAGE_SIZE,
    N_ACTION_STEPS,
    SETTLE_STEPS,
    max_steps_for,
    parse_int_list,
    references,
    task_count,
    task_language,
)

PROTOCOL_KEYS = (
    "suite",
    "max_steps",
    "init_mode",
    "image_size",
    "controller",
    "settle_steps",
    "policy",
    "revision",
    "n_action_steps",
)
PACKAGES = ("lerobot", "torch", "transformers", "numpy", "mujoco", "robosuite", "libero")


def resolve_policy(
    student: str | None, policy: str | None, revision: str | None
) -> tuple[students.Student, str, str | None]:
    """(student, checkpoint source, revision) for a run. Without --policy, the student's own
    checkpoint at its pinned revision; a student's own Hub checkpoint is always pinned unless
    --revision says otherwise. Raises ValueError when --student and --policy disagree.
    """
    from cap_harness.vla.policy import student_for

    if policy is None:
        chosen = students.get(student)
        return chosen, chosen.checkpoint, revision or chosen.revision
    chosen = student_for(policy, student)
    return chosen, policy, revision or (chosen.revision if policy == chosen.checkpoint else None)


def policy_seed_for(seed: int, task_id: int, scene: int) -> int:
    """A distinct, reproducible noise seed for every episode."""
    return 1_000_000 * seed + 1_000 * task_id + scene


def episode_dir(root: Path, suite: str, task_id: int, seed: int, scene: int) -> Path:
    return root / "episodes" / suite / f"task_{task_id:02d}" / f"seed_{seed}" / f"scene_{scene:02d}"


def _git_commit(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def versions() -> dict[str, Any]:
    found: dict[str, Any] = {}
    for name in PACKAGES:
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            found[name] = None
    found["python"] = platform.python_version()
    found["aspire_commit"] = _git_commit(REPOSITORY_ROOT)
    found["lerobot_commit"] = _git_commit(REPOSITORY_ROOT / "third_party" / "lerobot")
    try:
        import torch

        if torch.cuda.is_available():
            found["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:  # pragma: no cover - torch is a vla dependency
        pass
    return found


def wilson(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials == 0:
        return (0.0, 0.0)
    p = successes / trials
    centre = (p + z * z / (2 * trials)) / (1 + z * z / trials)
    half = (
        z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / (1 + z * z / trials)
    )
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarize(episodes: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Success per task and policy seed, the suite mean per seed, and the spread across seeds."""
    counts: dict[tuple[str, int], dict[int, list[int]]] = defaultdict(
        lambda: defaultdict(lambda: [0, 0])
    )
    steps: dict[tuple[str, int], list[int]] = defaultdict(list)
    families: dict[tuple[str, int], Counter[str]] = defaultdict(Counter)
    for episode in episodes:
        key = (str(episode["suite"]), int(episode["task_id"]))
        seed = int(episode["policy_seed_index"])
        if not episode["success"] and episode.get("labels"):
            families[key][str(episode["labels"].get("failure_family", "unknown"))] += 1
        counts[key][seed][0] += int(bool(episode["success"]))
        counts[key][seed][1] += 1
        if episode["success"]:
            steps[key].append(int(episode["steps"]))
    seeds = sorted({seed for per_seed in counts.values() for seed in per_seed})
    tasks = []
    for (suite, task_id), per_seed in sorted(counts.items()):
        successes = sum(ok for ok, _ in per_seed.values())
        trials = sum(n for _, n in per_seed.values())
        low, high = wilson(successes, trials)
        tasks.append(
            {
                "suite": suite,
                "task_id": task_id,
                "successes": successes,
                "episodes": trials,
                "success_rate": successes / trials if trials else 0.0,
                "ci95": [low, high],
                "per_seed": {str(seed): ok / n for seed, (ok, n) in sorted(per_seed.items()) if n},
                "mean_steps_when_successful": statistics.mean(steps[(suite, task_id)])
                if steps[(suite, task_id)]
                else None,
                "failure_families": dict(families[(suite, task_id)]),
            }
        )
    per_seed_means = {}
    for seed in seeds:
        rates = [
            per_seed[seed][0] / per_seed[seed][1]
            for per_seed in counts.values()
            if per_seed.get(seed) and per_seed[seed][1]
        ]
        if rates:
            per_seed_means[str(seed)] = statistics.mean(rates)
    successes = sum(task["successes"] for task in tasks)
    trials = sum(task["episodes"] for task in tasks)
    means = list(per_seed_means.values())
    return {
        "tasks": tasks,
        "per_seed_mean": per_seed_means,
        "mean": statistics.mean(task["success_rate"] for task in tasks) if tasks else 0.0,
        "std_across_seeds": statistics.stdev(means) if len(means) > 1 else 0.0,
        "successes": successes,
        "episodes": trials,
        "ci95": list(wilson(successes, trials)),
        "failure_families": dict(sum((Counter(f) for f in families.values()), Counter())),
    }


FAMILY_LEGEND = (
    "Failures (offline grading from simulator state): semantic = another object, placed "
    "correctly; physical = the requested object, not placed; both = another object, not "
    "placed; no_attempt = the gripper never closed on an object."
)


def family_text(families: Mapping[str, int]) -> str:
    return ", ".join(
        f"{name} {count}" for name, count in sorted(families.items(), key=lambda kv: -kv[1])
    )


def format_table(
    summary: Mapping[str, Any], title: str, notes: Iterable[str] = (), planned: int | None = None
) -> str:
    seeds = list(summary["per_seed_mean"])
    lines = [f"## {title}", ""]
    if planned is not None and summary["episodes"] < planned:
        lines += [f"_In progress: {summary['episodes']} of {planned} episodes._", ""]
    header = "| task | success | 95% interval | " + " | ".join(f"seed {s}" for s in seeds)
    lines += [header + " | mean steps (successes) | failures |", "|" + "---|" * (5 + len(seeds))]
    for task in summary["tasks"]:
        per_seed = " | ".join(f"{100 * task['per_seed'].get(s, 0):.0f}%" for s in seeds)
        steps = task["mean_steps_when_successful"]
        lines.append(
            f"| {task['suite']}:{task['task_id']} | {task['successes']}/{task['episodes']} "
            f"({100 * task['success_rate']:.1f}%) | {100 * task['ci95'][0]:.0f}-{100 * task['ci95'][1]:.0f}% | "
            f"{per_seed} | {'-' if steps is None else f'{steps:.0f}'} | "
            f"{family_text(task.get('failure_families') or {})} |"
        )
    per_seed = " | ".join(f"{100 * summary['per_seed_mean'][s]:.1f}%" for s in seeds)
    lines.append(
        f"| **mean** | {summary['successes']}/{summary['episodes']} "
        f"(**{100 * summary['mean']:.1f}%** ± {100 * summary['std_across_seeds']:.1f} across seeds) | "
        f"{100 * summary['ci95'][0]:.0f}-{100 * summary['ci95'][1]:.0f}% | {per_seed} | | "
        f"{family_text(summary.get('failure_families') or {})} |"
    )
    if summary.get("failure_families"):
        lines += ["", FAMILY_LEGEND]
    notes = list(notes)
    if notes:
        lines += ["", "Published numbers to compare: " + "; ".join(notes) + "."]
    return "\n".join(lines) + "\n"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def load_episodes(root: Path) -> list[dict[str, Any]]:
    episodes = []
    for path in sorted((root / "episodes").glob("*/task_*/seed_*/scene_*/episode.json")):
        try:
            episodes.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return episodes


def write_results(
    root: Path,
    title: str,
    episodes: list[Mapping[str, Any]] | None = None,
    notes: Iterable[str] = (),
    planned: int | None = None,
) -> dict[str, Any]:
    summary = summarize(load_episodes(root) if episodes is None else episodes)
    summary["references"] = list(notes)
    summary["planned"] = planned
    atomic_write(root / "results.json", json.dumps(summary, indent=2) + "\n")
    table = format_table(summary, title, notes, planned)
    atomic_write(root / "results.md", table)
    return {"summary": summary, "table": table}


def _interrupt_on_sigterm() -> None:
    """Let `kill` (and vla.sh stop) end a run like Ctrl-C: with a final status written."""

    def handler(signum: int, frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, handler)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--suite", required=True, help="LIBERO or LIBERO-Pro suite, e.g. libero_object_swap"
    )
    parser.add_argument("--tasks", default="all", help='task ids, e.g. "0,3,5-9" (default: all)')
    parser.add_argument(
        "--scenes", default="1-50", help="scene numbers (adapter seeds), default 1-50"
    )
    parser.add_argument("--seeds", default="0,1,2", help="policy noise seeds, default 0,1,2")
    parser.add_argument(
        "--student",
        choices=sorted(students.STUDENTS),
        default=None,
        help="which student (default: the one --policy points to, else smolvla); "
        "each runs in its own environment",
    )
    parser.add_argument(
        "--policy",
        default=None,
        help="Hub id or local checkpoint (default: the student's own, e.g. lerobot/smolvla_libero)",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Hub revision (default: the pinned one for a student's own checkpoint)",
    )
    parser.add_argument("--device", default=None, help="cuda or cpu (default: cuda when available)")
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=N_ACTION_STEPS,
        help=f"actions executed per predicted chunk (default {N_ACTION_STEPS}, see protocol.py; "
        "0 = the checkpoint's own value)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="episode limit (default: LeRobot's for the suite family)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=IMAGE_SIZE,
        help="camera render size (default 256, the training data's)",
    )
    parser.add_argument(
        "--init-mode",
        choices=("saved", "seeded"),
        default="saved",
        help="saved: the official scenes",
    )
    parser.add_argument("--video", choices=("all", "failures", "none"), default="all")
    parser.add_argument(
        "--out", type=Path, default=None, help="run directory (default outputs/vla/<time>-<suite>)"
    )
    parser.add_argument(
        "--label", default=None, help="a short name for the policy in titles (default: --policy)"
    )
    parser.add_argument(
        "--no-grading",
        action="store_true",
        help="skip the offline failure labels (which object was lifted and placed)",
    )
    parser.add_argument(
        "--report", action="store_true", help="only rebuild results from an existing --out"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    if args.n_action_steps is not None and args.n_action_steps <= 0:
        args.n_action_steps = None  # the checkpoint's own value
    ensure_runtime_env()
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    root = args.out or REPOSITORY_ROOT / "outputs" / "vla" / f"{stamp}-{args.suite}"
    if args.report and args.policy is None and args.student is None:
        # Rebuilding an existing run's table: its own policy and student, not the defaults.
        recorded = _read_json(root / "run.json")
        args.policy = (recorded.get("settings") or {}).get("policy")
        args.student = students.infer(recorded)[0] if recorded else None
    try:
        student, source, revision = resolve_policy(args.student, args.policy, args.revision)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    title = f"{args.label or source} on {args.suite}"
    reference = references(args.suite, source, student.key)
    notes = reference["notes"]
    if args.report:
        print(write_results(root, title, notes=notes)["table"])
        return 0
    # Each student runs in its own environment (cap_harness.vla.students).
    students.reexec(student, "cap_harness.vla.benchmark", argv)

    tasks = parse_int_list(args.tasks, default=range(task_count(args.suite)))
    scenes = parse_int_list(args.scenes)
    seeds = parse_int_list(args.seeds)
    max_steps = args.max_steps or max_steps_for(args.suite)
    root.mkdir(parents=True, exist_ok=True)

    from cap_harness.libero.adapter import LiberoAdapter
    from cap_harness.vla.grading import LiberoGrader
    from cap_harness.vla.policy import LeRobotLiberoPolicy, seed_everything
    from cap_harness.vla.rollout import EpisodeRecorder, EpisodeSpec, run_episode

    settings = {
        "suite": args.suite,
        "tasks": tasks,
        "scenes": scenes,
        "policy_seeds": seeds,
        "max_steps": max_steps,
        "init_mode": args.init_mode,
        "image_size": args.image_size,
        "controller": CONTROLLER,
        "settle_steps": SETTLE_STEPS,
        "policy": source,
        "revision": revision,
        "n_action_steps": args.n_action_steps,
        "video": args.video,
        # Not part of the protocol: the policy already is, and older runs have no student field.
        "student": student.key,
    }
    # Everything that changes an episode's outcome; a run directory holds one protocol only.
    protocol = {key: settings[key] for key in PROTOCOL_KEYS}
    running = read_progress(root)
    if running and running.get("status") == "running" and running.get("pid") != os.getpid():
        print(
            f"error: run {running.get('pid')} on {running.get('host')} is still writing to {root} "
            "(scripts/gpu/vla.sh stop to end it)",
            file=sys.stderr,
        )
        return 2
    run_path = root / "run.json"
    previous = json.loads(run_path.read_text(encoding="utf-8")) if run_path.is_file() else None
    if previous and previous.get("protocol") != protocol:
        print(
            f"error: {root} holds a run with a different protocol; use another --out\n"
            f"  there: {previous.get('protocol')}\n  now:   {protocol}",
            file=sys.stderr,
        )
        return 2

    policy = LeRobotLiberoPolicy(
        source,
        revision=revision,
        device=args.device,
        n_action_steps=args.n_action_steps,
        image_size=args.image_size,
        suite=args.suite,
        student=student.key,
    )
    run_path.write_text(
        json.dumps(
            {
                "created": (previous or {}).get("created", stamp),
                "updated": stamp,
                "command": sys.argv,
                "settings": settings,
                "protocol": protocol,
                "student": student.describe(),
                "policy": policy.info.to_dict(),
                "versions": versions(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    adapter = LiberoAdapter(
        controller=CONTROLLER,
        camera_height=args.image_size,
        camera_width=args.image_size,
        init_mode=args.init_mode,
        post_settle_steps=SETTLE_STEPS,
    )
    # Offline labels from simulator state (which object was lifted and placed): grading
    # only, the policy never sees it.
    grader = None if args.no_grading else LiberoGrader(adapter)
    plan = [
        (task_id, seed_index, scene)
        for task_id in tasks
        for seed_index in seeds
        for scene in scenes
    ]
    finished = {}
    for task_id, seed_index, scene in plan:
        path = episode_dir(root, args.suite, task_id, seed_index, scene) / "episode.json"
        if path.is_file():
            finished[(task_id, seed_index, scene)] = json.loads(path.read_text(encoding="utf-8"))
    monitor = RunMonitor(
        root,
        title=title,
        planned=len(plan),
        resumed=list(finished.values()),
        references=reference,
        student=student.describe(),
        write_results=lambda episodes: write_results(root, title, episodes, notes, len(plan)),
    )
    monitor.log(
        f"start: {title}: {len(tasks)} tasks x {len(seeds)} seeds x {len(scenes)} scenes = "
        f"{len(plan)} episodes ({len(finished)} already done), limit {max_steps} steps, "
        f"GPU {os.environ.get('CUDA_VISIBLE_DEVICES', 'default')}, results in {root}"
    )
    for note in notes:
        monitor.log(f"compare with: {note}")
    _interrupt_on_sigterm()
    status, error = "finished", None
    instructions: dict[int, str] = {}
    try:
        monitor.write()
        for task_id, seed_index, scene in plan:
            if (task_id, seed_index, scene) in finished:
                continue
            if task_id not in instructions:
                instructions[task_id] = task_language(args.suite, task_id)
            spec = EpisodeSpec(
                suite=args.suite,
                task_id=task_id,
                scene=scene,
                policy_seed=policy_seed_for(seed_index, task_id, scene),
                max_steps=max_steps,
                instruction=instructions[task_id],
                seed_index=seed_index,
            )
            monitor.start({"task_id": task_id, "policy_seed_index": seed_index, "scene": scene})
            directory = episode_dir(root, args.suite, task_id, seed_index, scene)
            recorder = EpisodeRecorder(directory, video=args.video)
            summary = run_episode(
                adapter, policy, spec, recorder=recorder, seed_fn=seed_everything, grader=grader
            )
            monitor.add(summary)
    except KeyboardInterrupt:
        status = "interrupted"
    except Exception as exc:
        status, error = "failed", f"{type(exc).__name__}: {exc}"
        monitor.log(f"ERROR: {error}")
        raise
    finally:
        adapter.close()
        monitor.finish(status, error)
        print("\n" + (root / "results.md").read_text(encoding="utf-8"))
        print(f"run directory: {root}")
    return 0 if status == "finished" else 130


if __name__ == "__main__":
    raise SystemExit(main())
