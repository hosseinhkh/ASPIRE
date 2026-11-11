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
    <out>/results.json     success per task and seed, means, spread across seeds
    <out>/results.md       the same as a table
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import datetime
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Any

from cap_harness.vla.libero_compat import REPOSITORY_ROOT, ensure_runtime_env
from cap_harness.vla.protocol import (
    CONTROLLER,
    IMAGE_SIZE,
    SETTLE_STEPS,
    max_steps_for,
    parse_int_list,
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
    for episode in episodes:
        key = (str(episode["suite"]), int(episode["task_id"]))
        seed = int(episode["policy_seed_index"])
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
    }


def format_table(summary: Mapping[str, Any], title: str) -> str:
    seeds = list(summary["per_seed_mean"])
    lines = [f"## {title}", ""]
    header = "| task | success | 95% interval | " + " | ".join(f"seed {s}" for s in seeds)
    lines += [header + " | mean steps (successes) |", "|" + "---|" * (4 + len(seeds))]
    for task in summary["tasks"]:
        per_seed = " | ".join(f"{100 * task['per_seed'].get(s, 0):.0f}%" for s in seeds)
        steps = task["mean_steps_when_successful"]
        lines.append(
            f"| {task['suite']}:{task['task_id']} | {task['successes']}/{task['episodes']} "
            f"({100 * task['success_rate']:.1f}%) | {100 * task['ci95'][0]:.0f}-{100 * task['ci95'][1]:.0f}% | "
            f"{per_seed} | {'-' if steps is None else f'{steps:.0f}'} |"
        )
    per_seed = " | ".join(f"{100 * summary['per_seed_mean'][s]:.1f}%" for s in seeds)
    lines.append(
        f"| **mean** | {summary['successes']}/{summary['episodes']} "
        f"(**{100 * summary['mean']:.1f}%** ± {100 * summary['std_across_seeds']:.1f} across seeds) | "
        f"{100 * summary['ci95'][0]:.0f}-{100 * summary['ci95'][1]:.0f}% | {per_seed} | |"
    )
    return "\n".join(lines) + "\n"


def load_episodes(root: Path) -> list[dict[str, Any]]:
    episodes = []
    for path in sorted((root / "episodes").glob("*/task_*/seed_*/scene_*/episode.json")):
        try:
            episodes.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return episodes


def write_results(root: Path, title: str) -> dict[str, Any]:
    summary = summarize(load_episodes(root))
    (root / "results.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    table = format_table(summary, title)
    (root / "results.md").write_text(table, encoding="utf-8")
    return {"summary": summary, "table": table}


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
        "--policy", default=None, help="Hub id or local checkpoint (default lerobot/smolvla_libero)"
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Hub revision (default: the pinned one for the default policy)",
    )
    parser.add_argument("--device", default=None, help="cuda or cpu (default: cuda when available)")
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=None,
        help="actions executed per chunk (default: checkpoint's)",
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
        "--report", action="store_true", help="only rebuild results from an existing --out"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ensure_runtime_env()
    from cap_harness.vla.policy import DEFAULT_POLICY, DEFAULT_REVISION

    source = args.policy or DEFAULT_POLICY
    revision = args.revision or (DEFAULT_REVISION if source == DEFAULT_POLICY else None)
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    root = args.out or REPOSITORY_ROOT / "outputs" / "vla" / f"{stamp}-{args.suite}"
    title = f"{source} on {args.suite}"
    if args.report:
        print(write_results(root, title)["table"])
        return 0

    tasks = parse_int_list(args.tasks, default=range(task_count(args.suite)))
    scenes = parse_int_list(args.scenes)
    seeds = parse_int_list(args.seeds)
    max_steps = args.max_steps or max_steps_for(args.suite)
    root.mkdir(parents=True, exist_ok=True)

    from cap_harness.libero.adapter import LiberoAdapter
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
    }
    # Everything that changes an episode's outcome; a run directory holds one protocol only.
    protocol = {key: settings[key] for key in PROTOCOL_KEYS}
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
    )
    run_path.write_text(
        json.dumps(
            {
                "created": (previous or {}).get("created", stamp),
                "updated": stamp,
                "command": sys.argv,
                "settings": settings,
                "protocol": protocol,
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
    total = len(tasks) * len(seeds) * len(scenes)
    done = 0
    started = time.monotonic()
    print(f"{title}: {len(tasks)} tasks x {len(seeds)} seeds x {len(scenes)} scenes = {total} episodes, "
          f"limit {max_steps} steps, results in {root}", flush=True)  # fmt: skip
    try:
        for task_id in tasks:
            instruction = task_language(args.suite, task_id)
            for seed_index in seeds:
                for scene in scenes:
                    done += 1
                    directory = episode_dir(root, args.suite, task_id, seed_index, scene)
                    if (directory / "episode.json").is_file():
                        continue
                    spec = EpisodeSpec(
                        suite=args.suite,
                        task_id=task_id,
                        scene=scene,
                        policy_seed=policy_seed_for(seed_index, task_id, scene),
                        max_steps=max_steps,
                        instruction=instruction,
                        seed_index=seed_index,
                    )
                    recorder = EpisodeRecorder(directory, video=args.video)
                    summary = run_episode(
                        adapter, policy, spec, recorder=recorder, seed_fn=seed_everything
                    )
                    elapsed = time.monotonic() - started
                    print(
                        f"[{done}/{total}] {spec.task_ref} seed {seed_index} scene {scene}: "
                        f"{'success' if summary['success'] else 'fail'} in {summary['steps']} steps "
                        f"({summary['wall_s']:.1f} s; {elapsed / 60:.1f} min so far)",
                        flush=True,
                    )
    finally:
        adapter.close()
        result = write_results(root, title)
        print("\n" + result["table"])
        print(f"run directory: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
