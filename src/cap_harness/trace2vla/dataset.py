"""Demonstrations -> a LeRobot dataset the student can be fine-tuned on.

* **Same shape as ``lerobot/libero``** (the data the checkpoint was trained on): cameras
  ``observation.images.image`` / ``image2`` at 256x256, an 8-value ``observation.state``, a
  7-value ``action``, 10 fps declared, robot type ``panda``. The two can be merged for replay.
* **Frames come from a fresh replay** of each demo's stored actions, so the dataset holds
  lossless frames and every episode is checked once more: it must still solve the task and
  reproduce the stored states.
* **Normalisation statistics are the base checkpoint's.** lerobot-train replaces a
  checkpoint's statistics with the dataset's when fine-tuning; with the checkpoint's own
  statistics written into the dataset, every update keeps the scaling the weights expect. The
  statistics computed from the new data are kept beside them (``meta/stats.computed.json``).
* **A dataset is for one student** (``manifest.json`` names it): the frames are the same for
  every student, the statistics are not. ``derive_dataset`` copies a dataset and swaps in
  another student's statistics without rendering the frames again.
* ``manifest.json`` lists every episode with its demo, its teacher run and their hashes.
"""

from __future__ import annotations

from collections.abc import Iterable
import json
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from cap_harness.trace2vla import replay
from cap_harness.trace2vla.store import now_iso, read_json, sha256_file, write_json
from cap_harness.vla import students
from cap_harness.vla.protocol import IMAGE_SIZE

FPS = 10  # as lerobot/libero declares (frames are consecutive 20 Hz control steps)
ROBOT_TYPE = "panda"
FROZEN_FEATURES = ("observation.state", "action")


def features(image_size: int = IMAGE_SIZE) -> dict[str, dict[str, Any]]:
    image = {
        "dtype": "video",
        "shape": (image_size, image_size, 3),
        "names": ["height", "width", "channel"],
    }
    return {
        "observation.images.image": dict(image),
        "observation.images.image2": dict(image),
        "observation.state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
        "action": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
    }


def demo_dirs(
    demo_root: Path, tasks: Iterable[int] | None = None, per_task: int | None = None
) -> list[Path]:
    """Accepted demos under a demo set, by task then seed; at most ``per_task`` of each."""
    wanted = None if tasks is None else {int(t) for t in tasks}
    chosen: list[Path] = []
    for task_dir in sorted(demo_root.glob("task_*")):
        task_id = int(task_dir.name.split("_")[1])
        if wanted is not None and task_id not in wanted:
            continue
        episodes = [
            p.parent
            for p in sorted(task_dir.glob("seed_*/episode.json"))
            if (read_json(p) or {}).get("accepted")
        ]
        chosen += episodes if per_task is None else episodes[:per_task]
    return chosen


def checkpoint_stats(checkpoint: Path) -> dict[str, dict[str, list[float]]]:
    """The normalisation statistics stored with a LeRobot checkpoint, per feature."""
    from safetensors.numpy import load_file

    files = sorted(
        Path(checkpoint).glob("policy_preprocessor_step_*_normalizer_processor.safetensors")
    )
    if not files:
        raise FileNotFoundError(f"no normalizer statistics in {checkpoint}")
    stats: dict[str, dict[str, list[float]]] = {}
    for key, value in load_file(str(files[0])).items():
        feature, _, name = key.rpartition(".")
        stats.setdefault(feature, {})[name] = np.asarray(value, dtype=np.float64).tolist()
    return stats


#: Largest difference between the states of the dataset's replay and the demo's stored states,
#: in metres and radians. The frames written always come from the dataset's own replay of the
#: stored actions, so they match those actions exactly; this only checks that the replay is the
#: same demo. Float32 rounding of the actions moves a replay by up to about 1e-4 after contact
#: (5e-5 measured on demos converted before the converter replayed the float32 actions); a wrong
#: scene, controller or code version moves it by centimetres or fails the task.
STATE_DRIFT_LIMIT = 1e-3


def replay_problem(success: bool, frames: int, expected: int, drift: float) -> str | None:
    """Why the dataset's replay of a demo cannot be used, or None when it can."""
    if not success:
        return "the replay did not solve the task"
    if frames != expected:
        return f"the replay solved the task after {frames} of {expected} steps"
    if not drift <= STATE_DRIFT_LIMIT:  # also catches NaN
        return f"state drift {drift:.2e} > {STATE_DRIFT_LIMIT:g}"
    return None


def freeze_stats(dataset_root: Path, checkpoint: Path) -> dict[str, Any]:
    """Write the checkpoint's state and action statistics into the dataset.

    The statistics computed from the data are kept in ``meta/stats.computed.json`` the first
    time and read from there afterwards, so freezing again (for another student) starts from
    the data's own statistics, not from the previous student's.
    """
    path = dataset_root / "meta" / "stats.json"
    original = dataset_root / "meta" / "stats.computed.json"
    if original.is_file():
        computed = json.loads(original.read_text(encoding="utf-8"))
    else:
        computed = json.loads(path.read_text(encoding="utf-8"))
        write_json(original, computed)
    base = checkpoint_stats(checkpoint)
    frozen = dict(computed)
    for feature in FROZEN_FEATURES:
        if feature not in base:
            raise KeyError(f"the checkpoint has no statistics for {feature}")
        frozen[feature] = {**computed.get(feature, {}), **base[feature]}
    path.write_text(json.dumps(frozen, indent=4) + "\n", encoding="utf-8")
    return {
        feature: {
            "checkpoint_mean": base[feature]["mean"],
            "checkpoint_std": base[feature]["std"],
            "data_mean": computed.get(feature, {}).get("mean"),
            "data_std": computed.get(feature, {}).get("std"),
        }
        for feature in FROZEN_FEATURES
    }


def build_dataset(
    demos: list[Path],
    out_dir: Path,
    *,
    name: str,
    checkpoint: Path | None,
    image_size: int = IMAGE_SIZE,
    vcodec: str = "libsvtav1",
    log: Any = print,
    student: str = students.DEFAULT_STUDENT,
    allow_empty: bool = False,
) -> dict[str, Any]:
    """Replay each demo and write it as one dataset episode. Returns the manifest.

    ``allow_empty``: when no demo could be written, return a manifest with no episodes (and
    the rejections) instead of raising; no dataset is left on disk then. Used for shards.
    """
    chosen = students.get(student)
    import datasets as hf_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    hf_datasets.disable_progress_bars()  # one bar per episode otherwise
    os.environ.setdefault("SVT_LOG", "1")  # the AV1 encoder prints its configuration per video
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} already exists (choose another dataset name)")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset = LeRobotDataset.create(
        repo_id=f"trace2vla/{name}",
        fps=FPS,
        features=features(image_size),
        root=out_dir,
        robot_type=ROBOT_TYPE,
        use_videos=True,
        vcodec=vcodec,
    )
    adapter = replay.osc_adapter(image_size)
    episodes: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    try:
        for demo in demos:
            summary = read_json(demo / "episode.json") or {}
            stored = np.load(demo / "steps.npz")
            count = 0

            def add(sample: dict[str, Any], instruction: str = summary["instruction"]) -> None:
                nonlocal count
                dataset.add_frame(
                    {
                        "observation.images.image": sample["image"],
                        "observation.images.image2": sample["image2"],
                        "observation.state": sample["state"].astype(np.float32),
                        "action": sample["action"].astype(np.float32),
                        "task": instruction,
                    }
                )
                count += 1

            states: list[np.ndarray] = []
            rollout = replay.open_loop(
                adapter,
                summary["scene"],
                stored["action"],
                on_frame=lambda sample, add=add, states=states: (
                    states.append(sample["state"]),
                    add(sample),
                ),
            )
            drift = (
                float(np.abs(np.stack(states) - stored["state"][: len(states)]).max())
                if states
                else float("inf")
            )
            problem = replay_problem(rollout.success, count, len(stored["action"]), drift)
            if problem:
                dataset.clear_episode_buffer()
                rejected.append(
                    {
                        "demo": str(demo),
                        "reason": problem,
                        "success": rollout.success,
                        "frames": count,
                        "state_drift": drift,
                    }
                )
                log(f"  rejected {demo.parent.name}/{demo.name}: {problem}")
                continue
            dataset.save_episode()
            episodes.append(
                {
                    "episode_index": len(episodes),
                    "demo": str(demo),
                    "frames": count,
                    "task_id": summary["scene"]["task_id"],
                    "scene": summary["scene"],
                    "instruction": summary["instruction"],
                    "source_run": summary["source_run"],
                    "source_steps_sha256": summary.get("source_steps_sha256"),
                    "actions_sha256": sha256_file(demo / "steps.npz"),
                    "state_drift": drift,
                }
            )
            log(f"  [{len(episodes)}/{len(demos)}] {demo.parent.name}/{demo.name}: {count} frames")
    finally:
        adapter.close()
    if not episodes:
        shutil.rmtree(out_dir, ignore_errors=True)
        if allow_empty:
            return {"name": name, "episodes": [], "rejected": rejected, "frames": 0}
        raise RuntimeError("no demo could be written to the dataset")
    dataset.finalize()
    frozen = freeze_stats(out_dir, checkpoint) if checkpoint is not None else None
    manifest = {
        "name": name,
        "created": now_iso(),
        "student": chosen.describe(),
        "repo_id": f"trace2vla/{name}",
        "root": str(out_dir),
        "fps": FPS,
        "image_size": image_size,
        "episodes": episodes,
        "rejected": rejected,
        "frames": sum(e["frames"] for e in episodes),
        "per_task": {
            str(t): sum(1 for e in episodes if e["task_id"] == t)
            for t in sorted({e["task_id"] for e in episodes})
        },
        "statistics": {
            "source": str(checkpoint) if checkpoint else "computed from this dataset",
            "student": chosen.key,
            "frozen": frozen,
        },
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def build_shard(
    index: int,
    demos: list[str],
    shard_dir: str,
    name: str,
    image_size: int,
    vcodec: str,
    student: str,
) -> dict[str, Any]:
    """Build one shard of a dataset (a worker's job): the demos as an ordinary dataset, with the
    statistics the data gives (they are frozen once, on the merged dataset).
    """
    return build_dataset(
        [Path(d) for d in demos],
        Path(shard_dir),
        name=f"{name}-part{index}",
        checkpoint=None,
        image_size=image_size,
        vcodec=vcodec,
        log=lambda line: print(f"[part {index}] {line.strip()}", flush=True),
        student=student,
        allow_empty=True,
    )


def build_dataset_parallel(
    demos: list[Path],
    out_dir: Path,
    *,
    name: str,
    checkpoint: Path | None,
    workers: int,
    image_size: int = IMAGE_SIZE,
    vcodec: str = "libsvtav1",
    log: Any = print,
    student: str = students.DEFAULT_STUDENT,
) -> dict[str, Any]:
    """``build_dataset`` with the demos split over ``workers`` processes.

    Each worker writes a shard (consecutive demos, so the order is the same as one process
    would give); LeRobot's ``aggregate_datasets`` merges the shards into ``out_dir``, and the
    checkpoint's statistics are frozen into the result as ``build_dataset`` does. Same
    manifest, with the episodes numbered in order. With one worker it is ``build_dataset``.
    """
    from cap_harness.trace2vla import parallel

    if workers <= 1 or len(demos) <= 1:
        return build_dataset(
            demos, out_dir, name=name, checkpoint=checkpoint, image_size=image_size,
            vcodec=vcodec, log=log, student=student,
        )  # fmt: skip
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} already exists (choose another dataset name)")
    chosen = students.get(student)
    shards_root = out_dir.parent / f".{out_dir.name}.shards"
    shutil.rmtree(shards_root, ignore_errors=True)  # left by an interrupted build
    shards_root.mkdir(parents=True)
    size = -(-len(demos) // workers)  # consecutive groups, the last one shorter
    groups = parallel.chunks([str(d) for d in demos], size)
    jobs = [
        (index, group, str(shards_root / f"part{index}"), name, image_size, vcodec, chosen.key)
        for index, group in enumerate(groups)
    ]
    log(f"building {len(demos)} demos in {len(jobs)} parts at once")
    manifests: dict[int, dict[str, Any]] = {}
    errors: list[str] = []
    try:
        parallel.run_jobs(
            build_shard,
            jobs,
            workers=len(jobs),
            on_result=lambda job, manifest: manifests.__setitem__(job[0], manifest),
            on_error=lambda job, error: errors.append(
                f"part {job[0]}: {type(error).__name__}: {error}"
            ),
        )
        if errors:
            raise RuntimeError("building the dataset failed: " + "; ".join(errors))
        written = [i for i in sorted(manifests) if manifests[i]["episodes"]]
        if not written:
            raise RuntimeError("no demo could be written to the dataset")
        from lerobot.datasets.aggregate import aggregate_datasets

        log(f"merging {len(written)} parts into {out_dir}")
        aggregate_datasets(
            repo_ids=[f"trace2vla/{name}-part{i}" for i in written],
            aggr_repo_id=f"trace2vla/{name}",
            roots=[shards_root / f"part{i}" for i in written],
            aggr_root=out_dir,
        )
    except BaseException:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(shards_root, ignore_errors=True)
    episodes: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for index in sorted(manifests):
        for episode in manifests[index]["episodes"]:
            episodes.append({**episode, "episode_index": len(episodes)})
        rejected.extend(manifests[index]["rejected"])
    frozen = freeze_stats(out_dir, checkpoint) if checkpoint is not None else None
    manifest = {
        "name": name,
        "created": now_iso(),
        "student": chosen.describe(),
        "repo_id": f"trace2vla/{name}",
        "root": str(out_dir),
        "fps": FPS,
        "image_size": image_size,
        "episodes": episodes,
        "rejected": rejected,
        "frames": sum(e["frames"] for e in episodes),
        "per_task": {
            str(t): sum(1 for e in episodes if e["task_id"] == t)
            for t in sorted({e["task_id"] for e in episodes})
        },
        "statistics": {
            "source": str(checkpoint) if checkpoint else "computed from this dataset",
            "student": chosen.key,
            "frozen": frozen,
        },
        "built": {"workers": len(jobs), "parts": len(written)},
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def derive_dataset(
    source_root: Path,
    out_dir: Path,
    *,
    name: str,
    checkpoint: Path,
    student: str,
) -> dict[str, Any]:
    """Copy a dataset for another student: same frames and actions, that student's statistics.

    Rendering 98 demos takes 10-20 minutes; this takes seconds. The new manifest names the
    dataset it came from and that dataset's manifest hash.
    """
    chosen = students.get(student)
    source_manifest = read_json(source_root / "manifest.json")
    if not source_manifest:
        raise FileNotFoundError(f"no dataset at {source_root} (t2v.sh dataset)")
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} already exists (choose another dataset name)")
    shutil.copytree(source_root, out_dir)
    try:
        frozen = freeze_stats(out_dir, checkpoint)
    except Exception:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise
    manifest = {
        **source_manifest,
        "name": name,
        "created": now_iso(),
        "student": chosen.describe(),
        "repo_id": f"trace2vla/{name}",
        "root": str(out_dir),
        "derived_from": {
            "name": source_manifest.get("name", source_root.name),
            "manifest_sha256": sha256_file(source_root / "manifest.json"),
            "student": students.infer(source_manifest)[0],
        },
        "statistics": {"source": str(checkpoint), "student": chosen.key, "frozen": frozen},
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest
