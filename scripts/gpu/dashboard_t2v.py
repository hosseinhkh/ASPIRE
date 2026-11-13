"""Dashboard data for Trace2VLA experiments (outputs/trace2vla/<name>/).

Reads the files the pipeline stages write (cap_harness.trace2vla): experiment.json, the ledger,
teacher results and grades, demo summaries, dataset manifests, training runs and student
evaluation runs. Standard library only, like the rest of the dashboard.
"""

from __future__ import annotations

import json
from pathlib import Path
import statistics
from typing import Any

from dashboard_store import Store, downsample, read_json, read_npz
from dashboard_vla import EPISODE_SERIES, progress_of, student_of

from cap_harness.trace2vla.train import repair_steps

LEDGER_TAIL = 300
LOG_TAIL = 200


def _events(root: Path) -> list[dict[str, Any]]:
    path = root / "ledger.jsonl"
    events = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                events.append(json.loads(line))
            except ValueError:
                continue
    return events


def _tail(path: Path, lines: int = LOG_TAIL) -> list[str]:
    if not path.is_file():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]


class T2vStore:
    def __init__(self, store: Store) -> None:
        self.store = store

    @property
    def root(self) -> Path:
        return self.store.outputs / "trace2vla"

    def _experiment(self, path: Path) -> Path:
        if self.root not in path.parents or not (path / "experiment.json").is_file():
            raise ValueError("not a Trace2VLA experiment (outputs/trace2vla/<name>)")
        return path

    def _inside(self, path: Path) -> Path:
        if self.root not in path.parents:
            raise ValueError("outside outputs/trace2vla")
        return path

    # ----------------------------------------------------------------------- pieces --
    def _teacher(self, root: Path) -> dict[str, Any]:
        sets = {}
        for path in sorted((root / "teacher").glob("*/results.json")):
            summary = read_json(path) or {}
            for task in summary.get("tasks", []):
                for seed in (task.get("seeds") or {}).values():
                    seed["run"] = self.store.rel(root / seed["run"])
            sets[path.parent.name] = summary
        return sets

    def _evals(self, root: Path) -> list[dict[str, Any]]:
        runs = []
        folders = sorted((root / "evals").glob("*/run.json"), key=lambda p: p.stat().st_mtime)
        for path in folders:
            run = path.parent
            meta = read_json(run / "eval.json") or {}
            record = read_json(path) or {}
            settings = record.get("settings") or {}
            results = read_json(run / "results.json") or {}
            student = student_of(meta, record)
            runs.append(
                {
                    "name": run.name,
                    "path": self.store.rel(run),
                    "student": student["key"],
                    "student_label": student["label"],
                    "suite": settings.get("suite"),
                    "checkpoint": meta.get("checkpoint_label") or settings.get("policy"),
                    "training_run": meta.get("training_run"),
                    "checkpoint_step": meta.get("checkpoint_step"),
                    "scene_set": meta.get("scene_set"),
                    "progress": progress_of(run),
                    "mean": results.get("mean"),
                    "successes": results.get("successes"),
                    "episodes": results.get("episodes"),
                    "ci95": results.get("ci95"),
                    "failure_families": results.get("failure_families") or {},
                    "tasks": results.get("tasks") or [],
                }
            )
        return runs

    def _training(self, root: Path) -> list[dict[str, Any]]:
        runs = []
        for path in sorted((root / "training").glob("*/run.json"), key=lambda p: p.stat().st_mtime):
            run = path.parent
            record = read_json(path) or {}
            metrics = []
            if (run / "metrics.jsonl").is_file():
                for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
                    try:
                        metrics.append(json.loads(line))
                    except ValueError:
                        continue
            # Runs logged before exact steps read 1000, 1000, ..., 2000 after step 1000.
            metrics = repair_steps(metrics, (record.get("settings") or {}).get("log_freq"))
            student = student_of(record)
            runs.append(
                {
                    "name": run.name,
                    "path": self.store.rel(run),
                    "student": student["key"],
                    "student_label": student["label"],
                    "record": record,
                    "progress": read_json(run / "progress.json") or {},
                    "loss": downsample([float(m["loss"]) for m in metrics]),
                    "lr": downsample([float(m["lr"]) for m in metrics]),
                    "steps_logged": [int(m["step"]) for m in metrics][-1:] or [0],
                    "log": self.store.rel(run / "train.log")
                    if (run / "train.log").is_file()
                    else None,
                    "log_tail": _tail(run / "train.log", 30),
                }
            )
        return runs

    def _datasets(self, root: Path) -> list[dict[str, Any]]:
        found = []
        for path in sorted(
            (root / "datasets").glob("*/manifest.json"), key=lambda p: p.stat().st_mtime
        ):
            manifest = read_json(path) or {}
            student = student_of(manifest)
            found.append(
                {
                    "name": manifest.get("name", path.parent.name),
                    "path": self.store.rel(path.parent),
                    "student": student["key"],
                    "student_label": student["label"],
                    "derived_from": (manifest.get("derived_from") or {}).get("name"),
                    "created": manifest.get("created"),
                    "episodes": len(manifest.get("episodes", [])),
                    "frames": manifest.get("frames"),
                    "per_task": manifest.get("per_task"),
                    "rejected": len(manifest.get("rejected", [])),
                    "statistics": manifest.get("statistics"),
                }
            )
        return found

    def _demos(self, root: Path) -> list[dict[str, Any]]:
        """Yield and step counts per demo set, from the demo files themselves."""
        sets = []
        for folder in sorted(p for p in (root / "demos").glob("*") if p.is_dir()):
            tasks = []
            for task_dir in sorted(folder.glob("task_*")):
                kept = [read_json(p) or {} for p in sorted(task_dir.glob("seed_*/episode.json"))]
                rejected = [
                    read_json(p) or {} for p in sorted(task_dir.glob("seed_*.rejected.json"))
                ]
                steps = [int(e.get("steps") or 0) for e in kept]
                tasks.append(
                    {
                        "task_id": int(task_dir.name.split("_")[1]),
                        "accepted": len(kept),
                        "rejected": len(rejected),
                        "rejected_reasons": sorted({str(e.get("rejected")) for e in rejected}),
                        "steps": {
                            "min": min(steps),
                            "mean": statistics.mean(steps),
                            "max": max(steps),
                        }
                        if steps
                        else None,
                        "teacher_steps_mean": statistics.mean(
                            e.get("teacher_steps") or 0 for e in kept
                        )
                        if kept
                        else None,
                        "idle_dropped_mean": statistics.mean(
                            (e.get("tracked_steps") or 0) - (e.get("steps") or 0) for e in kept
                        )
                        if kept
                        else None,
                        "kept_all_idle": sum(1 for e in kept if e.get("dwell") == "all"),
                        "over_student_limit": sum(
                            1 for e in kept if e.get("within_student_limit") is False
                        ),
                    }
                )
            sets.append(
                {
                    "name": folder.name,
                    "path": self.store.rel(folder),
                    "settings": read_json(folder / "settings.json"),
                    "tasks": tasks,
                    "accepted": sum(t["accepted"] for t in tasks),
                    "rejected": sum(t["rejected"] for t in tasks),
                }
            )
        return sets

    # ------------------------------------------------------------------------- list --
    def experiments(self) -> dict[str, Any]:
        items = []
        folders = (
            sorted(
                (p.parent for p in self.root.glob("*/experiment.json")),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if self.root.is_dir()
            else []
        )
        for root in folders:
            settings = read_json(root / "experiment.json") or {}
            events = _events(root)
            teacher = self._teacher(root)
            official = teacher.get("official") or {}
            evals = self._evals(root)
            training = self._training(root)
            items.append(
                {
                    "name": root.name,
                    "path": self.store.rel(root),
                    "suite": settings.get("suite"),
                    "tasks": settings.get("tasks"),
                    "teacher_official": {
                        "successes": official.get("successes"),
                        "episodes": official.get("episodes"),
                    }
                    if official
                    else None,
                    "demos": sum(d.get("accepted", 0) for d in self._demos(root)),
                    "datasets": len(self._datasets(root)),
                    "training": [
                        {
                            "name": t["name"],
                            "status": t["progress"].get("status"),
                            "student_label": t["student_label"],
                        }
                        for t in training
                    ],
                    "evals": [
                        {
                            "name": e["name"],
                            "mean": e["mean"],
                            "status": e["progress"].get("status"),
                            "checkpoint": e["checkpoint"],
                            "student_label": e["student_label"],
                        }
                        for e in evals
                    ],
                    "last_event": events[-1] if events else None,
                    "modified": root.stat().st_mtime,
                }
            )
        return {"items": items}

    # ------------------------------------------------------------------- experiment --
    def experiment(self, path: Path) -> dict[str, Any]:
        root = self._experiment(path)
        settings = read_json(root / "experiment.json") or {}
        logs = sorted((root / "logs").glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        return {
            "name": root.name,
            "path": self.store.rel(root),
            "settings": settings,
            "programs": read_json(root / "teacher" / "programs" / "programs.json") or {},
            "teacher": self._teacher(root),
            "demos": self._demos(root),
            "datasets": self._datasets(root),
            "training": self._training(root),
            "evals": self._evals(root),
            "ledger": _events(root)[-LEDGER_TAIL:],
            "logs": [
                {"path": self.store.rel(p), "name": p.name, "bytes": p.stat().st_size}
                for p in logs[:20]
            ],
            "report": self.store.rel(root / "reports" / "summary.md")
            if (root / "reports" / "summary.md").is_file()
            else None,
        }

    # ------------------------------------------------------------------------ demos --
    def demos(self, folder: Path) -> dict[str, Any]:
        self._inside(folder)
        episodes = []
        for path in sorted(folder.glob("task_*/seed_*/episode.json")) + sorted(
            folder.glob("task_*/seed_*.rejected.json")
        ):
            summary = read_json(path) or {}
            accepted = bool(summary.get("accepted"))
            episodes.append(
                {
                    "path": self.store.rel(path.parent) if accepted else None,
                    "task_id": (summary.get("scene") or {}).get("task_id"),
                    "seed": (summary.get("scene") or {}).get("seed"),
                    "accepted": accepted,
                    "rejected": summary.get("rejected"),
                    "teacher_steps": summary.get("teacher_steps"),
                    "tracked_steps": summary.get("tracked_steps"),
                    "steps": summary.get("steps"),
                    "dwell": summary.get("dwell"),
                    "within_student_limit": summary.get("within_student_limit"),
                    "lag_mean_mm": (summary.get("tracking") or {}).get("lag_mean_mm"),
                    "command_abs_mean": summary.get("command_abs_mean"),
                }
            )
        return {
            "path": self.store.rel(folder),
            "name": folder.name,
            "settings": read_json(folder / "settings.json"),
            "episodes": episodes,
        }

    def demo(self, folder: Path) -> dict[str, Any]:
        self._inside(folder)
        summary = read_json(folder / "episode.json")
        if not isinstance(summary, dict):
            raise FileNotFoundError(f"no episode.json in {self.store.rel(folder)}")
        series: dict[str, dict[str, list[float]]] = {}
        if (folder / "steps.npz").is_file():
            arrays = read_npz(folder / "steps.npz")
            for key, labels in EPISODE_SERIES.items():
                rows = (arrays.get(key) or {}).get("values") or []
                if rows and isinstance(rows[0], list):
                    series[key] = {
                        label: downsample([float(row[i]) for row in rows])
                        for i, label in enumerate(labels)
                        if i < len(rows[0])
                    }
        teacher_run = Path(summary.get("source_run", ""))
        teacher: dict[str, Any] | None = None
        if teacher_run.is_dir() and self.store.outputs in teacher_run.resolve().parents:
            videos = sorted((teacher_run / "media" / "videos").glob("*.mp4"))
            teacher = {
                "run": self.store.rel(teacher_run),
                "videos": [self.store.rel(v) for v in videos],
            }
        return {
            "path": self.store.rel(folder),
            "experiment": self.store.rel(folder.parents[3]),
            "demo_set": folder.parents[1].name,
            "summary": summary,
            "series": series,
            "video": self.store.rel(folder / "video.mp4")
            if (folder / "video.mp4").is_file()
            else None,
            "teacher": teacher,
        }

    # ------------------------------------------------------------------ side by side --
    def scene(self, path: Path, task_id: int, scene: int) -> dict[str, Any]:
        """Teacher and every student evaluation on one official scene of one task."""
        root = self._experiment(path)
        teacher = None
        for summary in self._teacher(root).values():
            if summary.get("scene_set") != "official":
                continue
            for task in summary.get("tasks", []):
                seed = (task.get("seeds") or {}).get(str(scene))
                if task["task_id"] == task_id and seed:
                    run = self.store.outputs / seed["run"]
                    teacher = {
                        **seed,
                        "videos": [
                            self.store.rel(v)
                            for v in sorted((run / "media" / "videos").glob("*.mp4"))
                        ],
                    }
        students = []
        for run in self._evals(root):
            for folder in sorted(
                (self.store.outputs / run["path"]).glob(
                    f"episodes/*/task_{task_id:02d}/seed_*/scene_{scene:02d}"
                )
            ):
                episode = read_json(folder / "episode.json")
                if isinstance(episode, dict):
                    students.append(
                        {
                            "eval": run["name"],
                            "student_label": run["student_label"],
                            "checkpoint": run["checkpoint"],
                            "suite": episode.get("suite"),
                            "path": self.store.rel(folder),
                            "seed": episode.get("policy_seed_index"),
                            "success": episode.get("success"),
                            "steps": episode.get("steps"),
                            "family": (episode.get("labels") or {}).get("failure_family"),
                            "chosen": (episode.get("labels") or {}).get("chosen"),
                            "video": self.store.rel(folder / "video.mp4")
                            if (folder / "video.mp4").is_file()
                            else None,
                        }
                    )
        return {"task_id": task_id, "scene": scene, "teacher": teacher, "students": students}
