"""Dashboard data for the VLA student: the benchmark runs under outputs/vla only.

Kept apart from ASPIRE's own runs (dashboard_store.Store): it reads run.json, progress.json,
results.json, benchmark.log and the episode folders that ``cap_harness.vla.benchmark`` writes,
and the parity reports of ``cap_harness.vla.parity``. Standard library only.
"""

from __future__ import annotations

import os
from pathlib import Path
import socket
from typing import Any

from dashboard_store import Store, downsample, read_json, read_npz

#: Series drawn on an episode page: npz key -> labels of its columns.
EPISODE_SERIES = {
    "action": ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"],
    "eef_pos": ["x", "y", "z"],
    "gripper_qpos": ["finger 1", "finger 2"],
}
LOG_TAIL = 300


def _alive(progress: dict[str, Any]) -> bool | None:
    pid, host = progress.get("pid"), progress.get("host")
    if not pid or host != socket.gethostname():
        return None
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def progress_of(run: Path) -> dict[str, Any]:
    progress = read_json(run / "progress.json")
    if not isinstance(progress, dict):
        return {"status": "no progress file"}
    if progress.get("status") == "running" and _alive(progress) is False:
        progress["status"] = "stopped"
        progress["warnings"] = [
            *(progress.get("warnings") or []),
            "the run's process is gone without a final status (killed or the machine restarted)",
        ]
    return progress


class VlaStore:
    def __init__(self, store: Store) -> None:
        self.store = store

    @property
    def root(self) -> Path:
        return self.store.outputs / "vla"

    def _require(self, path: Path) -> Path:
        if path != self.root and self.root not in path.parents:
            raise ValueError("not a VLA student run (outside outputs/vla)")
        return path

    # ------------------------------------------------------------------------ list --
    def runs(self) -> dict[str, Any]:
        runs = []
        if self.root.is_dir():
            folders = sorted(
                (p.parent for p in self.root.glob("*/run.json")),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for run in folders:
                meta = read_json(run / "run.json") or {}
                settings = meta.get("settings") or {}
                results = read_json(run / "results.json") or {}
                runs.append(
                    {
                        "path": self.store.rel(run),
                        "name": run.name,
                        "suite": settings.get("suite"),
                        "policy": settings.get("policy"),
                        "n_action_steps": settings.get("n_action_steps"),
                        "progress": progress_of(run),
                        "mean": results.get("mean"),
                        "std_across_seeds": results.get("std_across_seeds"),
                        "modified": run.stat().st_mtime,
                    }
                )
        parity = []
        for path in sorted(self.root.glob("parity-*.json")) if self.root.is_dir() else []:
            report = read_json(path)
            if isinstance(report, dict) and "environment" in report:
                parity.append(
                    {
                        "path": self.store.rel(path),
                        "suite": report.get("suite"),
                        "task": report.get("task"),
                        "scene": report.get("scene"),
                        "environment": bool(report["environment"].get("passed")),
                        "policy": (report.get("policy") or {}).get("passed"),
                        "policy_detail": report.get("policy"),
                        "modified": path.stat().st_mtime,
                    }
                )
        return {
            "root": self.store.rel(self.root) if self.root.exists() else "vla",
            "runs": runs,
            "parity": parity,
        }

    # ------------------------------------------------------------------------- run --
    def run(self, run: Path) -> dict[str, Any]:
        self._require(run)
        meta = read_json(run / "run.json") or {}
        episodes = []
        instructions: dict[str, str] = {}
        for path in sorted((run / "episodes").glob("*/task_*/seed_*/scene_*/episode.json")):
            episode = read_json(path)
            if not isinstance(episode, dict):
                continue
            diagnostics = episode.get("diagnostics") or {}
            instructions.setdefault(str(episode.get("task_id")), episode.get("instruction") or "")
            episodes.append(
                {
                    "path": self.store.rel(path.parent),
                    "task_id": episode.get("task_id"),
                    "seed": episode.get("policy_seed_index"),
                    "scene": episode.get("scene"),
                    "success": episode.get("success"),
                    "steps": episode.get("steps"),
                    "ended": episode.get("ended"),
                    "wall_s": episode.get("wall_s"),
                    "video": (path.parent / "video.mp4").is_file(),
                    "saturated": diagnostics.get("action_saturated_fraction"),
                    "hand_path_m": diagnostics.get("hand_path_m"),
                    "policy_ms_per_step": diagnostics.get("policy_ms_per_step"),
                }
            )
        log = run / "benchmark.log"
        lines = (
            log.read_text(encoding="utf-8", errors="replace").splitlines() if log.is_file() else []
        )
        return {
            "path": self.store.rel(run),
            "name": run.name,
            "meta": meta,
            "progress": progress_of(run),
            "results": read_json(run / "results.json") or {},
            "results_md": (run / "results.md").read_text(encoding="utf-8")
            if (run / "results.md").is_file()
            else "",
            "instructions": instructions,
            "episodes": episodes,
            "log": lines[-LOG_TAIL:],
            "log_lines": len(lines),
        }

    # --------------------------------------------------------------------- episode --
    def episode(self, folder: Path) -> dict[str, Any]:
        self._require(folder)
        summary = read_json(folder / "episode.json")
        if not isinstance(summary, dict):
            raise FileNotFoundError(f"no episode.json in {self.store.rel(folder)}")
        series: dict[str, dict[str, list[float]]] = {}
        npz = folder / "steps.npz"
        if npz.is_file():
            arrays = read_npz(npz)
            for key, labels in EPISODE_SERIES.items():
                rows = (arrays.get(key) or {}).get("values") or []
                if rows and isinstance(rows[0], list):
                    series[key] = {
                        label: downsample([float(row[i]) for row in rows])
                        for i, label in enumerate(labels)
                        if i < len(rows[0])
                    }
        run = folder.parents[4]  # <run>/episodes/<suite>/task_XX/seed_S/scene_KK
        files = sorted(p for p in folder.iterdir() if p.is_file())
        return {
            "path": self.store.rel(folder),
            "run": self.store.rel(run),
            "run_name": run.name,
            "summary": summary,
            "series": series,
            "video": self.store.rel(folder / "video.mp4")
            if (folder / "video.mp4").is_file()
            else None,
            "files": [
                {"path": self.store.rel(p), "name": p.name, "bytes": p.stat().st_size}
                for p in files
            ],
        }
