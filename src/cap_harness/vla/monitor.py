"""Live state of a benchmark run, written as it goes.

    <out>/benchmark.log    one line per episode (and every start, stop and warning), timestamped
    <out>/progress.json    status, planned/done episodes, success so far, speed, ETA, warnings
    <out>/results.json/md  the results table, refreshed after every episode

``progress.json`` is what ``python -m cap_harness.vla.status`` and the dashboard read.  Its
``status`` is running, finished, interrupted or failed; a run whose process is gone while it
still says running is reported as stopped.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime
import json
import os
from pathlib import Path
import socket
import time
from typing import Any

from cap_harness.vla.protocol import WARN_AFTER_EPISODES, WARN_BELOW_REFERENCE

#: Pipeline-health checks over the episodes run so far: (diagnostic, test, message).
HEALTH_CHECKS: tuple[tuple[str, Callable[[float], bool], str], ...] = (
    (
        "action_saturated_fraction",
        lambda value: value > 0.5,
        "actions sit at the controller limit in most steps (scale or frame mismatch?)",
    ),
    (
        "hand_path_m",
        lambda value: value < 0.02,
        "the hand barely moves (wrong inputs to the policy?)",
    ),
)
HEALTH_SHARE = 0.3
HEALTH_MIN_EPISODES = 10


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def process_alive(pid: int | None, host: str | None) -> bool | None:
    """True/False for a process on this host; None when it ran elsewhere or is unknown."""
    if not pid or host != socket.gethostname():
        return None
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_progress(root: Path) -> dict[str, Any] | None:
    try:
        progress = json.loads((root / "progress.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if progress.get("status") == "running":
        alive = process_alive(progress.get("pid"), progress.get("host"))
        if alive is False:
            progress["status"] = "stopped"
            progress.setdefault("warnings", []).append(
                "the run's process is gone without a final status (killed or the machine restarted)"
            )
    return progress


class RunMonitor:
    """Keeps benchmark.log, progress.json and the results files current."""

    def __init__(
        self,
        root: Path,
        *,
        title: str,
        planned: int,
        resumed: list[Mapping[str, Any]],
        references: Mapping[str, Any],
        write_results: Callable[[list[Mapping[str, Any]]], Any],
    ) -> None:
        self.root = root
        self.title = title
        self.planned = planned
        self.episodes: list[Mapping[str, Any]] = list(resumed)
        self.resumed = len(resumed)
        self.ran = 0
        self.ran_seconds = 0.0
        self.references = dict(references)
        self.write_results = write_results
        self.started_iso = now_iso()
        self.started = time.monotonic()
        self.status = "running"
        self.current: dict[str, Any] | None = None
        self.last: dict[str, Any] | None = None
        self._warnings: dict[str, str] = {}  # kind -> latest text; logged once per kind
        self.error: str | None = None
        self.log_path = root / "benchmark.log"

    # ------------------------------------------------------------------------------ log --
    def log(self, message: str) -> None:
        print(message, flush=True)
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(f"{now_iso()} {message}\n")

    @property
    def warnings(self) -> list[str]:
        return list(self._warnings.values())

    def warn(self, kind: str, message: str) -> None:
        """Record a warning; it is logged the first time and its text kept current after."""
        if kind not in self._warnings:
            self.log(f"WARNING: {message}")
        self._warnings[kind] = message

    # ------------------------------------------------------------------------- progress --
    def done(self) -> int:
        return len(self.episodes)

    def successes(self) -> int:
        return sum(1 for episode in self.episodes if episode.get("success"))

    def start(self, current: Mapping[str, Any]) -> None:
        self.current = dict(current)
        self.write()

    def add(self, summary: Mapping[str, Any]) -> None:
        self.episodes.append(summary)
        self.ran += 1
        self.ran_seconds += float(summary.get("wall_s") or 0.0)
        self.last = {
            key: summary.get(key)
            for key in ("task_id", "policy_seed_index", "scene", "success", "steps", "wall_s")
        }
        self.current = None
        done, successes = self.done(), self.successes()
        eta = self.eta_s()
        self.log(
            f"[{done}/{self.planned}] {summary['task_ref']} seed {summary['policy_seed_index']} "
            f"scene {summary['scene']}: {'success' if summary['success'] else 'fail'} in "
            f"{summary['steps']} steps ({summary['wall_s']:.1f} s) | so far {successes}/{done} "
            f"({100 * successes / done:.0f}%)" + (f" | ETA {eta / 60:.0f} min" if eta else "")
        )
        self._check()
        self.write_results(self.episodes)
        self.write()

    def eta_s(self) -> float | None:
        if not self.ran:
            return None
        return max(self.planned - self.done(), 0) * self.ran_seconds / self.ran

    def _check(self) -> None:
        done = self.done()
        expected = self.references.get("expected")
        rate = 100.0 * self.successes() / done if done else 0.0
        if (
            expected is not None
            and done >= WARN_AFTER_EPISODES
            and rate < expected - WARN_BELOW_REFERENCE
        ):
            self.warn(
                "below_reference",
                f"success so far {rate:.0f}% is far below the published {expected:.0f}%: check "
                "`vla.sh parity`, the image size and --n-action-steps before trusting this run",
            )
        recent = [episode.get("diagnostics") or {} for episode in self.episodes[self.resumed :]]
        if len(recent) >= HEALTH_MIN_EPISODES:
            for key, test, message in HEALTH_CHECKS:
                values = [d[key] for d in recent if isinstance(d.get(key), int | float)]
                if values and sum(1 for v in values if test(v)) / len(values) > HEALTH_SHARE:
                    self.warn(key, message)

    def finish(self, status: str, error: str | None = None) -> None:
        self.status = status
        self.error = error
        self.current = None
        self.write_results(self.episodes)
        self.write()
        self.log(
            f"run {status}: {self.successes()}/{self.done()} successful of {self.planned} planned"
        )

    def write(self) -> None:
        done, successes = self.done(), self.successes()
        elapsed = time.monotonic() - self.started
        progress = {
            "status": self.status,
            "title": self.title,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "started": self.started_iso,
            "updated": now_iso(),
            "planned": self.planned,
            "done": done,
            "resumed": self.resumed,
            "ran": self.ran,
            "successes": successes,
            "success_rate": successes / done if done else None,
            "elapsed_s": round(elapsed, 1),
            "s_per_episode": round(self.ran_seconds / self.ran, 2) if self.ran else None,
            "eta_s": round(self.eta_s(), 0) if self.eta_s() is not None else None,
            "current": self.current,
            "last": self.last,
            "expected": self.references.get("expected"),
            "notes": self.references.get("notes", []),
            "warnings": self.warnings,
            "error": self.error,
        }
        atomic_write(self.root / "progress.json", json.dumps(progress, indent=2) + "\n")
