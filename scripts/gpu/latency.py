#!/usr/bin/env python3
"""Latency report: how long model services, the simulator and motions took, per run set.

  python scripts/gpu/latency.py PATH [PATH ...] [--calls NAME ...]

Each PATH is a session, a campaign, a check folder or a single run directory; every run
under it is included. Pass two sessions (e.g. one recorded while the GPU was busy and one
after) to compare them side by side.

  Model services   every provider call (SAM3, Contact-GraspNet, PyRoki, cuRobo): count,
                   median, p90 and max seconds
  Simulator        per control step (adapter_step), per reset, video encoding, keyframes
  Motions          the API stages (plan_motion, execute_trajectory, ...) and, per motion,
                   wall seconds per simulated second (1.0 = real time)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys


def runs_under(path: Path) -> list[Path]:
    if (path / "run.json").exists():
        return [path]
    return sorted(p.parent for p in path.rglob("run.json") if "wandb" not in p.parts)


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1) + 0.5))]


def collect(path: Path) -> dict[str, object]:
    calls: dict[str, list[float]] = {}
    buckets: dict[str, list[float]] = {}
    realtime: list[float] = []
    runs = runs_under(path)
    for run in runs:
        events = run / "trace/events.jsonl"
        started: dict[str, float] = {}
        if events.exists():
            for line in events.read_text(encoding="utf-8").splitlines():
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get("event") == "span_start":
                    started[event.get("span_id")] = float(event.get("simulator_time_s") or 0.0)
                if event.get("event") != "span_end" or event.get("duration_s") is None:
                    continue
                if event.get("category") in ("provider", "api"):
                    key = f"{event['category']}: {event.get('name')}"
                    calls.setdefault(key, []).append(float(event["duration_s"]))
                # simulator_time_s is the absolute clock; a call's simulated time is the difference
                simulated = float(event.get("simulator_time_s") or 0.0) - started.get(
                    event.get("span_id"), 0.0
                )
                if event.get("category") == "api" and simulated > 0.2:
                    realtime.append(float(event["duration_s"]) / simulated)
        try:
            timing = json.loads((run / "outcome.json").read_text())["timing"]["buckets"]
        except (OSError, ValueError, KeyError, TypeError):
            continue
        for name, data in timing.items():
            count = data.get("calls") or 0
            if count and name in {"adapter_step", "reset", "video_encode", "keyframe", "setup"}:
                buckets.setdefault(name, []).append(float(data["seconds"]) / count)
    return {"runs": len(runs), "calls": calls, "buckets": buckets, "realtime": realtime}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--calls", nargs="*", help="only these call names (substring match)")
    args = parser.parse_args()
    reports = [(path, collect(path)) for path in args.paths]
    for path, report in reports:
        print(f"\n=== {path}  ({report['runs']} runs)")
        calls = report["calls"]
        if args.calls:
            calls = {k: v for k, v in calls.items() if any(c in k for c in args.calls)}
        if calls:
            print(f"  {'call':<44}{'count':>7}{'median s':>11}{'p90 s':>9}{'max s':>9}")
            for name in sorted(calls, key=lambda k: (not k.startswith("provider"), k)):
                values = calls[name]
                print(
                    f"  {name:<44}{len(values):>7}{statistics.median(values):>11.3f}"
                    f"{quantile(values, 0.9):>9.3f}{max(values):>9.3f}"
                )
        buckets = report["buckets"]
        if buckets:
            print("  simulator, per call (median over runs):")
            for name, values in sorted(buckets.items()):
                print(f"    {name:<20} {statistics.median(values) * 1000:9.1f} ms")
        if report["realtime"]:
            print(
                "  motions: wall seconds per simulated second, median "
                f"{statistics.median(report['realtime']):.2f} (1.0 = real time, lower is faster)"
            )
        if not report["runs"]:
            print("  no run directories found here")
    return 0


if __name__ == "__main__":
    sys.exit(main())
