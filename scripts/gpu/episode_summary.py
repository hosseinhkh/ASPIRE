#!/usr/bin/env python3
"""Explain episodes in plain text: outcome, failed calls, the program's result and its output.

  episode_summary.py RUN_DIR...            every run directory given
  episode_summary.py --unsolved SESSION    every episode in a session that did not succeed

For each episode: how it ended, each call that failed (with its error), what the program set as
`result`, and the last lines it printed. Enough to paste into a chat or an issue.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

TAIL_LINES = 15


def read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def failed_calls(run: Path) -> list[str]:
    spans = {}
    events = run / "trace/events.jsonl"
    if events.exists():
        for line in events.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("event") == "span_end":
                spans[event.get("span_id")] = event
    lines = []
    for call in sorted((run / "trace/calls").glob("call-*")):
        number = call.name.split("-")[1]
        span = spans.get(f"span-{number}", {})
        output = read_json(call / "output.json")
        error = output.get("error") if isinstance(output, dict) else None
        failure = read_json(call / "error.json")
        if span.get("ok") is False or isinstance(error, dict) or isinstance(failure, dict):
            if isinstance(error, dict):
                why = f"{error.get('code')}: {error.get('message')}"
            elif isinstance(failure, dict):
                why = f"{failure.get('type')}: {failure.get('message')}"
            else:
                why = "not ok"
            name = span.get("name") or call.name.split("-", 2)[-1]
            lines.append(f"  call {int(number)} {name} [{span.get('category', '?')}]: {why[:300]}")
    return lines


def summarize(run: Path) -> str:
    meta = read_json(run / "run.json") or {}
    outcome = read_json(run / "outcome.json") or {}
    result = read_json(run / "source/program-result.json") or {}
    title = (
        f"{meta.get('benchmark', '?')} {meta.get('suite', '?')}:{meta.get('task_id', '?')} "
        f"seed {meta.get('seed', '?')} ({meta.get('task_name', '')})"
    )
    out = [f"=== {title}", f"  dir: {run}"]
    keys = ("program_ok", "task_success", "protocol_success", "termination_reason",
            "steps_executed", "simulator_duration_s")  # fmt: skip
    out.append("  outcome: " + ", ".join(f"{k}={outcome.get(k)}" for k in keys))
    if isinstance(result.get("error"), dict):
        out.append(f"  program error: {result['error'].get('message')}")
    calls = failed_calls(run)
    out.append(f"  failed calls: {len(calls)}")
    out.extend(calls[:20])
    if len(calls) > 20:
        out.append(f"  ... {len(calls) - 20} more")
    value = json.dumps(result.get("result"), default=str)
    out.append(f"  program result: {value[:800]}{' ...' if len(value) > 800 else ''}")
    stdout = run / "logs/program.stdout"
    if stdout.exists():
        tail = stdout.read_text(encoding="utf-8", errors="replace").splitlines()[-TAIL_LINES:]
        out.append(f"  last {len(tail)} lines printed by the program:")
        out.extend(f"    | {line[:200]}" for line in tail)
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--unsolved", action="store_true", help="paths are session directories")
    args = parser.parse_args()
    runs: list[Path] = []
    for path in map(Path, args.paths):
        if args.unsolved:
            for outcome in sorted(path.rglob("outcome.json")):
                if "wandb" in outcome.parts:
                    continue
                data = read_json(outcome) or {}
                if data.get("task_success") is not True:
                    runs.append(outcome.parent)
        else:
            runs.append(path)
    if not runs:
        print("every episode succeeded" if args.unsolved else "no run directories given")
        return 0
    print("\n\n".join(summarize(run) for run in runs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
