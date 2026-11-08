#!/usr/bin/env python3
"""Readable live view of `claude -p --output-format stream-json --verbose` output.

Reads stream-json events on stdin, appends every raw line to --raw (the full transcript), and
prints one short line per step: the agent's text, each tool call (Bash commands in full, other
tools by target), tool errors, and the final cost / turns / duration. Unknown event shapes are
kept in the raw file and skipped on screen.

    claude -p "..." --output-format stream-json --verbose | stream_view.py --raw transcript.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

DIM, BOLD, RED, GREEN, CYAN, RESET = (
    "\033[2m",
    "\033[1m",
    "\033[31m",
    "\033[32m",
    "\033[36m",
    "\033[0m",
)


def _short(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _tool_line(name: str, tool_input: dict) -> str:
    if name == "Bash":
        return f"$ {_short(tool_input.get('command', ''), 220)}"
    for key in ("file_path", "path", "pattern", "url", "description", "prompt"):
        if key in tool_input:
            return f"{name}: {_short(tool_input[key], 160)}"
    return name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", required=True, help="append raw stream-json lines here")
    parser.add_argument("--summary", help="write the final result event here as JSON")
    args = parser.parse_args()
    color = sys.stdout.isatty()

    def paint(code: str, text: str) -> str:
        return f"{code}{text}{RESET}" if color else text

    started = time.monotonic()
    raw_path = Path(args.raw)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    final: dict | None = None
    with raw_path.open("a", encoding="utf-8") as raw:
        for line in sys.stdin:
            raw.write(line)
            raw.flush()
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                print(paint(DIM, _short(line, 200)))
                continue
            stamp = paint(DIM, f"[{int(time.monotonic() - started) // 60:3d}m]")
            kind = event.get("type")
            if kind == "system" and event.get("subtype") == "init":
                print(stamp, paint(BOLD, f"session started, model {event.get('model', '?')}"))
            elif kind == "assistant":
                for block in event.get("message", {}).get("content", []) or []:
                    if block.get("type") == "text" and block.get("text", "").strip():
                        print(stamp, paint(CYAN, _short(block["text"], 300)))
                    elif block.get("type") == "tool_use":
                        print(stamp, _tool_line(block.get("name", "?"), block.get("input") or {}))
            elif kind == "user":
                for block in event.get("message", {}).get("content", []) or []:
                    if (
                        isinstance(block, dict)
                        and block.get("type") == "tool_result"
                        and block.get("is_error")
                    ):
                        content = block.get("content")
                        if isinstance(content, list):
                            content = " ".join(
                                part.get("text", "") for part in content if isinstance(part, dict)
                            )
                        print(stamp, paint(RED, "tool error: " + _short(content or "", 200)))
            elif kind == "result":
                final = event
    if final is None:
        print(paint(RED, "no final result event (interrupted, timed out, or failed to start)"))
        return 1
    ok = final.get("subtype") == "success" and not final.get("is_error")
    cost = final.get("total_cost_usd")
    minutes = (final.get("duration_ms") or 0) / 60000
    print()
    print(
        paint(GREEN if ok else RED, f"agent finished: {final.get('subtype', '?')}"),
        f"| turns {final.get('num_turns', '?')}",
        f"| {minutes:.1f} min",
        f"| est. cost ${cost:.2f}" if isinstance(cost, int | float) else "",
    )
    if args.summary:
        Path(args.summary).write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
