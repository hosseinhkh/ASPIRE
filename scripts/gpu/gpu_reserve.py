#!/usr/bin/env python3
"""Hold GPU memory so it is there when you need it (run by scripts/gpu/reserve.sh).

Allocates up to --gib GiB on the visible GPU in 1 GiB blocks, takes what it can get if the
card has less free, reports what it holds, then sleeps until SIGTERM/SIGINT and frees it.
Run it with the torch from one of the repo venvs (.venv-sam3 or .venv-libero).

Note: the CUDA context itself costs a few hundred MiB on top of the blocks.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time

GIB = 1 << 30


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gib", type=float, required=True)
    parser.add_argument("--status-file", required=True)
    parser.add_argument("--device", default="cuda", help=argparse.SUPPRESS)  # "cpu" for tests
    args = parser.parse_args()

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        print("error: no CUDA device visible", file=sys.stderr)
        return 1
    blocks = []
    wanted = max(0, int(args.gib * 4))  # quarter-GiB blocks, rounded down: never above the request
    for _ in range(wanted):
        try:
            blocks.append(torch.empty(GIB // 4, dtype=torch.uint8, device=args.device))
        except (RuntimeError, MemoryError):  # torch.OutOfMemoryError subclasses RuntimeError
            break
    held = len(blocks) / 4
    status = {
        "pid": os.getpid(),
        "requested_gib": args.gib,
        "held_gib": held,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "since": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    Path(args.status_file).write_text(json.dumps(status) + "\n", encoding="utf-8")
    print(f"holding {held:g} GiB of {args.gib:g} GiB requested (pid {os.getpid()})", flush=True)

    stop = False

    def _stop(*_: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    while not stop:
        time.sleep(1)
    blocks.clear()
    return 0


if __name__ == "__main__":
    sys.exit(main())
