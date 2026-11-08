#!/usr/bin/env python3
"""One-line summary of the model-service calls recorded in one run directory.

Counts every recorded provider call (``trace/calls/call-*-<provider>-*/output.json``) and
separates infrastructure errors (service down, HTTP failure) from ordinary negative results
(SAM3 finding nothing, IK out of reach), which are the program's business, not the setup's.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROVIDERS = ("sam3", "contact-graspnet", "pyroki", "curobo")
INFRA_CODES = {"provider_unavailable", "provider_error"}


def main() -> int:
    run_dir = Path(sys.argv[1])
    counts: dict[str, list[int]] = {}
    infra: dict[str, set[str]] = {}
    for output in sorted((run_dir / "trace" / "calls").glob("call-*/output.json")):
        name = output.parent.name.split("-", 2)[-1]
        provider = next((p for p in PROVIDERS if name.startswith(p + "-")), None)
        if provider is None:
            continue
        calls_ok = counts.setdefault(provider, [0, 0])
        calls_ok[0] += 1
        try:
            record = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        error = record.get("error") or {}
        if record.get("ok", True):
            calls_ok[1] += 1
        elif error.get("code") in INFRA_CODES:
            infra.setdefault(provider, set()).add(error["code"])
    if not counts:
        print("no provider calls recorded")
        return 0
    parts = [f"{name} {total} calls ({ok} ok)" for name, (total, ok) in counts.items()]
    line = "services: " + ", ".join(parts)
    if infra:
        line += " | SERVICE ERRORS: " + ", ".join(
            f"{name} {'/'.join(sorted(codes))}" for name, codes in infra.items()
        )
    print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
