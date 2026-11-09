"""Print a LIBERO-Pro task's language instruction, read offline from its BDDL file.

  .venv-libero/bin/python scripts/gpu/checks/task_instruction.py SUITE TASK_ID

Prints one line, "INSTRUCTION=<text>", or nothing (exit 1) if the task cannot be resolved.
"""

from __future__ import annotations

import sys


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    try:
        from cap_harness.libero.registry import LiberoSuiteRegistry

        language = LiberoSuiteRegistry().resolve(f"{sys.argv[1]}:{sys.argv[2]}").language
    except Exception as exc:  # any failure just means "no instruction in the prompt"
        print(f"cannot resolve {sys.argv[1]}:{sys.argv[2]}: {exc}", file=sys.stderr)
        return 1
    print("INSTRUCTION=" + " ".join(str(language).split()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
