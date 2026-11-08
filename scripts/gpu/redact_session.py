#!/usr/bin/env python3
"""Replace secret values (Bedrock / AWS keys, Hugging Face and W&B tokens) in a session directory.

  redact_session.py SESSION_DIR

An agent that runs `env`, or a library that logs its request headers, can copy a credential into
a transcript or a log. This scans every text file under SESSION_DIR for the values of the
credential variables in the current environment (and the Hugging Face token file) and replaces
each with [REDACTED:<variable>]. Run it before sharing a session or uploading it to W&B;
scripts/gpu/session.sh does so automatically. Prints how many replacements it made.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import sys

SECRET_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", re.IGNORECASE)
TEXT_SUFFIXES = {
    ".json", ".jsonl", ".txt", ".log", ".md", ".html", ".tsv", ".csv", ".py", ".err",
    ".stderr", ".stdout", ".yaml", ".yml", ".diff", "",
}  # fmt: skip
MAX_BYTES = 512 * 1024 * 1024
MIN_SECRET_LEN = 12  # shorter values (flags like "1") would replace ordinary text


def secrets() -> dict[str, str]:
    found = {
        name: value
        for name, value in os.environ.items()
        if SECRET_NAME.search(name) and len(value) >= MIN_SECRET_LEN
    }
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    try:
        token = (hf_home / "token").read_text(encoding="utf-8").strip()
    except OSError:
        token = ""
    if len(token) >= MIN_SECRET_LEN:
        found.setdefault("HF_TOKEN", token)
    return found


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    session = Path(sys.argv[1])
    values = sorted(secrets().items(), key=lambda item: -len(item[1]))
    if not values:
        print("redact: no credential variables set; nothing to look for")
        return 0
    replaced = files = 0
    for path in session.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if path.stat().st_size > MAX_BYTES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        count = 0
        for name, value in values:
            hits = text.count(value)
            if hits:
                text = text.replace(value, f"[REDACTED:{name}]")
                count += hits
        if count:
            path.write_text(text, encoding="utf-8")
            replaced += count
            files += 1
            print(f"redact: {count} in {path.relative_to(session)}")
    print(f"redact: {replaced} secret value(s) replaced in {files} file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
