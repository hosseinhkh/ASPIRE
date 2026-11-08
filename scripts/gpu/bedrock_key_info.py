#!/usr/bin/env python3
"""Region and expiry of the Bedrock API key in $AWS_BEARER_TOKEN_BEDROCK, without printing it.

A short-term Bedrock API key ("bedrock-api-key-...") is a base64-encoded, SigV4-presigned URL:
it is scoped to one region and stops working X-Amz-Expires seconds (at most 12 hours) after
X-Amz-Date. A long ASPIRE run that outlives it fails midway with authentication errors, so
check before starting. Long-term keys carry no expiry and are reported as such.

  bedrock_key_info.py [--min-minutes 60] [--region-only]

Prints one line; exit code 0 = usable, 2 = expired or expiring within --min-minutes,
1 = no key set or not parseable.
"""

from __future__ import annotations

import argparse
import base64
import binascii
from datetime import datetime, timedelta, timezone
import os
import sys
from urllib.parse import parse_qs, urlsplit

PREFIX = "bedrock-api-key-"


def describe(token: str) -> dict[str, object]:
    token = token.strip()
    if not token.startswith(PREFIX):
        return {"kind": "long-term"}
    payload = token[len(PREFIX) :]
    try:
        url = base64.b64decode(payload + "=" * (-len(payload) % 4)).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError) as exc:
        raise ValueError("not a decodable short-term Bedrock API key") from exc
    query = parse_qs(urlsplit("https://" + url.split("://", 1)[-1]).query)
    try:
        credential = query["X-Amz-Credential"][0].split("/")
        issued = datetime.strptime(query["X-Amz-Date"][0], "%Y%m%dT%H%M%SZ").replace(
            tzinfo=timezone.utc
        )
        lifetime = timedelta(seconds=int(query["X-Amz-Expires"][0]))
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError("short-term key without region/date/expiry fields") from exc
    return {
        "kind": "short-term",
        "region": credential[2] if len(credential) > 2 else None,
        "issued": issued,
        "expires": issued + lifetime,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--min-minutes", type=float, default=0.0)
    parser.add_argument("--region-only", action="store_true", help="print only the key's region")
    args = parser.parse_args()
    token = os.environ.get("AWS_BEARER_TOKEN_BEDROCK", "")
    if not token:
        print("no AWS_BEARER_TOKEN_BEDROCK set")
        return 1
    try:
        info = describe(token)
    except ValueError as exc:
        print(f"could not read the key: {exc}")
        return 1
    if args.region_only:
        print(info.get("region") or "")
        return 0
    if info["kind"] == "long-term":
        print("long-term Bedrock API key (no built-in expiry)")
        return 0
    now = datetime.now(timezone.utc)
    left = info["expires"] - now
    minutes = left.total_seconds() / 60
    local = info["expires"].astimezone().strftime("%Y-%m-%d %H:%M %Z")
    if minutes <= 0:
        print(f"short-term key for {info['region']} EXPIRED at {local}; generate a new one")
        return 2
    line = f"short-term key for {info['region']}, expires {local} ({minutes / 60:.1f} h left)"
    if minutes < args.min_minutes:
        print(line + f"; that is less than the {args.min_minutes:.0f} min this run needs")
        return 2
    print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
