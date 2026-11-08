#!/usr/bin/env python3
"""Is the gated SAM3 model (facebook/sam3) downloadable with this HF token? Prints one status."""

from __future__ import annotations

import os
import sys

REPO = "facebook/sam3"


def main() -> int:
    from huggingface_hub import hf_hub_download, try_to_load_from_cache
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

    cached = try_to_load_from_cache(REPO, "sam3.pt")
    if isinstance(cached, str) and os.path.exists(cached):
        print(f"GRANTED weights already cached ({cached})")
        return 0
    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        from huggingface_hub import get_token

        if not get_token():
            print("NO_TOKEN set HF_TOKEN (scripts/gpu/set_credentials.sh) or run `hf auth login`")
            return 3
    try:
        hf_hub_download(REPO, "config.json")  # small file: proves access without the 3.4 GB
    except GatedRepoError as exc:
        text = str(exc).lower()
        if "awaiting a review" in text or "pending" in text:
            print("PENDING access request is awaiting review by the repo authors")
            return 2
        print("NO_ACCESS request access at https://huggingface.co/facebook/sam3")
        return 3
    except HfHubHTTPError as exc:
        print(f"ERROR {type(exc).__name__}: {str(exc).splitlines()[0][:160]}")
        return 1
    print("GRANTED config.json downloadable; sam3.pt (~3.4 GB) downloads on first service start")
    return 0


if __name__ == "__main__":
    sys.exit(main())
