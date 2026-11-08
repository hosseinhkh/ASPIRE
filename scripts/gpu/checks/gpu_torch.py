#!/usr/bin/env python3
"""Check that this venv's torch runs kernels on the visible GPU (sm_120 needs CUDA 12.8+)."""

from __future__ import annotations

import math
import sys
import time


def main() -> int:
    import torch

    if not torch.cuda.is_available():
        print(f"FAIL torch {torch.__version__}: CUDA not available")
        return 1
    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    capability = f"sm_{major}{minor}"
    started = time.perf_counter()
    a = torch.randn(2048, 2048, device="cuda")
    value = float((a @ a).sum())  # forces a real kernel launch on this architecture
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000
    if not math.isfinite(value):
        print("FAIL matmul returned NaN")
        return 1
    print(
        f"PASS torch {torch.__version__} (CUDA {torch.version.cuda}) on {name} {capability}, "
        f"2048x2048 matmul {elapsed_ms:.0f} ms"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
