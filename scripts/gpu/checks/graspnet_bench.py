"""Time Contact-GraspNet requests on a synthetic tabletop scene (numpy + standard library).

  python graspnet_bench.py URL [--requests 3] [--timeout 600]

One forward pass per request and no service-side retries, so each timing is one inference.
Prints one line per request and a final "median" line.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import statistics
import time
import urllib.error
import urllib.request

import numpy as np


def encode(array: np.ndarray) -> str:
    with io.BytesIO() as buffer:
        np.save(buffer, np.asarray(array, dtype=np.float32), allow_pickle=False)
        return base64.b64encode(buffer.getvalue()).decode("ascii")


def scene(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Build a table 0.8 m from the camera with a 5 cm cube on it (camera frame, z forward).

    Points are where a depth camera sees them: the table top, the cube's top face, and the two
    cube sides facing the camera.
    """
    table = np.column_stack(
        [rng.uniform(-0.35, 0.35, 20000), rng.uniform(-0.25, 0.25, 20000), np.full(20000, 0.8)]
    )
    table = table[~((np.abs(table[:, 0]) < 0.025) & (np.abs(table[:, 1]) < 0.025))]
    side = 0.05
    top = np.column_stack(
        [rng.uniform(-side / 2, side / 2, 1500), rng.uniform(-side / 2, side / 2, 1500)]
    )
    top = np.column_stack([top, np.full(len(top), 0.8 - side)])
    left = np.column_stack(
        [
            np.full(600, -side / 2),
            rng.uniform(-side / 2, side / 2, 600),
            rng.uniform(0.8 - side, 0.8, 600),
        ]
    )
    front = np.column_stack(
        [
            rng.uniform(-side / 2, side / 2, 600),
            np.full(600, -side / 2),
            rng.uniform(0.8 - side, 0.8, 600),
        ]
    )
    cube = np.vstack([top, left, front])
    return np.vstack([table, cube]), cube


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("url", help="service base URL, e.g. http://127.0.0.1:8115")
    parser.add_argument("--requests", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()
    full, cube = scene(np.random.default_rng(0))
    body = json.dumps(
        {
            "pc_full_base64": encode(full),
            "pc_segment_base64": encode(cube),
            "segmap_id": 1,
            "local_regions": True,
            "filter_grasps": True,
            "forward_passes": 1,
            "max_retries": 0,
        }
    ).encode()
    times = []
    for index in range(1, args.requests + 1):
        request = urllib.request.Request(
            args.url.rstrip("/") + "/plan_point_clouds",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                payload = json.loads(response.read())
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"request {index}: FAILED after {time.perf_counter() - started:.1f} s: {exc}")
            return 1
        elapsed = time.perf_counter() - started
        with io.BytesIO(base64.b64decode(payload["grasps_base64"])) as buffer:
            grasps = len(np.load(buffer, allow_pickle=False))
        times.append(elapsed)
        print(f"request {index}: {elapsed:.1f} s, {grasps} grasps ({len(full)} scene points)")
    print(f"median {statistics.median(times):.1f} s per inference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
