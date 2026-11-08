#!/usr/bin/env python3
"""Record a GPU timeline to CSV while a session runs (stops when --stop-file appears).

Two files, one row per sample:
  gpu.csv        time, gpu index, used/free MiB, utilization %, temperature C, power W
  processes.csv  time, gpu index, pid, role (sam3, curobo, simulator_worker, reservation, other),
                 yours (1/0), used MiB
Roles and ownership come from scripts/gpu/check_gpu.py, so other users' jobs show as "other".
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_gpu import _int, _nvidia_smi, list_gpus, list_processes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="directory for gpu.csv and processes.csv")
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--stop-file", required=True)
    parser.add_argument(
        "--t0", type=float, help="epoch seconds that t_s counts from (default: now)"
    )
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = args.t0 or time.time()
    with (
        (out / "gpu.csv").open("w", newline="") as gpu_file,
        (out / "processes.csv").open("w", newline="") as proc_file,
    ):
        gpu_writer = csv.writer(gpu_file)
        proc_writer = csv.writer(proc_file)
        gpu_writer.writerow(["t_s", "gpu", "used_mib", "free_mib", "util_pct", "temp_c", "power_w"])
        proc_writer.writerow(["t_s", "gpu", "pid", "role", "yours", "used_mib"])
        while not Path(args.stop_file).exists():
            now = round(time.time() - started, 1)
            try:
                gpus = list_gpus()
                extra = {
                    int(row[0]): row[1:]
                    for row in _nvidia_smi("--query-gpu=index,temperature.gpu,power.draw")
                }
                for gpu in gpus:
                    temp, power = (extra.get(gpu.index) or ["", ""])[:2]
                    gpu_writer.writerow(
                        [
                            now,
                            gpu.index,
                            gpu.used_mib,
                            gpu.free_mib,
                            gpu.utilization_pct,
                            _int(temp),
                            _int(power),
                        ]
                    )
                for process in list_processes(gpus):
                    proc_writer.writerow(
                        [
                            now,
                            process.gpu_index,
                            process.pid,
                            process.role,
                            int(process.mine),
                            process.used_mib,
                        ]
                    )
                gpu_file.flush()
                proc_file.flush()
            except (OSError, SystemExit, ValueError):
                pass  # a missed sample is fine; a crashed monitor loses the whole timeline
            time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    sys.exit(main())
