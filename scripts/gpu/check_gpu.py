#!/usr/bin/env python3
"""Will ASPIRE fit on a shared GPU, and how much does it actually use?

Standard library only, so it runs from any Python, including the conda env before the repo
venvs exist.

  check_gpu.py budget  [--gpu 0] [--no-curobo] [--sim-workers 1] [--json PATH]
      Pre-flight. Compares the GPU's *current* free memory with the planned footprint of the
      stack, and exits 1 if it does not fit with the safety margin.

  check_gpu.py measure [--gpu 0] [--watch SECONDS] [--interval 2] [--stop-file F] [--json PATH]
      Reads the real per-process usage of this repository's processes (provider services and
      simulator workers owned by you) from nvidia-smi. With --watch it samples for that long
      and reports the peak, e.g. while `demo.sh smoke` runs in another terminal.

The planned footprint is an estimate, not a measurement. Run `measure --watch` during the smoke
demo once, and trust that number for your machine. Other users' jobs on a shared card change
the free memory minute to minute, so re-run `budget` right before starting services.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Planned peak GPU memory per component, MiB. Conservative estimates from the code:
#:   sam3      ~848M-parameter model held in fp32 (~3.4 GB) + bf16 activations + CUDA context
#:   graspnet  small PointNet++ model on ~20k points + CUDA context
#:   curobo    robot spheres, 32 IK seeds, trajectory optimizer buffers, voxel/mesh world
#:   simulator MuJoCo EGL context per simulator worker (rendering only)
#:   pyroki    0: the pinned jaxlib 0.4.29 wheel is CPU-only
PLANNED_MIB = {
    "sam3": 6500,
    "contact_graspnet": 2500,
    "curobo": 3500,
    "simulator_worker": 800,
}
SAFETY_MARGIN_MIB = 2048


@dataclass(frozen=True)
class Gpu:
    index: int
    uuid: str
    name: str
    compute_cap: str
    total_mib: int
    used_mib: int
    free_mib: int
    utilization_pct: int | None


@dataclass(frozen=True)
class Process:
    pid: int
    gpu_index: int
    used_mib: int
    role: str
    mine: bool


def _nvidia_smi(*args: str) -> list[list[str]]:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        raise SystemExit("error: nvidia-smi not found; this check needs the NVIDIA driver")
    output = subprocess.run(
        [exe, *args, "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    rows = []
    for line in output.splitlines():
        if line.strip():
            rows.append([field.strip() for field in line.split(",")])
    return rows


def _int(value: str) -> int | None:
    try:
        return int(float(value))
    except ValueError:
        return None  # "[N/A]" on some boards


def list_gpus() -> list[Gpu]:
    rows = _nvidia_smi(
        "--query-gpu=index,uuid,name,compute_cap,memory.total,memory.used,memory.free,utilization.gpu"
    )
    return [
        Gpu(
            index=int(row[0]),
            uuid=row[1],
            name=row[2],
            compute_cap=row[3],
            total_mib=_int(row[4]) or 0,
            used_mib=_int(row[5]) or 0,
            free_mib=_int(row[6]) or 0,
            utilization_pct=_int(row[7]),
        )
        for row in rows
    ]


def _cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\0", b" ").decode(errors="replace")


def _owned_by_me(pid: int) -> bool:
    try:
        return Path(f"/proc/{pid}").stat().st_uid == os.getuid()
    except OSError:
        return False


def _role(cmdline: str) -> str | None:
    """Which part of this repository a process is, or None if it is not ours."""
    if "cap_harness.providers.sam3" in cmdline:
        return "sam3"
    if "cap_harness.providers.graspnet" in cmdline:
        return "contact_graspnet"
    if "cap_harness.providers.curobo" in cmdline:
        return "curobo"
    if "cap_harness.providers.pyroki" in cmdline:
        return "pyroki"
    if "gpu_reserve.py" in cmdline:
        return "reservation"
    if "cap-harness" in cmdline or "cap_harness" in cmdline or str(REPO_ROOT) in cmdline:
        return "simulator_worker"
    return None


def list_processes(gpus: list[Gpu]) -> list[Process]:
    index_by_uuid = {gpu.uuid: gpu.index for gpu in gpus}
    processes = []
    for row in _nvidia_smi("--query-compute-apps=gpu_uuid,pid,used_memory"):
        if len(row) < 3:
            continue
        pid = _int(row[1])
        if pid is None:
            continue
        cmdline = _cmdline(pid)
        role = _role(cmdline)
        mine = _owned_by_me(pid) and role is not None
        processes.append(
            Process(
                pid=pid,
                gpu_index=index_by_uuid.get(row[0], -1),
                used_mib=_int(row[2]) or 0,
                role=role or "other",
                mine=mine,
            )
        )
    return processes


def planned_footprint(with_curobo: bool, with_graspnet: bool, sim_workers: int) -> dict[str, int]:
    plan = {"sam3": PLANNED_MIB["sam3"]}
    if with_graspnet:
        plan["contact_graspnet"] = PLANNED_MIB["contact_graspnet"]
    if with_curobo:
        plan["curobo"] = PLANNED_MIB["curobo"]
    plan["simulator_workers"] = PLANNED_MIB["simulator_worker"] * sim_workers
    return plan


def _gib(mib: int) -> str:
    return f"{mib / 1024:.1f} GiB"


def _select(gpus: list[Gpu], index: int) -> Gpu:
    for gpu in gpus:
        if gpu.index == index:
            return gpu
    raise SystemExit(f"error: no GPU with index {index}; found {[g.index for g in gpus]}")


def cmd_budget(args: argparse.Namespace) -> int:
    gpus = list_gpus()
    print("GPUs right now:")
    for gpu in gpus:
        util = "?" if gpu.utilization_pct is None else f"{gpu.utilization_pct}%"
        print(
            f"  [{gpu.index}] {gpu.name} (sm_{gpu.compute_cap.replace('.', '')}) "
            f"free {_gib(gpu.free_mib)} of {_gib(gpu.total_mib)}, util {util}"
        )
    target = _select(gpus, args.gpu)
    already_mine = sum(
        p.used_mib for p in list_processes(gpus) if p.mine and p.gpu_index == target.index
    )
    on_cpu = set(filter(None, os.environ.get("ASPIRE_CPU_PROVIDERS", "").split(",")))
    plan = planned_footprint(
        not args.no_curobo and "curobo" not in on_cpu,
        not args.no_graspnet and "contact_graspnet" not in on_cpu,
        args.sim_workers,
    )
    need = sum(plan.values()) + SAFETY_MARGIN_MIB
    # Memory our own running services already hold counts as available to us.
    available = target.free_mib + already_mine
    print(f"\nPlanned ASPIRE footprint on GPU {target.index} (estimate):")
    for name, mib in plan.items():
        print(f"  {name:<18} {_gib(mib):>9}")
    print(f"  {'safety margin':<18} {_gib(SAFETY_MARGIN_MIB):>9}")
    print(f"  {'total needed':<18} {_gib(need):>9}")
    print(
        f"  {'available':<18} {_gib(available):>9}"
        + (
            f"  (includes {_gib(already_mine)} already held by your ASPIRE processes)"
            if already_mine
            else ""
        )
    )
    fits = available >= need
    if target.compute_cap and target.compute_cap != "12.0":
        print(
            f"\nnote: GPU {target.index} is sm_{target.compute_cap.replace('.', '')}; the "
            "rtxpro6000 profile assumes sm_120. Pick the matching profile in configs/profiles/."
        )
    if fits:
        print(f"\nOK: fits with {_gib(available - need)} to spare.")
    else:
        print(f"\nDOES NOT FIT: short by {_gib(need - available)}.")
        if not args.no_curobo:
            print("  - try without cuRobo (saves ~3.4 GiB): services.sh start --no-curobo")
        others = [g for g in gpus if g.index != target.index and g.free_mib > available]
        if others:
            best = max(others, key=lambda g: g.free_mib)
            print(f"  - GPU {best.index} has {_gib(best.free_mib)} free right now")
        print("  - or wait for other users' jobs to finish; free memory on a shared card moves")
    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "gpus": [asdict(g) for g in gpus],
                    "target": target.index,
                    "plan_mib": plan,
                    "safety_margin_mib": SAFETY_MARGIN_MIB,
                    "needed_mib": need,
                    "available_mib": available,
                    "fits": fits,
                },
                indent=2,
            )
            + "\n"
        )
    return 0 if fits else 1


def cmd_measure(args: argparse.Namespace) -> int:
    deadline = time.monotonic() + max(0.0, args.watch)
    peak_total = 0
    peak_by_role: dict[str, int] = {}
    samples = 0
    min_free: int | None = None
    while True:
        gpus = list_gpus()
        target = _select(gpus, args.gpu)
        mine = [p for p in list_processes(gpus) if p.mine and p.gpu_index == target.index]
        by_role: dict[str, int] = {}
        for process in mine:
            by_role[process.role] = by_role.get(process.role, 0) + process.used_mib
        total = sum(by_role.values())
        peak_total = max(peak_total, total)
        for role, mib in by_role.items():
            peak_by_role[role] = max(peak_by_role.get(role, 0), mib)
        min_free = target.free_mib if min_free is None else min(min_free, target.free_mib)
        samples += 1
        if time.monotonic() >= deadline:
            break
        if args.stop_file and Path(args.stop_file).exists():
            break
        time.sleep(args.interval)

    label = f"peak over {samples} samples" if samples > 1 else "now"
    print(f"Your ASPIRE processes on GPU {args.gpu} ({label}):")
    if not peak_by_role:
        print("  none found (are the services running? `scripts/gpu/services.sh status`)")
    for role, mib in sorted(peak_by_role.items(), key=lambda item: -item[1]):
        planned = PLANNED_MIB.get(role)
        note = f"  (planned {_gib(planned)})" if planned else ""
        print(f"  {role:<18} {_gib(mib):>9}{note}")
    print(f"  {'total':<18} {_gib(peak_total):>9}")
    if min_free is not None:
        print(f"Lowest free memory left on GPU {args.gpu} while sampling: {_gib(min_free)}")
        if min_free < 1024:
            print(
                "WARNING: under 1 GiB free at some point; an OOM in a provider or a "
                "neighbour's job is likely. Drop cuRobo or move to a quieter GPU."
            )
    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "gpu": args.gpu,
                    "samples": samples,
                    "peak_total_mib": peak_total,
                    "peak_by_role_mib": peak_by_role,
                    "min_free_mib": min_free,
                },
                indent=2,
            )
            + "\n"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    budget = sub.add_parser("budget", help="pre-flight: does the stack fit right now?")
    budget.add_argument("--gpu", type=int, default=int(os.environ.get("ASPIRE_GPU", "0")))
    budget.add_argument("--no-curobo", action="store_true")
    budget.add_argument("--no-graspnet", action="store_true")
    budget.add_argument("--sim-workers", type=int, default=1)
    budget.add_argument("--json")
    measure = sub.add_parser("measure", help="actual usage of your ASPIRE processes")
    measure.add_argument("--gpu", type=int, default=int(os.environ.get("ASPIRE_GPU", "0")))
    measure.add_argument("--watch", type=float, default=0.0, help="sample for this many seconds")
    measure.add_argument("--interval", type=float, default=2.0)
    measure.add_argument("--stop-file", help="stop sampling early once this file exists")
    measure.add_argument("--json")
    args = parser.parse_args(argv)
    if args.command == "budget":
        return cmd_budget(args)
    return cmd_measure(args)


if __name__ == "__main__":
    sys.exit(main())
