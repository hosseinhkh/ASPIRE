"""Run independent per-episode jobs (grading, conversion) in worker processes.

Each worker is a fresh Python process (``spawn``: no simulator or GPU state is copied from the
parent) that keeps its own simulator adapters between jobs. The parent decides which episodes
need work, hands them out, and reports each result as it arrives; a job that raises is reported
and the others go on. With one worker the jobs run in this process, one after another.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
import multiprocessing
from typing import Any


def run_jobs(
    function: Callable[..., dict[str, Any]],
    jobs: Sequence[tuple[Any, ...]],
    *,
    workers: int = 1,
    on_result: Callable[[tuple[Any, ...], dict[str, Any]], None],
    on_error: Callable[[tuple[Any, ...], BaseException], None],
    cleanup: Callable[[], None] | None = None,
) -> None:
    """Call ``function(*job)`` for every job; ``on_result`` or ``on_error`` gets each outcome.

    ``cleanup`` runs in this process after the jobs when they ran here (workers <= 1);
    worker processes clean up when they exit.
    """
    if workers <= 1 or len(jobs) <= 1:
        try:
            for job in jobs:
                try:
                    result = function(*job)
                except Exception as error:  # one bad episode must not stop the others
                    on_error(job, error)
                    continue
                on_result(job, result)
        finally:
            if cleanup is not None:
                cleanup()
        return
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=context) as pool:
        futures: dict[Future[dict[str, Any]], tuple[Any, ...]] = {
            pool.submit(function, *job): job for job in jobs
        }
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
            except BrokenProcessPool as error:
                # A worker died (killed, or crashed in native code): the jobs not yet done
                # are reported as failed; rerunning the command picks them up.
                on_error(job, error)
                continue
            except Exception as error:
                on_error(job, error)
                continue
            on_result(job, result)


def chunks(items: Iterable[Any], size: int) -> list[list[Any]]:
    """Consecutive groups of at most ``size`` items."""
    group: list[Any] = []
    out: list[list[Any]] = []
    for item in items:
        group.append(item)
        if len(group) == size:
            out.append(group)
            group = []
    if group:
        out.append(group)
    return out
