"""t2v.sh collect: the whole data collection of an experiment in one command.

    t2v.sh collect NAME --scenes alt1-train,alt2-train,alt3-train,alt4-train,train
        [--programs-from outputs/sessions/<session>] [--follow] [--workers 2] [--prepare-workers 6]

It repeats, until there is nothing left to do:

1. Programs. A task without a program takes the one its coding agent wrote in the session
   (``--programs-from``), once that agent has finished and its program solved at least
   ``--min-heldout`` of its held-out scenes. Programs registered by hand (teacher-add) are used
   as they are; a weak or missing agent program is reported, never registered.
2. Teacher. For each task that has a program (in task order), and each scene set (in the given
   order): record the scenes that have no counted episode yet (teacher.run_scene_set).
3. Grading and conversion. As soon as a (scene set, task) is recorded, a background process
   grades and converts it (``t2v.sh prepare``; test scene sets are graded only, never converted)
   while the teacher goes on with the next one.

With ``--follow`` it waits for the programs still missing until the session has finished (or,
without a session, until every task has one). An episode a model service failed is recorded
again; a (scene set, task) still missing scenes after ``--attempts`` tries is left incomplete and
named. Everything finished is skipped when the same command runs again, so a stopped collection
resumes where it stopped.

Progress is written to collect.json in the experiment folder after every change (t2v.sh status,
and the dashboard's Pipeline tab).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

from cap_harness.trace2vla.progress import STATUS_FILE
from cap_harness.trace2vla.scenes import SceneSet, compact
from cap_harness.trace2vla.store import Busy, Experiment, now_iso, read_json, write_json
from cap_harness.vla.libero_compat import REPOSITORY_ROOT

SERVICE_PORTS = (8114, 8115, 8116)
_SEED_DONE = re.compile(r"\[(\d+)/(\d+)\]\s*$")


# --------------------------------------------------------------------- agent programs --
def agent_programs(session: Path, suite: str) -> dict[int, dict[str, Any]]:
    """The coding agents of a session (``agents/llm-<benchmark>-<suite>-<task>/``): each one's
    program, whether the agent has finished, and how its program did on its held-out scenes.
    """
    found: dict[int, dict[str, Any]] = {}
    root = Path(session) / "agents"
    if not root.is_dir():
        return found
    for workspace in sorted(root.glob("llm-*")):
        head, _, task = workspace.name.rpartition("-")
        if not task.isdigit() or not head.endswith(f"-{suite}"):
            continue
        held = read_json(workspace / "heldout" / "batch.json") or {}
        seeds = held.get("seeds")
        program = workspace / "program_best.py"
        found[int(task)] = {
            "workspace": workspace,
            "program": program if program.is_file() else None,
            # result.json is written when the agent's process ends, whatever the outcome.
            "finished": (workspace / "result.json").is_file(),
            "heldout_passes": held.get("passes"),
            "heldout_seeds": len(seeds) if isinstance(seeds, list) else seeds,
        }
    return found


def judge(
    entry: dict[str, Any], min_heldout: float, *, session_over: bool = False
) -> tuple[str, str]:
    """(state, why) for an agent's program: ready, agent_running, agent_stopped, no_program,
    unchecked, weak.
    """
    if not entry["finished"]:
        if session_over:
            return "agent_stopped", (
                "its coding agent stopped before it finished (see the session); check its "
                "program_best.py, then t2v.sh teacher-add NAME --program FILE --task N"
            )
        return "agent_running", "its coding agent is still working"
    if entry["program"] is None:
        return "no_program", "its coding agent finished without a program (program_best.py)"
    passes, seeds = entry["heldout_passes"], entry["heldout_seeds"]
    if passes is None or not seeds:
        return "unchecked", (
            "its coding agent did not run its held-out scenes; check the program, then "
            f"t2v.sh teacher-add NAME --program {entry['program']} --task N"
        )
    if passes / seeds < min_heldout:
        return "weak", (
            f"its program solved {passes}/{seeds} held-out scenes, below {min_heldout:.0%}; "
            "not used (fix it, then t2v.sh teacher-add)"
        )
    return "ready", f"held-out {passes}/{seeds}"


# ------------------------------------------------------------------------------ cells --
@dataclass
class Cell:
    """One scene set of one task: the unit the teacher records and prepare converts."""

    scene_set: str
    task: int
    state: str = "pending"  # pending | recording | recorded | incomplete | busy
    attempts: int = 0
    missing: int | None = None
    note: str = ""
    prepare: str = ""  # "" | queued | running | finished | problems
    prepare_log: str = ""

    @property
    def key(self) -> str:
        return f"{self.scene_set}/{self.task}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "attempts": self.attempts,
            "missing": self.missing,
            "note": self.note,
            "prepare": self.prepare,
            "prepare_log": self.prepare_log,
        }


class Collector:
    def __init__(
        self,
        experiment: Experiment,
        sets: list[SceneSet],
        tasks: Iterable[int],
        *,
        workers: int = 2,
        prepare_workers: int = 4,
        prepare_jobs: int = 1,
        session: Path | None = None,
        min_heldout: float = 0.5,
        follow: bool = False,
        poll: float = 60.0,
        attempts: int = 3,
        videos: bool = True,
        service_wait: float = 1800.0,
        argv: list[str] | None = None,
        log: Callable[[str], None] = print,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.experiment = experiment
        self.sets = list(sets)
        self.by_name = {s.name: s for s in self.sets}
        self.tasks = list(tasks)
        self.workers, self.prepare_workers, self.prepare_jobs = (
            workers,
            prepare_workers,
            prepare_jobs,
        )
        self.session = Path(session).resolve() if session else None
        self.min_heldout, self.follow, self.poll = min_heldout, follow, poll
        self.attempts, self.videos, self.service_wait = attempts, videos, service_wait
        self.out, self.sleep = log, sleep
        self.cells = {(s.name, t): Cell(s.name, t) for t in self.tasks for s in self.sets}
        self.queue: list[Cell] = []
        self.jobs: list[tuple[Cell, subprocess.Popen[bytes], Any]] = []
        self.program_states: dict[int, dict[str, str]] = {}
        self.service_failures = 0
        self.recorded_seconds = 0.0
        self.recorded_episodes = 0
        self.status: dict[str, Any] = {
            "status": "starting",
            "pid": os.getpid(),
            "started": now_iso(),
            "command": argv,
            "scene_sets": [s.name for s in self.sets],
            "tasks": self.tasks,
            "workers": workers,
            "prepare_workers": prepare_workers,
            "programs_from": str(self.session) if self.session else None,
            "log_file": _own_log(),
            "follow": follow,
            "now": None,
            "log": [],
        }

    # ------------------------------------------------------------------------ status --
    def log(self, line: str) -> None:
        self.out(line)
        self.status["log"] = [*self.status["log"], f"{time.strftime('%H:%M:%S')} {line}"][-30:]

    def save(self, **changes: Any) -> None:
        self.status.update(changes)
        remaining = sum(c.missing or 0 for c in self.cells.values() if c.task in self._registered)
        per_episode = (
            self.recorded_seconds / self.recorded_episodes if self.recorded_episodes else None
        )
        self.status.update(
            updated=now_iso(),
            cells={c.key: c.to_dict() for c in self.cells.values()},
            programs={str(t): s for t, s in self.program_states.items()},
            prepare_running=[c.key for c, _, _ in self.jobs],
            prepare_queued=[c.key for c in self.queue],
            episodes_left=remaining,
            seconds_per_episode=round(per_episode, 1) if per_episode else None,
            eta_s=round(remaining * per_episode) if per_episode else None,
        )
        write_json(self.experiment.root / STATUS_FILE, self.status)

    # ---------------------------------------------------------------------- programs --
    @property
    def _registered(self) -> set[int]:
        from cap_harness.trace2vla import teacher

        return set(teacher.programs(self.experiment))

    def take_programs(self) -> None:
        from cap_harness.trace2vla import teacher

        registered = self._registered
        agents = agent_programs(self.session, self.experiment.suite) if self.session else {}
        over = self.session is not None and self.session_over()
        for task in self.tasks:
            previous = self.program_states.get(task, {}).get("state")
            if task in registered:
                if previous not in ("registered", "taken"):
                    self.program_states[task] = {"state": "registered", "note": "registered"}
                continue
            entry = agents.get(task)
            if entry is None:
                state, note = (
                    "waiting",
                    (
                        "no coding agent for it in the session yet"
                        if self.session
                        else "no program (t2v.sh teacher-add)"
                    ),
                )
            else:
                state, note = judge(entry, self.min_heldout, session_over=over)
                if state == "ready":
                    teacher.register_program(self.experiment, task, entry["program"])
                    state, note = "taken", f"from its coding agent ({note})"
                    self.log(f"task {task}: program taken from {entry['workspace']} ({note})")
                    for scenes in self.sets:  # its scenes count in the time left from now on
                        self.cells[(scenes.name, task)].missing = len(
                            teacher.missing_seeds(self.experiment, scenes, task)
                        )
            if state not in (previous, "taken"):
                self.log(f"task {task}: {note}")
            self.program_states[task] = {"state": state, "note": note}
            if entry is not None:
                self.program_states[task]["workspace"] = str(entry["workspace"])

    def waiting_for_programs(self) -> list[int]:
        return [t for t in self.tasks if t not in self._registered]

    def session_over(self) -> bool:
        """True when nothing more will come: the session finished, or there is none to wait on
        and every task has a program.
        """
        if self.session is None:
            return not self.waiting_for_programs()
        meta = read_json(self.session / "session.json") or {}
        return meta.get("status") == "complete" or bool(meta.get("finished"))

    # ------------------------------------------------------------------------- teacher --
    def scan(self) -> None:
        """How many scenes each cell of a task with a program still misses."""
        from cap_harness.trace2vla import teacher

        registered = self._registered
        for cell in self.cells.values():
            if cell.task in registered:
                cell.missing = len(teacher.missing_seeds(self.experiment, self.by_name[cell.scene_set], cell.task))  # fmt: skip

    def next_cell(self) -> Cell | None:
        """The next cell to record, task by task; cells found complete are queued to prepare."""
        from cap_harness.trace2vla import teacher

        registered = self._registered
        for task in self.tasks:
            if task not in registered:
                continue
            for scenes in self.sets:
                cell = self.cells[(scenes.name, task)]
                if cell.state in ("recorded", "incomplete", "busy"):
                    continue
                cell.missing = len(teacher.missing_seeds(self.experiment, scenes, task))
                if not cell.missing:
                    cell.state = "recorded"
                    self.queue_prepare(cell)
                    continue
                if cell.attempts >= self.attempts:
                    cell.state = "incomplete"
                    cell.note = (
                        f"{cell.missing} of {len(scenes.seeds)} scenes still not recorded after "
                        f"{cell.attempts} tries (see teacher/{scenes.name}/task_{task:02d}/logs)"
                    )
                    self.log(f"{cell.key}: {cell.note}")
                    self.queue_prepare(cell)
                    continue
                return cell
        return None

    def wait_for_services(self) -> None:
        from cap_harness.trace2vla import teacher

        waited = 0.0
        down = teacher.service_ports_down(SERVICE_PORTS)
        while down:
            if waited >= self.service_wait:
                raise RuntimeError(
                    f"the model services on ports {down} have not answered for "
                    f"{waited / 60:.0f} min; start them (scripts/gpu/services.sh start) and run "
                    "the same command again: it continues where it stopped"
                )
            if not waited:
                self.log(f"model services on ports {down} are not answering; waiting for them")
            self.save(now={"stage": "waiting", "detail": f"model services on ports {down}", "since": now_iso()})  # fmt: skip
            self.poll_jobs()
            self.sleep(60.0)
            waited += 60.0
            down = teacher.service_ports_down(SERVICE_PORTS)

    def _teacher_line(self, line: str) -> None:
        self.out(line)
        seen = _SEED_DONE.search(line)
        now = self.status.get("now") or {}
        if seen and now.get("stage") == "teacher":
            now.update(done=int(seen.group(1)), total=int(seen.group(2)))
            self.save(now=now)

    def record(self, cell: Cell) -> None:
        from cap_harness.trace2vla import teacher

        scenes = self.by_name[cell.scene_set]
        self.wait_for_services()
        cell.attempts += 1
        cell.state = "recording"
        before = cell.missing or 0
        self.log(f"{cell.key}: recording {before} scenes (try {cell.attempts})")
        self.save(now={"stage": "teacher", "scene_set": cell.scene_set, "task": cell.task,
                       "done": 0, "total": before, "since": now_iso()})  # fmt: skip
        started = time.monotonic()
        try:
            teacher.run_scene_set(
                self.experiment, scenes, [cell.task], workers=self.workers, videos=self.videos,
                log=self._teacher_line,
            )  # fmt: skip
        except Busy as exc:
            cell.attempts -= 1
            cell.state, cell.note = "busy", str(exc)
            self.log(f"{cell.key}: {exc}; coming back to it later")
            self.save(now=None)
            return
        except RuntimeError as exc:  # every episode failed on a model service
            # Not the cell's fault: it keeps its tries. Three such runs in a row mean the services
            # are broken (out of memory, say) while their ports still answer: stop and say so.
            cell.attempts -= 1
            cell.state, cell.note = "pending", str(exc)
            self.service_failures += 1
            self.log(f"{cell.key}: {exc}")
            if self.service_failures >= 3:
                raise RuntimeError(
                    "the model services failed every episode of three runs in a row; check "
                    "scripts/gpu/services.sh status and the GPU's free memory, then run the same "
                    "command again: it continues where it stopped"
                ) from exc
            self.save(now={"stage": "waiting", "detail": "model services failing; waiting before trying again", "since": now_iso()})  # fmt: skip
            self.sleep(self.poll * 5)
            return
        self.service_failures = 0
        cell.missing = len(teacher.missing_seeds(self.experiment, scenes, cell.task))
        if before > cell.missing:
            self.recorded_seconds += time.monotonic() - started
            self.recorded_episodes += before - cell.missing
        if not cell.missing:
            cell.state, cell.note = "recorded", ""
            self.queue_prepare(cell)
        else:
            cell.state = "pending"
            cell.note = f"{cell.missing} scenes left after try {cell.attempts}"
        self.save(now=None)

    # ------------------------------------------------------------------------- prepare --
    def queue_prepare(self, cell: Cell) -> None:
        if cell.prepare:
            return
        cell.prepare = "queued"
        self.queue.append(cell)
        self.poll_jobs()

    def prepare_command(self, scene_set: str, tasks: list[int]) -> list[str]:
        # Test scenes are graded (the teacher's failures get labels) but never become demos.
        stage = "teacher-grade" if self.by_name[scene_set].role == "test" else "prepare"
        return [
            sys.executable, "-m", "cap_harness.trace2vla", stage, str(self.experiment.root),
            "--scenes", scene_set, "--tasks", ",".join(str(t) for t in tasks),
            "--workers", str(self.prepare_workers),
        ]  # fmt: skip

    def _environment(self) -> dict[str, str]:
        environment = dict(os.environ)
        # t2v.sh passes its --gpu here: grading and conversion render there, while the teacher
        # keeps the GPU of ASPIRE's services.
        gpu = environment.get("T2V_PREPARE_GPU")
        if gpu:
            environment.update(CUDA_VISIBLE_DEVICES=gpu, MUJOCO_EGL_DEVICE_ID=gpu)
        return environment

    def _start(self, cell: Cell) -> None:
        path = self.logs / f"{time.strftime('%Y%m%d-%H%M%S')}-prepare-{cell.scene_set}-task{cell.task:02d}.log"  # fmt: skip
        stream = path.open("wb")
        process = subprocess.Popen(
            self.prepare_command(cell.scene_set, [cell.task]),
            stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            env=self._environment(), cwd=REPOSITORY_ROOT,
        )  # fmt: skip
        cell.prepare, cell.prepare_log = "running", self.experiment.rel(path)
        self.jobs.append((cell, process, stream))
        self.log(f"{cell.key}: grading and converting in the background (pid {process.pid})")

    def poll_jobs(self) -> None:
        for job in list(self.jobs):
            cell, process, stream = job
            code = process.poll()
            if code is None:
                continue
            stream.close()
            self.jobs.remove(job)
            # prepare also exits 1 when a task kept no demo (the teacher solved none of it).
            cell.prepare = "finished" if code == 0 else "problems"
            self.log(f"{cell.key}: grading and conversion {cell.prepare} (exit {code}; {cell.prepare_log})")  # fmt: skip
        while self.queue and len(self.jobs) < self.prepare_jobs:
            self._start(self.queue.pop(0))
        self.save()

    def drain(self) -> None:
        while self.jobs or self.queue:
            self.save(now={"stage": "prepare", "detail": "grading and converting the last ones", "since": now_iso()})  # fmt: skip
            self.sleep(5.0)
            self.poll_jobs()

    def final_pass(self) -> None:
        """One more grade-and-convert of every scene set over every task with a program: picks
        up anything a background job missed (cheap: finished work is skipped).
        """
        registered = sorted(self._registered & set(self.tasks))
        if not registered:
            return
        for scenes in self.sets:
            self.save(now={"stage": "prepare", "detail": f"final pass over '{scenes.name}'", "since": now_iso()})  # fmt: skip
            path = self.logs / f"{time.strftime('%Y%m%d-%H%M%S')}-final-{scenes.name}.log"
            with path.open("wb") as stream:
                code = subprocess.call(
                    self.prepare_command(scenes.name, registered),
                    stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                    env=self._environment(), cwd=REPOSITORY_ROOT,
                )  # fmt: skip
            if code:
                self.log(f"final pass over '{scenes.name}': exit {code} ({self.experiment.rel(path)})")  # fmt: skip

    @property
    def logs(self) -> Path:
        """The background jobs' logs: their own folder, so logs/ keeps the stages' logs."""
        folder = self.experiment.root / "logs" / "collect"
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    # ---------------------------------------------------------------------------- run --
    def summary(self) -> dict[str, Any]:
        cells = list(self.cells.values())
        return {
            "cells": len(cells),
            "recorded": sum(1 for c in cells if c.state == "recorded"),
            "incomplete": [c.key for c in cells if c.state == "incomplete"],
            "without_program": self.waiting_for_programs(),
            "prepare_problems": [c.key for c in cells if c.prepare == "problems"],
        }

    def loop(self) -> None:
        while True:
            self.take_programs()
            self.poll_jobs()
            cell = self.next_cell()
            if cell is not None:
                self.record(cell)
                continue
            busy = [c for c in self.cells.values() if c.state == "busy"]
            waiting = self.waiting_for_programs()
            if busy or (waiting and self.follow and not self.session_over()):
                detail = (
                    f"programs for tasks {compact(waiting)}"
                    if waiting
                    else f"{len(busy)} scene sets another process is recording"
                )
                self.save(status="waiting", now={"stage": "waiting", "detail": detail, "since": now_iso()})  # fmt: skip
                self.sleep(self.poll)
                for c in busy:
                    c.state = "pending"
                self.status["status"] = "running"
                continue
            return

    def run(self) -> int:
        with self.experiment.hold(["collect"], "t2v.sh collect"):
            self.experiment.log(
                "collect_started", scene_sets=self.status["scene_sets"], tasks=self.tasks,
                programs_from=self.status["programs_from"], follow=self.follow,
            )  # fmt: skip
            self.take_programs()
            self.scan()
            self.save(status="running")
            try:
                self.loop()
                self.drain()
                self.final_pass()
            except KeyboardInterrupt:
                self.save(status="stopped", now=None)
                raise
            except BaseException as exc:
                self.log(f"stopped: {type(exc).__name__}: {exc}")
                self.save(status="failed", error=f"{type(exc).__name__}: {exc}", now=None)
                self.drain()  # the background jobs need no services: let them finish
                self.save(status="failed", now=None)
                raise
            summary = self.summary()
            self.scan()
            self.save(status="finished", now=None, summary=summary)
            self.experiment.log("collect_finished", **summary)
            for key in summary["incomplete"]:
                self.log(f"incomplete: {key}: {self.cells[tuple_key(key)].note}")
            if summary["without_program"]:
                self.log(f"no program, not collected: tasks {compact(summary['without_program'])}")
            return 1 if summary["incomplete"] or summary["without_program"] else 0


def _own_log() -> str | None:
    """The file this process's output goes to (t2v.sh --detach writes it to the experiment's
    logs/), for the dashboard to link; None on a terminal.
    """
    try:
        target = os.readlink("/proc/self/fd/1")
    except OSError:
        return None
    return target if target.startswith("/") and target.endswith(".log") else None


def tuple_key(key: str) -> tuple[str, int]:
    scene_set, _, task = key.rpartition("/")
    return scene_set, int(task)
