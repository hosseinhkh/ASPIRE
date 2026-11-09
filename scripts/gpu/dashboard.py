#!/usr/bin/env python3
"""ASPIRE dashboard: browse every recorded run, download complete logs, drive live robots.

  python scripts/gpu/dashboard.py [--port 8765] [--host 127.0.0.1] [--outputs outputs]

Without --port it takes 8765 or the next free port, and prints the address and the
ssh tunnel command to reach it from a laptop.
Standard library only; the live consoles it starts run in the simulator venvs.

  Runs      sessions (scripts/gpu/session.sh), live-console runs, checks, demos, campaigns
  Episode   videos, keyframes, SAM3 overlays, trajectories, every call with its full input,
            output and error, the program, its output, and the console commands of a live run
  Agent     the whole Claude Code conversation: prompt, text, thinking, tool calls, results,
            images it was shown
  Download  any session / episode / folder as a zip, complete or logs only (no videos/images)
  Live      start a simulator (Robosuite, LIBERO-Pro) as a live console, see its cameras and
            state, call SAM3 / Contact-GraspNet / PyRoki / cuRobo through the robot API, run
            CaP programs, then close it: the session is recorded like any episode
"""

from __future__ import annotations

import argparse
import base64
import binascii
import errno
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, unquote, urlparse
from urllib.request import Request, urlopen
import zipfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import session_report as report  # noqa: E402

UI = HERE / "dashboard" / "index.html"
COLLECTION_KINDS = {  # outputs/<dir>: what one entry inside it is
    "sessions": "session",
    "gpu-demo": "demo",
    "gpu-check": "check",
    "runs": "campaign",
}
MEDIA_SUFFIXES = {".mp4", ".png", ".jpg", ".jpeg", ".npz", ".npy", ".webm", ".gif"}
MAX_POINTS = 600
LIVE_VENVS = {"robosuite": ".venv-robosuite", "libero-pro": ".venv-libero"}


def read_json(path: Path) -> object:
    return report.read_json(path)


def jsonable(value: object) -> object:
    return json.loads(json.dumps(value, default=str))


def downsample(values: list[float], limit: int = MAX_POINTS) -> list[float]:
    if len(values) <= limit:
        return values
    step = len(values) / limit
    return [values[int(i * step)] for i in range(limit)]


def dir_size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


# ---------------------------------------------------------------------------- discovery --
class Store:
    def __init__(self, outputs: Path) -> None:
        self.outputs = outputs.resolve()

    def resolve(self, rel: str) -> Path:
        """A path inside the outputs directory, or ValueError."""
        path = (self.outputs / unquote(rel or "")).resolve()
        if path != self.outputs and self.outputs not in path.parents:
            raise ValueError("path outside the outputs directory")
        return path

    def rel(self, path: Path) -> str:
        return path.resolve().relative_to(self.outputs).as_posix()

    def runs_in(self, root: Path) -> list[Path]:
        if (root / "run.json").exists():
            return [root]
        return sorted(
            p.parent
            for p in root.rglob("run.json")
            if "wandb" not in p.parts and "report" not in p.parts
        )

    def collections(self) -> list[dict[str, object]]:
        items: list[dict[str, object]] = []
        for folder, kind in COLLECTION_KINDS.items():
            base = self.outputs / folder
            if not base.is_dir():
                continue
            for entry in sorted(base.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                if entry.is_dir() and not entry.name.startswith("."):
                    items.append(self._summary(entry, kind))
        live = self.outputs / "live"
        if live.is_dir():
            for run in sorted(self.runs_in(live), key=lambda p: p.stat().st_mtime, reverse=True):
                items.append(self._summary(run, "live"))
        return items

    def _summary(self, path: Path, kind: str) -> dict[str, object]:
        meta = read_json(path / "session.json") or {}
        if not isinstance(meta, dict):
            meta = {}
        item: dict[str, object] = {
            "path": self.rel(path),
            "name": meta.get("name") or path.name,
            "kind": kind,
            "modified": path.stat().st_mtime,
            "status": meta.get("status"),
            "stage_counts": meta.get("stage_counts"),
        }
        if kind == "live":
            run = read_json(path / "run.json") or {}
            outcome = read_json(path / "outcome.json")
            item.update(
                name=f"{run.get('suite', '?')}:{run.get('task_id', '?')} seed {run.get('seed', '?')}",
                status="finalized" if outcome else "open or interrupted",
                episodes=1,
            )
        else:
            item["episodes"] = sum(1 for p in path.rglob("outcome.json") if "wandb" not in p.parts)
        return item

    # -- one collection --
    def collection(self, path: Path) -> dict[str, object]:
        meta = read_json(path / "session.json") or {}
        gpu = report.load_gpu(path)
        stages = report.load_stages(path, gpu)
        for number, row in enumerate(stages, 1):
            row["n"] = number
        times = gpu["times"]
        gpu_view = None
        if times:
            idx = list(range(len(times)))
            if len(idx) > MAX_POINTS:
                idx = [int(i * len(times) / MAX_POINTS) for i in range(MAX_POINTS)]
            gpu_view = {
                "target": gpu["target"],
                "t": [times[i] for i in idx],
                "used": [gpu["used"][times[i]] for i in idx],
                "util": [gpu["util"][times[i]] for i in idx],
                "yours": [gpu["yours"].get(times[i], 0.0) for i in idx],
                "reserved": [gpu["reserved"].get(times[i], 0.0) for i in idx],
            }
        episodes = []
        for run in self.runs_in(path):
            run_meta = read_json(run / "run.json") or {}
            outcome = read_json(run / "outcome.json") or {}
            calls = len(list((run / "trace/calls").glob("call-*")))
            episodes.append(
                {
                    "path": self.rel(run),
                    "where": run.relative_to(path).as_posix(),
                    "benchmark": run_meta.get("benchmark"),
                    "suite": run_meta.get("suite"),
                    "task_id": run_meta.get("task_id"),
                    "task_name": run_meta.get("task_name"),
                    "seed": run_meta.get("seed"),
                    "program_ok": outcome.get("program_ok"),
                    "task_success": outcome.get("task_success"),
                    "termination": outcome.get("termination_reason"),
                    "steps": outcome.get("steps_executed"),
                    "wall_s": outcome.get("wall_duration_s"),
                    "calls": calls,
                    "agent": run.relative_to(path).parts[0] == "agents",
                    "finalized": bool(outcome),
                }
            )
        agents = []
        transcripts = [p for p in path.rglob("transcript.jsonl") if "wandb" not in p.parts]
        transcripts += sorted((path / "claude-sessions").rglob("*.jsonl"))
        for transcript in transcripts:
            agents.append(
                {
                    "path": self.rel(transcript),
                    "title": transcript.relative_to(path).as_posix(),
                    **self._agent_stats(transcript),
                }
            )
        files = []
        for candidate in (path / "session.log", path / "stages.tsv", path / "session.json"):
            if candidate.exists():
                files.append(candidate)
        for folder in ("env", "preflight", "service-logs", "episodes/logs", "gpu"):
            files += sorted(p for p in (path / folder).glob("*") if p.is_file())
        return {
            "path": self.rel(path),
            "meta": meta,
            "stages": jsonable(stages),
            "gpu": gpu_view,
            "episodes": episodes,
            "agents": agents,
            "files": [
                {
                    "path": self.rel(f),
                    "name": f.relative_to(path).as_posix(),
                    "size": f.stat().st_size,
                }
                for f in files
            ],
            "has_report": (path / "report/index.html").exists(),
        }

    def _agent_stats(self, transcript: Path) -> dict[str, object]:
        stats: dict[str, object] = {"tool_calls": 0, "tool_errors": 0}
        for event in report.transcript_events(transcript):
            if event.get("type") == "result":
                stats.update(
                    result=event.get("subtype"),
                    turns=event.get("num_turns"),
                    cost_usd=event.get("total_cost_usd"),
                    minutes=round((event.get("duration_ms") or 0) / 60000, 1),
                )
            for block in (event.get("message") or {}).get("content") or []:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    stats["tool_calls"] = int(stats["tool_calls"]) + 1
                if (
                    isinstance(block, dict)
                    and block.get("type") == "tool_result"
                    and block.get("is_error")
                ):
                    stats["tool_errors"] = int(stats["tool_errors"]) + 1
        return stats

    # -- one episode --
    def episode(self, run: Path) -> dict[str, object]:
        meta = read_json(run / "run.json") or {}
        outcome = read_json(run / "outcome.json") or {}
        result = read_json(run / "source/program-result.json") or {}
        calls = []
        for row in report.call_rows(run):
            output = read_json(row["dir"] / "output.json")
            calls.append(
                {
                    "index": row["index"],
                    "name": row["name"],
                    "category": row["category"],
                    "depth": row["depth"],
                    "ok": row["ok"],
                    "duration_s": row["duration_s"],
                    "sim_time_s": row["sim_time_s"],
                    "error": row["error"],
                    "dir": self.rel(row["dir"]),
                    "summary": _short(output),
                }
            )
        charts = {
            title: {name: downsample(values) for name, values in series.items()}
            for title, series in report.trajectories(run).items()
        }
        media = {
            "videos": [self.rel(p) for p in sorted((run / "media/videos").glob("*.mp4"))],
            "keyframes": [self.rel(p) for p in sorted((run / "media/keyframes").glob("*/*.png"))],
            "overlays": [self.rel(p) for p in sorted((run / "media/overlays").rglob("*.png"))],
        }
        texts = {}
        for name, rel in (("program", "source/program.py"), ("stdout", "logs/program.stdout"),
                          ("stderr", "logs/program.stderr")):  # fmt: skip
            path = run / rel
            if path.exists():
                texts[name] = path.read_text(encoding="utf-8", errors="replace")[-200_000:]
        commands = []
        log = run / "logs/live_commands.jsonl"
        if log.exists():
            for line in log.read_text(encoding="utf-8").splitlines():
                try:
                    commands.append(json.loads(line))
                except ValueError:
                    pass
        files = [
            {"path": self.rel(p), "name": p.relative_to(run).as_posix(), "size": p.stat().st_size}
            for p in sorted(run.rglob("*"))
            if p.is_file()
            and "trace/calls" not in p.relative_to(run).as_posix()
            and p.suffix.lower() not in MEDIA_SUFFIXES
        ]
        return {
            "path": self.rel(run),
            "meta": meta,
            "outcome": outcome,
            "result": jsonable(result.get("result")),
            "program_error": jsonable(result.get("error")),
            "timing": outcome.get("timing"),
            "calls": calls,
            "charts": charts,
            "media": media,
            "texts": texts,
            "commands": jsonable(commands),
            "files": files,
        }

    def call(self, call_dir: Path) -> dict[str, object]:
        data: dict[str, object] = {"dir": self.rel(call_dir)}
        for name in ("input", "output", "error"):
            path = call_dir / f"{name}.json"
            if path.exists():
                data[name] = read_json(path)
        data["other_files"] = [
            self.rel(p) for p in sorted(call_dir.iterdir())
            if p.is_file() and p.name not in {"input.json", "output.json", "error.json"}
        ]  # fmt: skip
        return data

    # -- one agent conversation --
    def agent(self, transcript: Path) -> dict[str, object]:
        workspace = transcript.parent
        events = []
        names: dict[str, str] = {}
        for i, event in enumerate(report.transcript_events(transcript)):
            kind = event.get("type")
            message = event.get("message") or {}
            if kind == "system" and event.get("subtype") == "init":
                events.append({"kind": "init", "model": event.get("model"),
                               "tools": event.get("tools"), "cwd": event.get("cwd")})  # fmt: skip
            elif kind == "result":
                events.append(
                    {
                        "kind": "result",
                        "subtype": event.get("subtype"),
                        "turns": event.get("num_turns"),
                        "cost_usd": event.get("total_cost_usd"),
                        "minutes": round((event.get("duration_ms") or 0) / 60000, 1),
                        "text": event.get("result"),
                    }
                )
            for j, block in enumerate(message.get("content") or []):
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    events.append({"kind": "text", "role": message.get("role") or kind,
                                   "text": block.get("text", "")})  # fmt: skip
                elif btype == "thinking":
                    events.append({"kind": "thinking", "text": block.get("thinking", "")})
                elif btype == "tool_use":
                    names[block.get("id", "")] = block.get("name", "?")
                    events.append({"kind": "tool_use", "name": block.get("name"),
                                   "input": block.get("input")})  # fmt: skip
                elif btype == "tool_result":
                    parts: list[dict[str, object]] = []
                    content = block.get("content")
                    for k, part in enumerate(content if isinstance(content, list) else [content]):
                        if isinstance(part, dict) and part.get("type") == "image":
                            parts.append({"type": "image", "ref": f"{i}.{j}.{k}"})
                        elif isinstance(part, dict):
                            parts.append(
                                {"type": "text", "text": str(part.get("text", ""))[:20000]}
                            )
                        elif part is not None:
                            parts.append({"type": "text", "text": str(part)[:20000]})
                    events.append(
                        {
                            "kind": "tool_result",
                            "name": names.get(block.get("tool_use_id", ""), "tool"),
                            "is_error": bool(block.get("is_error")),
                            "parts": parts,
                        }
                    )
        texts = {}
        for name in ("prompt.md", "REPORT.md"):
            if (workspace / name).exists():
                texts[name] = (workspace / name).read_text(encoding="utf-8", errors="replace")
        programs = {
            p.name: p.read_text(encoding="utf-8", errors="replace")
            for p in sorted(workspace.glob("program_*.py"))
        }
        return {"path": self.rel(transcript), "events": events, "texts": texts,
                "programs": programs, **self._agent_stats(transcript)}  # fmt: skip

    def agent_image(self, transcript: Path, ref: str) -> tuple[bytes, str]:
        i, j, k = (int(x) for x in ref.split("."))
        event = report.transcript_events(transcript)[i]
        block = (event.get("message") or {}).get("content")[j]
        content = block.get("content")
        part = (content if isinstance(content, list) else [content])[k]
        source = part.get("source") or {}
        return base64.b64decode(source.get("data", "")), source.get("media_type", "image/png")


def _short(value: object, limit: int = 160) -> str:
    if isinstance(value, dict):
        error = value.get("error")
        if isinstance(error, dict):
            return f"{error.get('code')}: {error.get('message')}"[:limit]
        keys = [k for k in value if not k.startswith("$")][:6]
        return ("{" + ", ".join(keys) + "}")[:limit]
    return str(value)[:limit] if value is not None else ""


# ---------------------------------------------------------------------------- live --
class LiveManager:
    """Start, list, proxy and stop live consoles (python -m cap_harness.live)."""

    def __init__(self, store: Store, external: list[str]) -> None:
        self.store = store
        self.consoles: dict[str, dict[str, object]] = {}
        self.lock = threading.Lock()
        for number, url in enumerate(external, 1):
            self.consoles[f"ext{number}"] = {"id": f"ext{number}", "url": url.rstrip("/"),
                                             "external": True, "label": url}  # fmt: skip

    @staticmethod
    def _free_port() -> int:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    def start(self, spec: dict[str, object]) -> dict[str, object]:
        benchmark = str(spec.get("benchmark", "robosuite"))
        if benchmark not in LIVE_VENVS:
            raise ValueError(f"benchmark must be one of {', '.join(LIVE_VENVS)}")
        python = REPO / LIVE_VENVS[benchmark] / "bin/python"
        if not python.exists():
            raise ValueError(f"{python} is missing (run scripts/gpu/setup_conda_env.sh)")
        suite = str(spec.get("suite") or "")
        if not re.fullmatch(r"[A-Za-z0-9_]+", suite):
            raise ValueError("suite must be a benchmark suite name")
        port = self._free_port()
        console_id = f"live{port}"
        output_root = self.store.outputs / "live"
        output_root.mkdir(parents=True, exist_ok=True)
        log = output_root / f"console-{port}.log"
        command = [
            str(python), "-m", "cap_harness.live", "--benchmark", benchmark, "--suite", suite,
            "--task-id", str(int(spec.get("task_id", 0))), "--seed", str(int(spec.get("seed", 1))),
            "--port", str(port), "--output-root", str(output_root),
            "--camera-width", str(int(spec.get("camera_width", 512))),
            "--camera-height", str(int(spec.get("camera_height", 512))),
        ]  # fmt: skip
        if benchmark == "libero-pro" and spec.get("init_mode") in ("saved", "seeded"):
            command += ["--init-mode", str(spec["init_mode"])]
        handle = log.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True
        )
        entry = {
            "id": console_id,
            "url": f"http://127.0.0.1:{port}",
            "external": False,
            "label": f"{benchmark} {suite}:{spec.get('task_id', 0)} seed {spec.get('seed', 1)}",
            "process": process,
            "log": self.store.rel(log),
            "started": time.time(),
        }
        with self.lock:
            self.consoles[console_id] = entry
        return self.view(entry)

    def view(self, entry: dict[str, object]) -> dict[str, object]:
        process = entry.get("process")
        state = "external"
        if isinstance(process, subprocess.Popen):
            code = process.poll()
            state = "running" if code is None else f"exited ({code})"
        ready = False
        try:
            with urlopen(f"{entry['url']}/api/info", timeout=1.5) as response:
                ready = response.status == 200
        except (OSError, URLError, HTTPError):
            ready = False
        return {key: value for key, value in entry.items() if key != "process"} | {
            "state": state,
            "ready": ready,
        }

    def list(self) -> list[dict[str, object]]:
        with self.lock:
            entries = list(self.consoles.values())
        return [self.view(entry) for entry in entries]

    def get(self, console_id: str) -> dict[str, object]:
        with self.lock:
            entry = self.consoles.get(console_id)
        if entry is None:
            raise KeyError(console_id)
        return entry

    def stop(self, console_id: str) -> dict[str, object]:
        entry = self.get(console_id)
        try:
            request = Request(f"{entry['url']}/api/close", data=b"{}", method="POST")
            with urlopen(request, timeout=10) as response:
                response.read()
        except (OSError, URLError, HTTPError):
            pass
        process = entry.get("process")
        if isinstance(process, subprocess.Popen):
            try:
                process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                process.terminate()
        return self.view(entry)

    def proxy(
        self, console_id: str, path: str, method: str, body: bytes | None
    ) -> tuple[int, str, bytes]:
        entry = self.get(console_id)
        request = Request(f"{entry['url']}{path}", data=body, method=method,
                          headers={"Content-Type": "application/json"})  # fmt: skip
        try:
            with urlopen(request, timeout=30) as response:
                return response.status, response.headers.get("Content-Type", ""), response.read()
        except HTTPError as exc:
            return exc.code, exc.headers.get("Content-Type", "application/json"), exc.read()
        except (OSError, URLError) as exc:
            return (
                502,
                "application/json",
                json.dumps({"error": f"console unreachable: {exc}"}).encode(),
            )

    def shutdown(self) -> None:
        for entry in list(self.consoles.values()):
            if (
                isinstance(entry.get("process"), subprocess.Popen)
                and entry["process"].poll() is None
            ):
                self.stop(str(entry["id"]))


# ---------------------------------------------------------------------------- http --
class _ZipStream:
    """A write-only file object over the response socket (zipfile streams to it)."""

    def __init__(self, wfile: object) -> None:
        self.wfile = wfile
        self.position = 0

    def write(self, data: bytes) -> int:
        self.wfile.write(data)  # type: ignore[attr-defined]
        self.position += len(data)
        return len(data)

    def tell(self) -> int:
        return self.position

    def flush(self) -> None:
        self.wfile.flush()  # type: ignore[attr-defined]


def make_handler(store: Store, live: LiveManager) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "aspire-dashboard/1"

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            if os.environ.get("ASPIRE_DASHBOARD_ACCESS_LOG"):
                super().log_message(format, *args)

        def _json(self, payload: object, status: int = 200) -> None:
            self._bytes(
                json.dumps(payload, default=str).encode("utf-8"), "application/json", status
            )

        def _bytes(self, body: bytes, content_type: str, status: int = 200,
                   extra: dict[str, str] | None = None) -> None:  # fmt: skip
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for key, value in (extra or {}).items():
                self.send_header(key, value)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _query(self) -> dict[str, str]:
            return {key: values[0] for key, values in parse_qs(urlparse(self.path).query).items()}

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            query = self._query()
            try:
                if path in ("/", "/index.html"):
                    return self._bytes(UI.read_bytes(), "text/html; charset=utf-8")
                if path == "/api/collections":
                    return self._json({"outputs": str(store.outputs), "items": store.collections()})
                if path == "/api/collection":
                    return self._json(store.collection(store.resolve(query["path"])))
                if path == "/api/episode":
                    return self._json(store.episode(store.resolve(query["path"])))
                if path == "/api/call":
                    return self._json(store.call(store.resolve(query["path"])))
                if path == "/api/agent":
                    return self._json(store.agent(store.resolve(query["path"])))
                if path == "/api/agent-image":
                    data, kind = store.agent_image(store.resolve(query["path"]), query["ref"])
                    return self._bytes(data, kind)
                if path == "/api/size":
                    return self._json({"bytes": dir_size(store.resolve(query["path"]))})
                if path.startswith("/files/"):
                    return self._file(store.resolve(path[len("/files/") :]))
                if path == "/download":
                    return self._zip(store.resolve(query["path"]), query.get("media", "1") == "1")
                if path == "/api/live":
                    return self._json({"consoles": live.list()})
                if path.startswith("/live/"):
                    return self._live("GET")
                return self._json({"error": "not found"}, 404)
            except (KeyError, ValueError, IndexError, binascii.Error) as exc:
                return self._json({"error": f"{type(exc).__name__}: {exc}"}, 400)
            except FileNotFoundError as exc:
                return self._json({"error": str(exc)}, 404)
            except (BrokenPipeError, ConnectionResetError):
                return None

        def do_HEAD(self) -> None:
            self.do_GET()

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            try:
                if path == "/api/live/start":
                    return self._json(live.start(json.loads(body or b"{}")))
                match = re.fullmatch(r"/api/live/([\w-]+)/stop", path)
                if match:
                    return self._json(live.stop(match.group(1)))
                if path.startswith("/live/"):
                    return self._live("POST", body)
                return self._json({"error": "not found"}, 404)
            except (KeyError, ValueError) as exc:
                return self._json({"error": f"{type(exc).__name__}: {exc}"}, 400)

        def _live(self, method: str, body: bytes | None = None) -> None:
            _, _, console_id, rest = self.path.split("/", 3)
            status, kind, data = live.proxy(console_id, "/" + rest, method, body)
            return self._bytes(data, kind or "application/octet-stream", status)

        def _file(self, path: Path) -> None:
            if not path.is_file():
                raise FileNotFoundError(f"no such file: {store.rel(path)}")
            size = path.stat().st_size
            kind = mimetypes.guess_type(path.name)[0] or "text/plain"
            if path.suffix in {".log", ".jsonl", ".tsv", ".stdout", ".stderr", ".err", ".diff",
                               ".py", ".md", ".txt", ".csv"}:  # fmt: skip
                kind = "text/plain; charset=utf-8"
            start, end = 0, size - 1
            ranged = self.headers.get("Range", "")
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", ranged.strip()) if ranged else None
            if match and size:
                if match.group(1):
                    start = int(match.group(1))
                    end = int(match.group(2)) if match.group(2) else size - 1
                else:
                    start = max(0, size - int(match.group(2) or 0))
                end = min(end, size - 1)
            length = max(0, end - start + 1)
            self.send_response(206 if match else 200)
            self.send_header("Content-Type", kind)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            if match:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            if self._query().get("download"):
                self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
            self.end_headers()
            if self.command == "HEAD":
                return
            with path.open("rb") as handle:
                handle.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = handle.read(min(1 << 20, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

        def _zip(self, path: Path, media: bool) -> None:
            if not path.exists():
                raise FileNotFoundError(f"no such path: {store.rel(path)}")
            name = f"{path.name}{'' if media else '-logs'}.zip"
            self.send_response(200)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Disposition", f'attachment; filename="{name}"')
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
            stream = _ZipStream(self.wfile)
            with zipfile.ZipFile(stream, "w", allowZip64=True) as archive:  # type: ignore[arg-type]
                for file in files:
                    if not media and file.suffix.lower() in MEDIA_SUFFIXES:
                        continue
                    arcname = (
                        (Path(path.name) / file.relative_to(path)).as_posix()
                        if path.is_dir()
                        else file.name
                    )
                    compress = (
                        zipfile.ZIP_STORED
                        if file.suffix.lower() in MEDIA_SUFFIXES
                        else zipfile.ZIP_DEFLATED
                    )
                    try:
                        archive.write(file, arcname, compress_type=compress)
                    except OSError:
                        continue

    return Handler


def bind_server(
    host: str, port: int | None, handler: type[BaseHTTPRequestHandler], default: int
) -> ThreadingHTTPServer:
    """Bind the dashboard to an explicit port, or to the first free port from the default.

    Shared machines often have the default taken; after 50 tries any free port will do.
    """
    candidates = [port] if port is not None else [*range(default, default + 50), 0]
    for candidate in candidates:
        try:
            return ThreadingHTTPServer((host, candidate), handler)
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                raise
    raise SystemExit(
        f"error: port {port} on {host} is already in use; pick another with --port, "
        "or leave --port out to use a free one"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, help="default: 8765, or the next free port")
    parser.add_argument("--outputs", default=str(REPO / "outputs"))
    parser.add_argument("--live", action="append", default=[],
                        help="URL of a live console started elsewhere (repeatable)")  # fmt: skip
    args = parser.parse_args()
    outputs = Path(args.outputs)
    outputs.mkdir(parents=True, exist_ok=True)
    store = Store(outputs)
    live = LiveManager(store, args.live)
    server = bind_server(args.host, args.port, make_handler(store, live), default=8765)
    server.daemon_threads = True
    port = server.server_address[1]
    print(f"ASPIRE dashboard on http://{args.host}:{port}  (outputs: {store.outputs})")
    if args.host in ("127.0.0.1", "localhost"):
        print(f"from your laptop: ssh -L {port}:localhost:{port} {socket.gethostname()}, "
              f"then open http://localhost:{port}")  # fmt: skip

    def interrupt(*_: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)  # kill / timeout: still close our live consoles
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        print("stopping live consoles started by the dashboard (their runs are finalized)")
        live.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
