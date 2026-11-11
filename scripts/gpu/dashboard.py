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
from dashboard_store import MEDIA_SUFFIXES, Store, dir_size  # noqa: E402
from dashboard_vla import VlaStore  # noqa: E402

UI_DIR = HERE / "dashboard"
UI = UI_DIR / "index.html"
LIVE_VENVS = {"robosuite": ".venv-robosuite", "libero-pro": ".venv-libero"}
STATIC_TYPES = {".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml"}


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
    vla = VlaStore(store)

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
                if path.startswith("/static/"):
                    return self._static(path[len("/static/") :])
                if path == "/api/benchmark":
                    paths = parse_qs(urlparse(self.path).query).get("path", [])
                    if not paths:
                        raise ValueError("pass one or more ?path= runs to compare")
                    return self._json(store.benchmark([store.resolve(p) for p in paths]))
                if path == "/api/log":
                    return self._json(
                        store.log(
                            store.resolve(query["path"]),
                            query.get("q", ""),
                            query.get("alerts") == "1",
                            int(query.get("tail", "4000")),
                        )
                    )
                if path == "/api/collections":
                    return self._json({"outputs": str(store.outputs), "items": store.collections()})
                if path == "/api/collection":
                    return self._json(store.collection(store.resolve(query["path"])))
                if path == "/api/episode":
                    return self._json(store.episode(store.resolve(query["path"])))
                if path == "/api/call":
                    return self._json(store.call(store.resolve(query["path"])))
                if path == "/api/scene":
                    return self._json(store.scene(store.resolve(query["path"])))
                if path == "/api/steps":
                    return self._json(
                        store.steps(
                            store.resolve(query["path"]),
                            float(query.get("start", "0")),
                            float(query.get("end", "1e9")),
                        )
                    )
                if path == "/api/array":
                    return self._json(store.array(store.resolve(query["path"])))
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
                    paths = parse_qs(urlparse(self.path).query).get("path", [])
                    if not paths:
                        raise ValueError("pass one or more ?path= to download")
                    return self._zip(
                        [store.resolve(p) for p in paths],
                        query.get("media", "1") == "1",
                        query.get("name"),
                    )
                if path == "/api/vla/runs":
                    return self._json(vla.runs())
                if path == "/api/vla/run":
                    return self._json(vla.run(store.resolve(query["path"])))
                if path == "/api/vla/episode":
                    return self._json(vla.episode(store.resolve(query["path"])))
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

        def _static(self, rel: str) -> None:
            base = (UI_DIR / "static").resolve()
            path = (base / unquote(rel)).resolve()
            if base not in path.parents or not path.is_file():
                raise FileNotFoundError(f"no such asset: {rel}")
            kind = STATIC_TYPES.get(path.suffix, "application/octet-stream")
            return self._bytes(path.read_bytes(), f"{kind}; charset=utf-8")

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

        def _zip(self, paths: list[Path], media: bool, name: str | None = None) -> None:
            """Stream a zip of one or more files or folders, with or without media."""
            for path in paths:
                if not path.exists():
                    raise FileNotFoundError(f"no such path: {store.rel(path)}")
            # Entries are named from the folder that holds them all, so a zip of several
            # runs keeps their layout (dev_v03/.../0051/run-x/...).
            common = Path(os.path.commonpath([str(p) for p in paths]))
            anchor = common.parent
            stem = re.sub(r"[^\w.-]+", "-", name or common.name or "runs")[:80]
            if len(paths) > 1 and not name:
                stem = f"{stem}-{len(paths)}-runs"
            filename = f"{stem}{'' if media else '-logs'}.zip"
            self.send_response(200)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            files: list[Path] = []
            for path in paths:
                files += (
                    [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
                )
            stream = _ZipStream(self.wfile)
            with zipfile.ZipFile(stream, "w", allowZip64=True) as archive:  # type: ignore[arg-type]
                for file in dict.fromkeys(files):
                    if not media and file.suffix.lower() in MEDIA_SUFFIXES:
                        continue
                    compress = (
                        zipfile.ZIP_STORED
                        if file.suffix.lower() in MEDIA_SUFFIXES
                        else zipfile.ZIP_DEFLATED
                    )
                    try:
                        archive.write(
                            file, file.relative_to(anchor).as_posix(), compress_type=compress
                        )
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
