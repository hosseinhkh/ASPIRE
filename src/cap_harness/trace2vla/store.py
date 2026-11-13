"""The experiment folder and its ledger."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
from typing import Any
import uuid

from cap_harness.vla.libero_compat import REPOSITORY_ROOT

OUTPUTS = REPOSITORY_ROOT / "outputs" / "trace2vla"
SCHEMA_VERSION = 1


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_write(path: Path, text: str) -> None:
    """Write through a temporary file and rename it into place. The temporary name is unique
    to the process and the call, so processes writing the same file at once (parallel
    conversion workers updating a summary) never share one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Experiment:
    """One experiment folder. Every stage reads and writes through this object."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.settings: dict[str, Any] = read_json(self.root / "experiment.json", {}) or {}

    # ------------------------------------------------------------------ open / create --
    @staticmethod
    def resolve(name: str | Path) -> Path:
        path = Path(name)
        if path.is_absolute() or path.exists():
            return path.resolve()
        return OUTPUTS / str(name)

    @classmethod
    def open(cls, name: str | Path) -> Experiment:
        root = cls.resolve(name)
        if not (root / "experiment.json").is_file():
            raise FileNotFoundError(
                f"no experiment at {root} (create one: t2v.sh init NAME --suite SUITE --tasks 0,1)"
            )
        return cls(root)

    @classmethod
    def create(cls, name: str | Path, settings: Mapping[str, Any]) -> Experiment:
        root = cls.resolve(name)
        if (root / "experiment.json").is_file():
            raise FileExistsError(f"experiment {root} already exists")
        root.mkdir(parents=True, exist_ok=True)
        write_json(
            root / "experiment.json",
            {"schema_version": SCHEMA_VERSION, "name": root.name, "created": now_iso(), **settings},
        )
        experiment = cls(root)
        experiment.log("experiment_created", settings=dict(settings))
        return experiment

    def save_settings(self) -> None:
        write_json(self.root / "experiment.json", self.settings)

    # ------------------------------------------------------------------------- ledger --
    def log(self, event: str, **fields: Any) -> dict[str, Any]:
        """Append one event to the ledger: what ran, on what, and what it produced."""
        record = {
            "time": now_iso(),
            "event": event,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            **fields,
        }
        with (self.root / "ledger.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        return record

    def events(self, kinds: Iterable[str] | None = None) -> list[dict[str, Any]]:
        wanted = set(kinds) if kinds else None
        records = []
        path = self.root / "ledger.jsonl"
        if not path.is_file():
            return records
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if wanted is None or record.get("event") in wanted:
                records.append(record)
        return records

    # -------------------------------------------------------------------------- paths --
    @property
    def suite(self) -> str:
        return str(self.settings["suite"])

    @property
    def benchmark(self) -> str:
        return str(self.settings.get("benchmark", "libero-pro"))

    @property
    def tasks(self) -> list[int]:
        return [int(t) for t in self.settings.get("tasks", [])]

    def rel(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return str(path)

    def program(self, task_id: int) -> Path:
        return self.root / "teacher" / "programs" / f"task_{task_id:02d}.py"

    def teacher_dir(self, scene_set: str, task_id: int) -> Path:
        return self.root / "teacher" / scene_set / f"task_{task_id:02d}"

    def grade_dir(self, scene_set: str, task_id: int) -> Path:
        return self.root / "grades" / scene_set / f"task_{task_id:02d}"

    def demo_dir(self, name: str) -> Path:
        return self.root / "demos" / name

    def dataset_dir(self, name: str) -> Path:
        return self.root / "datasets" / name

    def training_dir(self, name: str) -> Path:
        return self.root / "training" / name

    def eval_dir(self, name: str) -> Path:
        return self.root / "evals" / name

    # -------------------------------------------------------------------------- locks --
    @contextmanager
    def hold(self, keys: Iterable[str], what: str) -> Iterator[None]:
        """Claim each key for this process until the block ends, or raise Busy naming the
        process that holds one. A claim is an flock on ``locks/<key>.lock``: it ends with the
        process however that ends (Ctrl-C, kill, crash), so a claim is never left behind.
        """
        import fcntl

        folder = self.root / "locks"
        folder.mkdir(parents=True, exist_ok=True)
        held: list[Any] = []
        try:
            for key in keys:
                path = folder / f"{key}.lock"
                stream = path.open("a+", encoding="utf-8")
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    stream.seek(0)
                    holder = stream.read().strip() or "another process"
                    stream.close()
                    raise Busy(f"{key} is in use by {holder}; wait for it or stop it") from None
                stream.seek(0)
                stream.truncate()
                stream.write(
                    f"{what} (pid {os.getpid()} on {socket.gethostname()}, since {now_iso()})"
                )
                stream.flush()
                held.append(stream)
            yield
        finally:
            for stream in held:
                # Empty means free (read by the dashboard); a holder that died leaves its line,
                # with a pid that no longer runs.
                stream.seek(0)
                stream.truncate()
                stream.close()  # closing the file ends the flock


class Busy(RuntimeError):
    """Another process is working on the same scene set, demo set or task right now."""


def is_probe(name: str, meta: Mapping[str, Any]) -> bool:
    """True for a short evaluation of one saved training step (``t2v.sh eval-steps``).

    eval.json says so; evaluations written before it did are recognised by the name
    eval-steps gives them, ``RUN@STEP-TAG``.
    """
    if "probe" in meta:
        return bool(meta["probe"])
    run, step = meta.get("training_run"), meta.get("checkpoint_step")
    return run is not None and step is not None and name.startswith(f"{run}@{step}-")
