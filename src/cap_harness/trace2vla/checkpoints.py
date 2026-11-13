"""Where training checkpoints live, and moving them to a bigger disk.

A training run keeps LeRobot's output (``checkpoints/<step>/pretrained_model`` and
``training_state``, gigabytes each) in ``training/<run>/lerobot``. With a checkpoint root set, that
folder lives at ``<root>/<experiment>/<run>`` and ``training/<run>/lerobot`` is a link to it. Every
path the pipeline has recorded keeps working through the link, and the dashboard's downloads do
not follow it.

    t2v.sh checkpoints --set-root /data/hossein/aspire-checkpoints   once per machine
    t2v.sh checkpoints NAME            where each run's checkpoints are, and their size
    t2v.sh checkpoints NAME --move     move existing ones to the root (finished runs only)

The root is ``ASPIRE_CHECKPOINT_ROOT`` when set (empty: none), else the path in
``checkpoint-root.txt`` next to the experiments (``outputs/trace2vla/``), else none: checkpoints
stay in the run folder, as before. Standard library only.
"""

from __future__ import annotations

from collections.abc import Callable
import os
from pathlib import Path
import shutil
from typing import Any

from cap_harness.trace2vla.store import now_iso, read_json, write_json

ENV = "ASPIRE_CHECKPOINT_ROOT"
SETTING = "checkpoint-root.txt"
LINK = "lerobot"


# ----------------------------------------------------------------------------- the root --
def checkpoint_root(experiments_dir: Path) -> Path | None:
    """The checkpoint root for the experiments in ``experiments_dir``, if one is set."""
    if ENV in os.environ:
        value = os.environ[ENV].strip()
        return Path(value).expanduser() if value else None
    try:
        value = (Path(experiments_dir) / SETTING).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return Path(value).expanduser() if value else None


def set_root(experiments_dir: Path, root: Path | None) -> Path | None:
    """Remember ``root`` for the experiments in ``experiments_dir`` (None forgets it)."""
    setting = Path(experiments_dir) / SETTING
    if root is None:
        setting.unlink(missing_ok=True)
        return None
    root = Path(root).expanduser()
    if not root.is_absolute():
        raise ValueError(f"the checkpoint root must be an absolute path, not {root}")
    root.mkdir(parents=True, exist_ok=True)
    probe = root / ".aspire-write-test"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as error:
        raise ValueError(f"cannot write to {root}: {error}") from None
    setting.parent.mkdir(parents=True, exist_ok=True)
    setting.write_text(f"{root}\n", encoding="utf-8")
    return root


def external_dir(root: Path, run_dir: Path) -> Path:
    """Where a run's checkpoints go under ``root``: ``<root>/<experiment>/<run>``."""
    return Path(root) / run_dir.parent.parent.name / run_dir.name


def prepare_output(run_dir: Path, root: Path | None) -> Path:
    """The folder to give lerobot-train as its output for a new run.

    Without a root: ``run_dir/lerobot``, as before. With one: a folder under the root that does
    not exist yet (LeRobot creates it and refuses an existing one), with ``run_dir/lerobot``
    linking to it from the start, so the run's checkpoints are found the usual way.
    """
    link = run_dir / LINK
    if root is None:
        return link
    target = external_dir(root, run_dir)
    if target.exists() or target.is_symlink():
        raise FileExistsError(
            f"{target} already exists (another run wrote there): choose another run name"
        )
    if link.exists() or link.is_symlink():
        raise FileExistsError(f"{link} already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=True)
    return target


# --------------------------------------------------------------------------- what is where --
def folder_size(path: Path) -> int:
    """Bytes of the files under ``path`` (links are not followed)."""
    total = 0
    for folder, _dirs, files in os.walk(path):
        for name in files:
            file = Path(folder) / name
            if not file.is_symlink():
                try:
                    total += file.stat().st_size
                except OSError:
                    continue
    return total


def human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def location(run_dir: Path) -> dict[str, Any]:
    """Where a run's checkpoints are: ``inside`` the run folder, ``linked`` elsewhere,
    ``missing`` (a link whose target is gone), or ``none`` yet.
    """
    link = run_dir / LINK
    if link.is_symlink():
        target = Path(os.readlink(link))
        target = target if target.is_absolute() else (link.parent / target)
        if not target.is_dir():
            return {"kind": "missing", "path": str(target), "bytes": 0}
        return {"kind": "linked", "path": str(target.resolve()), "bytes": folder_size(target)}
    if link.is_dir():
        return {"kind": "inside", "path": str(link), "bytes": folder_size(link)}
    return {"kind": "none", "path": None, "bytes": 0}


def _alive(pid: object) -> bool:
    try:
        os.kill(int(pid), 0)  # type: ignore[arg-type]
    except (TypeError, ValueError, ProcessLookupError):
        return False
    except PermissionError:
        return True
    return True


def _running(folder: Path) -> bool:
    progress = read_json(folder / "progress.json") or {}
    return progress.get("status") == "running" and _alive(progress.get("pid"))


def busy(run_dir: Path) -> str | None:
    """Why the run's checkpoints must not move now (its training, or an evaluation of one of
    its checkpoints, is running), or None.
    """
    if _running(run_dir):
        return "its training is running"
    users = []
    for meta_path in sorted((run_dir.parent.parent / "evals").glob("*/eval.json")):
        meta = read_json(meta_path) or {}
        uses = meta.get("training_run") == run_dir.name or str(run_dir / LINK) in str(
            meta.get("checkpoint") or ""
        )
        if uses and _running(meta_path.parent):
            users.append(meta_path.parent.name)
    if users:
        return f"evaluation {', '.join(users)} is using it"
    return None


# ---------------------------------------------------------------------------------- move --
def _listing(root: Path) -> dict[str, tuple[str, int | str]]:
    """Every entry under ``root``: relative path -> (kind, size or link target)."""
    entries: dict[str, tuple[str, int | str]] = {}
    for folder, dirs, files in os.walk(root):
        for name in dirs + files:
            path = Path(folder) / name
            relative = str(path.relative_to(root))
            if path.is_symlink():
                entries[relative] = ("link", os.readlink(path))
            elif path.is_dir():
                entries[relative] = ("dir", 0)
            else:
                entries[relative] = ("file", path.stat().st_size)
    return entries


def move(
    run_dir: Path,
    root: Path,
    *,
    dry_run: bool = False,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Move a finished run's checkpoints to ``root`` and leave a link in their place.

    Copies first, checks that every file arrived with the same size and every link with the
    same target, and only then replaces the folder with the link and deletes the original.
    Returns {"run", "action", "detail", "bytes", "to"}; ``action`` is moved, would move,
    skipped or error.
    """
    result: dict[str, Any] = {"run": run_dir.name, "bytes": 0, "to": None}
    source = run_dir / LINK
    if source.is_symlink():
        return {**result, "action": "skipped", "detail": f"already at {location(run_dir)['path']}"}
    if not source.is_dir():
        return {**result, "action": "skipped", "detail": "no checkpoints"}
    why = busy(run_dir)
    if why:
        return {**result, "action": "skipped", "detail": f"{why}; move it when that has finished"}
    target = external_dir(root, run_dir)
    size = folder_size(source)
    result.update(bytes=size, to=str(target))
    if target.exists() or target.is_symlink():
        return {**result, "action": "error", "detail": f"{target} already exists"}
    existing = target
    while not existing.exists():
        existing = existing.parent
    free = shutil.disk_usage(existing).free
    if size > 0.95 * free:
        return {
            **result,
            "action": "error",
            "detail": f"needs {human(size)}, {human(free)} free on {existing}",
        }
    if dry_run:
        return {**result, "action": "would move", "detail": f"{human(size)} to {target}"}
    partial = target.with_name(f"{target.name}.partial")
    if partial.exists():
        shutil.rmtree(partial)  # left by an interrupted move
    partial.parent.mkdir(parents=True, exist_ok=True)
    log(f"{run_dir.name}: copying {human(size)} to {target} ...")
    try:
        shutil.copytree(source, partial, symlinks=True)
        if _listing(source) != _listing(partial):
            raise RuntimeError("the copy differs from the original (files or sizes)")
    except Exception as error:
        shutil.rmtree(partial, ignore_errors=True)
        return {**result, "action": "error", "detail": f"copy failed, nothing changed: {error}"}
    partial.rename(target)
    old = run_dir / f"{LINK}.moving"
    try:
        source.rename(old)
    except OSError as error:
        shutil.rmtree(target, ignore_errors=True)
        return {**result, "action": "error", "detail": f"nothing changed: {error}"}
    try:
        source.symlink_to(target, target_is_directory=True)
    except OSError as error:
        old.rename(source)
        shutil.rmtree(target, ignore_errors=True)
        return {**result, "action": "error", "detail": f"could not link, nothing changed: {error}"}
    shutil.rmtree(old)
    write_json(
        run_dir / "checkpoints-moved.json",
        {"from": str(source), "to": str(target), "bytes": size, "time": now_iso()},
    )
    return {**result, "action": "moved", "detail": f"{human(size)} to {target}"}
