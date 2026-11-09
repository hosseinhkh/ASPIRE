"""Data model behind the dashboard: runs, sessions, episodes, agents, benchmarks, logs.

Reads the recorded files only (run directories, session folders, transcripts); standard library
plus scripts/gpu/session_report.py and latency.py. scripts/gpu/dashboard.py serves it over HTTP.
"""

from __future__ import annotations

import base64
from collections import defaultdict
import json
import os
from pathlib import Path
import re
import statistics
import sys
from typing import Any
from urllib.parse import unquote

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import latency  # noqa: E402
import session_report as report  # noqa: E402

COLLECTION_KINDS = {  # outputs/<dir>: what one entry inside it is
    "sessions": "session",
    "gpu-demo": "demo",
    "gpu-check": "check",
    "runs": "campaign",
}
MEDIA_SUFFIXES = {".mp4", ".png", ".jpg", ".jpeg", ".npz", ".npy", ".webm", ".gif"}
MAX_POINTS = 600
#: Calls the ProgramExecutor brackets with before/after keyframes, in this order.
MOTION_TOOLS = {
    "step", "execute_trajectory", "set_gripper", "set_grippers", "move_to_joints",
    "move_to_pose", "move_synchronized", "open_gripper", "close_gripper", "go_home",
}  # fmt: skip
PHASE = re.compile(r"^(dev_v\d+|heldout|unseen\w*|dev\w*|validation\w*)$")
#: Below this SAM3 score the segmenter is guessing: the top mask is often another object.
LOW_SAM3_SCORE = 0.3
LOG_ALERT = re.compile(r"Traceback|Error|ERROR|Exception|FAIL|CRITICAL|WARN|Killed|out of memory")


def read_json(path: Path) -> Any:
    return report.read_json(path)


def jsonable(value: Any) -> Any:
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


def short(value: Any, limit: int = 160) -> str:
    if isinstance(value, dict):
        error = value.get("error")
        if isinstance(error, dict):
            return f"{error.get('code')}: {error.get('message')}"[:limit]
        keys = [k for k in value if not k.startswith("$")][:6]
        return ("{" + ", ".join(keys) + "}")[:limit]
    return str(value)[:limit] if value is not None else ""


class Store:
    def __init__(self, outputs: Path) -> None:
        self.outputs = outputs.resolve()

    # ---------------------------------------------------------------------- paths --
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

    # ---------------------------------------------------------------- collections --
    def collections(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
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

    def _summary(self, path: Path, kind: str) -> dict[str, Any]:
        meta = read_json(path / "session.json") or {}
        if not isinstance(meta, dict):
            meta = {}
        outcomes = [
            read_json(p) or {} for p in path.rglob("outcome.json") if "wandb" not in p.parts
        ]
        item: dict[str, Any] = {
            "path": self.rel(path),
            "name": meta.get("name") or path.name,
            "kind": kind,
            "modified": path.stat().st_mtime,
            "status": meta.get("status"),
            "stage_counts": meta.get("stage_counts"),
            "episodes": len(outcomes),
            "successes": sum(1 for o in outcomes if o.get("task_success") is True),
            "llm_cost_usd": meta.get("llm_cost_usd"),
            "gpu": meta.get("gpu"),
            "git_commit": meta.get("git_commit"),
        }
        if kind == "live":
            run = read_json(path / "run.json") or {}
            item.update(
                name=f"{run.get('suite', '?')}:{run.get('task_id', '?')} seed {run.get('seed', '?')}",
                status="finalized" if outcomes else "open or interrupted",
            )
        return item

    # ----------------------------------------------------------------- collection --
    def collection(self, path: Path) -> dict[str, Any]:
        meta = read_json(path / "session.json") or {}
        gpu = report.load_gpu(path)
        stages = report.load_stages(path, gpu)
        for number, row in enumerate(stages, 1):
            row["n"] = number
        gpu_view = None
        times = gpu["times"]
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
        episodes = [self._episode_row(run, path) for run in self.runs_in(path)]
        transcripts = [p for p in path.rglob("transcript.jsonl") if "wandb" not in p.parts]
        transcripts += sorted((path / "claude-sessions").rglob("*.jsonl"))
        agents = [
            {
                "path": self.rel(t),
                "title": t.relative_to(path).as_posix(),
                **self._agent_stats(t),
            }
            for t in transcripts
        ]
        files = [f for f in (path / "session.log", path / "stages.tsv", path / "session.json")
                 if f.exists()]  # fmt: skip
        for folder in ("env", "preflight", "service-logs", "episodes/logs", "gpu"):
            files += sorted(p for p in (path / folder).glob("*") if p.is_file())
        return {
            "path": self.rel(path),
            "meta": meta,
            "stages": jsonable(stages),
            "gpu": gpu_view,
            "episodes": episodes,
            "tasks": self._tasks(episodes),
            "failures": _failures(episodes),
            "agents": agents,
            "files": [
                {
                    "path": self.rel(f),
                    "name": f.relative_to(path).as_posix(),
                    "size": f.stat().st_size,
                    "log": f.suffix in {".log", ".err", ".txt", ".tsv"},
                }
                for f in files
            ],
            "has_report": (path / "report/index.html").exists(),
        }

    def _episode_row(self, run: Path, root: Path) -> dict[str, Any]:
        meta = read_json(run / "run.json") or {}
        outcome = read_json(run / "outcome.json") or {}
        where = run.relative_to(root).parts if run != root else ()
        phase = run_phase(where)
        agent = next((part for part in where if part.startswith("llm-")), None)
        result = read_json(run / "source/program-result.json") or {}
        error = result.get("error") or outcome.get("error")
        if isinstance(error, dict):
            error = f"{error.get('code') or error.get('type') or 'error'}: {error.get('message')}"
        keyframes = sorted((run / "media/keyframes").glob("*/*.png"))
        thumb = next(
            (p for p in reversed(keyframes) if p.stem in {"agentview", "head", "top"}),
            keyframes[-1] if keyframes else None,
        )
        return {
            "path": self.rel(run),
            "where": "/".join(where),
            "benchmark": meta.get("benchmark"),
            "suite": meta.get("suite"),
            "task_id": meta.get("task_id"),
            "task_name": meta.get("task_name"),
            "seed": meta.get("seed"),
            "init_mode": meta.get("init_mode"),
            "program_ok": outcome.get("program_ok"),
            "task_success": outcome.get("task_success"),
            "termination": outcome.get("termination_reason"),
            "steps": outcome.get("steps_executed"),
            "wall_s": outcome.get("wall_duration_s"),
            "sim_s": outcome.get("simulator_duration_s"),
            "calls": len(list((run / "trace/calls").glob("call-*"))),
            "phase": phase,
            "agent": agent,
            "finalized": bool(outcome),
            "thumb": self.rel(thumb) if thumb else None,
            "error": str(error)[:400] if error else None,
            "program": self.rel(run / "source/program.py")
            if (run / "source/program.py").exists()
            else None,
        }

    @staticmethod
    def _tasks(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
        for e in episodes:
            key = (e["agent"] or "", e["phase"], e["benchmark"], e["suite"], e["task_id"])
            groups[key].append(e)
        rows = []
        for (agent, phase, benchmark, suite, task_id), items in sorted(
            groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])
        ):
            done = [e for e in items if e["finalized"]]
            rows.append(
                {
                    "key": "|".join(str(x) for x in (agent, phase, benchmark, suite, task_id)),
                    # The folder holding every run of the group, for one-click downloads.
                    "folder": os.path.commonpath([e["path"] for e in items]),
                    "runs": sorted(
                        ({k: e.get(k) for k in RUN_FIELDS} for e in items),
                        key=lambda r: (r["seed"] is None, r["seed"] or 0),
                    ),
                    "agent": agent or None,
                    "phase": phase,
                    "benchmark": benchmark,
                    "suite": suite,
                    "task_id": task_id,
                    "task_name": items[0]["task_name"],
                    "episodes": len(items),
                    "successes": sum(1 for e in done if e["task_success"] is True),
                    "program_ok": sum(1 for e in done if e["program_ok"] is True),
                    "seeds": sorted({e["seed"] for e in items if e["seed"] is not None}),
                    "mean_steps": _mean([e["steps"] for e in done]),
                    "mean_wall_s": _mean([e["wall_s"] for e in done]),
                }
            )
        return rows

    # -------------------------------------------------------------------- episode --
    def episode(self, run: Path) -> dict[str, Any]:
        meta = read_json(run / "run.json") or {}
        outcome = read_json(run / "outcome.json") or {}
        result = read_json(run / "source/program-result.json") or {}
        rows = {row["index"]: row for row in report.call_rows(run)}
        spans = self._timeline(run, rows)
        media = {
            "videos": [self.rel(p) for p in sorted((run / "media/videos").glob("*.mp4"))],
            "keyframes": [self.rel(p) for p in sorted((run / "media/keyframes").glob("*/*.png"))],
            "overlays": [
                self.rel(p)
                for p in sorted((run / "media/overlays").rglob("*.png"))
                if "-mask-" not in p.name
            ],
            "geometry": len(list((run / "media/geometry").glob("*.json"))),
        }
        video_index = read_json(run / "media/video-index.json") or {}
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
        charts = {
            title: {name: downsample(values) for name, values in series.items()}
            for title, series in report.trajectories(run).items()
        }
        return {
            "path": self.rel(run),
            "meta": meta,
            "outcome": outcome,
            "language": self._language(run),
            "result": jsonable(result.get("result")),
            "program_error": jsonable(result.get("error")),
            "timing": outcome.get("timing"),
            "duration_s": max(
                [s["end"] for s in spans] + [outcome.get("simulator_duration_s") or 0]
            ),
            "video_fps": video_index.get("fps"),
            "spans": spans,
            "warnings": _warnings(self._mark_empty_grasps(run, spans), outcome),
            "calls": [s for s in spans if s.get("dir")],
            "charts": charts,
            "chart_steps": len(next(iter(next(iter(charts.values()), {}).values()), [])),
            "media": media,
            "texts": texts,
            "commands": jsonable(commands),
            "files": files,
        }

    def _timeline(self, run: Path, rows: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
        """Every span with its simulator start/end, wall time, parent, call folder and media."""
        events = run / "trace/events.jsonl"
        starts: dict[str, dict[str, Any]] = {}
        spans: list[dict[str, Any]] = []
        if events.exists():
            for line in events.read_text(encoding="utf-8").splitlines():
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get("event") == "span_start":
                    starts[event["span_id"]] = event
                elif event.get("event") == "span_end":
                    start = starts.get(event["span_id"], {})
                    number = int(event["span_id"].split("-")[1])
                    row = rows.get(number, {})
                    spans.append(
                        {
                            "id": event["span_id"],
                            "index": number,
                            "name": event.get("name"),
                            "category": event.get("category"),
                            "parent": event.get("parent_id"),
                            "ok": event.get("ok"),
                            "start": float(start.get("simulator_time_s") or 0.0),
                            "end": float(event.get("simulator_time_s") or 0.0),
                            "wall_s": event.get("duration_s"),
                            "started_at": start.get("timestamp"),
                            "error": row.get("error") or "",
                            "dir": self.rel(row["dir"]) if row.get("dir") else None,
                            "depth": row.get("depth", 0),
                        }
                    )
        spans.sort(key=lambda s: s["index"])
        self._pair_keyframes(run, spans)
        self._pair_overlays(run, spans)
        self._pair_evidence(run, spans)
        return spans

    def _pair_keyframes(self, run: Path, spans: list[dict[str, Any]]) -> None:
        """Attach the before/after keyframes the runtime takes around each motion call.

        Folders are numbered in capture order. A call that raised has no "after" image,
        and a motion tool called inside another one nests its pair, so each "after"
        closes the latest open "before" of the same tool, and pairs are handed to the
        motion calls of that tool in the order they started.
        """
        pairs: list[dict[str, Any]] = []
        open_pairs: list[int] = []
        for folder in sorted((run / "media/keyframes").glob("*")):
            match = re.match(r"\d+-(before|after)-(.+)$", folder.name)
            if not match:
                continue
            kind, tool = match.group(1), _slug(match.group(2))
            images = [self.rel(p) for p in sorted(folder.glob("*.png"))]
            if kind == "before":
                open_pairs.append(len(pairs))
                pairs.append({"tool": tool, "before": images})
                continue
            index = next((i for i in reversed(open_pairs) if pairs[i]["tool"] == tool), None)
            if index is None:
                pairs.append({"tool": tool, "after": images})
            else:
                open_pairs.remove(index)
                pairs[index]["after"] = images
        queues: dict[str, list[dict[str, Any]]] = {}
        for pair in pairs:
            queues.setdefault(pair.pop("tool"), []).append(pair)
        for span in spans:
            if span["category"] == "program" and span["name"] in MOTION_TOOLS:
                queue = queues.get(_slug(str(span["name"])))
                if queue:
                    span["keyframes"] = queue.pop(0)

    def _pair_overlays(self, run: Path, spans: list[dict[str, Any]]) -> None:
        """Attach each SAM3 overlay image to the request that produced it.

        Newer recordings name the call folder in the overlay metadata. Older ones are
        paired in order with the requests that returned masks: failed or empty requests
        save no image, so a plain zip with every SAM3 call would drift.
        """
        sam_calls = [s for s in spans if s["category"] == "provider" and "sam3" in str(s["name"])]
        by_folder = {Path(s["dir"]).name: s for s in sam_calls if s.get("dir")}
        unnamed = []
        for meta in sorted((run / "media/overlays").glob("sam3-*.json")):
            data = read_json(meta) or {}
            image = meta.with_suffix(".png")
            base = meta.parent / str(data.get("image") or "")
            candidates = []
            for candidate in data.get("candidates", []):
                mask = meta.parent / str(candidate.get("mask") or "")
                candidates.append({**candidate, "mask": self.rel(mask) if mask.is_file() else None})
            overlay = {
                "image": self.rel(image) if image.exists() else None,
                "base": self.rel(base) if data.get("image") and base.is_file() else None,
                "camera": data.get("camera_name"),
                "image_shape": data.get("image_shape"),
                "candidates": candidates,
            }
            span = by_folder.get(str(data.get("call")))
            if span is not None:
                span["overlay"] = overlay
            else:
                unnamed.append(overlay)
        if not unnamed:
            return
        with_masks = [
            s for s in sam_calls if "overlay" not in s and s.get("ok") and _has_masks(run, s)
        ]
        for span, overlay in zip(with_masks, unnamed, strict=False):
            span["overlay"] = overlay

    @staticmethod
    def _mark_empty_grasps(run: Path, spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Flag close_gripper calls that closed on nothing (diagnostics.closed_empty)."""
        for span in spans:
            if span.get("name") == "close_gripper" and span.get("dir"):
                output = read_json(run / "trace/calls" / Path(span["dir"]).name / "output.json")
                diagnostics = output.get("diagnostics") if isinstance(output, dict) else None
                if isinstance(diagnostics, dict) and diagnostics.get("closed_empty"):
                    span["closed_empty"] = True
        return spans

    def _pair_evidence(self, run: Path, spans: list[dict[str, Any]]) -> None:
        """Attach grasp overlays and 3D geometry files to the calls that produced them."""
        by_folder = {Path(s["dir"]).name: s for s in spans if s.get("dir")}
        for meta in sorted((run / "media/overlays").glob("grasps-*.json")):
            data = read_json(meta) or {}
            span = by_folder.get(str(data.get("call")))
            image = meta.with_suffix(".png")
            if span is not None and image.exists():
                span.setdefault("grasp_overlays", []).append(
                    {
                        "image": self.rel(image),
                        "camera": data.get("camera_name"),
                        "candidates": data.get("candidates", []),
                    }
                )
        for path in sorted((run / "media/geometry").glob("*.json")):
            data = read_json(path) or {}
            span = by_folder.get(str(data.get("call")))
            if span is not None:
                span.setdefault("geometry", []).append(
                    {
                        "kind": data.get("kind"),
                        "frame": data.get("frame"),
                        "points": data.get("point_count"),
                        "grasps": len(data.get("grasps") or []),
                        "center": data.get("center"),
                        "extents": data.get("extents"),
                    }
                )

    def scene(self, run: Path) -> dict[str, Any]:
        """Everything with a position, in the robot base frame, for the 3D view."""
        items = []
        for path in sorted((run / "media/geometry").glob("*.json")):
            data = read_json(path)
            if isinstance(data, dict):
                items.append({**data, "file": path.name})
        targets = []
        for call in sorted((run / "trace/calls").glob("call-*")):
            if not any(name in call.name for name in ("move-to-pose", "plan-motion")):
                continue
            pose = _find_pose(read_json(call / "input.json"))
            if pose is not None:
                targets.append({"call": call.name, **pose})
        path_xyz: list[list[float]] = []
        steps = run / "episode/steps.jsonl"
        if steps.exists():
            with steps.open(encoding="utf-8") as stream:
                for number, line in enumerate(stream):
                    try:
                        state = json.loads(line).get("robot_state") or {}
                    except ValueError:
                        continue
                    for pose in (state.get("end_effector_poses") or {}).values():
                        position = _inline(pose.get("position") if pose else None)
                        if position and number % 2 == 0:
                            path_xyz.append([round(float(v), 4) for v in position[:3]])
                        break
        return {"items": items, "targets": targets, "ee_path": path_xyz[:5000]}

    def steps(self, run: Path, start_s: float, end_s: float, limit: int = 4000) -> dict[str, Any]:
        """Commanded and measured joints, per arm, for the steps in a time window."""
        out: dict[str, Any] = {"t": [], "arms": {}}
        path = run / "episode/steps.jsonl"
        if not path.exists():
            return out
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    step = json.loads(line)
                except ValueError:
                    continue
                t = float(step.get("simulator_time_s") or 0.0)
                if t < start_s - 1e-6:
                    continue
                if t > end_s + 1e-6 or len(out["t"]) >= limit:
                    break
                out["t"].append(round(t, 3))
                state = step.get("robot_state") or {}
                commands = ((step.get("action") or {}).get("arms")) or {}
                for arm, joints in (state.get("joint_positions") or {}).items():
                    entry = out["arms"].setdefault(arm, {"measured": [], "commanded": []})
                    entry["measured"].append(_inline(joints) or [])
                    entry["commanded"].append(
                        _inline((commands.get(arm) or {}).get("target")) or []
                    )
        return out

    def array(self, path: Path, limit: int = 20000) -> dict[str, Any]:
        """A saved NumPy array (trace payload) as nested lists; long arrays keep evenly spaced rows."""
        data = read_npy(path)
        rows = data["values"]
        if len(rows) > limit:
            step = len(rows) / limit
            rows = [rows[int(i * step)] for i in range(limit)]
        return {"shape": data["shape"], "dtype": data["dtype"], "values": rows}

    @staticmethod
    def _language(run: Path) -> str | None:
        """The task instruction, from the first task-context or task-metadata call's output."""
        for call in sorted((run / "trace/calls").glob("call-*")):
            if any(key in call.name for key in ("get-task-context", "get-task-metadata")):
                output = read_json(call / "output.json")
                if isinstance(output, dict) and output.get("language"):
                    return str(output["language"])
        return None

    def call(self, call_dir: Path) -> dict[str, Any]:
        data: dict[str, Any] = {"dir": self.rel(call_dir)}
        for name in ("input", "output", "error"):
            path = call_dir / f"{name}.json"
            if path.exists():
                data[name] = read_json(path)
        data["other_files"] = [
            self.rel(p)
            for p in sorted(call_dir.iterdir())
            if p.is_file() and p.name not in {"input.json", "output.json", "error.json"}
        ]
        return data

    # ---------------------------------------------------------------------- agent --
    def _agent_stats(self, transcript: Path) -> dict[str, Any]:
        stats: dict[str, Any] = {"tool_calls": 0, "tool_errors": 0}
        for event in report.transcript_events(transcript):
            if event.get("type") == "result":
                stats.update(
                    result=event.get("subtype"),
                    turns=event.get("num_turns"),
                    cost_usd=event.get("total_cost_usd"),
                    minutes=round((event.get("duration_ms") or 0) / 60000, 1),
                    # What Claude Code really ran with (window, thinking), per model.
                    models={
                        name: {
                            "context_window": usage.get("contextWindow"),
                            "thinking_tokens": usage.get("thinkingTokens"),
                        }
                        for name, usage in (event.get("modelUsage") or {}).items()
                        if isinstance(usage, dict)
                    },
                )
            for block in (event.get("message") or {}).get("content") or []:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    stats["tool_calls"] += 1
                if (
                    isinstance(block, dict)
                    and block.get("type") == "tool_result"
                    and block.get("is_error")
                ):
                    stats["tool_errors"] += 1
        return stats

    def agent(self, transcript: Path) -> dict[str, Any]:
        workspace = transcript.parent
        events: list[dict[str, Any]] = []
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
                    parts: list[dict[str, Any]] = []
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
        texts = {
            name: (workspace / name).read_text(encoding="utf-8", errors="replace")
            for name in ("prompt.md", "REPORT.md", "DEMO_SUMMARY.md", "kickoff_prompt.md")
            if (workspace / name).exists()
        }
        programs = {
            p.name: p.read_text(encoding="utf-8", errors="replace")
            for p in sorted(workspace.glob("program_*.py"))
        }
        versions = []
        for batch in sorted(workspace.glob("*/batch.json")):
            data = read_json(batch) or {}
            versions.append({"phase": batch.parent.name, "passes": data.get("passes"),
                             "seeds": data.get("seeds")})  # fmt: skip
        return {"path": self.rel(transcript), "events": events, "texts": texts,
                "programs": programs, "versions": versions,
                "story": self._story(events, workspace),
                **self._agent_stats(transcript)}  # fmt: skip

    def _story(self, events: list[dict[str, Any]], workspace: Path) -> list[dict[str, Any]]:
        """The debugging loop as chapters: what the agent thought, what it changed, what ran.

        A chapter ends at each program written or edited, and at each batch it ran; the
        thinking and text before it are its reasoning.
        """
        chapters: list[dict[str, Any]] = []
        reasoning: list[dict[str, str]] = []
        previous: dict[str, str] = {}  # program name -> last full text seen
        for event in events:
            kind = event.get("kind")
            if kind in ("thinking", "text") and event.get("role", "assistant") == "assistant":
                text = str(event.get("text") or "").strip()
                if text:
                    reasoning.append({"kind": kind, "text": text[:6000]})
                continue
            if kind == "tool_result" and chapters and chapters[-1].get("awaiting"):
                chapter = chapters[-1]
                chapter.pop("awaiting")
                chapter["ok"] = not event.get("is_error")
                text = " ".join(p.get("text", "") for p in event.get("parts") or [])
                chapter["output_tail"] = text[-1500:]
                continue
            if kind != "tool_use":
                continue
            name, data = event.get("name"), event.get("input") or {}
            program = str(data.get("file_path") or "")
            if name in ("Write", "Edit", "MultiEdit") and PROGRAM_FILE.search(program):
                label = Path(program).name
                if name == "Write":
                    new = str(data.get("content") or "")
                    old = previous.get(_version_family(label), "")
                    previous[_version_family(label)] = new
                    diff = _diff(old, new, label) if old else None
                    code = None if old else new[:20000]
                else:
                    edits = data.get("edits") or [data]
                    diff = "\n".join(
                        _diff(str(e.get("old_string", "")), str(e.get("new_string", "")), label)
                        for e in edits
                    )
                    code = None
                chapters.append({"kind": "code", "file": label, "tool": name,
                                 "reasoning": reasoning, "diff": diff, "code": code,
                                 "awaiting": True})  # fmt: skip
                reasoning = []
            elif name == "Bash" and "run-batch" in str(data.get("command") or ""):
                command = str(data.get("command"))
                root = _flag(command, "--output-root")
                chapters.append({"kind": "run", "command": command[:2000],
                                 "seeds": _flag(command, "--seeds", many=True),
                                 "output": root, "reasoning": reasoning,
                                 "results": self._batch_results(workspace, root),
                                 "awaiting": True})  # fmt: skip
                reasoning = []
        if reasoning:
            chapters.append({"kind": "end", "reasoning": reasoning})
        for chapter in chapters:
            chapter.pop("awaiting", None)
        return chapters

    def _batch_results(self, workspace: Path, root: str | None) -> dict[str, Any] | None:
        """The runs a batch wrote, found under the agent's workspace by the path's tail."""
        if not root:
            return None
        tail = [part for part in root.rstrip("/").split("/") if part and "$" not in part]
        tail = tail[-3:]
        folder = None
        for size in range(len(tail), 0, -1):
            candidate = workspace.joinpath(*tail[-size:])
            if candidate.is_dir():
                folder = candidate
                break
        if folder is None:
            return None
        runs = [self._episode_row(run, folder) for run in self.runs_in(folder)]
        return {
            "path": self.rel(folder),
            "passes": sum(1 for r in runs if r["task_success"] is True),
            "total": len(runs),
            "runs": [
                {
                    "path": r["path"],
                    "seed": r["seed"],
                    "task_success": r["task_success"],
                    "termination": r["termination"],
                }
                for r in runs
            ],
        }

    def agent_image(self, transcript: Path, ref: str) -> tuple[bytes, str]:
        i, j, k = (int(x) for x in ref.split("."))
        event = report.transcript_events(transcript)[i]
        block = (event.get("message") or {}).get("content")[j]
        content = block.get("content")
        part = (content if isinstance(content, list) else [content])[k]
        source = part.get("source") or {}
        return base64.b64decode(source.get("data", "")), source.get("media_type", "image/png")

    # ------------------------------------------------------------------ benchmark --
    def benchmark(self, paths: list[Path]) -> dict[str, Any]:
        """Side-by-side results of several runs: per task, per agent, latency and cost."""
        columns = []
        for path in paths:
            episodes = [self._episode_row(run, path) for run in self.runs_in(path)]
            meta = read_json(path / "session.json") or {}
            agent_costs = []
            agent_models: dict[str, dict[str, Any]] = {}
            for transcript in path.rglob("transcript.jsonl"):
                stats = self._agent_stats(transcript)
                if stats.get("cost_usd") is not None:
                    agent_costs.append(float(stats["cost_usd"]))
                for model, usage in (stats.get("models") or {}).items():
                    entry = agent_models.setdefault(
                        model, {"context_window": None, "thinking_tokens": 0}
                    )
                    if usage.get("context_window"):
                        entry["context_window"] = max(
                            entry["context_window"] or 0, int(usage["context_window"])
                        )
                    entry["thinking_tokens"] += int(usage.get("thinking_tokens") or 0)
            lat = latency.collect(path)
            providers = {
                name: {
                    "count": len(values),
                    "median_s": statistics.median(values),
                    "p90_s": latency.quantile(values, 0.9),
                }
                for name, values in lat["calls"].items()
                if name.startswith("provider")
            }
            simulator = {name: statistics.median(v) for name, v in lat["buckets"].items()}
            done = [e for e in episodes if e["finalized"]]
            columns.append(
                {
                    "path": self.rel(path),
                    "name": meta.get("name") or _column_name(self.rel(path)),
                    "git_commit": meta.get("git_commit"),
                    "gpu": meta.get("gpu"),
                    "started": meta.get("started"),
                    "episodes": len(episodes),
                    "successes": sum(1 for e in done if e["task_success"] is True),
                    "tasks": self._tasks(episodes),
                    "llm_cost_usd": round(sum(agent_costs), 2) if agent_costs else None,
                    "agent_models": agent_models,
                    "claude_effort": meta.get("claude_effort"),
                    # fixed / published (docs/harness-fixes.md); older sessions only
                    # recorded the motion-timing switch.
                    "harness": meta.get("harness")
                    or {"1": "re-timed motion", "0": "published motion"}.get(
                        str(meta.get("motion_retiming"))
                    ),
                    "providers": providers,
                    "simulator_ms": {k: round(v * 1000, 1) for k, v in simulator.items()},
                    "realtime_factor": statistics.median(lat["realtime"])
                    if lat["realtime"]
                    else None,
                }
            )
        keys = sorted(
            {
                (
                    t["agent"] or "",
                    t["phase"],
                    t["benchmark"],
                    t["suite"],
                    t["task_id"],
                    t["task_name"],
                )
                for c in columns
                for t in c["tasks"]
            },
            key=lambda k: tuple(str(x) for x in k),
        )
        return {
            "columns": columns,
            "rows": [
                {
                    "agent": k[0] or None,
                    "phase": k[1],
                    "benchmark": k[2],
                    "suite": k[3],
                    "task_id": k[4],
                    "task_name": k[5],
                }
                for k in keys
            ],
        }

    # ----------------------------------------------------------------------- logs --
    def log(
        self, path: Path, query: str = "", alerts: bool = False, tail: int = 4000
    ) -> dict[str, Any]:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        needle = query.lower()
        selected = [
            (number, line)
            for number, line in enumerate(lines, 1)
            if (not needle or needle in line.lower()) and (not alerts or LOG_ALERT.search(line))
        ]
        return {
            "path": self.rel(path),
            "total_lines": len(lines),
            "matched": len(selected),
            "lines": [
                {"n": n, "text": t[:2000], "alert": bool(LOG_ALERT.search(t))}
                for n, t in selected[-tail:]
            ],
            "size": path.stat().st_size,
        }


def _warnings(spans: list[dict[str, Any]], outcome: dict[str, Any]) -> list[dict[str, Any]]:
    """Things worth a look before trusting an episode, each linked to its call."""
    notes: list[dict[str, Any]] = []
    for span in spans:
        candidates = (span.get("overlay") or {}).get("candidates") or []
        scores = sorted((float(c.get("score") or 0.0) for c in candidates), reverse=True)
        if not scores or scores[0] >= LOW_SAM3_SCORE:
            continue
        label = candidates[0].get("label") or span.get("name")
        text = f'SAM3 is unsure what "{label}" is: best score {scores[0]:.2f}'
        if len(scores) > 1 and scores[0] - scores[1] < 0.05:
            text += f", next {scores[1]:.2f}"
        text += ". The program may have used the wrong object: compare the candidates."
        notes.append({"level": "warn", "text": text, "span": span["id"]})
    for span in spans:
        if span.get("closed_empty"):
            notes.append(
                {
                    "level": "warn",
                    "text": "close_gripper closed on nothing: the grasp missed, so the moves "
                    "after it carried nothing.",
                    "span": span["id"],
                }
            )
    if outcome.get("program_ok") is True and outcome.get("task_success") is False:
        notes.append(
            {
                "level": "info",
                "text": "The program finished without an error, but the benchmark's success "
                "check never passed. Check the final frame and which object each call used.",
                "span": None,
            }
        )
    return notes


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


#: Generated programs: the quick agent's program_vNN.py and the campaign's code/vNN_slug.py.
PROGRAM_FILE = re.compile(r"(program_[\w-]+|(^|/)v\d+[\w-]*|fix_code[\w-]*)\.py$")


def _version_family(name: str) -> str:
    """Files that are versions of one program share a family (diffs are taken within it)."""
    return "program" if re.match(r"(program_|v\d+)", name) else name


def _diff(old: str, new: str, label: str) -> str:
    import difflib

    lines = difflib.unified_diff(
        old.splitlines(), new.splitlines(), "before", label, lineterm="", n=2
    )
    return "\n".join(list(lines)[:600])


def _flag(command: str, flag: str, many: bool = False) -> Any:
    """A flag's value(s) in a shell command line (quotes stripped)."""
    import shlex

    try:
        words = shlex.split(command.replace("\\\n", " "))
    except ValueError:
        words = command.split()
    if flag not in words:
        return None
    rest = words[words.index(flag) + 1 :]
    if not many:
        return rest[0] if rest else None
    values = []
    for word in rest:
        if word.startswith("--"):
            break
        values.append(word)
    return values


def _inline(value: Any) -> list[Any] | None:
    """A recorded array value: plain list or {"$array_inline": [...]}."""
    if isinstance(value, dict):
        value = value.get("$array_inline")
    return value if isinstance(value, list) else None


def _find_pose(value: Any, depth: int = 0) -> dict[str, Any] | None:
    """The first recorded Pose inside a call's input (the target of a motion call)."""
    if depth > 6:
        return None
    if isinstance(value, dict):
        if value.get("$type") == "Pose":
            position = _inline(value.get("position"))
            if position:
                return {
                    "position": [float(v) for v in position[:3]],
                    "quaternion_wxyz": _inline(value.get("quaternion_wxyz")),
                    "frame": value.get("frame"),
                }
        children = list(value.values())
    elif isinstance(value, list):
        children = value
    else:
        return None
    for child in children:
        found = _find_pose(child, depth + 1)
        if found is not None:
            return found
    return None


_NPY_TYPES = {"<f8": "d", "<f4": "f", "<i8": "q", "<i4": "i", "<u1": "B", "|u1": "B", "|b1": "B"}


def read_npy(path: Path) -> dict[str, Any]:
    """Read a C-order .npy file with the standard library (the dashboard needs no NumPy)."""
    import array
    import ast

    raw = path.read_bytes()
    if raw[:6] != b"\x93NUMPY":
        raise ValueError(f"not a .npy file: {path.name}")
    major = raw[6]
    header_len = int.from_bytes(raw[8:10] if major == 1 else raw[8:12], "little")
    start = (10 if major == 1 else 12) + header_len
    header = ast.literal_eval(raw[(10 if major == 1 else 12) : start].decode("latin1"))
    code = _NPY_TYPES.get(header["descr"])
    if code is None or header.get("fortran_order"):
        raise ValueError(f"unsupported array layout: {header}")
    values = array.array(code)
    values.frombytes(raw[start:])
    flat = [bool(v) for v in values] if header["descr"] == "|b1" else list(values)
    shape = list(header["shape"])
    if len(shape) >= 2:
        width = 1
        for dim in shape[1:]:
            width *= dim
        flat = [flat[i : i + width] for i in range(0, len(flat), width)]
    return {"shape": shape, "dtype": header["descr"], "values": flat}


def _has_masks(run: Path, span: dict[str, Any]) -> bool:
    """Whether a recorded SAM3 request returned at least one segmentation."""
    if not span.get("dir"):
        return True
    output = read_json(run / "trace/calls" / Path(span["dir"]).name / "output.json")
    if not isinstance(output, dict):
        return True
    return output.get("ok", True) is not False and bool(output.get("segmentations", True))


RUN_FIELDS = ("path", "seed", "task_success", "program_ok", "termination", "steps", "wall_s",
              "error", "program", "finalized")  # fmt: skip


def run_phase(parts: tuple[str, ...]) -> str:
    """Which stage a run belongs to, from its folders.

    Quick agents: dev_vNN/, heldout/. Campaigns (iterative-debugging): task_N/debug/initial
    (v00), debug/attempts/vNN, debug/holdout (the unseen check), validation/ (held-out),
    rerun_NN/ (backward transfer) and verification/ (the skill verifier).
    """
    phase = None
    for index, part in enumerate(parts):
        previous = parts[index - 1] if index else ""
        following = parts[index + 1] if index + 1 < len(parts) else ""
        if part == "verification":
            phase = "verifier"
            break
        if part == "validation":
            phase = "heldout"
        elif part == "holdout" and previous == "debug":
            phase = "unseen"
        elif part == "initial" and previous == "debug":
            phase = "dev_v00"
        elif part == "attempts" and re.fullmatch(r"v\d+", following):
            phase = f"dev_{following}"
        elif PHASE.match(part):
            phase = part
    rerun = next((part for part in parts if re.fullmatch(r"rerun_\d+", part)), None)
    phase = phase or "episode"
    return f"{rerun} {phase}" if rerun and phase != "verifier" else phase


def _column_name(rel: str) -> str:
    """A short name for a compared folder: its agent and phase when it has them."""
    parts = Path(rel).parts
    agent = next((p for p in parts if p.startswith("llm-")), None)
    phase = next((p for p in reversed(parts) if PHASE.match(p)), None)
    if agent or phase:
        return " · ".join(x for x in (agent, phase) if x)
    return "/".join(parts[1:3]) if len(parts) > 2 else (parts[-1] if parts else rel)


def _failure_reason(episode: dict[str, Any]) -> str:
    """One line that groups episodes failing the same way (numbers and paths blanked)."""
    if not episode["finalized"]:
        return "not finished (stopped or crashed before the outcome was written)"
    if episode["program_ok"] is False:
        text = str(episode.get("error") or episode.get("termination") or "program error")
        text = re.sub(r"/[^\s'\"]+", "<path>", text.splitlines()[0] if text else text)
        text = re.sub(r"-?\d+(\.\d+)?(e-?\d+)?", "N", text)
        return f"program failed: {text[:160]}"
    return f"program finished, task not done ({episode.get('termination') or 'unknown'})"


def _failures(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Unsolved episodes grouped by how they failed, most common first."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        if episode["task_success"] is not True:
            groups[_failure_reason(episode)].append(episode)
    return [
        {
            "reason": reason,
            "count": len(items),
            "example": next((e["error"] for e in items if e.get("error")), None),
            "episodes": [
                {k: e.get(k) for k in ("path", "seed", "suite", "task_id", "agent", "phase")}
                for e in items
            ],
        }
        for reason, items in sorted(groups.items(), key=lambda kv: -len(kv[1]))
    ]


def _mean(values: list[Any]) -> float | None:
    numbers = [float(v) for v in values if isinstance(v, (int, float))]
    return round(sum(numbers) / len(numbers), 2) if numbers else None
