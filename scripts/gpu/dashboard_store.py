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
        phase = next((part for part in where if PHASE.match(part)), "episode")
        agent = next((part for part in where if part.startswith("llm-")), None)
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
            "overlays": [self.rel(p) for p in sorted((run / "media/overlays").rglob("*.png"))],
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
            overlay = {
                "image": self.rel(image) if image.exists() else None,
                "camera": data.get("camera_name"),
                "candidates": data.get("candidates", []),
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
                **self._agent_stats(transcript)}  # fmt: skip

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
            for transcript in path.rglob("transcript.jsonl"):
                cost = self._agent_stats(transcript).get("cost_usd")
                if cost is not None:
                    agent_costs.append(float(cost))
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
                    "name": meta.get("name") or path.name,
                    "git_commit": meta.get("git_commit"),
                    "gpu": meta.get("gpu"),
                    "started": meta.get("started"),
                    "episodes": len(episodes),
                    "successes": sum(1 for e in done if e["task_success"] is True),
                    "tasks": self._tasks(episodes),
                    "llm_cost_usd": round(sum(agent_costs), 2) if agent_costs else None,
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


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _has_masks(run: Path, span: dict[str, Any]) -> bool:
    """Whether a recorded SAM3 request returned at least one segmentation."""
    if not span.get("dir"):
        return True
    output = read_json(run / "trace/calls" / Path(span["dir"]).name / "output.json")
    if not isinstance(output, dict):
        return True
    return output.get("ok", True) is not False and bool(output.get("segmentations", True))


def _mean(values: list[Any]) -> float | None:
    numbers = [float(v) for v in values if isinstance(v, (int, float))]
    return round(sum(numbers) / len(numbers), 2) if numbers else None
