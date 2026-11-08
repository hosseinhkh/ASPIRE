#!/usr/bin/env python3
"""Log a recorded ASPIRE session to Weights & Biases (run from .venv-tools).

  wandb_log.py SESSION_DIR [--project aspire] [--entity TEAM] [--mode offline|online]
                           [--no-artifacts]

One W&B run per episode, per agent conversation, and one for the session, all grouped under
the session name:

  episode   config = run.json; summary = outcome + timing; per-step joint / gripper /
            end-effector trajectories; every camera video; keyframes; SAM3 overlays; a table of
            every tool, API-stage and model-service call with inputs, outputs and errors;
            the run directory as an artifact.
  agent     summary = turns, cost, minutes, tool calls / errors; the full transcript as a table
            (thinking, text, tool calls, tool results); every image the agent was shown;
            REPORT.md and the programs it wrote; its workspace as an artifact.
  session   GPU memory / utilization timeline, stages, environment snapshot, service logs and
            the HTML report as artifacts.

--mode offline (default) needs no account: runs land in SESSION_DIR/wandb and upload later with
`wandb sync SESSION_DIR/wandb/wandb/offline-run-*` after `wandb login`.
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
from pathlib import Path
import sys

import wandb

MAX_IMAGES = 120  # per kind and run; keyframes can number in the hundreds on long episodes
MAX_CELL = 4000


def read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def clip(text: object) -> str:
    text = text if isinstance(text, str) else json.dumps(text)
    return text if len(text) <= MAX_CELL else text[:MAX_CELL] + " ...[truncated]"


def values(value: object) -> list[float]:
    if isinstance(value, dict) and "$array_inline" in value:
        return [float(v) for v in value["$array_inline"]]
    return []


def init(args: argparse.Namespace, session: Path, name: str, job_type: str, config: dict):
    return wandb.init(
        project=args.project,
        entity=args.entity,
        group=session.name,
        name=name,
        job_type=job_type,
        config=config,
        dir=str(session / "wandb"),
        mode=args.mode,
        reinit="finish_previous",
        settings=wandb.Settings(silent=True),  # one summary line at the end, not per-run tables
    )


def log_episode(args: argparse.Namespace, session: Path, run_dir: Path) -> None:
    meta = read_json(run_dir / "run.json") or {}
    outcome = read_json(run_dir / "outcome.json") or {}
    where = run_dir.relative_to(session)
    job = "agent-episode" if where.parts[0] == "agents" else "episode"
    name = f"{meta.get('suite', '?')}-t{meta.get('task_id', '?')}-s{meta.get('seed', '?')}"
    if job == "agent-episode":
        name = f"{where.parts[1]}/{name}"
    run = init(args, session, name, job, {**meta, "run_dir": str(where)})

    steps = run_dir / "episode/steps.jsonl"
    if steps.exists():
        with steps.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                state = record.get("robot_state") or {}
                metrics: dict[str, float] = {}
                for arm, joints in (state.get("joint_positions") or {}).items():
                    for j, value in enumerate(values(joints)):
                        metrics[f"joint/{arm}/j{j}"] = value
                for arm, width in (state.get("gripper_positions") or {}).items():
                    metrics[f"gripper/{arm}"] = float(width)
                for arm, pose in (state.get("end_effector_poses") or {}).items():
                    for axis, value in zip(
                        "xyz", values((pose or {}).get("position")), strict=False
                    ):
                        metrics[f"ee/{arm}/{axis}"] = value
                for arm, command in ((record.get("action") or {}).get("arms") or {}).items():
                    for j, value in enumerate(values((command or {}).get("target"))):
                        metrics[f"command/{arm}/j{j}"] = value
                if metrics:
                    run.log(metrics, step=index)

    media: dict[str, object] = {}
    for video in sorted((run_dir / "media/videos").glob("*.mp4")):
        media[f"video/{video.stem}"] = wandb.Video(str(video), format="mp4")
    keyframes = sorted((run_dir / "media/keyframes").glob("*/*.png"))[:MAX_IMAGES]
    if keyframes:
        media["keyframes"] = [
            wandb.Image(str(p), caption=f"{p.parent.name}/{p.stem}") for p in keyframes
        ]
    overlays = sorted((run_dir / "media/overlays").rglob("*.png"))[:MAX_IMAGES]
    if overlays:
        media["sam3_overlays"] = [wandb.Image(str(p), caption=p.stem) for p in overlays]

    spans = {}
    events = run_dir / "trace/events.jsonl"
    if events.exists():
        for line in events.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("event") == "span_end":
                spans[event["span_id"]] = event
    table = wandb.Table(
        columns=[
            "index",
            "call",
            "kind",
            "parent",
            "ok",
            "duration_s",
            "sim_time_s",
            "error",
            "input",
            "output",
        ]
    )
    failed = 0
    for call_dir in sorted((run_dir / "trace/calls").glob("call-*")):
        number = call_dir.name.split("-")[1]
        span = spans.get(f"span-{number}", {})
        output = read_json(call_dir / "output.json")
        error = output.get("error") if isinstance(output, dict) else None
        failed += span.get("ok") is False
        table.add_data(
            int(number),
            span.get("name", call_dir.name),
            span.get("category", ""),
            span.get("parent_id") or "",
            span.get("ok"),
            span.get("duration_s"),
            span.get("simulator_time_s"),
            clip(f"{error.get('code')}: {error.get('message')}") if isinstance(error, dict) else "",
            clip(read_json(call_dir / "input.json")),
            clip(output),
        )
    media["tool_calls"] = table
    program = run_dir / "source/program.py"
    if program.exists():
        media["program"] = wandb.Html(f"<pre>{program.read_text(encoding='utf-8')}</pre>")
    run.log(media)

    summary = {
        k: outcome.get(k)
        for k in (
            "program_ok",
            "task_success",
            "protocol_success",
            "steps_executed",
            "wall_duration_s",
            "simulator_duration_s",
            "termination_reason",
        )
    }
    for bucket, data in ((outcome.get("timing") or {}).get("buckets") or {}).items():
        summary[f"timing/{bucket}_s"] = data.get("seconds")
    summary.update(calls=len(table.data), failed_calls=failed)
    run.summary.update(summary)
    if not args.no_artifacts:
        artifact = wandb.Artifact(f"episode-{run.id}", type="episode")
        artifact.add_dir(str(run_dir))
        run.log_artifact(artifact)
    run.finish()


def log_agent(args: argparse.Namespace, session: Path, transcript: Path) -> None:
    workspace = transcript.parent
    name = str(workspace.relative_to(session))
    run = init(args, session, name, "agent", {"transcript": name})
    table = wandb.Table(columns=["i", "kind", "name", "text", "is_error"])
    images = []
    tool_names: dict[str, str] = {}
    stats = {"tool_calls": 0, "tool_errors": 0}
    for i, line in enumerate(transcript.read_text(encoding="utf-8", errors="replace").splitlines()):
        try:
            event = json.loads(line)
        except ValueError:
            continue
        kind, message = event.get("type"), event.get("message") or {}
        if kind == "system" and event.get("subtype") == "init":
            table.add_data(i, "init", str(event.get("model")), clip(event.get("tools")), False)
            run.config.update({"model": event.get("model")}, allow_val_change=True)
        elif kind == "assistant":
            for block in message.get("content") or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    table.add_data(i, "text", "", clip(block.get("text", "")), False)
                elif block.get("type") == "thinking":
                    table.add_data(i, "thinking", "", clip(block.get("thinking", "")), False)
                elif block.get("type") == "tool_use":
                    stats["tool_calls"] += 1
                    tool_names[block.get("id", "")] = block.get("name", "?")
                    table.add_data(
                        i, "tool_use", block.get("name", "?"), clip(block.get("input")), False
                    )
        elif kind == "user":
            for block in message.get("content") or []:
                if not (isinstance(block, dict) and block.get("type") == "tool_result"):
                    continue
                error = bool(block.get("is_error"))
                stats["tool_errors"] += int(error)
                name_ = tool_names.get(block.get("tool_use_id", ""), "tool")
                content = block.get("content")
                texts = []
                for part in content if isinstance(content, list) else [content]:
                    if isinstance(part, dict) and part.get("type") == "image":
                        source = part.get("source") or {}
                        if source.get("type") == "base64" and len(images) < MAX_IMAGES:
                            from PIL import Image

                            data = base64.b64decode(source.get("data", ""))
                            images.append(
                                wandb.Image(
                                    Image.open(io.BytesIO(data)), caption=f"event {i} ({name_})"
                                )
                            )
                        texts.append("[image]")
                    elif isinstance(part, dict):
                        texts.append(part.get("text", ""))
                    elif part is not None:
                        texts.append(str(part))
                table.add_data(i, "tool_result", name_, clip("\n".join(texts)), error)
        elif kind == "result":
            run.summary.update(
                {
                    "result": event.get("subtype"),
                    "turns": event.get("num_turns"),
                    "cost_usd": event.get("total_cost_usd"),
                    "minutes": (event.get("duration_ms") or 0) / 60000,
                }
            )
    payload: dict[str, object] = {"transcript": table}
    if images:
        payload["images_shown_to_agent"] = images
    report = workspace / "REPORT.md"
    if report.exists():
        payload["report"] = wandb.Html(f"<pre>{report.read_text(encoding='utf-8')}</pre>")
    run.log(payload)
    run.summary.update(stats)
    if not args.no_artifacts:
        artifact = wandb.Artifact(f"agent-{run.id}", type="agent")
        for path in workspace.iterdir():
            if path.is_file():
                artifact.add_file(str(path))
        run.log_artifact(artifact)
    run.finish()


def log_session(args: argparse.Namespace, session: Path) -> None:
    meta = read_json(session / "session.json") or {}
    run = init(args, session, "session", "session", meta)
    target = str(meta.get("gpu", "0"))
    gpu_csv, proc_csv = session / "gpu/gpu.csv", session / "gpu/processes.csv"
    yours: dict[str, float] = {}
    if proc_csv.exists():
        with proc_csv.open() as handle:
            for row in csv.DictReader(handle):
                if row["gpu"] == target and row["yours"] == "1":
                    yours[row["t_s"]] = yours.get(row["t_s"], 0.0) + float(row["used_mib"] or 0)
    if gpu_csv.exists():
        with gpu_csv.open() as handle:
            for index, row in enumerate(r for r in csv.DictReader(handle) if r["gpu"] == target):
                run.log(
                    {
                        "gpu/t_s": float(row["t_s"]),
                        "gpu/used_gib": float(row["used_mib"] or 0) / 1024,
                        "gpu/yours_gib": yours.get(row["t_s"], 0.0) / 1024,
                        "gpu/util_pct": float(row["util_pct"] or 0),
                    },
                    step=index,
                )
    stages = session / "stages.tsv"
    if stages.exists():
        table = wandb.Table(columns=["stage", "status", "detail", "start_s", "duration_s"])
        counts: dict[str, int] = {}
        for line in stages.read_text().splitlines():
            if not line.strip():
                continue
            stage, status, detail, start, end = (line.split("\t") + [""] * 5)[:5]
            try:
                begin, duration = float(start), float(end) - float(start)
            except ValueError:
                begin = duration = None
            table.add_data(stage, status, detail, begin, duration)
            if not stage.startswith("preflight: "):
                counts[status] = counts.get(status, 0) + 1
        run.log({"stages": table})
        run.summary.update({f"stages/{k}": v for k, v in counts.items()})
    if not args.no_artifacts:
        for folder, kind in (
            ("env", "environment"),
            ("service-logs", "service-logs"),
            ("report", "report"),
            ("gpu", "gpu-timeline"),
        ):
            if (session / folder).is_dir():
                artifact = wandb.Artifact(f"{kind}-{run.id}", type=kind)
                artifact.add_dir(str(session / folder))
                run.log_artifact(artifact)
    run.finish()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("session")
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "aspire"))
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--mode", choices=("offline", "online"), default="offline")
    parser.add_argument("--no-artifacts", action="store_true")
    args = parser.parse_args()
    session = Path(args.session).resolve()
    (session / "wandb").mkdir(exist_ok=True)

    episodes = sorted(p.parent for p in session.rglob("outcome.json") if "wandb" not in p.parts)
    transcripts = sorted(p for p in session.rglob("transcript.jsonl") if "wandb" not in p.parts)
    transcripts += sorted((session / "claude-sessions").rglob("*.jsonl"))
    for run_dir in episodes:
        log_episode(args, session, run_dir)
    for transcript in transcripts:
        log_agent(args, session, transcript)
    log_session(args, session)
    print(
        f"wandb: logged {len(episodes)} episodes, {len(transcripts)} agent conversations "
        f"and the session ({args.mode}) under group {session.name!r}"
    )
    if args.mode == "offline":
        print(
            f"upload later: wandb login && wandb sync {session / 'wandb' / 'wandb'}/offline-run-*"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
