#!/usr/bin/env python3
"""Build a browsable HTML report for a recorded ASPIRE session (standard library only).

  session_report.py SESSION_DIR

Writes SESSION_DIR/report/index.html plus one page per episode and per agent conversation:

  * episode pages: outcome, timing, every camera video, joint / gripper / end-effector
    trajectories, keyframes before and after each motion call, SAM3 overlays, and the full
    tool-call tree (program call -> API stage -> model service) with inputs and outputs;
  * agent pages: the whole Claude Code conversation -- prompt, the model's text and thinking,
    every tool call and its result, images it was shown -- with turns, duration and cost;
  * the index: session metadata, the GPU-memory timeline, and links to everything.

Open report/index.html in a browser (videos play from the run directories). On a remote server:
`python -m http.server 8000` in the session directory, then browse to /report/.
"""

from __future__ import annotations

from bisect import bisect_left
import csv
from html import escape
import json
import os
from pathlib import Path
import sys

STYLE = """
body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;margin:24px;color:#1d2330;
     background:#fbfbfd;line-height:1.45;max-width:1400px}
h1,h2,h3{margin:.9em 0 .4em} a{color:#2457c5} code,pre{font-family:ui-monospace,Menlo,monospace}
pre{background:#f1f3f8;padding:10px;border-radius:6px;overflow:auto;max-height:420px;
    font-size:12.5px}
table{border-collapse:collapse;margin:8px 0;font-size:13.5px} td,th{border:1px solid #d9dde7;
    padding:4px 8px;vertical-align:top;text-align:left} th{background:#eef1f7}
.ok{color:#18794e;font-weight:600}.bad{color:#c4271d;font-weight:600}.muted{color:#6b7385}
.warn{color:#b26a00;font-weight:600}
.grid{display:flex;flex-wrap:wrap;gap:10px}.grid figure{margin:0;font-size:12px}
.grid img{max-width:260px;border:1px solid #d9dde7;border-radius:4px}
video{max-width:480px;border-radius:6px;background:#000}
details{margin:4px 0} summary{cursor:pointer}
.msg{border-left:4px solid #c9d1e3;padding:4px 10px;margin:8px 0;background:#fff}
.assistant{border-color:#2457c5}.thinking{border-color:#8a63d2;background:#f8f5ff}
.tool{border-color:#c98a00;background:#fffaf0}.result{border-color:#18794e}
.error{border-color:#c4271d;background:#fff4f3}.system{border-color:#6b7385}
svg{background:#fff;border:1px solid #d9dde7;border-radius:6px}
"""
PALETTE = ["#2457c5", "#c4271d", "#18794e", "#c98a00", "#8a63d2", "#0f8b8d", "#d4447c", "#5b6273"]


def page(title: str, body: str) -> str:
    return (
        f"<!doctype html><html><head><meta charset='utf-8'><title>{escape(title)}</title>"
        f"<style>{STYLE}</style></head><body>{body}</body></html>"
    )


def read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def rel(target: Path, start: Path) -> str:
    return escape(os.path.relpath(target, start))


def okfmt(value: object) -> str:
    if value is True:
        return "<span class='ok'>yes</span>"
    if value is False:
        return "<span class='bad'>no</span>"
    return "<span class='muted'>-</span>"


def array(value: object) -> list[float] | None:
    if isinstance(value, dict) and "$array_inline" in value:
        return [float(v) for v in value["$array_inline"]]
    if isinstance(value, list):
        return [float(v) for v in value]
    return None


def line_chart(
    series: dict[str, list[float]],
    title: str,
    width: int = 640,
    height: int = 220,
    bands: list[tuple[float, float, str, str]] | None = None,
) -> str:
    """Inline SVG line chart; x is the sample index. bands: (start, end, label, tooltip)."""
    series = {k: v for k, v in series.items() if v}
    if not series:
        return ""
    values = [y for ys in series.values() for y in ys]
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        lo, hi = lo - 1, hi + 1
    n = max(len(ys) for ys in series.values())
    pad_l, pad_b, pad_t = 52, 20, 22
    plot_w, plot_h = width - pad_l - 10, height - pad_b - pad_t

    def xy(i: int, y: float) -> str:
        x = pad_l + plot_w * (i / max(n - 1, 1))
        return f"{x:.1f},{pad_t + plot_h * (1 - (y - lo) / (hi - lo)):.1f}"

    parts = [
        f"<svg width='{width}' height='{height + 18 * ((len(series) + 3) // 4)}'>",
        f"<text x='{pad_l}' y='14' font-size='12' font-weight='600'>{escape(title)}</text>",
        f"<text x='4' y='{pad_t + 10}' font-size='10'>{hi:.3g}</text>",
        f"<text x='4' y='{pad_t + plot_h}' font-size='10'>{lo:.3g}</text>",
        f"<line x1='{pad_l}' y1='{pad_t + plot_h}' x2='{width - 10}' "
        f"y2='{pad_t + plot_h}' stroke='#ccc'/>",
    ]
    for index, (start, end, label, tip) in enumerate(bands or []):
        x0 = pad_l + plot_w * (start / max(n - 1, 1))
        x1 = pad_l + plot_w * (end / max(n - 1, 1))
        fill = "#e8eefb" if index % 2 == 0 else "#f3ecfb"
        parts.append(
            f"<rect x='{x0:.1f}' y='{pad_t}' width='{max(x1 - x0, 1):.1f}' height='{plot_h}' "
            f"fill='{fill}'><title>{escape(tip)}</title></rect>"
            f"<text x='{x0 + 2:.1f}' y='{pad_t + 10}' font-size='9' fill='#5b6273'>"
            f"{escape(label)}</text>"
        )
    for index, (name, ys) in enumerate(series.items()):
        color = PALETTE[index % len(PALETTE)]
        step = max(1, len(ys) // 800)  # keep the SVG light on long episodes
        points = " ".join(xy(i, ys[i]) for i in range(0, len(ys), step))
        parts.append(
            f"<polyline fill='none' stroke='{color}' stroke-width='1.4' points='{points}'/>"
        )
        lx = pad_l + (index % 4) * 150
        ly = height + 14 + 18 * (index // 4)
        parts.append(
            f"<rect x='{lx}' y='{ly - 9}' width='10' height='10' fill='{color}'/>"
            f"<text x='{lx + 14}' y='{ly}' font-size='11'>{escape(name)}</text>"
        )
    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------------------------- episodes --
def trajectories(run: Path) -> dict[str, dict[str, list[float]]]:
    charts: dict[str, dict[str, list[float]]] = {}
    path = run / "episode/steps.jsonl"
    if not path.exists():
        return charts
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                state = json.loads(line).get("robot_state") or {}
            except ValueError:
                continue
            for arm, joints in (state.get("joint_positions") or {}).items():
                values = array(joints) or []
                chart = charts.setdefault(f"{arm}: joint positions (rad)", {})
                for j, value in enumerate(values):
                    chart.setdefault(f"j{j}", []).append(value)
            for arm, width in (state.get("gripper_positions") or {}).items():
                charts.setdefault("grippers (0 closed, 1 open)", {}).setdefault(arm, []).append(
                    float(width)
                )
            for arm, pose in (state.get("end_effector_poses") or {}).items():
                position = array((pose or {}).get("position")) or []
                chart = charts.setdefault(f"{arm}: end-effector position (m)", {})
                for axis, value in zip("xyz", position, strict=False):
                    chart.setdefault(axis, []).append(value)
    return charts


def call_rows(run: Path) -> list[dict[str, object]]:
    spans: dict[str, dict[str, object]] = {}
    started: dict[str, float] = {}  # simulator clock when each span began
    events = run / "trace/events.jsonl"
    if events.exists():
        for line in events.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("event") == "span_start" and event.get("simulator_time_s") is not None:
                started[event["span_id"]] = float(event["simulator_time_s"])
            if event.get("event") == "span_end":
                spans[event["span_id"]] = event
    depth: dict[str, int] = {}
    rows = []
    for call_dir in sorted((run / "trace/calls").glob("call-*")):
        number = call_dir.name.split("-")[1]
        span = spans.get(f"span-{number}", {})
        parent = span.get("parent_id")
        depth[f"span-{number}"] = depth.get(parent, -1) + 1 if parent else 0
        output = read_json(call_dir / "output.json")
        error = (output or {}).get("error") if isinstance(output, dict) else None
        failure = read_json(call_dir / "error.json")
        rows.append(
            {
                "index": int(number),
                "name": span.get("name") or call_dir.name.split("-", 2)[-1],
                "category": span.get("category", ""),
                "depth": depth[f"span-{number}"],
                "ok": span.get("ok"),
                "duration_s": span.get("duration_s"),
                # Every event carries the absolute simulator clock: the call's own
                # simulated time is its end clock minus its start clock.
                "sim_time_s": (
                    round(float(span["simulator_time_s"]) - started[f"span-{number}"], 3)
                    if span.get("simulator_time_s") is not None and f"span-{number}" in started
                    else None
                ),
                "error": (
                    f"{error.get('code')}: {error.get('message')}"
                    if isinstance(error, dict)
                    else (
                        f"{failure.get('type')}: {failure.get('message')}"
                        if isinstance(failure, dict)
                        else ""
                    )
                ),
                "dir": call_dir,
            }
        )
    return rows


def episode_page(run: Path, out_dir: Path) -> tuple[str, dict[str, object]]:
    outcome = read_json(run / "outcome.json") or {}
    meta = read_json(run / "run.json") or {}
    name = (
        f"{meta.get('benchmark', '?')} / {meta.get('suite', '?')} / task {meta.get('task_id', '?')}"
        f" ({meta.get('task_name', '')}) / seed {meta.get('seed', '?')}"
    )
    body = [
        f"<p><a href='index.html'>&larr; session</a></p><h1>{escape(name)}</h1>",
        f"<p class='muted'>{escape(str(run))}</p>",
        "<table><tr><th>program ok</th><th>task success</th><th>termination</th>"
        "<th>steps</th><th>sim time</th><th>wall</th><th>error</th></tr>",
        f"<tr><td>{okfmt(outcome.get('program_ok'))}</td>"
        f"<td>{okfmt(outcome.get('task_success'))}</td>"
        f"<td>{escape(str(outcome.get('termination_reason', '')))}</td>"
        f"<td>{outcome.get('steps_executed', '')}</td>"
        f"<td>{outcome.get('simulator_duration_s', '')} s</td>"
        f"<td>{round(outcome.get('wall_duration_s') or 0, 1)} s</td>"
        f"<td>{escape(json.dumps(outcome.get('error'))[:300]) if outcome.get('error') else ''}"
        "</td></tr></table>",
    ]

    videos = sorted((run / "media/videos").glob("*.mp4"))
    if videos:
        body.append("<h2>Videos</h2><div class='grid'>")
        body += [
            f"<figure><video controls preload='metadata' src='{rel(v, out_dir)}'></video>"
            f"<figcaption>{escape(v.stem)}</figcaption></figure>"
            for v in videos
        ]
        body.append("</div>")

    charts = trajectories(run)
    if charts:
        body.append("<h2>Trajectories (one point per control step)</h2><div class='grid'>")
        body += [line_chart(series, title) for title, series in charts.items()]
        body.append("</div>")

    timing = (outcome.get("timing") or {}).get("buckets") or {}
    if timing:
        body.append(
            "<h2>Where the time went</h2><table><tr><th>bucket</th><th>seconds</th>"
            "<th>calls</th></tr>"
        )
        body += [
            f"<tr><td>{escape(k)}</td><td>{v.get('seconds', 0):.2f}</td>"
            f"<td>{v.get('calls', '')}</td></tr>"
            for k, v in timing.items()
        ]
        body.append("</table>")

    # Annotated images only: per-candidate masks are for the dashboard's mask viewer.
    overlays = [
        p for p in sorted((run / "media/overlays").rglob("*.png")) if "-mask-" not in p.name
    ]
    if overlays:
        body.append("<h2>SAM3 and grasp overlays</h2><div class='grid'>")
        body += [
            f"<figure><a href='{rel(o, out_dir)}'><img src='{rel(o, out_dir)}'></a>"
            f"<figcaption>{escape(str(o.relative_to(run / 'media/overlays')))}"
            "</figcaption></figure>"
            for o in overlays
        ]
        body.append("</div>")

    keyframes = sorted(p for p in (run / "media/keyframes").glob("*") if p.is_dir())
    if keyframes:
        body.append("<h2>Keyframes (before / after each motion call)</h2>")
        for frame_dir in keyframes:
            images = sorted(frame_dir.glob("*.png"))
            body.append(f"<details><summary>{escape(frame_dir.name)}</summary><div class='grid'>")
            body += [
                f"<figure><img loading='lazy' src='{rel(i, out_dir)}'>"
                f"<figcaption>{escape(i.stem)}</figcaption></figure>"
                for i in images
            ]
            body.append("</div></details>")

    rows = call_rows(run)
    if rows:
        body.append(
            "<h2>Tool calls</h2><p class='muted'>program = the generated program's "
            "call; api = a stage inside it; provider = a model service. Click a row "
            "for its full input and output.</p><table><tr><th>#</th><th>call</th>"
            "<th>kind</th><th>ok</th><th>s</th><th>sim t</th><th>error</th></tr>"
        )
        for row in rows:
            indent = "&nbsp;" * 4 * int(row["depth"])
            io = "".join(
                f"<details><summary>{part}</summary><pre>"
                f"{escape((row['dir'] / f'{part}.json').read_text(encoding='utf-8')[:20000])}"
                "</pre></details>"
                for part in ("input", "output", "error")
                if (row["dir"] / f"{part}.json").exists()
            )
            duration = row["duration_s"]
            body.append(
                f"<tr><td>{row['index']}</td><td>{indent}<b>{escape(str(row['name']))}</b>{io}"
                f"</td><td>{escape(str(row['category']))}</td><td>{okfmt(row['ok'])}</td>"
                f"<td>{'' if duration is None else f'{duration:.2f}'}</td>"
                f"<td>{row['sim_time_s'] if row['sim_time_s'] is not None else ''}</td>"
                f"<td>{escape(str(row['error'])[:300])}</td></tr>"
            )
        body.append("</table>")

    for title, path in (
        ("Program", run / "source/program.py"),
        ("Program stdout", run / "logs/program.stdout"),
        ("Program result", run / "source/program-result.json"),
    ):
        if path.exists() and path.stat().st_size:
            text = path.read_text(encoding="utf-8")
            if path.suffix == ".json":
                data = read_json(path)
                text = json.dumps((data or {}).get("result", data), indent=2)
            body.append(f"<h2>{title}</h2><pre>{escape(text[:60000])}</pre>")

    summary = {
        "name": name,
        "program_ok": outcome.get("program_ok"),
        "task_success": outcome.get("task_success"),
        "steps": outcome.get("steps_executed"),
        "wall_s": outcome.get("wall_duration_s"),
        "termination": outcome.get("termination_reason"),
        "calls": len(rows),
        "failed_calls": sum(1 for r in rows if r["ok"] is False),
    }
    return page(name, "".join(body)), summary


# ----------------------------------------------------------------------------------- agents --
def content_html(content: object, out_dir: Path) -> str:
    if isinstance(content, str):
        return f"<pre>{escape(content[:20000])}</pre>"
    parts = []
    for block in content or []:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text":
            parts.append(f"<pre>{escape(block.get('text', '')[:20000])}</pre>")
        elif kind == "image":
            source = block.get("source") or {}
            if source.get("type") == "base64":
                parts.append(
                    f"<img style='max-width:420px' src='data:"
                    f"{escape(source.get('media_type', 'image/png'))};base64,"
                    f"{source.get('data', '')}'>"
                )
        else:
            parts.append(f"<pre>{escape(json.dumps(block)[:4000])}</pre>")
    return "".join(parts)


def transcript_events(path: Path) -> list[dict]:
    events = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            continue
    return events


def agent_page(path: Path, out_dir: Path, title: str) -> tuple[str, dict[str, object]]:
    body = [
        f"<p><a href='index.html'>&larr; session</a></p><h1>{escape(title)}</h1>",
        f"<p class='muted'>{escape(str(path))}</p>",
    ]
    prompt = path.parent / "prompt.md"
    if prompt.exists():
        body.append(
            f"<details open><summary><b>Prompt given to the agent</b></summary>"
            f"<pre>{escape(prompt.read_text(encoding='utf-8'))}</pre></details>"
        )
    stats = {
        "title": title,
        "turns": None,
        "cost_usd": None,
        "minutes": None,
        "tool_calls": 0,
        "tool_errors": 0,
        "result": None,
    }
    tool_names: dict[str, str] = {}
    for event in transcript_events(path):
        kind = event.get("type")
        message = event.get("message") or {}
        if kind == "system" and event.get("subtype") == "init":
            body.append(
                f"<div class='msg system'><b>session start</b> model "
                f"<code>{escape(str(event.get('model')))}</code>, tools: "
                f"{escape(', '.join(map(str, event.get('tools') or [])))}</div>"
            )
        elif kind == "assistant":
            for block in message.get("content") or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text" and block.get("text", "").strip():
                    body.append(
                        f"<div class='msg assistant'><b>assistant</b>"
                        f"<pre>{escape(block['text'])}</pre></div>"
                    )
                elif block.get("type") == "thinking":
                    text = block.get("thinking") or "(thinking hidden by the API)"
                    body.append(
                        f"<div class='msg thinking'><details><summary>thinking"
                        f"</summary><pre>{escape(text)}</pre></details></div>"
                    )
                elif block.get("type") == "tool_use":
                    stats["tool_calls"] += 1
                    tool_names[block.get("id", "")] = block.get("name", "?")
                    tool_input = block.get("input") or {}
                    shown = (
                        tool_input.get("command")
                        if block.get("name") == "Bash"
                        else json.dumps(tool_input, indent=2)
                    )
                    body.append(
                        f"<div class='msg tool'><b>tool call: "
                        f"{escape(str(block.get('name')))}</b>"
                        f"<pre>{escape(str(shown)[:20000])}</pre></div>"
                    )
        elif kind == "user":
            for block in message.get("content") or []:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    error = bool(block.get("is_error"))
                    stats["tool_errors"] += int(error)
                    name = tool_names.get(block.get("tool_use_id", ""), "tool")
                    body.append(
                        f"<div class='msg {'error' if error else 'result'}'><details>"
                        f"<summary>{escape(name)} result{' (error)' if error else ''}"
                        f"</summary>{content_html(block.get('content'), out_dir)}"
                        "</details></div>"
                    )
                elif isinstance(block, dict) and block.get("type") == "text":
                    body.append(
                        f"<div class='msg system'><b>user</b>"
                        f"<pre>{escape(block.get('text', '')[:20000])}</pre></div>"
                    )
            if isinstance(message.get("content"), str):
                body.append(
                    f"<div class='msg system'><b>user</b>"
                    f"<pre>{escape(message['content'][:20000])}</pre></div>"
                )
        elif kind == "result":
            stats.update(
                turns=event.get("num_turns"),
                cost_usd=event.get("total_cost_usd"),
                minutes=round((event.get("duration_ms") or 0) / 60000, 1),
                result=event.get("subtype"),
            )
            body.append(
                f"<div class='msg result'><b>finished: {escape(str(event.get('subtype')))}"
                f"</b> &middot; {stats['turns']} turns &middot; {stats['minutes']} min"
                f" &middot; est. ${stats['cost_usd']}</div>"
            )
    report = path.parent / "REPORT.md"
    if report.exists():
        body.insert(
            3,
            f"<details open><summary><b>Agent's REPORT.md</b></summary>"
            f"<pre>{escape(report.read_text(encoding='utf-8'))}</pre></details>",
        )
    programs = sorted(path.parent.glob("program_*.py")) + sorted(path.parent.glob("*_check.py"))
    if programs:
        body.insert(
            3,
            "<details><summary><b>Programs written</b></summary>"
            + "".join(
                f"<h3>{escape(p.name)}</h3><pre>{escape(p.read_text(encoding='utf-8'))}</pre>"
                for p in programs
            )
            + "</details>",
        )
    return page(title, "".join(body)), stats


# ------------------------------------------------------------------------------------ index --
def load_gpu(session: Path) -> dict[str, object]:
    """Samples of the session's GPU: times, total used / utilization, and your processes."""
    gpu_csv, proc_csv = session / "gpu/gpu.csv", session / "gpu/processes.csv"
    target = str((read_json(session / "session.json") or {}).get("gpu", "0"))
    used: dict[float, float] = {}
    util: dict[float, float] = {}
    yours: dict[float, float] = {}
    reserved: dict[float, float] = {}
    if gpu_csv.exists():
        with gpu_csv.open() as handle:
            for row in csv.DictReader(handle):
                if row["gpu"] == target:
                    used[float(row["t_s"])] = float(row["used_mib"] or 0) / 1024
                    util[float(row["t_s"])] = float(row["util_pct"] or 0)
    if proc_csv.exists():
        with proc_csv.open() as handle:
            for row in csv.DictReader(handle):
                if row["gpu"] == target and row["yours"] == "1":
                    t = float(row["t_s"])
                    # The reservation is memory held in reserve, not what ASPIRE needs.
                    bucket = reserved if row.get("role") == "reservation" else yours
                    bucket[t] = bucket.get(t, 0.0) + float(row["used_mib"] or 0) / 1024
    return {
        "target": target,
        "times": sorted(used),
        "used": used,
        "util": util,
        "yours": yours,
        "reserved": reserved,
    }


def load_stages(session: Path, gpu: dict[str, object]) -> list[dict[str, object]]:
    """stages.tsv rows (stage, status, detail, start_s, end_s) plus your peak GPU memory."""
    path = session / "stages.tsv"
    rows: list[dict[str, object]] = []
    if not path.exists():
        return rows
    yours = gpu["yours"]
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cells = (line.split("\t") + [""] * 5)[:5]
        row: dict[str, object] = dict(zip(("stage", "status", "detail"), cells[:3], strict=True))
        try:
            row["start"], row["end"] = float(cells[3]), float(cells[4])
        except ValueError:
            row["start"] = row["end"] = None
        if row["start"] is not None:
            inside = [v for t, v in yours.items() if row["start"] <= t <= row["end"] + 2]
            row["peak"] = max(inside) if inside else None
        rows.append(row)
    return rows


def gpu_section(gpu: dict[str, object], stages: list[dict[str, object]]) -> str:
    times, used, util, yours = gpu["times"], gpu["used"], gpu["util"], gpu["yours"]
    reserved = gpu["reserved"]
    target = gpu["target"]
    if not times:
        return ""

    def index_of(t: float) -> float:
        return float(bisect_left(times, t)) if t <= times[-1] else float(len(times) - 1)

    bands = [
        (index_of(r["start"]), index_of(r["end"]), str(r["n"]),
         f"#{r['n']} {r['stage']}: {r['status']} ({r['end'] - r['start']:.0f} s)")
        for r in stages
        if r.get("start") is not None and r["end"] > r["start"]
    ]  # fmt: skip
    chart = line_chart(
        {
            "GPU total used (GiB)": [used[t] for t in times],
            "your services and simulators (GiB)": [yours.get(t, 0.0) for t in times],
            "your reservation (GiB)": [reserved.get(t, 0.0) for t in times],
        },
        f"GPU {target} memory over the session ({times[-1]:.0f} s); "
        "shaded bands are the numbered stages",
        900,
        220,
        bands,
    )
    util_chart = line_chart(
        {"utilization %": [util[t] for t in times]},
        f"GPU {target} utilization (all users)",
        900,
        160,
        bands,
    )
    peak = max(yours.values()) if yours else 0.0
    return (
        f"<h2>GPU {target}</h2><p>Peak memory of your services and simulators: "
        f"<b>{peak:.1f} GiB</b> (the reservation is shown separately)</p>{chart}<br>{util_chart}"
    )


STATUS_CLASS = {
    "PASS": "ok",
    "FAIL": "bad",
    "WARN": "warn",
    "BLOCKED": "warn",
    "INTERRUPTED": "bad",
}


def stages_table(stages: list[dict[str, object]]) -> str:
    if not stages:
        return ""

    def fmt(value: object, digits: int = 0) -> str:
        return "" if value is None else f"{value:.{digits}f}"

    rows = []
    for r in stages:
        status = escape(str(r["status"]))
        css = STATUS_CLASS.get(str(r["status"]), "muted")
        duration = None if r.get("start") is None else r["end"] - r["start"]
        rows.append(
            f"<tr><td>{r['n']}</td><td>{escape(str(r['stage']))}</td>"
            f"<td class='{css}'>{status}</td><td>{escape(str(r['detail']))}</td>"
            f"<td>{fmt(r.get('start'))}</td><td>{fmt(duration)}</td>"
            f"<td>{fmt(r.get('peak'), 1)}</td></tr>"
        )
    return (
        "<h2>Stages</h2><table><tr><th>#</th><th>stage</th><th>status</th><th>detail</th>"
        "<th>start s</th><th>duration s</th><th>your GPU peak GiB</th></tr>"
        + "".join(rows)
        + "</table>"
    )


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    session = Path(sys.argv[1]).resolve()
    out_dir = session / "report"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = read_json(session / "session.json") or {}

    episodes = []
    runs = sorted(p for p in session.rglob("outcome.json") if "wandb" not in p.parts)
    for index, run in enumerate(runs):
        run_dir = run.parent
        html, summary = episode_page(run_dir, out_dir)
        name = f"episode-{index:03d}.html"
        (out_dir / name).write_text(html, encoding="utf-8")
        where = os.path.relpath(run_dir, session)
        summary.update(page=name, where=where)
        episodes.append(summary)

    agents = []
    transcripts = sorted(p for p in session.rglob("transcript.jsonl") if "wandb" not in p.parts)
    transcripts += sorted((session / "claude-sessions").rglob("*.jsonl"))
    for index, path in enumerate(transcripts):
        title = os.path.relpath(path, session)
        html, stats = agent_page(path, out_dir, title)
        name = f"agent-{index:03d}.html"
        (out_dir / name).write_text(html, encoding="utf-8")
        stats["page"] = name
        agents.append(stats)

    body = [
        f"<h1>ASPIRE session: {escape(str(meta.get('name', session.name)))}</h1>",
        "<table>"
        + "".join(
            f"<tr><th>{escape(k)}</th><td>{escape(str(v))}</td></tr>" for k, v in meta.items()
        )
        + "</table>",
    ]
    gpu = load_gpu(session)
    stages = load_stages(session, gpu)
    for number, row in enumerate(stages, 1):
        row["n"] = number
    body.append(stages_table(stages))
    body.append(
        gpu_section(gpu, [r for r in stages if not str(r["stage"]).startswith("preflight: ")])
    )
    body.append(f"<h2>Agent conversations ({len(agents)})</h2>")
    if agents:
        body.append(
            "<table><tr><th>conversation</th><th>result</th><th>turns</th>"
            "<th>tool calls</th><th>tool errors</th><th>minutes</th><th>est. cost</th>"
            "</tr>"
            + "".join(
                f"<tr><td><a href='{a['page']}'>{escape(a['title'])}</a></td>"
                f"<td>{escape(str(a['result']))}</td><td>{a['turns']}</td>"
                f"<td>{a['tool_calls']}</td><td>{a['tool_errors']}</td>"
                f"<td>{a['minutes']}</td><td>{a['cost_usd']}</td></tr>"
                for a in agents
            )
            + "</table>"
        )
    body.append(f"<h2>Episodes ({len(episodes)})</h2>")
    if episodes:
        body.append(
            "<table><tr><th>episode</th><th>program ok</th><th>task success</th>"
            "<th>steps</th><th>wall s</th><th>calls (failed)</th><th>termination</th>"
            "<th>directory</th></tr>"
            + "".join(
                f"<tr><td><a href='{e['page']}'>{escape(e['name'])}</a></td>"
                f"<td>{okfmt(e['program_ok'])}</td><td>{okfmt(e['task_success'])}</td>"
                f"<td>{e['steps']}</td><td>{round(e['wall_s'] or 0, 1)}</td>"
                f"<td>{e['calls']} ({e['failed_calls']})</td>"
                f"<td>{escape(str(e['termination']))}</td>"
                f"<td class='muted'>{escape(e['where'])}</td></tr>"
                for e in episodes
            )
            + "</table>"
        )
    files = [session / "session.log", session / "stages.tsv"]
    for folder in ("env", "preflight", "service-logs", "episodes/logs"):
        files += sorted(p for p in (session / folder).glob("*") if p.is_file())
    links = [(p, os.path.relpath(p, out_dir)) for p in files if p.exists()]
    if links:
        body.append(
            "<h2>Console, environment snapshot, preflight and service logs</h2><ul>"
            + "".join(
                f"<li><a href='{escape(r)}'>{escape(os.path.relpath(p, session))}</a></li>"
                for p, r in links
            )
            + "</ul>"
        )
    (out_dir / "index.html").write_text(page("ASPIRE session", "".join(body)), encoding="utf-8")
    print(
        f"report: {out_dir / 'index.html'} ({len(episodes)} episodes, "
        f"{len(agents)} agent conversations)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
