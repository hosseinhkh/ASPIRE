"""The recorded-session tools in scripts/gpu: HTML report and credential redaction (CPU only)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
GPU_SCRIPTS = REPO / "scripts/gpu"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _fake_session(root: Path) -> Path:
    session = root / "20260101-000000-test"
    session.mkdir()
    (session / "session.json").write_text(json.dumps({"name": "test", "gpu": "0"}))
    (session / "stages.tsv").write_text(
        "preflight: torch\tPASS\tok\t\t\n"
        "episode cube seed 1\tPASS\tprogram_ok=True task_success=True\t2.0\t9.0\n"
        "episode drawer seed 1\tSKIP\tservice down: sam3\t9.0\t9.1\n"
    )
    gpu = session / "gpu"
    gpu.mkdir()
    (gpu / "gpu.csv").write_text(
        "t_s,gpu,used_mib,free_mib,util_pct,temp_c,power_w\n"
        + "".join(f"{t},0,{60000 + 100 * t},30000,{t},50,100\n" for t in range(0, 12, 2))
    )
    (gpu / "processes.csv").write_text(
        "t_s,gpu,pid,role,yours,used_mib\n"
        + "".join(f"{t},0,{100 + t},sam3,1,{4096 + 512 * t}\n" for t in range(0, 12, 2))
        + "".join(f"{t},0,999,reservation,1,30720\n" for t in range(0, 12, 2))
    )

    run = session / "episodes/robosuite/cube_lifting/00-lift/0001/run-x"
    (run / "source").mkdir(parents=True)
    (run / "run.json").write_text(
        json.dumps({"benchmark": "robosuite", "suite": "cube_lifting", "task_id": 0, "seed": 1})
    )
    (run / "outcome.json").write_text(
        json.dumps({"program_ok": True, "task_success": True, "steps_executed": 2})
    )
    (run / "source/program.py").write_text("move_to_pose(target)\n")
    state = {
        "joint_positions": {"primary": {"$array_inline": [0.1, 0.2]}},
        "gripper_positions": {"primary": 1.0},
        "end_effector_poses": {"primary": {"position": {"$array_inline": [0.4, 0.0, 0.9]}}},
    }
    _write_jsonl(run / "episode/steps.jsonl", [{"robot_state": state}, {"robot_state": state}])
    _write_jsonl(
        run / "trace/events.jsonl",
        [
            {"event": "span_start", "span_id": "span-000001", "name": "move_to_pose"},
            {
                "event": "span_end",
                "span_id": "span-000001",
                "name": "move_to_pose",
                "category": "program",
                "ok": True,
                "duration_s": 1.5,
            },
        ],
    )
    call = run / "trace/calls/call-000001-move-to-pose"
    call.mkdir(parents=True)
    (call / "input.json").write_text(json.dumps({"target": [0.4, 0.0, 1.0]}))
    (call / "output.json").write_text(json.dumps({"ok": True}))

    _write_jsonl(
        session / "agents/llm-demo/transcript.jsonl",
        [
            {"type": "system", "subtype": "init", "model": "opus", "session_id": "s1"},
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "thinking", "thinking": "try a staged sweep"},
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "Bash",
                            "input": {"command": "ls"},
                        },
                    ]
                },
            },
            {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "is_error": True,
                            "content": "boom",
                        }
                    ]
                },
            },
            {"type": "result", "subtype": "success", "num_turns": 2, "total_cost_usd": 0.5},
        ],
    )
    return session


def test_session_report_links_stages_episodes_and_agents(tmp_path: Path) -> None:
    session = _fake_session(tmp_path)
    subprocess.run(
        [sys.executable, str(GPU_SCRIPTS / "session_report.py"), str(session)],
        check=True,
        capture_output=True,
    )
    report = session / "report"
    index = (report / "index.html").read_text()
    assert "episode cube seed 1" in index and "service down: sam3" in index
    assert "episode-000.html" in index and "agent-000.html" in index
    assert "<rect" in index  # stage bands on the GPU chart
    # Your GPU peak during the cube stage, in GiB; the 30 GiB reservation is charted apart.
    assert "<td>9.0</td>" in index and "your reservation" in index
    episode = (report / "episode-000.html").read_text()
    assert "move_to_pose" in episode and "end-effector position" in episode
    agent = (report / "agent-000.html").read_text()
    assert "try a staged sweep" in agent and "boom" in agent


def test_redact_session_replaces_credential_values(tmp_path: Path) -> None:
    session = _fake_session(tmp_path)
    secret = "sk-test-0123456789abcdef"
    leak = session / "agents/llm-demo/transcript.jsonl"
    leak.write_text(leak.read_text() + json.dumps({"env": f"MY_API_KEY={secret}"}) + "\n")
    env = {**os.environ, "MY_API_KEY": secret, "HF_HOME": str(tmp_path / "hf")}
    result = subprocess.run(
        [sys.executable, str(GPU_SCRIPTS / "redact_session.py"), str(session)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert secret not in leak.read_text()
    assert "[REDACTED:MY_API_KEY]" in leak.read_text()
    assert "1 secret value(s) replaced in 1 file(s)" in result.stdout


def test_dashboard_serves_runs_and_streams_zips(tmp_path: Path) -> None:
    import io
    import threading
    from urllib.request import urlopen
    import zipfile

    sys.path.insert(0, str(GPU_SCRIPTS))
    import dashboard

    outputs = tmp_path / "outputs"
    (outputs / "sessions").mkdir(parents=True)
    session = _fake_session(outputs / "sessions")
    video = next(session.rglob("run-x")) / "media/videos/agentview.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"\x00" * 2048)
    store = dashboard.Store(outputs)
    server = dashboard.ThreadingHTTPServer(
        ("127.0.0.1", 0), dashboard.make_handler(store, dashboard.LiveManager(store, []))
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def get(path: str) -> bytes:
        with urlopen(base + path, timeout=10) as response:
            return response.read()

    try:
        items = json.loads(get("/api/collections"))["items"]
        assert [item["kind"] for item in items] == ["session"]
        rel = items[0]["path"]
        collection = json.loads(get(f"/api/collection?path={rel}"))
        assert len(collection["episodes"]) == 1 and len(collection["agents"]) == 1
        episode = json.loads(get(f"/api/episode?path={collection['episodes'][0]['path']}"))
        assert episode["calls"][0]["name"] == "move_to_pose" and episode["media"]["videos"]
        call = json.loads(get(f"/api/call?path={episode['calls'][0]['dir']}"))
        assert call["input"] == {"target": [0.4, 0.0, 1.0]}
        agent = json.loads(get(f"/api/agent?path={collection['agents'][0]['path']}"))
        assert [e["kind"] for e in agent["events"]][:3] == ["init", "thinking", "tool_use"]
        full = zipfile.ZipFile(io.BytesIO(get(f"/download?path={rel}&media=1")))
        logs = zipfile.ZipFile(io.BytesIO(get(f"/download?path={rel}&media=0")))
        assert full.testzip() is None and logs.testzip() is None
        assert any(n.endswith(".mp4") for n in full.namelist())
        assert not any(n.endswith(".mp4") for n in logs.namelist())
        assert any(n.endswith("transcript.jsonl") for n in logs.namelist())
        # Tasks list their runs and the folder holding them; failures group unsolved runs.
        task = collection["tasks"][0]
        assert task["folder"] == collection["episodes"][0]["path"] and len(task["runs"]) == 1
        assert collection["failures"] == []
        # Several paths in one zip keep their layout under the folder that holds them all.
        two = zipfile.ZipFile(
            io.BytesIO(get(f"/download?media=0&name=pair&path={rel}/stages.tsv&path={rel}/gpu"))
        )
        assert sorted(two.namelist()) == [
            f"{session.name}/gpu/gpu.csv",
            f"{session.name}/gpu/processes.csv",
            f"{session.name}/stages.tsv",
        ]
        try:
            get("/api/collection?path=../..")
            raise AssertionError("a path outside the outputs directory was served")
        except Exception as exc:  # urllib raises HTTPError 400
            assert "400" in str(exc)
    finally:
        server.shutdown()


def test_latency_report_uses_simulated_durations(tmp_path: Path) -> None:
    run = tmp_path / "run"
    (run / "trace").mkdir(parents=True)
    (run / "run.json").write_text("{}")
    _write_jsonl(
        run / "trace/events.jsonl",
        [
            # simulator_time_s is the absolute clock: 10 s in when the motion starts
            {"event": "span_start", "span_id": "span-000001", "simulator_time_s": 10.0},
            {"event": "span_start", "span_id": "span-000002", "simulator_time_s": 10.0},
            {
                "event": "span_end",
                "span_id": "span-000002",
                "name": "pyroki.solve_ik",
                "category": "provider",
                "duration_s": 0.25,
                "simulator_time_s": 10.0,
            },
            {
                "event": "span_end",
                "span_id": "span-000001",
                "name": "execute_trajectory",
                "category": "api",
                "duration_s": 4.0,
                "simulator_time_s": 12.0,
            },
        ],
    )
    out = subprocess.run(
        [sys.executable, str(GPU_SCRIPTS / "latency.py"), str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "provider: pyroki.solve_ik" in out and "0.250" in out
    assert "median 2.00" in out  # 4 s of wall time for 2 simulated seconds


def _sam3_call(run: Path, number: int, *, ok: bool, masks: int, clock: float) -> list[dict]:
    """One recorded SAM3 provider request: call folder plus its two trace events."""
    call = run / f"trace/calls/call-{number:06d}-sam3-segment-text"
    call.mkdir(parents=True)
    (call / "input.json").write_text(json.dumps({"prompt": "soup"}))
    segmentations = [{"label": "soup", "score": 0.5}] * masks
    (call / "output.json").write_text(json.dumps({"ok": ok, "segmentations": segmentations}))
    span = f"span-{number:06d}"
    return [
        {"event": "span_start", "span_id": span, "simulator_time_s": clock},
        {
            "event": "span_end",
            "span_id": span,
            "name": "sam3.segment_text",
            "category": "provider",
            "ok": ok,
            "duration_s": 1.2,
            "simulator_time_s": clock,
        },
    ]


def test_dashboard_store_pairs_media_compares_runs_and_filters_logs(tmp_path: Path) -> None:
    sys.path.insert(0, str(GPU_SCRIPTS))
    from dashboard_store import Store

    outputs = tmp_path / "outputs"
    (outputs / "sessions").mkdir(parents=True)
    session = _fake_session(outputs / "sessions")
    run = next(session.rglob("run-x"))
    # Keyframes: one before/after pair for the move_to_pose call, plus periodic frames.
    for name in ("000001-reset", "000002-before-move-to-pose", "000003-after-move-to-pose"):
        (run / "media/keyframes" / name).mkdir(parents=True)
        (run / "media/keyframes" / name / "agentview.png").write_bytes(b"png")
    # Three SAM3 requests: the first fails (no image saved), the next two return masks.
    events = [json.loads(line) for line in (run / "trace/events.jsonl").read_text().splitlines()]
    events += _sam3_call(run, 2, ok=False, masks=0, clock=1.0)
    events += _sam3_call(run, 3, ok=True, masks=2, clock=2.0)
    events += _sam3_call(run, 4, ok=True, masks=1, clock=3.0)
    _write_jsonl(run / "trace/events.jsonl", events)
    overlays = run / "media/overlays"
    overlays.mkdir(parents=True)
    # An older recording without the call name, then a newer one that names it.
    (overlays / "sam3-000001-camera-agentview.json").write_text(
        json.dumps({"camera_name": "agentview", "candidates": [{"label": "first", "score": 0.12}]})
    )
    (overlays / "sam3-000002-camera-agentview.json").write_text(
        json.dumps(
            {
                "camera_name": "agentview",
                "candidates": [{"label": "second", "score": 0.9}],
                "call": "call-000004-sam3-segment-text",
            }
        )
    )
    (overlays / "sam3-000001-camera-agentview.png").write_bytes(b"png")

    store = Store(outputs)
    episode = store.episode(run)
    spans = {span["index"]: span for span in episode["spans"]}
    assert spans[1]["keyframes"]["before"][0].endswith("000002-before-move-to-pose/agentview.png")
    assert spans[1]["keyframes"]["after"][0].endswith("000003-after-move-to-pose/agentview.png")
    assert "overlay" not in spans[2]  # the failed request saved no image
    # Older recordings kept no per-candidate masks: the viewer falls back to the image.
    assert spans[3]["overlay"]["candidates"] == [{"label": "first", "score": 0.12, "mask": None}]
    assert spans[3]["overlay"]["image"].endswith("sam3-000001-camera-agentview.png")
    assert spans[4]["overlay"]["candidates"] == [{"label": "second", "score": 0.9, "mask": None}]
    # Only the low-confidence segmentation is flagged, linked to its call.
    assert [(w["level"], w["span"]) for w in episode["warnings"]] == [("warn", "span-000003")]
    assert "first" in episode["warnings"][0]["text"]
    # A close that ended empty is flagged too (diagnostics.closed_empty from close_gripper).
    grip = run / "trace/calls/call-000009-close-gripper"
    grip.mkdir(parents=True)
    (grip / "output.json").write_text(
        json.dumps({"ok": True, "diagnostics": {"closed_empty": True}})
    )
    _write_jsonl(
        run / "trace/events.jsonl",
        events
        + [
            {"event": "span_start", "span_id": "span-000009", "simulator_time_s": 4.0},
            {"event": "span_end", "span_id": "span-000009", "name": "close_gripper",
             "category": "program", "ok": True, "simulator_time_s": 4.5},
        ],
    )  # fmt: skip
    warnings = store.episode(run)["warnings"]
    assert ("warn", "span-000009") in [(w["level"], w["span"]) for w in warnings]
    assert spans[4]["overlay"]["image"] is None
    assert episode["duration_s"] == 3.0

    # A second agent that reports the window and thinking Claude Code really used.
    usage = {"opus-4-6": {"contextWindow": 1_000_000, "thinkingTokens": 7, "costUSD": 1.0}}
    _write_jsonl(
        session / "agents/llm-two/transcript.jsonl",
        [{"type": "result", "subtype": "success", "total_cost_usd": 1.0, "modelUsage": usage}],
    )
    meta = json.loads((session / "session.json").read_text())
    meta.update(claude_effort="max", harness="published")
    (session / "session.json").write_text(json.dumps(meta))

    # Comparing a session with itself gives two identical columns over the same task rows.
    compared = store.benchmark([session, session])
    assert [column["successes"] for column in compared["columns"]] == [1, 1]
    assert [row["task_id"] for row in compared["rows"]] == [0]
    column = compared["columns"][0]
    assert column["llm_cost_usd"] == 1.5
    assert column["agent_models"] == {
        "opus-4-6": {"context_window": 1_000_000, "thinking_tokens": 7}
    }
    assert (column["claude_effort"], column["harness"]) == ("max", "published")
    assert "provider: sam3.segment_text" in compared["columns"][0]["providers"]

    log = session / "logs/agent.log"
    log.parent.mkdir()
    log.write_text("starting\nTraceback (most recent call last)\nretrying sam3\ndone\n")
    assert [line["n"] for line in store.log(log, alerts=True)["lines"]] == [2]
    assert [line["n"] for line in store.log(log, query="SAM3")["lines"]] == [3]
    assert [line["n"] for line in store.log(log, tail=2)["lines"]] == [3, 4]


def test_dashboard_serves_static_assets_only_from_its_folder(tmp_path: Path) -> None:
    import threading
    from urllib.error import HTTPError
    from urllib.request import urlopen

    sys.path.insert(0, str(GPU_SCRIPTS))
    import dashboard

    (tmp_path / "outputs").mkdir()
    store = dashboard.Store(tmp_path / "outputs")
    server = dashboard.ThreadingHTTPServer(
        ("127.0.0.1", 0), dashboard.make_handler(store, dashboard.LiveManager(store, []))
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urlopen(base + "/static/app.js", timeout=10) as response:
            assert "javascript" in response.headers["Content-Type"]
            assert b"import" in response.read()
        for name in [
            "lib/charts.js",
            "lib/ui.js",
            "lib/util.js",
            "views/episode.js",
            "views/compare.js",
        ]:
            with urlopen(f"{base}/static/{name}", timeout=10) as response:
                assert response.status == 200
        for escape in ["../../dashboard.py", "..%2F..%2Fdashboard.py"]:
            try:
                urlopen(f"{base}/static/{escape}", timeout=10)
                raise AssertionError(f"served a file outside the static folder: {escape}")
            except HTTPError as exc:
                assert exc.code in (400, 404)
    finally:
        server.shutdown()


def test_dashboard_keyframes_follow_failed_and_nested_motion_calls(tmp_path: Path) -> None:
    sys.path.insert(0, str(GPU_SCRIPTS))
    from dashboard_store import Store

    run = tmp_path / "run"
    # move_to_pose #1 raised (no "after"); go_home then called move_to_pose inside it.
    for name in [
        "000001-before-move-to-pose",
        "000002-before-go-home",
        "000003-before-move-to-pose",
        "000004-after-move-to-pose",
        "000005-after-go-home",
    ]:
        (run / "media/keyframes" / name).mkdir(parents=True)
        (run / "media/keyframes" / name / "cam.png").write_bytes(b"png")
    spans = [
        {"index": i, "name": n, "category": "program"}
        for i, n in enumerate(["move_to_pose", "go_home", "move_to_pose"], 1)
    ]
    Store(tmp_path)._pair_keyframes(run, spans)

    def folders(span: dict) -> dict:
        return {k: v[0].split("/")[-2][:6] for k, v in span["keyframes"].items()}

    assert [folders(span) for span in spans] == [
        {"before": "000001"},
        {"before": "000002", "after": "000005"},
        {"before": "000003", "after": "000004"},
    ]


def test_dashboard_reads_arrays_without_numpy_and_builds_the_3d_scene(tmp_path: Path) -> None:
    import numpy as np

    sys.path.insert(0, str(GPU_SCRIPTS))
    from dashboard_store import Store, read_npy

    np.save(tmp_path / "a.npy", np.arange(6, dtype=np.float64).reshape(3, 2))
    np.save(tmp_path / "b.npy", np.array([True, False]))
    assert read_npy(tmp_path / "a.npy") == {
        "shape": [3, 2],
        "dtype": "<f8",
        "values": [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]],
    }
    assert read_npy(tmp_path / "b.npy")["values"] == [True, False]

    run = tmp_path / "run"
    geometry = run / "media/geometry"
    geometry.mkdir(parents=True)
    (geometry / "000001-cloud.json").write_text(
        json.dumps({"kind": "cloud", "call": "call-000004-x", "points_mm": [1, 2, 3]})
    )
    call = run / "trace/calls/call-000005-move-to-pose"
    call.mkdir(parents=True)
    pose = {"$type": "Pose", "frame": "robot_base", "position": {"$array_inline": [0.5, 0, 0.3]}}
    (call / "input.json").write_text(json.dumps({"args": [pose], "kwargs": {}}))
    steps = []
    for k in range(6):
        joints = {"$array_inline": [0.1 * k, 0.0]}
        hand = {"position": {"$array_inline": [0.4, 0.0, 0.1 * k]}}
        steps.append(
            {
                "simulator_time_s": 0.05 * k,
                "robot_state": {
                    "joint_positions": {"primary": joints},
                    "end_effector_poses": {"primary": hand},
                },
                "action": {"arms": {"primary": {"target": {"$array_inline": [0.5, 0.0]}}}},
            }
        )
    _write_jsonl(run / "episode/steps.jsonl", steps)

    store = Store(tmp_path)
    scene = store.scene(run)
    assert [item["kind"] for item in scene["items"]] == ["cloud"]
    assert scene["targets"] == [
        {
            "call": "call-000005-move-to-pose",
            "position": [0.5, 0.0, 0.3],
            "quaternion_wxyz": None,
            "frame": "robot_base",
        }
    ]
    assert len(scene["ee_path"]) == 3  # every other step
    window = store.steps(run, 0.1, 0.2)
    assert window["t"] == [0.1, 0.15, 0.2]
    assert window["arms"]["primary"]["measured"][0] == [0.2, 0.0]
    assert window["arms"]["primary"]["commanded"][0] == [0.5, 0.0]


def test_agent_story_links_reasoning_code_changes_and_batch_results(tmp_path: Path) -> None:
    sys.path.insert(0, str(GPU_SCRIPTS))
    from dashboard_store import Store

    workspace = tmp_path / "agents/llm-x"
    run = workspace / "dev_v01/libero-pro/s/00-t/0051/run-a"
    run.mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"suite": "s", "task_id": 0, "seed": 51}))
    (run / "outcome.json").write_text(json.dumps({"task_success": False, "program_ok": True}))
    v1 = "soup = localize_object('soup')\nresult = 1\n"
    v2 = "soup = localize_object('blue soup can')\nresult = 1\n"
    ws = "/somewhere/else/agents/llm-x"

    def use(tool_id: str, name: str, data: dict) -> dict:
        return {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": tool_id, "name": name, "input": data}],
            },
        }

    def said(kind: str, text: str) -> dict:
        key = "thinking" if kind == "thinking" else "text"
        return {
            "type": "assistant",
            "message": {"role": "assistant", "content": [{"type": kind, key: text}]},
        }

    def done(tool_id: str, text: str) -> dict:
        block = {"type": "tool_result", "tool_use_id": tool_id, "content": text}
        return {"type": "user", "message": {"role": "user", "content": [block]}}

    command = (
        f"cap-harness run-batch --seeds 51 --program {ws}/program_v01.py --output-root {ws}/dev_v01"
    )
    _write_jsonl(
        workspace / "transcript.jsonl",
        [
            said("thinking", "start with the plain name"),
            use("t1", "Write", {"file_path": f"{ws}/program_v01.py", "content": v1}),
            done("t1", "written"),
            use("t2", "Bash", {"command": command}),
            done("t2", "0/1 succeeded"),
            said("text", "SAM3 picked the wrong can; describe it by colour"),
            use("t3", "Write", {"file_path": f"{ws}/program_v02.py", "content": v2}),
            done("t3", "written"),
        ],
    )
    _write_jsonl(
        workspace / "transcript.jsonl",
        [json.loads(line) for line in (workspace / "transcript.jsonl").read_text().splitlines()]
        + [
            said("text", "the colour prompt is a reusable lesson"),
            use("t4", "Edit", {"file_path": f"{ws}/skills/localize.md",
                               "old_string": "| Object |", "new_string": "| Object |\n| soup | blue can |"}),
            done("t4", "edited"),
        ],
    )  # fmt: skip
    (workspace / "skills.diff").write_text("+| soup | blue can |\n")
    agent = Store(tmp_path).agent(workspace / "transcript.jsonl")
    story = agent["story"]
    assert [c["kind"] for c in story] == ["code", "run", "code", "skill"]
    assert story[3]["file"] == "localize.md" and "+| soup | blue can |" in story[3]["diff"]
    assert agent["texts"]["skills.diff"].startswith("+| soup")
    assert story[0]["reasoning"] == [{"kind": "thinking", "text": "start with the plain name"}]
    assert story[0]["code"] == v1 and story[0]["diff"] is None
    assert story[1]["seeds"] == ["51"] and story[1]["output_tail"] == "0/1 succeeded"
    assert story[1]["results"]["total"] == 1 and story[1]["results"]["passes"] == 0
    assert story[1]["results"]["runs"][0]["seed"] == 51
    assert "+soup = localize_object('blue soup can')" in story[2]["diff"]
    assert story[2]["reasoning"][0]["text"].startswith("SAM3 picked the wrong can")


def test_failures_group_unsolved_runs_by_reason_with_numbers_blanked() -> None:
    sys.path.insert(0, str(GPU_SCRIPTS))
    from dashboard_store import _column_name, _failures

    def episode(seed: int, ok: bool | None, program_ok: bool = True, error: str | None = None):
        return {"path": f"s/run-{seed}", "seed": seed, "suite": "s", "task_id": 0, "agent": None,
                "phase": "dev_v01", "finalized": ok is not None, "task_success": ok,
                "program_ok": program_ok, "termination": "program_completed", "error": error}  # fmt: skip

    groups = _failures(
        [
            episode(1, True),
            episode(2, False, False, "timeout: joint error 0.28 rad at /tmp/x/run-2"),
            episode(3, False, False, "timeout: joint error 0.19 rad at /tmp/y/run-3"),
            episode(4, False),
            episode(5, None),
        ]
    )
    assert [(g["count"], g["reason"]) for g in groups] == [
        (2, "program failed: timeout: joint error N rad at <path>"),
        (1, "program finished, task not done (program_completed)"),
        (1, "not finished (stopped or crashed before the outcome was written)"),
    ]
    assert [e["seed"] for e in groups[0]["episodes"]] == [2, 3]
    assert _column_name("sessions/x/agents/llm-a/dev_v03/libero-pro/s/00-t") == "llm-a · dev_v03"
    assert _column_name("sessions/20261109-tasks") == "20261109-tasks"


def test_run_phase_reads_quick_agent_and_campaign_layouts() -> None:
    sys.path.insert(0, str(GPU_SCRIPTS))
    from dashboard_store import run_phase

    task = ("libero-pro", "libero_goal_swap", "task_3")
    cases = {
        ("agents", "llm-x", "dev_v02", "libero-pro"): "dev_v02",
        ("agents", "llm-x", "heldout", "libero-pro"): "heldout",
        (*task, "debug", "initial", "run-a"): "dev_v00",
        (*task, "debug", "attempts", "v04", "run-a"): "dev_v04",
        (*task, "debug", "holdout", "run-a"): "unseen",
        (*task, "validation", "runs", "run-a"): "heldout",
        (*task, "rerun_02", "debug", "attempts", "v01", "run-a"): "rerun_02 dev_v01",
        ("verification", "libero-pro__libero_goal_swap__task_3", "run-a"): "verifier",
        ("episodes", "libero-pro", "libero_90", "33-x", "0001", "run-a"): "episode",
    }
    for parts, phase in cases.items():
        assert run_phase(parts) == phase, parts


def test_library_script_saves_empties_and_restores_a_library(tmp_path: Path) -> None:
    import shutil

    skills = tmp_path / "skills"
    shutil.copytree(REPO / ".claude/skills/iterative-debugging/skills", skills)
    original = {p.name: p.read_text() for p in skills.glob("*.md")}
    env = {
        **os.environ,
        "ASPIRE_SKILLS_DIR": str(skills),
        "ASPIRE_LIBRARY_STORE": str(tmp_path / "store"),
    }

    def library(*args: str) -> str:
        return subprocess.run(
            ["bash", str(GPU_SCRIPTS / "library.sh"), *args],
            env=env, check=True, capture_output=True, text=True,
        ).stdout  # fmt: skip

    library("save", "base")
    library("empty")
    # Learned entries are gone; headers, the README and the exploration procedure stay.
    assert "## " not in (skills / "grasp.md").read_text()
    assert (skills / "grasp.md").read_text().startswith(original["grasp.md"][:40])
    assert (skills / "task-exploration.md").read_text() == original["task-exploration.md"]
    assert "base" in library("list") and "backup-" in library("list")
    library("load", "base")
    assert {p.name: p.read_text() for p in skills.glob("*.md")} == original


def test_campaign_results_reports_held_out_success_per_task(tmp_path: Path) -> None:
    campaign = tmp_path / "2026-11-10_goal"
    for task_id, wins in ((0, 3), (5, 0)):
        for seed in range(1, 6):
            run = campaign / f"libero-pro/libero_goal_swap/task_{task_id}/validation/runs/r{seed}"
            run.mkdir(parents=True)
            (run / "outcome.json").write_text(json.dumps({"task_success": seed <= wins}))
    (campaign / "session.json").write_text(json.dumps({"name": "goal", "memory": "off"}))
    (campaign / "result.part1.json").write_text(json.dumps({"total_cost_usd": 10.0}))
    (campaign / "result.json").write_text(json.dumps({"total_cost_usd": 2.5}))
    out = subprocess.run(
        [sys.executable, str(GPU_SCRIPTS / "campaign_results.py"), str(campaign)],
        check=True, capture_output=True, text=True,
    ).stdout  # fmt: skip
    assert "3/5 (60%)" in out and "0/5 (0%)" in out
    assert "30.0% (2 tasks)" in out  # mean over tasks
    assert "cost $12.50" in out and "2 part(s)" in out
    assert "55.6%" in out  # the paper's empty-library Opus 4.6 mean for this suite
