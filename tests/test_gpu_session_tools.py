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
