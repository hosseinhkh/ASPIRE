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
