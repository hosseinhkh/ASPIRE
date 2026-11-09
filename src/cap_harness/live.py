"""Live console: keep one robot (simulated or real) running and drive it over HTTP.

  python -m cap_harness.live --benchmark robosuite --suite cube_lifting --task-id 0 --seed 1
      [--host 127.0.0.1] [--port 8770] [--output-root outputs/live]
      [--camera-width 512] [--camera-height 512] [--max-steps 100000]
      [--init-mode saved|seeded]                      (libero-pro)
      [--station NAME] [--station-config-root DIR] [--allow-motion]     (yam_real)

The robot is the same adapter + CapApi + ToolRegistry that `cap-harness run` builds, wrapped in
the LiveAgentSession that serves the real-robot System 2 agent: calls run one at a time as
background jobs, and every call, program, frame and model-service request is recorded like an
episode (trace/calls, videos, keyframes, overlays) in a run directory under --output-root. The
run is finalized (outcome.json) when the console closes: POST /api/close, Ctrl-C or SIGTERM.

HTTP API (JSON unless noted; scripts/gpu/dashboard.py gives it a web UI):
  GET  /api/info                   robot, task, cameras, arms, tools with JSON schemas
  GET  /api/state?after=SEQ        phase, current job, events, latest robot state, step count
  GET  /api/camera/NAME.jpg        latest frame of a camera (image/jpeg), updated every step
  GET  /api/history                every tool call / program so far, with its result
  GET  /api/job/ID                 one job's status and result
  POST /api/tool    {"name": ..., "arguments": {...}}   start a registered tool
  POST /api/program {"code": "...", "label": "..."}     start a Code-as-Policy program
  POST /api/close                  finalize the recording and stop the console

Physical robots: --benchmark yam_real builds the station exactly like `cap-harness run`; it
cannot move unless --allow-motion is given (the adapter refuses motion otherwise).
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import inspect
import io
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import threading
import time
from typing import Any, get_type_hints
from urllib.parse import parse_qs, urlparse

import numpy as np

from .agent.mcp import (
    _cap_input_schema,
    _jsonable,
    _motion_strategy_input_schema,
    _pose_input_schema,
)
from .agent.session import _READ_ONLY_TOOL_NAMES, LiveAgentSession, SessionBusyError
from .artifacts import RunRecorder, TracedProvider
from .contracts import MotionStrategy, Pose
from .geometry import pose_from_mapping
from .registry import ToolRegistry

BENCHMARKS = ("robosuite", "libero-pro", "yam_real")
# The tools the ProgramExecutor brackets with before/after keyframes.
MOTION_TOOLS = {
    "step",
    "execute_trajectory",
    "set_gripper",
    "set_grippers",
    "move_to_joints",
    "move_to_pose",
    "move_synchronized",
    "open_gripper",
    "close_gripper",
    "go_home",
}


class LiveRecorder(RunRecorder):
    """A RunRecorder that also keeps the latest frames and robot state for the live view."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._live_lock = threading.Lock()
        self.live_frames: dict[str, np.ndarray] = {}
        self.live_state: dict[str, Any] = {}
        self.live_steps = 0

    def _remember(self, observation: Any) -> None:
        cameras = getattr(observation, "cameras", None) or {}
        frames = {
            str(name): np.asarray(camera.rgb, dtype=np.uint8).copy()
            for name, camera in cameras.items()
            if getattr(camera, "rgb", None) is not None
        }
        state = getattr(observation, "robot_state", None)
        with self._live_lock:
            if frames:
                self.live_frames.update(frames)
            if state is not None:
                self.live_state = _robot_state_view(state)

    def on_reset(self, observation: Any, metadata: Mapping[str, object]) -> None:
        super().on_reset(observation, metadata)
        self._remember(observation)

    def after_step(self, action: Any, result: Any) -> None:
        super().after_step(action, result)
        with self._live_lock:
            self.live_steps += 1
        if getattr(result, "observation", None) is not None:
            self._remember(result.observation)

    def frame(self, camera: str) -> np.ndarray | None:
        with self._live_lock:
            frame = self.live_frames.get(camera)
            return None if frame is None else frame.copy()

    def snapshot(self) -> dict[str, Any]:
        with self._live_lock:
            return {
                "cameras": sorted(self.live_frames),
                "robot_state": self.live_state,
                "steps": self.live_steps,
            }


def _robot_state_view(state: Any) -> dict[str, Any]:
    view: dict[str, Any] = {}
    for key in ("joint_positions", "gripper_positions"):
        values = getattr(state, key, None)
        if isinstance(values, Mapping):
            view[key] = {
                arm: np.round(np.asarray(value, dtype=float), 6).tolist()
                for arm, value in values.items()
            }
    poses = getattr(state, "end_effector_poses", None)
    if isinstance(poses, Mapping):
        view["end_effector_poses"] = {
            arm: {
                "position": np.round(np.asarray(pose.position, dtype=float), 6).tolist(),
                "quaternion_wxyz": np.round(
                    np.asarray(pose.quaternion_wxyz, dtype=float), 4
                ).tolist(),
                "frame": str(pose.frame),
            }
            for arm, pose in poses.items()
        }
    for key in ("base_frame", "embodiment"):
        if getattr(state, key, None) is not None:
            view[key] = str(getattr(state, key))
    return view


# ----------------------------------------------------------------------------- building --
def build_robot(args: argparse.Namespace) -> dict[str, Any]:
    """Build adapter, providers, CapApi and tools the way cap_harness.run does, minus a program."""
    from .providers.curobo.client import DEFAULT_CUROBO_URL, CuRoboProvider
    from .providers.graspnet.client import ContactGraspNetProvider
    from .providers.pyroki.client import PyRokiProvider
    from .providers.sam3.client import DEFAULT_SAM3_URL, Sam3Provider

    task_registry: Any
    if args.benchmark == "libero-pro":
        from .libero.registry import LiberoSuiteRegistry

        task_registry = LiberoSuiteRegistry()
        allowlist = None
    elif args.benchmark == "robosuite":
        from .registry import ROBOSUITE_PUBLIC_TOOL_NAMES
        from .robosuite.registry import RobosuiteTaskRegistry

        task_registry = RobosuiteTaskRegistry()
        allowlist = ROBOSUITE_PUBLIC_TOOL_NAMES
    elif args.benchmark == "yam_real":
        from .registry import YAM_REAL_PUBLIC_TOOL_NAMES
        from .yam_real.registry import YamRealTaskRegistry

        task_registry = YamRealTaskRegistry()
        allowlist = YAM_REAL_PUBLIC_TOOL_NAMES
    else:
        raise ValueError(f"--benchmark must be one of {', '.join(BENCHMARKS)}")
    metadata = task_registry.resolve(args.suite, args.task_id)

    ik_names = ("mink", "curobo") if args.benchmark == "yam_real" else ("pyroki", "curobo")
    recorder = LiveRecorder(
        output_root=Path(args.output_root),
        benchmark=args.benchmark,
        suite=metadata.suite_name,
        task_id=metadata.task_id,
        task_name=metadata.task_name,
        seed=args.seed,
        max_steps=args.max_steps,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        init_mode=args.init_mode,
        capture_videos=not args.no_videos,
        providers={
            "segmentation": "sam3",
            "grasping": ("contact-graspnet",),
            "ik": ik_names,
            "trajectory_planning": ("interpolation", "curobo"),
            "pose_planning": ("composed", "curobo-integrated"),
        },
        step_video=args.benchmark != "yam_real",
    )
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as handle:
        handle.write(
            "# Live console session: the commands sent to this robot are in\n"
            "# logs/live_commands.jsonl; each one's calls are in trace/.\n"
            "result = None\n"
        )
    recorder.save_program(Path(handle.name))
    os.unlink(handle.name)

    if args.benchmark == "yam_real":
        from .yam_real.adapter import YamRealAdapter
        from .yam_real.station import build_real_station

        adapter = YamRealAdapter(
            build_real_station(args.station, config_root=args.station_config_root),
            registry=task_registry,
            run_observer=recorder,
            allow_physical_motion=args.allow_motion,
        )
        recorder.frequency = adapter.control_frequency
    else:
        kwargs: dict[str, Any] = {
            "registry": task_registry,
            "run_observer": recorder,
            "camera_width": args.camera_width,
            "camera_height": args.camera_height,
            "horizon": args.max_steps,
        }
        if args.benchmark == "libero-pro":
            from .libero.adapter import LiberoAdapter

            kwargs["init_mode"] = args.init_mode
            adapter = LiberoAdapter(**kwargs)
        else:
            from .robosuite.adapter import RobosuiteAdapter

            adapter = RobosuiteAdapter(**kwargs)
    adapter.reset(metadata, args.seed)
    rate = getattr(adapter, "control_frequency", None)
    if rate:
        recorder.frequency = float(rate)

    segmentation = TracedProvider(
        Sam3Provider(base_url=os.environ.get("CAP_HARNESS_SAM3_URL", DEFAULT_SAM3_URL)),
        recorder,
        "sam3",
    )
    grasping = TracedProvider(ContactGraspNetProvider(), recorder, "contact_graspnet")
    curobo = TracedProvider(
        CuRoboProvider(
            base_url=os.environ.get("CAP_HARNESS_CUROBO_URL", DEFAULT_CUROBO_URL),
            **(
                {"control_period_s": adapter.control_period_s}
                if args.benchmark == "yam_real"
                else {}
            ),
        ),
        recorder,
        "curobo",
    )
    if args.benchmark == "yam_real":
        from .providers.yam_kinematics import YamKinematicsIKProvider
        from .yam_real.kinematics import YamKinematics

        ik = {
            "mink": TracedProvider(
                YamKinematicsIKProvider(YamKinematics(adapter.native_env.config.model_xml)),
                recorder,
                "mink",
            ),
            "curobo": curobo,
        }
        default_camera = adapter.native_env.config.camera.role
    else:
        ik = {"pyroki": TracedProvider(PyRokiProvider(), recorder, "pyroki"), "curobo": curobo}
        default_camera = "agentview"

    from .api import CapApi

    api = CapApi(
        adapter,
        segmentation_provider=segmentation,
        grasp_providers={"contact-graspnet": grasping},
        ik_providers=ik,
        trajectory_planning_providers={"curobo": curobo},
        integrated_pose_planning_providers={"curobo-integrated": curobo},
        default_camera=default_camera,
        recorder=recorder,
    )
    registry = api.register_tools(
        ToolRegistry(public_extension_allowlist=allowlist) if allowlist else ToolRegistry()
    )
    return {
        "adapter": adapter,
        "api": api,
        "registry": registry,
        "recorder": recorder,
        "metadata": metadata,
    }


# ------------------------------------------------------------------------------ console --
def _tool_schema(spec: Any) -> tuple[dict[str, Any], bool]:
    """The tool's JSON schema, and whether every required argument can be given as JSON."""
    schema = _cap_input_schema(spec)
    properties = schema["properties"]
    try:
        hints = get_type_hints(spec.function)
    except Exception:
        hints = {}
    for name, parameter in spec.signature.parameters.items():
        annotation = hints.get(name, parameter.annotation)
        text = str(annotation)
        is_pose = annotation is Pose or ("Pose" in text and "Mapping" not in text)
        if is_pose and name in properties:
            properties[name] = _pose_input_schema()
        if annotation is MotionStrategy or "MotionStrategy" in text:
            properties[name] = {"anyOf": [_motion_strategy_input_schema(), {"type": "null"}]}
    if "targets" in properties:
        properties["targets"] = {
            "type": "object",
            "additionalProperties": {
                "oneOf": [_pose_input_schema(), {"type": "array", "items": {"type": "number"}}]
            },
        }
    callable_as_json = all(properties.get(name) for name in schema.get("required", []))
    return schema, callable_as_json


def _pose(value: Mapping[str, Any]) -> Pose:
    """pose_from_mapping, accepting a hand-typed or rounded quaternion (normalized here)."""
    value = dict(value)
    quaternion = value.get("quaternion_wxyz")
    if quaternion is not None:
        q = np.asarray(quaternion, dtype=float)
        norm = float(np.linalg.norm(q))
        if norm < 1e-6:
            raise ValueError("quaternion_wxyz must not be zero")
        value["quaternion_wxyz"] = (q / norm).tolist()
    return pose_from_mapping(value)


def _convert_arguments(spec: Any, arguments: Mapping[str, Any]) -> dict[str, Any]:
    try:
        hints = get_type_hints(spec.function)
    except Exception:
        hints = {}
    converted: dict[str, Any] = {}
    for name, value in arguments.items():
        parameter = spec.signature.parameters.get(name)
        annotation = hints.get(name, parameter.annotation if parameter else inspect.Parameter.empty)
        text = str(annotation)
        if value is None:
            converted[name] = None
        elif name == "targets" and isinstance(value, Mapping):
            converted[name] = {
                arm: _pose(target)
                if isinstance(target, Mapping)
                else np.asarray(target, dtype=float)
                for arm, target in value.items()
            }
        elif isinstance(value, Mapping) and ("Pose" in text or name.endswith("pose")):
            converted[name] = _pose(value)
        elif isinstance(value, Mapping) and "MotionStrategy" in text:
            converted[name] = MotionStrategy(**value)
        elif isinstance(value, list) and ("ndarray" in text or name in {"target", "joints"}):
            converted[name] = np.asarray(value, dtype=float)
        else:
            converted[name] = value
    return converted


class RobotThreadSession(LiveAgentSession):
    """A LiveAgentSession whose jobs run on the one thread that owns the robot.

    MuJoCo's offscreen GL context belongs to the thread that created the simulator: stepping
    or rendering from any other thread returns garbage depth (negative / non-finite) and the
    adapter fails. Jobs still get their own monitoring thread; it only waits for the robot
    thread to run the operation.
    """

    def __init__(self, *args: Any, on_robot_thread: Callable[..., Any], **kwargs: Any) -> None:
        self._on_robot_thread = on_robot_thread
        self._console_recorder = kwargs.get("recorder")
        super().__init__(*args, **kwargs)

    def _start_job(self, *, operation: Any, **kwargs: Any) -> str:
        return super()._start_job(operation=lambda: self._on_robot_thread(operation), **kwargs)

    def start_tool(self, name: str, arguments: Mapping[str, Any] | None = None) -> str:
        """Like LiveAgentSession.start_tool, but traced like a program call.

        A program's calls are recorded by the ProgramExecutor (span, inputs, outputs, and
        before/after keyframes for motion); a direct console call would otherwise leave no
        trace/calls entry of its own.
        """
        spec = self.registry.get_spec(str(name))
        if not spec.public:
            raise ValueError(f"tool {name!r} is not public")
        kwargs = dict(arguments or {})
        recorder = self._console_recorder
        readonly = name in _READ_ONLY_TOOL_NAMES or spec.capability == "metadata"

        def operation() -> Any:
            if recorder is None:
                return spec.function(**kwargs)
            if name in MOTION_TOOLS:
                recorder.keyframe(f"before-{name}")
            with recorder.span(
                name, category="program", inputs={"args": (), "kwargs": kwargs}
            ) as span:
                result = spec.function(**kwargs)
                ok = getattr(result, "ok", True)
                span.output(result, ok=ok if type(ok) is bool else True)
            if name in MOTION_TOOLS:
                recorder.keyframe(f"after-{name}")
            return result

        return self._start_job(
            kind="tool",
            label=name,
            phase="perceive" if readonly else "act",
            operation=operation,
            controls_runtime=not readonly,
        )

    def terminal_robot_state(self) -> Any:
        try:
            return self._on_robot_thread(self.api.get_robot_state)
        except Exception:
            return None


class LiveConsole:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self._robot_thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="robot")
        built = self.on_robot_thread(build_robot, args)
        self.adapter = built["adapter"]
        self.api = built["api"]
        self.registry = built["registry"]
        self.recorder: LiveRecorder = built["recorder"]
        self.metadata = built["metadata"]
        self.session = RobotThreadSession(
            self.api,
            self.registry,
            recorder=self.recorder,
            on_robot_thread=self.on_robot_thread,
        )
        self.history: list[dict[str, Any]] = []
        self._history_lock = threading.Lock()
        self._closed = False
        self.started = time.time()
        self.commands_path = self.recorder.root / "logs/live_commands.jsonl"
        self.commands_path.parent.mkdir(parents=True, exist_ok=True)
        self.tools = {}
        for spec in self.registry.all_specs(public_only=True):
            schema, as_json = _tool_schema(spec)
            self.tools[spec.name] = {
                "name": spec.name,
                "signature": spec.documented_signature,
                "capability": spec.capability,
                "layer": spec.layer,
                "doc": (spec.docstring or "").strip().split("\n\n")[0],
                "schema": schema,
                "json_callable": as_json,
                "spec": spec,
            }

    def on_robot_thread(self, function: Callable[..., Any], *args: Any) -> Any:
        """Run on the robot's own thread (simulator and GL context live there) and wait."""
        return self._robot_thread.submit(function, *args).result()

    # -- information --
    def info(self) -> dict[str, Any]:
        snapshot = self.recorder.snapshot()
        return {
            "benchmark": self.args.benchmark,
            "suite": self.metadata.suite_name,
            "task_id": self.metadata.task_id,
            "task_name": self.metadata.task_name,
            "language": getattr(self.metadata, "language", None),
            "seed": self.args.seed,
            "embodiment": getattr(self.adapter, "embodiment", None),
            "allow_motion": bool(getattr(self.args, "allow_motion", False)),
            "cameras": snapshot["cameras"],
            "default_camera": self.api.default_camera,
            "configured_providers": _jsonable(self.api.configured_providers),
            "run_dir": str(self.recorder.root),
            "started": self.started,
            "tools": [
                {key: value for key, value in tool.items() if key != "spec"}
                for tool in self.tools.values()
            ],
        }

    def state(self, after: int) -> dict[str, Any]:
        state = self.session.state(after=after)
        state.pop("tools", None)
        snapshot = self.recorder.snapshot()
        state.update(robot_state=snapshot["robot_state"], steps=snapshot["steps"],
                     closed=self._closed)  # fmt: skip
        return _jsonable(state)

    def history_view(self) -> list[dict[str, Any]]:
        with self._history_lock:
            return [dict(item) for item in self.history]

    # -- commands --
    def _track(self, job_id: str, request: dict[str, Any]) -> None:
        entry = {"job_id": job_id, "started": time.time(), "status": "running", **request}
        with self._history_lock:
            self.history.append(entry)

        def wait() -> None:
            job = self.session.wait(job_id, timeout_s=None, detail="full") or {}
            entry.update(
                status="error" if job.get("error") else "done",
                finished=time.time(),
                error=job.get("error"),
                result=_jsonable(job.get("result")),
                elapsed_s=job.get("elapsed_s"),
                steps_after=self.recorder.snapshot()["steps"],
            )
            with self._history_lock, self.commands_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, default=str) + "\n")

        threading.Thread(target=wait, name=f"live-wait-{job_id}", daemon=True).start()

    def start_tool(self, name: str, arguments: Mapping[str, Any]) -> str:
        tool = self.tools.get(name)
        if tool is None:
            raise KeyError(f"unknown tool {name!r}")
        spec = tool["spec"]
        kwargs = _convert_arguments(spec, arguments)
        job_id = self.session.start_tool(name, kwargs)
        self._track(job_id, {"kind": "tool", "name": name, "arguments": _jsonable(arguments)})
        return job_id

    def start_program(self, code: str, label: str) -> str:
        job_id = self.session.start_program(code, label=label or "CaP program")
        self._track(job_id, {"kind": "program", "name": label or "CaP program", "code": code})
        return job_id

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.session.close()
        except Exception:
            pass
        try:
            self.on_robot_thread(self._finalize)
        finally:
            self._robot_thread.shutdown(wait=True)

    def _finalize(self) -> None:
        try:
            success = self.adapter.check_success()
        except Exception:
            success = None
        finish = getattr(self.adapter, "finish_episode_recording", None)
        if callable(finish):
            try:
                finish()
            except Exception:
                pass
        try:
            self.recorder.finalize(
                program_ok=True,
                task_success=success,
                # Closing the console ends the session's "program"; the reason set is fixed.
                termination_reason="task_succeeded" if success else "program_completed",
                program_result={"commands": len(self.history)},
            )
        finally:
            try:
                self.adapter.close()
            except Exception:
                pass


def _jpeg(frame: np.ndarray, quality: int = 85) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def make_handler(console: LiveConsole, stop: threading.Event) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "cap-live/1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            if os.environ.get("CAP_LIVE_ACCESS_LOG"):
                super().log_message(format, *args)

        def _send(self, status: int, payload: Any, content_type: str = "application/json") -> None:
            body = (
                payload
                if isinstance(payload, bytes)
                else json.dumps(payload, default=str).encode("utf-8")
            )
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            data = json.loads(raw or b"{}")
            if not isinstance(data, dict):
                raise ValueError("request body must be a JSON object")
            return data

        def do_GET(self) -> None:
            url = urlparse(self.path)
            query = parse_qs(url.query)
            try:
                if url.path == "/api/info":
                    return self._send(200, console.info())
                if url.path == "/api/state":
                    return self._send(200, console.state(int(query.get("after", ["0"])[0])))
                if url.path == "/api/history":
                    return self._send(200, {"history": console.history_view()})
                if url.path.startswith("/api/job/"):
                    job = console.session.job(url.path.rsplit("/", 1)[-1], detail="full")
                    if job is None:
                        for entry in console.history_view():
                            if entry["job_id"] == url.path.rsplit("/", 1)[-1]:
                                return self._send(200, entry)
                        return self._send(404, {"error": "unknown job"})
                    return self._send(200, _jsonable(job))
                if url.path.startswith("/api/camera/") and url.path.endswith(".jpg"):
                    name = url.path[len("/api/camera/") : -len(".jpg")]
                    frame = console.recorder.frame(name)
                    if frame is None:
                        return self._send(404, {"error": f"no frame for camera {name!r}"})
                    return self._send(200, _jpeg(frame), "image/jpeg")
                return self._send(404, {"error": "not found"})
            except Exception as exc:
                return self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

        def do_POST(self) -> None:
            url = urlparse(self.path)
            try:
                body = self._body()
                if url.path == "/api/tool":
                    job_id = console.start_tool(str(body["name"]), body.get("arguments") or {})
                    return self._send(202, {"job_id": job_id})
                if url.path == "/api/program":
                    job_id = console.start_program(str(body["code"]), str(body.get("label", "")))
                    return self._send(202, {"job_id": job_id})
                if url.path == "/api/close":
                    self._send(200, {"ok": True, "run_dir": str(console.recorder.root)})
                    stop.set()
                    return None
                return self._send(404, {"error": "not found"})
            except SessionBusyError as exc:
                return self._send(409, {"error": str(exc)})
            except (KeyError, ValueError, TypeError) as exc:
                return self._send(400, {"error": f"{type(exc).__name__}: {exc}"})
            except Exception as exc:
                return self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--benchmark", choices=BENCHMARKS, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--output-root", default="outputs/live")
    parser.add_argument("--camera-width", type=int, default=512)
    parser.add_argument("--camera-height", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--init-mode", choices=("saved", "seeded"), default="saved")
    parser.add_argument("--no-videos", action="store_true")
    parser.add_argument("--station", default="yam-example")
    parser.add_argument("--station-config-root", type=Path)
    parser.add_argument("--allow-motion", action="store_true", help="yam_real: enable motion")
    args = parser.parse_args(argv)

    console = LiveConsole(args)
    stop = threading.Event()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(console, stop))
    server.daemon_threads = True
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    thread = threading.Thread(target=server.serve_forever, name="live-http", daemon=True)
    thread.start()
    print(
        f"live console: {args.benchmark} {console.metadata.suite_name}:{console.metadata.task_id} "
        f"seed {args.seed} on http://{args.host}:{args.port}  (run dir {console.recorder.root})",
        flush=True,
    )
    stop.wait()
    server.shutdown()
    console.close()
    print(f"live console closed; recording finalized in {console.recorder.root}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
