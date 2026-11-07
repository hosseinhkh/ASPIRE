from __future__ import annotations

import contextlib

import numpy as np
import pytest

from cap_harness.contracts import Segmentation, SegmentationSet
from cap_harness.errors import ApiError, ErrorCode
from cap_harness.registry import ToolRegistry
from cap_harness.runtime import ProgramExecutor


def _registry() -> tuple[ToolRegistry, list[str]]:
    registry = ToolRegistry()
    calls: list[str] = []

    def get_task_context() -> dict[str, object]:
        calls.append("task")
        return {"language": "put cup away", "reward": 9.0, "success": True}

    def get_task_metadata() -> dict[str, object]:
        calls.append("metadata")
        return {"suite": "suite", "task_id": 1}

    def internal_secret() -> str:
        calls.append("secret")
        return "secret"

    registry.register(
        get_task_context,
        name="get_task_context",
        layer="shared",
        capability="task",
        public=True,
    )
    registry.register(
        get_task_metadata,
        name="libero.get_task_metadata",
        layer="extension",
        capability="metadata",
        public=True,
    )
    registry.register(
        internal_secret,
        name="internal_secret",
        layer="runtime",
        capability="private",
        public=False,
    )
    return registry, calls


def test_runtime_builds_hierarchical_allowlisted_namespace_and_trace() -> None:
    registry, calls = _registry()
    executor = ProgramExecutor(registry)

    result = executor.execute_program(
        "task = get_task_context()\nmeta = libero.get_task_metadata()\nresult = (task, meta)\n"
    )

    assert result.ok
    assert calls == ["task", "metadata"]
    assert [entry.tool for entry in result.calls] == [
        "get_task_context",
        "libero.get_task_metadata",
    ]
    task, metadata = result.result
    assert "reward" not in task
    assert "success" not in task
    assert metadata["suite"] == "suite"


def test_runtime_does_not_expose_private_tools_or_adapter_names() -> None:
    registry, calls = _registry()
    executor = ProgramExecutor(registry)

    for code in (
        "result = internal_secret()",
        "result = adapter",
        "result = reward",
        "result = success",
        "result = mujoco",
    ):
        result = executor.execute_program(code)
        assert not result.ok
    assert "secret" not in calls


def test_runtime_rejects_import_and_callable_introspection() -> None:
    registry, _ = _registry()
    executor = ProgramExecutor(registry)

    imported = executor.execute_program("import os\nresult = os.environ")
    introspected = executor.execute_program("result = get_task_context.__closure__")

    assert not imported.ok
    assert "Import" in imported.error.message
    assert not introspected.ok
    assert "private" in introspected.error.message


def test_runtime_allows_a_lambda_but_still_guards_its_body() -> None:
    """A lambda is a ``def`` with less syntax, and ``def`` was always allowed.

    Rejecting it bought no safety and forced sort keys into index-tuple shapes.
    The guards that do the work are on names and attributes, and those recurse
    into the lambda body -- which is what the second half pins.
    """
    registry, _ = _registry()
    executor = ProgramExecutor(registry)

    allowed = executor.execute_program(
        "result = sorted([(2, 'b'), (1, 'a')], key=lambda row: row[0])[0][1]"
    )
    guarded = executor.execute_program("result = (lambda: get_task_context.__closure__)()")

    assert allowed.ok, allowed.error
    assert allowed.result == "a"
    assert not guarded.ok
    assert "private" in guarded.error.message


def test_execute_program_does_not_implicitly_step_robot() -> None:
    registry = ToolRegistry()
    steps: list[int] = []

    def step(action: object) -> object:
        steps.append(1)
        return action

    registry.register(
        step,
        name="step",
        layer="shared",
        capability="control",
        public=True,
    )
    result = ProgramExecutor(registry).execute_program("result = 42")

    assert result.ok
    assert result.result == 42
    assert steps == []


def test_keyboard_interrupt_propagates_out_of_generated_tool_call() -> None:
    registry = ToolRegistry()

    def step(action: object) -> object:
        del action
        raise KeyboardInterrupt

    registry.register(
        step,
        name="step",
        layer="shared",
        capability="control",
        public=True,
    )

    with pytest.raises(KeyboardInterrupt):
        ProgramExecutor(registry).execute_program("step(None)")


def test_runtime_exposes_safe_contract_constructors() -> None:
    executor = ProgramExecutor(ToolRegistry())

    result = executor.execute_program(
        "pose = Pose(position=[0, 0, 0], quaternion_wxyz=[1, 0, 0, 0], frame='base')\n"
        "command = ArmCommand(mode='joint_position', target=[0, 0, 0, 0, 0, 0, 0])\n"
        "strategy = MotionStrategy(ik_solver='pyroki', trajectory_planner='curobo')\n"
        "result = (RobotAction(arms={'primary': command}), strategy)\n"
    )

    assert result.ok
    action, strategy = result.result
    assert tuple(action.arms) == ("primary",)
    assert strategy.ik_solver == "pyroki"
    assert strategy.trajectory_planner == "curobo"


def test_trace_summarizes_arrays_provider_frame_and_failure_code() -> None:
    registry = ToolRegistry()

    def segment_points(camera_name: str, points: np.ndarray) -> SegmentationSet:
        del camera_name, points
        return SegmentationSet(
            ok=True,
            segmentations=(
                Segmentation(
                    mask=np.ones((2, 2), dtype=bool),
                    label="object",
                    score=1.0,
                    camera_name="agentview",
                    frame="camera/agentview",
                ),
            ),
            diagnostics={"provider": "sam3"},
        )

    def generate_grasps(camera_name: str, mask: object) -> object:
        del camera_name, mask
        from cap_harness.contracts import GraspSet

        return GraspSet(
            ok=False,
            error=ApiError(code=ErrorCode.NO_GRASP, message="none"),
            diagnostics={"provider": "contact_graspnet"},
        )

    registry.register(
        segment_points,
        name="segment_points",
        layer="shared",
        capability="perception",
        public=True,
    )
    registry.register(
        generate_grasps,
        name="generate_grasps",
        layer="shared",
        capability="grasping",
        public=True,
    )
    executor = ProgramExecutor(registry)

    successful = executor.execute_program("result = segment_points('agentview', [[10, 20]])")
    assert successful.ok
    trace = successful.calls[0]
    assert trace.provider == "sam3"
    assert trace.frame == "camera/agentview"
    assert trace.outputs_summary["count"] == 1
    assert "mask" not in repr(trace.outputs_summary).lower()

    failed = executor.execute_program("result = generate_grasps('agentview', [[1]])")
    assert failed.ok
    trace = failed.calls[0]
    assert trace.ok is False
    assert trace.provider == "contact_graspnet"
    assert trace.failure_code == ErrorCode.NO_GRASP.value
    assert trace.inputs_summary["positional"][1]["type"] == "list"


# ---------------------------------------------------------------------------
# Sandbox escapes that the private-name rule alone did not stop.
# ---------------------------------------------------------------------------


class _HostSecret:
    def __init__(self) -> None:
        self.object_xyz = [0.123, -0.456, 0.789]


class _HostApi:
    """Stands in for CapApi: a bound-method tool whose owner holds privileged state."""

    def __init__(self) -> None:
        self._adapter = _HostSecret()

    def get_task_context(self) -> dict[str, object]:
        return {"language": "demo"}


def _host_registry(api: _HostApi) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        api.get_task_context,
        name="get_task_context",
        layer="shared",
        capability="task",
        public=True,
    )
    return registry


@pytest.mark.parametrize(
    "code",
    [
        "result = '{0.__closure__}'.format(get_task_context)",
        "result = str.format('{0.__closure__}', get_task_context)",
        "result = '{f.__closure__}'.format_map({'f': get_task_context})",
        "template = '{0.__closure__}'\nfmt = template.format\nresult = fmt(get_task_context)",
        "result = '{0:{1.__class__}}'.format(1, get_task_context)",
    ],
)
def test_str_format_cannot_traverse_private_attributes(code: str) -> None:
    result = ProgramExecutor(_host_registry(_HostApi())).execute_program(code)

    assert not result.ok
    assert "format fields" in result.error.message


def test_str_format_still_formats_ordinary_values() -> None:
    code = (
        "pose = Pose((0.1, 0.2, 0.3), (1.0, 0.0, 0.0, 0.0), 'robot_base')\n"
        "result = (\n"
        "    '{:.2f}'.format(3.14159),\n"
        "    '{} {}'.format('a', 2),\n"
        "    '{0[k]}'.format({'k': 7}),\n"
        "    '{x}'.format_map({'x': 'y'}),\n"
        "    '{0.frame}'.format(pose),\n"
        "    str.format('{}-{}', 1, 2),\n"
        ")\n"
    )
    result = ProgramExecutor(ToolRegistry()).execute_program(code)

    assert result.ok, result.error
    assert result.result == ("3.14", "a 2", "7", "y", "robot_base", "1-2")


def test_generator_frames_cannot_reach_host_locals() -> None:
    """``gi_frame.f_back.f_locals`` used to hand a program the host's live adapter."""
    adapter = _HostSecret()  # a host local, like ``adapter`` in run.run_program
    code = (
        "def walk(frame):\n"
        "    while frame is not None:\n"
        "        if 'adapter' in frame.f_locals:\n"
        "            return frame.f_locals['adapter']\n"
        "        frame = frame.f_back\n"
        "frames = (walk(gen.gi_frame) for i in range(1))\n"
        "gen = frames\n"
        "for found in frames:\n"
        "    found.object_xyz[0] = 99.0\n"
    )

    result = ProgramExecutor(ToolRegistry()).execute_program(code)

    assert not result.ok
    assert "gi_frame" in result.error.message or "f_locals" in result.error.message
    assert adapter.object_xyz[0] == 0.123


@pytest.mark.parametrize(
    "attribute", ["gi_frame", "f_back", "f_locals", "f_globals", "tb_frame", "tofile", "ctypes"]
)
def test_introspection_and_numpy_io_attributes_are_rejected(attribute: str) -> None:
    result = ProgramExecutor(ToolRegistry()).execute_program(
        f"value = 1\nresult = value.{attribute}"
    )

    assert not result.ok
    assert attribute in result.error.message


def test_bare_except_still_handles_ordinary_errors() -> None:
    code = "try:\n    raise ValueError('x')\nexcept:\n    result = 'handled'\n"

    result = ProgramExecutor(ToolRegistry()).execute_program(code)

    assert result.ok
    assert result.result == "handled"


# ---------------------------------------------------------------------------
# Step budget: a program must stop once the run cannot advance any more.
# ---------------------------------------------------------------------------


class _BudgetRecorder:
    """The slice of RunRecorder that ProgramExecutor touches."""

    def __init__(self, max_steps: int) -> None:
        self.max_steps = max_steps
        self.step_count = 0
        self.terminal_reason: str | None = None

    def keyframe(self, name: str) -> None:
        del name

    @contextlib.contextmanager
    def span(self, name: str, *, category: str, inputs: object):
        del name, category, inputs

        class _Span:
            @staticmethod
            def output(result: object, *, ok: bool) -> object:
                del ok
                return result

        yield _Span()


def _budget_registry(recorder: _BudgetRecorder) -> tuple[ToolRegistry, list[int]]:
    attempts: list[int] = []
    registry = ToolRegistry()

    def step(action: object) -> dict[str, object]:
        del action
        attempts.append(1)
        # What CapApi.step does with the adapter's StepLimitReached: a failed result.
        if recorder.step_count >= recorder.max_steps:
            recorder.terminal_reason = "step_limit"
            return {"ok": False}
        recorder.step_count += 1
        return {"ok": True}

    registry.register(step, name="step", layer="shared", capability="control", public=True)
    return registry, attempts


@pytest.mark.parametrize("handler", ["except Exception:\n        pass", "except:\n        pass"])
def test_program_stops_when_step_budget_is_exhausted(handler: str) -> None:
    """A retry loop used to spin forever: StepLimitReached arrived as a failed result."""
    recorder = _BudgetRecorder(max_steps=3)
    registry, attempts = _budget_registry(recorder)
    code = f"while True:\n    try:\n        step(None)\n    {handler}\n"

    result = ProgramExecutor(registry, recorder=recorder).execute_program(code)

    assert not result.ok
    assert result.error.code is ErrorCode.TRUNCATED
    assert "max_steps=3" in result.error.message
    assert len(attempts) == 4  # three steps, then the refused fourth


def test_program_that_finishes_inside_the_budget_is_unaffected() -> None:
    recorder = _BudgetRecorder(max_steps=3)
    registry, _ = _budget_registry(recorder)

    result = ProgramExecutor(registry, recorder=recorder).execute_program(
        "for i in range(3):\n    step(None)\nresult = 'done'\n"
    )

    assert result.ok
    assert result.result == "done"
