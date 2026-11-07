"""Restricted generated-program execution over a ToolRegistry allowlist."""

from __future__ import annotations

import _string  # CPython's field-name parser; the same one string.Formatter uses
import ast
from collections.abc import Callable, Mapping
import contextlib
from dataclasses import dataclass
import importlib
import io
import string
import time
from types import MappingProxyType

import numpy as np

from cap_harness.contracts import (
    ArmCommand,
    ExecutionResult,
    GraspSet,
    LocalizationResult,
    MotionStrategy,
    PlanResult,
    PointCloud,
    Pose,
    RobotAction,
    SegmentationSet,
    StepResult,
    SynchronizedPlanResult,
    SynchronizedTrajectory,
    Trajectory,
)
from cap_harness.errors import ApiError, ErrorCode
from cap_harness.registry import ToolRegistry

ALLOWED_IMPORTS = frozenset({"math"})
"""Modules a generated program may import; everything else is injected or forbidden."""

FORBIDDEN_ATTRIBUTES = frozenset(
    {
        # Frame and code introspection. None of these start with an underscore, so the
        # private-name rule alone let a program walk ``(g.gi_frame for i in ...)`` ->
        # ``f_back`` -> ``f_locals`` up into the host's frames and take the live adapter
        # (and through it the native simulator) out of ``run_program``.
        "ag_await",
        "ag_code",
        "ag_frame",
        "cr_await",
        "cr_code",
        "cr_frame",
        "cr_origin",
        "f_back",
        "f_builtins",
        "f_code",
        "f_globals",
        "f_locals",
        "f_trace",
        "gi_code",
        "gi_frame",
        "gi_yieldfrom",
        "tb_frame",
        "tb_next",
        # NumPy escape hatches on the read-only arrays every contract returns:
        # arbitrary file writes and raw memory access.
        "ctypes",
        "dump",
        "dumps",
        "tofile",
    }
)
"""Public-looking attribute names a generated program may not read (beyond ``_*``)."""

_FORMAT_ATTRIBUTES = frozenset({"format", "format_map"})
_FORMAT_HELPER_NAME = "_cap_format_attribute"


def is_forbidden_attribute(name: str) -> bool:
    """Whether generated programs may not access attribute ``name``."""
    return name.startswith("_") or name in FORBIDDEN_ATTRIBUTES


class ProgramAborted(BaseException):
    """Stops a generated program at a tool boundary.

    Derives from ``BaseException`` so a program's own ``except Exception`` cannot
    swallow it (bare ``except:`` is rewritten to ``except Exception:``).
    """


class _SafeFormatter(string.Formatter):
    """``str.format`` with the same attribute policy as the AST validator.

    ``'{0.__closure__[0].cell_contents.__self__._adapter}'.format(tool)`` performs
    attribute traversal at runtime, where the AST validator never sees it.
    """

    def get_field(self, field_name: str, args: object, kwargs: object) -> object:
        _, rest = _string.formatter_field_name_split(field_name)
        for is_attribute, key in rest:
            if is_attribute and is_forbidden_attribute(str(key)):
                raise ValueError(
                    "private, dunder, and introspection attributes are not allowed "
                    "in generated-program format fields"
                )
        return super().get_field(field_name, args, kwargs)  # type: ignore[arg-type]


_SAFE_FORMATTER = _SafeFormatter()


def _format_attribute(value: object, name: str) -> object:
    """Runtime stand-in for ``value.format`` / ``value.format_map`` in programs."""
    if isinstance(value, str):
        if name == "format":
            return lambda *args, **kwargs: _SAFE_FORMATTER.vformat(value, args, kwargs)
        return lambda mapping: _SAFE_FORMATTER.vformat(value, (), mapping)
    if value is str:
        if name == "format":
            return lambda template, *args, **kwargs: _SAFE_FORMATTER.vformat(
                str(template), args, kwargs
            )
        return lambda template, mapping: _SAFE_FORMATTER.vformat(str(template), (), mapping)
    return getattr(value, name)


def _restricted_import(name: str, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
    if level != 0 or name not in ALLOWED_IMPORTS:
        raise ImportError(f"import of {name!r} is not allowed in generated programs")
    return importlib.import_module(name)


_SAFE_BUILTINS = MappingProxyType(
    {
        "__import__": _restricted_import,
        "Exception": Exception,
        "ValueError": ValueError,
        "TypeError": TypeError,
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "float": float,
        "int": int,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "print": print,
        "range": range,
        "reversed": reversed,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
    }
)

_SAFE_CONSTRUCTORS = MappingProxyType(
    {
        "ArmCommand": ArmCommand,
        "MotionStrategy": MotionStrategy,
        "Pose": Pose,
        "RobotAction": RobotAction,
        "SynchronizedTrajectory": SynchronizedTrajectory,
        "Trajectory": Trajectory,
    }
)

_SENSITIVE_KEYS = (
    "ground_truth",
    "groundtruth",
    "mujoco",
    "native_env",
    "object_pose",
    "privileged",
    "qpos",
    "qvel",
    "raw_observation",
    "raw_state",
    "reward",
    "sim_state",
    "success",
)


@dataclass(frozen=True, slots=True)
class CallTrace:
    """One public tool invocation without raw arguments or privileged values."""

    index: int
    tool: str
    ok: bool
    duration_s: float
    positional_count: int
    keyword_names: tuple[str, ...]
    inputs_summary: Mapping[str, object]
    outputs_summary: Mapping[str, object] | None = None
    provider: str | None = None
    frame: str | None = None
    failure_code: str | None = None
    result_type: str | None = None
    error_type: str | None = None
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class ProgramExecutionResult:
    """Result of executing one complete generated Python program."""

    ok: bool
    result: object | None
    stdout: str
    calls: tuple[CallTrace, ...]
    error: ApiError | None = None

    @property
    def trace(self) -> tuple[CallTrace, ...]:
        return self.calls


class _ToolNamespace:
    __slots__ = ("_entries",)

    def __init__(self, entries: Mapping[str, object]) -> None:
        object.__setattr__(self, "_entries", MappingProxyType(dict(entries)))

    def __getattr__(self, name: str) -> object:
        try:
            return self._entries[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("tool namespaces are immutable")

    def __repr__(self) -> str:
        return "<allowlisted tool namespace>"


class _ProgramValidator(ast.NodeVisitor):
    _forbidden_nodes = (
        ast.AsyncFunctionDef,
        ast.Await,
        ast.ClassDef,
        ast.Delete,
        ast.Global,
        ast.ImportFrom,
        ast.Nonlocal,
        ast.With,
    )
    # ``ast.Lambda`` is deliberately absent. ``ast.FunctionDef`` is allowed, so a
    # lambda grants nothing a two-line ``def`` did not already grant, and the
    # name and attribute guards below recurse into its body either way. Banning
    # it only pushed ordinary code -- a sort key, say -- into clumsier shapes.

    def generic_visit(self, node: ast.AST) -> None:
        if isinstance(node, self._forbidden_nodes):
            # Preserve the invalid-data ValueError contract.
            raise ValueError(f"{type(node).__name__} is not allowed in generated programs")
        super().generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        # ``import math`` is the one import a program may write; it grants nothing the
        # sandbox does not already allow and spares programs hand-rolled trigonometry.
        for alias in node.names:
            if alias.name not in ALLOWED_IMPORTS or alias.asname is not None:
                raise ValueError(
                    f"Import of {alias.name!r} is not allowed; only {sorted(ALLOWED_IMPORTS)} "
                    "may be imported, without aliasing"
                )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("_"):
            raise ValueError("private and dunder attribute access is not allowed")
        if node.attr in FORBIDDEN_ATTRIBUTES:
            raise ValueError(
                f"attribute {node.attr!r} is not allowed in generated programs "
                "(frame/code introspection and raw NumPy I/O are forbidden)"
            )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.startswith("_"):
            raise ValueError("private and dunder names are not allowed")
        self.generic_visit(node)


class _ProgramRewriter(ast.NodeTransformer):
    """Applied after validation; may therefore introduce private helper names.

    * ``x.format`` / ``x.format_map`` become ``_cap_format_attribute(x, name)`` so
      format-field attribute traversal obeys the validator's attribute policy.
    * A bare ``except:`` becomes ``except Exception:`` so a program cannot swallow
      :class:`ProgramAborted` (or ``KeyboardInterrupt``).
    """

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        self.generic_visit(node)
        if node.attr in _FORMAT_ATTRIBUTES and isinstance(node.ctx, ast.Load):
            return ast.copy_location(
                ast.Call(
                    func=ast.Name(id=_FORMAT_HELPER_NAME, ctx=ast.Load()),
                    args=[node.value, ast.Constant(node.attr)],
                    keywords=[],
                ),
                node,
            )
        return node

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> ast.AST:
        self.generic_visit(node)
        if node.type is None:
            node.type = ast.copy_location(ast.Name(id="Exception", ctx=ast.Load()), node)
        return node


class ProgramExecutor:
    """Execute code against wrapped public tools, separately from robot ``step``."""

    def __init__(self, registry: ToolRegistry, *, recorder: object | None = None) -> None:
        if not isinstance(registry, ToolRegistry):
            # Preserve the invalid-data ValueError contract.
            raise ValueError("registry must be a ToolRegistry")
        self._registry = registry
        self._recorder = recorder
        self._calls: list[CallTrace] = []

    def public_namespace(self) -> Mapping[str, object]:
        """Build a fresh hierarchical namespace from public registry entries."""
        root: dict[str, object] = {}
        branches: dict[tuple[str, ...], dict[str, object]] = {(): root}
        for spec in self._registry.all_specs(public_only=True):
            parts = tuple(spec.name.split("."))
            parent_path = parts[:-1]
            for depth in range(1, len(parts)):
                path = parts[:depth]
                branches.setdefault(path, {})
            branches[parent_path][parts[-1]] = self._wrapped_tool(spec.name, spec.function)

        for path in sorted(branches, key=len, reverse=True):
            if not path:
                continue
            parent = branches[path[:-1]]
            parent[path[-1]] = _ToolNamespace(branches[path])
        return MappingProxyType(root)

    def execute_program(self, code: str) -> ProgramExecutionResult:
        """Execute one complete program; this never performs an implicit robot step."""
        self._calls = []
        if not isinstance(code, str) or not code.strip():
            return self._failure(ErrorCode.INVALID_REQUEST, "code must be a non-empty string")
        try:
            tree = ast.parse(code, mode="exec")
            _ProgramValidator().visit(tree)
            tree = ast.fix_missing_locations(_ProgramRewriter().visit(tree))
            compiled = compile(tree, "<generated-program>", "exec")
        except (SyntaxError, ValueError) as exc:
            return self._failure(
                ErrorCode.INVALID_REQUEST,
                f"program rejected: {exc}",
            )

        environment: dict[str, object] = {
            "__builtins__": _SAFE_BUILTINS,
            **_SAFE_CONSTRUCTORS,
            **self.public_namespace(),
            _FORMAT_HELPER_NAME: _format_attribute,
        }
        output = io.StringIO()
        try:
            with contextlib.redirect_stdout(output):
                # Execute AST-validated programs; process isolation is still required for untrusted code.
                exec(compiled, environment, environment)
            result = _public_value(environment.get("result"))
        except ProgramAborted as exc:
            return ProgramExecutionResult(
                ok=False,
                result=None,
                stdout=output.getvalue(),
                calls=tuple(self._calls),
                error=ApiError(
                    code=ErrorCode.TRUNCATED,
                    message=f"program stopped: {exc}",
                    details={"reason": "step_limit"},
                ),
            )
        # Report generated-program failures as unsuccessful execution results.
        except Exception as exc:
            return ProgramExecutionResult(
                ok=False,
                result=None,
                stdout=output.getvalue(),
                calls=tuple(self._calls),
                error=ApiError(
                    code=ErrorCode.EXECUTION_FAILED,
                    message=f"program execution failed: {type(exc).__name__}: {exc}",
                    details={"exception_type": type(exc).__name__},
                ),
            )
        return ProgramExecutionResult(
            ok=True,
            result=result,
            stdout=output.getvalue(),
            calls=tuple(self._calls),
        )

    def _wrapped_tool(self, name: str, function: Callable[..., object]) -> Callable[..., object]:
        def invoke(*args: object, **kwargs: object) -> object:
            self._stop_if_step_budget_exhausted(name)
            if self._recorder is not None:
                result = self._recorded_tool_call(name, function, args, kwargs)
            else:
                result = self._invoke_tool(name, function, args, kwargs)
            self._stop_if_step_budget_exhausted(name)
            return result

        return invoke

    def _stop_if_step_budget_exhausted(self, name: str) -> None:
        """End the program once the run's step budget is spent.

        The adapter's step hook raises ``StepLimitReached``, but every API layer
        between it and the program converts exceptions into failed results, so the
        program used to keep running -- a retry loop spun forever against a budget
        that could not grow.
        """
        if getattr(self._recorder, "terminal_reason", None) == "step_limit":
            max_steps = getattr(self._recorder, "max_steps", None)
            raise ProgramAborted(
                f"the run's step budget (max_steps={max_steps}) is exhausted; "
                f"{name}() and every later tool call cannot advance the episode"
            )

    def _recorded_tool_call(
        self,
        name: str,
        function: Callable[..., object],
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
    ) -> object:
        motion = name in {
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
        if motion:
            self._recorder.keyframe(f"before-{name}")  # type: ignore[attr-defined]
        with self._recorder.span(  # type: ignore[attr-defined]
            name,
            category="program",
            inputs={"args": args, "kwargs": kwargs},
        ) as span:
            result = self._invoke_tool(name, function, args, kwargs)
            ok = getattr(result, "ok", True)
            span.output(result, ok=ok if type(ok) is bool else True)
        if motion:
            self._recorder.keyframe(f"after-{name}")  # type: ignore[attr-defined]
        return result

    def _invoke_tool(
        self,
        name: str,
        function: Callable[..., object],
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
    ) -> object:
        started = time.perf_counter()
        index = len(self._calls)
        inputs_summary = MappingProxyType(
            {
                "positional": tuple(_value_summary(item) for item in args),
                "keywords": MappingProxyType(
                    {key: _value_summary(value) for key, value in sorted(kwargs.items())}
                ),
            }
        )
        try:
            result = function(*args, **kwargs)
        except Exception as exc:
            error = getattr(exc, "error", None)
            self._calls.append(
                CallTrace(
                    index=index,
                    tool=name,
                    ok=False,
                    duration_s=time.perf_counter() - started,
                    positional_count=len(args),
                    keyword_names=tuple(sorted(kwargs)),
                    inputs_summary=inputs_summary,
                    failure_code=(error.code.value if isinstance(error, ApiError) else None),
                    error_type=type(exc).__name__,
                    error_message=str(exc)[:256],
                )
            )
            raise
        public_result = _public_value(result)
        provider = _result_provider(public_result)
        frame = _result_frame(public_result)
        failure_code = _result_failure_code(public_result)
        call_ok = getattr(public_result, "ok", True)
        if type(call_ok) is not bool:
            call_ok = True
        self._calls.append(
            CallTrace(
                index=index,
                tool=name,
                ok=call_ok,
                duration_s=time.perf_counter() - started,
                positional_count=len(args),
                keyword_names=tuple(sorted(kwargs)),
                inputs_summary=inputs_summary,
                outputs_summary=_value_summary(public_result),
                provider=provider,
                frame=frame,
                failure_code=failure_code,
                result_type=type(public_result).__name__,
            )
        )
        return public_result

    def _failure(self, code: ErrorCode, message: str) -> ProgramExecutionResult:
        return ProgramExecutionResult(
            ok=False,
            result=None,
            stdout="",
            calls=tuple(self._calls),
            error=ApiError(code=code, message=message),
        )


def _public_value(value: object) -> object:
    if isinstance(value, StepResult):
        return value.public_view()
    if isinstance(value, ExecutionResult):
        return value.public_view()
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                key: _public_value(item)
                for key, item in value.items()
                if isinstance(key, str) and not any(term in key.lower() for term in _SENSITIVE_KEYS)
            }
        )
    if isinstance(value, list):
        return tuple(_public_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_public_value(item) for item in value)
    return value


def _value_summary(value: object) -> Mapping[str, object]:
    """Summarize values without retaining image, mask, depth, or joint contents."""
    summary: dict[str, object] = {"type": type(value).__name__}
    if value is None or isinstance(value, bool | int | float):
        summary["value"] = value
    elif isinstance(value, str):
        summary["length"] = len(value)
    elif isinstance(value, np.ndarray):
        summary.update(shape=tuple(value.shape), dtype=str(value.dtype))
    elif isinstance(value, Pose):
        summary["frame"] = value.frame
    elif isinstance(value, MotionStrategy):
        summary.update(
            ik_solver=value.ik_solver,
            trajectory_planner=value.trajectory_planner,
            pose_planner=value.pose_planner,
        )
    elif isinstance(value, ArmCommand):
        summary.update(mode=value.mode, has_gripper=value.gripper_position is not None)
    elif isinstance(value, RobotAction):
        summary["arms"] = tuple(sorted(value.arms))
    elif isinstance(value, Trajectory):
        summary.update(
            arm=value.arm,
            waypoint_count=len(value.joint_positions),
            planner=value.planner,
            collision_aware=value.collision_aware,
        )
    elif isinstance(value, SynchronizedTrajectory):
        summary.update(
            arms=tuple(sorted(value.joint_positions)),
            waypoint_count=value.waypoint_count,
            planner=value.planner,
            collision_aware=value.collision_aware,
        )
    elif isinstance(value, PointCloud):
        summary.update(frame=value.frame, point_count=len(value.points))
    elif isinstance(value, SegmentationSet):
        summary.update(ok=value.ok, count=len(value.segmentations))
        if value.segmentations:
            summary["frame"] = value.segmentations[0].frame
    elif isinstance(value, GraspSet):
        summary.update(ok=value.ok, count=len(value.grasps))
        if value.grasps:
            summary["frame"] = value.grasps[0].frame
    elif isinstance(value, LocalizationResult):
        summary["ok"] = value.ok
        if value.geometry is not None:
            summary["frame"] = value.geometry.frame
    elif isinstance(value, PlanResult):
        summary["ok"] = value.ok
        if value.trajectory is not None:
            summary["waypoint_count"] = len(value.trajectory.joint_positions)
    elif isinstance(value, SynchronizedPlanResult):
        summary["ok"] = value.ok
        if value.trajectory is not None:
            summary.update(
                arms=tuple(sorted(value.trajectory.joint_positions)),
                waypoint_count=value.trajectory.waypoint_count,
            )
    elif isinstance(value, StepResult):
        summary.update(ok=value.ok, terminated=value.terminated, truncated=value.truncated)
    elif isinstance(value, ExecutionResult):
        summary.update(
            ok=value.ok,
            steps_executed=value.steps_executed,
            terminated=value.terminated,
            truncated=value.truncated,
            status=value.status.value,
            final_errors=dict(value.final_errors),
        )
    elif isinstance(value, Mapping):
        summary["keys"] = tuple(
            sorted(
                key
                for key in value
                if isinstance(key, str) and not any(term in key.lower() for term in _SENSITIVE_KEYS)
            )
        )
    elif isinstance(value, list | tuple | set | frozenset):
        summary["length"] = len(value)
        summary["item_types"] = tuple(sorted({type(item).__name__ for item in value}))
    else:
        frame = getattr(value, "frame", None)
        if isinstance(frame, str):
            summary["frame"] = frame
        ok = getattr(value, "ok", None)
        if type(ok) is bool:
            summary["ok"] = ok
    return MappingProxyType(summary)


def _result_provider(value: object) -> str | None:
    diagnostics = getattr(value, "diagnostics", None)
    if isinstance(diagnostics, Mapping):
        provider = diagnostics.get("provider")
        if isinstance(provider, str):
            return provider
    return None


def _result_frame(value: object) -> str | None:
    frame = getattr(value, "frame", None)
    if isinstance(frame, str):
        return frame
    if isinstance(value, SegmentationSet) and value.segmentations:
        return value.segmentations[0].frame
    if isinstance(value, GraspSet) and value.grasps:
        return value.grasps[0].frame
    if isinstance(value, LocalizationResult) and value.geometry is not None:
        return value.geometry.frame
    if isinstance(value, PointCloud):
        return value.frame
    return None


def _result_failure_code(value: object) -> str | None:
    error = getattr(value, "error", None)
    return error.code.value if isinstance(error, ApiError) else None


ProgramResult = ProgramExecutionResult

__all__ = [
    "FORBIDDEN_ATTRIBUTES",
    "CallTrace",
    "ProgramAborted",
    "ProgramExecutionResult",
    "ProgramExecutor",
    "ProgramResult",
    "is_forbidden_attribute",
]
