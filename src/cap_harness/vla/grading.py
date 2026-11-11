"""Offline grading of an episode from simulator state: what was grasped, lifted and placed.

The benchmark's own success flag says whether the task was solved. It does not say why not.
For a pick-and-place task the two useful questions are the proposal's two labels:

    semantic_correct   did the robot go for the requested object?
    physical_success   did the object it went for end up where the task wanted an object?

    failure_family     success | semantic | physical | both | no_attempt | unknown
                       semantic:   wrong object, placed correctly
                       physical:   right object, not placed
                       both:       wrong object, not placed
                       no_attempt: the gripper never closed near an object

This reads the simulator (object positions, the task's goal predicate), which is privileged.
It is for grading only: nothing here is returned to a policy or written where the teacher agent
reads. It supports goals of the form (In|On <object> <region>); other tasks get family
``success`` or ``unknown``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

LIFT_M = 0.05  # an object is "lifted" once it is this far above where it started
NEAR_M = 0.08  # the hand "went for" the nearest object within this distance when it closed


def sim_state(adapter: Any) -> np.ndarray:
    """The full MuJoCo state (privileged): enough to restore this exact moment."""
    env = adapter.native_env
    sim = getattr(env, "sim", None) or env.env.sim
    return np.asarray(sim.get_state().flatten(), dtype=np.float64)


class LiberoGrader:
    """Follows one episode. Call start() after reset, step() after every action, finish() last."""

    def __init__(self, adapter: Any) -> None:
        self.adapter = adapter
        self.supported = False
        self.target: str | None = None
        self.region: str | None = None
        self.predicate: str | None = None
        self.names: list[str] = []
        self._start: dict[str, np.ndarray] = {}
        self._lifted: dict[str, int] = {}
        self._approached: list[dict[str, Any]] = []
        self._closed = False
        self._step = 0

    # ---------------------------------------------------------------------- internals --
    def _inner(self) -> Any:
        env = self.adapter.native_env
        return getattr(env, "env", env)

    def _sim(self) -> Any:
        env = self.adapter.native_env
        return getattr(env, "sim", None) or env.env.sim

    def _position(self, name: str) -> np.ndarray:
        inner = self._inner()
        return np.array(self._sim().data.body_xpos[inner.obj_body_id[name]], dtype=np.float64)

    # -------------------------------------------------------------------------- hooks --
    def start(self) -> None:
        self.supported = False
        self._lifted, self._approached, self._closed, self._step = {}, [], False, 0
        try:
            inner = self._inner()
            goal = inner.parsed_problem["goal_state"]
            objects = list(inner.objects_dict)
            if len(goal) == 1 and len(goal[0]) == 3 and goal[0][1] in objects:
                self.predicate, self.target, self.region = (str(part) for part in goal[0])
                # The container (or support) the goal names is not a candidate to be picked.
                self.names = [n for n in objects if not self.region.startswith(n)]
                self._start = {name: self._position(name) for name in self.names}
                self.supported = self.target in self.names
        except (AttributeError, KeyError, TypeError, IndexError):
            self.supported = False

    def step(self, raw: Mapping[str, Any], gripper_closed_command: bool) -> None:
        """After an action: ``raw`` is the new observation, the flag is the command just sent."""
        self._step += 1
        if not self.supported:
            return
        for name in self.names:
            if name not in self._lifted and self._position(name)[2] - self._start[name][2] > LIFT_M:
                self._lifted[name] = self._step
        if gripper_closed_command and not self._closed:
            hand = np.asarray(raw["robot0_eef_pos"], dtype=np.float64)
            distances = {
                name: float(np.linalg.norm(self._position(name) - hand)) for name in self.names
            }
            nearest = min(distances, key=distances.get)
            self._approached.append(
                {
                    "step": self._step,
                    "object": nearest if distances[nearest] <= NEAR_M else None,
                    "distance_m": round(distances[nearest], 4),
                }
            )
        self._closed = bool(gripper_closed_command)

    def _satisfies(self, inner: Any, name: str) -> bool:
        """Whether ``name`` is where the goal wants the target (the goal with ``name`` in it)."""
        try:
            return bool(inner._eval_predicate([self.predicate, name, self.region]))
        except (AttributeError, KeyError, TypeError, ValueError, IndexError):
            return False  # a predicate that cannot be evaluated for this object: not placed

    def finish(self, success: bool) -> dict[str, Any]:
        if not self.supported:
            return {"supported": False, "failure_family": "success" if success else "unknown"}
        inner = self._inner()
        placed = [name for name in self.names if self._satisfies(inner, name)]
        lifted = sorted(self._lifted, key=self._lifted.get)
        approached = [a["object"] for a in self._approached if a["object"]]
        # The object the robot went for: the first one it lifted, else the first it closed on.
        chosen = lifted[0] if lifted else (approached[0] if approached else None)
        semantic = None if chosen is None else chosen == self.target
        physical = None if chosen is None else chosen in placed
        if success:
            family = "success"
        elif chosen is None:
            family = "no_attempt"
        elif semantic and not physical:
            family = "physical"
        elif not semantic and physical:
            family = "semantic"
        else:
            family = "both"
        return {
            "supported": True,
            "target": self.target,
            "region": self.region,
            "chosen": chosen,
            "lifted": lifted,
            "placed": placed,
            "approached": self._approached[:8],
            "target_lifted": self.target in self._lifted,
            "target_placed": self.target in placed,
            "semantic_correct": semantic,
            "physical_success": physical,
            "failure_family": family,
            "moved_m": {
                name: round(
                    float(np.linalg.norm(self._position(name)[:2] - self._start[name][:2])), 3
                )
                for name in self.names
                if np.linalg.norm(self._position(name)[:2] - self._start[name][:2]) > 0.02
            },
        }
