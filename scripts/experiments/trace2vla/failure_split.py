"""How does the untrained student fail on libero_object_swap? Offline grading with simulator state
(never shown to the policy): which object, if any, ends in the basket, and which object the hand
closed on. One episode per task, scene 1, policy seed 0, the benchmark protocol.
"""

import json
import sys

import numpy as np

from cap_harness.libero.adapter import LiberoAdapter
from cap_harness.vla.benchmark import policy_seed_for
from cap_harness.vla.libero_compat import ensure_libero_alias
from cap_harness.vla.policy import LeRobotLiberoPolicy, seed_everything
from cap_harness.vla.protocol import (
    CONTROLLER,
    N_ACTION_STEPS,
    SETTLE_STEPS,
    max_steps_for,
    task_language,
)

ensure_libero_alias()
suite = "libero_object_swap"
tasks = [int(t) for t in sys.argv[1].split(",")]
policy = LeRobotLiberoPolicy(device="cpu", n_action_steps=N_ACTION_STEPS, suite=suite)
adapter = LiberoAdapter(
    controller=CONTROLLER,
    camera_height=256,
    camera_width=256,
    init_mode="saved",
    post_settle_steps=SETTLE_STEPS,
)
rows = []
for task in tasks:
    adapter.reset(f"{suite}:{task}", seed=1)
    inner = adapter.native_env.env
    sim = adapter.native_env.sim
    goal = inner.parsed_problem["goal_state"]
    target = goal[0][1]
    names = [n for n in inner.objects_dict if not n.startswith("basket")]

    def pos(n, sim=sim, inner=inner):
        return np.array(sim.data.body_xpos[inner.obj_body_id[n]])

    start = {n: pos(n) for n in names}
    raw = adapter.raw_observation
    policy.reset()
    seed_everything(policy_seed_for(0, task, 1))
    instruction = task_language(suite, task)
    lifted = {}
    success = False
    for step in range(max_steps_for(suite)):
        raw = adapter.step_native_action(policy.act(raw, instruction))
        for n in names:
            if pos(n)[2] - start[n][2] > 0.05:
                lifted.setdefault(n, step)
        if adapter.check_success():
            success = True
            break
    in_basket = []
    for n in names:
        try:
            if inner._eval_predicate(["in", n, goal[0][2]]):
                in_basket.append(n)
        except Exception as exc:
            in_basket.append(f"{n}:{type(exc).__name__}")
    moved = {n: round(float(np.linalg.norm(pos(n)[:2] - start[n][:2])), 3) for n in names}
    rows.append(
        {
            "task": task,
            "instruction": instruction,
            "target": target,
            "success": success,
            "lifted": list(lifted),
            "in_basket": in_basket,
            "moved_m": {n: d for n, d in moved.items() if d > 0.02},
        }
    )
    print("ROW" + json.dumps(rows[-1]), flush=True)
adapter.close()
