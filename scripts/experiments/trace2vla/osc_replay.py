"""Prototype (scratch, not product code): re-execute a recorded ASPIRE teacher episode in OSC_POSE,
the student's action space, following the teacher's hand path at a chosen speed-up.

    osc_replay.py RUN_DIR SPEED [SPEED ...]

Reads only what the recorder already stores (episode/steps.jsonl, reset.json, run.json).
Reports, per speed: task success, steps used, tracking error, command saturation.
"""

import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from cap_harness.libero.adapter import LiberoAdapter
from cap_harness.vla.libero_compat import ensure_libero_alias

POS_MAX, ROT_MAX = 0.05, 0.5  # OSC_POSE output_max per action unit
LOOKAHEAD = 3.0  # teacher steps ahead of the playhead to aim at
ERR_HOLD = 0.02  # m: do not advance the playhead while this far behind
IDLE_SPEED = 2e-4  # m/step: teacher hand "standing still" (gripper moving)
SLOW_WINDOW = 12  # teacher steps around a gripper change run at 1x
TAIL_STEPS = 40  # extra steps holding the last pose


def arr(v):
    return np.asarray(v["$array_inline"] if isinstance(v, dict) else v, dtype=np.float64)


def load(run: Path):
    meta = json.loads((run / "run.json").read_text())
    reset = json.loads((run / "episode/reset.json").read_text())
    poses, grips = [], []

    def pose_of(state):
        ee = state["end_effector_poses"]["primary"]
        return arr(ee["position"]), arr(ee["quaternion_wxyz"])

    poses.append(pose_of(reset["observation"]["robot_state"]))
    for line in (run / "episode/steps.jsonl").read_text().splitlines():
        step = json.loads(line)
        if not step.get("ok") or not step.get("robot_state"):
            continue
        poses.append(pose_of(step["robot_state"]))
        grips.append(float(step["action"]["arms"]["primary"]["gripper_position"]))
    return meta, poses, np.asarray(grips)


def replay(run: Path, speed: float, image: int = 64):
    meta, poses, grips = load(run)
    n = len(grips)
    adapter = LiberoAdapter(
        controller="OSC_POSE",
        camera_height=image,
        camera_width=image,
        init_mode=meta["init_mode"],
        post_settle_steps=10,
    )
    adapter.reset(f"{meta['suite']}:{meta['task_id']}", seed=int(meta["seed"]))
    world_from_base = np.linalg.inv(adapter._base_from_world())
    pos = np.array([world_from_base[:3, :3] @ p + world_from_base[:3, 3] for p, _ in poses])
    rot_base = R.from_quat([[q[1], q[2], q[3], q[0]] for _, q in poses])
    rot = R.from_matrix(world_from_base[:3, :3]) * rot_base
    slerp = Slerp(np.arange(n + 1), rot)
    step_len = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    change = np.flatnonzero(np.diff(grips) != 0) + 1
    slow = np.zeros(n + 1, dtype=bool)
    for c in change:
        slow[max(c - SLOW_WINDOW, 0) : c + SLOW_WINDOW] = True
    idle = np.concatenate([[False], step_len < IDLE_SPEED])

    def target(i):
        j = min(i + LOOKAHEAD, n)
        k = int(math.floor(j))
        f = j - k
        p = pos[k] if k >= n else (1 - f) * pos[k] + f * pos[k + 1]
        return p, slerp([j])[0]

    raw = adapter.raw_observation
    start_gap = float(np.linalg.norm(np.asarray(raw["robot0_eef_pos"]) - pos[0]))
    i, steps, tail, success = 0.0, 0, 0, False
    errs, sat, actions = [], 0, []
    while steps < 1500:
        p_now = np.asarray(raw["robot0_eef_pos"], dtype=np.float64)
        r_now = R.from_quat(np.asarray(raw["robot0_eef_quat"], dtype=np.float64))
        k = min(int(round(i)), n)
        p_t, r_t = target(i)
        a = np.zeros(7)
        a[:3] = np.clip((p_t - p_now) / POS_MAX, -1, 1)
        a[3:6] = np.clip((r_t * r_now.inv()).as_rotvec() / ROT_MAX, -1, 1)
        a[6] = -1.0 if grips[min(k, n - 1)] > 0.5 else 1.0
        sat += int(np.any(np.abs(a[:6]) >= 0.999))
        actions.append(a)
        raw = adapter.step_native_action(a)
        steps += 1
        if adapter.check_success():
            success = True
            break
        err = float(np.linalg.norm(np.asarray(raw["robot0_eef_pos"]) - pos[k]))
        errs.append(err)
        if i >= n:
            tail += 1
            if tail > TAIL_STEPS:
                break
        elif err < ERR_HOLD:
            rate = 1.0 if (slow[k] or idle[k]) else speed
            i = min(i + rate, float(n))
    adapter.close()
    actions = np.asarray(actions)
    return {
        "speed": speed,
        "success": success,
        "steps": steps,
        "teacher_steps": n,
        "start_gap_mm": round(1000 * start_gap, 1),
        "track_err_mean_mm": round(1000 * float(np.mean(errs)), 1) if errs else None,
        "track_err_max_mm": round(1000 * float(np.max(errs)), 1) if errs else None,
        "saturated_steps": round(sat / steps, 3),
        "action_abs_mean": round(float(np.abs(actions[:, :3]).mean()), 3),
        "gripper_changes": int(len(change)),
    }


if __name__ == "__main__":
    ensure_libero_alias()
    run = Path(sys.argv[1])
    for speed in [float(s) for s in sys.argv[2:]]:
        print(
            "RES"
            + json.dumps({"run": run.parts[-2] + "/" + run.parts[-1][-8:], **replay(run, speed)}),
            flush=True,
        )
