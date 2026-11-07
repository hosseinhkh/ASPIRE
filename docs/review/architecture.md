# How ASPIRE is built

ASPIRE has three layers. The coding agents and their prompts form the *learning loop*. The
`cap_harness` Python package is the *robot execution engine*. Below that sit the simulators and
the real robot.

```
 ┌──────────────────────────── Learning loop (prompts + small scripts) ─────────────────────────────┐
 │ .claude/skills/iterative-debugging/   main-agent-prompt.md   = coordinator (curriculum, dispatch, │
 │                                       subagent-prompt.md     = actor (debug one task)    promote) │
 │                                       verifier-prompt.md     = verifier (reproduce a proposed skill)│
 │                                       skills/*.md            = THE SKILL LIBRARY (markdown)       │
 │                                       scripts/*.py           = campaign init, dispatch, validation│
 │ .claude/skills/evosearch/             K=8 candidates/round, top-3 conditioning (prompt-driven)    │
 │ .codex/skills/                        thin pointers to the same files                             │
 └───────────────▲──────────────────────────────────────────────────────────────┬────────────────────┘
                 │ reads run artifacts (trace/, media/, outcome.json)            │ writes program.py,
                 │                                                               │ calls `cap-harness run`
 ┌───────────────┴──────────────────── cap_harness (src/cap_harness) ───────────▼────────────────────┐
 │ cli.py ─► run.py::run_program ─► RunRecorder (artifacts.py) ── per-call trace + keyframes + video │
 │              │                                                                                    │
 │              ├─► Adapter.reset(task, seed)          libero/ robosuite/ behavior/ yam_real/        │
 │              ├─► Providers (HTTP clients)           providers/{sam3,graspnet,pyroki,curobo}      │
 │              ├─► CapApi(adapter, providers)         api.py  ─ 24 shared tools + `<bench>.*` ext.  │
 │              ├─► ToolRegistry (allowlist)           registry.py                                   │
 │              └─► ProgramExecutor.execute_program    runtime.py ─ AST-validated exec() sandbox     │
 │                    └─ generated program.py calls tools; `result = ...` is returned                │
 │              ◄── adapter.check_success()  (privileged, never exposed to the program)             │
 └───────────────┬────────────────────────────────────────────────────────────────────────────────────┘
                 │
 ┌───────────────▼────────────── Simulators / robot ──────────────────────────────────────────────────┐
 │ LIBERO-Pro (MuJoCo, Panda, 20 Hz)   Robosuite (MuJoCo, 1–2 Panda, 20 Hz)                          │
 │ BEHAVIOR-1K (Isaac Sim/OmniGibson, R1 Pro holonomic base + torso + 2×7-DoF, 30 Hz)                │
 │ Real YAM (2× 6-DoF I2RT arms, DaMiao CAN motors, RealSense) — also a MuJoCo sim stand-in          │
 └────────────────────────────────────────────────────────────────────────────────────────────────────┘

 Model services (separate venvs, HTTP):  SAM3 :8114 · Contact-GraspNet :8115 · PyRoki :8116 · cuRobo :8118
 System 2 (real robot): Codex/Claude ─MCP─► agent/server.py :8222 ─► LiveAgentSession ─► CapApi ─► YAM
```

## 1. The life of one run (`cap-harness run`)

1. **`cli.py` → `run.run_program()`** (`run.py:66`). It picks the task registry and adapter by
   benchmark name, using hard-coded if/elif dispatch.
2. **`RunRecorder`** (`artifacts.py`) opens a run directory:
   - `source/program.py` and `provenance.json`;
   - `trace/calls/call-NNNNNN-<tool>/{input,output}.json` for every tool **and** nested API stage
     **and** provider call;
   - `episode/steps.jsonl`;
   - `media/keyframes` (before and after every motion call) and `media/videos`;
   - `outcome.json` with timing buckets.

   This is the paper's "per-primitive multimodal trace". See `docs/run-artifacts.md`.
3. **`adapter.reset(metadata, seed)`**. For LIBERO the seed either selects a saved init state
   (`saved`) or seeds procedural placement (`seeded`; LIBERO-Pro position perturbation).
   Robosuite and BEHAVIOR seed their own samplers.
4. **Providers are wrapped in `TracedProvider`** so their calls appear as child spans.
5. **`CapApi.register_tools()`** registers 24 shared tools plus the embodiment extensions the
   adapter happens to implement. `ToolRegistry` drops anything not on the allowlist and anything
   whose name contains a privileged term (`qpos`, `reward`, `success`, …).
6. **`ProgramExecutor.execute_program(source)`:**
   - parses the source and rejects imports other than `math`, classes, `with`, `global`,
     `_`-names, and (after this branch) frame-introspection attributes;
   - rewrites `.format` and bare `except:`;
   - `exec`s the program with a tiny builtins table and the tool namespace;
   - records a `CallTrace` per tool call;
   - post-filters every returned mapping to strip privileged keys.
7. **`adapter.check_success()`** gives task success. For bimanual Robosuite and BEHAVIOR, a
   *protocol evaluator* records extra witness checks.
8. **`run-batch`** (`batch.py`) runs many seeds in spawned worker processes with
   `PR_SET_PDEATHSIG`, so MuJoCo, EGL or CUDA state is never forked.

## 2. Contracts (`contracts.py`)

All data crossing a boundary is a frozen, validated dataclass with read-only arrays:

| Contract | What it carries |
| --- | --- |
| `Pose` | `position` (m), `quaternion_wxyz` (unit), and an explicit `frame` string. Frames are never implicit. |
| `RobotState` | Per-arm maps (`"primary"`, `"secondary"`, or `left`/`right` on YAM) of joints, velocities, end-effector `Pose`, and normalized gripper (**0 = closed, 1 = open**), plus `base_frame` and `embodiment`. Arms must be 6- or 7-DoF (`JOINT_DIMENSIONS`). **There is no base pose.** |
| `CameraObservation` | RGB, metric depth, intrinsics, and `camera_pose` in the robot base frame. |
| `Trajectory` / `SynchronizedTrajectory` | Timed joint waypoints, `expected_start: RobotState` (used to reject stale plans, `api.py:1828`), and `collision_aware`. |
| Results | `IKResult`, `PlanResult`, `ExecutionResult`, `SegmentationSet`, `GraspSet`, `LocalizationResult`. Each has `ok` plus a typed `ApiError(code, message, details)`. Failures are values, not exceptions. That is what makes traces debuggable by an LLM. |

## 3. The robot API (`api.py`, `docs/api-reference.md`)

**Atomic tools**

| Purpose | Tools |
| --- | --- |
| Read state | `get_task_context`, `get_observation`, `get_robot_state` |
| Perception | `segment_text`, `segment_points` |
| Geometry | `mask_to_point_cloud`, `crop_point_cloud`, `estimate_geometry` |
| Grasping and kinematics | `generate_grasps`, `solve_ik` |
| Planning | `plan_motion`, `plan_synchronized_motion` |
| Control | `step`, `execute_trajectory`, `set_gripper(s)` |

**High-level tools** (compositions of the atomic tools)

| Tool | What it does |
| --- | --- |
| `localize_object` | segment → project → geometry |
| `select_grasp` | |
| `move_to_joints` | servo loop |
| `move_to_pose` | plan + execute + verify within 1 cm / 10° |
| `move_synchronized` | |
| `open_gripper`, `close_gripper` | |
| `go_home` | |

`MotionStrategy(ik_solver, trajectory_planner, pose_planner)` selects backends. The default is
PyRoki IK plus straight joint interpolation, which is *not* collision-aware. cuRobo is opt-in, and
the harness never falls back implicitly.

**Extensions** are namespaced by benchmark:

- `libero.get_task_metadata()` and `robosuite.get_controller_metadata()`;
- the mobile-manipulation set: `behavior.navigate_to_pose(x, y, yaw, planner=...)`,
  `behavior.get_base_pose()`, `behavior.move_torso()`, `behavior.plan_standoff_pose()`,
  `behavior.base_pose_is_free()`.

The BEHAVIOR set is the closest existing template for a legged base.

## 4. Providers (`providers/`)

Each provider is a FastAPI service in its own venv, reached over HTTP with bounded retries
(`providers/http.py`). Ports and venvs come from `configs/environments.json`, and GPU placement
from `configs/profiles/*.json`.

| Provider | Role | Runs on |
| --- | --- | --- |
| SAM3 (`sam3/service.py`) | Text or point → instance masks | GPU; HF-gated weights |
| Contact-GraspNet (`graspnet/`) | 6-DoF grasp candidates from object + scene clouds | GPU |
| PyRoki (`pyroki/service.py`) | JAX least-squares IK for one robot per process | **CPU fine** |
| cuRobo (`curobo/service.py`) | Collision-aware joint/pose planning against a voxelized depth scene, with robot points carved out by its sphere model; Panda and YAM configs | GPU; source build |
| Mink (`yam_kinematics/`) | MuJoCo-native differential IK, used for YAM | **CPU fine** |

## 5. Benchmarks and adapters

| Adapter | Robot | What is privileged (adapter-side only) |
| --- | --- | --- |
| `libero/adapter.py` | Panda, OSC → joint codec | MuJoCo state; `check_success` via BDDL predicates |
| `robosuite/adapter.py` | 1–2 Panda | `_check_success`; bimanual protocol evaluator |
| `behavior/adapter.py` (+ `planning.py`) | R1 Pro: virtual holonomic base, 4-joint torso, 2×7 arms | Ground-truth base pose, collision meshes, traversability map, held-object id; pickup witness |
| `yam_real/adapter.py` (+ `env.py`, `server/`) | 2× YAM 6-DoF on CAN; MuJoCo stand-in in `sim.py` | n/a (real); station calibration bundle |

Perception of task objects is non-privileged everywhere: SAM3 on rendered RGB-D. In simulation,
planning and evaluation use simulator truth.

## 6. The learning loop: paper ↔ code

| Paper element | Lives in | Enforced by |
| --- | --- | --- |
| Robot execution engine (§2.1) | `cap_harness` (recorder, typed errors, traces) | **Code** |
| Iterative debugging (actor) | `iterative-debugging/subagent-prompt.md` | Prompt; the harness just runs what it is given |
| Skill library (§2.2) | `iterative-debugging/skills/{grasp,localize,transport,manipulation,task-exploration}.md`, read whole into context | Prompt (merge rules); `lint_skills.py` checks structure and the sandbox API |
| Skill admission: verifier reproduces the fix | `verifier-prompt.md`, `scripts/dispatch.py verify`, `gen_progress.py` | Lint is code. The verdict is self-reported JSON, and promotion is a hand edit by the coordinator. |
| Evolutionary search (§2.3, Alg. 1) | `evosearch/*.md`, `evosearch_eval.py` (scores a candidate × seed matrix) | Prompt (K, top-3, stopping, selection) |
| Curriculum and backward transfer (§2.4) | `main-agent-prompt.md` (curriculum), `gen_progress.py:484-493` (rerun candidates) | Rerun selection is code, but it uses held-out scores (see findings B5) |
| Seed partitions (LIBERO 51–65 dev / 66–80 val / 1–50 held-out; Robosuite 101–125 / 1–100; BEHAVIOR 26–35 / 1–25) | `campaign.py:55-57`, `init_run.py`, `run_validation.py:533-535` | Disjointness and the held-out runner's range are code. **Who may run which seeds is prompt-only.** |
| System 2 on the real robot (App. D) | `agent/{server,mcp,session}.py`, `AGENTS.md`, `.agents/skills/*` | Code (MCP allowlist, one job at a time, motion interlock) plus prompt (task recipes) |

The design choice is deliberate. ASPIRE treats the coding agent as the optimizer and keeps the
harness small and strict. The consequence is that reproducibility and evaluation hygiene depend
on how well the model follows long prompts. See [findings.md](findings.md) §B and
[paper-assessment.md](paper-assessment.md).

## 7. Extension points, and where they are hard-coded

To add a robot today you implement the `EnvironmentAdapter` protocol (`protocols.py:145-195`):

- `embodiment`
- `reset`
- `get_task_context`
- `get_observation`
- `get_robot_state`
- `step`
- `execute_trajectory`
- `set_gripper` / `set_grippers`
- `close`

You can add any of these optional hooks, which `CapApi` finds with `getattr`:

- `resolve_arm`
- `go_home`
- `close_gripper` (contact-aware)
- `get_controller_metadata`
- `get_planning_context`
- `planning_workspace`
- `ik_request`
- `planning_scene_request`
- `gripper_holds_intermediate`
- `monitor_snapshot`
- the named `<embodiment>.*` extensions

You then also have to edit the lists in [findings.md](findings.md) §C1. The roadmap proposes
replacing them with one `EmbodimentSpec` registry.
