# Roadmap: ASPIRE for a quadruped ("dog") robot and a robot arm

The goal is to reuse ASPIRE's execution engine, typed API, traces and skill pipeline for:

1. a standalone arm, and
2. a legged mobile manipulator (quadruped with the arm mounted on its back),

in simulation first and on hardware later.

This plan is ordered so that every phase produces something runnable on a CPU-only machine
(`docs/cpu-dev-setup.md`), and the GPU-only providers become drop-in upgrades.

> **Hardware assumptions to confirm.** This plan does not yet know your exact robots. Typical
> pairings are a Unitree Go2/B2 with a Z1, Piper or D1 arm, or Spot with its arm. Phases 2–3
> name MuJoCo Menagerie models as stand-ins. Swap in your own MJCF/URDF as soon as you have it.

## Phase 0: Environment (done on this branch)

- `scripts/bootstrap_cpu_dev.sh` builds LIBERO, Robosuite, PyRoki-on-CPU and a full dev venv.
  The full test suite runs, and two provider-free or IK-only simulation episodes run.
- Bugs that blocked or undermined development are fixed: the LIBERO init crash, the sandbox
  escapes, and the runaway step limit. See [findings.md](findings.md) §A.
- **For campaigns, get a GPU host.** Use an RTX 4090/5090 or L40, plus a Hugging Face token for
  SAM3. CPU rendering costs about 0.35 s per tick.

## Phase 1: Make embodiments pluggable (harness refactor)

Today a new robot means editing about 10 hard-coded lists ([findings.md](findings.md) §C1).
Replace them with one registry, so a new embodiment is one package plus one registration.

```python
# src/cap_harness/embodiments.py  (new)
@dataclass(frozen=True)
class EmbodimentSpec:
    name: str                                   # "go2_z1", "my_arm"
    arm_joint_dims: Mapping[str, int]           # {"primary": 6}; any DoF, not just 6/7
    adapter_factory: Callable[..., EnvironmentAdapter]
    task_registry_factory: Callable[[], TaskRegistry]
    extension_tools: Mapping[str, str]          # {"base.navigate_to_pose": "execution", ...}
    default_camera: str
    default_motion_strategy: MotionStrategy
    planning_model: str | None                  # cuRobo/PyRoki robot id, never a silent "panda"
    provider_factory: Callable[[EnvironmentAdapter], ProviderSet] | None = None

register_embodiment(spec)                       # entry point: [project.entry-points."cap_harness.embodiments"]
```

Concrete changes:

1. **Joint dimensions.** In `contracts.py`, replace `JOINT_DIMENSIONS` /
   `SUPPORTED_JOINT_DIMENSIONS` with a lookup into the registry, and allow any DoF ≥ 1.
2. **Unknown embodiments.** In `api.py:108`, raise for an unknown embodiment instead of falling
   back to `robosuite`.
3. **Extension tools.** In `api.py:1567`, ask the adapter (or spec) for its `extension_tools()`
   instead of a fixed name tuple.
4. **Allowlists.** In `registry.py`, build the allowlists and prefixes from registered specs.
   Keep the privileged-name filter.
5. **Run wiring.** In `run.py` and `cli.py`, look up the spec. Provider wiring comes from
   `spec.provider_factory`.
6. **Planning context.** In `api.py:1732`, remove the Panda fallback and require
   `get_planning_context()` whenever a planner is used.
7. **Contract tests.** Add a test that runs the same `EnvironmentAdapter` checks against every
   registered embodiment (reset, observation, step, trajectory, gripper, close).

Do this first. It is about two to three days of work, entirely testable on CPU, and every later
phase depends on it.

## Phase 2: Your arm in simulation (MuJoCo, CPU)

Goal: `cap-harness run --benchmark my_arm ...` runs your arm on tabletop tasks with the existing
API.

1. **Model.**
   - Use your vendor MJCF/URDF. Menagerie has candidates: `unitree_z1`, `agilex_piper`,
     `ufactory_xarm7`, `franka_fr3`, `universal_robots_ur5e`, `kinova_gen3`, `i2rt_yam`.
   - Add a table, objects and cameras. Use an agent-view camera plus a wrist camera, rendered
     with `mujoco.Renderer`, with depth from the depth buffer → metric.
2. **Adapter.**
   - Model it on `robosuite/adapter.py`, the smallest complete adapter, at about 900 lines.
   - Use joint-position control at 20–30 Hz.
   - `CameraObservation.camera_pose` must be in the arm base frame.
   - Implement:
     - `get_controller_metadata()`: action bounds **and real velocity limits**;
     - `get_planning_context()`;
     - `planning_workspace()`;
     - `gripper_holds_intermediate`, which is `True` for position-controlled grippers.
3. **IK.**
   - Generalize `yam_real/kinematics.py` + `providers/yam_kinematics` into a
     `MinkIKProvider(model_xml, site_name, joint_names)`. It is MuJoCo-native, runs on CPU, and
     already works for YAM.
   - Alternatively, run a second PyRoki instance with `--robot <your>_description`.
   - Add multi-seed restarts; the current IK is single-seed.
4. **Collision-aware motion without a GPU.** cuRobo needs CUDA, so until you have one:
   - Add an interpolation planner that checks MuJoCo collisions along the path (`mj_collision`
     on a scratch `MjData`) and rejects colliding plans, or
   - use OMPL or a simple RRT-Connect in joint space with MuJoCo as the collision checker.

   Register it as `trajectory_planner="mujoco_rrt"`. It is a far safer default than blind
   interpolation.
5. **Perception without a GPU.**
   - Add a *development-only* `SimSegmentationProvider` that renders MuJoCo segmentation masks
     (`Renderer.enable_segmentation_rendering`) and maps body names to text labels.
   - Mark it `privileged=True` in run provenance, and refuse it in held-out validation runs.
   - On a GPU host, switch to SAM3 with no program changes.
6. **Tasks and evaluators.**
   - Use a task registry with procedural, seed-controlled object placement.
   - Write success predicates as host-side `check_success()`, never as tools.
   - Define dev, validation and held-out seed bands per task in the registry, enforced in code
     (Phase 5).
7. **Smoke.** Port `examples/robosuite/cube_lifting_seed1.py` to the new benchmark as the first
   end-to-end program.

## Phase 3: The legged mobile base (MuJoCo, CPU)

The only mobile manipulator today is BEHAVIOR's R1 Pro, whose base is a position-servoed virtual
holonomic joint driven with simulator-truth maps. A quadruped needs a real abstraction.

### 3a. Base protocol and state

Add `BaseState` to `RobotState`, or carry it alongside it:

```python
@dataclass(frozen=True)
class BaseState:
    pose_xy_yaw: tuple[float, float, float]     # in "odom"
    height_m: float; roll: float; pitch: float  # body pose
    twist: tuple[float, float, float]           # vx, vy, wz
    mode: Literal["standing", "walking", "sitting", "fallen", "estop"]
    localization_ok: bool
```

Then define the protocol:

```python
class MobileBase(Protocol):                     # adapter mixin; tools registered as `base.*`
    def get_base_state(self) -> BaseState: ...
    def navigate_to_pose(self, x, y, yaw, *, tolerance_m=0.05, tolerance_rad=0.1,
                         max_steps=..., timeout_s=...) -> ExecutionResult: ...
    def command_velocity(self, vx, vy, wz, *, duration_s) -> ExecutionResult: ...
    def stop_and_hold(self, *, timeout_s=2.0) -> ExecutionResult: ...   # before any arm motion
    def set_body_pose(self, height_m, pitch, roll) -> ExecutionResult: ...  # quadruped "torso"
    def plan_approach_pose(self, object_cloud, distance_m=0.6) -> tuple[float, float, float]: ...
    def base_pose_is_free(self, x, y, yaw) -> bool: ...   # from a *perceived* costmap on real HW
```

Rules the harness must enforce in code:

- `expected_start` captures the base pose and body pose. `execute_trajectory` returns
  `STALE_PLAN` if the base has drifted beyond tolerance since planning. This fixes the gap noted
  in findings E3.
- Arm motion requires `mode == "standing"` and a settled twist.
- New error codes: `BASE_BLOCKED`, `BASE_FALLEN`, `LOCALIZATION_LOST`, `GOAL_UNREACHABLE`, kept
  separate from arm `PLANNING_FAILED`.
- Camera extrinsics are chained body → odom at observation time. Arm planning happens in the
  **arm base** frame, which moves with the body.

### 3b. Locomotion in simulation, in two steps

1. **Kinematic base first.**
   - Give the robot planar `x`, `y`, `yaw` slide/hinge joints plus height, and drive them with
     velocity commands. Legs are cosmetic or fixed.
   - This is exactly how BEHAVIOR treats its base.
   - It unblocks all manipulation and navigation work on CPU in days.
2. **Learned gait second.**
   - Run a velocity-tracking locomotion policy (ONNX, through `onnxruntime` on CPU) inside the
     adapter's `step()`: `(vx, vy, wz) + proprioception → 12 joint targets`, at 50 Hz under the
     30 Hz harness tick.
   - Candidates: Menagerie `unitree_go2` or `boston_dynamics_spot` (with `spot_arm.xml`) as
     models. Train the policy with MuJoCo Playground, mjlab or Isaac Lab on a rented GPU.
   - Pretrained MuJoCo-ready Go2 checkpoints are not reliably published, so plan to train one.
     References: `unitree_rl_mjlab`, `rl_sar`, `Go2_ARX_mjlab` (quadruped + arm).
   - Treat these as starting points to verify. Licences vary per model.

### 3c. Navigation stack

- **In simulation:** use an occupancy grid from depth (perceived, not privileged), A* over it,
  and a pure-pursuit or DWA local controller that emits `command_velocity`.
- Reuse BEHAVIOR's fallback ladder:
  1. direct plan;
  2. hop waypoints;
  3. servo;
  4. multi-angle approach (paper Fig. 2).
- Reuse BEHAVIOR's diagnostics vocabulary: `route`, `hops`, `moved_*`, `direct_status`. The
  skills the paper learned on BEHAVIOR (multi-angle approach, way-hopping) transfer as text.

## Phase 4: The learning loop for your benchmarks

1. **Copy the skill.** Copy `.claude/skills/iterative-debugging/` to a new benchmark profile.
   Before any campaign, fix the pipeline issues in [findings.md](findings.md) §B:
   - Fill every placeholder from `campaign.json`, and fail on leftover `<...>`.
   - Derive verifier seeds from `dev_seeds`.
   - Select rerun and evosearch tasks on dev or validation scores, never held-out.
   - Strip privileged evaluator metrics from dev-run artifacts.
2. **Enforce partitions in code.** Make `cap-harness run` / `run-batch` refuse held-out seeds
   unless `run_validation` passes a token. Run actors in a container without read access to
   validation outputs.
3. **Skill library per embodiment.** Keep `skills/shared/*.md` for API-level debugging workflows
   and `skills/<embodiment>/*.md` for robot-specific knowledge. Give each entry front-matter:
   `id`, `trigger`, `preconditions`, `evidence` (task, seeds, harness commit), `verified_by`.
   Retrieve with BM25 or embeddings keyed on failure codes instead of whole-file context.
4. **Seed the library.** Use the paper's transferable entries: multi-object disambiguation, OBB
   grasp yaw, push for flat objects, multi-angle approach, way hopping.

## Phase 5: Hardware

Lessons from the YAM review ([findings.md](findings.md) §D) are requirements, not nice-to-haves.

### Arm

- Write an adapter over the vendor SDK: `z1_sdk`, `xArm-Python-SDK`, `ur_rtde`, `libfranka`, or
  ROS 2 `ros2_control` / MoveIt 2 `moveit_py`.
- Keep the YAM split: a thin per-arm **server** owns the bus and does safety; the harness
  **client** plans.
- The server must enforce:
  - joint-position **and velocity** limits;
  - a per-tick step clamp;
  - a tracking-error trip;
  - a command-age watchdog that holds or brakes when commands stop;
  - torque and gain bounds.
- If its control loop dies, it must disable the motors and report unhealthy.

### Base (example: Unitree Go2)

- Use `unitree_sdk2_python` `SportClient.Move(vx, vy, vyaw)`, `StopMove()` and `BalanceStand()`
  for `command_velocity` / `stop_and_hold`.
- Use onboard odometry or SLAM for `get_base_state` and a perceived costmap.
- `unitree_mujoco` bridges SDK2 low-level commands to MuJoCo for software-in-the-loop testing.

### Both

- **Software e-stop.** Expose an MCP `cap_stop` tool and a UI button that halt both base and arm.
  This needs `BaseException`-safe job handling (YAM D5).
- **Remember commanded gripper widths.** Never re-command a measured width (YAM D3).
- **Plan joint moves.** `move_to_joints` must plan a velocity-limited trajectory, not stream the
  goal (YAM D1).
- **Network.** Bind servers to localhost or a private interface, and authenticate them.
- **System 2.** Reuse `agent/` (MCP bridge, `LiveAgentSession`, viser). Generalize its YAM
  imports through the Phase 1 registry.

## Suggested order and effort

| # | Work item | Rough effort | Runs on CPU |
| --- | --- | --- | --- |
| 1 | Phase 1 registry refactor + adapter conformance tests | 2–3 days | yes |
| 2 | Arm MuJoCo adapter + Mink IK provider + sim segmentation (dev-only) | 3–5 days | yes |
| 3 | MuJoCo collision-checked planner | 2–3 days | yes |
| 4 | Kinematic-base quadruped + `MobileBase` + `base.*` tools + stale-base checks | 4–6 days | yes |
| 5 | Pipeline hardening (§B fixes, code-enforced partitions) | 3–4 days | yes |
| 6 | First campaign on 5–10 arm tasks | 1–2 days of agent time | GPU recommended |
| 7 | Learned gait in sim (train on GPU, run ONNX on CPU) | 1–2 weeks | inference only |
| 8 | Hardware adapters + safety servers + e-stop | 2–4 weeks | — |
