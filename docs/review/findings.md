# Findings: bugs, risks, and technical debt

Line numbers refer to the `ASPIRE v1.0 release` commit (`fa00f0e`) unless an entry says otherwise.

Each finding is tagged with how it was established:

| Tag | Meaning |
| --- | --- |
| **fixed + test** | Reproduced, fixed on this branch, and covered by a regression test that fails without the fix. |
| **verified** | Confirmed by reading the code and, where possible, running it on this branch. |
| **spot-checked** | Reported by a review pass and confirmed at the cited lines. |
| **reported** | Reported by a review pass and not independently re-checked. |
| **suspected** | Depends on behaviour of external software (Isaac Sim, OmniGibson, DaMiao firmware) that could not be run here. |

Severity reflects impact on correctness, evaluation integrity, or physical safety.

---

## A. Fixed on this branch

### A1. LIBERO "saved" init mode crashed on every real task — High · fixed + test

- **Where:** `src/cap_harness/libero/adapter.py:438`, `elif not init_states:`.
- **What:** LIBERO's `get_task_init_states()` returns the init-state bank from `torch.load`, as an
  `ndarray`. The truth value of a multi-element array raises `ValueError`, so
  `cap-harness run --benchmark libero-pro` failed at `reset()` whenever `--init-mode` was left at
  its default, `saved`. That includes the README's example commands. The unit tests used Python
  lists, so they never hit this.
- **Impact:** The research pipeline passes `--init-mode seeded` for LIBERO-Pro, which never reaches
  this branch, so the paper's numbers are unaffected. The documented single-run path was broken.
- **Fix:** `len(init_states) == 0`. Tests:
  `tests/test_libero_adapter_fake.py::test_saved_init_mode_accepts_an_ndarray_init_state_bank`
  and `::test_saved_init_mode_rejects_an_empty_ndarray_bank`. Verified end-to-end:
  `libero_90:33` close-microwave reaches `task_success=True` on CPU.

### A2. Sandbox: `str.format` read privileged simulator state — High · fixed + test

- **Where:** `src/cap_harness/runtime.py:192-200` (`_ProgramValidator.visit_Attribute`).
- **What:** The validator rejects `_`-prefixed attributes in the AST. Format-field traversal
  happens at runtime, so the validator never sees it. This program returned the adapter's
  privileged state as a string:

  ```python
  '{0.__closure__[0].cell_contents.__self__._adapter._native_env...}'.format(get_task_context)
  ```

  Any MuJoCo state could then be parsed back with `float()`, including object poses and contact
  data. That contradicts `docs/api-reference.md` ("no ... native simulator, reward, or
  success-predicate access").
- **Fix:** Programs are rewritten after validation. Every `x.format` / `x.format_map` goes
  through `_SafeFormatter`, which applies the validator's attribute policy to format fields.
  Ordinary formatting still works: `{:.2f}`, `{0[key]}`, `{0.frame}`, and `str.format(...)`.
  Tests: `tests/test_runtime.py::test_str_format_*`.

### A3. Sandbox: generator frames reached the host's live objects — High · fixed + test

- **Where:** `src/cap_harness/runtime.py:159-200`.
- **What:** `gi_frame`, `f_back` and `f_locals` contain no underscore. This program ran:

  ```python
  def walk(frame):
      while frame is not None:
          if 'adapter' in frame.f_locals: return frame.f_locals['adapter']
          frame = frame.f_back
  frames = (walk(gen.gi_frame) for i in range(1)); gen = frames
  ```

  It returned the `adapter` local of the calling host frame (`run.run_program`). With that, a
  program could read success predicates or teleport simulated objects. On `yam_real` it could call
  adapter internals around the API-level checks.
- **Also:** The read-only arrays returned by contracts exposed `ndarray.tofile` and `.dump`
  (arbitrary file writes) and `.ctypes` (raw memory).
- **Fix:** A `FORBIDDEN_ATTRIBUTES` denylist (frame, code and traceback introspection, plus NumPy
  I/O) is enforced both in the AST and inside format fields. Tests:
  `tests/test_runtime.py::test_generator_frames_cannot_reach_host_locals` and
  `::test_introspection_and_numpy_io_attributes_are_rejected`.
- **Still true:** Programs run in-process. This is defence in depth, not a security boundary;
  the README already says so. See C9.

### A4. The step limit did not stop the program — Medium-High · fixed + test

- **Where:**
  - `src/cap_harness/artifacts.py:473-476`: `before_step` raises `StepLimitReached(RuntimeError)`.
  - `src/cap_harness/api.py:863-874` and `api.py:923-944` catch `Exception` and return a failed
    result instead.
- **What:** Once the budget was spent, every motion call returned `ok=False` at once and the
  program kept running. A retry loop such as `while not move_to_pose(...).ok: ...` spun forever.
  There is no wall-clock timeout (see C6), so the worker hung. The `except StepLimitReached`
  branch in `run.py:414` could not be reached from API-mediated steps.
- **Fix:** `ProgramExecutor` checks the recorder at every tool boundary. Once
  `terminal_reason == "step_limit"`, it raises `ProgramAborted`, a `BaseException`. Bare `except:`
  in programs is rewritten to `except Exception:`, so the abort cannot be swallowed. The run ends
  with `termination_reason=step_limit` and a `truncated` error.
- **Tests:** `tests/test_runtime.py::test_program_stops_when_step_budget_is_exhausted`. On the
  old code this test hangs.
- **End-to-end check:** An infinite retry program on Robosuite with `--max-steps 12` now stops
  after 12 steps.

### A5. Doctor reported a bogus commit mismatch for uninitialized submodules — Low · fixed + test

- **Where:** `src/cap_harness/doctor.py:560-575` and `doctor.py:668-690`.
- **What:** `git rev-parse HEAD` inside an empty, uninitialized submodule resolves the **parent**
  repository's HEAD. The doctor therefore reported "does not match the dependency lock" instead
  of "not initialized". `bootstrap_libero.sh` already guarded against this; the doctor did not.
- **Fix:** `_submodule_head()` checks `--show-toplevel` first. The behavior doctor test now
  accepts an absent BEHAVIOR-1K checkout.

### A6. Test and doc hygiene — Low · fixed

- Four test modules imported optional extras unconditionally (`jsonschema`, `fastapi`, `msgspec`,
  `viser`, `mink`). Collection therefore aborted in `.venv-robosuite` and `.venv-libero`. They
  now use `pytest.importorskip`.
- Three release-asset tests failed because the public README no longer linked
  `docs/environments.md`, `api-reference.md`, `libero-pro.md` and `robosuite.md`. The README now
  has a Documentation table.

---

## B. Evaluation integrity and the skill pipeline (`.claude/skills/**`, `validation/**`)

Most of the paper's method is enforced by prompts that the coding agent is asked to follow, not
by code:

- seed-partition discipline;
- the verifier gate;
- evosearch's K and top-3 parameters;
- curriculum;
- the "non-privileged API audit".

What code enforces:

- the program sandbox;
- seed-band disjointness at campaign init (`campaign.py:55-57`, `init_run.py:76-90`);
- the held-out seed-range check in `run_validation.py:533-535`;
- structural linting of skill proposals (`campaign.py:389-497`, `lint_skills.py`).

### B1. The verifier runs Robosuite skill admission on held-out seeds — High · spot-checked

- **Where:** `.claude/skills/iterative-debugging/verifier-prompt.md:92,121,139,164` hard-code
  seeds `51-55`, and `scripts/dispatch.py` never substitutes them.
- **Why it matters:** The Robosuite held-out band is 1-100 and its dev band is 101-125. Skills are
  therefore admitted, and `v02` re-debugged, on evaluation scenes.
- **Fix:** Derive the verifier seeds from `campaign.json` `dev_seeds[:5]`, and fail dispatch on
  any unfilled `<...>` placeholder.

### B2. The evosearch sanity check can discard the evolved winner — Medium-High · spot-checked

- **Where:** `.claude/skills/evosearch/subagent-prompt.md:315-345`. The "baseline" is
  `max(pass_rate)` over every iteration's `candidate_A`.
- **What:** `candidate_A` is the verbatim `fix_code.py` only in iteration 0; every later
  iteration rewrites `candidate_A` through `candidate_H` (line 226). If the global best is a later
  `candidate_A`, the check `best_rate > baseline` is false and the script falls back to
  `fix_code.py`.
- **Fix:** Compare against iteration 0's `candidate_A` only.

### B3. Evosearch is LIBERO-specific but documented for Robosuite — High if used there · reported

- **Where:** `evosearch/subagent-prompt.md:57,173-176,241-243,297,373-374`;
  `evosearch_eval.py:44,66`.
- **What:** Seeds 51-65 and 1-50, `--init-mode seeded` and `.venv-libero` are hard-coded. On
  Robosuite, the "dev" seeds 51-65 are held-out seeds.

### B4. Evosearch never uses the validation split — Medium · reported

- The paper's Table 10 says the per-task winner is "selected on a held-out validation set
  (seeds 66–80)".
- The released evosearch picks its winner on the same dev seeds (51-65) that drove the search.
- In iterative debugging, seeds 66-80 are a one-time overfitting check and do not drive
  selection.

### B5. Held-out scores feed back into what gets re-run — Medium (methodology) · reported

- **Where:** `gen_progress.py:484-493` and `evosearch/main-agent-prompt.md:121`.
- **What:** Tasks are picked for backward-transfer reruns by held-out pass rate below 0.8, and
  for evosearch by fix-loop held-out below 90%. They are then re-scored on the same held-out
  seeds.
- **Why it matters:** Regression to the mean inflates measured gains. This applies to the paper's
  backward-transfer table (+7.6 pp).
- **Fix:** Select on dev or unseen-check scores.

### B6. Privileged evaluator metrics are written into development runs — Medium-High · reported

- **Where:** `run.py:269-291` attaches `RobosuiteBimanualProtocolEvaluator` to every
  `two_arm_lift` / `two_arm_handover` run, debug runs included.
  `robosuite_bimanual.py:208-246` then writes `stage_metrics` (receiver-to-handle offsets, contact
  geoms) to `evaluation/protocol.json`.
- **Why it matters:** Actors can read these files. That contradicts `docs/run-artifacts.md:44-45`.

### B7. BEHAVIOR "held-out" numbers are best-of-k on the evaluation seed — Methodology · verified

- **Where:** `behavior/subagent-prompt.md` (Stage 2) and `behavior/score_campaign.py:96-108`.
- **What:** The worker iterates on each evaluation seed. The score is
  `any(r["success"] for r in final)` across repeated runs of the final program. That is
  oracle-guided search, not a frozen evaluation, and it should be reported separately from the
  tabletop held-out numbers.

### B8. Nothing in code stops an actor running held-out seeds — Design · verified

`cap-harness run --seed 1` and `run-batch` have no partition guard; only the prompts forbid it.

- **Fix:** A campaign token, or a held-out allowlist that only `run_validation` can lift.
- **Fix:** Run actors under a separate OS user or container with no read access to validation
  outputs.

### B9. Worked solutions are inside the agent's reach — Medium · reported

`.claude/skills/iterative-debugging/robosuite/*_fix_code.py` are held-out-validated solutions.
`evosearch-iteration.md:60` tells actors to `find .claude/skills -name "*.md"`, which surfaces
`SUCCESS_RATES.md`. Only the verifier is barred from `examples/`.

### B10. Smaller pipeline bugs · reported

Each item names its file and the failure.

- **`dispatch.py:251-261`:** never fills `BENCHMARK` (the key mismatch is at line 255),
  `MAX_STEPS`, `DEV_SEEDS`, `UNSEEN_SEEDS` or `HELDOUT_COUNT`.
- **`dispatch.py:143-156`:**
  - `manifests[-1]` is ordered by hash, so it may pick a subset run or one still in progress.
  - The denominator is hard-coded as `/50`; Robosuite has 100 held-out seeds.
  - It tells the verifier the proposer's held-out score.
- **`dispatch.py`:** has no rerun/attempt option. Rerun proposals stay "proposed" forever
  (`gen_progress.py:462-465`).
- **`run_validation.py:98-125`:** the run identity excludes the harness commit, so `--resume`
  after a harness fix mixes pre-fix and post-fix seeds in one manifest.
- **`validation/runner.py:223-227`:** `_find_run_dir` takes the newest `outcome.json`. A rerun
  that crashes before writing its outcome reuses the previous result (a stale pass).
- **`evosearch_eval.py:309-324`:** a `--candidates` subset run truncates `iter_summary.json`.
- **`selectors_libero.py:299-306`:** ignores case seeds.
- **`trace_stats.py:43-44`:** recognizes only `libero_` suites.
- **Machine-specific assumptions:**
  - GPU ids `3 4 5 6 7`;
  - `/root/.claude/projects/*/*/subagents` (`watchdog.py:28`, `trace_stats.py:36`);
  - `model="opus"`;
  - `groot/cap` (`lint_skills.py:37`).
- **Codex port:** the root `AGENTS.md` that Codex auto-loads is the *real-robot* runbook ("act
  only through `mcp__cap__*` tools"). That conflicts with the shell-driven simulation pipeline.

---

## C. Harness core (`src/cap_harness/*.py`)

### C1. Adding an embodiment means editing about 10 hard-coded lists — High (extensibility) · verified

Each place below has to change by hand for a new embodiment:

- **`contracts.py:15-24`:** `JOINT_DIMENSIONS`. Only 6- or 7-DoF arms are accepted anywhere
  (`SUPPORTED_JOINT_DIMENSIONS`, `contracts.py:190-201`).
- **`api.py:108-110`:** silently maps an unknown embodiment to `"robosuite"`. The next
  `get_robot_state()` then fails with a confusing mismatch.
- **`api.py:1567-1577`:** the extension tool names are a fixed tuple.
- **`registry.py:41-83`:** per-benchmark allowlists and `_APPROVED_EMBODIMENT_PREFIXES`.
- **`run.py:86-133`, `run.py:138-154`, `run.py:247-357`:** if/elif dispatch on the benchmark name,
  with provider wiring per benchmark.
- **`cli.py:141-196`:** benchmark choices.
- **`api.py:1732-1743`:** when an adapter has no `get_planning_context()`, the planner is told the
  robot is `"panda"` / `"dual_panda"`.
- **Real-robot stack:** the cuRobo service model literal (`providers/curobo/service.py:68`),
  `agent/server.py`, `agent/mcp.py:150` and `agent/visualizer.py`.

See [roadmap-quadruped-arm.md](roadmap-quadruped-arm.md) §2 for a registry-based design.

### C2. The default motion path is not collision-aware — Medium · verified

- **Where:** `api.py:599-682`.
- **What:** `MotionStrategy()` defaults to PyRoki IK plus straight joint interpolation, and marks
  the result `collision_aware=False`. Only cuRobo plans around obstacles, and it needs a GPU.
  Every `move_to_pose` on a CPU-only stack can sweep through the table or other objects.
- **Note:** `move_to_joints` and `go_home` are direct servo moves on every embodiment.

### C3. IK quality — Low-Medium · verified

- **Where:** `providers/pyroki/service.py:47-104`.
- **What:**
  - It uses one seed, zeros when none is given, and has no restarts and no collision or
    manipulability cost.
  - Each service process serves one robot (`--robot`, `--target-link`).
  - `_apply_joint_margin` (lines 120-128) subtracts 0.15 rad from both ends of every revolute
    range, which inverts any range narrower than 0.3 rad.
  - The client rejects only on error thresholds; IK failures look like `IK_FAILED` with no
    alternatives.

### C4. `localize_object` trusts `segmentations[0]` — Low-Medium · verified

- **Where:** `api.py:1159`.
- **What:** With several matches, which instance is picked depends on the provider's ordering.
  The paper's "multi-object disambiguation" skill exists to work around this. A `select=`
  argument (by score, by spatial qualifier, or nearest to a point) would make it explicit.

### C5. Masks are re-applied to a newer frame without a freshness check — Low-Medium · verified

- **Where:** `api.py:306-368` (`generate_grasps`) and `api.py:243-254`.
- **What:** `generate_grasps` re-reads the camera and applies a mask produced from an earlier
  image. After any arm, base or torso motion, the mask no longer lines up with the depth image.
- **Fix:** Carry the source timestamp on `Segmentation` and reject a mismatch.

### C6. No wall-clock budget per program — Medium · verified

- **What:** A pure-Python infinite loop that calls no tools hangs a run or batch worker forever.
  `batch.py` has no per-seed timeout, and A4 only fires at tool boundaries.
- **Fix:** A per-seed `future.result(timeout=...)` plus worker kill in `batch.py`, or a
  `signal.setitimer` budget in the spawned worker.

### C7. Minor API inconsistencies · verified

- **`api.py:1391`:** the orientation tolerance for `move_to_pose` is a fixed 10° and cannot be
  configured. `final_errors` records only the position error.
- **`api.py:641` vs `api.py:586`:** the composed plan passes `_planning_context(state)` (current
  state), while the integrated plan uses `planning_state` (the requested start). They differ
  when `start_joints` is given.
- **`api.py:437-455`:** some `IKResult` failures omit `arm=`.

### C8. Silent downgrade of failures to "planning failed" · reported

- **Where:**
  - `behavior/planning.py:509-514`: any exception from cuRobo, including CUDA OOM, becomes
    `PLANNING_FAILED` and triggers fallbacks.
  - The many `except Exception` blocks in `api.py` turn infrastructure faults into ordinary
    failed results.
- **Why it matters:** That is good for the agent loop but bad for diagnosis. Keep the exception
  type in `details` everywhere; most sites already do.

### C9. The sandbox is in-process — Design · verified

Even with A2–A4, programs share the interpreter with the adapter and simulator. For untrusted
models, or any real-hardware use, run programs in a subprocess that only sees serialized contracts
over an RPC boundary. `ProgramExecutor` + `ToolRegistry` is already the right seam for that.

---

## D. Real YAM robot and System 2 (`yam_real/**`, `agent/**`)

These matter directly if you reuse this stack for your own arm.

### D1. `move_to_joints` on YAM is effectively a step command — High (safety) · spot-checked

- **Where:**
  - `api.py:1242-1256` loops `step()` with the **final** goal.
  - `yam_real/adapter.py:536-538` submits each step with `duration_s=control_period_s` (1/30 s).
  - `adapter.py:504-512` builds a two-waypoint batch `[held, goal]`.
- **What:** A 1 rad request is streamed within about 33 ms at kp=80, and nothing checks for
  collisions on joint moves.

### D2. Joint velocity limits are parsed but never enforced — High (safety) · spot-checked

- **Where:** `max_joint_velocity_rad_s`, `max_tracking_error_rad`, `rpc_timeout_s`,
  `final_joint_tolerance_rad` and `max_acquisition_elapsed_s` are parsed in
  `yam_real/config.py:155,423`, and nothing else reads them.
- **What:** The interpolation step limit comes from action bounds ÷ control rate, capped at
  0.05 rad per tick (about 1.5 rad/s at 30 Hz). The URDF velocity limits are 1000.

### D3. Commanding one arm releases the other arm's grip — High · spot-checked

- **Where:** `yam_real/adapter.py:617` and `adapter.py:630` (`held_grip` = **measured** width
  for unaddressed arms); also `step` (`adapter.py:498-503`), `env.py:337-339` and
  `action.py:110-113`.
- **What:** The adapter's own docstring explains that re-commanding the measured width "releases
  the squeeze". This breaks any bimanual task where one hand holds while the other works.
- **Fix:** Keep a commanded-width memory per arm in `RealYamEnv`.

### D4. A dead hold loop is invisible — High · reported

- **Where:** `server/yam_robot.py:450-452` exits the hold thread after 20 faults without
  disabling the motors. `arm_server.py:173-185` watches only the RPC threads. The harness never
  calls `get_health`.

### D5. No software stop path — High · reported

- **Where:** `control/controller.py:42-82` defines `request_stop` and pause, and nothing calls
  them. There is no MCP cancel tool and no viser stop button.
- **Latent bugs:**
  - A `KeyboardInterrupt` used as a stop would escape `except Exception` in `session.py:449` and
    be reported as `ok=True`.
  - Pause/resume does not shift `t0` (`controller.py:193-197`).

### D6. Transport — Medium-High · reported

- **Where:** `transport.py:101-116`.
- **What:**
  - Commands are fire-and-forget, so errors surface on a later call from any thread.
  - There are no RPC timeouts.
  - portal `autoconn` may replay queued targets after a reconnect (**suspected**).

### D7. Other real-robot issues · reported

Each item names its file and the risk.

- **`arm_server.py:158`:** arm servers bind `0.0.0.0` with no authentication.
- **`yam_robot.py:301-308`:** client-supplied `kp`/`kd` are not range-checked.
- **`cameras.py:213-229`:** a failed camera keeps serving its last frame. No age check exists
  before perception (`codec.py:133-156`).
- **`yam_robot.py:345-347`:** the gripper torque cap is probably about 2.7× the configured value,
  because the DaMiao ratio is scaled by T_max (**suspected**).
- **`adapter.py:343`:** `reset()` closes both grippers (`go_home(keep_grippers=False)`).
  System-2 Home opens them.
- **`motors.py:130-138`:** a partial motor `connect` failure leaves earlier motors enabled.

---

## E. BEHAVIOR-1K mobile manipulation (`behavior/**`)

Isaac Sim could not run here, so OmniGibson-dependent items are suspected.

### E1. Navigation re-commands finger targets — Medium · suspected

- **Where:** `adapter.py:1207`.
- **What:** `execute_joint_targets` sends cuRobo's full joint vector, with the fingers set from
  the measured state. After `close_gripper`, a `curobo` navigation may loosen the grasp, which
  matters when carrying an object.

### E2. `solve_ik` returns arm joints from a torso-moving solution — Medium · reported

- **Where:** `planning.py:203-208` and `planning.py:555`.
- **What:** The torso is free in IK, but only the arm joints are returned. Executing them with
  the current torso misses the target.

### E3. Stale-plan checks ignore base pose and torso — Medium · reported

- **Where:** `api.py:1814-1826` and `adapter.py:1135-1181`.
- **What:** `RobotState` has no base pose, so an arm plan made before `navigate_to_pose` is not
  invalidated.

### E4. Yaw seam at ±π in `_servo_base` — Medium if real · suspected

- **Where:** `adapter.py:1434-1442`.

### E5. Privileged information in the harness — Methodology · reported

- **What:** The harness uses:
  - the ground-truth base pose (`adapter.py:494-496`);
  - simulator collision meshes, and which object is held (`planning.py:259-330`);
  - the precomputed traversability map (`config.py:92-95`).
- **Exposed to programs:** `behavior.base_pose_is_free()` answers from that map.
- **Tuning:** Harness constants were tuned on evaluation seeds (`adapter.py:57-60,80-91`).
- Target perception itself (SAM3) is non-privileged.
