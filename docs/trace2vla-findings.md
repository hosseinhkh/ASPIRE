# Trace2VLA: findings log

Everything learned while building and running the project that is not obvious from the code:
what was measured, what went wrong and why, what was fixed, and what is still open. Each entry
says how sure it is. Add new entries at the top of their section, with the date.

The goal and status are in the [overview](trace2vla.md); the work ahead is in the
[roadmap](trace2vla-roadmap.md). Detailed evidence for the teacher-agent problems is in
[harness-fixes.md](harness-fixes.md), and for the design measurements in
[trace2vla-review.md](trace2vla-review.md).

Last updated: 12 November 2026.

## Open items at a glance

| # | Finding | Status |
|---|---|---|
| T1 | Grasps suggested by the grasp planner fail every time they are executed (12 of 12) | **Open.** Cause not established. Workaround: the program's own top-down grasp |
| T4 | A confident SAM3 mask on the wrong object cannot be detected from scores | Open by nature. The agent must check the image |
| T5 | The basket centre from the median of points is 5 cm off; task 0's program still uses it | Program still fragile (4 of 50 lowering moves stalled, all 4 still succeeded) |
| I1 | The GPU memory reservation is not handed over on demand | Open. Workaround documented |
| I2 | A busy machine makes episodes 3–15× slower; CPU load is not recorded | Open. How to check is documented |
| I4 | Bedrock's daily token quota | Outside the repository. Teacher collection and all student stages use no LLM |
| P3 | Task 0 teacher episodes are long (506–636 steps); converted length may be near the student's 280-step limit | To be read off the first conversion |
| P5 | Fine-tuning on the GPU has never been run | Next step of the preflight |

## Teacher (the ASPIRE agent and its programs)

### T1. Grasps suggested by the grasp planner fail when executed (12 November)

* **What.** `generate_grasps` (Contact-GraspNet) followed by `select_grasp(strategy="top_down")`
  sometimes returns a candidate. Every episode that executed such a candidate closed the gripper
  on nothing: 8 of 8 in the first run of task 1 (cream cheese) and 4 of 4 in task 0 (alphabet
  soup) on the official scenes. In the same runs the programs' own top-down grasp (hand pointing
  down at the object's centre) succeeded 137 of 137.
* **Evidence.** `outputs/trace2vla/preflight2/teacher/official/task_00/{0019,0028,0032,0033}`;
  the `select-grasp` and `close-gripper` calls of each run.
  * The candidates have low scores (0.15–0.22).
  * Each candidate's position is 0.103–0.109 m from its own `contact_point`, the distance from
    Contact-GraspNet's gripper base to its fingertips (0.1034 m). The hand went to the
    candidate's position and stopped about 10 cm from the object.
  * `select_grasp("top_down")` accepts an approach up to 37° from vertical
    (`vertical_threshold=0.8`); the four candidates were tilted 24–35°.
* **What was tested.** Moving the four recorded candidates 0.1034 m along their approach
  direction and executing them in the simulator. The hand then reached the can, and still closed
  on nothing in all four (two moves stalled against the can). So "the pose is the gripper base,
  shift it by 10 cm" is **not** a sufficient fix, and it is not established whether the harness
  mixes two frames or the candidates are simply poor for these objects.
* **Also.** For the 2 cm cream cheese box the planner often returns no grasp at all, and a call
  costs 16–42 s when the planner runs on the CPU (I2).
* **Practice until this is understood.** Teacher programs for cans and boxes should use their
  own top-down grasp at the object's geometry centre, and treat the planner's candidate as
  optional. The agent prompt does not say this yet.
* **To do.** Compare with the ASPIRE paper's released programs, if any use these candidates
  successfully; test candidates on an object with a clear side grasp; decide whether the
  candidate pose should be converted to the public end-effector frame.

### T2. SAM3 does not recognise product names (11–12 November, fixed)

* "alphabet soup" scores about 0.11 on every object, and the best mask is the milk carton.
  "cream cheese" scores 0.011. Descriptions of appearance work: "blue soup can" 0.75, "small blue
  rectangular box" 0.52, "woven basket" 0.82.
* `localize_object` returns `ok=True` for a guess. Two agent sessions lost every version to this.
* **Fixed** by giving the quick agent the pipeline's exploration step (look at the scene, probe
  prompts, check masks) and by warnings in `run-batch`. [harness-fixes.md](harness-fixes.md)
  section 5.

### T3. The agent's batches were cut off at 10 minutes (12 November, fixed)

* The prompt asked for a 600000 ms timeout; on a busy machine three seeds did not fit, and two
  versions lost seeds. **Fixed**: one-hour timeout, and `demo.sh` raises Claude Code's limit.
  [harness-fixes.md](harness-fixes.md) section 6.

### T4. A confident mask can still be the wrong object (12 November)

* With the exploration step, the task 0 agent chose "brown and orange cylindrical can" (0.83,
  runner-up 0.41): a clean mask on the tomato sauce. Scores cannot catch this.
* It was caught one version later by the check "the object is in the basket and the task still
  fails, so it is the wrong object", which the prompt's diagnosis order leads to.
* The alphabet soup is the smaller blue can; the tomato sauce is the red and green one.

### T5. The basket centre from the median of points is 5 cm off (three sessions)

* One camera sees mostly the basket's near wall, so the median of its points sits about 5 cm
  toward the camera. `skills/grasp.md` uses that median for a target's centre, and agents copy it.
  Result: the object lands on the near rim (task 1, version 2), or the held object touches the
  near wall and the lowering move stalls (task 0, version 2).
* `localize_object(...).geometry.pose.position` is within about 1 cm. The prompt now says so.
* **Task 0's registered program still uses the median** and releases from 8 cm above the rim.
  It works: 46 of 50, with the lowering move stalled in 4 of the 46 successes. It is the first
  thing to fix if task 0's teacher success matters more than it does today.

### T6. Other program-level observations

* The teacher's episode ends at the first success, usually while the gripper is still opening.
  Calls that were in progress are recorded as `terminated`; that is not a failure.
* cuRobo plans normally take 5–9 s. Under load they exceeded the 120 s read timeout twice
  (`provider_unavailable`). Programs that use cuRobo for one move (task 1) depend on a GPU that is
  not saturated.
* The quick agent costs \$4–7 per task and about 25 minutes of model time when it succeeds in
  three versions.

## Machine and services

### I1. The GPU memory reservation is not handed over on demand (12 November)

* `services.sh start` pauses the reservation, starts the services, and takes the reservation
  back sized to what the services use **while idle**, leaving 2 GiB. SAM3 holds 4.3 GiB idle and
  needs about 6–7 GiB while segmenting. Nothing shrinks the reservation during a run.
* On GPU 1, with other jobs filling the rest, SAM3 found 4 MiB free and every segmentation
  failed with `CUDA out of memory`. 16 teacher episodes died on it.
* **Workaround:** start the services with more headroom, which is an existing setting:

  ```bash
  ASPIRE_RESERVE_SIM_HEADROOM_GIB=8 scripts/gpu/services.sh start
  ```

* **Now:** episodes that fail on a service are not counted and are recorded again; `teacher-run`
  stops when a whole task fails on a service; `run-batch` prints "a model service did not
  answer".
* **Open:** the reservation still does not give memory back when a service needs it.

### I2. A busy machine makes everything slower, and CPU work dominates (12 November)

Measured per teacher episode:

| | Idle machine | Busy machine |
|---|---|---|
| Whole episode | about 31–35 s | median 69–91 s, up to 508 s |
| Grasp planner (Contact-GraspNet), one call | 2.3 s on the GPU | 16–42 s on the CPU |
| Video encoding (x264, two cameras) | under 1 s | 14–22 s typical, up to 173 s |
| cuRobo, one plan | 5–9 s | over 120 s (timeout) |
| 50 episodes, 2 workers | about 15 minutes | 31–44 minutes |

* The grasp planner runs on the **CPU** on cali because `ASPIRE_CPU_PROVIDERS=contact_graspnet`
  is set in the conda env. That was chosen when another user kept the GPU at 100% and the planner
  was slower there than on the CPU. Video encoding is always on the CPU.
* In the 12 November agent session another user's job held GPU 0 at 91–95% until 04:27; runs
  before that moment were about 10× slower than runs after it.
* **CPU load is not recorded** in the session's timeline, so the CPU side of this is inferred
  from timings, not measured.
* **How to check where the grasp planner should run.** This starts two temporary copies (GPU and
  CPU), times the same request on each, and stops them. It does not touch the running services:

  ```bash
  scripts/gpu/check_graspnet.sh
  ```

  ```bash
  echo "$ASPIRE_CPU_PROVIDERS"
  ```

  If the GPU is faster and the second command prints `contact_graspnet`, move it back to the GPU
  (not while a `teacher-run` is using the services):

  ```bash
  conda env config vars unset -n aspire ASPIRE_CPU_PROVIDERS
  ```

  ```bash
  conda deactivate && conda activate aspire
  ```

  ```bash
  scripts/gpu/services.sh stop
  ```

  ```bash
  ASPIRE_RESERVE_SIM_HEADROOM_GIB=8 scripts/gpu/services.sh start
  ```

* Task 1's program does not call the grasp planner at all, and task 0's uses its result only in
  the episodes that then fail (T1). Removing the call from task 0's program would save more time
  than moving the planner.

### I3. Which GPU each part uses

* The model services, `session.sh` and `teacher-run` follow the conda env (`ASPIRE_GPU`; switch
  with `scripts/gpu/use_gpu.sh N`).
* `t2v.sh` stages that simulate, train or evaluate default to GPU 0 (`--gpu N` or
  `ASPIRE_T2V_GPU`). `vla.sh` defaults to GPU 1 (`--gpu N` or `ASPIRE_VLA_GPU`).
* A CUDA error naming "GPU 0" inside a service means that service's own first device, which is
  the physical GPU it was placed on.

### I4. Bedrock's daily token quota (12 November)

* An agent finished its work and was then refused its closing message: HTTP 429, "Too many
  tokens per day". The session reported the stage as FAIL; it now reports WARN with the message.
* The agent runs at maximum effort with the 1M-token context, as the ASPIRE paper did.
* Only teacher *development* uses the LLM. Collecting teacher episodes, converting, training and
  evaluating use none.

### I5. Recording and reporting details

* A headless agent run that waits on a background batch ends and resumes; the turn count is now
  summed over the parts. Cost was always cumulative.
* Interrupting a stage is safe: `teacher-run`, `teacher-grade`, `convert` and `eval` keep what
  finished and continue when the same command is run again. `teacher-run` now prints one line
  per finished episode, also into the log that `t2v.sh logs` follows.

## Pipeline (teacher episodes → student training data)

### P1. Teacher episodes replay exactly (12 November)

* All 100 episodes of `preflight2` on the official scenes replayed with a joint difference of
  0.0 rad, and the replayed outcome equals the recorded one in all 100. Every success placed the
  requested object.
* A scene is reproducible from (suite, task, init mode, seed) **only in a freshly built
  environment**; a reused environment drifted by 0.18 rad on a contact-rich task. Every replay
  builds a new one.

### P2. Teacher on the official scenes (12 November)

| Task | Solved | Steps | Failures |
|---|---|---|---|
| 0, alphabet soup | 46 / 50 | 506–636, mean 550 | 4: planner grasp closed on nothing (T1) |
| 1, cream cheese | 50 / 50 | 412–450, mean 419 | none |

The ASPIRE paper reports about 98% for the suite, on its own seeded layouts.

### P3. Episode length against the student's limit

* The student has 280 steps. The prototype converted 466–534-step episodes to 190–208 steps at
  3×, a ratio of about 0.4.
* Task 0's current episodes are longer (506–636), so expect roughly 200–255 after conversion,
  with little margin for the longest. The converter records `within_student_limit` per demo;
  if demos exceed it, convert with `--speed 4` (163–177 steps in the prototype, floor about 160).

### P4. Facts that shaped the design

From [trace2vla-review.md](trace2vla-review.md):

* The 50 scenes of a LIBERO task differ by 0–4 cm. A different seed is nearly the same scene, so
  the main experiment must hold out a *layout*.
* "Swap" exchanges the target with one of five candidate objects; the official suite is one draw.
* LeRobot's trainer replaces the checkpoint's normalisation statistics with the dataset's; the
  pipeline keeps the checkpoint's.
* LeRobot's default LoRA adapts only the action expert, not the vision-language backbone where
  the grounding failure lives.
* `lerobot/libero` declares 10 fps although control runs at 20 Hz; the dataset writer matches it
  so the two can be merged.

### P5. Not yet verified

* Fine-tuning on the GPU, and whether it raises the student's success.
* The converter on any task other than task 0.
* The dashboard's Trace2VLA pages against a full experiment.

## Student (SmolVLA)

* **Baseline:** 0 / 1,500 on `libero_object_swap`; 462 / 500 on standard `libero_object` (paper
  96%).
* **How it fails:** it never lifts the requested object. In 5 of 10 tasks it lifts whatever sits
  where the target used to be, and places it in 4 of those; in 5 it lifts nothing.
* **Actions per prediction:** 10 of the 50 predicted, by measurement (78% / 93% / 83% for
  50 / 10 / 1 on standard `libero_object`).
* **Parity:** the student's observations, the simulator state and the policy's actions are
  identical between ASPIRE's adapter and LeRobot's own LIBERO environment.
* **Checkpoint:** `lerobot/smolvla_libero`, already fine-tuned on LIBERO. `smolvla_base` was
  trained on 6-joint arms and cannot command this robot.

## Process

* Read the recordings before changing code. Each of the teacher problems above looked like a
  motion or grasp bug from the outcome alone, and was something else in the recordings.
* Run one task first when a change is untested: a failed agent session costs \$4–7 and an hour.
* A result on the two preflight tasks shows the pipeline works. It does not show the student
  generalises; only the held-out-layout experiment does.
