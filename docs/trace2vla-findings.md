# Trace2VLA: findings log

Everything learned while building and running the project that is not obvious from the code:
what was measured, what went wrong and why, what was fixed, and what is still open. Each entry
says how sure it is. Add new entries at the top of their section, with the date.

The goal and status are in the [overview](trace2vla.md); the work ahead is in the
[roadmap](trace2vla-roadmap.md). Detailed evidence for the teacher-agent problems is in
[harness-fixes.md](harness-fixes.md), and for the design measurements in
[trace2vla-review.md](trace2vla-review.md).

Last updated: 12 November 2026 (evening).

## Open items at a glance

| # | Finding | Status |
|---|---|---|
| T1 | Grasps suggested by the grasp planner fail every time they are executed (12 of 12) | **Cause found, not fixed; avoided** (13 November): the teacher prompt (`scripts/gpu/prompts/llm_quick_demo.md`) now tells new programs to use their own top-down grasp for upright cans, boxes and bottles and not to call the planner. The harness fix is still open |
| P8 | Seeded layouts barely vary: wide objects sit at their region's centre for every seed | Variety comes from the sibling layouts (`libero_object_swap_alt1…4`), built 13 November |
| T4 | A confident SAM3 mask on the wrong object cannot be detected from scores | Open by nature. The agent must check the image |
| T5 | The basket centre from the median of points is 5 cm off; task 0's program still uses it | Program still fragile (4 of 50 lowering moves stalled, all 4 still succeeded) |
| I1 | The GPU memory reservation is not handed over on demand | Open. Workaround documented |
| I6 | An episode stopped by Ctrl-C was counted, and `teacher-grade` stopped on it | Fixed 12 November; rerun `teacher-run`, nothing else to redo |
| I2 | A busy machine makes episodes 3–15× slower; CPU load is not recorded | Open. How to check is documented |
| I4 | Bedrock's daily token quota | Outside the repository. Teacher collection and all student stages use no LLM |
| S1 | Fine-tuned SmolVLA: 0% → 86% on the swap suite, 89% → ~0% on the standard layouts, forgetting complete by step 500 | Open: most likely a position shortcut; next: replay mixing, LoRA, π0.5 for comparison |
| P10 | The preflight's teacher programs do not transfer to the new layouts: in the pilot (13 November) task 0 solved 0 of 12 episodes on alt1–4 (3 of 3 on the official partner's layout), task 1 failed on alt2; perception was right in all 30 | **Partly fixed:** an agent developing against all five layouts solved 4 of 5 for task 0 (held-out 8/10); alt1 fails because a finger lands on the neighbouring milk carton (not reach, as the agent thought). Prompt now explains stalls; repair mode (`agent-fix`) to fix alt1 from the current program |
| P6 | The dataset step rejected successful demos for a state drift of 1–5×10⁻⁵ (`d50`: 77 of 98 kept, all 21 rejections in task 1) | **Fixed 12 November** (cause: float32 rounding of the stored actions). A dataset built before the fix is valid, only smaller; the preflight trains on `d50` as it is |

## Teacher (the ASPIRE agent and its programs)

### T1. Grasps from the grasp planner fail (12 November; cause found, not fixed)

**In short.** Every grasp suggested by the grasp planner (Contact-GraspNet) that a program
executed closed on nothing: 12 of 12. The programs' own top-down grasp succeeded every time. The
recordings point to two defects in how the harness calls the planner, both present in the
original ASPIRE v1.0 release. Neither blocks Trace2VLA: teacher episodes that used a planner
grasp all failed, and only successes become training data.

**What the recordings show.** Every planner call in the recordings, read for this entry:

| Where | Object | Calls | Candidates | Score | Approach, degrees from straight down | Executed | Result |
|---|---|---|---|---|---|---|---|
| Agent session `20261112-033519`, task 0 v01 | tomato sauce can (the agent's wrong pick) | 3 | 15 | 0.15–0.29 | **127–152**: pointing up, from under the can; contact points 4–5 cm below the can's centre, at table height | 0 | none passed `top_down`; the program used its own grasp |
| Same session, task 0 v02–v03 and held-out | alphabet soup can | 8 | 25 | 0.15–0.23 | 55–112, mostly horizontal | 0 | none passed `top_down` |
| Same session, task 1 v01 | cream cheese box, 2 cm tall | 3 | 1 | 0.15 | 40 | 0 | two calls returned no grasp at all |
| Experiment `preflight`, official scenes, task 1 | cream cheese box | 9 | | 0.15–0.17 | 11–36 | 9 | **8 closed on nothing, 1 out of reach** |
| Experiment `preflight2`, official scenes, task 0, all 50 episodes | alphabet soup can | 50 | 152 | 0.15–0.28, median 0.17 | 24–127, median 93: **89 from below**, 57 from the side, 6 within 37° of straight down | 4: the only 4 calls with a top-down candidate (seeds 19, 28, 32, 33) | **4 closed on nothing**; the other 46 episodes used the program's own grasp and all succeeded |
| Experiment `preflight2`, training layouts, task 0 | alphabet soup can | 2 failures | | | | | not read; probably the same |

* `select_grasp(strategy="top_down")` accepts a candidate whose approach is within 37° of
  straight down (`vertical_threshold=0.8`). Most candidates were far outside that, so the
  programs fell back to their own grasp, and that is why the teacher still works.
* **Every candidate's position is 0.103–0.109 m from its own contact point**: 41 of 41 in the
  session and 152 of 152 in `preflight2`, whatever the object. That is Contact-GraspNet's gripper depth (0.1034 m from the
  gripper base to the fingertips) plus half the finger opening.
* Executed candidates stopped about 10 cm above the object and closed on air.
* In `preflight2` the planner decided task 0's outcome exactly: the 4 episodes in which it
  offered a top-down candidate are the 4 failures, and the 46 in which it did not all succeeded.

**Cause 1, likely: the planner is run in the robot's base frame.** `generate_grasps`
(`src/cap_harness/api.py`) projects the object and the whole scene into the robot base frame
and sends both. The grasp service (`src/cap_harness/providers/graspnet/service.py`) runs the
network on the points as received. Contact-GraspNet was trained on point clouds in a camera
frame, with z pointing from the camera into the scene. In the base frame z points up, so the
network sees the scene as if from a camera under the table. That matches what it proposes:
grasps reaching up from under the tomato can, side grasps on the soup can, nothing on the flat
box, all with low scores. The service does try other viewpoints, but only when the first pass
returns nothing, and its virtual cameras are placed within 25 cm of the robot base's origin,
not around the object.

**Cause 2, likely: the pose is the gripper base, used as the fingertip point.** A public
`Pose` addresses the point between the fingertips (`robot0_eef`). The IK service moves 0.1 m
back from it to the hand (`PANDA_EEF_TO_HAND_OFFSET_M` in
`src/cap_harness/providers/pyroki/client.py`). Contact-GraspNet's pose is the gripper base, so
the fingertips end about 10 cm short of the contact point. The client corrects only the
rotation of the fingers about the approach axis (`_LOCAL_Z_90_DEG`).

**What was tested.** The four failed candidates of `preflight2`, moved 0.1034 m along their
approach direction and executed in the simulator (CPU, PyRoki IK). The hand then reached the can,
and still closed on nothing in all four; two moves stalled against the can. So fixing the
position alone is not enough: the candidates themselves are poor, which fits cause 1.

**Both are in the original release.** The base-frame request and the service's first pass and
retry are in ASPIRE v1.0 (commit `fa00f0e`). This fork changed only the service's logging and
its loading on a CPU. The ASPIRE paper's teacher results may therefore rest on programs that
also fell back to their own grasps; do not claim anything about that without checking.

**What it costs today.**

* 0.3 s per call on an idle GPU, 2.6 s on the CPU, up to 42 s on a loaded machine, for a
  candidate that is almost never used and fails when it is.
* Agent versions: the session of 11 November lost a version to `select_grasp("highest_score")`
  candidates the arm could not reach, and its report proposed avoiding them.
* About 4–8% of teacher episodes on the official scenes (4 of 50 in task 0, 8 of 50 in task 1
  with the earlier program).

**What to do, in order of cost.**

1. **Now, free:** teacher programs for cans and boxes use their own top-down grasp at the
   object's geometry centre and do not call the planner. The current task 1 program already
   does; task 0's still calls it and uses a candidate when one passes. The agent prompt does not
   say this yet.
2. **Fix, small:** in `generate_grasps`, express both point clouds in the camera's own frame
   (OpenCV convention, z into the scene) before calling the planner, transform the grasps back,
   and move each pose 0.1034 m along its approach so that it addresses the fingertip point.
   Then rerun the four failed seeds and about 20 scenes of tasks 0 and 1, and compare the
   approach angles, the scores and the empty-grasp rate with the table above.
3. Keep the published behaviour selectable (`CAP_HARNESS_COMPAT=published`), as for the other
   harness fixes, so paper comparisons stay like for like.

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

### I6. An episode stopped by Ctrl-C was counted as finished (12 November, fixed)

* **What happened.** The first `teacher-run --scenes train` was stopped with Ctrl-C while
  seed 1011 of task 0 was still resetting its scene. Ctrl-C reaches every process of a foreground
  command, so the episode's own process caught it, wrote its outcome as `user_interrupt` and
  marked the run finished, but it had no reset state yet (`episode/reset.json`). The pipeline
  counted it: the rerun skipped seed 1011, the table showed task 0 at 47 / 50 with one
  `user_interrupt` "failure", and `teacher-grade` stopped at that episode with
  `No such file or directory: .../1011/run-20261112T170526.../episode/reset.json`.
* `t2v.sh stop` (used for the `--detach` run) does not cause this: it ends the processes
  without finalizing their run, so the half-run episode is ignored and recorded again.
* **Fixed.** An episode that ended as `user_interrupt` or `harness_error`, or whose recording
  lacks `reset.json` or `steps.jsonl`, is not counted; `teacher-run` records that seed again and
  the teacher table lists it as "not counted: ... stopped before they ended". `teacher-grade` and
  `convert` skip a recording they cannot read, say which, and carry on with the rest.
* **Nothing has to be redone.** Rerunning `teacher-run` records only the missing seed;
  `teacher-grade` keeps the grades already written.

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

### P2b. Teacher on the training layouts (12 November)

| Task | Solved | Steps | Not solved |
|---|---|---|---|
| 0, alphabet soup | 48 / 50 (seed 1011 recorded again after I6, and solved) | 499–577, mean 537 | 2 program errors, cause not yet read |
| 1, cream cheese | 50 / 50 | 413–427, mean 420 | none |

* This run was about 3× faster than the run on the official scenes: 38 episodes in 586 s and
  50 in 605 s with two workers, against 50 in 2,663 s and 1,848 s. The machine was less loaded;
  `check_graspnet.sh` that evening measured the grasp planner at 0.3 s on the idle GPU against
  2.6 s on the CPU.

### P2. Teacher on the official scenes (12 November)

| Task | Solved | Steps | Failures |
|---|---|---|---|
| 0, alphabet soup | 46 / 50 | 506–636, mean 550 | 4: planner grasp closed on nothing (T1) |
| 1, cream cheese | 50 / 50 | 412–450, mean 419 | none |

The ASPIRE paper reports about 98% for the suite, on its own seeded layouts.

### P3. Episode length against the student's limit

* The student has 280 steps. The prototype converted 466–534-step episodes to 190–208 steps at
  3×, a ratio of about 0.4.
* **Measured on `preflight2`, training layouts, 3× (12 November):** **98 kept of 98 teacher
  successes, 0 rejected**. Task 0: 48 demos, 188–216 steps from a teacher mean of 537. Task 1:
  50 demos, 162–174 steps from a teacher mean of 420. None over the 280-step limit, so 3×
  leaves a margin of more than 60 steps and `--speed 4` is not needed.
* Idle frames dropped per demo: about 6 (task 0) and 4 (task 1). In one demo per task, dropping
  them made the stored sequence fail on replay, so all were kept ("kept all idle frames").
* Tracking lag, the distance between the re-executed hand and the teacher's recorded hand path:
  about 11 mm throughout.

### P6. Successful demos rejected for a tiny state drift (12 November, fixed)

**Seen.** Building `d50` on `preflight2`, the dataset step rejected demos with
`rejected task_01/seed_1002: success True, state drift 3.40e-05`. Final count: **77 kept of 98,
21 rejected, all from task 1** (task 0: 48 kept, task 1: 29 kept, 14,584 frames). Every
rejected replay solved the task; their drift was 1.1e-5 to 5.5e-5. Of the 77 kept, 44 also
drifted, by less than 1e-5. The rule was: reject when the drift is above 1e-5.

**Decision for the preflight:** train on this `d50` (77 demos) now, so SmolVLA and π0.5 learn
from the same frames (π0.5 gets a copy with its own statistics); build the full 98-demo dataset
with the fix for the experiments that follow. Task 1 has fewer demos (29 against 48), which
the per-task results should be read against.

**Cause.** The converter's final replay played the actions in float64 and recorded the states
of that replay, but `steps.npz` stores the actions as float32. The dataset step replays the
stored float32 actions. The simulator itself is deterministic (measured on two demos: the same float32
actions replayed twice in fresh environments differ by 0.0), so the rounding, about 1e-8, is the only
difference between the two replays; contact amplifies it to 1e-5–5e-5 in task 1's demos.

**Effect on data.** None on correctness: the dataset's frames always come from its own replay
of the stored actions, so the frames and actions written match exactly, and a rejected demo is
just left out. A dataset built before the fix is valid, only smaller.

**Fix.** The converter now replays exactly the float32 actions it stores, so its recorded
states match any later replay bit for bit. The dataset step accepts a successful replay of the
full length whose states are within 1e-3 (1 mm, 1 mrad) of the stored ones, far above
rounding and far below what a wrong scene, controller or code version gives, and records the
reason for each rejection. Demos converted before the fix pass too; nothing has to be converted
again. Tests: `tests/test_trace2vla_replay_precision.py`.

### P7. Training progress showed 1000, 1000, ..., 2000 (13 November, fixed)

LeRobot prints the step rounded to thousands from step 1000 on (`step:1K`, and 1500 already
as `2K`, 2500 still as `2K`), and the training wrapper read that literally: the log, the progress
and the dashboard showed `step 1000/3000` for steps 1000-1450 and `2000/3000` from 1500. The
training itself was not affected (the learning rate kept falling step by step). The wrapper
now takes the previous step plus the logging interval whenever that agrees with the printed
value, and the dashboard repairs `metrics.jsonl` files written before the fix when it reads
them. Tests: `tests/test_training_steps.py`.

### P8. Seeded layouts barely vary; the sibling layouts do (13 November)

**Measured** in the simulator, `libero_object_swap_alt1` task 0, seeds 7, 8, 9 and 1001: the
alphabet soup and the salad dressing stood at exactly the same spot for every seed (to 0.1 mm);
the milk moved by about 5 mm. Each object's starting region is a 5 × 5 cm square, and objects
about that wide end up at its centre. The demos of `preflight2` differ by 1–1.5 cm between
seeds (start states), the same picture. So 50 seeds of one layout are close to one scene, and
the 50 official scenes of a task differ by centimetres too (P4).

**Consequence:** variety for training has to come from layouts. LIBERO-Pro draws each task's
swap partner from five candidates; the other four give layouts with the target about 20 cm
elsewhere, same instruction, same objects. They are generated with LIBERO-Pro's own rule (it
reproduces the ten official swap files byte for byte) into `configs/libero_layouts/` and
registered as suites `libero_object_swap_alt1…alt4` (`src/cap_harness/libero/layouts.py`).
Checked: in `alt1` task 0 the soup stands where the salad dressing stands in the standard
layout (within 1 mm), 22 cm from its own place, and the instruction is word for word the
official one. See [trace2vla-full-run.md](trace2vla-full-run.md).

### P9. A re-execution that runs into the simulator's limit stopped the conversion (13 November, fixed)

The converter's tracking loop allowed 1,500 steps; the simulator ends its episode at 1,000 and
then raises `executing action in terminated episode` on the next step. A teacher episode whose
re-execution had not succeeded by then stopped the whole `convert` command (seen with a recorded
episode replayed from another seed's starting state). The loop now stops there and the demo is
rejected as `re-execution_failed`, like any other failed re-execution. Conversion and grading also
report a failing episode and go on with the others (`--workers`, P9's test run).

### P10. Programs written on the official layout fail where the target has moved (13 November)

The pilot (`object_collection.sh pilot`: tasks 0–1, the five layouts, 3 scenes each, the
preflight's programs) recorded 30 episodes. The teacher found the right object in all 30 (graded
`chosen` is the target every time), so perception is not the problem. What failed is physical:

| Layout | Task 0 (alphabet soup) | Task 1 (cream cheese) |
|---|---|---|
| official partner's (`train`) | 3/3 | 3/3 |
| alt1 | 0/3: "grasp descent failed": the soup stands between other objects and the open gripper stops on one | 3/3 |
| alt2 | 0/3: lifted, then dropped 3 cm from where it stood | 0/3: never lifted (one "closed on nothing") |
| alt3 | 0/3: lifted, then dropped | 3/3 solved, all 3 rejected at conversion (below) |
| alt4 | 0/3: lifted, then dropped | 3/3 |

Task 0's program grasps straight down at the soup's point-cloud median plus 1 cm with a fixed
gripper yaw (its grasp-planner call returns candidates below the table, T1, so it always used
this fallback). On the layout it was written on the gripper closes to 0.60 and holds the can; on
alt2–4 it closes to 0.64–0.70 and the can slips out in transit. The program was debugged on one
layout only (development seeds 51–53 of the official swap), so nothing pushed it to work where the
target stands elsewhere. Evidence: `object-pilot/teacher/*/task_00/*/run-*/source/program-result.json`,
`trace/calls/call-000024-move-to-pose/input.json` (the grasp pose), `close-gripper/output.json`
(`gripper_position`), `grades/*/task_00/seed_*.json` (`target_lifted`, `moved_m`).

Alt3, task 1: the teacher solved all 3, but the conversion's 3× re-execution lost the cream cheese
(8 mm mean tracking lag, 21 mm worst). At 2× it failed too; at 1× two of three passed but took
395–400 steps, over the student's 280. The teacher's grasp there is marginal: it works only when
the hand follows the recording within about a millimetre. `convert --fallback-speeds 2` now
exists (each demo records its speed), off by default.

Response: the coding agent develops one program per task against all five layouts. `demo.sh llm
--layouts all` (and `session.sh --llm-layouts all`) runs every development and held-out batch on
each layout (`scripts/gpu/layouts_batch.py`, one combined `batch.json` with each layout's result)
and tells the agent that one program must solve all of them and never hard-code a position. One
agent per task rather than one per (task, layout): ten sessions instead of fifty, one program per
task, and a teacher that does not need re-programming for each placement. A task whose program
still fails on some layout can get a per-layout agent afterwards.

#### P10, the pilot with the repaired program (14 November)

`object_collection.sh pilot` with task 0's repaired program: **15/15 on all five layouts**, alt1
included (it chose the 90-degree yaw from the neighbours, corrected the centre by 1.5 cm toward
the far side, and needed no retry). Task 1 still ran the preflight's program (alt2 0/3; alt3's 3
successes rejected at conversion) and gets a new agent in the full run.

Alt1's three demos came out at 283-293 steps, over the student's 280: the teacher episodes there
are 771-803 steps (the wrist turn makes each move slow at the arm's fixed joint speed), and 3x
replay is not enough. Fix: `convert` now tries 4x, then 5x, for a demo over the student's step
limit and keeps the first that still solves the task and fits (`--limit-speeds`, default 4,5;
each demo records its speed and tries). Checked on the pilot's alt1 episodes: 233-241 steps at
4x, all three kept. A tracker change made at the same time (a wrist turn counts as motion, and
the playhead waits for a lagging turn) left every pilot demo's length unchanged: the hand moves
during these turns, so they were already played at 3x.

#### P10, alt1 settled by replaying the contacts (13 November, late)

The repair agent (`agent-fix 0`, 32 min, \$17.55) turned the gripper 90 degrees as advised and
still failed alt1 (held-out 8/10 again). Replaying its recorded alt1 run in the simulator and
listing every robot-object contact gave the cause, which neither the agent nor the images
showed:

* default yaw: the first contact is the **hand** (`gripper0_hand_collision`) on the milk carton
  at 8.8 cm height, then a finger; the carton stands 10.8 cm from the soup along the finger
  direction, inside the hand's reach;
* yaw 90: the milk is clear, but a finger lands **on the soup itself**. The program's centre
  (`geometry.pose.position`) is 1.6 cm toward the camera from the true centre (0.667 vs 0.651 m):
  the camera sees only the near half of the can. Across the line of sight (default yaw) that does
  not matter; along it, the open gripper leaves about 7 mm each side of the can.

So alt1 is solvable with both a turn and a centre corrected away from the camera. The agent
prompt now states these facts (hand width, the camera-side shift and its correction, yaw within
±90 degrees, which also cost the agent a version). Tools: `stall_contacts.py`-style replay of a
recorded run with the simulator's contact list (analysis only, never given to the agent).

#### P10, continued: one agent against all five layouts (task 0, 13 November)

`object_collection.sh agent-test 0`: one agent, every batch on the five layouts (development
seeds 51–53, held-out 1–2 on each), 26 minutes, 191 turns, \$15.75.

| | official | alt1 | alt2 | alt3 | alt4 |
|---|---|---|---|---|---|
| preflight program (pilot) | 3/3 | 0/3 | 0/3 | 0/3 | 0/3 |
| v01, development | 3/3 | 0/3 | 3/3 | 3/3 | 3/3 |
| v03 (a "disambiguation" that picked wrong objects) | 2/3 | 0/3 | 0/3 | 0/3 | 3/3 |
| best (v01), held-out | 2/2 | 0/2 | 2/2 | 2/2 | 2/2 |

Developing against every layout fixed three of the four layouts the old program failed on,
in its first version. Its remaining versions all went to alt1, and none fixed it, because the
agent misread the cause: it reported a reach limit at x = 0.667 m. The trace says otherwise: the
inverse kinematics for the grasp pose solved (`call-*-pyroki-solve-ik`, `ok`), and the descent
stalled 0.036 rad short (`stalled: true`). The wrist camera at the stall shows why: in alt1 the
soup stands between the milk carton and a tall bottle, the open fingers span the image's
left-right direction, and that direction points at the carton; the finger comes down on it.
Front and back of the can are free. Turning the gripper 90 degrees about the vertical axis lets
the fingers straddle the can without touching a neighbour. Evidence: the held-out runs
`heldout/libero_object_swap_alt1/.../run-*/media/keyframes/00000{4,6}-after-move-to-pose/robot0_eye_in_hand.png`
and `trace/calls/call-000019-move-to-pose/output.json`.

Responses:

* The agent prompt now says how to read such a stall: reachable (the IK solved) means contact;
  look at the wrist image after the move; turn the gripper about the vertical axis so the
  fingers open toward free table, chosen from where the neighbours are, never per layout.
* Repair instead of rewrite: `demo.sh llm --start-from PROGRAM` (`session.sh --llm-start-from`,
  `object_collection.sh agent-fix TASK`). The existing program runs first as version 1 (no new
  program written); the agent then fixes what fails and keeps every scene it solves. This is the
  incremental way to extend a program to a new layout or to repair one layout: the LLM is paid
  only for what fails, and the program stays one program for every layout.

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
* The converter on tasks other than 0 and 1.
* The dashboard's Trace2VLA pages against a full experiment.

## Student (SmolVLA)

* **Baseline:** 0 / 1,500 on `libero_object_swap`; 462 / 500 on standard `libero_object` (paper
  96%).
* **Before training, `preflight2` official scenes (12 November):** 0 / 100 (tasks 0 and 1,
  seed 0). Graded failures: 92 `semantic` (another object placed in the basket) and 8 `both`
  (another object, not placed). Task 0 placed the butter.
* **The same two tasks on the standard layouts** (`before-standard`, 12 November): **89 / 100**
  (task 0: 42 / 50, task 1: 47 / 50); all 11 failures `physical` (the right object, not
  placed). The instruction and the objects are identical to the swap suite; only the layout
  differs. This pair of runs, 89% against 0% with 92 of 100 placing the wrong object, is the
  clearest statement of the problem the project attacks.
* **How it fails:** it never lifts the requested object. In 5 of 10 tasks it lifts whatever sits
  where the target used to be, and places it in 4 of those; in 5 it lifts nothing.
* **Actions per prediction:** 10 of the 50 predicted, by measurement (78% / 93% / 83% for
  50 / 10 / 1 on standard `libero_object`).
* **Parity:** the student's observations, the simulator state and the policy's actions are
  identical between ASPIRE's adapter and LeRobot's own LIBERO environment.
* **Checkpoint:** `lerobot/smolvla_libero`, already fine-tuned on LIBERO. `smolvla_base` was
  trained on 6-joint arms and cannot command this robot.

### S1. First fine-tuning: it learns the swap and forgets the standard layouts (13 November)

`ft-full`: the whole model (the checkpoint trains the vision-language part too:
`train_expert_only` and `freeze_vision_encoder` are false), 3,000 steps, batch 32, lr 5e-5, on
`d50` (77 teacher demos: task 0 48, task 1 29). `ft-curve`: the same, saving every 500 steps,
with 20-episode probes per step. Numbers as of 13 November, about 01:00; the two marked
running were not finished.

| Evaluation | Suite, scenes | Before | After |
|---|---|---|---|
| `before-official` → `after-official` | swap, official (100) | 0 / 100 | **86 / 100** (14 `physical`) |
| `before-standard` → `after-standard` | standard, official (100) | 89 / 100 | **0 / 77** running (73 `both`, 4 `semantic`) |
| curve, swap `dev` probe (20) | step 0 / 500 / 1000 / 1500 / 2000 | 0 | 6, 9, 9, 19 |
| curve, standard probe (20) | step 0 / 500 / 1000 / 1500 | 17 | 0, 0, 0 |

* **The forgetting is total by step 500**, the first checkpoint: there is no step that keeps
  both. Choosing an earlier checkpoint does not help.
* **How it fails on the standard layouts:** almost all `both` or `semantic`, i.e. the *wrong
  object*, not clumsy grasps. Its manipulation is intact (86% on the swap suite); what changed
  is which object it goes for.
* **Most likely reading (to confirm on the videos):** with two tasks whose layouts differ by
  centimetres, the instruction alone tells where to reach. The model learned "this instruction:
  go to the swapped position" instead of "find this object". On the standard layouts that
  position holds the other object. The 86% on the swap suite then measures re-learned positions,
  not object grounding. How to confirm: watch `after-standard` episodes; the hand should go to
  where the target sits in the *swap* layout.
* **Consequences:** more swap-only demos (98 instead of 77) will not change this. What can:
  data where the same instruction needs different positions, i.e. the original LIBERO
  demonstrations mixed in (replay, roadmap S2) or teacher demos on the standard layouts;
  training only part of the model (LoRA on the action expert keeps the vision-language part);
  and an evaluation on layouts neither seen in training nor in the swap draw (held-out pairs).
* **π0.5 before training on the swap suite** (`pi-before-official`, running): 56 / 96, failures
  20 `semantic`, 20 `physical`. Far above the 17% the LIBERO-PRO paper reports for openpi's own
  π0.5 checkpoint; this is LeRobot's checkpoint. π0.5 already finds the swapped object about
  half the time, where SmolVLA never does.

## Process

* Read the recordings before changing code. Each of the teacher problems above looked like a
  motion or grasp bug from the outcome alone, and was something else in the recordings.
* Run one task first when a change is untested: a failed agent session costs \$4–7 and an hour.
* A result on the two preflight tasks shows the pipeline works. It does not show the student
  generalises; only the held-out-layout experiment does.
