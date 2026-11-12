# Trace2VLA: roadmap to the paper

What is left to do, in order, with the gate that ends each step; the experiments a paper needs;
the risks; and the decisions that are still open. The goal and current status are in the
[project overview](trace2vla.md). What may be claimed is in
[related work](trace2vla-related-work.md).

Last updated: 12 November 2026. Update the status columns when a step finishes.

## 1. The order of work

| Step | What | Gate | Status |
|---|---|---|---|
| **A** | **Preflight on two tasks** (`libero_object_swap` 0 and 1): teacher on the official scenes, teacher demos on training layouts, convert, dataset, student before, fine-tune, student after, report. Commands: [trace2vla-preflight.md](trace2vla-preflight.md) | The loop closes once on the GPU. Teacher at or near 90% on `official`; most successes convert; student after training above 0% | Running. Teacher on `official` 96 / 100, on training layouts 98 / 100; conversion at 3× keeps every success so far, all under 280 steps. Next: dataset, fine-tune, student after |
| **B** | **Read the preflight** and fix what it shows (table in section 2) | A written decision on training method, steps, demo count and conversion speed for step D | |
| **C** | **Build what the full run needs** (section 3) | Each item has a test | |
| **D** | **Full M1 run on ten tasks**: same-layout and held-out-layout experiments, retention, three training settings, demo-count curve | The M1 tables and figures in section 4, with three training seeds | |
| **E** | **M2**: failure-aware training and its ablation | The grounding-vs-manipulation table | |
| **F** | **M3**: one persistent student over a task sequence, with the gate | The forgetting matrix | |
| **G** | **Breadth**: a second suite (`libero_goal_swap`), a second student as a check, LIBERO-Plus | Optional; strengthens the paper | |

M4 (the agent proposes tasks, on-device updates) is out of scope for a first paper.

## 2. Reading the preflight (step B)

| What the preflight shows | What it means | Next action |
|---|---|---|
| Teacher well below 90% on `official` | The student cannot learn what the teacher fails | Improve that task's program first: rerun the agent with more versions (`session.sh --llm-max-versions`) |
| Many demos rejected, or idle frames kept for most | The converter needs tuning for that object | Try `--speed 2`; check the tracking lag on the Demos tab |
| Student after training well above 0% on `official` | The pipeline works end to end | Go to step C |
| Student still 0%, failures still `semantic` | It did not relearn which object to take | `--method lora-vlm` or full fine-tuning with more steps; more demos |
| Student improves, standard suite drops | Forgetting | Mix original LIBERO data into training (replay) |
| Student improves on the two tasks | **This shows the pipeline works. It does not show generalisation**, because the 50 scenes of a task differ by centimetres | The held-out-layout experiment in step D is the real test |

## 3. What still has to be built (step C)

| Item | Why | Notes |
|---|---|---|
| **Sibling-layout scene sets** | The main experiment trains on layouts the test never shows | Generate the other swap partners with LIBERO-Pro's own tool (five candidates per task in `libero_ood/ood_spatial_relation.yaml`; the official suite is one draw). Add them as scene sets with role `train` |
| **Replay mixing** | Retention | Merge the teacher dataset with `lerobot/libero` (`lerobot-edit-dataset`, operation `merge`); the dataset writer already matches its feature names, image size and declared fps |
| **Converter checked on all ten tasks** | It is validated on task 0 only | Thin or tall objects may need slower contact phases. Record the yield per task |
| **Teacher programs for all ten tasks** | Only tasks 0 and 1 exist | About \$4–7 of LLM cost per task in the last session. Consider the full campaign (`demo.sh campaign`), which also grows the skill library |
| **More demonstrations per task** | SEAR: under 50 demos about 21%, 300 or more about 94% | Add scene sets of 200 seeded layouts; teacher episodes cost about 30 s of simulator time each and no LLM money |
| **Accept-or-roll-back gate and checkpoint registry** | M1's update loop and M3 | Each checkpoint records its parent, its dataset and its evaluation; a candidate replaces the current student only if it is not worse on the development scenes |
| **Failure labels from the agent's perception** | M2 must not train on simulator ground truth | Today's labels come from offline grading with simulator state. Keep those as the oracle and add labels the agent derives from camera images |
| **Recording the student's failure state for teacher takeover** | M2's physical corrections | The adapter can start from a saved state; the student runner already stores start and end states |
| **A round script** | M3 | One command that runs evaluate → collect → convert → dataset → train → evaluate → gate, reading the ledger |
| **A second student** as a sanity check | Confirms the setup against published LIBERO-Pro numbers | `lerobot/pi05_libero` should reproduce about 0.17 (object) and 0.38 (goal) if the setup is right. Planned and not yet run |

Teacher quality ([findings T1](trace2vla-findings.md)): every grasp the grasp planner
suggested that a program executed closed on nothing (12 of 12), while the programs' own
top-down grasps succeeded. The likely causes are in the harness, from the original release:
the planner runs on points in the robot base frame instead of a camera frame, and its pose
(the gripper base) is used as the fingertip point. A fix is small (findings T1, "What to do");
until then, teacher programs should use their own top-down grasp.

Smaller items: record CPU load in the session timeline (a busy machine made runs 10× slower and
the cause had to be inferred); a faster video preset for development runs; keep teacher episodes
in which the program retried and recovered, and mark them (Scaling Up & Distilling Down found
they add about 35 points).

## 3b. π0.5 as a second student (readiness check and implementation, 12 November)

**Status: implemented, not yet run on the GPU.** Every command that loads a policy takes
`--student smolvla|pi05` (default SmolVLA, whose behaviour, run names, protocol keys and outputs
are unchanged). π0.5 runs in its own `.venv-pi05`, built by `scripts/gpu/vla.sh setup-pi05`,
and a command moves itself there. Datasets carry their student's statistics
(`t2v.sh dataset --from-dataset d50 --student pi05` copies a SmolVLA dataset), records name
their student, the report and the dashboard show it, and `vla.sh label-students` writes it into
records from before. Tests: `tests/test_students.py`, `tests/test_students_pipeline.py`,
`tests/test_label_students.py`. Built and tested on CPU: the environment (same shared package
versions as `.venv-libero`), the routing between environments, the dataset copy, the training
command, the reports and the dashboard. Not yet done: a π0.5 policy load and evaluation on the
GPU, and measured memory. Commands:
[trace2vla-preflight.md](trace2vla-preflight.md#π05-the-same-experiment-with-the-second-student);
design: [vla-student.md](vla-student.md#two-students-smolvla-and-π05).

The readiness check, done before writing any code (nothing here changes the SmolVLA setup):

| Question | Answer |
|---|---|
| Checkpoint | `lerobot/pi05_libero_finetuned_v044`: LeRobot's own, made for LeRobot v0.4.4 (our version), fine-tuned on LIBERO (`HuggingFaceVLA/libero`, 6,000 steps, batch 32, lr 2.5e-5). 7.5 GB, bfloat16, not gated. LeRobot reports 99% on standard LIBERO-Object with it. A quantile-normalised variant also exists |
| Same data format as our demos? | **Yes.** Cameras `observation.images.image` / `image2` (LIBERO's own names, no renaming), 8-number state, 7-number action. The 98 converted demos can be reused; only the normalisation statistics differ (the checkpoint's mean and standard deviation instead of SmolVLA's) |
| Actions per prediction | 10, as LeRobot and openpi use for it; the same as our protocol |
| Can it share `.venv-libero`? | **No.** LeRobot v0.4.4's π0.5 needs `transformers` from the branch `fix/lerobot_openpi` (version 4.53.3, `tokenizers` < 0.22). SmolVLA's environment has `transformers` 4.57.6 and `tokenizers` 0.22.2, and LeRobot's SmolVLA extra requires `transformers` ≥ 4.57.1. Installing π0.5 there would downgrade both |
| A separate environment | **Works.** A dry run and a CPU install in a throwaway environment: LeRobot 0.4.4 with the `pi` extra, the transformers branch, torch 2.9.1, numpy 1.26.4, mujoco 3.3.0, robosuite 1.4.0 and peft 0.21.2 resolve together when the repository's own overrides are used with the transformers line replaced. LeRobot's check for that branch passes, the π0.5 and SmolVLA modules import, the checkpoint's config loads, and LeRobot has default LoRA targets for π0.5. Pin `pandas` to the SmolVLA environment's 3.0.3: the newest pandas fails to import with numpy 1.26.4 |
| Must not touch | `pyproject.toml` and `uv.lock`: relocking could move SmolVLA's environment. The π0.5 environment is installed by its own script with its own pins |
| Gated download | The PaliGemma tokenizer (`google/paligemma-3b-pt-224`) is gated: the Hugging Face account whose token is on the server must accept its licence. The weights are under the Gemma licence |
| GPU | Same torch build (2.9.1, CUDA 12.8) as SmolVLA, which already runs on the RTX PRO 6000 (Blackwell). The transformers branch is pure Python. The checkpoint asks for `torch.compile`; turn it off for evaluation if it is slow to start or fails |
| Memory (estimates, to be measured) | Evaluation about 10 GB. LoRA training with gradient checkpointing roughly 20–35 GB. Full fine-tuning roughly 50–70 GB: an otherwise empty 96 GB card |
| Code changes | Additive: a setup command for `.venv-pi05`; the dataset writer and trainer take the student checkpoint as a parameter (statistics source, camera renaming only when the checkpoint needs it, learning rate); reference numbers for π0.5 in the reports. SmolVLA stays the default and its tests are unchanged. **Done as planned**, with one addition: the student registry `src/cap_harness/vla/students.py` holds all of these per-student settings in one place |

Order: after the SmolVLA preflight. First π0.5 before training on the same 100 official-scene
episodes and the standard layout (it should land near the LIBERO-PRO paper's 17% on the swap
suite and near 99% on the standard one), then LoRA fine-tuning on the same demos.

## 4. Experiments for the paper

The proposal's matrix, extended with what the survey and the measurements added.

| ID | System | Data and training | Answers | Status |
|---|---|---|---|---|
| **T0** | ASPIRE teacher | none | Teacher success and cost on the official scenes | Tasks 0 and 1: 46 / 50 and 50 / 50. Eight tasks have no program yet |
| **S0** | Frozen SmolVLA (`smolvla_libero`) | none | The student's starting point | **Done: 0 / 1,500** on `libero_object_swap`; 462 / 500 on standard |
| **S1** | SmolVLA, success only | 25 / 50 / 100 / 200 verified teacher demos per task | Data efficiency (H1) | Preflight covers 50 demos on two tasks |
| **S1-layout** | SmolVLA, success only | trained on sibling layouts only, tested on `official` | **Main result:** did it learn to find the named object, or a position? | Needs sibling scene sets |
| **S1-method** | SmolVLA | default LoRA vs LoRA with the vision-language backbone vs full fine-tuning | Where the grounding failure is fixed | Trainer supports all three |
| **S2** | SmolVLA + replay | teacher data plus original LIBERO data | Retention on the standard suite | Needs replay mixing |
| **S3** | Fixed-data baseline | the same demonstrations, trained once | Does updating step by step add anything? | |
| **J** | Joint upper bound | all tasks together | The ceiling for the sequential student | |
| **M2-a…e** | SmolVLA | success only vs + generic failure replay vs + physical corrections vs + semantic corrections vs both, at matched demo and step budgets | Does failure decomposition matter (H2)? | Labels exist |
| **D** | DAgger baseline | the teacher relabels the student's own states | Is M2 more than DAgger? | |
| **M3** | One persistent student | tasks in sequence; naive sequential vs replay vs self-rollout rehearsal (Garg et al.); gate on vs off | Retention across tasks (H3) | |
| **C** | Cost | teacher LLM dollars and seconds per episode; student inference latency | The cost side of "compact" | Teacher cost is already recorded per session |

**Figures and tables to aim for**

| # | Content | From |
|---|---|---|
| Table 1 | Teacher vs student before vs student after, per task, on the same 50 official scenes | T0, S0, S1 |
| Figure 1 | The pipeline (the diagram in the overview) | |
| Figure 2 | Student success against number of teacher demos | S1 |
| Table 2 | Same-layout vs held-out-layout, with retention on the standard suite | S1, S1-layout, S2 |
| Table 3 | Training method: default LoRA, LoRA with backbone, full | S1-method |
| Figure 3 | Failure families before and after training (semantic vs physical) | grades |
| Table 4 | M2 ablation, with semantic accuracy and physical success reported separately | M2-a…e, D |
| Figure 4 | Forgetting matrix: success on every earlier task after every update | M3 |
| Table 5 | Cost: teacher dollars and time per success, student latency | C |

**What each candidate claim needs**

| Claim | Minimum evidence |
|---|---|
| Teacher trajectories lift a compact student from 0% on shifted layouts, without privileged state or reinforcement learning | Table 1 on ten tasks, three training seeds |
| The student learned to find the object, not a position | Table 2: held-out layout well above 0%, standard suite retained |
| Separating semantic from physical failures beats generic replay and DAgger | Table 4 at matched budgets |
| One small student can be updated task after task without forgetting | Figure 4 against the joint upper bound and the restart-per-task baseline |

Only the first two depend on work that is already built. If H2 or H3 come out neutral, the first
two with the cost table are still a complete, honest paper; say so in the plan rather than
discovering it at the deadline.

## 5. Protocol for paper-grade numbers

* **Test:** the 50 official saved scenes per task, the official step limit (280 for object
  suites), three policy seeds. With one run of 50 episodes the 95% range is about ±14 points at
  50% success.
* **Training seeds:** at least three for every reported training setting.
* **Development vs test:** tune on the `dev` scene set and on training layouts. Run `official`
  only at declared checkpoints. Never debug a teacher program on `official`.
* **Teacher and student on identical scenes**, each under its own controller and step limit;
  state both limits (teacher 1,000 steps in joint control, student 280 in hand-movement
  control).
* **Report** numerators and denominators, and confidence intervals.
* **Per checkpoint, record:** overall success; semantic accuracy (the right object was
  selected); physical success given the right object; the failure-family mix; training time;
  student inference latency; cumulative teacher cost; dataset size; success on earlier tasks.
* **State the deviations** listed in the overview (section 6): the student checkpoint, 10 actions
  executed per prediction, conversion by re-execution at 3×.

## 6. Recording: the proposal's specification against what exists

The proposal defined three capture tiers. Status of each data family:

| Data family | Recorded now | Not recorded |
|---|---|---|
| Identity and provenance | run ID, task, instruction, suite, seed, init mode, program hash, harness commit, scene set, ledger with lineage | a layout hash per scene |
| Camera observations | both cameras as MP4 for every step; lossless keyframes before and after each motion call; camera intrinsics and pose | lossless frames for every step |
| Proprioception | joint positions and velocities, hand pose, gripper, per step | joint effort |
| Motion | the command sent and the resulting state per step; for demos, the actions before idle frames were dropped and the tracking lag | |
| Agent and tool decisions | every tool call with input and output, errors, retries, timing; the agent's transcript; each program version | |
| Perception | SAM3 masks, scores and overlays for every segmentation call; grasp candidates; point clouds and boxes | masks at every step; per-step depth |
| Outcomes and failures | benchmark success, termination reason, warnings read from the calls, failure family, which object was lifted and placed (offline, with simulator state) | labels from the agent's own perception |
| Training and evaluation | exact command, log, loss and learning rate, dataset lineage, checkpoint; per student episode: actions, state, video, labels | checkpoint parent and child registry, gate decisions |
| Simulator state | at the start and end of graded teacher episodes and of student episodes; never in policy inputs | |

## 7. Risks

| Risk | Guard |
|---|---|
| The student memorises layouts instead of learning to look | The held-out-layout experiment is the headline, not the same-layout one |
| The teacher is not strong enough on the official scenes | Measure T0 first; improve programs before collecting; report teacher failures by family |
| Fine-tuning forgets the standard suite | Replay with original LIBERO data, fixed normalisation statistics, the retention run |
| The converter fails on other objects | Check all ten tasks before the full run; the success filter stays |
| Too few demonstrations | Plan for 200 per task; the demo-count curve shows where it saturates |
| A closer paper appears | Repeat the related-work search before writing; keep the claims list in section 4 current |
| Shared, busy GPUs | Record the machine's load; run teacher collection when the GPU is free; batches now tolerate slow seeds |
| LLM quota and cost | Bedrock's daily token quota was reached once on 12 November; teacher collection and all student stages use no LLM |
| False success labels | Benchmark success is the authority; every teacher episode is replayed and must reproduce; unknown stays unknown |
| Leakage | Scene sets by name and role; lineage for every training sample; simulator state only for grading |

## 8. Decisions still open

These are the project owner's to make. Each changes the work list.

1. **Target venue and date.** They set how far down section 4 the first paper goes.
2. **How many tasks and suites** in the first paper: ten tasks of `libero_object_swap` only, or
   also `libero_goal_swap` (where the ASPIRE paper's teacher scores 0% on several tasks without
   its skill library).
3. **A second student** (π0.5 or OpenVLA) as more than a sanity check. SEAR and SynthDemo-RL
   both use π0.5; a 450M student is our difference, and a reviewer may still ask.
4. **Reinforcement-learning baselines** (SimpleVLA-RL, SynthDemo-RL): reproduce, or compare with
   published numbers only.
5. **Failure labels for M2:** how the agent derives them from images, and how much oracle
   comparison to report.
6. **Whether the proposal PDF belongs in the repository.** It is summarised in the overview and
   is not committed.
