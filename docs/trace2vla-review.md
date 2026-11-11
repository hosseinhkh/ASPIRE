# Trace2VLA design review (11 November 2026)

A review of the proposal (*Trace2VLA: Distilling Agent-Executed Manipulation Trajectories into
Compact Vision-Language-Action Models*), the code and the results so far, made before the
teacher-data phase starts. It records what was measured, what each measurement changes in the
design, and the order of the next steps.

## 1. Where the project stands

| Proposal item (M0) | Status |
|---|---|
| Student runs in the same simulator as the teacher | Done. SmolVLA runs through ASPIRE's `LiberoAdapter` in `OSC_POSE`. Parity with LeRobot's own LIBERO env is exact: identical images, robot and MuJoCo state over 120 actions, and identical SmolVLA actions over one episode |
| Student baseline (S0) | Done. `libero_object_swap`: **0 / 1,500** (10 tasks × 50 official scenes × 3 seeds). Standard `libero_object`: 462 / 500 = 92.4% (paper 96%) |
| Teacher baseline (T0) on the same scenes | Not done. Six recorded successes of the agent's program on task 0 exist (seeded scenes), none on the official scenes |
| Action / camera / state bridge | Prototyped and measured in this review (section 2.2). Not product code yet |
| Archive of all episodes with labels | Partly. Both actors record every step; they use two formats with no common index, and only `task_success` is labelled |
| LeRobotDataset export | Not done |
| M0 gate: one verified ASPIRE success exported without ambiguity | Not met yet |

Deviations from the proposal, to state in any write-up:

* **S0 is `lerobot/smolvla_libero`, not `lerobot/smolvla_base`.** The base model was pretrained on
  6-joint SO-100 arms and cannot command a Panda, so its LIBERO score would be 0 for an
  uninteresting reason. The proposal already calls the LIBERO checkpoint a "preadapted reference".
* **The student executes 10 of each 50 predicted actions** before predicting again. This was
  measured, not assumed: 78% / 93% / 83% on LIBERO-Object for 50 / 10 / 1 actions.
* **Teacher and student have different step limits and controllers.** The teacher has 1,000 steps
  in joint control; the student has LIBERO's 280 in hand-movement control.

## 2. What was measured in this review

All on `libero_object_swap` unless noted. Scripts are in the session scratchpad (prototypes).

### 2.1 Teacher episodes are 2× too long for the student's limit

Seven successful teacher episodes recorded under the fixed harness (six from the agent's program
on seeded scenes, one from the example program on a saved scene):

| | Teacher (joint control) | Student limit | SmolVLA's own successes |
|---|---|---|---|
| Steps per episode | **466–534** | 280 | about 105–240, typically 115–150 |

About 3–7% of teacher steps are idle (the hand stands still while the gripper moves). The hand
travels about 1.03 m at no more than 5 mm per step.

### 2.2 Re-executing teacher episodes in the student's control works

Prototype: reset the same scene in `OSC_POSE`, then follow the teacher's recorded hand path with
a simple tracker (aim 3 teacher steps ahead, never run more than 2 cm ahead of the hand, play the
±12 steps around a gripper change and all idle steps at 1×, the rest at the chosen speed-up).
It reads only what the recorder already stores (`steps.jsonl`, `reset.json`, `run.json`).

| Speed-up | Replays that still solve the task | Steps | Commands at the limit | Mean command size |
|---|---|---|---|---|
| 1× | 6 / 7 | 477–518 | 0% | 0.08 |
| 2× | 7 / 7 | 258–282 | 0% | 0.15 |
| **3×** | **7 / 7** | **190–208** | 0% | 0.20 |
| 4× | 7 / 7 | 163–177 | 0% | 0.23 |
| 6× and 8× (3 episodes) | 3 / 3 each | 156–161 | 0% | 0.26 |

* A scene is reproducible from (suite, task, init mode, seed): two fresh processes gave an
  identical simulator state, and an episode recorded on cali replayed here from a start gap of
  0.0 mm.
* Above 4× the step count stops falling (about 160): the tracker's own limits set the pace.
* So the plan "leave the teacher untouched, convert afterwards, keep only replays that succeed"
  holds, and conversion at 3× also solves the episode-length problem. One replay failed at 1×,
  which is why the success filter stays.
* **Limits of this evidence:** one task, two programs, seven episodes. It must be repeated on
  the other nine tasks during the preflight.

### 2.3 The 50 test scenes of a LIBERO task are nearly one scene

Object positions over the 50 official scenes (spread = max − min):

| Suite : task | Target object | Other objects | Container |
|---|---|---|---|
| `libero_object` : 0 | 0 mm (fixed) | 0–10 mm | 30 mm |
| `libero_object_swap` : 0 | 0 mm (fixed) | 0–10 mm | 29 mm |
| `libero_object_swap` : 5 | 20 mm | 0–10 mm | 30 mm |
| `libero_goal` : 2, `libero_goal_swap` : 2 | 29–39 mm for every object | | |

Fifty seeded layouts of `libero_object_swap` : 0 have the same ranges as the 50 saved scenes.

* Seeded training scenes and saved test scenes come from the same distribution. That part of the
  plan is valid.
* But "different seeds" does not mean "different scenes": train and test differ by millimetres.
  A student can pass the test by memorising one trajectory per task. This is true of every LIBERO
  result, and it is the point of the LIBERO-Pro paper.
* For ASPIRE, 15 development seeds carry little more information than 3.

### 2.4 What "swap" is, and how the student fails

* In the swap suite the target exchanges places with one other object. Task 0: the alphabet soup
  and the butter. LIBERO-Pro's generator picks the partner from five candidates per task
  (`libero_ood/ood_spatial_relation.yaml`); the official suite is one draw.
* Offline grading with simulator state (never shown to the policy), one episode per task
  (scene 1, seed 0) for tasks 0–5:

  | Task | Asked for | What the student lifted | Ended in the basket |
  |---|---|---|---|
  | 0 | alphabet soup | butter | butter |
  | 1 | cream cheese | alphabet soup | alphabet soup |
  | 2 | salad dressing | nothing | nothing |
  | 3 | bbq sauce | chocolate pudding | chocolate pudding |
  | 4 | ketchup | cream cheese | nothing |
  | 5 | tomato sauce | nothing | nothing |

  The student never lifted the requested object. In four of six episodes it lifted a different
  object, and in three it placed that object in the basket: the proposal's "semantic = false,
  physical = true" case. In two it lifted nothing. In task 0 the object it took is the one
  sitting where the target used to be. Tasks 6–9 were not graded here; grade all 1,500 baseline
  episodes on cali for the full split.

### 2.5 Facts from LeRobot's training code (v0.4.4)

* **Fine-tuning replaces the checkpoint's normalisation statistics** with the new dataset's
  (`lerobot_train.py`, the `normalizer_processor` and `unnormalizer_processor` overrides). A small
  teacher dataset with smaller actions would rescale the model's inputs and outputs at step 0,
  and every later update would rescale them again.
* **The default LoRA adapts only the action expert** (its q/v projections, and the state and
  action projections). The vision-language backbone is not touched. The checkpoint itself was
  trained with full fine-tuning (vision encoder unfrozen, lr 1e-4, batch 32, 25k steps).
* `lerobot/libero` is in the v3.0 dataset format, which v0.4.4 reads. It declares 10 fps although
  the control rate is 20 Hz; action chunks are counted in frames. `MultiLeRobotDataset` is
  disabled, so mixing two datasets means merging them (`lerobot-edit-dataset`, operation `merge`).

### 2.6 Facts from ASPIRE's tooling

* `run_validation.py` already takes `--init-mode saved`, `--seeds`, `--output-dir` and
  `--workers 4`. Running a task's final program (`task_<id>/fix_code.py`) on the 50 official
  scenes needs only a wrapper and a results table.
* A teacher episode takes about 31 s of wall time (simulation 12 s, grasp planning 4 s, SAM3
  1 s, recording 8 s, reset 5.5 s) and ends at the first success.
* `demo.sh campaign` has a small protocol (3 development seeds, 3 held-out) and the paper
  protocol (15 + 15 + 50).

## 3. What changes in the design

1. **Conversion is re-execution at 3× with a success filter.** No change to the teacher, the
   student's step limit or the harness. Record the yield (replays kept / tried) and the step
   counts per task.
2. **Split by layout, not by seed.** Seeds only move objects by millimetres. Define named scene
   sets:
   * `official`: the 50 saved scenes of the official suite. Test only.
   * `sibling`: the other swap partners for each task (up to four), generated with LIBERO-Pro's
     own tool. Training and development.
   * `standard`: the original suite. Retention.

   Two experiments follow:
   * **Same layout (pipeline check):** train on teacher demos from the official layout's seeded
     scenes, test on `official`. It should work, and proves the pipeline. It does not prove
     generalisation.
   * **Held-out layout (main result):** train on `sibling` layouts only, test on `official`.
     Success here means the student learned to find the named object, not a position.
3. **Retention now has a sharper meaning.** The instruction is identical in the standard and the
   swap suite; only the scene differs. A student that scores well on both after training is
   looking at the scene.
4. **Training rules.**
   * Keep the checkpoint's normalisation statistics fixed for every update.
   * Write the dataset with the same feature names, image size and declared fps as
     `lerobot/libero`, so the two can be merged for replay.
   * Compare three settings: default LoRA, LoRA that also covers the vision-language backbone,
     and full fine-tuning. The failure is in grounding, which the default LoRA does not touch.
5. **Teacher baseline with the small protocol.** Because scenes barely vary, 3 development seeds
   are enough to develop each program. Then run `fix_code.py` on `official` in saved mode.
6. **M2 is concrete on this benchmark.** Most of the baseline's graded failures are semantic. The label "which
   object ended in the container" can be graded offline from the simulator, for grading only. For
   physical failures, the teacher's program can continue from the student's failure state if that
   state is recorded, since the adapter can already start from any saved state.
7. **Record the simulator state** at reset and at the end of every teacher and student episode.
   Conversion does not need it. Recovery data and reproducibility after code changes do.

## 4. Structure for the full pipeline

One simulator layer, three actors, one data layer, one loop.

| Stage | Exists | Missing |
|---|---|---|
| Simulator: `LiberoAdapter`, joint and hand-movement control | yes | simulator state in recordings |
| Teacher development: `demo.sh campaign` → `fix_code.py` | yes | |
| Teacher evaluation on a named scene set | `run_validation.py` | wrapper, results in the student's table format |
| Demo collection on training scene sets | batch runner | scene-set definitions, sibling layouts |
| Converter (re-execution in hand-movement control) | prototype | product code, alignment tests, yield report |
| Dataset versions (LeRobot format) | | writer, manifest with lineage, frozen statistics |
| Trainer | `lerobot-train` | wrapper, logs, the three settings |
| Student evaluation | `cap_harness.vla.benchmark` | failure labels, scene-set argument |
| Gate and checkpoint registry | | accept / roll back, parent and child of each checkpoint |
| Ledger and dashboard | | section 5 |

Rules that keep it coherent:

* **One experiment folder with a ledger.** Every stage appends an event (teacher run, conversion,
  dataset version, training, evaluation, gate decision) with paths, hashes and parents. ASPIRE's
  own run folders stay untouched and are referenced.
* **Scene sets are named and versioned.** Every episode records its scene set and a layout hash,
  so a training episode can never be confused with a test episode.
* **Each stage is one command that reads and writes the ledger.** A script runs the round first.
  Later the agent can call the same commands, which is how the online loop (M3/M4) plugs in:

  1. evaluate the current student on development scene sets;
  2. pick what to collect (first by rule: tasks below a threshold; later the agent decides);
  3. run the teacher's program, or have the agent develop one;
  4. convert;
  5. build the next dataset version (old + new + replay);
  6. train a candidate from the current student;
  7. evaluate the candidate, including retention;
  8. accept or roll back;
  9. test on `official` only at declared checkpoints.

## 5. Dashboard and logging upgrades

* **Experiment page:** rounds over time; per task, teacher vs student before vs student after;
  success against number of demos; retention.
* **Side by side:** teacher and student on the same scene, two videos.
* **Dataset page:** each converted episode with a link to its teacher run, its speed-up, steps
  and tracking lag, and a viewer that steps through frame, state and action together.
* **Training page:** loss, learning rate, GPU, time, and the checkpoint's parent and dataset.
* **Failure breakdown:** semantic vs physical per task, with example episodes.
* **Logging:** simulator state at reset and end; scene set and layout hash per episode; the
  ledger; conversion log; costs (LLM dollars, GPU hours, simulator time).

## 6. Risks

* The converter is validated on one task. Thin or tall objects may need slower contact phases.
* The student cannot learn a task or layout the teacher fails. The preflight measures this.
* The student may memorise layouts. The held-out-layout experiment is the guard.
* Fine-tuning may hurt the standard suite. Replay and fixed statistics are the guard; the
  retention run measures it.
* Both GPUs are shared and busy, so training times are uncertain until measured.

## 7. Order of work

1. **Preflight on two tasks:** small campaign, `fix_code.py` on `official`, conversion at 3×,
   yield and step counts. Record simulator state from here on.
2. **Build:** converter, dataset writer with tests, scene sets including sibling layouts, ledger.
3. **Small fine-tune** on those two tasks (about an hour), evaluated on `official` and on the
   standard suite. This closes the loop once.
4. **Full run:** ten tasks; same-layout and held-out-layout experiments; retention; the three
   training settings; demo-count curve.
5. **Then** failure-aware training (M2) and the automated loop (M3/M4).
