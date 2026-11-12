# Trace2VLA preflight: teacher data → fine-tuned SmolVLA → comparison

A first complete pass of the pipeline on two tasks of `libero_object_swap`. It answers three
questions before the full ten-task run:

1. Does the teacher solve these tasks on the official scenes, and what do its failures look like?
2. Do its episodes convert to student demonstrations (yield, length, idle frames)?
3. Does fine-tuning SmolVLA on those demonstrations move its success from 0%?

Background and the measurements behind the design: [trace2vla-review.md](trace2vla-review.md).
The student side (SmolVLA baseline, protocol): [vla-student.md](vla-student.md).

Everything lives in one folder, `outputs/trace2vla/preflight/`, with a ledger of what produced
what. One command drives every stage: `scripts/gpu/t2v.sh`.

## What gets recorded

| Stage | Recorded | Where |
|---|---|---|
| Teacher episodes | ASPIRE's full recording: every control step (command, joints, hand pose, gripper), both camera videos, every tool call with input and output, SAM3 masks and overlays, grasp candidates, point clouds, keyframes, the program | `teacher/<scene set>/task_<id>/<seed>/run-*/` |
| Teacher grading | replay fidelity, which object was lifted and placed, failure family, simulator state at start and end | `grades/<scene set>/task_<id>/seed_<n>.json`, `.npz` |
| Demonstrations | stored actions and states, the actions before idle frames were dropped, which frames were kept, tracking lag, checks, simulator states, video as the student sees it | `demos/<name>/task_<id>/seed_<n>/` |
| Dataset | LeRobot dataset, lineage manifest (demo → teacher run, hashes), both sets of statistics | `datasets/<name>/` |
| Training | exact command, log, loss and learning rate per logged step, checkpoint | `training/<run>/` |
| Student evaluation | per episode: actions, policy state, joints, hand pose, gripper, video, failure labels, simulator states; live progress and results | `evals/<name>/` |
| Everything | one event per stage run | `ledger.jsonl` |

Simulator state is privileged. It is used for replay and grading only and is written only under
the experiment folder; the teacher agent's files and the student's inputs never contain it.

**Failure families** (student and teacher): `semantic` = another object, placed correctly;
`physical` = the requested object, not placed; `both` = another object, not placed;
`no_attempt` = the gripper never closed on an object.

## Before you start

```bash
git pull origin claude/dazzling-pascal-4r6ipf
```

This update adds one package to `.venv-libero` (`peft`, for LoRA training). Install it with the
same setup command as before; it takes about a minute and does not touch the model services:

```bash
scripts/gpu/vla.sh setup
```

```bash
scripts/gpu/services.sh status
```

The four model services must be up for the teacher stages (`scripts/gpu/services.sh start` if
not). Restart the dashboard so it loads the new pages:

```bash
python scripts/gpu/dashboard.py
```

Open the **Trace2VLA** tab (`http://localhost:8765/#/t2v` through your SSH tunnel).

## Step 1: the teacher's programs (the only step that costs LLM money)

The coding agent develops one program per task. This is the command you already used, with two
tasks and a total cap of $50:

```bash
scripts/gpu/session.sh run --tasks quick --no-preflight --llm libero-pro:libero_object_swap:0 --llm libero-pro:libero_object_swap:1 --llm-budget 25 --llm-total-budget 50 --reserve 30
```

Each agent ends with a line such as `dev_v02 3/3; held-out task_success 3/3`. A program
with no held-out success is not a teacher: look at the `warning` lines under it (they say, for
example, that SAM3 did not recognise a prompt) before going on.

It prints the session folder (`outputs/sessions/<date>-<time>-tasks`). Each agent leaves its
best program in `agents/llm-libero-pro-libero_object_swap-<id>/program_best.py`. Check on the
session's dashboard page that both agents finished with held-out successes before going on.

## Step 2: create the experiment and register the programs

```bash
scripts/gpu/t2v.sh init preflight --suite libero_object_swap --tasks 0,1
```

Replace the folder with the one step 1 printed:

```bash
scripts/gpu/t2v.sh teacher-add preflight --from outputs/sessions/<date>-<time>-tasks
```

Sanity check: LeRobot, the base checkpoint, GPUs, disk, programs, services, and one simulator
reset per task:

```bash
scripts/gpu/t2v.sh check preflight
```

Every line should say PASS before you continue.

## Step 3: teacher baseline on the official scenes

The 50 official scenes of each task, the same scenes SmolVLA is tested on. No LLM cost.

```bash
scripts/gpu/t2v.sh teacher-run --detach preflight --scenes official --workers 2
```

```bash
scripts/gpu/t2v.sh status preflight
```

```bash
scripts/gpu/t2v.sh logs preflight
```

When it has finished, grade the episodes (replay in the simulator, no services needed):

```bash
scripts/gpu/t2v.sh teacher-grade --detach preflight --scenes official
```

Check: the Trace2VLA page's **Teacher** tab shows success per task, how unsolved episodes
ended, their failure families, and a Replay link per episode (videos, tool calls, SAM3 masks).
"not reproduced" must be 0.

## Step 4: teacher demonstrations on training layouts

Fifty seeded layouts per task (seeds 1001–1050), disjoint from the official scenes and from the
seeds the agent debugged on.

```bash
scripts/gpu/t2v.sh teacher-run --detach preflight --scenes train --workers 2
```

```bash
scripts/gpu/t2v.sh teacher-grade --detach preflight --scenes train
```

## Step 5: convert to student demonstrations and build the dataset

Re-executes each successful teacher episode in the student's hand-movement control at 3× speed,
drops idle frames where the shortened sequence still solves the task on replay, and records what
the student sees.

```bash
scripts/gpu/t2v.sh convert --detach preflight --scenes train --speed 3
```

Check on the **Demos** tab: kept vs rejected per task, steps (should be under 280), idle frames
dropped. Open a demo to see it next to the teacher's own video.

```bash
scripts/gpu/t2v.sh dataset preflight --demos train-3x --name d50
```

It replays every demo once more while writing frames, and prints the action spread of this data
next to the checkpoint's.

## Step 6: student before training

On the official scenes of the two tasks, and on the standard suite for retention:

```bash
scripts/gpu/t2v.sh eval preflight --name before-official --checkpoint base --scenes official --seeds 0
```

```bash
scripts/gpu/t2v.sh eval preflight --name before-standard --checkpoint base --suite libero_object --scenes official --seeds 0
```

## Step 7: fine-tune

```bash
scripts/gpu/t2v.sh train --detach preflight --dataset d50 --name ft-full --method full --steps 3000 --batch-size 32
```

```bash
scripts/gpu/t2v.sh status preflight
```

The **Training** tab shows the loss curve live. These settings are a first guess (about five
passes over 100 demos); the preflight is what tells us whether they need changing. Other
methods to compare afterwards: `--method lora` (LeRobot's default LoRA: action expert only) and
`--method lora-vlm` (LoRA that also covers the vision-language backbone).

## Step 8: student after training, and the report

```bash
scripts/gpu/t2v.sh eval preflight --name after-official --checkpoint ft-full --scenes official --seeds 0
```

```bash
scripts/gpu/t2v.sh eval preflight --name after-standard --checkpoint ft-full --suite libero_object --scenes official --seeds 0
```

```bash
scripts/gpu/t2v.sh report preflight
```

The report (`outputs/trace2vla/preflight/reports/summary.md`) and the **Compare** tab show, per
task: teacher on the official scenes, student before, student after, the student's failure
families, and retention on the standard suite. "Same scene, side by side" plays the teacher's
and the student's videos for one official scene.

## Optional: failure types of the 1,500-episode baseline

Replays the stored actions of an existing student run and labels each episode (no model needed).
Seed 0 is 500 episodes:

```bash
scripts/gpu/t2v.sh grade-student outputs/vla/baseline-smolvla-libero_object_swap-n10 --seeds 0
```

## Reading the result

| What you see | What it means |
|---|---|
| Teacher well below 90% on `official` | Fix the program first (rerun step 1 for that task): the student cannot learn what the teacher fails |
| Many demos rejected, or "kept all idle frames" for most | Try `--speed 2`, and tell me: the converter needs tuning for that object |
| Student after ≫ 0% on `official` | The pipeline works end to end |
| Student after still 0%, failures still `semantic` | The model did not relearn which object to take: try `--method lora-vlm`, more steps, or more demos |
| `after-standard` far below `before-standard` | Forgetting: the next step is mixing original LIBERO data into training |

Because the 50 scenes of a task differ by only centimetres, success here shows the student can
learn the swapped layout from the teacher. It does not yet show that it finds objects in
layouts it has not seen; that is the held-out-layout experiment that follows the preflight.

## Commands for watching and stopping

| | |
|---|---|
| Everything at a glance | `scripts/gpu/t2v.sh status preflight` |
| Follow the stage running in the background | `scripts/gpu/t2v.sh logs preflight` |
| Stop it (finished episodes are kept; rerun the same command to continue) | `scripts/gpu/t2v.sh stop preflight` |
| All experiments | `scripts/gpu/t2v.sh status` |
| Choose the GPU for simulate / train / evaluate stages | `scripts/gpu/t2v.sh --gpu 1 <stage> ...` |

`teacher-run`, `teacher-grade`, `convert` and `eval` skip what is already done, so rerunning a
command after an interruption continues it. `dataset` and `train` refuse to overwrite: give a
new `--name`.

Two kinds of teacher episode are kept on disk but not counted, and `teacher-run` records their
scenes again: episodes in which a model service did not answer (for example SAM3 out of GPU
memory), and episodes of a program you have since replaced with `teacher-add --force`. The
teacher table lists them as "not counted". `teacher-add` prints what the agent measured for a
program and warns when it solved none of its held-out scenes.
