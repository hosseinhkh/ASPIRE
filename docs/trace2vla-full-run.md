# Trace2VLA: readiness for the full run (13 November)

The preflight (two tasks, `preflight2`) closed the loop and showed what the full run must change.
This page lists every issue found so far with its status, the scope and cost of the full run,
the data plan for the paper, and what still has to be built. Details of each issue are in the
[findings log](trace2vla-findings.md); the plan of record is the [roadmap](trace2vla-roadmap.md).

**Verdict:** ready to start the **teacher side** now (programs for the other eight tasks, and
the teacher's check on the new layouts). **Not yet ready** for the full student run: the
dataset for the main experiment needs the new layouts collected, and replay mixing and faster
dataset and evaluation steps are still to build (section 4).

## Decisions (13 November)

* **Both suites, one after the other:** `libero_object_swap` first, finished for **both students
  (SmolVLA and π0.5)**, then `libero_goal_swap`.
* **Training data:** 50 seeds per training layout, used as 25 and 50 (the demo-count curve).
  More seeds cost no LLM money: the LLM is used only to write each task's teacher program, once.
  More seeds cost simulator time only (section 5).
* **Evaluation:** the standard LIBERO protocol below, so every number is comparable with
  published ones.

## Standard evaluation: what every reported number means

The protocol of OpenVLA, LeRobot and LIBERO-Pro, which ours already matches (parity with
LeRobot's LIBERO environment checked; `src/cap_harness/vla/protocol.py`):

| | Standard |
|---|---|
| Episodes | **All 10 tasks of the suite × the 50 fixed initial states each = 500 per suite**, per policy seed |
| Initial states | The suite's own saved ones (`official` scene set; LIBERO-Pro's for the swap suite). Never seeded layouts |
| Step limit | The suite's: 280 for LIBERO-Object, 300 for LIBERO-Goal (and their swaps); 10 still steps after reset, not counted |
| Success | The task's own predicate, checked after every step |
| Policy | 256 × 256 images (both cameras), 8-number state, the benchmark's instruction, 10 actions per prediction |
| Seeds | **3 policy seeds (1,500 episodes) for every number in a table**; 1 seed (500) while developing |
| Reported | Success over the 500 (or 1,500) episodes, per task and mean, with the 95% interval |
| Suites | The swap suite (the problem) **and** the standard suite (retention), for every model |

Not part of the standard, and reported as our own diagnostics only: the sibling layouts
(`alt1…4`), the `dev` probes, and the failure families.

Anchors that show the setup is right (each base model on the standard suites):

| Model | Standard LIBERO-Object | LIBERO-Pro object swap |
|---|---|---|
| SmolVLA (`smolvla_libero`) | 462 / 500 = 92.4% (paper 96%) | 0 / 1,500 |
| π0.5 (`pi05_libero_finetuned_v044`) | to run: LeRobot reports 99% | to run: LIBERO-PRO reports 17% for openpi's checkpoint; ours was 60% on tasks 0–1 |

## 1. What the preflight taught

| # | Finding | Consequence for the full run |
|---|---|---|
| S1 | Fine-tuned SmolVLA goes from 0% to 86% on the official swap scenes, and from 89% to 0% on the standard layouts, already at step 500; nearly all failures pick the **wrong object**. LoRA on the action expert only (vision and language frozen) does the same | The student learned **where to reach for each instruction**, not which object to find. Training on one layout per task can never show grounding. The main experiment must vary the target's position per instruction and test on positions never trained on |
| P8 | Seeded layouts barely vary: in a task, objects land at the centre of their 5 cm region whenever they are about that wide, so the soup and the salad dressing sit at the same spot for every seed; narrower objects move by millimetres | 50 seeds of one layout are close to one scene. **Variety must come from layouts**, not from more seeds |
| — | LIBERO-Pro's swap draws each task's partner from **five** candidates; the official suite is one draw | The other four are layouts with the same instruction and objects and the target ~20 cm elsewhere: **built now** as `libero_object_swap_alt1…alt4` (section 3) |
| — | π0.5 before training is at 60% on the official swap scenes (SmolVLA 0%); during LoRA training it kept 65% on the standard probe at step 500 where SmolVLA had 0% | π0.5 is worth keeping as the second student: it shows whether a stronger grounding prior changes the picture |

## 2. Every issue so far, and its status

| ID | Issue | Status |
|---|---|---|
| T1 | Grasp-planner candidates fail every time they are executed (12 of 12): run in the base frame, gripper base used as the fingertip point | **Avoided:** the teacher prompt now tells new programs to use their own top-down grasp and not the planner (13 November). The harness fix (camera frame, 0.1034 m offset) is not done; not needed for LIBERO-Object |
| T2 | SAM3 does not know product names | Fixed (11–12 November) |
| T3 | Agent batches cut off at 10 minutes | Fixed |
| T4 | A confident SAM3 mask can be the wrong object | Open by nature; the prompt makes the agent check overlays; the teacher's success filter guards the data |
| T5 | Basket centre from the median of points is 5 cm off | Prompt fixed; task 0's registered program still uses the median and works 46/50 |
| I1 | GPU reservation not handed over on demand | Open; workaround documented. `services.sh start` also counts student jobs as ASPIRE processes (13 November) |
| I2 | A busy machine slows everything 3–15×; CPU load is not recorded | Open |
| I4 | Bedrock's daily token quota | Outside the repository; only teacher program development uses the LLM |
| I6 | Ctrl-C counted an episode as finished; `teacher-grade` stopped on it | Fixed |
| P6 | Dataset step rejected successful demos (float32 rounding drift) | Fixed: 98 of 98 kept from now on |
| P7 | Training progress showed 1000, 1000, …, 2000 | Fixed |
| — | `t2v.sh stop` reached only the newest background stage | Fixed |
| — | Probes, curves, step 0, checkpoint evaluation by step | Built (`eval-steps`, dashboard Training tab) |
| — | Checkpoints filling the home disk | Built: `t2v.sh checkpoints --set-root /data/...` and `--move` |
| — | π0.5 loading warnings (`embed_tokens`) | Checked harmless |
| C1 | Two commands could work on the same data at once (two `teacher-run` on one scene set and task record the same seeds twice; two conversions into one demo set write the same files) | **Fixed 13 November:** each `teacher-run`, `teacher-grade` and `convert` claims its (scene set or demo set, task) and a second one is refused with the holder's command and pid. The claim ends with its process however it ends, so nothing is left to clean up. Checked live |
| C2 | A train scene set could share scenes with a test or dev set (same suite, init mode and seed) | **Fixed:** `scenes --add` refuses it; the same scenes in two training sets get a warning |
| C3 | A second scene set converted into an existing demo set (`--demos`) was taken for the first one's seeds | **Fixed:** a demo set records its scene set (`source.json`; older ones are recognised by their episodes) and refuses another. Checked live |
| C4 | `teacher-run` over ten tasks stopped at the first task without a program, after hours of the others | **Fixed:** every task is checked before anything runs |
| C5 | Nothing checked the disk before thousands of teacher episodes | **Fixed:** `teacher-run` refuses when the episodes (about 10 MB each with videos, counted generously) plus 10 GB do not fit; `--no-videos` or `--force` |
| C6 | One scene set per `teacher-run` command | **Built:** `--scenes a,b,c` runs them one after another in one (detachable) command |
| C7 | Collection needed one command per stage and per scene set, and waited for every program first | **Built 13 November:** `t2v.sh collect` (section 7): programs taken from the session as each agent finishes, teacher task by task, grading and conversion of each finished (scene set, task) in the background, retries, a status file. Tested with stand-ins (15 tests) and run for real with recorded episodes (background jobs, final pass, status, a second collection refused) |
| C8 | A dataset could be built from test or dev scenes | **Fixed:** `dataset` refuses demo sets made from a test or dev scene set (`--allow-heldout` to override) |
| C9 | `check` loaded only the experiment's own suite | **Fixed:** it also loads every task on each sibling layout a scene set uses |
| P10 | The preflight's programs fail on the new layouts (pilot: task 0 solved 0/12 on alt1–4, 3/3 on the official partner's; task 1 failed on alt2; perception right in all 30) | **Being addressed:** agents develop one program per task against all five layouts (`--llm-layouts all`); `setup` no longer reuses the preflight's programs; test one task first with `object_collection.sh agent-test N`. Details: findings P10 |
| C10 | The dashboard showed no collection progress | **Built:** Pipeline tab (per task and scene set), a "Now running" card, a Collection column on the experiment list, the demo sets behind each dataset |

## 3. Built for the full run (13 November): sibling layouts

`configs/libero_layouts/` holds 40 task files: for each of the ten `libero_object_swap` tasks, the
target swapped with each of its four other LIBERO-Pro candidates. They are made with LIBERO-Pro's
own rule, which reproduces all ten official swap files byte for byte when given the official
partner (tested). They are registered as suites `libero_object_swap_alt1` … `alt4`, so every
command that takes a suite takes them; they have no saved layouts and are used with seeded scene
sets. A scene set can now name its suite:

```bash
scripts/gpu/t2v.sh scenes NAME --add alt1-train --suite libero_object_swap_alt1 --seeds 1001-1050 --role train
```

Checked in the simulator: in `alt1` task 0 the alphabet soup stands exactly where the salad
dressing stands in the standard layout (1 mm), 22 cm from its own place; the instruction is
word for word the official one; the teacher's batch runner runs a program on it; `eval` and
`eval-steps` on such a scene set evaluate that suite. Generation: `python -m
cap_harness.libero.layouts` (`--check` confirms the committed files are current).

Which partner each layout uses, per task: `configs/libero_layouts/manifest.json`.

## 4. Still to build before the full student run

| Item | Why | Size |
|---|---|---|
| **Replay mixing** | Retention (experiment S2): mix original LIBERO-Object demonstrations into training | Medium: LeRobot's dataset merge, feature names already match |
| **Faster dataset building** | One writer renders ~30 s per demo: 1,500 demos is about 12 hours | **Built 13 November:** `dataset --workers N` builds N parts at once and merges them with LeRobot's `aggregate_datasets`; checked identical to a one-process build (every action, state, index, decoded frame and statistic) |
| **Parallel evaluation** | One process: ~15 s per episode; 10 tasks × 50 scenes × 3 seeds = 1,500 episodes per model and suite | Medium: split by task across processes, one result table |
| **A dataset from several layouts** | The main experiment trains on `alt1`, `alt2` and `alt3` together, 25 or 50 per layout | **Built 13 November:** `dataset --demos a,b,c --per-task 25` (the limit applies to each demo set) |
| **An experiment over ten tasks** | `preflight2` has tasks 0 and 1 | Commands only: `t2v.sh init` with `--tasks 0-9`, register the programs |
| Accept-or-roll-back gate, M2 labels from images, round script | M2 and M3 | Later; not needed for the M1 tables |

## 4b. Collecting in parallel (built 13 November)

| Stage | Parallel how | Limit |
|---|---|---|
| `teacher-run` | `--workers N` runs N seeds at once; `--scenes a,b,c` runs several scene sets one after another in one command | The model services (SAM3 one request at a time): 2–3 workers, and one `teacher-run` at a time |
| `teacher-grade`, `convert` | `--workers N`: N episodes at once, each worker with its own simulator; `--scenes a,b,c` for several scene sets in one command | CPU cores: about one worker per free core |
| `prepare` | `teacher-grade` then `convert`, for every listed scene set, one command | as above |
| `dataset` | `--workers N`: N parts built at once, then merged; `--demos a,b,c` combines layouts | CPU cores and memory; the AV1 encoder uses several threads per part, so fewer workers than cores |

What does not wait for what: a scene set's episodes can be graded and converted as soon as its
`teacher-run` is done, while the teacher runs the next scene set (grading and conversion use no
model services). Rerunning `prepare` later adds only what is new. A dataset is a fixed snapshot of
the demos converted at the time it is built: build the main experiment's dataset once all ten
tasks' layouts are converted (or build early ones under other names).

What is safe at once, and what is refused: grading and converting a scene set while
`teacher-run` is still recording it is safe (only finished, finalized episodes are read; rerun
`prepare` afterwards for the rest). A second `teacher-run` on the same scene set and task, a second
`teacher-grade` of the same scene set and task, and a second conversion into the same demo set and
task are refused while the first runs (C1). Every stage can be stopped (`t2v.sh stop NAME STAGE` or
Ctrl-C) and resumed by running the same command: an interrupted teacher episode is recorded again,
an unfinished grade or demo is made again (its `seed_N.json` or `episode.json` is written last).

Checked for real on CPU: three workers graded three teacher episodes at once; parallel and
one-process conversion gave the same results; a demo whose re-execution ran into the simulator's
step limit is now rejected (`re-execution_failed`) instead of stopping the command; a failing
episode is reported, the others go on, and a rerun retries it.

## 5. Scope and cost of the full run (estimates)

| Stage | Amount | Time | LLM cost |
|---|---|---|---|
| Teacher programs, tasks 2–9 | 8 sessions | ~25 min of agent time each when it succeeds | ~\$4–7 each, ≤ \$25 cap: ~\$40–60, at most \$200 |
| Teacher on the official scenes (T0) | 10 × 50 | ~35 s each: ~5 h on one worker | none |
| Teacher on the new layouts (training) | 10 tasks × 3 layouts × 25–50 seeds = 750–1,500 | ~4–15 h (two workers) | none |
| Teacher on the held-out layout (dev) | 10 × 10 | ~1 h | none |
| Conversion | 750–1,500 demos, ~36 s each | ~8–15 h on one process; split by task (`convert --tasks`) | none |
| Dataset | as above | ~6–12 h until parallel building exists | none |
| Training | SmolVLA ~25 min per 3,000 steps (larger datasets need more steps); π0.5 LoRA ~4 h | per setting × 3 seeds | none |
| Evaluation | per model: 10 × 50 × 3 seeds on swap, 10 × 50 on standard | ~2 h per 500 episodes on one process | none |

## 6. Data plan for the paper

Per task, with the target in a different place in each layout:

| Scene set | Suite | Seeds | Role | Used for |
|---|---|---|---|---|
| `official` | `libero_object_swap` | saved 1–50 | test | Table 1, the headline numbers |
| `standard` | `libero_object` (via `--suite`) | saved 1–50 | test | Retention, and a second held-out position |
| `alt1-train`, `alt2-train`, `alt3-train` | `…_alt1`, `…_alt2`, `…_alt3` | seeded 1001–1025 (or –1050) | train | The main experiment's training data |
| `alt4-dev` | `…_alt4` | seeded 2001–2010 | dev | Choosing checkpoints on a held-out position |
| `train` | `libero_object_swap` | seeded 1001–1050 | train | The same-layout control (the preflight's setting) |

Experiments, in order of what a reviewer needs:

1. **Main (S1-layout):** train on `alt1–3`, test on `official`, `standard` and `alt4`. If the
   student found objects, it succeeds on all of them; if it learned positions, it fails on all
   three. SmolVLA and π0.5; full and LoRA.
2. **Same-layout control:** train on `train` (official partner), test the same way. Expected,
   from the preflight: high on `official`, near 0 on the others.
3. **Demo count (H1):** 25 / 50 / 75 / 150 demos per task from the `alt` pool.
4. **Replay (S2):** the main setting plus original LIBERO-Object data.
5. Three training seeds for the settings that go into tables.

## 7. Collecting the object suite in one command (`t2v.sh collect`)

Decided 13 November (evening): every layout of `libero_object_swap` (`train`, the official
partner's, and `alt1`–`alt4`), 25 seeded scenes each, all ten tasks: 1,250 teacher episodes. The
coding agents write the programs of tasks 2–9 while the teacher already records the tasks that
have one.

`t2v.sh collect` does it all and resumes where it stopped when run again:

* **Programs as they arrive.** With `--programs-from SESSION`, a task without a program takes the
  one its coding agent wrote, once the agent has finished and its program solved at least half of
  its held-out scenes (`--min-heldout`). A weaker program, or none, is reported and not used. With
  `--follow` it waits for the missing ones until the session has finished.
* **The teacher, task by task**, on every listed scene set in order (`--workers` episodes at once).
* **Grading and conversion in the background**: as soon as a (scene set, task) is recorded, a
  separate process grades and converts it (`--prepare-workers`) while the teacher goes on. Test
  scene sets are graded only, never converted. A final pass at the end catches anything missed.
* **Failures.** An episode a model service failed is recorded again. A (scene set, task) still
  missing scenes after `--attempts` tries is left incomplete and named. Services that stop
  answering are waited for (`--service-wait`, 30 min); services that answer but fail every
  episode three times in a row stop the collection with a message. Two collections of one
  experiment, or two commands on the same scene set and task, are refused.
* **Progress**: `collect.json` in the experiment folder; `t2v.sh status` prints a table (one row
  per task, one column per scene set); the dashboard's **Pipeline** tab shows the same, live:
  what runs now, episodes recorded / solved / graded / kept per cell, each task's program and where
  it came from, the background conversions with their logs, and the time left.

The LeRobot dataset is built once at the end, from the demo sets an experiment uses (section 6):
which layouts go into a dataset is a choice per experiment, not part of the collection.

### One command per step: `scripts/gpu/object_collection.sh`

Each step prints every command before it runs it, stops at the first error, and keeps its log in
the experiment's `logs/` (`<time>-object_collection-<step>.log`):

| Step | What it does |
|---|---|
| `agent-test N` | One coding agent writes task N's program against all five layouts (in the terminal, about 45–75 min, one agent's LLM cost), then the program records 3 new scenes of each layout (experiment `object-agent-test`) and a verdict says whether it works on every layout |
| `agent-fix N [PROGRAM]` | The same, but the agent repairs an existing program (default: task N's program from the last `agent-test` or `agent-fix`): it runs it first as version 1, then fixes only what fails, keeping every layout it solves. The LLM is paid only for what fails |
| `pilot` | Pipeline test (about 20–30 min): experiment `object-pilot`, tasks 0–1 × 5 layouts × 3 scenes with the preflight's programs through `collect`, a dataset, and a verdict |
| `setup` | Experiment `object-full`: 10 tasks, 25 scenes on each layout (`--replace`, so it also fixes an existing one), `t2v.sh check` (missing programs are expected). It registers no program unless `PROGRAMS=folder` is set: every program comes from an agent that saw all five layouts |
| `start` | In the background: one session of coding agents for the tasks without a program, each developing against all five layouts (`AGENT_LAYOUTS=all`; log `logs/<time>-agents.log`), then `collect --follow` taking each program as its agent finishes. Refuses when a collection already runs |
| `status` | The collection table, and each agent: working, or finished with its held-out score |
| `stop` | Stops the collection (and its background jobs) and the agents' session; finished work is kept; `start` continues |
| `datasets` | After the collection: `main-25` (alt1–3), `control-25` (official partner's layout), and their π0.5 copies |

Where things are, per experiment (`outputs/trace2vla/<name>/`): `logs/` holds the steps' logs,
the agents' log and `collect`'s own log; `logs/collect/` one log per background grading and
conversion job (`<time>-prepare-<scene set>-task<NN>.log`) and the final pass; `collect.json` the
state; `teacher/<scene set>/task_<NN>/` the teacher's runs; `demos/<scene set>-3x/`;
`datasets/<name>/`. The agents' session is `outputs/sessions/<time>-objagents/`, linked from the
Pipeline tab, as is each agent's conversation.

Checked for real (13 November, CPU, stand-in programs and services; the agents' session a
stand-in): `pilot` (30 episodes on all five layouts, background jobs, final pass, verdict);
`setup`; `start` with an agent finishing mid-run (its program taken, its task recorded on all
layouts); a second `start` refused; `stop` in the middle (status "stopped", no process left,
every claim released) and `start` again (only the 13 missing episodes recorded; the episode cut
off by the stop kept on disk and not counted); the Pipeline tab and its links (session, agent
conversation, full log, each job's log).

Pulling while things run is safe: `git pull` writes new files instead of editing them in
place, so a running shell script keeps reading its old copy (checked), and running Python
processes keep the code they loaded. The services' and reservation scripts do not change.
Restart the dashboard after a pull to get the new pages.

Start the model services yourself (`scripts/gpu/services.sh start`) before the session and the
collection: a session that started the services stops them when it ends, and the collection
would then wait for them. A session that finds them running leaves them up.
