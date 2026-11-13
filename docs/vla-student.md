# VLA student: SmolVLA and π0.5 on LIBERO and LIBERO-Pro

Part of Trace2VLA; the project overview is [trace2vla.md](trace2vla.md).

This is the first part of the teacher → student plan: measure a small vision-language-action
model (SmolVLA, 450M parameters) on the same scenes ASPIRE is measured on, in the same
simulator, before any ASPIRE data is used to train it.

| Step | What | Command |
|---|---|---|
| 0 | Install LeRobot + SmolVLA into `.venv-libero` | `scripts/gpu/vla.sh setup` |
| 0 | Check that our simulator path equals LeRobot's | `scripts/gpu/vla.sh parity --policy` |
| 0 | Check the checkpoint on standard LIBERO-Object (paper: 96%) | `scripts/gpu/vla.sh sanity` |
| 1 | **SmolVLA baseline on LIBERO-Pro `libero_object_swap`** | `scripts/gpu/vla.sh baseline` |

Everything runs on GPU 1 by default (`ASPIRE_VLA_GPU`, or `scripts/gpu/vla.sh --gpu N ...`),
so ASPIRE keeps GPU 0. Results go to `outputs/vla/`.

## Two students: SmolVLA and π0.5

SmolVLA is the main student and the default everywhere. π0.5 is a second student for the same
experiments: a larger model with published LIBERO and LIBERO-Pro numbers, so it also checks our
setup independently. Every command that loads a policy takes `--student smolvla|pi05`; leaving
it out means SmolVLA, exactly as before π0.5 support.

| | SmolVLA | π0.5 |
|---|---|---|
| Checkpoint | `lerobot/smolvla_libero` (0.45B) | `lerobot/pi05_libero_finetuned_v044` (about 3.6B, 7.5 GB, bfloat16): LeRobot's own, made for LeRobot v0.4.4 |
| Pinned revision | `a24f1f3…` | `7152f13…` |
| Environment | `.venv-libero` | `.venv-pi05` (its own: see below) |
| Cameras | LIBERO's two cameras renamed to `camera1` / `camera2` | LIBERO's names, no renaming |
| Inputs and actions | two 256×256 images, 8-number state, instruction → 7-number hand movement | the same |
| Actions per prediction | 10 | 10 |
| Normalisation statistics of a training dataset | SmolVLA checkpoint's | π0.5 checkpoint's (mean and standard deviation). For these two checkpoints they are identical to the last bit (checked 13 November): both were fine-tuned on the same LIBERO data. The copy still records the student, and keeps another π0.5 checkpoint right |
| Training methods (learning rate) | `full` (5e-5), `lora` (1e-4), `lora-vlm` (1e-4) | `full` (2.5e-5, the checkpoint's own), `lora` (1e-4). `lora-vlm` is refused: its module names are not checked for π0.5 yet |
| Evaluation settings | the checkpoint's | `torch.compile` off (the checkpoint asks for it: minutes of compiling per process) |
| Standard LIBERO reference (spatial / object / goal / 10) | 90 / 96 / 92 / 71 (SmolVLA paper) | 97 / 99 / 98 / 96 (LeRobot docs) |
| LIBERO-Pro reference | none published | the LIBERO-PRO paper's π0.5 (openpi's own checkpoint, not this one): object 98% original, 17% positions swapped |

### Why π0.5 has its own environment

LeRobot v0.4.4's π0.5 needs `transformers` from the branch `fix/lerobot_openpi` (4.53.3, with
`tokenizers` < 0.22). SmolVLA's environment has `transformers` 4.57.6, and LeRobot's SmolVLA
requires ≥ 4.57.1. One environment cannot hold both, so π0.5 gets `.venv-pi05`:

```bash
scripts/gpu/vla.sh setup-pi05
```

`setup-pi05` (`scripts/bootstrap_pi05.sh`, then the checkpoint download):

* pins every package to the version `.venv-libero` has, except `transformers`, `tokenizers` and
  `scipy` (LeRobot's π extra asks for < 1.15), and reuses the repository's own uv overrides
  (numpy 1.26.4, mujoco 3.3.0, …) and the same editable checkouts (this repository,
  LIBERO-PRO, robosuite, LeRobot);
* never writes `pyproject.toml`, `uv.lock` or `.venv-libero`: SmolVLA's environment is only read;
* checks afterwards that numpy, MuJoCo, robosuite, LIBERO, LeRobot, torch, pandas, peft and the
  rest have the same versions in both environments, that `transformers` is LeRobot's branch, and
  that the CUDA runtime matches; it then writes `.venv-pi05/pi05-environment.json`;
* downloads the checkpoint at its pinned revision (about 7.5 GB).

The PaliGemma tokenizer (`google/paligemma-3b-pt-224`) is gated: the Hugging Face account whose
token is on the server must have accepted its licence. `--force` rebuilds the environment.

You never call `.venv-pi05` yourself. A command started for π0.5 from the usual scripts moves
itself into `.venv-pi05` before it imports LeRobot (it prints `pi0.5: continuing in
.venv-pi05`). It keeps its process id, so `status` and `stop` work as for SmolVLA. If
`.venv-pi05` is missing, the command stops at once and names `setup-pi05`.

### Commands for π0.5

```bash
scripts/gpu/vla.sh sanity --student pi05 --quick
```

```bash
scripts/gpu/vla.sh sanity --student pi05
```

```bash
scripts/gpu/vla.sh baseline --student pi05 --quick
```

```bash
scripts/gpu/vla.sh baseline --student pi05
```

The runs are `sanity-pi05-libero_object-n10` and `baseline-pi05-libero_object_swap-n10`. SmolVLA's
run names are unchanged (`sanity-libero_object-n10`, `baseline-smolvla-libero_object_swap-n10`),
so its existing runs resume as before. Trace2VLA's `dataset`, `train` and `eval` take the student
too: [trace2vla-preflight.md](trace2vla-preflight.md#π05-the-same-experiment-with-the-second-student).

Memory (estimates, to be measured): evaluation about 10 GB; LoRA training roughly 20–35 GB; full
fine-tuning roughly 50–70 GB, so an otherwise empty 96 GB card.

### First run on the GPU (13 November)

`setup-pi05` on the server: the environment check passed (transformers 4.53.3, tokenizers
0.21.4; numpy, MuJoCo, robosuite, LIBERO, LeRobot, torch, pandas, peft and the rest identical to
`.venv-libero`), the checkpoint downloaded (7.47 GB), and its 19 tests passed. The 2-episode
light test on GPU 0 solved both (`libero_object` task 0, 137 and 142 steps, 9–12 s each).

Loading prints two warnings that are **harmless**, checked against the checkpoint and the code:

* `Could not remap state dict keys … Missing key(s): …language_model.embed_tokens.weight`.
  PaliGemma shares one matrix between its token embedding and its output layer (`lm_head`).
  The checkpoint stores it once, as `paligemma.lm_head.weight` (257,152 × 2,048), and LeRobot's
  model ties the two (the same tensor, checked in `.venv-pi05`). LeRobot loads with
  `strict=True`, which copies every weight it finds *before* raising about the missing name, and
  catches the error to print this line. So `embed_tokens` holds the checkpoint's values and every
  other weight is loaded (checked with a small PaliGemma: the same load leaves nothing unloaded).
* `Vision embedding key might need handling: …patch_embedding…`: an information line in
  LeRobot's key fixer; those two tensors are loaded unchanged.

The `huggingface/tokenizers: The current process just got forked` lines came from the
tokenizer's threads; π0.5 commands now start with `TOKENIZERS_PARALLELISM=false`.

### Which student a record belongs to

Every new record names its student: `run.json` and `progress.json` of a benchmark or evaluation,
`eval.json`, a training run's `run.json` and a dataset's `manifest.json`. A record written before
π0.5 support has no such field and is read as SmolVLA, the only student until then; nothing has
to be changed for that. The dashboard shows the student on every run, evaluation, training run
and dataset; the "VLA student" page has a student filter, and the Trace2VLA page shows "before →
after" per student. Mixing students is refused where it would give a wrong number: an evaluation
folder holds one student's episodes, a training run uses a dataset built with its own student's
statistics, and a checkpoint is checked against `--student`.

To write the student into the old records as well (optional, for a tidy archive):

```bash
scripts/gpu/vla.sh label-students
```

```bash
scripts/gpu/vla.sh label-students --apply
```

The first only lists what it would label. `--apply` adds the `student` field and nothing else,
keeps each file's modification time (the dashboard orders runs by it), copies every file to
`outputs/.student-labels/<time>/` first, skips a run whose process is still running, and never
touches teacher runs or episodes. Running it again changes nothing.

## Results so far

Measured on cali (RTX PRO 6000), `lerobot/smolvla_libero`, 10 actions per prediction, the
official saved scenes, 256×256 rendering:

| Suite | Episodes | Success | 95% interval | Published |
|---|---|---|---|---|
| `libero_object` (standard, sanity) | 10 tasks × 50 scenes × 1 seed = 500 | **92.4%** | 90–94% | SmolVLA paper 96% |
| `libero_object_swap` (LIBERO-Pro, positions swapped; **step 1 baseline**) | 10 tasks × 50 scenes × 3 seeds = 1,500 | **0.0%** | 0–0.3% | π0.5 17%; OpenVLA and π0 0% (LIBERO-PRO paper) |

* **Standard LIBERO-Object:** the checkpoint is within about 4 points of the paper. The paper's
  model was trained longer: 100k steps against 25k for this checkpoint. Per task: 80–100%.
* **Swapped positions:** no success in any task, seed or scene. That matches what LIBERO-PRO
  reports for OpenVLA and π0, and the paper's finding that models above 90% on LIBERO drop to
  0 when objects move. In a CPU spot check, asked for the alphabet soup, the policy put the
  butter in the basket. The parity check shows the swap scenes run identically in LeRobot's
  own env.
* This 0% is the "before" number for the student. Training on ASPIRE's demonstrations is
  measured on these same 1,500 episodes.

## How the student and the teacher share one simulator

* **One repository, one environment.** LeRobot is the `third_party/lerobot` submodule (branch
  `aspire-v0.4.4` of the fork, LeRobot release v0.4.4) and installs into ASPIRE's simulator
  venv `.venv-libero` through the `vla` extra. LeRobot main needs Python 3.12 and numpy 2,
  and the simulator stack is pinned to Python 3.11 and numpy 1.26; v0.4.4 fits it
  without moving numpy, PyTorch (2.9.1), MuJoCo (3.3.0) or robosuite (1.4.0). The `vla` extra
  pins `pyarrow<25` (25 requires numpy 2) and `torchcodec<0.10` (0.10 is built for PyTorch
  2.10), and is resolved apart from the SAM3/GraspNet services, which need a newer
  `huggingface-hub` (`[tool.uv] conflicts` in `pyproject.toml`).
* **One simulator code path.** `LiberoAdapter` builds every environment: ASPIRE's teacher
  runs and the student's runs. It picks the scene for a seed, settles it and checks success
  the same way for both.
* **Controllers.** The teacher keeps `JOINT_POSITION`, so ASPIRE's results stay comparable to
  the paper. The student runs the adapter in `OSC_POSE`, LIBERO's default controller and the
  action space of every published LIBERO checkpoint: 7 values in [-1, 1], a hand position
  change, a hand rotation change (axis-angle) and the gripper (-1 open, +1 close). In
  `OSC_POSE` mode ASPIRE's joint-space tools refuse to run, and the student steps the
  adapter with `step_native_action`. Teacher demonstrations will be replayed into this action
  space before training (step 2 of the plan).
* **Same inputs as LeRobot.** The policy gets LeRobot's own processing chain, built with
  LeRobot's factories, as in `lerobot-eval`. Both cameras are rendered at 256×256 (the
  training data's size) and rotated 180 degrees. The 8-value state is hand position, hand
  axis-angle and the two gripper fingers. The task instruction is the benchmark task's
  `language`.
* **LIBERO-Pro as the LIBERO package.** LeRobot imports `libero.libero`; LIBERO-Pro installs
  as plain `libero`. `cap_harness.vla.libero_compat` registers the upstream names. Neither
  package is modified.

### Same folder, same environment: what changes for ASPIRE

The student lives in the ASPIRE checkout and the `aspire` conda env. The conda env itself is
not modified: the packages go into `.venv-libero`, the simulator venv inside the repository.

* **Packages that change in `.venv-libero`.** LeRobot and its dependencies are added (about
  65 packages: transformers 4.57, datasets, accelerate, wandb, and others). Four existing
  packages move: einops 0.4.1 → 0.8.2 (same API), and, with the `vla` extra, fsspec,
  packaging 26.2 → 25.0 and protobuf 7.35 → 6.33. numpy, PyTorch, MuJoCo, robosuite, LIBERO-Pro
  and OpenCV keep their versions. The SAM3, GraspNet, PyRoki and cuRobo venvs are untouched,
  and W&B logging keeps its own `.venv-tools`.
* **Checked after installing.** The full test suite passes. ASPIRE's close-microwave program
  runs in `.venv-libero` with LeRobot installed (`program_ok=True task_success=True`, joint
  control, PyRoki service).
* **Re-running setup.** `uv sync` removes every package the requested extras do not list.
  `scripts/bootstrap_libero.sh` (and so `scripts/gpu/setup_conda_env.sh`) therefore keeps the
  `vla` extra when LeRobot is already installed; `--no-vla` removes it.
* **GPU.** `vla.sh` sets `CUDA_VISIBLE_DEVICES` and `MUJOCO_EGL_DEVICE_ID` for its own command
  only (GPU 1 by default) and says so. It warns when that is also ASPIRE's GPU (`ASPIRE_GPU`).
  SmolVLA needs about 3 GB.
* **Outputs.** Runs go to `outputs/vla/`, which ASPIRE never reads. The dashboard shows them on
  their own page (VLA student); ASPIRE's Runs page does not list them.
* **Running both at once** is fine: separate processes, the same simulator code. Do not run
  `vla.sh setup` while an ASPIRE run is going: it rewrites `.venv-libero` in place.

### Parity: what was checked

`python -m cap_harness.vla.parity` builds LeRobot's `LiberoEnv` and our adapter on the same
task and scene:

1. After the reset and after every one of 120 scripted actions, the camera images, the hand
   pose, the gripper, the joints and the full MuJoCo state were **identical** (differences
   0.0). This held on `libero_object` task 0 scene 1 and on `libero_object_swap` task 3 scene 7.
2. With `--policy`, one SmolVLA episode ran through LeRobot's own `rollout` and one through
   our `run_episode`, from the same noise seed (CPU, `libero_object` task 0 scene 1). Both
   succeeded in 138 steps, and the largest action difference was 0.0.

`tests/test_vla.py` also checks that our numpy state and image rotation equal LeRobot's
`LiberoProcessorStep` output. That test runs wherever LeRobot is installed.

## The evaluation protocol

| | |
|---|---|
| Scenes | the task's 50 saved init states (`--scenes 1-50`, init mode `saved`); scene k is state k-1. These are the official LIBERO / LIBERO-Pro test scenes |
| Policy seeds | 3 (`--seeds 0,1,2`). The scene and physics are deterministic; the seed changes only SmolVLA's sampling noise. LeRobot recommends averaging over 3 seeds |
| Reset | saved state, then 10 still steps with the gripper open (not counted), as lerobot-eval |
| Episode limit | LeRobot's per-suite limit; LIBERO-Pro suites inherit their base suite's (object 280, goal 300, spatial 280, libero_10 520) |
| Success | the task's own predicate after every step; the episode ends at the first success |
| Checkpoint | `lerobot/smolvla_libero` at revision `a24f1f38…` (the one LeRobot's docs pin) |
| Actions per prediction | **10** of each predicted 50-action chunk, then the policy looks again (`--n-action-steps`; 0 = the checkpoint's 50). Chosen by measurement, below |
| Rendering | 256×256 (`--image-size`; lerobot-eval's default is 360×360) |

Reported per task: successes / episodes, a 95% Wilson interval and the success rate per seed.
The suite mean is the mean of the task rates. The spread across seeds is the standard
deviation of the per-seed suite means. With 150 episodes per task the 95% interval is about
±8 points at 50% success.

**Do not develop or tune on these 50 scenes.** They are the test set. Training demonstrations
will come from seeded layouts (`--init-mode seeded`), checked for near-duplicates of the test
scenes. ASPIRE's own development seeds stay in seeded mode, because in saved mode seeds 51–65
would be the same scenes as 1–15.

### Actions per prediction: measured, not assumed

SmolVLA predicts 50 actions at a time. How many of them run before it looks again changed the
LIBERO-Object result more than anything else (`lerobot/smolvla_libero`, 10 tasks × 10 scenes,
1 seed, same scenes for all three):

| Actions run per prediction | Success | 95% interval | Time per episode |
|---|---|---|---|
| 50 (the value stored in the checkpoint) | 78% | 69–85% | ~5 s |
| **10 (protocol default)** | **93%** | **86–97%** | ~7 s |
| 1 (as the paper describes for simulation) | 83% | 74–89% | ~27 s |

The SmolVLA paper reports 96%, inside the interval for 10. Fifty actions (2.5 s) run blind and
cannot correct a grasp. Re-planning every step draws fresh flow-matching noise each time, and the
hand switches between plans. Community reruns also pointed at this setting (lerobot#2354). Every
SmolVLA run here uses 10, so the baseline and the trained student share one protocol.

## What every episode records

`outputs/vla/<run>/episodes/<suite>/task_<id>/seed_<s>/scene_<k>/`:

| File | Contents |
|---|---|
| `episode.json` | task, instruction, scene and init state, policy seed, success, steps, how it ended, wall time, policy time, and `diagnostics`: share of hand commands at the controller limit, hand path length and net displacement, gripper switches, policy ms per step |
| `steps.npz` | per step: `action` (7), `state` (8, the policy input), `joint_pos` (7), `eef_pos` (3), `eef_quat_xyzw` (4), `gripper_qpos` (2); `final_*` hold the observation after the last action |
| `video.mp4` | both cameras as the policy saw them, side by side, 20 fps (`--video all/failures/none`) |

Each episode is also **graded offline** from simulator state (never a policy input): which
object was lifted, which ended where the task wanted one, and a failure family (`semantic`,
`physical`, `both`, `no_attempt`). The labels are in `episode.json` under `labels`, the
simulator state at the start and end in `steps.npz`, and the counts per task in the results
table. `--no-grading` turns it off. `scripts/gpu/t2v.sh grade-student RUN` adds the labels to
a run made before grading existed, by replaying its stored actions.

And for the whole run, in `outputs/vla/<run>/`:

| File | Contents |
|---|---|
| `run.json` | settings, protocol, policy (source, revision, local path, chunk size), versions (LeRobot commit, PyTorch, MuJoCo, robosuite, GPU) |
| `benchmark.log` | one timestamped line per episode (outcome, steps, time, success so far, ETA), warnings, start and stop |
| `progress.json` | live: status (running, finished, interrupted, failed; stopped if the process died), done/planned, success so far, s per episode, ETA, current episode, warnings |
| `results.json`, `results.md` | the table, refreshed after every episode, with the published numbers to compare against |
| `console.log` | everything the process printed, for runs started with `--detach` |

**Warnings while it runs.** The log, `progress.json`, `vla.sh status` and the dashboard show a
warning when:
* a run with a published SmolVLA number (standard LIBERO) trails it by more than 25 points
  after 20 episodes, which points at the setup rather than the model;
* more than 30% of episodes have most hand commands at the controller limit (a scale or frame
  mismatch);
* more than 30% of episodes have a hand that barely moves (wrong inputs).

A run directory is resumable: rerunning the same command skips finished episodes. It refuses a
different protocol, and a second process while one is still writing.

## Watching a run

* **Dashboard:** `scripts/gpu/dashboard.py`, page **VLA student**. It reads only
  `outputs/vla/`, so ASPIRE's Runs page is unchanged. It shows:
  - the parity checks and every run, with status, progress, success so far, the published
    number and ETA;
  - per run, the per-task table (success, 95% interval, per seed), the episode list (filter by
    task, seed, outcome), the log and the settings;
  - per episode, the video (agentview | wrist, as the policy saw them) synced with charts of
    the hand-position, rotation and gripper commands, the hand position and the gripper
    fingers.

  Pages refresh every 10 s while a run is going.
* **Terminal:** `scripts/gpu/vla.sh status` lists every run. `scripts/gpu/vla.sh status RUN`
  shows one: progress, warnings, the table and the last log lines. Or follow
  `tail -f outputs/vla/RUN/benchmark.log`.
* **Long runs:** add `--detach` (sanity, baseline, benchmark). The run continues after the
  terminal closes. `scripts/gpu/vla.sh stop RUN` ends it with status `interrupted`; the episode in
  progress is dropped and rerun on resume.

## Commands

```bash
scripts/gpu/vla.sh setup
```

```bash
scripts/gpu/vla.sh parity --policy
```

```bash
scripts/gpu/vla.sh sanity --quick
```

```bash
scripts/gpu/vla.sh sanity
```

```bash
scripts/gpu/vla.sh baseline --quick
```

```bash
scripts/gpu/vla.sh baseline
```

```bash
scripts/gpu/vla.sh baseline --detach
```

```bash
scripts/gpu/vla.sh status
```

```bash
scripts/gpu/vla.sh status baseline-smolvla-libero_object_swap-n10
```

```bash
scripts/gpu/vla.sh stop baseline-smolvla-libero_object_swap-n10
```

```bash
scripts/gpu/vla.sh report baseline-smolvla-libero_object_swap-n10
```

* `setup` installs the `vla` extra into `.venv-libero`, downloads the checkpoint and runs
  `tests/test_vla.py`. The first policy load also downloads the SmolVLM2 backbone, about
  2.8 GB in all, into the Hugging Face cache.
* `parity --policy` runs the environment check on `libero_object` and `libero_object_swap`,
  and the SmolVLA episode comparison on CPU. On CPU both paths are bit-exact. GPU kernels are
  not run-to-run deterministic, and the comparison checks the pipeline, not GPU numerics.
* `sanity` runs 10 LIBERO-Object tasks × 50 scenes × 1 seed (`--quick`: 10 scenes) into
  `sanity-libero_object-n10`. Expect close to the paper's 96%.
* `baseline` runs 10 `libero_object_swap` tasks × 50 scenes × 3 seeds = 1,500 episodes
  (`--quick`: 10 scenes, 1 seed) into `baseline-smolvla-libero_object_swap-n10`.
* `--quick` and the full run share a run folder, so the full run resumes from the quick one and
  only runs the missing episodes.

Other suites and options go through `benchmark`:

```bash
scripts/gpu/vla.sh benchmark --suite libero_goal_swap --scenes 1-50 --seeds 0,1,2 --out baseline-smolvla-libero_goal_swap-n10
```

LeRobot's own evaluator, on the same LIBERO-Pro install:

```bash
scripts/gpu/vla.sh lerobot-eval --policy.path=lerobot/smolvla_libero --env.type=libero --env.task=libero_object --eval.batch_size=1 --eval.n_episodes=10 --env.observation_height=256 --env.observation_width=256 --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'
```

lerobot-eval also resets the env internally after a success, so its episodes walk the init
states differently. Use it as an independent check, not for the reported numbers.

## Known differences from published numbers

* No source reports SmolVLA on LIBERO-Pro, so `baseline` is a new measurement. π0.5 is the
  only open model with LIBERO-Pro numbers (object: 0.98 original / 0.17 positions swapped), with
  openpi's own checkpoint. `baseline --student pi05` runs LeRobot's π0.5 checkpoint on the same
  scenes: a number near 17% would confirm the LIBERO-Pro setup independently. It is not the
  same checkpoint, so a gap of several points is not by itself a problem.
* lerobot-eval renders at 360×360 by default. This benchmark renders at 256×256, the size of
  the LIBERO training data.
