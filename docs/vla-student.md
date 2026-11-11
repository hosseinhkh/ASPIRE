# VLA student: SmolVLA on LIBERO and LIBERO-Pro

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
* **Outputs.** Runs go to `outputs/vla/`. The dashboard does not list that folder yet, and
  nothing there is read by ASPIRE.
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
| Checkpoint | `lerobot/smolvla_libero` at revision `a24f1f38…` (the one LeRobot's docs pin); 50-action chunks as stored in the checkpoint (`--n-action-steps` to change) |
| Rendering | 256×256 (`--image-size`; lerobot-eval's default is 360×360) |

Reported per task: successes / episodes, a 95% Wilson interval and the success rate per seed.
The suite mean is the mean of the task rates. The spread across seeds is the standard
deviation of the per-seed suite means. With 150 episodes per task the 95% interval is about
±8 points at 50% success.

**Do not develop or tune on these 50 scenes.** They are the test set. Training demonstrations
will come from seeded layouts (`--init-mode seeded`), checked for near-duplicates of the test
scenes. ASPIRE's own development seeds stay in seeded mode, because in saved mode seeds 51–65
would be the same scenes as 1–15.

## What every episode records

`outputs/vla/<run>/episodes/<suite>/task_<id>/seed_<s>/scene_<k>/`:

| File | Contents |
|---|---|
| `episode.json` | task, instruction, scene and init state, policy seed, success, steps, how it ended, wall time, policy time |
| `steps.npz` | per step: `action` (7), `state` (8, the policy input), `joint_pos` (7), `eef_pos` (3), `eef_quat_xyzw` (4), `gripper_qpos` (2); `final_*` hold the observation after the last action |
| `video.mp4` | both cameras as the policy saw them, side by side, 20 fps (`--video all/failures/none`) |

`run.json` holds the settings, the protocol, the policy (source, revision, local path, chunk
size) and the versions (LeRobot commit, PyTorch, MuJoCo, robosuite, GPU). `results.json` and
`results.md` hold the table. A run directory is resumable: rerunning the same command skips
finished episodes. A different protocol in the same directory is refused.

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
scripts/gpu/vla.sh report outputs/vla/baseline-smolvla-libero_object_swap
```

* `setup` installs the `vla` extra into `.venv-libero`, downloads the checkpoint and runs
  `tests/test_vla.py`. The first policy load also downloads the SmolVLM2 backbone, about
  2.8 GB in all, into the Hugging Face cache.
* `parity --policy` runs the environment check on `libero_object` and `libero_object_swap`,
  and the SmolVLA episode comparison on CPU. On CPU both paths are bit-exact. GPU kernels are
  not run-to-run deterministic, and the comparison checks the pipeline, not GPU numerics.
* `sanity` runs 10 LIBERO-Object tasks × 50 scenes × 1 seed (`--quick`: 10 scenes). Expect
  close to the paper's 96%. Community reruns have reported 91%, and differences in
  `n_action_steps` were suspected (lerobot#2354).
* `baseline` runs 10 `libero_object_swap` tasks × 50 scenes × 3 seeds = 1,500 episodes
  (`--quick`: 10 scenes, 1 seed).

Other suites and options go through `benchmark`:

```bash
scripts/gpu/vla.sh benchmark --suite libero_goal_swap --scenes 1-50 --seeds 0,1,2 --out outputs/vla/baseline-smolvla-libero_goal_swap
```

LeRobot's own evaluator, on the same LIBERO-Pro install:

```bash
scripts/gpu/vla.sh lerobot-eval --policy.path=lerobot/smolvla_libero --env.type=libero --env.task=libero_object --eval.batch_size=1 --eval.n_episodes=10 --env.observation_height=256 --env.observation_width=256 --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'
```

lerobot-eval also resets the env internally after a success, so its episodes walk the init
states differently. Use it as an independent check, not for the reported numbers.

## Known differences from published numbers

* No source reports SmolVLA on LIBERO-Pro, so `baseline` is a new measurement. π0.5 is the
  only open model with LIBERO-Pro numbers (object: 0.98 original / 0.17 positions swapped). A
  π0.5 run would confirm the LIBERO-Pro setup independently.
* lerobot-eval renders at 360×360 by default. This benchmark renders at 256×256, the size of
  the LIBERO training data.
