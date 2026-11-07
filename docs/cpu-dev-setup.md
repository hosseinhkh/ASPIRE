# CPU-only development setup

This is the setup used to review and develop ASPIRE on a machine **without an NVIDIA GPU**
(validated on Ubuntu 24.04, x86_64, 4 vCPU, 15 GB RAM). It runs the harness, the full unit-test
suite, the Robosuite and LIBERO-Pro simulators, and PyRoki IK. It cannot run the CUDA-only
providers. For the full stack, follow [`host-setup.md`](host-setup.md) on a GPU host.

## What runs where

| Component | CPU-only host | Needs | Notes |
| --- | --- | --- | --- |
| `cap_harness` core, sandbox, recorder | yes | Python 3.10–3.12 | |
| Unit tests (`tests/`) | yes | `.venv-dev` | 700+ tests; about 6.5 min on 4 vCPU |
| Robosuite simulator | yes | OSMesa | ≈0.35 s per control tick at 256×256 with 3 cameras |
| LIBERO-Pro simulator | yes | OSMesa | The venv includes torch (CUDA wheels, used on CPU); about 7 GB |
| PyRoki IK provider (port 8116) | yes | JAX on CPU | Panda by default (`--robot`, `--target-link`) |
| Joint-interpolation planner | yes | | Built into `CapApi.plan_motion`; **not collision-aware** |
| SAM3 segmentation (8114) | GPU recommended | CUDA, gated HF weights | Every `localize_object` / `segment_text` call needs it |
| Contact-GraspNet (8115) | no | CUDA | Non-commercial license |
| cuRobo (8118) | no | CUDA, source build | Collision-aware planning and integrated pose planning |
| BEHAVIOR-1K | no | Isaac Sim 5.1, RTX GPU, 40 GB datasets | |
| Real YAM / System 2 | sim mode only | `.venv-dev` | `scripts/start_system2.sh --sim` (synthetic cameras, no objects) |

So on CPU you can only run programs that need no perception or grasp service, such as
`examples/robosuite/bimanual_control_smoke.py` and `examples/libero/close_microwave.py`. Every
other shipped example calls SAM3.

## One command

```bash
scripts/bootstrap_cpu_dev.sh --install-system-deps
```

What the script does:

1. **Pick a Python 3.10–3.12 interpreter.** Ubuntu 24.04 can ship a newer `python3`, and the GPU
   bootstraps default to `python3`. The script prefers `python3.11`, and otherwise lets uv
   download one.
2. **Install Mesa OSMesa/EGL** (`libosmesa6 libegl1 libegl-mesa0 libglu1-mesa ffmpeg`).
3. **Shallow-clone the submodules** LIBERO-PRO, robosuite, sam3 and contact_graspnet_pytorch.
   BEHAVIOR-1K is skipped.
4. **Run `bootstrap_robosuite.sh`, `bootstrap_libero.sh` and `bootstrap_pyroki.sh`.** They use
   the same frozen `uv.lock` as the GPU path.
5. **Build `.venv-dev`** with the `dev`, `agent`, `yam-real` and `robosuite` extras, which are
   needed to collect the whole test suite.

## Environment for headless rendering

EGL needs an NVIDIA device; with Mesa only, `MUJOCO_GL=egl` fails with an `EGLError`. Use OSMesa:

```bash
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export LIBERO_CONFIG_PATH="$PWD/.libero"
```

## Smoke tests that pass on CPU

Run the full test suite:

```bash
.venv-dev/bin/python -m pytest -q
```

Provider-free Robosuite smoke. Expected: `program_ok=True`. `task_success=False` is expected,
because this is a control check, not a lift.

```bash
.venv-robosuite/bin/cap-harness run --benchmark robosuite --suite two_arm_lift --task-id 0 \
  --seed 1 --program examples/robosuite/bimanual_control_smoke.py \
  --output-root outputs/cpu-smoke --camera-width 256 --camera-height 256
```

LIBERO-90 close-microwave, using PyRoki IK. Start the IK service first:

```bash
JAX_PLATFORMS=cpu .venv-pyroki/bin/python -m cap_harness.providers.pyroki.service --port 8116 &
```

Expected: `task_success=True`.

```bash
.venv-libero/bin/cap-harness run --benchmark libero-pro --suite libero_90 --task-id 33 \
  --seed 1 --program examples/libero/close_microwave.py --output-root outputs/cpu-smoke \
  --max-steps 2000 --camera-width 256 --camera-height 256
```

The LIBERO command needs the init-state fix in `libero/adapter.py`. Before it, every LIBERO reset
in the default `--init-mode saved` crashed with `truth value of an array ... is ambiguous`. See
[`review/findings.md`](review/findings.md).

## Performance notes

- On CPU, rendering dominates. In the Robosuite smoke, `adapter_step` took about 23 s of a 36 s
  run for 65 steps, rendering two eye-in-hand cameras plus agent-view RGB-D. Keep cameras at
  256×256 or smaller for iteration. Use a GPU with `MUJOCO_GL=egl` for campaigns.
- A 1000-step LIBERO episode takes about 6 minutes on 4 vCPU, so paper-scale campaigns (15 dev
  seeds × many attempts per task) are not practical on CPU.

## Next step: a GPU host

For SAM3, Contact-GraspNet and cuRobo you need:

- An Ada (sm_89) or Blackwell (sm_120) GPU. For Ampere (sm_86), add an arch entry to
  `configs/dependency-lock.json` and a profile; see upstream issue NVlabs/ASPIRE#28.
- A Hugging Face token with access to the gated SAM3 model.

Then follow [`host-setup.md`](host-setup.md) and [`environments.md`](environments.md).
