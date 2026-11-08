# GPU quickstart: conda env, Opus 4.6 on Bedrock, and a light demo

This is a step-by-step path from a fresh clone to Claude Opus 4.6 writing and debugging robot
programs in simulation. It was written for a shared workstation with 2× NVIDIA RTX PRO 6000
Blackwell (96 GB each). Other users' jobs keep about 20 GB free on GPU 0, and GPU 1 is nearly
full, so **everything runs on GPU 0**.

The demos are deliberately small. Each one proves that one layer works.

| Demo | What it proves | LLM | Wall time | API cost |
| --- | --- | --- | --- | --- |
| `demo.sh smoke` | Simulators, GPU rendering, SAM3, Contact-GraspNet, PyRoki (and cuRobo) work end to end | none | 10–20 min | $0 |
| `demo.sh llm` | Claude Opus 4.6 on Bedrock can write, run and debug a program on 3 dev seeds, then evaluate it on 3 held-out seeds | Opus 4.6 | 20–60 min | roughly $3–15 |
| `demo.sh campaign` | The full ASPIRE loop: coordinator, worker sub-agent, held-out validation, verifier, skill library | Opus 4.6 | about 1 h | roughly $10–30 |

The cost figures are estimates. For scale, the paper's full-size Opus 4.6 runs average about
$24 per LIBERO-Pro task, on 15 dev seeds plus 50 held-out seeds. Bedrock bills at its own rates.
`demo.sh llm` caps spend with `--budget-usd` (default 20).

## 0. Will it fit in about 20 GB on GPU 0?

Yes, with margin. Planned peak memory:

| Component | Estimate | Note |
| --- | --- | --- |
| SAM3 | ~6.3 GiB | 848M-parameter model in fp32 plus bf16 activations |
| Contact-GraspNet | ~2.4 GiB | |
| cuRobo | ~3.4 GiB | optional (`--no-curobo`) |
| PyRoki | 0 | CPU-only JAX build |
| one simulator (MuJoCo EGL) | ~0.8 GiB | the demos run one episode at a time |
| safety margin | 2.0 GiB | |
| **total** | **~15 GiB** (~11.6 GiB without cuRobo) | |

These are estimates from the code, not measurements. The smoke demo records the real peak, so
trust that number after the first run. Free memory on a shared card moves, so check right
before you start:

```bash
python scripts/gpu/check_gpu.py budget            # exit code 0 = fits, 1 = does not
python scripts/gpu/check_gpu.py budget --no-curobo
```

If another card has more room today, move everything there with one command. It sets the
rendering GPU in the env, and `services.sh` places every model service on the same card:

```bash
scripts/gpu/use_gpu.sh 1
```

Then reload the env with `conda deactivate && conda activate aspire`. You can also choose the card
at setup time with `scripts/gpu/setup_conda_env.sh --gpu 1`.

## 1. Get the code

```bash
git clone https://github.com/hosseinhkh/ASPIRE.git
```

```bash
cd ASPIRE
```

```bash
git checkout claude/dazzling-pascal-4r6ipf
```

## 2. Create the conda env and install every dependency

```bash
scripts/gpu/setup_conda_env.sh
```

This creates conda env `aspire` (from conda-forge only) with:

- Python 3.11
- uv, git, git-lfs, ffmpeg
- Node 22, boto3
- the **Claude Code CLI**

It then builds the component venvs from the frozen `uv.lock`:

| Venv | Purpose |
| --- | --- |
| `.venv-robosuite`, `.venv-libero` | simulators |
| `.venv-sam3`, `.venv-graspnet`, `.venv-pyroki`, `.venv-curobo` | model services |
| `.venv-dev` | tests |

Why both conda and venvs: the model services pin conflicting stacks (JAX vs torch builds), so
upstream keeps one venv per service. The conda env is the toolchain that builds and launches
them.

The script also stores the rendering and GPU variables in the env: `MUJOCO_GL=egl`,
`EGL_PLATFORM=device`, `CUDA_VISIBLE_DEVICES=0`, `MUJOCO_EGL_DEVICE_ID=0`, `LIBERO_CONFIG_PATH`
and the profile `rtxpro6000`. Finally it runs the unit tests.

What to expect:

- About 30 minutes on first run.
- About 15 GB of downloads; torch is fetched once and hard-linked between venvs.
- About 30 GB on disk.
- No system CUDA toolkit is needed: every GPU wheel is prebuilt, and cuRobo compiles its kernels
  at runtime.

Options:

| Option | Effect |
| --- | --- |
| `--no-curobo` | skip the cuRobo planner |
| `--gpu N` | use a different card |
| `--env NAME` | use a different env name |
| `--no-tests` | skip the unit-test pass |

Then reload the env so the stored variables apply:

```bash
conda activate aspire
```

## 3. Credentials: AWS Bedrock (Opus 4.6) and Hugging Face (SAM3)

Before you start:

- In the AWS console, enable **Claude Opus 4.6** under Bedrock → Model access, in your region.
- Request access to the gated SAM3 model at https://huggingface.co/facebook/sam3.

### Option A: interactive (recommended; nothing lands in shell history)

```bash
scripts/gpu/set_credentials.sh
```

It asks for:

1. **Region:** default `us-east-1`.
2. **Authentication method:**
   - access key + secret (+ session token), or
   - a Bedrock API key (`AWS_BEARER_TOKEN_BEDROCK`), or
   - an AWS profile / SSO.
3. **Hugging Face token.**

Secret input is hidden. The script then:

- **Finds your Opus 4.6 inference profile** through `ListInferenceProfiles`, e.g.
  `us.anthropic.claude-opus-4-6-v1`. The bare model ID is rejected for on-demand use.
- **Stores the variables in the conda env**, and makes its state file readable only by you:
  - `CLAUDE_CODE_USE_BEDROCK=1` and `AWS_REGION`;
  - your credentials;
  - `ANTHROPIC_MODEL`, plus `ANTHROPIC_DEFAULT_OPUS_MODEL`, `ANTHROPIC_DEFAULT_SONNET_MODEL` and
    `ANTHROPIC_DEFAULT_HAIKU_MODEL`, all set to Opus 4.6. ASPIRE's coordinator spawns sub-agents
    with `model="opus"`, and pinning the aliases keeps every agent on 4.6.
- **Verifies Claude Code** with `claude -p "Reply with exactly the word OK."`.
- **Pre-downloads the SAM3 weights** (about 3.4 GB) so the first service start is quick.

Flags:

| Flag | Effect |
| --- | --- |
| `--region eu-west-1` | set the region |
| `--model-id ID` | skip detection and use this model ID |
| `--1m` | use the 1M-token context window |
| `--unset` | remove everything the script stored |

Reload when done:

```bash
conda deactivate
```

```bash
conda activate aspire
```

### Option B: plain commands

This is the same result as option A, but the keys end up in your shell history.

```bash
conda activate aspire
```

```bash
conda env config vars set CLAUDE_CODE_USE_BEDROCK=1 AWS_REGION=us-east-1 \
  AWS_ACCESS_KEY_ID='AKIA...' AWS_SECRET_ACCESS_KEY='...' \
  ANTHROPIC_MODEL='us.anthropic.claude-opus-4-6-v1' \
  ANTHROPIC_DEFAULT_OPUS_MODEL='us.anthropic.claude-opus-4-6-v1' \
  ANTHROPIC_DEFAULT_SONNET_MODEL='us.anthropic.claude-opus-4-6-v1' \
  ANTHROPIC_DEFAULT_HAIKU_MODEL='us.anthropic.claude-opus-4-6-v1' \
  HF_TOKEN='hf_...'
```

```bash
conda deactivate
```

```bash
conda activate aspire
```

```bash
claude -p "Reply with exactly the word OK." --max-turns 1
```

For temporary credentials, add `AWS_SESSION_TOKEN=...`. For a Bedrock API key, use
`AWS_BEARER_TOKEN_BEDROCK=...` instead of the key pair. To list the Opus 4.6 profiles your
account can use:

```bash
python -c "import boto3; print([p['inferenceProfileId'] for p in boto3.client('bedrock', region_name='us-east-1').list_inference_profiles(maxResults=1000)['inferenceProfileSummaries'] if 'opus-4-6' in p['inferenceProfileId']])"
```

## 4. Start the model services on GPU 0

```bash
python scripts/gpu/check_gpu.py budget
```

```bash
scripts/gpu/services.sh start
```

Notes:

- Add `--no-curobo` if memory is tight.
- `start` launches SAM3 (:8114), Contact-GraspNet (:8115), PyRoki (:8116) and cuRobo (:8118),
  detached from the terminal, so they survive a closed SSH session.
- It waits until every service is ready, then prints status and real GPU usage.
- cuRobo's first plan takes about 50 s while it warms up; later plans take milliseconds.

Other service commands:

| Command | Effect |
| --- | --- |
| `scripts/gpu/services.sh status` | ports plus GPU memory of your services |
| `scripts/gpu/services.sh logs sam3` | follow one service's log |
| `scripts/gpu/services.sh stop` | stop everything |

## 5. Demo 1: smoke test, no LLM

```bash
scripts/gpu/demo.sh smoke
```

It runs, in order:

1. `cap-harness doctor` for both simulators: environment, GPU, EGL, services.
2. Three full episodes of the shipped example programs:
   - Robosuite cube lifting;
   - LIBERO-Pro object swap ("pick up the alphabet soup and place it in the basket");
   - cuRobo cube lifting, when cuRobo is running.
3. A line per episode with the result, and a line per episode with the model services it called
   (e.g. `services: sam3 2 calls (2 ok), contact-graspnet 1 calls (1 ok), pyroki 9 calls (9 ok)`).
4. Peak GPU memory of your processes.
5. A **SMOKE PASS / FAIL** verdict.

PASS requires every program to finish, no service errors, and at least one successful SAM3
segmentation. `task_success` is reported but not required, because the examples succeed on most
seeds, not all. Add `--with-bimanual` to also run the two-arm lift.

Results go to `outputs/gpu-demo/smoke-<time>/`. Each run directory has:

| Path | Contents |
| --- | --- |
| `outcome.json` | the result |
| `trace/` | every API call and service call |
| `media/videos/` | episode video |
| `media/keyframes/` | before and after each motion |

## 6. Demo 2: Opus 4.6 writes and debugs a program

```bash
scripts/gpu/demo.sh llm
```

Claude Code runs headless with Opus 4.6. It reads the API reference and the skill library, then:

1. It writes its own program for LIBERO-Pro `libero_object_swap` task 0. It is told not to read
   the shipped examples.
2. It runs the program on development seeds 51–53 and reads the traces and keyframes.
3. It fixes failures, for up to 4 versions.
4. It evaluates the best version once on held-out seeds 1–3.
5. It writes `REPORT.md`: versions, success rates, failure modes, and candidate skills.

You see a live one-line-per-step view. The final line shows turns, minutes and estimated cost.

Other tasks and benchmarks:

```bash
scripts/gpu/demo.sh llm --benchmark robosuite
```

The Robosuite run uses cube lifting with dev seeds 101–103 and held-out seeds 1–3.

```bash
scripts/gpu/demo.sh llm --suite libero_goal_swap --task-id 8 --budget-usd 10
```

Output goes to `outputs/gpu-demo/llm-<benchmark>-<suite>-<task>-<time>/`:

| File | Contents |
| --- | --- |
| `prompt.md` | the prompt the agent received |
| `transcript.jsonl` | full transcript |
| `program_v*.py` | each program version |
| `dev_v*/`, `heldout/` | run directories |
| `REPORT.md` | the agent's report |
| `result.json` | cost, turns, duration |

## 7. Demo 3: the full ASPIRE loop on one task

```bash
scripts/gpu/demo.sh campaign --run
```

This initializes a tiny campaign under `outputs/runs/<time>_gpu-demo/`:

- 1 task;
- dev seeds 51–53;
- unseen-check seed 66;
- held-out seeds 1–3.

It then starts an **interactive** Claude Code session as the ASPIRE coordinator, in single-GPU
mode. The coordinator:

1. dispatches the worker sub-agent;
2. runs every episode itself;
3. runs Stage-2 held-out validation;
4. dispatches the verifier if the worker proposed a skill, and promotes verified skills.

It finishes with `DEMO_SUMMARY.md`. You can watch and steer it.

Without `--run`, the script prints the exact `claude` command instead. Check progress any time:

```bash
python .claude/skills/iterative-debugging/scripts/gen_progress.py --print
```

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `budget` says it does not fit | Use `--no-curobo`, wait for other users' jobs, or move to the emptier card with `scripts/gpu/use_gpu.sh N` |
| `qt-main_activate.sh: QT_XCB_GL_INTEGRATION: unbound variable` | A conda activation hook that breaks under `set -u`. Fixed in the scripts; `git pull` and re-run. |
| `PermissionError: '/tmp/robosuite.log'` | robosuite logs to a fixed `/tmp` file that another user owns. Setup now writes `third_party/robosuite/robosuite/macros_private.py`, which turns that log off. `git pull` and re-run `setup_conda_env.sh`; you don't need to delete anyone's file. |
| CUDA OOM in a service log | Another user's job grew. Run `services.sh stop`, `check_gpu.py budget`, then restart, without cuRobo if needed. |
| `EGLError` / blank images | Make sure the env is active (`MUJOCO_GL=egl`, `EGL_PLATFORM=device`, `MUJOCO_EGL_DEVICE_ID=0`). An NVIDIA EGL vendor file must exist (`ls /usr/share/glvnd/egl_vendor.d/`). |
| SAM3 fails to start with a 401 | The HF token lacks access to `facebook/sam3`. Request access, then re-run `set_credentials.sh`. |
| `on-demand throughput isn't supported` | You passed the bare model ID. Use the inference profile (`us.` / `eu.` / `global.` + `anthropic.claude-opus-4-6-v1`). |
| `AccessDeniedException` from Bedrock | Enable Opus 4.6 in Bedrock model access. The IAM principal needs `bedrock:InvokeModel`, `bedrock:InvokeModelWithResponseStream`, `bedrock:ListInferenceProfiles`. |
| Services died with the terminal | Use `scripts/gpu/services.sh start`. It runs them under `setsid`, so closing SSH does not kill them. |
| conda asks to accept Terms of Service | The setup uses `--override-channels -c conda-forge`. If your base conda still prompts, run `conda config --remove channels defaults`. |
| `claude` refuses `--permission-mode bypassPermissions` | Run the demos as your normal user, not root. |
| LIBERO reset crashes with "truth value of an array is ambiguous" | You are on the upstream release. This branch fixes it (`docs/review/findings.md` A1). |

Safety note from upstream: generated programs run with full Python access on this machine.
Don't run the demos on a host that holds credentials you can't afford to expose; the Bedrock
keys stored in this env are such credentials.
