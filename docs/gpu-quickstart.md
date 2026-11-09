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
| `session.sh run` | Many tasks with everything recorded: videos, trajectories, every model call, the agent's conversation, the GPU timeline; HTML report and W&B ([section 8](#8-recorded-test-sessions-videos-every-decision-wb)) | optional | 5 min–1 h | $0 without `--llm` |

The cost figures are estimates. For scale, the paper's full-size Opus 4.6 runs average about
$24 per LIBERO-Pro task, on 15 dev seeds plus 50 held-out seeds. Bedrock bills at its own rates.
`demo.sh llm` caps spend with `--budget-usd` (default 20).

## Already set up? Update, then check everything

If the `aspire` env already exists from an earlier version:

```bash
git pull
```

```bash
scripts/gpu/setup_conda_env.sh --gpu 1 --no-tests
```

```bash
conda deactivate && conda activate aspire
```

The re-run is quick: finished venvs are skipped. It adds:

- the robosuite `/tmp/robosuite.log` fix;
- the GPU choice (use the card with free memory);
- the per-user cuRobo cache.

Your Bedrock and Hugging Face settings stay as they are.

Then check the whole stack. The check needs no SAM3, so it works while the access request is
pending:

```bash
scripts/gpu/check_stack.sh --quick
```

```bash
scripts/gpu/check_stack.sh
```

```bash
scripts/gpu/check_stack.sh --with-llm
```

| Command | What it covers | Time / cost |
| --- | --- | --- |
| `--quick` | sections 1–7 | about 2 min, $0.01 |
| (no flag) | everything except the agent check | about 10 min |
| `--with-llm` | everything, including a small agent coding task | about $1–3 |

| # | Check | Passes when |
| --- | --- | --- |
| 1 | env | the conda env variables are loaded and point rendering at your GPU |
| 2 | gpu | the driver answers and the stack fits in the GPU's free memory right now |
| 3 | reserve | `reserve.sh` can hold 1 GiB, `nvidia-smi` shows it, and release frees it |
| 4 | torch | every torch venv runs a CUDA kernel on the Blackwell card (catches a wrong CUDA build) |
| 5 | egl | MuJoCo renders through NVIDIA EGL; it warns if it fell back to slow software rendering |
| 6 | sam3 | Hugging Face access to `facebook/sam3`: **BLOCKED** while your request is pending |
| 7 | bedrock | API-key expiry; Opus 4.6 answers through boto3 Converse and ConverseStream; Claude Code answers |
| 8 | services | PyRoki, Contact-GraspNet and cuRobo start and answer (SAM3 too once granted) |
| 9 | sims | a Robosuite two-arm episode (no services) and a LIBERO close-microwave episode (PyRoki IK) |
| 10 | probe | one Robosuite episode calls every service through the robot API: PyRoki IK and motion, cuRobo IK and both planners, Contact-GraspNet grasps from an image-region mask, and a SAM3 call |
| 11 | agent | `--with-llm` only: Opus 4.6 writes a small program (lift the arm 8 cm and back), runs it, reads the result and reports |

Statuses:

| Status | Meaning |
| --- | --- |
| PASS | works |
| WARN | works, with a caveat |
| FAIL | something on this machine needs fixing; the summary lists each with a hint |
| BLOCKED | waiting on something outside this machine |
| SKIP | not run (e.g. `--no-curobo`) |

The exit code is 1 if anything FAILed. Logs go to `outputs/gpu-check/<time>/`. The check stops
any services it started; add `--keep-services` to keep them running for the demos.

**While SAM3 is pending,** expect BLOCKED only for SAM3 (sections 6, 8 and 10) and PASS
elsewhere. Every demo needs SAM3 except the two-arm smoke and the microwave episode, which section
9 already runs. Once Hugging Face approves the request, re-run the check: SAM3 turns PASS, and
the services start it automatically. To download its weights (~3.4 GB) ahead of time:

```bash
.venv-sam3/bin/python -c "from huggingface_hub import hf_hub_download as d; print(d('facebook/sam3', 'sam3.pt'))"
```

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

### If you use a Bedrock API key (`AWS_BEARER_TOKEN_BEDROCK`)

This is the same credential AWS's own Python example uses: `boto3.client("bedrock-runtime",
region_name="us-east-1").converse(...)`, with the key in `AWS_BEARER_TOKEN_BEDROCK`. Claude Code
accepts it directly. Choose **2** in the menu, or skip the menu:

```bash
scripts/gpu/set_credentials.sh --method 2
```

With a key, the script does three extra things:

- **Region.** It reads the key's region and uses that region, because a short-term key only works
  in the region it was created in.
- **Expiry.** It prints when the key expires. Short-term keys
  (`bedrock-api-key-...`) last at most **12 hours**. `demo.sh llm` and `demo.sh campaign` refuse
  to start if the key won't outlive the run.
- **Two checks.** It calls Bedrock directly through boto3 Converse, as in AWS's example but with
  Opus 4.6. This separates "the key or model access is wrong" from "Claude Code is misconfigured".
  Then it verifies Claude Code.

With a key there is no account call to discover the model ID, so it uses
`us.anthropic.claude-opus-4-6-v1` (or `eu.` / `apac.` for keys from those regions). Pass
`--model-id` to override.

When the key expires, replace only the key:

```bash
scripts/gpu/set_credentials.sh --api-key
```

```bash
conda deactivate && conda activate aspire
```

A key is a password: don't paste it into chats, issues or commits. For runs longer than a few
hours, use a long-term API key or IAM access keys (option 1).

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

## 4. Start the model services

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

### Keep GPU memory for yourself (optional)

On a busy server you can hold memory on a card until you need it, e.g. 30 GiB on GPU 1:

```bash
scripts/gpu/reserve.sh hold 30 --gpu 1
```

```bash
scripts/gpu/reserve.sh status
```

```bash
scripts/gpu/reserve.sh release
```

A small detached process holds the memory, so it survives closing the terminal. It takes as much
as is free if the card has less than you asked for.

`services.sh` hands the memory over automatically:

- **`services.sh start` on the same GPU:** releases the hold, starts the services, then re-holds
  your target minus what the services need at peak, leaving about 2 GiB free for the simulator.
  Your total stays near 30 GiB.
- **`services.sh stop`:** restores the full hold.

Memory is unheld for the minute or so the services take to start, so on a busy card someone may
take part of it. Holding idle memory also blocks colleagues, so agree on it with whoever shares
the machine.

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

You see a live one-line-per-step view. The final line shows turns, minutes and estimated cost,
and the context window and thinking tokens the model really used.

Claude runs with the paper's Opus 4.6 settings: max effort and the 1M-token context. For Claude
Code's defaults (high effort, 200k window; cheaper), put `ASPIRE_EFFORT=high ASPIRE_1M=0` in
front of the command. Episodes use the harness defaults the paper used (800×512 cameras, the
task's own episode length).

This demo is a smoke test: 3 development seeds, 3 held-out seeds, at most 4 versions and a short
prompt. Its score is not comparable with the paper's; for that, run
`scripts/gpu/demo.sh campaign --paper` (section 7).

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

Without `--run`, the script prints the exact `claude` command instead.

**The paper's protocol.** `--paper` uses the paper's seed partitions (LIBERO-Pro: dev 51–65,
unseen-check 66–80, held-out 1–50; Robosuite: 101–125, 126–130, 1–100), lifts the 4-version cap
so workers debug until the threshold or a plateau, and caps the run at $40 per task:

```bash
scripts/gpu/demo.sh campaign --paper --headless
```

The default task is LIBERO-Pro `libero_object_swap` task 0, where the paper reports 100% on
50 held-out seeds with Opus 4.6 (Table 10). Pass `--tasks FILE` for others.

Check progress any time:

```bash
python .claude/skills/iterative-debugging/scripts/gen_progress.py --print
```

## 8. Recorded test sessions (videos, every decision, W&B)

`session.sh run` runs a list of simulator tasks, and optionally Opus 4.6 agents, with everything
recorded in one directory. It then builds an HTML report and Weights & Biases runs. Use it when
you want to look back at what happened: what each camera saw, every trajectory, every call to
SAM3 / Contact-GraspNet / PyRoki / cuRobo and its output, and the agent's whole conversation.

```bash
scripts/gpu/session.sh tasks
```

```bash
scripts/gpu/session.sh run --tasks quick --reserve 30
```

```bash
scripts/gpu/session.sh run --tasks standard --llm libero-pro:libero_object_swap:0 --llm-budget 10 --reserve 30
```

| Preset | Episodes per seed | Wall time (estimate) |
| --- | --- | --- |
| `quick` | 3: cube lift, microwave, object swap | 5–10 min |
| `standard` (default) | 7: adds the cuRobo cube lift, two-arm lift, two-arm handover and the drawer | 20–30 min |
| `full` | 14: every Robosuite and LIBERO-Pro example program | about 1 h |

In order, a session:

1. records the git commit and diff, `nvidia-smi`, versions, the package list of every venv and
   the ASPIRE environment variables. Credential values are never written, only `<set, N chars>`;
2. runs `check_stack.sh --quick` (skip with `--no-preflight`);
3. starts a GPU timeline: memory, utilization, temperature and power every 2 s, plus each process
   with its role (SAM3, cuRobo, simulator, reservation, other users);
4. starts the model services if they are not running. A reservation on the GPU is handed over
   automatically. Services that were already running are used as they are and never stopped;
5. runs each task once per seed (`--seeds "1 2 3"`). A task whose service is down is skipped;
6. runs each `--llm BENCHMARK:SUITE:TASK_ID` agent (the same agent as `demo.sh llm`, with
   `--llm-budget`, `--llm-max-turns`, `--llm-dev-seeds` and `--llm-heldout-seeds`);
7. stops the services it started, then holds `--reserve GIB` on the GPU for you. A
   reservation you hold on another GPU is left where it is;
8. scrubs credential values from every file, builds the report, and logs to W&B.

Ctrl-C stops a session cleanly:

- the running episode or agent is marked INTERRUPTED;
- the services the session started are stopped;
- `--reserve` is still applied;
- a report of everything recorded so far is written.

Editing or pulling the scripts while a session runs is safe.

Each session lands in `outputs/sessions/<date>-<time>-<name>/`:

| Path | Contents |
| --- | --- |
| `report/index.html` | **start here**: stages with timing and your GPU peak per stage, GPU charts with the stages shaded, links to every episode and agent page |
| `report/episode-NNN.html` | one episode: outcome, every camera's video, joint / gripper / end-effector trajectories, keyframes before and after each motion, SAM3 mask overlays, timing, and the tool-call tree (program call → API stage → model service) with each call's input, output and error, plus the program and its stdout |
| `report/agent-NNN.html` | one agent conversation: the prompt; Opus 4.6's text and thinking; every tool call (shell commands, file reads) and its result; every image it was shown; turns, minutes and cost; its report and the programs it wrote |
| `episodes/` | the raw run directories (`outcome.json`, `trace/`, `media/videos/*.mp4`, `episode/steps.jsonl`), and `logs/` with each episode's console |
| `agents/llm-*/` | each agent's workspace: `prompt.md`, `transcript.jsonl`, `program_v*.py`, `dev_v*/` and `heldout/` runs, `REPORT.md` |
| `claude-sessions/` | Claude Code's own transcripts of sub-agents (and of interactive sessions, see below) |
| `gpu/` | `gpu.csv` and `processes.csv` (the timeline) |
| `service-logs/` | what each model service logged during the session |
| `preflight/`, `env/` | the quick check and the environment snapshot |
| `stages.tsv`, `session.json`, `session.log` | stage results with start and end times, session metadata, the whole console |
| `wandb/` | offline W&B runs |

To view the report on the shared server, serve the session directory. Then open an SSH tunnel
from your laptop and browse to `http://localhost:8000/report/`:

```bash
cd outputs/sessions/<session> && python -m http.server 8000 --bind 127.0.0.1
```

```bash
ssh -L 8000:localhost:8000 <you>@<server>
```

**Weights & Biases.** By default the runs are written offline. The first session installs wandb
into `.venv-tools`, so no account is needed to record. Each episode, agent and session is one W&B
run, grouped by session name:

- episode runs log the joint, gripper, end-effector and commanded trajectories as charts, the
  videos, keyframes and SAM3 overlays, and a `tool_calls` table with inputs and outputs;
- agent runs log the transcript as a table and the images the agent saw;
- the session run logs the GPU timeline and the stage table;
- the raw directories are uploaded as artifacts.

To upload, log in once (the key is read at a hidden prompt), then sync:

```bash
.venv-tools/bin/wandb login
```

```bash
.venv-tools/bin/wandb sync outputs/sessions/<session>/wandb/wandb/offline-run-*
```

Or log straight to W&B with `--wandb online`. `--wandb off` skips it; the HTML report is
complete on its own. To rebuild the report and W&B runs for an existing session:

```bash
scripts/gpu/session.sh report outputs/sessions/<session> --wandb offline
```

**Interactive campaigns.** `demo.sh campaign --run` is interactive, so it runs outside
`session.sh`. Afterwards, copy its Claude Code transcripts, sub-agents included, into a session
and rebuild the report:

```bash
scripts/gpu/session.sh claude-logs outputs/sessions/<session>
```

```bash
scripts/gpu/session.sh report outputs/sessions/<session>
```

Your own task list uses the columns that `session.sh tasks` prints, one task per line. Pass it
with `--tasks FILE`.

To see why episodes did not succeed, print a plain-text summary of each one: how it ended,
every failed call with its error, the program's `result`, and its last printed lines:

```bash
scripts/gpu/session.sh why outputs/sessions/<session>
```

Add `--graspnet-cpu` to `session.sh run` to run Contact-GraspNet on the CPU (see
Troubleshooting).

## 9. Dashboard and live robots

The dashboard shows everything that was recorded and can drive a running robot. It needs no
extra packages:

```bash
python scripts/gpu/dashboard.py
```

It uses port 8765, or the next free one if someone else has it, and prints the address and
the `ssh -L` command to use from your laptop. For example, if it prints port 8766:

```bash
ssh -L 8766:localhost:8766 <you>@<server>
```

Then open http://localhost:8766. To choose the port yourself, pass `--port N`.

| Page | What you get |
| --- | --- |
| Runs | Every session, live-console run, check, demo and campaign under `outputs/`, with success rate, LLM cost, GPU and git commit. Filter them, tick two or more and press **Compare**. Each has a **zip download**: complete (videos and images included) or logs only |
| Session | Summary cards and a stage timeline. Tabs: **Tasks** (success per task, agent and phase; seeds link to their replays; click a row for every run with how it ended, its error and Replay / Logs / Full / Program buttons; **Logs** and **Full** zips per row; tick rows to compare them, for example two program versions), **Failures** (unsolved episodes grouped by how they failed, numbers and paths blanked, with a zip of each group's logs), **Episodes** (thumbnails, filter by task or failed), **Agents**, **Stages**, **GPU** (memory and utilization with the stages shaded), **Logs** (open in the viewer or download) and the session details |
| Episode | A replay. Every camera plays in sync with play, pause, frame step, speed and a scrub bar showing the simulated time and step. Below it, a **timeline** of every call in three lanes: the program's calls, the API stages inside them (IK, planning, execution) and each model request (SAM3, Contact-GraspNet, PyRoki, cuRobo). The **chain of calls** shows how they nest, with a failed-only filter. Clicking a call jumps the videos there and shows: its keyframes before and after; its SAM3 masks, which you can switch on one candidate at a time over the clean image, with scores (low ones are flagged); the grasp candidates drawn on the camera image; the point clouds and boxes it located; for a motion, a **joint tracking** chart (commanded vs measured joints and the gap between them); and its full input, output and error. Warnings at the top point at low-confidence SAM3 calls and at runs that ended without the task done. Tabs: trajectories (with a line that follows the video), **perception** (every mask and grasp), **3D scene** (point clouds, boxes, grasps, motion targets and the hand's path in the robot base frame; drag to rotate), keyframes, the program and its output, live-console commands and files |
| Agent | **How it debugged**: each step of the loop as a card: the agent's reasoning, the code change (as a diff), and the batch it ran with every seed solved or not, each linking to that episode's replay. Below it, the whole Claude Code conversation: prompt, text, thinking, each command, each result, every image the agent was shown, its report and the programs it wrote |
| Compare | Runs side by side: success per task, agent and phase (with the change in points for two runs), model-service latency (median and p90), simulator time per step, motion speed and LLM cost |
| Log viewer | Any log or text file: search, errors and warnings only, last N lines, and follow while it grows |
| Live robots | Start a simulator as a **live console**. See its cameras and state, call any robot tool with JSON arguments (`segment_text`, `localize_object`, `generate_grasps`, `plan_motion`, `move_to_pose`, grippers, ...) or run a Code-as-Policy program, one at a time. **Close & save** finalizes the recording, which then appears under Runs like any episode |

The dashboard is plain files with no build step. The server, `scripts/gpu/dashboard.py`, only
routes; `scripts/gpu/dashboard_store.py` reads recordings into JSON (`/api/...`). The page is
`scripts/gpu/dashboard/index.html` plus ES modules in `scripts/gpu/dashboard/static/`: one per
page in `views/`, shared pieces in `lib/` (`charts.js`, `ui.js`, `util.js`). To add a page,
add a `views/<name>.js` that exports `render(app, params, ctx)` and a route in `app.js`.

The model services must be running for perception and motion
(`scripts/gpu/services.sh start`). A live console can also be started from a terminal and
attached to the dashboard:

```bash
.venv-libero/bin/python -m cap_harness.live --benchmark libero-pro --suite libero_object_swap --task-id 0 --seed 1
```

It prints the address it listens on (port 8770, or the next free one). Pass that to the
dashboard:

```bash
python scripts/gpu/dashboard.py --live http://127.0.0.1:8770
```

How the pieces fit, and how to add your own robot or model: [big-picture.md](big-picture.md).

## 10. Benchmarks with Opus 4.6 under a cost cap

There are two ways to run Opus 4.6 under a dollar cap. Claude Code enforces it
(`--max-budget-usd`), but only in headless runs. The cap is Claude Code's own estimate from
list prices and is checked between turns, so a run can end slightly over it. For a hard
backstop, also set an AWS Budgets alert on the account.

**A. Many tasks, one agent each.** This is the quick demo's agent: it writes a program,
debugs it on the dev seeds, then evaluates it on the held-out seeds. Put one
`benchmark:suite:task_id` per line in a text file (`#` starts a comment), for example
`outputs/llm-tasks.txt`:

```text
libero-pro:libero_object_swap:0
libero-pro:libero_goal_swap:0
libero-pro:libero_90:33
robosuite:cube_lifting:0
```

```bash
scripts/gpu/session.sh run --tasks quick --llm-tasks outputs/llm-tasks.txt --llm-budget 8 --llm-total-budget 25 --reserve 30
```

- `--llm-budget` caps each agent.
- `--llm-total-budget` caps the whole session. Each agent's cap is lowered to what is left,
  and an agent that would start with less than $0.50 is skipped (stage SKIP).
- `session.json` records `llm_cost_usd`.

**B. The full ASPIRE pipeline, unattended.** This runs the coordinator, the worker
sub-agents, the unseen check, held-out validation, the verifier and the skill updates:

```bash
scripts/gpu/demo.sh campaign --headless --budget-usd 30
```

- `--tasks FILE` takes a JSON task list, in the same format as
  `outputs/gpu-demo/campaign-tasks-*.json`.
- `--dev-seeds "51 52 53"`, `--unseen-seeds "66"` and `--heldout-count 3` set its size.
- `--max-turns` limits the coordinator's turns.

The campaign folder, `outputs/runs/<time>_gpu-demo/`, gets:

- the coordinator's transcript and the sub-agents' transcripts;
- `result.json`, with the cost;
- `DEMO_SUMMARY.md`.

The dashboard lists it under Runs.

## 11. Testing, latency and GPU checks

| What | Command | Time |
| --- | --- | --- |
| Unit tests (CPU) | `.venv-dev/bin/python -m pytest -q -p no:cacheprovider` | about 6 min |
| One test file | `.venv-dev/bin/python -m pytest -q tests/test_live_console.py` | seconds |
| Quick stack check (no services) | `scripts/gpu/check_stack.sh --quick` | about 2 min |
| Full stack check | `scripts/gpu/check_stack.sh` | about 10 min |
| ... with an Opus 4.6 coding check | `scripts/gpu/check_stack.sh --with-llm` | about $1–3 |
| Example programs end to end | `scripts/gpu/demo.sh smoke` | 10–20 min |
| Contact-GraspNet: GPU vs CPU right now | `scripts/gpu/check_graspnet.sh` | about 2 min |
| Latency of every service and the simulator | `python scripts/gpu/latency.py SESSION [SESSION ...]` | seconds |

`latency.py` reads recorded runs. To compare, give it a session recorded while the GPU was
busy and one recorded now. It reports:

- for each model service: median, p90 and max seconds per call;
- per simulator step and per reset;
- how far motions run from real time.

**Harness fixes.** Two defects of the published release fail programs for reasons outside the
program, and are fixed by default ([harness-fixes.md](harness-fixes.md) has the evidence from
recorded runs):

- **Motion timing.** The planners send joint targets six times faster than the arm can follow
  (0.05 against about 0.008 rad per step), so long moves fell behind, swept low and ended with
  "final trajectory waypoint did not converge". Trajectories are now re-timed to the speed each
  adapter derives from its controller's gains, and settling continues while the arm is still
  converging. Each call's output reports `planned_waypoints`, `executed_waypoints`,
  `max_joint_step_rad` and `settle_steps`.
- **Point clouds and boxes.** Depth-edge pixels at a mask's border landed in mid-air or behind
  the object, and `estimate_geometry` sized boxes from the extreme points: a 7 cm can became a
  40 cm box with its centre 5 cm off. Edge pixels are now left out and boxes ignore stray points.

`localize_object` also reports how sure SAM3 was (`low_confidence`, `ambiguous`), and
`close_gripper` reports `closed_empty` when the grasp missed.

`CAP_HARNESS_COMPAT=published` restores the published behaviour, and `demo.sh campaign --paper`
sets it, so paper comparisons stay like for like. Compare a session with it against one without
on the Compare page.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| Bedrock `ExpiredTokenException` / `UnrecognizedClientException` mid-run | The short-term API key expired (12 h max). Run `scripts/gpu/set_credentials.sh --api-key`, reload the env, and re-run. |
| `budget` says it does not fit | Use `--no-curobo`, wait for other users' jobs, or move to the emptier card with `scripts/gpu/use_gpu.sh N` |
| `qt-main_activate.sh: QT_XCB_GL_INTEGRATION: unbound variable` | A conda activation hook that breaks under `set -u`. Fixed in the scripts; `git pull` and re-run. |
| `PermissionError: '/tmp/robosuite.log'` | robosuite logs to a fixed `/tmp` file that another user owns. Setup now writes `third_party/robosuite/robosuite/macros_private.py`, which turns that log off. `git pull` and re-run `setup_conda_env.sh`; you don't need to delete anyone's file. |
| CUDA OOM in a service log | Another user's job grew. Run `services.sh stop`, `check_gpu.py budget`, then restart, without cuRobo if needed. |
| `EGLError` / blank images | Make sure the env is active (`MUJOCO_GL=egl`, `EGL_PLATFORM=device`, `MUJOCO_EGL_DEVICE_ID=0`). An NVIDIA EGL vendor file must exist (`ls /usr/share/glvnd/egl_vendor.d/`). |
| SAM3 fails to start with a 401 | The HF token lacks access to `facebook/sam3`. Request access, then re-run `set_credentials.sh`. |
| `on-demand throughput isn't supported` | You passed the bare model ID. Use the inference profile (`us.` / `eu.` / `global.` + `anthropic.claude-opus-4-6-v1`). |
| `AccessDeniedException` from Bedrock | Enable Opus 4.6 in Bedrock model access. The IAM principal needs `bedrock:InvokeModel`, `bedrock:InvokeModelWithResponseStream`, `bedrock:ListInferenceProfiles`. |
| `probe: contact_graspnet` fails with `Read timed out (read timeout=120.0)` | Contact-GraspNet is very slow on a GPU that another user keeps at 100 % utilization (its PointNet++ is thousands of tiny GPU kernels, each waiting for a time slice). Measure it with `scripts/gpu/check_graspnet.sh`. If the CPU wins, run it there: `services.sh start --graspnet-cpu`, or permanently with `conda env config vars set -n aspire ASPIRE_CPU_PROVIDERS=contact_graspnet`. That also frees about 2.4 GiB of GPU memory. |
| `services` stage FAILs although every service is up | Fixed: `services.sh` exited non-zero while printing reservation status (`git pull`). |
| Services died with the terminal | Use `scripts/gpu/services.sh start`. It runs them under `setsid`, so closing SSH does not kill them. |
| conda asks to accept Terms of Service | The setup uses `--override-channels -c conda-forge`. If your base conda still prompts, run `conda config --remove channels defaults`. |
| `claude` refuses `--permission-mode bypassPermissions` | Run the demos as your normal user, not root. |
| LIBERO reset crashes with "truth value of an array is ambiguous" | You are on the upstream release. This branch fixes it (`docs/review/findings.md` A1). |

Safety note from upstream: generated programs run with full Python access on this machine.
Don't run the demos on a host that holds credentials you can't afford to expose; the Bedrock
keys stored in this env are such credentials.
