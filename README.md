# ASPIRE: Agentic /Skills Discovery for Robotics

[Project Page](https://research.nvidia.com/labs/gear/aspire/) &ensp;|&ensp; [Paper](https://arxiv.org/abs/2607.00272)

<img src="assets/media/skill_discovery.gif" alt="ASPIRE discovering /skills across simulation and real-robot tasks" width="100%">

ASPIRE is a new type of continual learning: "training" is skill refinement instead of gradient descent. 

"Trained model" is a repo of sensorimotor skills instead of floating weights. 

“Distributed training” is a panel of agents each practicing a different skill instead of sharded minibatches.

## This fork: Trace2VLA

This fork adds **Trace2VLA**, a research project built on ASPIRE: use ASPIRE as a *teacher* and
train a compact vision-language-action model (SmolVLA, 450M parameters) as a *student* on the
teacher's verified executed trajectories, then measure how much of the teacher-student gap
closes. On `libero_object_swap` the student starts at 0 of 1,500 episodes.

| | |
|---|---|
| What it is, how it works, where it stands | [`docs/trace2vla.md`](docs/trace2vla.md) |
| What is left to do, and the experiments for the paper | [`docs/trace2vla-roadmap.md`](docs/trace2vla-roadmap.md) |
| Run the first end-to-end pass | [`docs/trace2vla-preflight.md`](docs/trace2vla-preflight.md) |

Everything else below is ASPIRE itself, with the GPU-server tooling this fork added.

## News

- **[2026-10] ASPIRE v1.0 is out!** ASPIRE can now solve a Rubik's Cube, and it achieves 99% success on 80 LIBERO-Pro Goal / Object / Spatial / Long tasks and selected Robosuite, BEHAVIOR-1K tasks.

- **[2026-06] ASPIRE is out!** We achieve SOTA results on LIBERO-Pro, Robosuite, and BEHAVIOR-1K with Claude Opus 4.6! Also check out our table bussing sim-to-real demos!

## Real-World Results

<table>
  <tr>
    <th width="50%">Cross-embodiment sim-to-real</th>
    <th width="50%">Online real-world learning</th>
  </tr>
  <tr>
    <td><video src="https://github.com/user-attachments/assets/ce86d1c1-582e-4388-9eb8-2b9027252eec" width="100%" controls></video></td>
    <td><video src="https://github.com/user-attachments/assets/7fa4ea5c-bcae-4eb8-b0e6-3109af08c94d" width="100%" controls></video></td>
  </tr>
  <tr>
    <td><a href="https://github.com/NVlabs/ASPIRE/blob/v0.1/aspire/real/README.md">Long Horizon Table Bussing</a><br>Claude Opus 4.6 (sim), GPT-5.5 (real)</td>
    <td><a href="https://github.com/NVlabs/ASPIRE/blob/main/docs/system2.md#solve-a-rubiks-cube">Rubik's Cube</a><br>GPT-6 Astra</td>
  </tr>
</table>

<table>
  <tr>
    <td width="33%"><video src="https://github.com/user-attachments/assets/15e5946f-b26d-466e-aa44-c971e5756933" width="100%" controls></video></td>
    <td width="33%"><video src="https://github.com/user-attachments/assets/620eb33d-b9da-40cb-8d84-eb487f264585" width="100%" controls></video></td>
    <td width="33%"><video src="https://github.com/user-attachments/assets/18a8484f-a0bd-4574-9891-f5c49d906bc5" width="100%" controls></video></td>
  </tr>
  <tr>
    <td><a href="https://github.com/NVlabs/ASPIRE/blob/main/docs/system2.md#inverted-cup-nesting">Stack Cups</a><br>GPT-6 Astra</td>
    <td><a href="https://github.com/NVlabs/ASPIRE/blob/main/docs/system2.md#unscrew-and-pour">Unscrew Bottle Cap</a><br>GPT-6 Astra</td>
    <td><a href="https://github.com/NVlabs/ASPIRE/blob/main/docs/system2.md#pill-insertion-and-bottle-closure">Pill in Bottle</a><br>GPT-6 Astra</td>
  </tr>
</table>

## Model Scaling

<table>
  <tr>
    <th width="50%">Success vs. API cost</th>
    <th width="50%">LIBERO-Pro, Robosuite, BEHAVIOR-1K</th>
  </tr>
  <tr>
    <td><img src="assets/media/success_vs_api_cost.gif" alt="ASPIRE success vs. API cost across twelve coding-agent models" width="100%"></td>
    <td><img src="assets/media/benchmark_results.gif" alt="ASPIRE success rates on LIBERO-Pro, Robosuite and BEHAVIOR-1K by model" width="100%"></td>
  </tr>
  <tr>
    <th colspan="2">Stronger models discover better strategies</th>
  </tr>
  <tr>
    <td><video src="https://github.com/user-attachments/assets/f4b77674-2ada-4f10-a3de-8126d9512342" width="100%" controls></video></td>
    <td><video src="https://github.com/user-attachments/assets/9645e221-7f6c-4c6d-99b2-dce515afbf6e" width="100%" controls></video></td>
  </tr>
  <tr>
    <td><b>Book under a shelf</b><br>Opus 5.5 and Astra let go of the book and push it under the low shelf.</td>
    <td><b>Push a plate</b><br>Opus 5.5 clears the clutter first; Astra lifts the plate over it.</td>
  </tr>
  <tr>
    <td><video src="https://github.com/user-attachments/assets/d66e195f-93f8-403d-ab47-e098cba2f58b" width="100%" controls></video></td>
    <td><video src="https://github.com/user-attachments/assets/9c795d80-761a-48b4-838e-d20bcd905fd5" width="100%" controls></video></td>
  </tr>
  <tr>
    <td><b>Open the bottom drawer</b><br>Opus 5.5 and Astra tilt the gripper and hook the low handle, as the human does.</td>
    <td><b>Mug in the microwave</b><br>Opus 5.5 and Astra grasp the handle and slide the mug in upright.</td>
  </tr>
</table>

ASPIRE improves with the underlying model. On 10 LIBERO-Pro tasks, success rises from 15% (GPT-6 Luna) to 99% (Claude Opus 5.5), and the strongest models are not the most expensive: the green line marks the cost/success frontier. Across LIBERO-Pro, Robosuite and BEHAVIOR-1K, Claude Opus 5.5 reaches 97–100% success on every task.

## Quick Start

### Run with a coding agent

To get started with ASPIRE, launch a coding agent such as Codex or Claude Code and enter the following prompt:

```text
Clone the main branch of https://github.com/NVlabs/ASPIRE/,
read the iterative-debugging skill (.claude/skills/iterative-debugging/SKILL.md) and
docs/libero-pro.md, and run the complete ASPIRE LIBERO-Pro Goal-Swap Quick Start for
all ten tasks in the libero_goal_swap suite.

Before executing, report the required GPUs, credentials, gated weights,
services, expected runtime, seed partitions, and output paths. Wait for
my confirmation before launching. Do not access real-robot code or instructions
(AGENTS.md, docs/system2*.md, docs/yam-real.md, src/cap_harness/yam_real/).
```

Before running, set up the host (CUDA, uv, and the Hugging Face token for the gated SAM3 weights) with [`docs/host-setup.md`](docs/host-setup.md).

**Reference agent environments:** ASPIRE is coding-agent agnostic. The simulation workflow is packaged as agent skills for Claude Code ([`.claude/skills/`](.claude/skills/)) and Codex ([`.codex/skills/`](.codex/skills/)): `iterative-debugging` and `evosearch`. [`AGENTS.md`](AGENTS.md) holds the execution rules for the real-robot System 2 agent ([operator runbook](docs/system2.md)) and is not needed for simulation runs.

### Reproduce full paper results

Name the suite and experiment explicitly. If neither is named, the agent should present this table and stop for selection.

<table>
  <thead>
    <tr>
      <th>Suite</th>
      <th>Experiment</th>
      <th>Runbook</th>
      <th>Video</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>LIBERO-Pro</td>
      <td>Iterative-debugging loop</td>
      <td><a href="https://github.com/NVlabs/ASPIRE/blob/main/.claude/skills/iterative-debugging/SKILL.md"><code>iterative-debugging/</code></a></td>
      <td rowspan="3"><video src="https://github.com/user-attachments/assets/15c0b425-9fe9-4313-8a5b-8c7da54c24b7" width="240" controls></video></td>
    </tr>
    <tr>
      <td>Robosuite</td>
      <td>Iterative-debugging loop</td>
      <td><a href="https://github.com/NVlabs/ASPIRE/blob/main/.claude/skills/iterative-debugging/SKILL.md"><code>iterative-debugging/</code></a></td>
    </tr>
    <tr>
      <td>BEHAVIOR-1K</td>
      <td>Iterative-debugging loop</td>
      <td><a href="https://github.com/NVlabs/ASPIRE/blob/main/.claude/skills/iterative-debugging/behavior/SKILL.md"><code>iterative-debugging/behavior/</code></a></td>
    </tr>
    <tr>
      <td>LIBERO-Pro</td>
      <td>Evolutionary Search</td>
      <td><a href="https://github.com/NVlabs/ASPIRE/blob/main/.claude/skills/evosearch/SKILL.md"><code>evosearch/</code></a></td>
      <td rowspan="2"><video src="https://github.com/user-attachments/assets/edd2e5e1-b7d6-408a-bb62-23729df14db2" width="240" controls></video></td>
    </tr>
    <tr>
      <td>Robosuite</td>
      <td>Evolutionary Search</td>
      <td><a href="https://github.com/NVlabs/ASPIRE/blob/main/.claude/skills/evosearch/SKILL.md"><code>evosearch/</code></a></td>
    </tr>
    <tr>
      <td>LIBERO</td>
      <td>Zero-Shot Transfer</td>
      <td><a href="https://github.com/NVlabs/ASPIRE/blob/v0.1/aspire/sim/.claude/libero/zeroshot-transfer/INSTRUCTIONS.md"><code>libero/zeroshot-transfer/</code></a></td>
      <td rowspan="4"><video src="https://github.com/user-attachments/assets/8676fa10-6719-477f-9d60-a4ad241a3de3" width="240" controls></video></td>
    </tr>
    <tr>
      <td>LIBERO-Long-Pro</td>
      <td>Library-Size Scaling</td>
      <td><a href="https://github.com/NVlabs/ASPIRE/blob/v0.1/aspire/sim/.claude/libero/library-size-scaling/INSTRUCTIONS.md"><code>libero/library-size-scaling/</code></a></td>
    </tr>
    <tr>
      <td>LIBERO-Long-Pro</td>
      <td>Inference-Time Scaling</td>
      <td><a href="https://github.com/NVlabs/ASPIRE/blob/v0.1/aspire/sim/.claude/libero/inference-time-scaling/INSTRUCTIONS.md"><code>libero/inference-time-scaling/</code></a></td>
    </tr>
    <tr>
      <td>Robosuite</td>
      <td>Debugging-Token-Ablation</td>
      <td><a href="https://github.com/NVlabs/ASPIRE/blob/v0.1/aspire/sim/.claude/robosuite/training-law/INSTRUCTIONS.md"><code>robosuite/training-law/</code></a></td>
    </tr>
  </tbody>
</table>


Before a paper-scale launch, the agent must report the selected tasks, seed schedule, expected trial count and runtime, GPU and credential requirements, services, and output paths, then wait for explicit confirmation.

### Reproduce the real-robot manipulation demos

Use the [System 2 operator runbook](docs/system2.md#reproduce-the-manipulation-demos)
to prepare a calibrated YAM station and launch a fresh agent with the appropriate
text-based task skill:

- [Rubik's Cube](examples/yam_real/rubiks_cube/TASK_CONTEXT.md).
- [Unscrew a bottle and pour](examples/yam_real/unscrew_bottle/TASK_CONTEXT.md).
- [Insert a pill and close the bottle](examples/yam_real/pill_in_bottle/TASK_CONTEXT.md).
- [Nest cups upside down](examples/yam_real/cup_stacking/TASK_CONTEXT.md).

The [manipulation skill entrypoint](.agents/skills/manipulation-tasks/SKILL.md)
routes the agent to these recipes. They provide grasp strategies, recovery, and
physical completion checks without requiring episode logs or fixed-coordinate
replays. The three new recipes are distilled from demo sessions, not a new
real-robot validation; the pill recipe explicitly distinguishes a seated cap from
verified final tightness. Re-ground the current scene and verify the actual outcome.

> [!WARNING]
> ASPIRE executes language-model-generated Python with full import access. Trial processes and watchdogs are not a security sandbox. Run generated code on an isolated host without credentials or sensitive mounts, restrict network access, and never grant a simulation agent access to physical hardware. Real-robot work requires the controls in [`aspire/real/AGENTS.md`](https://github.com/NVlabs/ASPIRE/blob/v0.1/aspire/real/AGENTS.md) and separate operator authorization.

## Step by step on a GPU server: check, test, run, view

This assumes the conda env from [`docs/gpu-quickstart.md`](docs/gpu-quickstart.md). Run
every command from the repository folder, with the env active:

```bash
cd ~/robot2/ASPIRE && conda activate aspire
```

**1. See how busy the GPUs are.**

```bash
gpustat
```

```bash
python scripts/gpu/check_gpu.py budget
```

**2. Check the model services.**

```bash
scripts/gpu/services.sh status
```

If any of the four services is `down`, start them. Wait for "all provider services ready".

```bash
scripts/gpu/services.sh start
```

**3. Decide where Contact-GraspNet runs: GPU or CPU.**

```bash
scripts/gpu/check_graspnet.sh
```

Then check the current setting:

```bash
echo "$ASPIRE_CPU_PROVIDERS"
```

- If the check says the GPU is faster and the setting prints `contact_graspnet`, switch back
  to the GPU:

  ```bash
  conda env config vars unset -n aspire ASPIRE_CPU_PROVIDERS
  ```

- If the check says the CPU is faster and the setting prints nothing, switch to the CPU:

  ```bash
  conda env config vars set -n aspire ASPIRE_CPU_PROVIDERS=contact_graspnet
  ```

If you changed the setting, reload the env:

```bash
conda deactivate && conda activate aspire
```

Then restart the services:

```bash
scripts/gpu/services.sh stop && scripts/gpu/services.sh start
```

**4. Check the whole stack.** This takes about 10 minutes. It uses the running services and
leaves them up. Every line should be PASS, including `probe: contact_graspnet`.

```bash
scripts/gpu/check_stack.sh
```

**5. Start the dashboard** in a second tmux window, so it keeps running:

```bash
python scripts/gpu/dashboard.py
```

It prints a port and an `ssh -L ...` line. Run that line on your laptop, then open the
`http://localhost:...` address it shows.

**6. Run tasks** in the first window. Three tasks, no LLM, about 5–10 minutes:

```bash
scripts/gpu/session.sh run --tasks quick --no-preflight --reserve 30
```

Or the same three tasks plus one Opus 4.6 agent, capped at $10:

```bash
scripts/gpu/session.sh run --tasks quick --no-preflight --llm libero-pro:libero_object_swap:0 --llm-budget 10 --reserve 30
```

**7. Look at the run in the dashboard.** Refresh **Runs** and open the newest session.

- **Session → Tasks:** success per task, per agent version (dev / held-out) and per seed.
  Click a seed to replay it, or a row to see every run with how it ended and its error. Each
  row and each run has **Logs** and **Full** zip downloads. Tick two rows (say `dev_v03` and
  `dev_v04`) and press **Compare selected**.
- **Session → Failures:** every unsolved episode grouped by how it failed, with the logs of
  each group as one zip.
- **Session → Episodes:** a thumbnail per episode; filter by task or by failed.
- **An episode:** a replay. Every camera plays in sync, frame by frame if you like, under a
  timeline of every call: the program's calls, the API stages inside them (IK, planning,
  execution) and each model request (SAM3, Contact-GraspNet, PyRoki, cuRobo). Click a call to
  jump the videos there and see its keyframes before and after, each SAM3 mask (switch them on
  one by one), the grasps drawn on the image, and its full input, output and error. Motions
  get a chart of commanded against measured joints. Failed calls are red, and warnings flag
  low-confidence SAM3 results.
- **3D scene tab:** the point clouds, boxes, grasps, motion targets and hand path of the
  episode, in the robot base frame. Drag to rotate.
- **The agent:** *How it debugged*, one card per step (its reasoning, the code change, the
  seeds that passed), then its whole conversation.
- **Logs:** open any log in the viewer (search, errors only, follow), or download it.
- **Download:** a zip, complete or logs only.

**8. If an episode failed,** print why:

```bash
scripts/gpu/session.sh why "$(ls -td outputs/sessions/* | head -1)/episodes"
```

**9. When you are done,** stop the services. Stop the dashboard with Ctrl-C in its window.

```bash
scripts/gpu/services.sh stop
```

### Compare runs: a new idea, another simulator, a real robot

Record each variant as its own session with a name, on the same tasks and seeds:

```bash
scripts/gpu/session.sh run --name baseline --tasks standard --seeds "1 2 3"
```

Change the code (a new skill, model, prompt or planner), then:

```bash
scripts/gpu/session.sh run --name my-idea --tasks standard --seeds "1 2 3"
```

In the dashboard, tick both under **Runs** and press **Compare**. You get, side by side:

- success per task, per agent and per phase, with the change in points;
- model-service latency (median and p90), simulator time per step, motion speed;
- the agent's model, context window and effort, its cost, the motion timing, the GPU and the
  git commit of each run.

The same page compares any recordings: two simulators, a simulator and a real robot (live
console recordings), or two versions of an agent. From a terminal:

```bash
python scripts/gpu/latency.py outputs/sessions/*-baseline outputs/sessions/*-my-idea
```

The harness fixes defects of the published release that fail programs for reasons outside the
program: motions planned six times faster than the arm can follow, and point clouds and object
boxes corrupted by depth-edge pixels ([`docs/harness-fixes.md`](docs/harness-fixes.md) has the
evidence). To measure what they change, record a session with `CAP_HARNESS_COMPAT=published` in
front of the command and compare it with one without.

### Check that you match the paper (Opus 4.6)

The `--llm` agent in step 6 is a smoke test, and its score is not comparable with the paper. The
paper's Opus 4.6 numbers come from the full iterative-debugging pipeline:

| Setting | Paper | `demo.sh campaign --paper` | `--llm` agent (step 6) |
| --- | --- | --- | --- |
| Claude | Opus 4.6, max effort, 1M-token context | same | same |
| Pipeline | coordinator, worker, unseen check, held-out validation, verifier | same | one agent, short prompt |
| LIBERO-Pro seeds | dev 51–65, unseen 66–80, held-out 1–50 | same | dev 51–53, held-out 1–3 |
| Program versions | until the dev threshold or a plateau | same | at most 4 |
| Cameras, episode length | 800×512, harness default | same | same |
| Harness behaviour | published release | published (`CAP_HARNESS_COMPAT=published`, set by `--paper`) | fixed ([harness-fixes.md](docs/harness-fixes.md)) |
| Cost | about $24 and an hour of agent time per task (Table 18) | capped at $40 per task | the cap you set |

For example, the paper reports 100% on 50 held-out seeds for LIBERO-Pro Object "alphabet soup"
(`libero_object_swap` task 0; Table 10). To check that task:

```bash
scripts/gpu/demo.sh campaign --paper --headless
```

It runs unattended under a $40 cap and takes a few hours of GPU time (15 dev seeds per program
version, 15 unseen-check seeds, 50 held-out seeds). At the end it prints the campaign summary
with the held-out score, and the context window the model really used: it should say
1,000,000. For other tasks, pass a task list with `--tasks FILE` (format in
`.claude/skills/iterative-debugging/scripts/gen_progress.py`). The paper's per-task results
are in Tables 10, 11, 13 and 19.

Even with the same protocol, expect spread between runs: the paper's three Opus 4.6 runs differ
by up to 47 points on one task (Table 25).

### Does the skill library (memory) help? A three-campaign experiment

The skill library in `.claude/skills/iterative-debugging/skills/` is the only memory between
tasks (Claude Code's own memory and CLAUDE.md notes are switched off for these runs). Three
campaigns on one suite, all with the paper's protocol and the same harness:

| Run | Memory | Starts from | Question |
|---|---|---|---|
| 1 | off | empty library | How good is the agent alone? (paper, Opus 4.6: 55.6%) |
| 2 | on | empty library | Do skills learned on earlier tasks help later ones? |
| 3 | on | run 2's library | Does a full library help on a second pass? |

Run them one after another, in a tmux window. Each takes most of a day; see the cost note below.

```bash
scripts/gpu/demo.sh campaign --paper --harness fixed --suite libero_goal_swap --memory off --library empty --name goal-nomem --headless --budget-usd 300
```

```bash
scripts/gpu/demo.sh campaign --paper --harness fixed --suite libero_goal_swap --memory on --library empty --name goal-mem1 --headless --budget-usd 350
```

```bash
scripts/gpu/demo.sh campaign --paper --harness fixed --suite libero_goal_swap --memory on --library goal-mem1-end --name goal-mem2 --headless --budget-usd 350
```

To see how the agent writes skills before spending on campaigns, give a quick agent its own
copy of the library to write into (the shared library is not touched; the run prints what it
wrote, saves it as `skills.diff`, and the dashboard shows it on the agent page):

```bash
scripts/gpu/session.sh run --tasks quick --no-preflight --llm libero-pro:libero_object_swap:0 --llm-budget 10 --llm-write-skills --reserve 30
```

Compare them, task by task, next to the paper's numbers:

```bash
python scripts/gpu/campaign_results.py outputs/runs/*_goal-nomem outputs/runs/*_goal-mem1 outputs/runs/*_goal-mem2
```

or tick the three campaigns under **Runs** in the dashboard and press **Compare**.

- `--memory off` is the no-memory baseline: no skill is promoted, and if anything edits the
  library anyway it is restored afterwards. `--memory on` is the full loop: verifier, promotion,
  curriculum and the backward-transfer rerun sweep.
- Every campaign saves the library it started and ended with (`scripts/gpu/library.sh list`).
  `scripts/gpu/library.sh status` shows the current one, `diff A B` what was learned, and
  `released` puts the shipped library back.
- A stopped campaign (budget, expired credentials, Ctrl-C) continues where it left off:
  `scripts/gpu/demo.sh campaign --resume outputs/runs/<campaign> --headless --budget-usd 100`.
  Progress at any time: `python .claude/skills/iterative-debugging/scripts/gen_progress.py --print`.
- Credentials: a short-term Bedrock API key lasts at most 12 hours, shorter than a campaign. Use
  access keys or a long-term API key (`scripts/gpu/set_credentials.sh`), or renew and resume.
- **Cost and time.** The paper reports $23.58 and about an hour of agent time per task for Opus 4.6
  on this suite (Table 18), so about $240 for run 1, and more with memory on (verifier and
  reruns): plan for roughly $800-900 and two to three days for all three. `--budget-usd` is a hard
  cap per campaign; the budget can be raised on `--resume`.
- `--harness fixed` keeps the harness fixes on in all three, so only the memory differs; the
  paper's 55.6% used the published harness (`--harness published`).
- One campaign per condition is a small sample: with 50 held-out seeds a task's 95% interval is
  about +/-14 points, and the paper's own runs of one task varied by up to 47 points.

### Switch to another GPU (e.g. GPU 1), keeping your reservation on GPU 0

The memory reservation stays on GPU 0 the whole time. Services only touch a reservation on
their own GPU.

Stop the services. Your GPU 0 reservation then grows back to its full size: it lets go of
the memory for a moment and takes it back.

```bash
scripts/gpu/services.sh stop
```

Point the env at GPU 1. This command also shows whether the stack fits there right now:

```bash
scripts/gpu/use_gpu.sh 1
```

Reload the env:

```bash
conda deactivate && conda activate aspire
```

Start the services. They now go to GPU 1, and the simulators render there. Your GPU 0
reservation is not touched.

```bash
scripts/gpu/services.sh start
```

Check that the reservation is still on GPU 0 and the services are on GPU 1:

```bash
scripts/gpu/reserve.sh status
```

```bash
gpustat
```

Run sessions as usual, for example
`scripts/gpu/session.sh run --tasks quick --no-preflight --reserve 30`. When your reservation
is held on another GPU, `--reserve` leaves it there and says so.

To go back, run the same steps with `scripts/gpu/use_gpu.sh 0`. When the services start on
GPU 0 again, they share your reservation automatically.

To move the reservation itself to GPU 1 instead, run `scripts/gpu/reserve.sh hold 30 --gpu 1`.
That releases GPU 0.

## Documentation

| Topic | Guide |
| --- | --- |
| Host prerequisites (GPU, uv, Hugging Face token) | [`docs/host-setup.md`](docs/host-setup.md) |
| Simulator/provider topology, venvs, profiles | [`docs/environments.md`](docs/environments.md) |
| Generated-program API | [`docs/api-reference.md`](docs/api-reference.md) |
| Run artifacts and traces | [`docs/run-artifacts.md`](docs/run-artifacts.md) |
| Harness fixes (motion timing, point clouds, boxes, diagnostics) and why | [`docs/harness-fixes.md`](docs/harness-fixes.md) |
| LIBERO-Pro | [`docs/libero-pro.md`](docs/libero-pro.md) |
| **Trace2VLA overview**: goal, design, status, results so far, glossary | [`docs/trace2vla.md`](docs/trace2vla.md) |
| Trace2VLA roadmap: work ahead, experiments for the paper, risks, open decisions | [`docs/trace2vla-roadmap.md`](docs/trace2vla-roadmap.md) |
| Trace2VLA related work: closest papers, what is published, what is open, expected baselines | [`docs/trace2vla-related-work.md`](docs/trace2vla-related-work.md) |
| VLA student: SmolVLA on LIBERO / LIBERO-Pro in ASPIRE's simulator (setup, parity, baseline) | [`docs/vla-student.md`](docs/vla-student.md) |
| Trace2VLA design review: measurements, design changes, pipeline structure, order of work | [`docs/trace2vla-review.md`](docs/trace2vla-review.md) |
| Trace2VLA preflight: teacher data → fine-tuned SmolVLA → comparison, every command | [`docs/trace2vla-preflight.md`](docs/trace2vla-preflight.md) |
| Robosuite | [`docs/robosuite.md`](docs/robosuite.md) |
| BEHAVIOR-1K | [`docs/behavior.md`](docs/behavior.md) |
| Optional backends | [`docs/optional-backends.md`](docs/optional-backends.md) |
| CPU-only development setup (no GPU) | [`docs/cpu-dev-setup.md`](docs/cpu-dev-setup.md) |
| GPU workstation: conda env, Opus 4.6 on Bedrock, light demos | [`docs/gpu-quickstart.md`](docs/gpu-quickstart.md) |
| Code review, known issues, and extension roadmap | [`docs/review/README.md`](docs/review/README.md) |
| How the core works and how to add a robot or a model | [`docs/big-picture.md`](docs/big-picture.md) |
| Dashboard, comparing runs, live robot console | [`docs/gpu-quickstart.md`](docs/gpu-quickstart.md#9-dashboard-and-live-robots) |

## Contribution Guidelines

Start with [CONTRIBUTING.md](https://github.com/NVlabs/ASPIRE/blob/v0.1/CONTRIBUTING.md). Third-party contributions must include a Developer Certificate of Origin sign-off.

## License

ASPIRE material owned by NVIDIA or contributed under the project license is
available under the [Apache License 2.0](https://github.com/NVlabs/ASPIRE/blob/main/LICENSE). Third-party materials retain
their original terms. See [NOTICE](https://github.com/NVlabs/ASPIRE/blob/main/NOTICE) and the central
[licensing and compliance index](https://github.com/NVlabs/ASPIRE/blob/v0.1/LICENSES/README.md) before using or
redistributing the full stack.

## Citation

If you find ASPIRE useful in your research, please cite:

```bibtex
@article{lu2026aspire,
  title   = {ASPIRE: Agentic /Skills Discovery for Robotics},
  author  = {Runyu Lu and Yubo Wu and Ethan Kou and Haotian Lin and Letian Fu and Wenli Xiao and
             Ajay Mandlekar and Yinzhen Xu and Guanya Shi and Ken Goldberg and
             Ang Chen and Mosharaf Chowdhury and Jan Kautz and Yuke Zhu and Linxi "Jim" Fan and Guanzhi Wang},
  year    = {2026},
  journal = {arXiv preprint arXiv:2607.00272},
  url     = {https://arxiv.org/abs/2607.00272}
}
```

ASPIRE was developed by researchers from NVIDIA, the University of Michigan, the University of Illinois Urbana-Champaign, UC Berkeley, and Carnegie Mellon University.
