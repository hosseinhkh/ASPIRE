{{SCOPE}}

Ignore AGENTS.md and .agents/: they are the real-robot System-2 runbook and do not apply here.
Do not read or run anything under src/cap_harness/yam_real, docs/system2*.md or docs/yam-real.md.

1. Read `.claude/skills/iterative-debugging/SKILL.md`, then `main-agent-prompt.md`, and act as
   the coordinator.
2. The campaign is already initialized at RUN_ROOT=`{{RUN_ROOT}}` (`outputs/runs/LATEST` points to
   it). Keep its `campaign.json` settings:
   - development seeds {{DEV_SEEDS}};
   - unseen-check seeds {{UNSEEN_SEEDS}};
   - held-out seeds 1-{{HELDOUT_COUNT}}.

   The task list is `{{RUN_ROOT}}/tasks.json`.
3. Hardware: one simulation GPU, GPU {{GPU}}. The campaign is single-GPU (GPUS=1), so follow
   "Single-GPU Operating Mode": you run every episode yourself, one at a time, and sub-agents only
   analyse run directories. Other users' jobs share this GPU: never kill or touch processes you did
   not start, and judge GPU ownership only by your ledger.
4. The perception services are already running (SAM3 :8114, Contact-GraspNet :8115,
   PyRoki :8116{{CUROBO_NOTE}}). Do not start, stop or restart them. If one is down, stop and tell me.
5. The conda env has already set the environment variables: MUJOCO_GL=egl, EGL_PLATFORM=device,
   MUJOCO_EGL_DEVICE_ID, CUDA_VISIBLE_DEVICES, LIBERO_CONFIG_PATH,
   TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD. Use `.venv-libero/bin/cap-harness` for LIBERO and
   `.venv-robosuite/bin/cap-harness` for Robosuite.
6. Spawn sub-agents with model "opus", as the prompts say. Here it resolves to Claude Opus 4.6.
7. {{VERSION_RULE}}
8. When every task is done, write `{{RUN_ROOT}}/DEMO_SUMMARY.md` with the status of each stage:
   - exploration;
   - development results per version;
   - unseen-check;
   - held-out;
   - skill proposals and the verifier's verdict;
   - any change to the skill library.

   Also record the elapsed time and any pipeline problems you hit.
