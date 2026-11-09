You are running a small end-to-end demo of the ASPIRE code-as-policy harness in this repository,
on a shared GPU workstation. The goal is to show that every part works together (simulator,
perception services, the generated-program sandbox, run traces, and you as the coding agent).
It is not a benchmark run: keep it small and stop when the budget below is used.

Ignore AGENTS.md and .agents/: they are the real-robot System-2 runbook and do not apply here.
Do not read or run anything under src/cap_harness/yam_real, docs/system2*.md or docs/yam-real.md.

## Task

Write a Code-as-Policy program for {{BENCHMARK}} suite `{{SUITE}}`, task id {{TASK_ID}}.
{{TASK_NOTE}}
The suite name is the benchmark variant, not the task: in LIBERO-Pro, a suffix such as `_swap`
names how the scene layout is perturbed. Do what the instruction says, and keep reading it from
`get_task_context().language` inside the program rather than hard-coding it.

1. Read `docs/api-reference.md` and `docs/run-artifacts.md`, then
   `{{SKILLS_DIR}}/README.md` and the skill files it indexes.
   Do NOT read `examples/` or any `*_fix_code.py`: write your own program.
2. Write `{{WS}}/program_v01.py` and run it on the development seeds:

   ```
   {{CAP_HARNESS}} run-batch --benchmark {{BENCHMARK}} --suite {{SUITE}} --task-id {{TASK_ID}} \
     --seeds {{DEV_SEEDS}} --init-mode {{INIT_MODE}} --program {{WS}}/program_v01.py \
     --output-root {{WS}}/dev_v01 --workers 1
   ```

   Keep run-batch's defaults for the cameras (800x512) and the episode length: they are the
   ones the published results use.

   Each seed takes 1-5 minutes. Run the batch with the Bash tool in the background (or with a
   600000 ms timeout) and wait for it. Never run two batches at once: the GPU is shared.
3. Diagnose failures from `{{WS}}/dev_vNN/batch.json`, each run's `outcome.json`,
   `trace/events.jsonl`, `trace/calls/*`, `logs/program.stdout`, and the keyframes in `media/`.
   Fix the failure mode, not the seed. Save each version as `program_vNN.py`.
4. Budget: at most {{MAX_VERSIONS}} program versions. Stop early once every development seed
   succeeds.
5. Copy the best version to `{{WS}}/program_best.py`, then run it once on the held-out seeds,
   which you must never debug against:

   ```
   ... --seeds {{HELDOUT_SEEDS}} --program {{WS}}/program_best.py --output-root {{WS}}/heldout
   ```

6. Write `{{WS}}/REPORT.md`:
   - versions tried, with the development success of each;
   - held-out success;
   - the main failure modes you saw, with the evidence (file paths);
   - {{LESSONS_LINE}}
{{SKILLS_STEP}}
The perception services are already running (SAM3 :8114, Contact-GraspNet :8115, PyRoki :8116{{CUROBO_NOTE}}).
Do not start, stop or restart them. If one is down, stop and say so in REPORT.md.
The environment variables for rendering and LIBERO are already set in this shell.
