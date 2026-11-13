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
{{LAYOUTS_NOTE}}
1. Read `docs/api-reference.md` and `docs/run-artifacts.md`, then
   `{{SKILLS_DIR}}/README.md` and the skill files it indexes.
   Do NOT read `examples/` or any `*_fix_code.py`: write your own program.
2. Look at the scene before you write a program. This is the pipeline's exploration step
   (`{{SKILLS_DIR}}/task-exploration.md`); probe runs do not count as program versions.
   Copy `.claude/skills/iterative-debugging/scripts/scene_snapshot.py` to `{{WS}}/probe.py` and
   run it on the first development seed:

   ```
   {{CAP_HARNESS}} run --benchmark {{BENCHMARK}} --suite {{SUITE}} --task-id {{TASK_ID}} \
     --seed {{FIRST_DEV_SEED}} --init-mode {{INIT_MODE}} --program {{WS}}/probe.py \
     --output-root {{WS}}/explore --flat-run-dir --no-videos
   ```

   It prints `run: <run_dir>`. Open that run's `media/keyframes/*/agentview.png` with the Read
   tool and find every object the instruction names. Objects are small in the 800x512 image:
   when you cannot tell two of them apart, crop and enlarge that region
   (`{{PYTHON}} -c "from PIL import Image; ..."`, saved under `{{WS}}/explore/`) and look
   again. SAM3 segments by visual description; it
   often does not know a product or brand name, and then every candidate scores low and the
   best one is a guess. So write, for each object, two or three prompts that describe what you
   see (colour, shape, material), put them in `PROBE_PROMPTS`, run the probe again, and check
   both the scores in `source/program-result.json` and the masks in `media/overlays/*.png`.
   Keep a prompt only if its best mask is on the right object, scores at least 0.5, and leads
   the next candidate clearly. If two similar objects both score high, decide how the program
   will tell them apart from what it can observe (`{{SKILLS_DIR}}/localize.md`,
   "Disambiguation"), and probe that too. At most 4 probe runs.
3. Write `{{WS}}/program_v01.py` with the prompts you verified, and run it on the development
   seeds:

   ```
   {{BATCH}} \
     --seeds {{DEV_SEEDS}} --program {{WS}}/program_v01.py --output-root {{WS}}/dev_v01
   ```

   Keep run-batch's defaults for the cameras (800x512) and the episode length: they are the
   ones the published results use.

   A seed normally takes under a minute, but on a busy machine one seed can take 5-10
   minutes. Run the batch with the Bash tool in the background with `timeout: 3600000`, do
   not wrap it in a shorter shell `timeout`, and wait for it: a batch that is cut off loses
   its unfinished seeds and writes no `batch.json` (the finished seeds are in
   `batch-results.jsonl`). Never run two batches at once: the GPU is shared.

   In the program, `localize_object(...).diagnostics` has `segmentation_score`,
   `low_confidence` and `ambiguous`. `ok` is true even when the mask is a guess, so do not act
   on a `low_confidence` result: try the next prompt, and raise an error if none is confident.
   Handle `ambiguous` with the rule you chose in step 2. After `close_gripper()`, check
   `diagnostics["closed_empty"]`.

   For the place position of a container (basket, bowl, box) use
   `localize_object(...).geometry.pose.position`, not the median of its point cloud: one
   camera sees mostly the near wall, so the median sits about 5 cm toward the camera and the
   object lands on the rim.

   For grasping cans, boxes and bottles standing upright, use your own top-down grasp at the
   object's `geometry.pose.position` (descend from above, close, then check `closed_empty`),
   and do not call `generate_grasps` / `select_grasp`. In this harness the grasp planner's
   candidates are unreliable: in recorded runs every candidate a program executed closed on
   nothing (12 of 12), because the planner is run on points in the robot's base frame and its
   pose is the gripper base, not the fingertip point. Use it only for an object a top-down
   grasp cannot take, and then verify the grasp before lifting.
4. Diagnose failures in this order, and fix the failure mode, not the seed:
   a. `{{WS}}/dev_vNN/batch.json`: the `warnings` list first (run-batch also prints it). It
      names causes that the program's own error does not: a guessed or ambiguous object, a
      grasp that closed on nothing, a model service that did not answer.
   b. The right object: open `media/overlays/sam3-*.png` of a failed run and confirm the mask
      the program used is on the object the instruction names. Do this before you change any
      grasp or motion code; a wrong object makes every later step look broken.
   c. Then each run's `outcome.json`, `trace/events.jsonl`, `trace/calls/*`,
      `logs/program.stdout`, and the keyframes in `media/`.
   Save each version as `program_vNN.py`.
5. Budget: at most {{MAX_VERSIONS}} program versions. Stop early once every development seed
   succeeds.
6. Copy the best version to `{{WS}}/program_best.py`, then run it once on the held-out seeds,
   which you must never debug against:

   ```
   ... --seeds {{HELDOUT_SEEDS}} --program {{WS}}/program_best.py --output-root {{WS}}/heldout
   ```

7. Write `{{WS}}/REPORT.md`:
   - versions tried, with the development success of each;
   - held-out success;
   - the main failure modes you saw, with the evidence (file paths);
   - {{LESSONS_LINE}}
{{SKILLS_STEP}}
The perception services are already running (SAM3 :8114, Contact-GraspNet :8115, PyRoki :8116{{CUROBO_NOTE}}).
Do not start, stop or restart them. If one is down, stop and say so in REPORT.md.
The environment variables for rendering and LIBERO are already set in this shell.
