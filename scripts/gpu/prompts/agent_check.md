This is a small capability check of the ASPIRE code-as-policy harness in this repository. SAM3
perception is not available yet, so use no perception tools. Be brief and finish quickly.

Ignore AGENTS.md and .agents/: they are the real-robot System-2 runbook and do not apply here.

1. Read the shared-tools table and the "Cartesian pose inputs" section of `docs/api-reference.md`.
2. Write `{{WS}}/lift_check.py`, a CaP program for Robosuite cube lifting that:
   - reads the end-effector pose of arm "primary";
   - moves it 8 cm straight up with `move_to_pose`, keeping the orientation;
   - moves it back to the start pose;
   - sets `result` to a dict with the measured end-effector heights (z) before, at the top, and
     after, and the `ok` flag of both moves.
3. Run it:

   ```
   .venv-robosuite/bin/cap-harness run --benchmark robosuite --suite cube_lifting --task-id 0 \
     --seed 1 --program {{WS}}/lift_check.py --output-root {{WS}}/run --max-steps 600 \
     --camera-width 256 --camera-height 256
   ```

4. Read the run's `source/program-result.json` and `outcome.json`. If the program failed, fix it
   and run it again, at most 3 attempts in total.
5. Write `{{WS}}/REPORT.md` with:
   - the number of attempts;
   - the measured heights;
   - whether the top was 0.08 m ± 0.01 m above the start;
   - anything in the API docs that was unclear.
