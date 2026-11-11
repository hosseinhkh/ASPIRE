# Review prototypes (not product code)

The scripts behind the measurements in [docs/trace2vla-review.md](../../../docs/trace2vla-review.md).
Run them with `.venv-libero/bin/python` and `LIBERO_CONFIG_PATH=.libero MUJOCO_GL=egl`.

| Script | What it measures |
|---|---|
| `teacher_stats.py DIR...` | Length, idle share and hand speed of recorded teacher episodes |
| `osc_replay.py RUN_DIR SPEED...` | Re-executes a recorded teacher episode in `OSC_POSE` at each speed-up; reports success, steps, tracking lag |
| `scenes.py SUITE TASK saved\|seeded SEEDS` | Object positions per scene, and whether a scene is reproducible |
| `failure_split.py TASKS` | Which object the untrained student lifts and places (offline grading with simulator state) |

`osc_replay.py` is the starting point for the converter; it has no dataset writer and no tests.
