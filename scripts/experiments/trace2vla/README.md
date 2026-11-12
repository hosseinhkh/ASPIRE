# Review prototypes (not product code)

The scripts behind the measurements in [docs/trace2vla-review.md](../../../docs/trace2vla-review.md).
Run them with `.venv-libero/bin/python` and `LIBERO_CONFIG_PATH=.libero MUJOCO_GL=egl`.

| Script | What it measures |
|---|---|
| `teacher_stats.py DIR...` | Length, idle share and hand speed of recorded teacher episodes |
| `osc_replay.py RUN_DIR SPEED...` | Re-executes a recorded teacher episode in `OSC_POSE` at each speed-up; reports success, steps, tracking lag |
| `scenes.py SUITE TASK saved\|seeded SEEDS` | Object positions per scene, and whether a scene is reproducible |
| `failure_split.py TASKS` | Which object the untrained student lifts and places (offline grading with simulator state) |

`osc_replay.py` was the starting point for the converter. The product code is now
`src/cap_harness/trace2vla/` (`replay.py`, `convert.py`, `dataset.py`), with tests in
`tests/test_trace2vla.py`. Project overview: [docs/trace2vla.md](../../../docs/trace2vla.md).
