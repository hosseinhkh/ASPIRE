"""Object positions per scene, and whether a scene is reproducible.

    scenes.py SUITE TASK saved|seeded SEED[,SEED...]

Prints one JSON line (prefixed JSON): per seed, each movable object's position and a checksum of
the full simulator state. Run it twice to compare processes; compare saved against seeded output
to compare layout distributions.
"""

import json
import sys

import numpy as np

from cap_harness.libero.adapter import LiberoAdapter
from cap_harness.vla.libero_compat import ensure_libero_alias


def main() -> None:
    ensure_libero_alias()
    suite, task, mode = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    seeds = [int(s) for s in sys.argv[4].split(",")]
    adapter = LiberoAdapter(
        controller="OSC_POSE", camera_height=64, camera_width=64, init_mode=mode,
        post_settle_steps=10,
    )  # fmt: skip
    rows = {}
    for seed in seeds:
        adapter.reset(f"{suite}:{task}", seed=seed)
        env = adapter.native_env
        inner, sim = env.env, env.sim
        objects = {
            name: np.array(sim.data.body_xpos[inner.obj_body_id[name]]).round(5).tolist()
            for name in inner.objects_dict
        }
        state = np.asarray(sim.get_state().flatten())
        rows[str(seed)] = {"objects": objects, "state_sum": float(np.round(state.sum(), 9))}
    adapter.close()
    print("JSON" + json.dumps(rows))


if __name__ == "__main__":
    main()
