r"""LeRobot's own ``lerobot-eval``, run against the LIBERO-Pro package installed here.

    python -m cap_harness.vla.lerobot_eval \
        --policy.path=lerobot/smolvla_libero --env.type=libero --env.task=libero_object \
        --eval.batch_size=1 --eval.n_episodes=10 \
        --rename_map='{"observation.images.image": "observation.images.camera1",
                       "observation.images.image2": "observation.images.camera2"}'

Arguments are lerobot-eval's.  The only difference from calling ``lerobot-eval`` directly is
the ``libero.libero`` import alias (see ``libero_compat``).  Use it as an independent check
of the numbers from ``cap_harness.vla.benchmark``; note that lerobot-eval renders 360x360 by
default (``--env.observation_height/width``) and steps through init states differently.
"""

from __future__ import annotations

from cap_harness.vla.libero_compat import ensure_libero_alias


def main() -> None:
    ensure_libero_alias()
    from lerobot.scripts.lerobot_eval import main as lerobot_eval_main

    lerobot_eval_main()


if __name__ == "__main__":
    main()
