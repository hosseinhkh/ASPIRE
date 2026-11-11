"""VLA student on LIBERO: run LeRobot policies (SmolVLA first) through ASPIRE's own simulator.

The student and the ASPIRE teacher share one simulator code path: ``LiberoAdapter`` builds
every environment, picks the scene for a seed, settles it and checks success.  The student
drives the adapter in ``OSC_POSE`` mode, LIBERO's own controller and the action space of the
published LIBERO checkpoints; the teacher keeps ``JOINT_POSITION``.

Modules:
    libero_compat  lets LeRobot's ``libero.libero`` imports find the LIBERO-Pro package
    protocol       episode limits, settle steps and task language, as lerobot-eval uses them
    observation    what the student sees: two cameras rotated 180 degrees, an 8-value state
    policy         a LeRobot policy with LeRobot's own pre- and post-processing
    rollout        one recorded episode through the adapter
    benchmark      a suite benchmark: tasks x scenes x policy seeds (``python -m``)
    parity         LeRobot's LIBERO env vs the adapter on the same scene and actions
    lerobot_eval   LeRobot's own ``lerobot-eval``, runnable against the LIBERO-Pro package

Requires the ``vla`` extra (``scripts/bootstrap_libero.sh --vla``).
"""
