"""Trace2VLA: turn ASPIRE teacher episodes into training data for a VLA student, train, compare.

One experiment folder (``outputs/trace2vla/<name>/``) holds every stage and a ledger of what
produced what:

    experiment.json              suite, tasks, scene sets, settings
    ledger.jsonl                 one event per stage run (inputs, outputs, hashes, counts)
    teacher/programs/            the frozen program of each task (the teacher)
    teacher/<scene-set>/         ASPIRE run folders: every step, video, tool call, SAM3 mask
    grades/<scene-set>/          offline grading of each teacher episode (simulator state)
    demos/<name>/                teacher successes re-executed in the student's control
    datasets/<name>/             LeRobot datasets built from demos, with a lineage manifest
    training/<run>/              lerobot-train output, log, metrics
    evals/<run>/                 student benchmark runs (the format of outputs/vla)
    reports/                     comparison tables

Modules:
    store      experiment folder, ledger, hashing
    scenes     named scene sets (official test scenes, training layouts, development)
    grading    offline labels from simulator state: which object was lifted and placed
    episodes   read a recorded ASPIRE teacher episode
    replay     joint replay (fidelity, grading), re-execution in OSC_POSE, open-loop replay
    teacher    register programs, run them on a scene set, grade the episodes
    convert    teacher successes -> verified student-control demos, idle frames dropped
    dataset    demos -> LeRobotDataset with the base checkpoint's normalisation statistics
    train      lerobot-train wrapper, log and metrics
    report     teacher vs student before vs student after
    cli        ``python -m cap_harness.trace2vla`` (scripts/gpu/t2v.sh)

Simulator state is privileged. It is used here for replay and grading only, is written only
under the experiment folder, and never reaches the teacher agent or the student's inputs.
"""
