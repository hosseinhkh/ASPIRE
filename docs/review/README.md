# ASPIRE review: architecture, findings, setup, and extension plan

This is a deep review of ASPIRE v1.0 (commit `fa00f0e`) and its paper (arXiv 2607.00272v2),
written to prepare the codebase for extension to a quadruped ("dog") robot and a robot arm.

| Document | Contents |
| --- | --- |
| [architecture.md](architecture.md) | How the system is built: the life of a run, contracts, API, providers, adapters, paper↔code mapping, extension points |
| [findings.md](findings.md) | Bugs, risks and tech debt with file:line, severity and verification status, including what this branch fixed |
| [paper-assessment.md](paper-assessment.md) | Strengths and weaknesses of the paper, and related work |
| [dependencies.md](dependencies.md) | Every dependency: host, venvs, submodules, services and ports, credentials, licences, hardware |
| [roadmap-quadruped-arm.md](roadmap-quadruped-arm.md) | Phased plan to support your arm and a legged mobile manipulator, simulation first |
| [../cpu-dev-setup.md](../cpu-dev-setup.md) | Running the stack without a GPU |

## How the review was done

- Read the full paper, including all appendices.
- Read the harness core line by line: `api.py`, `contracts.py`, `runtime.py`, `run.py`,
  `registry.py`, `artifacts.py`, providers, and the LIBERO and Robosuite adapters.
- Ran dedicated review passes over the real-robot stack (`yam_real/`, `agent/`), the BEHAVIOR-1K
  adapter, and the skill pipeline (`.claude/skills/**`, `validation/**`). Their highest-impact
  claims were spot-checked against the code.
- Ran web research on related work, the upstream issue tracker, simulators and SDKs.
- Built the environments on a CPU-only Ubuntu 24.04 machine. Ran the test suite, a Robosuite
  bimanual control smoke (`program_ok=True`), and a LIBERO-90 close-microwave episode with PyRoki
  IK on CPU (`task_success=True`).

## Verdict in one paragraph

ASPIRE's core idea is to give a frontier coding agent **typed, per-primitive, multimodal
execution traces** and let it debug robot programs like software. That idea is sound, well
engineered and worth building on. The harness has validated frame-explicit contracts, an
allowlisted tool registry and a thorough trace recorder, and it serves three simulators and a
real robot through one API.

The weaker parts:

- **Evaluation integrity depends on prompts.** Seed discipline, the verifier gate and evosearch
  selection are instructions to the model. Several released prompts and scripts break the
  Robosuite partitions.
- **The sandbox had real escapes**, now fixed on this branch.
- **The real-robot control path lacks basic safety features** such as velocity limiting,
  a stop path, and grip preservation across arms.
- **New embodiments are hard to add.** Robots are wired through about ten hard-coded lists, and
  arms are limited to 6 or 7 DoF.

The paper's near-saturated numbers mostly measure what a frontier model can do when allowed to
debug one program per task with rich traces. Cross-task transfer is real but modest (30.5%
zero-shot on LIBERO-Pro Long).

## Fixed on this branch

All entries were reproduced first and have regression tests.

| # | Fix | Files |
| --- | --- | --- |
| A1 | LIBERO `--init-mode saved` (the CLI default) crashed at every reset: ndarray truthiness | `libero/adapter.py` |
| A2 | Programs could read privileged simulator state through `str.format` field traversal | `runtime.py` |
| A3 | Programs could reach the host's live adapter through `gi_frame.f_back.f_locals`; NumPy `tofile`/`dump`/`ctypes` exposed file and memory access | `runtime.py` |
| A4 | Exhausting the step budget did not stop the program; retry loops hung forever | `runtime.py` |
| A5 | Doctor reported a bogus commit mismatch for uninitialized submodules | `doctor.py` |
| A6 | Tests broke collection in the simulator venvs; README stopped linking docs, so 3 release tests failed | `tests/`, `README.md`, `pyproject.toml` |

Test status after the fixes:

| Venv | Before | After |
| --- | --- | --- |
| `.venv-dev` (all extras) | 699 passed, 4 failed | **723 passed**, 1 skipped (cuRobo not installed) |
| `.venv-robosuite`, `.venv-libero` | collection aborted | **596 passed**, 69 skipped (missing extras, reported by name) |

## Top open issues

See [findings.md](findings.md) for details.

1. **Real robot (D1–D5).**
   - `move_to_joints` streams the goal in one 33 ms tick.
   - Velocity limits are parsed but never enforced.
   - Commanding one arm releases the other arm's grip.
   - A dead hold loop goes unnoticed.
   - There is no software stop.
2. **Evaluation (B1–B8).**
   - The verifier uses seeds 51–55, which are held-out seeds for Robosuite.
   - Evosearch can discard its own winner.
   - Held-out scores drive which tasks are re-run.
   - Privileged evaluator metrics land in dev-run artifacts.
   - BEHAVIOR's "held-out" score is best-of-k.
3. **Extensibility (C1).** About ten hard-coded embodiment lists, a 6/7-DoF limit, a silent
   fallback to `robosuite`, and a silent fallback to the Panda planning model.
4. **Default motion is not collision-aware (C2)** without cuRobo, which needs a GPU.
