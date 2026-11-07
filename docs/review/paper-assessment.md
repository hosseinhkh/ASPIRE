# Paper assessment: *ASPIRE: Agentic /Skills Discovery for Robotics* (arXiv 2607.00272v2)

This assessment is based on a full read of the 41-page v2 PDF (20 Oct 2026) and the released code.

## Summary of the claim

A frontier coding agent (Claude Code or Codex) writes a Python robot program against a typed API
covering perception, grasping, IK, planning and control. It improves that program through three
mechanisms:

1. **Iterative debugging** with per-primitive multimodal traces: return values, keyframes and
   overlays for each API call.
2. **Evolutionary search**: K candidates per round, conditioned on the top 3 programs and their
   failure traces.
3. **A markdown skill library** of validated repairs. A verifier agent must reproduce each repair
   before it is admitted, and later tasks reuse the library as in-context guidance.

Each task produces one program, evaluated on disjoint held-out seeds.

## Strengths

1. **The execution engine is the real contribution, and the ablation supports it.**
   - Adding per-primitive traces raises LIBERO-Pro macro success from 14% to 62% with Opus 4.6.
     Evolutionary search then adds +10 pp, reaching 72% (Fig. 7, Tables 10–11).
   - The code backs this up. Every tool, nested API stage and provider call is recorded with
     typed inputs and outputs, before/after keyframes and timing, and failures are typed values
     (`ApiError(code, details)`) rather than tracebacks.
   - This is the part most worth reusing for your robots.
2. **Program-level policies generalize across layouts where VLAs memorize.**
   - On LIBERO-Pro position and task perturbations, OpenVLA, π0 and π0.5 score 0–38%, while
     ASPIRE scores 45–98% (Table 5).
   - LIBERO-Pro was built to expose memorization; see [Zhou et al. 2025](https://arxiv.org/abs/2510.03827).
3. **Skills transfer, and the effect grows with library size.**
   - Zero-shot LIBERO-Pro Long rises from 4.7% to 13.7%, 21.5% and 30.5% as the library grows
     from 0 to 25, 50 and 90 source tasks (Table 8).
   - On the real robot, retrieved simulation skills cut tokens to first success. Soda can:
     61.9M → 6.6M tokens. Drawer: 0/20 → 11/20 success (Table 1).
4. **Honest scaling and cost reporting.**
   - Twelve models with cost per task: Sonnet 5.5 reaches 98.8% at $12/task (Fig. 8, Table 18).
   - Tables 20–23 give tokens, LLM calls, simulator rollouts and wall time.
   - Run-to-run variance is reported for Opus 4.6 (Table 25). Few agentic-robotics papers do this.
5. **Engineering quality.**
   - Frame-explicit, validated contracts.
   - An allowlisted tool registry and stale-plan rejection.
   - Spawned (not forked) batch workers.
   - Pinned submodules plus a frozen `uv.lock`.
   - A provider topology that is a single declarative source (`configs/environments.json`).
   - 700+ unit tests.
   - The same API serves three simulators and a real bimanual robot.
6. **Agent-agnostic.** The pipeline is plain markdown and scripts. It ran with Claude Code and
   Codex, using proprietary and open-weight models.

## Weaknesses and open questions

### Evaluation design

1. **Within-task generalization, not task generalization.** The headline numbers
   (98–100% on LIBERO-Pro and Robosuite with Opus 5.5) come from debugging **one program per
   task** on 15 dev seeds of that same task, then evaluating it on 50 held-out layouts of that
   task. The cross-task result, zero-shot on LIBERO-Pro Long, is far lower: 30.5% with Opus 4.6.
2. **Comparisons span different information regimes.**
   - The VLA baselines run zero-shot from demonstrations.
   - ASPIRE gets per-task debugging budgets, a frontier LLM, SAM3, Contact-GraspNet and cuRobo,
     and simulator-truth collision geometry in BEHAVIOR.
   - CaP-Agent0 regenerates a program per seed with retries.

   The comparison establishes that perception + planning + program search beats memorized
   policies under layout shift. It does not show that one method is better at equal information.
3. **Variance is mostly unreported.**
   - Table 25 shows per-task standard deviations up to 47 pp between three Opus 4.6 runs; *Open
     middle drawer* scored 0%, 100% and 0%.
   - Most other tables come from single campaigns without confidence intervals.
   - At n = 50 seeds per task, a 95% binomial CI on 80% is roughly ±11 pp.
4. **Held-out results leak into method decisions in the released pipeline**
   ([findings.md](findings.md) §B):
   - Backward-transfer reruns and evosearch tasks are chosen by held-out score and then re-scored
     on the same seeds. That is regression to the mean, so the +7.6 pp in Table 2 is optimistically
     biased.
   - The released Robosuite verifier uses seeds 51–55, which lie in Robosuite's held-out band.
   - Table 10's "selected on seeds 66–80" is not implemented in the released evosearch, which
     selects on the dev seeds.
5. **BEHAVIOR-1K numbers mean something different.**
   - Stage 2 iterates on each evaluation seed. The score is "the final program succeeded at least
     once" across repeats (`behavior/score_campaign.py:96-108`), which is best-of-k with oracle
     feedback.
   - Navigation and planning use privileged simulator state: ground-truth base pose, collision
     meshes, the precomputed traversability map, and the held-object identity.
   - `behavior.base_pose_is_free()` answers programs from that ground-truth map.
6. **Integrity depends on model compliance.**
   - Seed discipline, the verifier gate, evosearch parameters and the "non-privileged API audit"
     are prompt instructions.
   - The sandbox let programs read or modify simulator state through `str.format` and generator
     frames; this branch fixes both.
   - Nothing suggests a model exploited this. Nothing in the pipeline would have detected it
     either.

### Method

7. **The predefined API bounds what can be learned.** The paper acknowledges this. "Skills" are
   prompt text and code sketches over fixed primitives, not new sensorimotor capabilities.
   Contact-rich behaviour such as *nut assembly* rises from 9% to 100% only with stronger models,
   which suggests the model does the work rather than the library.
8. **The skill library does not scale as built.**
   - The library is whole markdown files loaded into context, with no retrieval, ranking,
     per-entry IDs or re-validation.
   - The prompts note that `transport.md` reached 28k words.
   - The paper acknowledges stale skills.
9. **The gains are largely the model's.**
   - Fig. 8 runs from 15% to 99% across models on the same pipeline.
   - With Opus 5.5, evolutionary search moves suite averages by at most 2 pp.
   - It is hard to separate how much is ASPIRE and how much is the model's capacity to debug
     with good traces. A strong-model ablation without the skill library or traces on all suites
     would clarify this.
10. **Compute cost.** Opus 4.6 used about 2.0B tokens (≈$1.4k) for 60 LIBERO-Pro tasks, plus
    about 4.9B tokens (≈$2.8k) for evolutionary search on 31 tasks (Table 20). Real-robot runs
    spend 5–335M tokens per task.

### Real-robot evidence

11. **The sample is small.**
    - Three transfer tasks with 20 trials each, on one station (Table 1).
    - Token counts "until first success" are single runs.
    - The Rubik's Cube result is three episodes with one model; a human scrambles the cube and
      resets the scene.
12. **The real-robot stack has safety gaps** that matter before anyone copies it
    ([findings.md](findings.md) §D):
    - no velocity limiting;
    - `move_to_joints` is effectively a step command;
    - one arm's command releases the other arm's grip;
    - no software stop path;
    - dead hold loops go unnoticed.

## Related work

| Work | Relation |
| --- | --- |
| [Code as Policies](https://arxiv.org/abs/2209.07753) | Origin of the paradigm. One-shot generation with no execution-feedback loop. |
| [Voyager](https://arxiv.org/abs/2305.16291) | Skill library plus iterative prompting in Minecraft. ASPIRE is the embodied version with verifier-gated admission. |
| [CaP-X / CaP-Agent0](https://arxiv.org/abs/2603.22435) ([code](https://github.com/capgym/cap-x)) | Direct predecessor: same benchmarks and API family, with an explicit privileged-vs-perceived perception switch. ASPIRE adds per-primitive traces, verified skills, evolutionary search, and one program per task. |
| [AlphaEvolve](https://arxiv.org/abs/2506.13131), [Code evolution for control](https://arxiv.org/abs/2601.06845) | Program evolution with automatic evaluators. ASPIRE's search is the robotics instance and shares the same dependence on cheap, reliable evaluators. |
| [RoboCodeX](https://arxiv.org/abs/2402.16117) | Fine-tunes a code LLM, the training-based alternative to a frozen frontier model. |
| [Uni-Skill](https://arxiv.org/abs/2603.02623), [SkillFlow](https://arxiv.org/abs/2604.17308) | Self-evolving skill repositories and benchmarks. |
| [ENPIRE](https://huggingface.co/papers/2606.19980) | Sibling work that improves *learned policies* on real YAM stations with automatic reset and verification. Complementary to ASPIRE's program search. |
| [LIBERO-Plus](https://arxiv.org/abs/2510.13626) | Broader robustness perturbations (camera, robot initial state) that ASPIRE has not been evaluated on. |

## Bottom line

ASPIRE is a well-engineered system with a convincing core idea: give a coding agent
**per-primitive, typed, multimodal traces**, and treat robot programming as software debugging.
Distilling repairs into reusable text is useful and transfers modestly. The headline
near-saturation numbers mostly measure what a frontier model can do when allowed to debug one
program per task with rich traces.

For your project, the reusable assets are:

- the typed API and contracts;
- the trace recorder;
- the provider topology;
- the System-2 MCP pattern.

Treat the pipeline's evaluation guarantees as something you must enforce in code yourself.
