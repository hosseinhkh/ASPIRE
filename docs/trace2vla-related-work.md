# Trace2VLA: related work and what is still open

A survey made on 10–11 November 2026, before the first experiments. About 50 papers were
checked against their arXiv page or full text. It decides what the paper may claim, which
baselines it needs, and which numbers to compare with.

**Before a submission:** repeat the search (this field moves monthly), reread the five papers in
section 1 in full, and recheck every number quoted here against its source. Numbers below come
from abstracts or full text as noted in the survey; none was reproduced by us.

Back to the [project overview](trace2vla.md). The experiments that follow from this page are in
the [roadmap](trace2vla-roadmap.md).

## 1. The five closest papers

| Paper | What it does | What it means for us |
|---|---|---|
| **SEAR: Self-Evolving Agentic Robotics** ([2610.18792](https://arxiv.org/abs/2610.18792)) | A code agent writes per-task policies and fine-tunes π0.5 on its own rollouts with supervised learning. One checkpoint per suite; data and checkpoints do not carry over between tasks. VLA vs code teacher: LIBERO 94.0 vs 94.4, LIBERO-PRO 90.6 vs 91.8. Tasks with fewer than 50 demos score 20.8%, with 300 or more 94.0%. Lists learning from failures as future work | "Code agent → fine-tune a VLA on its successes" is published. Demo count dominates: plan for hundreds, not tens. Learning from failures and carrying one student across tasks are open |
| **SynthDemo-RL** ([2609.21650](https://arxiv.org/abs/2609.21650)) | An LLM-guided teacher with privileged simulator state trains a VLA student by supervised learning, then PPO with a binary reward. 96.0% on LIBERO with no human demos; on LIBERO-PRO it rescues all 27 tasks where π0.5 scores 0% (97.8% position, 97.1% task) | The closest pipeline. It uses privileged state and reinforcement learning; we use neither. It sets the bar for "rescue the 0% tasks" |
| **Robo-COP** ([2610.09228](https://arxiv.org/abs/2610.09228)) | An orchestrator mixes scripted motion with π0.5; judges curate skill segments, including good segments from failed episodes; a LoRA is retrained from base per task; a candidate is adopted only if it is not worse than the incumbent by more than 0.10, otherwise reverted | The accept-or-roll-back gate and agent-curated self-data are published. Use the gate as a component, not as a contribution |
| **LfH** ([2607.09042](https://arxiv.org/abs/2607.09042)) | A vision-language model relabels failed VLA rollouts (for example "wrong object placed") with the task actually achieved, inside GRPO reinforcement learning. Matches GRPO on LIBERO-PRO with about 5× fewer rollouts | Overlaps the semantic half of M2. Ours would be imitation only, with teacher corrections for the physical half |
| **Garg et al., self-demonstrated generative control** ([2608.19490](https://arxiv.org/abs/2608.19490)) | Rollouts from the frozen policy serve as pseudo-targets alongside expert data, to retain old skills without stored replay. Old RoboTwin tasks: 90.8 (base) → 16.6 (expert only) vs 70.6 (theirs). Full fine-tuning beat LoRA | A retention baseline, and evidence that fine-tuning on new data alone forgets badly |

## 2. What is already published, and what still looks open

**Published. Do not claim:**

| Idea | Where |
|---|---|
| A code agent as teacher, then supervised fine-tuning of a VLA on LIBERO / LIBERO-PRO | SEAR, SynthDemo-RL |
| Agent-curated self-data with a verify-then-adopt-or-revert gate | Robo-COP |
| Retry and recovery data in distillation | Scaling Up & Distilling Down, SEAR, RESample |
| Hindsight relabelling of wrong-object failures | LfH, SOAR, CAST, DIAL |
| A teacher correcting the learner on the learner's own states | DAgger, HG-DAgger, on-policy distillation |
| LoRA with a small replay buffer for continual VLA learning | TAIL, 2603.03818, 2603.11653 |
| Retaining skills with the policy's own rollouts | Garg et al. |

So **M1 (success-only distillation) is a baseline for us, not a contribution.**

**Apparently open (as of the survey):**

1. **One small student trained task after task from a code agent.** SEAR and Robo-COP both
   restart from the base model per suite or per task. A 450M student carried across a task
   sequence, with retention measured after every update, was not found.
2. **Failure-decomposed imitation.** An agent that sorts the student's failures into physical
   and semantic and treats each differently with imitation alone: teacher takeover from the
   student's own state for physical failures, relabelling plus contrast demonstrations for
   semantic ones. The pieces exist separately; the combination was not found.
3. **Whether failure data adds value beyond more successes.** SEAR finds demo count dominates.
   A matched-budget comparison is the test.
4. **SmolVLA on LIBERO-PRO.** No published number was found. Ours (0 / 1,500 on
   `libero_object_swap`) is new, small as a contribution, and necessary as a baseline.
5. **No privileged state in the teacher.** ASPIRE perceives through cameras. SynthDemo-RL's
   teacher uses simulator state. This is a framing point, and a reason our teacher's data is
   closer to what a real robot could produce.

**The proposal's "potential main contribution" still stands as a hypothesis:**
failure-decomposed continual distillation, tested against standard replay and self-demonstration
baselines.

## 3. Baselines a reviewer will expect

1. Success-only supervised fine-tuning on teacher data (SEAR-style), with a "more successes at
   matched compute" control.
2. One model trained on all tasks together, as the upper bound for the sequential student.
3. DAgger with the code agent labelling the student's states, and retry data in the style of
   Scaling Up & Distilling Down.
4. Continual baselines: sequential LoRA, replay with a small buffer, and the gate on vs off.
5. Garg-style self-rollout regularisation for retention.
6. Where feasible: teacher fine-tuning followed by sparse-reward reinforcement learning
   (SimpleVLA-RL, SynthDemo-RL), and LfH for the semantic-failure treatment. If these are out of
   budget, say so and compare with their published numbers on matching tasks only.
7. Evaluation on LIBERO-PRO, with retention on standard LIBERO. LIBERO-Plus is a further check.

## 4. The rest of the survey, by topic

**Distilling planners, code or scripted data into policies**

| Paper | Finding |
|---|---|
| Scaling Up & Distilling Down ([2307.14535](https://arxiv.org/abs/2307.14535)) | An LLM plans and writes its own success check; failed attempts are retried without resetting, so a kept success contains the recovery. Distilled policy +33.2% over its data-collection policy; without retries 34.8% lower. **Keep episodes in which the program retried and recovered** |
| GenSim ([2310.01361](https://arxiv.org/abs/2310.01361)), GenSim2 ([2410.03645](https://arxiv.org/abs/2410.03645)), RoboGen ([2311.01455](https://arxiv.org/abs/2311.01455)) | LLMs generate simulation tasks and demos. They generate tasks; they do not distil a VLA on a fixed benchmark |
| MimicGen ([2310.17596](https://arxiv.org/abs/2310.17596)), SkillMimicGen ([2410.18907](https://arxiv.org/abs/2410.18907)), DexMimicGen ([2410.24185](https://arxiv.org/abs/2410.24185)), RoboCasa ([2406.02523](https://arxiv.org/abs/2406.02523)) | Many demos from a few human ones. A data-generation baseline at matched demo count |
| RoboTwin 2.0 ([2506.18088](https://arxiv.org/abs/2506.18088)) | A multimodal LLM writes expert code refined in simulation |
| From Code to Action ([2509.24917](https://arxiv.org/abs/2509.24917)), BLAZER ([2510.08572](https://arxiv.org/abs/2510.08572)) | Code-API or LLM-planner demos used to train a policy or the LLM itself |
| FAEA ([2601.20334](https://arxiv.org/abs/2601.20334)) | The Claude Agent SDK reaches 84.9% on LIBERO with privileged state; it proposes agent trajectories as VLA training data and does not test it |

**Self-improvement loops**

| Paper | Finding |
|---|---|
| RoboCat ([2306.11706](https://arxiv.org/abs/2306.11706)) | The model generates its own next-round data |
| AutoRT ([2401.12963](https://arxiv.org/abs/2401.12963)) | LLM orchestration of 20+ robots, 77K episodes |
| SOAR ([2407.20635](https://arxiv.org/abs/2407.20635)) | A vision-language model proposes and judges tasks; retrains on judged successes; using failures is future work |
| Self-Improving EFMs ([2509.15155](https://arxiv.org/abs/2509.15155)) | A steps-to-go reward lets robots practise alone |
| A caution ([2411.01813](https://arxiv.org/abs/2411.01813)) | Autonomous imitation data gave modest gains; more human data helped more |

**Interactive imitation**

| Paper | Finding |
|---|---|
| DAgger ([1011.0686](https://arxiv.org/abs/1011.0686)), HG-DAgger ([1810.02890](https://arxiv.org/abs/1810.02890)) | Correcting the learner on its own states is standard. M2's "teacher takes over from the student's state" is DAgger with a code agent as the expert |
| RoboCopilot ([2503.07771](https://arxiv.org/abs/2503.07771)), DexHiL ([2603.09121](https://arxiv.org/abs/2603.09121)) | Human-in-the-loop versions for VLAs |
| RLDG ([2412.09858](https://arxiv.org/abs/2412.09858)) | Data from reinforcement-learning specialists, then a generalist fine-tune |
| Multi-teacher on-policy distillation ([2610.21522](https://arxiv.org/abs/2610.21522)) | The VLA student rolls out and teachers label its states: +6.4–9.5 over supervised learning on teacher rollouts |
| On-policy expert corrections ([2512.14895](https://arxiv.org/abs/2512.14895)) | The student starts and an expert takes over mid-trajectory (language-model agents) |

**Reinforcement-learning fine-tuning of VLAs**

SimpleVLA-RL ([2509.09674](https://arxiv.org/abs/2509.09674)), RIPT-VLA
([2505.17016](https://arxiv.org/abs/2505.17016)), VLA-RL
([2505.18719](https://arxiv.org/abs/2505.18719)), RL4VLA
([2505.19789](https://arxiv.org/abs/2505.19789)), ConRFT
([2502.05450](https://arxiv.org/abs/2502.05450)), PA-RL
([2412.06685](https://arxiv.org/abs/2412.06685)), iRe-VLA
([2501.16664](https://arxiv.org/abs/2501.16664)), π\*0.6 / RECAP
([2511.14759](https://arxiv.org/abs/2511.14759)). These reach high LIBERO scores from few demos
with a reward signal. We use imitation only; they are the comparison for "what a reward would
have bought".

**Failures and hindsight relabelling**

HER ([1707.01495](https://arxiv.org/abs/1707.01495)), DIAL
([2211.11736](https://arxiv.org/abs/2211.11736)), Sumers et al.
([2301.12507](https://arxiv.org/abs/2301.12507)), CAST
([2508.13446](https://arxiv.org/abs/2508.13446)), LfH (section 1), ExpReS-VLA
([2511.06202](https://arxiv.org/abs/2511.06202): separate success and failure buffers with a
contrastive loss), AFIL ([2605.08434](https://arxiv.org/abs/2605.08434)), RESample
([2510.17640](https://arxiv.org/abs/2510.17640): deviation-plus-recovery augmentation).

**Continual learning**

| Paper | Finding |
|---|---|
| LIBERO ([2306.03310](https://arxiv.org/abs/2306.03310)) | The benchmark's own continual baselines (replay, EWC, PackNet) |
| TAIL ([2310.05905](https://arxiv.org/abs/2310.05905)), LOTUS ([2311.02058](https://arxiv.org/abs/2311.02058)), M2Distill ([2410.00064](https://arxiv.org/abs/2410.00064)), CLARE ([2601.09512](https://arxiv.org/abs/2601.09512)) | Adapter-per-task and distillation methods |
| VLAs resist forgetting ([2603.03818](https://arxiv.org/abs/2603.03818)), sequential LoRA with on-policy RL ([2603.11653](https://arxiv.org/abs/2603.11653)) | A small replay buffer often gives no forgetting |
| RECALL ([2606.23617](https://arxiv.org/abs/2606.23617)) | Fine-tuning on recovery data alone causes catastrophic forgetting, so M2 needs replay |

**Benchmarks and the student**

| Paper | Finding |
|---|---|
| LIBERO-PRO ([2510.03827](https://arxiv.org/abs/2510.03827)) | Models above 90% on LIBERO fall to 0.0% in the generalised setting: they keep grasping at the old position when the target is swapped, and ignore corrupted instructions |
| LIBERO-Plus ([2510.13626](https://arxiv.org/abs/2510.13626)) | Success drops from about 95% to below 30% under modest camera or initial-state changes |
| SmolVLA ([2506.01844](https://arxiv.org/abs/2506.01844)) | 450M parameters; LIBERO 90 / 96 / 92 / 71 (Spatial / Object / Goal / Long), average 87.3. Community reruns reported lower numbers ([lerobot#2354](https://github.com/huggingface/lerobot/issues/2354)); ours on Object is 92.4% |

## 5. Wording to avoid

* "ASPIRE is a better teacher than SEAR": not without a matched comparison. Higher reported
  success on different benchmarks is not evidence.
* "First to distil an agent into a VLA": false.
* "Zero-shot baseline" for `lerobot/smolvla_libero`: it was fine-tuned on LIBERO. Call it the
  LIBERO-adapted student.
* Head-to-head comparisons with SEAR or the ASPIRE paper's numbers, unless tasks, scenes, seeds
  and budgets match. The ASPIRE paper's held-out seeds are seeded layouts, not the official saved
  scenes.
