# Research Questions and Falsifiable Claims

This document distinguishes **exploratory findings already observed** from **confirmatory questions to be tested on a frozen held-out evaluation**.

## Scope

The study concerns performance inside the Panopticon simulation. It does not measure real-world intelligence, cybersecurity competence, employee risk, or general AI safety.

## RQ1 — Does V5 fine-tuning improve average task performance?

**Population:** Episodes sampled from the frozen five-tier Panopticon evaluation distribution.

**Compared policies:** Untrained `Qwen/Qwen2.5-1.5B-Instruct` and the raw merged V5 LoRA model, with identical prompts, tokenizer, decoding settings, parser rules, and paired seeds.

**Primary outcome:** Paired difference in level-macro composite grade.

**Directional hypothesis:** Raw V5 has a positive mean paired macro-grade difference relative to the base model.

**Exploratory observation:** The checked-in 20-episode-per-cell summary reports `0.701627 - 0.641110 = 0.060517` absolute, or approximately 9.44% relative.

**Falsification:** On the final held-out seed plan, the 95% confidence interval for the paired macro-grade difference includes zero or favors the base. A positive point estimate alone is not confirmation.

## RQ2 — Does average improvement satisfy advanced security requirements?

**Advanced tiers:** Level 4 and Level 5.

**Mandatory episode conditions:** Final security at least 90, all spawned sleepers caught, no sleeper missed, and zero false accusations. Candidate acceptance additionally uses the frozen comparison checks against base behavior.

**Primary safety outcomes:** Per-tier pass rate and probability of any mandatory-condition failure.

**Exploratory observation:** Raw V5 passes 50% on Level 4 and 5% on Level 5 and fails nine stored acceptance checks.

**Falsification of a safety claim:** Any final held-out episode violates a mandatory condition under a zero-tolerance release rule. Statistical uncertainty must still be reported when zero sampled failures are observed.

## RQ3 — Is the task/gate solvable by a verified controller?

**Policy:** The deterministic security-first supervisor, evaluated independently.

**Purpose:** Diagnostic positive control. If it fails, the environment, policy implementation, or gate may be inconsistent. If it passes, the controller path is demonstrably capable of passing sampled episodes.

**Exploratory observation:** The checked-in supervisor macro grade is `0.790471`, with 100% pass rate on both advanced levels.

**Interpretation boundary:** Supervisor success is not raw-model success. It cannot confirm that learned representations caused the executed actions.

## RQ4 — Which components account for learned and hybrid performance?

Confirmatory ablations will isolate:

1. curriculum order;
2. rare-action weighting;
3. security-first reward terms;
4. hard grader gates;
5. parsing-only cleanup;
6. semantic repair/fallback;
7. deterministic supervision; and
8. LoRA target modules/rank.

**Primary analysis:** Paired differences from the full raw V5 configuration for learning ablations, and from the raw policy for runtime-intervention ablations.

## RQ5 — Does behavior generalize beyond the training distribution?

Test held-out perturbation families separately:

- sleeper schedules shifted by bounded offsets;
- unseen worker-name assignments and department layouts;
- revenue/security cost perturbations;
- changed leak/activation probabilities;
- longer horizons where supported; and
- at least one alternative scripted adversary policy.

Report in-distribution and out-of-distribution results separately. Do not average them into a single number without displaying each family.

## RQ6 — Does learned HYDRA expose defender overfitting beyond scripted memory?

**Compared adversaries:** `scripted_memory_v1`, untrained neural HYDRA, single-defender-trained neural HYDRA, and population-trained neural HYDRA.

**Primary outcome:** Paired change in advanced-tier ARGUS operational pass rate under identical frozen seeds.

**Current status:** The neural policy boundary and trainer are implemented; no held-out result is available. The implementation must not be described as stronger, self-improving in production, or strategically general.

**Falsification:** Population-trained neural HYDRA does not reliably reduce held-out defender pass rate versus the scripted baseline, its interval includes no effect, or gains disappear outside the defender used during training. Full details are in `learned_hydra_protocol.md`.

## Claim language

Use “improved in the checked-in preliminary evaluation” until the final confirmatory run is complete. Use “passed sampled gates” rather than “proved safe.” Never use “real-world counter-intelligence capability.”
