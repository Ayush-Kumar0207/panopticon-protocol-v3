# Ablation and Diagnostic Experiment Plan

An ablation changes one component while holding the evaluation seed plan, environment, base model, decoding, and other components fixed. If retraining is required, use at least three training seeds when compute permits and report between-run variation.

## Core ablations

| ID | Full condition | Ablated condition | Question | Primary readout |
|---|---|---|---|---|
| A1 | Five-tier curriculum | All tiers mixed from start | Does curriculum ordering help advanced reliability? | Level-4/5 pass rate; macro grade |
| A2 | Configured action weighting | One copy per expert step | Does oversampling rare actions help or distort policy behavior? | action recall; false actions; advanced pass |
| A3 | Security-first reward | Remove threat/security-deficit terms | Does shaped reward change learned PPO/security behavior? | final security; catch/miss rates |
| A4 | Hard grader gate | Composite threshold only | How many unsafe episodes would average-only grading accept? | unsafe-acceptance count |
| A5 | Completion-only loss | Loss on full conversation | Does masking prompt tokens improve action learning? | valid action rate; grade |
| A6 | Attention+MLP LoRA targets | Attention projections only | Is added adapter capacity needed? | grade, pass rate, trainable parameters |
| A7 | Rank 16 | Rank 8 and rank 32 | Sensitivity to adaptation rank | performance/compute frontier |
| A8 | 512-token context fitting | 256 and 1024 where feasible | Does critical evidence survive truncation? | advanced errors by evidence age |
| A9 | Raw parser only | Syntax cleanup | How much failure is formatting rather than policy choice? | parse validity; unchanged-action rate |
| A10 | Syntax cleanup | Semantic repair/fallback | How much performance comes from changing decisions? | repair rate; grade delta |
| A11 | Raw V5 | V5 + supervisor | What is the operational benefit and intervention cost? | gate pass; supervisor-use rate |
| A12 | Expert-only SFT | DAgger-style relabeling | Does learner-state coverage reduce compounding error? | advanced pass; state-distribution shift |

## Adversary/generalization ablations

| ID | Change | Purpose |
|---|---|---|
| G1 | Shift sleeper spawn turns within prespecified bounds | Test schedule memorization. |
| G2 | Randomize cost/damage parameters | Test sensitivity to reward/dynamics constants. |
| G3 | Alternative channel-selection adversary | Test robustness beyond current scripted avoidance. |
| G4 | Unseen department/name assignments | Test superficial token memorization. |
| G5 | Disable adversary memory | Estimate the difficulty contribution of adaptive avoidance. |

## Diagnostic slices

For every experiment, stratify errors by:

- generation and active threat count;
- dead-switch state;
- false-flag evidence present;
- confirmed versus merely suspicious target;
- time since relevant evidence;
- context truncation status;
- raw parse/semantic validity;
- repair category; and
- supervisor intervention.

## Interpretation rules

1. Retraining ablations answer causal questions only across the randomized training seeds used.
2. A supervisor gain is a system gain, not a raw-model gain.
3. Removing the gate cannot improve safety by definition; it measures hidden unsafe acceptance.
4. If an ablation changes dataset size, report both fixed-example and naturally resulting-size comparisons when feasible.
5. Do not select only favorable ablations for the paper. Publish the complete registered matrix or explain missing cells.
