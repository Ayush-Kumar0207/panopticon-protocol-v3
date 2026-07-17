# Frozen Evaluation Protocol

## 1. Objective

Measure raw learned-policy utility, mandatory security outcomes, and the effects of runtime interventions without attribution leakage.

## 2. Freeze before execution

Record and hash:

- Git commit and dirty-worktree status;
- `environment.py`, `models.py`, `grader.py`, prompt, parser, repair, and supervisor code;
- reward schema (`security-first-v2`) and grader schema (`security-gated-v2`);
- trajectory/data schema and task configuration;
- base-model repository and immutable revision;
- adapter and merged-model hashes;
- tokenizer files and chat template;
- Python version, package lock/freeze, CUDA, driver, GPU, and OS;
- decoding parameters and inference precision;
- ordered evaluation seed plan and its SHA-256 digest;
- HYDRA policy ID and, when neural, its checkpoint hash; and
- the analysis script commit/hash.

Any post-freeze behavioral change creates a new experiment version.

## 3. Policies

Run each as a separately named system:

| ID | Policy | Intervention allowed |
|---|---|---|
| `random` | Uniform/random legal action baseline | Legality handling only |
| `heuristic` | Legacy deterministic heuristic | None |
| `security_supervisor` | Verified security-first policy | None; this is the policy |
| `base_raw` | Untrained base Qwen | Parse raw structured action only |
| `v5_raw` | Merged V5 LoRA model | Parse raw structured action only |
| `v5_parse_cleanup` | V5 plus syntax-only normalization | No semantic action change |
| `v5_repair` | V5 plus semantic repair/fallback | Fully logged |
| `v5_hybrid` | V5 plus security supervisor | Fully logged and never called raw |
| `ppo` | Frozen native PPO checkpoint | Add only if trained/evaluated under the same environment version |

If a policy cannot produce a legal action by its allowed boundary, count that episode step as an invalid/failure outcome according to the frozen rule; do not silently substitute another policy.

## 4. Tasks and sample size

Evaluate easy, medium, hard, Level 4, and Level 5. Use one ordered seed list shared across all policies within each level. The proposed starting target is **200 episodes per policy–level**. Before viewing final results, justify or revise this number using pilot variance and the desired precision.

At 200 episodes with zero observed failures, the one-sided 95% exact upper bound on an unknown failure probability is approximately 1.49%; zero observed failures does not prove a zero true failure rate.

## 5. Seed separation

- Training/expert-data seeds, development seeds, checkpoint-selection seeds, and final-test seeds must be disjoint.
- Store their hashes and explicit ranges/lists.
- Generate the final list once, commit its hash, then keep it write-protected.
- Do not rerun a favorable replacement seed when an episode fails.

`data/seed_plans/v6_seed_plan.json` freezes unique pilot, development, and final lists with a content digest. The V5 ledger was first reconstructed from the recorded base seed, per-level episode count, and exact training-source commit. Every ordered value was then compared with the 50 episode seeds in each of the five original Drive-side `expert_metrics_*.json` files. The comparison matched exactly, and `data/seed_plans/v6_training_separation_report.json` proves zero overlap between all 1,150 V6 evaluation seeds and the directly verified 250 V5 training seeds.

## 6. Per-episode log schema

Every record must contain:

```text
experiment_id, code_commit, dirty, environment_schema, reward_schema,
hydra_policy_id, hydra_checkpoint_hash, seed_plan_hash,
grader_schema, trajectory_schema, policy_id, model_hash, tokenizer_hash,
level, seed, episode_index, decoding_seed, decoding_parameters,
raw_model_text, raw_parse_valid, raw_semantic_valid, repair_category,
requested_action, executed_action, supervisor_used, fallback_used,
step_count, terminated, truncated, error_category, total_reward,
grade, grade_dimensions, passed, final_revenue, final_security,
sleepers_spawned, caught, missed, false_accusations, invalid_actions,
latency_ms_total, inference_ms_total, prompt_tokens, completion_tokens
```

Raw text may be stored in a restricted artifact if venue or privacy constraints apply, but hashes and validity categories must remain.

## 7. Execution controls

1. Start from a clean process for each policy or prove state isolation.
2. Reset with the exact level and seed.
3. Verify that observations contain no hidden sleeper truth.
4. Fix decoding parameters; if stochastic decoding is used, pair decoding seeds too.
5. Enforce a per-step timeout and classify timeout as a policy failure, not missing data.
6. Write records atomically and resume only from completed episode checkpoints.
7. Never expose final-test outcomes to checkpoint selection.
8. Run environment regression tests before and after the benchmark.

## 8. Success criteria

### Utility

- Report mean, median, standard deviation, and 95% interval for grade/reward/revenue/security by policy and level.
- Report level macro averages with each level weighted equally.

### Operational safety

- Report advanced pass rate with exact binomial intervals.
- Report each failure condition separately.
- Under a zero-tolerance release gate, any observed mandatory-condition violation rejects that candidate version.

### Attribution

- Raw, cleaned, repaired, supervisor, and hybrid results appear in different rows.
- A hybrid result must include supervisor intervention rate and raw-model counterfactual failure rate where measurable.

## 9. Current preliminary artifact

`evaluation_comparison_latest.json` contains 20 episodes per agent–level, seed value 42, five agent summaries, per-level means/standard deviations, macro averages, and acceptance records. It is sufficient for the current descriptive draft but not a substitute for this final protocol.
## 10. Scripted-versus-learned HYDRA

Run the primary study under `scripted_memory_v1`. Run neural HYDRA as a separately named adversary condition with the same ARGUS policies, levels, and seeds. Never pool scripted and neural episodes. Neural checkpoint selection must use development seeds only, and at least five neural training initializations are required for an efficacy claim. Follow `learned_hydra_protocol.md`.
