# Statistical Analysis Plan

Freeze this document before the final held-out run. Changes after seeing final outcomes must be dated and labeled exploratory.

## 1. Experimental unit and blocking

The experimental unit is one complete episode. Policies are paired by `(level, environment_seed, decoding_seed)` wherever stochastic decoding applies. Difficulty level is a prespecified block, not a random sample of all possible environments.

## 2. Outcomes

### Confirmatory primary outcome for RQ1

Level-macro mean of paired raw V5 minus raw base composite grade.

### Confirmatory safety outcomes for RQ2

- Level-4 pass probability;
- Level-5 pass probability;
- probability of any mandatory security failure across advanced episodes.

These safety outcomes are not replaced by the composite outcome.

### Secondary outcomes

Reward, final revenue, final security, catch rate, missed sleepers, false accusations, invalid-action rate, parse validity, intervention rate, latency, and token usage.

## 3. Descriptive reporting

For each policy and level report `n`, missing/error count, mean, standard deviation, median, interquartile range, minimum, maximum, and 95% confidence interval. For binary outcomes report numerator/denominator and Clopper–Pearson intervals. Show episode-level distributions where space permits.

## 4. Effect estimation

- Continuous paired outcomes: mean paired difference with a 10,000-resample paired bootstrap percentile interval; also report median paired difference.
- Confirmatory continuous test: two-sided paired permutation test under exchangeability, with the directional hypothesis stated separately.
- Paired binary outcomes: discordant-pair table, McNemar exact test, and paired risk difference.
- Unpaired fallback, if pairing is broken: Welch interval/test, explicitly labeled and justified.
- Effect sizes: standardized mean paired difference plus raw units. Raw units remain primary for interpretability.

Statistical significance is not operational acceptance. A small positive grade effect cannot override one advanced mandatory-condition failure.

## 5. Multiplicity

RQ1 has one confirmatory primary comparison. For the three RQ2 confirmatory safety outcomes, control family-wise error with Holm's procedure. All per-level secondary comparisons and ablations are labeled exploratory unless separately preregistered.

## 6. Missing data and failures

- Timeout, crash, malformed output, or exhausted repair budget is a policy failure and remains in the denominator.
- Do not drop episodes after observing outcomes.
- Infrastructure-wide failures affecting every policy on a paired seed may be rerun only under a written, policy-blind rule.
- Report exclusions with reason, policy, level, and seed.

## 7. Macro aggregation

Compute each level mean first, then average the five level means equally. Also provide a micro average over all episodes. Do not mix the two. If sample sizes are equal they may coincide, but both definitions remain explicit.

## 8. Precision and sample-size rationale

The proposed 200 episodes per policy–level is a starting target. For rare failures, if zero are observed in 200 independent episodes, the one-sided 95% exact upper confidence bound is

```text
1 - 0.05^(1/200) ≈ 0.01487 (1.49%)
```

Before final execution, use pilot variance to estimate the paired-grade interval width and document compute constraints. Do not choose sample size after observing whether a result is significant.

## 9. Sensitivity analyses

- bootstrap BCa versus percentile intervals;
- winsorized and untrimmed continuous summaries for heavy tails;
- with and without infrastructure-only reruns;
- raw parsing failures counted as minimum score versus environment-executed invalid action;
- level-weighted macro versus episode micro average; and
- alternative operational thresholds shown as a curve, while retaining the preregistered gate as primary.

## 10. Reproducible analysis outputs

The final analysis must create one machine-readable result table from episode logs, one immutable analysis report, and a software-session manifest. Tables in the paper must be generated from that result table, not manually transcribed.
