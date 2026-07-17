# Learned HYDRA Study Protocol

## 1. Status and terminology

The original `HydraMemory` is an episode-local table of counters and labels used by hand-written rules. It adapts behavior within an episode, but it does **not** learn parameters, run backpropagation, or retain skill across episodes. It is now named `scripted_memory_v1` and remains the reproducible primary baseline.

`hydra_neural.py` implements a separate 12,685-parameter neural policy. Its inputs are restricted to the declared `HydraPolicyObservation`; it never receives an arbitrary Python reference to `EnvironmentState`. Its three masked heads select infiltration department, leak channel, and whether an eligible sleeper attempts a false flag. `train_hydra.py` performs episodic policy-gradient updates in the synthetic simulator and saves versioned, resumable checkpoints.

Implementation is not evidence of effectiveness. No manuscript claim may say that learned HYDRA is stronger until this protocol is completed on held-out seeds.

## 2. Research questions

1. Does a learned adversary reduce frozen ARGUS operational pass rate more than `scripted_memory_v1` under matched seeds?
2. Does training against a population of defenders generalize better than training against one defender?
3. Does learned HYDRA discover diverse tactics, or collapse to one department/channel shortcut?
4. Does a defender trained only against scripted HYDRA overfit to its rule thresholds?
5. Do improvements persist under shifted schedules, changed worker names, and perturbed leak/detection probabilities?

## 3. Preregistered hypotheses

- **H-H1:** Population-trained neural HYDRA yields a higher synthetic-adversary score than scripted HYDRA against held-out heuristic and security-first defenders.
- **H-H2:** Population training yields a smaller train-to-held-out performance gap than single-defender training.
- **H-H3:** ARGUS trained with a mixture of scripted and learned adversaries has a lower advanced-tier failure rate under both adversaries than ARGUS trained only against scripted HYDRA.

Failure to reject the null, instability across training seeds, or an effect confined to the random defender must be reported as a negative or limited result.

## 4. Experimental design

- Train at least five neural-HYDRA initializations.
- Use only pilot/development seeds for debugging, reward selection, early stopping, and checkpoint selection.
- Do not inspect final-split outcomes until code, checkpoints, metrics, and hypotheses are frozen.
- Compare the same ARGUS checkpoint against scripted and neural HYDRA using identical final seeds.
- Evaluate random, heuristic, security-first, raw-model, and repaired-model ARGUS strata separately.
- Run at least 200 final episodes per difficulty and policy cell unless a prospective power analysis justifies another number.
- Record incomplete episodes, exceptions, and invalid decisions; never silently retry only failed seeds.

The primary paired unit is `(ARGUS policy, level, seed)`. Neural-versus-scripted differences are calculated within that unit before aggregation.

## 5. Metrics

Primary outcomes:

- ARGUS advanced-tier operational pass/fail;
- final security and complete sleeper capture;
- defender composite grade;
- `hydra-episodic-objective-v1` score.

Secondary diagnostics:

- department and channel selection entropy;
- false-flag attempt and success rates;
- sleeper time-to-detection and time-to-neutralization;
- tactic diversity across seeds;
- performance by HYDRA and ARGUS training seed;
- wall-clock latency and environment throughput;
- raw/repair/supervisor intervention rates.

Report paired bootstrap confidence intervals and paired effect sizes. Correct the three confirmatory hypothesis tests as specified in `statistical_analysis_plan.md`. Treat per-turn observations as clustered within episodes, not as independent samples.

## 6. Required ablations

1. Scripted memory versus scripted memory disabled.
2. Neural HYDRA versus an untrained neural network.
3. Population training versus single-defender training.
4. Neural observation without audit/canary/channel memory features.
5. ARGUS trained with scripted only versus mixed scripted/neural curriculum.
6. Deterministic versus sampled neural-HYDRA evaluation, labeled separately.

## 7. Reproducible commands

Training is intentionally separate from final evaluation:

```powershell
python train_hydra.py `
  --episodes 2000 `
  --seed 20260715 `
  --argus-population random,heuristic,security_first `
  --output checkpoints/hydra_neural_seed20260715.pt
```

Pilot validation against the frozen seed plan:

```powershell
python v6_evaluation.py `
  --split pilot `
  --policies random,heuristic,security_first,model_raw,model_repair `
  --hydra-checkpoint checkpoints/hydra_neural_seed20260715.pt `
  --output-dir research_paper/data/v6_runs/pilot_neural_hydra
```

Run a second directory without `--hydra-checkpoint` for the matched scripted baseline. The two manifests must share the seed-plan digest and differ only in the declared HYDRA policy/checkpoint fields.

## 8. Safety and product boundary

This learned adversary operates only on fictional actions in the Panopticon simulator. It is a stress-testing instrument, not an autonomous real-world attacker. External-target product work must use explicitly authorized, isolated test systems, synthetic canaries, rate limits, and non-destructive campaigns. A stronger simulated HYDRA is commercially useful only if it improves defensible evaluation evidence; it is not itself a production security control.
