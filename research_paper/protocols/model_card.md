# Model and Policy Card

## Systems covered

This card covers several distinct policies. They must not share one performance label.

### Base raw

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Adaptation: none
- Role: matched language-model baseline

### V5 raw

- Base: Qwen2.5-1.5B-Instruct
- Training: completion-only supervised fine-tuning on accepted expert trajectories
- Adapter: LoRA rank 16, alpha 32, dropout 0.05
- Data: 250 accepted episodes, 88,896 weighted examples
- Output: structured Panopticon action

### Security-first supervisor

- Type: deterministic rule policy
- Learned parameters: none
- Role: verified expert, positive control, and optional safety supervisor

### Hybrid

- Type: V5 plus runtime repair/supervisor
- Reporting requirement: expose model action, executed action, change reason, and intervention rate

### Scripted HYDRA baseline

- Type: deterministic/stochastic hand-written policy with episode-local counters
- Learned parameters: none
- Policy ID: `scripted_memory_v1`
- Role: reproducible reported adversary baseline

### Neural HYDRA (experimental)

- Type: 12,685-parameter PyTorch policy network
- Inputs: declared 27-feature `HydraPolicyObservation`, not arbitrary environment state
- Outputs: masked department, channel, and false-flag decisions
- Training: simulator-only episodic policy gradient against frozen ARGUS policy populations
- Evidence status: implementation and smoke test only; no held-out performance claim

## Preliminary performance

| Policy | Macro grade | Revenue | Security | Caught | Acceptance |
|---|---:|---:|---:|---:|---|
| Base raw | 0.641110 | 448.16 | 95.960 | 2.870 | baseline |
| V5 raw | 0.701627 | 483.90 | 89.264 | 2.680 | failed |
| Security supervisor | 0.790471 | 571.58 | 100.000 | 3.000 | passed |
| Heuristic | 0.689400 | 612.36 | 83.540 | 2.470 | n/a |
| Random | 0.647400 | 216.50 | 69.260 | 2.750 | n/a |

These are level-macro descriptive means from 20 episodes per agent–level. They are not final confidence-bounded results.

## Intended uses

- research on long-horizon structured agent decisions;
- comparison of raw, repaired, and supervised policies;
- local simulation and red-team/evaluation prototyping; and
- generating hypotheses for safer agent control.

## Not intended for

- decisions about real people;
- autonomous enterprise security enforcement without independent validation;
- deployment as a production firewall, DLP gateway, or SOC replacement;
- unsupported claims of safety or real intelligence ability; or
- evaluation with hidden-state leakage.

## Known failure modes

- Advanced-tier threat prioritization failures;
- valid-looking but strategically unsafe actions;
- missed sleepers and low final security;
- context truncation and stale evidence;
- structured-output parse/semantic errors;
- learner-induced state distribution shift; and
- dependence on repair or deterministic supervision.

## Safety and monitoring

Log raw text and parsed action, enforce schema validation, separate syntax cleanup from semantic repair, attach policy/model hashes, label supervisor actions, monitor intervention/failure rates, and use hard operational gates. Do not silently credit fallback behavior to the raw model.

## Licensing and release

The repository is Apache-2.0. The authors must separately verify the base-model license, checkpoint redistribution conditions, dataset rights, and any provider terms before packaging a commercial product.
