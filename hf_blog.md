# The Panopticon Protocol v3: What We Actually Trained

**Team:** Ayush Kumar & Ravi Prashant  
**Hackathon:** Meta PyTorch OpenEnv x Scaler - Grand Finale 2026  
**Demo Space:** [panopticon-protocol-v3](https://huggingface.co/spaces/Ayush-Kumar0207/panopticon-protocol-v3)  
**Model Repo:** [panopticon-argus-qwen-1.5B](https://huggingface.co/Ayush-Kumar0207/panopticon-argus-qwen-1.5B)

---

## The environment

The Panopticon Protocol v3 is an OpenEnv-compatible counter-espionage simulator where **ARGUS** defends a corporate network against **HYDRA** sleeper agents. It stresses:

- multi-agent deception and theory-of-mind
- long-horizon planning across 60 to 160 turns
- irreversible decisions under partial observability
- adaptive adversaries that punish fixed strategies

The five curriculum levels progressively introduce canary traps, rotating leak channels, false flags, dead-man's switches, and finally double-agent turning plus disinformation.

---

## The latest V5 run

The latest selected Drive artifacts come from `panopticon-security-v5-ep50`, a **50-episode-per-level Security-First V5 run** using:

- base model: `Qwen/Qwen2.5-1.5B-Instruct`
- method: TRL SFT + LoRA
- curriculum: `easy -> medium -> hard -> level_4 -> level_5`
- training data: **250 expert episodes** and **88,896 supervised examples**
- max sequence length: `512`
- seed: `42`
- output: Drive-saved adapters, checkpoints, logs, evaluation JSONs, acceptance reports, and merged model

| Level | Episodes | Supervised Examples |
|---|---:|---:|
| Easy | 50 | 7,430 |
| Medium | 50 | 13,166 |
| Hard | 50 | 18,414 |
| Level 4 | 50 | 23,889 |
| Level 5 | 50 | 25,997 |
| **Total** | **250** | **88,896** |

The full optimizer event log remains in Drive as `training_events.jsonl`; the repo keeps compact checked-in summaries and regenerated plots so reviewers can inspect the latest results without multi-GB downloads.

---

## What the benchmark taught us

The final V5 benchmark gives a more precise answer than training loss alone:

| Policy | Macro Grade | Security | Sleepers Caught | Gate |
|---|---:|---:|---:|---|
| Base untrained Qwen | 0.64111 | 95.96 | 2.87 | Reference |
| Raw V5 trained model | 0.701627 | 89.26 | 2.68 | Failed |
| Security-first supervisor | 0.790471 | 100.00 | 3.00 | Passed |
| Heuristic | 0.6894 | 83.54 | 2.47 | Baseline |

The raw V5 trained model **did improve** over the untrained base and slightly beat the heuristic on macro grade. But it did **not** pass the strict security-first acceptance gate: Level 4 and Level 5 still had pass-rate, security, missed-sleeper, and false-accusation failures.

The security-first supervisor diagnostic passed every check. That is the strongest operational result, but it should be interpreted honestly: it validates the controller/policy path, not the raw neural model alone.

So the project outcome is not a simple victory lap. It is better and more useful than that:

- the environment exposes a real advanced-tier safety failure mode;
- the raw V5 model learned meaningful behavior but is not fully aligned;
- the security-first controller solves the measured gate;
- the benchmark now tells us exactly what must improve next.

---

## Where judges should look

1. Try the environment Space: [panopticon-protocol-v3](https://huggingface.co/spaces/Ayush-Kumar0207/panopticon-protocol-v3)
2. Open the README's Security-First V5 benchmark and refreshed plot gallery
3. Inspect [`evaluation_comparison_latest.json`](evaluation_comparison_latest.json)
4. Use [`COLAB_SECURITY_V5_TRAINING.md`](COLAB_SECURITY_V5_TRAINING.md) for the checkpoint-resumable Colab workflow

---

## Why this still matters

The main claim is not "we solved deception forever." The main claim is: **we built an OpenEnv environment where deception-detection failure modes become trainable, measurable, and visible.**

The latest V5 results make that claim stronger, because they distinguish raw model improvement from true security acceptance.
