# The Panopticon Protocol v3: What We Actually Trained

**Team:** Ayush Kumar & Ravi Prashant  
**Hackathon:** Meta PyTorch OpenEnv x Scaler - Grand Finale 2026  
**Demo Space:** [panopticon-protocol-v3](https://huggingface.co/spaces/Ayush-Kumar0207/panopticon-protocol-v3)  
**Model Repo:** [panopticon-argus-qwen-1.5B](https://huggingface.co/Ayush-Kumar0207/panopticon-argus-qwen-1.5B)

---

## The environment

The Panopticon Protocol v3 is an OpenEnv-compatible counter-espionage simulator where **ARGUS** defends a corporate network against **HYDRA** sleeper agents. The environment is built to stress:

- multi-agent deception and theory-of-mind
- long-horizon planning across 60 to 160 turns
- irreversible decisions under partial observability
- adaptive adversaries that punish fixed strategies

The five curriculum levels progressively introduce:

1. canary traps  
2. rotating leak channels  
3. false flags  
4. dead-man's switches  
5. double-agent turning and disinformation

This is why we believe the project fits the OpenEnv judging themes around **multi-agent interactions**, **long-horizon planning**, and **self-improving curricula**.

---

## The training run

The final public artifacts come from a **50-episode-per-level A10G run** using:

- base model: `Qwen/Qwen2.5-1.5B-Instruct`
- method: TRL SFT + LoRA
- curriculum: `easy -> medium -> hard -> level_4 -> level_5`
- total expert demonstrations: **29,000**
- final merged model uploaded to the model repo

### Important notebook clarification

The repo contains **two different notebooks**:

- **Actual end-to-end training notebook:** [Panopticon_Training_FINAL.ipynb](Panopticon_Training_FINAL.ipynb)
- **Plot regeneration notebook:** [Panopticon_Plots_Colab.ipynb](Panopticon_Plots_Colab.ipynb)

The training notebook runs the curriculum, saves `output_logs.txt`, merges the adapter, runs evaluation, and uploads artifacts.  
The plot notebook is only for rebuilding the figure suite from a finished log or evaluation JSON.

---

## Evidence that training really happened

The strongest submission-side evidence is public and reproducible:

- raw worker log: [`output_logs.txt`](output_logs.txt)
- local plot source: [`plots/`](plots)
- uploaded training metrics: [training_metrics/](https://huggingface.co/Ayush-Kumar0207/panopticon-argus-qwen-1.5B/tree/main/training_metrics)
- uploaded benchmark JSON: [evaluationResults.json](https://huggingface.co/Ayush-Kumar0207/panopticon-argus-qwen-1.5B/blob/main/evaluationResults.json)
- uploaded showcase JSON: [showcaseResults.json](https://huggingface.co/Ayush-Kumar0207/panopticon-argus-qwen-1.5B/blob/main/showcaseResults.json)

From the finished 50-episode run:

| Level | Examples | Avg Tokens | Final Loss | Loss Reduction |
|---|---:|---:|---:|---:|
| Easy | 3,000 | 573 | 0.0249 | 99.0% |
| Medium | 4,500 | 589 | 0.0220 | 90.6% |
| Hard | 6,000 | 608 | 0.0238 | 98.9% |
| Level 4 | 7,500 | 629 | 0.0212 | 91.0% |
| Level 5 | 8,000 | 647 | 0.0226 | 98.8% |

The training plots in the README are regenerated directly from that saved run, not hand-entered.

---

## What the benchmark taught us

The final structured benchmark is useful for a very specific reason: it shows that **training curves can look healthy while deployment behavior is still reward-misaligned**.

On the uploaded held-out evaluation:

- the **heuristic baseline** remains the best balanced security-preserving policy
- the **trained model** still accumulates too many invalid or operationally weak actions on harder levels
- the environment therefore exposes a real and interesting failure mode: token-level imitation success does **not automatically** produce robust strategic oversight behavior

That is not a bug in the README story. It is one of the most interesting research outcomes of the project.

In other words:

- **the training pipeline works**
- **the reward/loss evidence is real**
- **the environment is difficult enough to reveal where the trained agent still breaks**

For an OpenEnv submission, that is valuable. It means the environment is not a toy benchmark that every policy can fake its way through.

---

## Where judges should look

If you want the fastest path through the submission:

1. Try the environment Space: [panopticon-protocol-v3](https://huggingface.co/spaces/Ayush-Kumar0207/panopticon-protocol-v3)
2. Open the README plots and training summary in the repo
3. Open the actual training notebook: [Panopticon_Training_FINAL.ipynb](Panopticon_Training_FINAL.ipynb)
4. Inspect uploaded artifacts in the model repo:
   - [plots/](https://huggingface.co/Ayush-Kumar0207/panopticon-argus-qwen-1.5B/tree/main/plots)
   - [training_metrics/](https://huggingface.co/Ayush-Kumar0207/panopticon-argus-qwen-1.5B/tree/main/training_metrics)
   - [evaluationResults.json](https://huggingface.co/Ayush-Kumar0207/panopticon-argus-qwen-1.5B/blob/main/evaluationResults.json)

---

## Why we still think this submission matters

The Panopticon Protocol is not just a game skin. It is a deliberately structured environment for:

- hidden-state reasoning
- adversarial social inference
- long-horizon resource/security tradeoffs
- adaptive curriculum learning

The main claim of the project is not "we solved deception forever."  
The main claim is: **we built an OpenEnv environment where those failure modes become trainable, measurable, and visible.**
