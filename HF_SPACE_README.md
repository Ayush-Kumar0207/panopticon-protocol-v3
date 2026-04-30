---
title: "The Panopticon Protocol v3: Teaching AI to Catch AI Spies"
emoji: "👁️"
colorFrom: purple
colorTo: red
sdk: docker
pinned: true
license: apache-2.0
---

# 👁️ The Panopticon Protocol v3

> **Counter-Espionage RL Environment — "Among Us… for AIs"**

An OpenEnv-compliant reinforcement learning environment where an AI security agent (**ARGUS**) defends a corporate network against an adaptive adversary (**HYDRA**) that infiltrates multi-generational sleeper agents.

## What Makes This Unique

**7 Stacking Espionage Mechanics** that create emergent strategic depth:

| Mechanic | Description |
|----------|-------------|
| **Canary Traps** | Plant unique tracking data to trace leaks |
| **5-Gen Sleepers** | From amateur leakers to Manchurian candidates |
| **False Flags** | Gen-3+ sleepers frame innocents |
| **Dead-Man's Switches** | Gen-4 explode on careless termination |
| **Double Agent Turning** | Convert enemies into YOUR assets |
| **Disinformation** | Feed false intel back to the adversary |
| **HYDRA Adaptive Memory** | The enemy evolves its counter-tactics |

## 6-Phase Narrative Arc

```
Orientation → First Contact → Escalation → Deep Cover → Crisis → Counterstrike
```

## Why This Matters

This environment trains the exact capabilities needed for **Scalable AI Oversight**:
- **Theory-of-Mind**: Modeling hidden states and intentions of other agents
- **Deception Detection**: Identifying false signals in noisy, adversarial data
- **Strategic Planning**: Multi-step reasoning under partial observability
- **Adaptive Response**: Adjusting tactics as the adversary evolves

## API Endpoints

- `GET /health` — Health check
- `POST /reset` — Start a new episode
- `POST /step` — Take an action
- `GET /tasks` — List available difficulty levels
- `POST /grade/{level}` — Grade an episode

## Quick Start

```bash
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 8000
```

## Links

- [OpenEnv Spec](https://github.com/OpenEnvs/openenv)
- [Canonical Training Notebook in Repo](https://github.com/Ayush-Kumar0207/panopticon-protocol-v3/blob/main/Panopticon_Training_FINAL.ipynb)
- [Compatibility Copy For Submitted Colab Link](https://github.com/Ayush-Kumar0207/panopticon-protocol-v3/blob/main/Panopticon_Plots_Colab.ipynb)
- [Submitted Colab URL](https://colab.research.google.com/drive/1-MIjo3qqII3s-Y6v4xfcRN7jLS4WQ3qe?usp=sharing)
- [Blog Post](https://huggingface.co/spaces/Ayush-Kumar0207/panopticon-protocol-v3/blob/main/blog.md)
- [Model Repo](https://huggingface.co/Ayush-Kumar0207/panopticon-argus-qwen-1.5B)
- [Training Metrics](https://huggingface.co/Ayush-Kumar0207/panopticon-argus-qwen-1.5B/tree/main/training_metrics)

> For avoidance of doubt: `Panopticon_Training_FINAL.ipynb` is the canonical end-to-end training notebook. `Panopticon_Plots_Colab.ipynb` keeps the submitted filename and now mirrors the training flow so the already-shared Colab link still shows a real training script.

## License

Apache-2.0
