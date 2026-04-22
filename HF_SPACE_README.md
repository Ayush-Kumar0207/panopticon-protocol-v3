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
- [Training Notebook (Colab)](https://colab.research.google.com/)
- [HuggingFace Blog Post](https://huggingface.co/blog/)

## License

Apache-2.0
