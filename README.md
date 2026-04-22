# 👁️ The Panopticon Protocol v3

> **Counter-Espionage RL Environment — "Among Us… for AIs"**
>
> *Meta PyTorch OpenEnv Hackathon x Scaler — Grand Finale*
> *Theme: Multi-Agent Interactions | Sub-theme: Fleet AI Scalable Oversight*

[![HF Space](https://img.shields.io/badge/🤗_HuggingFace-Space-blue)](https://huggingface.co/spaces/)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-v0.2.1-green)](https://github.com/OpenEnvs/openenv)
[![License](https://img.shields.io/badge/License-Apache_2.0-red)](LICENSE)

An OpenEnv-compliant reinforcement learning environment where an AI security agent (**ARGUS**) defends a corporate network against an adaptive adversary (**HYDRA**) that infiltrates multi-generational sleeper agents of escalating sophistication.

**This environment trains the exact capabilities needed for Scalable AI Oversight: theory-of-mind, deception detection, strategic planning, and adaptive response.**

---

## 🎯 Why This Environment?

| AI Capability | How We Train It |
|---|---|
| **Theory-of-Mind** | Modeling hidden worker states from partial observations |
| **Deception Detection** | Distinguishing real leaks from false flags planted by Gen-3+ sleepers |
| **Strategic Planning** | Multi-step canary → monitor → investigate → neutralize reasoning chains |
| **Adaptive Response** | Re-calibrating strategy as HYDRA's adaptive memory evolves counter-tactics |
| **Risk Assessment** | Interrogate vs. terminate decisions with irreversible dead-man's switch consequences |

---

## 🌟 7 Stacking Espionage Mechanics

No single mechanic is complex — but **together they create emergent strategic depth** that demands genuine multi-step reasoning:

| # | Mechanic | Description | Gen |
|---|----------|-------------|-----|
| 1 | **Canary Traps** | Plant unique tracking data in departments to trace leaks back to their source | 1+ |
| 2 | **Multi-Gen Sleepers** | From amateur leakers (Gen-1) to Manchurian candidates (Gen-5) with active counter-intelligence | 1-5 |
| 3 | **False Flags** | High-gen sleepers frame innocent workers — verify before you act! | 3+ |
| 4 | **Dead-Man's Switches** | Terminate a Gen-4 without interrogation → devastating data breach | 4+ |
| 5 | **Double Agent Turning** | Convert caught sleepers into YOUR assets (4-turn investment, high risk/reward) | 3+ |
| 6 | **Disinformation Campaigns** | Feed false intel back to HYDRA through turned double agents | 3+ |
| 7 | **HYDRA Adaptive Memory** | The adversary remembers your strategies and evolves counter-tactics | All |

## 🎮 6-Phase Narrative Arc

Every episode follows a dramatic arc from calm to crisis to triumph:

```
Phase 1: Orientation    (T0-30)   — Build baseline, plant canaries
Phase 2: First Contact  (T30-60)  — Gen-1 activates, first leak detected
Phase 3: Escalation     (T60-90)  — False flags appear, trust no one
Phase 4: Deep Cover     (T90-120) — Dead switches, deepest crisis
Phase 5: Crisis         (T120-140)— Manchurian candidate, overwhelming pressure
Phase 6: Counterstrike  (T140-160)— Double agent payoff, REWARD SURGE
```

## 📊 5-Dimension Grading System

| Dimension | Weight | What It Measures |
|-----------|--------|------------------|
| **Security Outcome** | 30% | Catch rate, false accusation penalty, final security score |
| **Enterprise Revenue** | 25% | Revenue stability, peak revenue, V-recovery shape |
| **Intelligence Craft** | 20% | Canary effectiveness, double agent mastery, disinformation |
| **Adaptability** | 15% | Strategy diversity, phase-appropriate tactics |
| **Efficiency** | 10% | Action economy, invalid action rate |

## 🏆 5 Difficulty Tiers

| Level | Name | Sleepers | Key Mechanic | Turns |
|-------|------|----------|-------------|-------|
| easy | Amateur | 1x Gen-1 | Canary basics | 60 |
| medium | Professional | Gen-1 + Gen-2 | Channel monitoring | 90 |
| hard | Spy Network | Gen-1/2/3 | False flags, verification | 120 |
| level_4 | Terror Cell | Gen-1-4 | Dead-man's switches | 150 |
| level_5 | Manchurian | Gen-1-5 | Full toolkit required | 160 |

---

## ⚡ Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server (OpenEnv-compliant)
uvicorn server:app --host 0.0.0.0 --port 8000

# Verify all 5 levels pass
python smoke_test.py

# Train with native PPO (fast)
python train_rl.py --curriculum

# Train LLM with HuggingFace TRL (fine-tune Qwen 0.5B)
python train_trl.py --curriculum

# Generate training reward curves
python plot_training.py

# Run LLM inference
HF_TOKEN=your_token python inference.py

# Full E2E pipeline verification
python e2e_verify.py
```

## 🕹️ Action Space

8 action types with sub-action modifiers — `MultiDiscrete([8, 8, 7])`:

| Category | Action | Sub-actions | Target |
|----------|--------|-------------|--------|
| **Productivity** | `work` | — | Department |
| **Productivity** | `hire` | — | Department |
| **Intelligence** | `canary` | — | Department |
| **Intelligence** | `monitor` | — | Leak Channel |
| **Intelligence** | `investigate` | `audit` / `verify` / `correlate` | Worker / Leak / Dept |
| **Enforcement** | `neutralize` | `terminate` / `interrogate` / `turn` | Worker |
| **Enforcement** | `deploy_double` | — | Double Agent |
| **Meta** | `noop` | — | — |

## 🏗️ Architecture

```
models.py           — Pydantic v2 data models (Worker, Leak, Canary, DoubleAgent, etc.)
environment.py      — Core game engine (6 phases, HYDRA AI, 7 mechanics, 1100+ LOC)
grader.py           — 5-dimension programmatic grader (OpenEnv-compliant)
tasks/              — 5 difficulty tiers with grader registry
gym_wrapper.py      — Gymnasium adapter (136-dim obs, MultiDiscrete action space)
train_rl.py         — Native PyTorch PPO with 3-head actor network
train_trl.py        — HuggingFace TRL PPOTrainer with LoRA (Qwen 0.5B)
_server.py          — FastAPI server (11 endpoints, OpenEnv-compliant)
inference.py        — LLM agent inference (any OpenAI-compatible API)
smoke_test.py       — Heuristic verification across all 5 levels
benchmark_suite.py  — Comparative agent evaluation
plot_training.py    — Publication-quality reward curve generation
```

## 🤝 Training Pipeline

### Traditional RL (PPO)
```
gym_wrapper.py → 136-dim observation vector → 3-head actor network → MultiDiscrete action
```

### LLM Fine-Tuning (TRL)
```
environment.py → JSON observation → LLM prompt → JSON action → PPO reward signal
```

Both pipelines share the same environment and grading system, enabling direct comparison.

## 🐳 Docker

```bash
docker build -t panopticon-v3 .
docker run -p 8000:8000 panopticon-v3
```

## 📜 Team

- **Ayush Kumar** — Environment Design, RL Pipeline, System Architecture
- **Ravi Prashant** — Training, Evaluation, Deployment

## 📜 License

Apache-2.0
