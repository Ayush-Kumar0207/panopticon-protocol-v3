# The Panopticon Protocol v3: Teaching AI to Catch AI Spies

*An OpenEnv-compliant RL environment for training scalable AI oversight agents*

---

## The Problem

As AI systems become more autonomous, we face a critical challenge: **How do you train an AI to detect when other AIs are being deceptive?** This isn't science fiction — it's the core problem of AI alignment and scalable oversight.

## Our Solution: "Among Us... for AIs"

We built **The Panopticon Protocol v3**, a counter-espionage RL environment where an AI defender (ARGUS) must identify and neutralize hidden adversarial agents within a corporate network. Think of it as a high-stakes game of deception where every employee might be a spy.

### 7 Stacking Mechanics Create Emergent Depth

What makes our environment unique is that no single mechanic is complex — but **together they create emergent strategic behavior** that demands genuine reasoning:

1. **Canary Traps** — Plant traceable data in departments. When it leaks, you know where the spy is.
2. **Multi-Generation Sleepers** — Gen-1 is obvious. Gen-5 is a Manchurian candidate with active counter-intelligence.
3. **False Flags** — High-gen sleepers frame innocent workers. Act without verifying and *you* damage the company.
4. **Dead-Man's Switches** — Terminate a Gen-4 without interrogating first? It triggers a devastating data breach.
5. **Double Agent Turning** — Risky 4-turn investment, but a turned sleeper becomes your most powerful asset.
6. **Disinformation Campaigns** — Use double agents to feed false intel back to the adversary.
7. **HYDRA Adaptive Memory** — The adversary remembers what worked and evolves its strategy.

### 6-Phase Narrative Arc

Every episode follows a dramatic arc:

```
Orientation → First Contact → Escalation → Deep Cover → Crisis → Counterstrike
```

The first phases are about information gathering. By Phase 5, you're in crisis mode. Phase 6 is where trained agents *shine* — turning the adversary's own assets against them for a massive reward surge.

## Training Results

We trained agents across 5 difficulty tiers using both PPO (for RL) and HF TRL (for LLM fine-tuning):

- **Amateur**: Single Gen-1 sleeper. Optimal agents achieve 100% security score.
- **Professional**: Gen-1 + Gen-2. Requires active channel monitoring.
- **Spy Network**: Gen-1/2/3 with false flags. Verification becomes critical.
- **Terror Cell**: Gen-1-4 with dead-man's switches. Interrogation-first strategy required.
- **Manchurian**: Full 5-gen gauntlet. Only agents that master double-agent deployment survive.

The reward curves show clear learning signal across all 5 levels — agents progress from random behavior to sophisticated counter-intelligence playbooks.

## Why This Environment Matters

The Panopticon Protocol directly trains capabilities needed for **Scalable AI Oversight**:

| Capability | How We Train It |
|---|---|
| Theory-of-Mind | Modeling hidden worker states from partial observations |
| Deception Detection | Distinguishing real leaks from false flags |
| Strategic Planning | Multi-step canary → monitor → investigate → neutralize chains |
| Adaptive Response | Re-calibrating strategy as HYDRA evolves counter-tactics |
| Risk Assessment | Interrogate vs. terminate decisions with irreversible consequences |

## Technical Stack

- **Environment**: Custom Gymnasium-compatible env with Pydantic v2 models
- **Server**: FastAPI with OpenEnv-compliant grading endpoints
- **RL Training**: Native PyTorch PPO with 3-head actor network
- **LLM Training**: HuggingFace TRL PPOTrainer with LoRA adapters
- **Grading**: 5-dimension programmatic grader (Security, Revenue, Intelligence, Adaptability, Efficiency)

## Links

- **HuggingFace Space**: [Coming soon]
- **GitHub**: [panopticon-protocol-v3](https://github.com/Ayush-Kumar0207/panopticon-protocol-v3)
- **Colab Notebook**: [Training Demo](https://colab.research.google.com/)

---

*Built for the Meta PyTorch OpenEnv Hackathon x Scaler School of Technology Grand Finale, April 2026.*
*Team: Ayush Kumar & Ravi Prashant*