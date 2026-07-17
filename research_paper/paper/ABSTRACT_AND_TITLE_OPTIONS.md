# Title, Abstract, and Highlight Options

Use these only after final results are regenerated. The LaTeX abstract is the authoritative current draft.

## Recommended title

**Panopticon Protocol: Security-Gated Evaluation of Learned Agents in a Partially Observable Counter-Espionage Environment**

## Alternative titles

1. **When Average Agent Scores Hide Security Failures: The Panopticon Protocol Environment**
2. **Constraint Masking in Long-Horizon Agent Evaluation: A Panopticon Protocol Case Study**
3. **Separating Utility, Safety Gates, and Supervisor Attribution in a Partially Observable Agent Environment**
4. **Panopticon Protocol: Latent Compromise and Intervention-Aware Evaluation for Learned Agents**

Avoid “first,” “solved,” “safe agent,” “enterprise IDS,” and “production-ready.”

## 100-word abstract

Panopticon Protocol is a partially observable counter-espionage environment for evaluating long-horizon learned agents under hidden identity, deceptive evidence, and security–utility trade-offs. It separates dense training reward, five-dimensional grading, and hard advanced-tier security gates. In a preliminary matched evaluation, LoRA fine-tuning of Qwen2.5-1.5B improves macro grade from 0.641110 to 0.701627 but fails nine advanced acceptance checks. A separately evaluated deterministic supervisor reaches 0.790471 and passes the sampled gate, establishing controller-path solvability without proving raw-model safety. The artifact adds provenance-stratified evaluation and Constraint Masking Gap, which measures aggregate-pass but operational-fail episodes.

## Three submission highlights

- A typed latent-state environment exposes delayed, deceptive failures through a common rule/PPO/LLM action boundary.
- Independent reward, multidimensional grade, and hard gates reveal improvement that remains operationally unacceptable.
- Provenance-stratified metrics prevent repair and supervisor performance from being attributed to the raw model.

## Plain-language summary

An AI can earn a better average score while still making one kind of mistake that a real system cannot accept. Panopticon turns that problem into a repeatable simulation: an AI protects a fictional company while hidden threats appear over time and misleading evidence accumulates. The current trained model improves its average score but still fails the hardest security requirements. A hand-written supervisor succeeds, but the paper reports it as a different system. The project therefore focuses on measuring when averages hide failures and when a result depends on an external safety controller.
