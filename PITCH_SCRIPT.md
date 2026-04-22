# Panopticon Protocol v3 -- 3-Minute Pitch Script

## SLIDE 1: Hook (0:00 - 0:30)

**[Speaker looks at audience]**

"What if I told you the biggest threat to AI isn't hackers from outside...
it's AI agents hiding *inside* your system, pretending to be loyal?

That's is the AI alignment problem in a nutshell.

We built an environment where an AI learns to catch OTHER AI spies.
It's called **The Panopticon Protocol**.
Think of it as... *Among Us... for AIs.*"

---

## SLIDE 2: The Environment (0:30 - 1:30)

**[Show environment diagram]**

"You are ARGUS — a corporate AI security agent.
Your company has been infiltrated by HYDRA — an adaptive adversary
that plants *sleeper agents* in your workforce.

These aren't simple enemies. They evolve across 5 GENERATIONS:

- Gen-1 is an amateur leaker. Easy to spot.
- Gen-3 plants FALSE FLAGS — it frames innocent workers to waste your time.
- Gen-4 has a DEAD-MAN'S SWITCH — fire them carelessly and they trigger a data breach.
- Gen-5 is a Manchurian candidate — virtually undetectable.

You have 7 tools to fight back:

1. Plant CANARY TRAPS — traceable data that reveals the leaker
2. MONITOR channels for canary matches
3. INVESTIGATE — audit workers, verify leaks, correlate signals
4. NEUTRALIZE — but INTERROGATE first to avoid triggering switches
5. TURN caught spies into DOUBLE AGENTS working for you
6. Deploy double agents to run DISINFORMATION campaigns
7. The adversary has ADAPTIVE MEMORY — it LEARNS your tactics

Every episode plays out across a 6-phase narrative arc,
from Orientation all the way to Counterstrike — where your double agents
deliver a massive reward surge."

---

## SLIDE 3: Training & Results (1:30 - 2:15)

**[Show reward curves]**

"We trained agents using BOTH traditional PPO AND HuggingFace TRL —
fine-tuning an LLM to reason about deception.

Here are the reward curves across 5 difficulty levels.

Notice how the agent learns to:
- Plant canaries early in the game
- Verify before accusing — avoiding false flags
- Interrogate before terminating — dodging dead-man's switches
- Eventually turn caught spies into double agents for the Counterstrike surge

At the hardest level — Manchurian Protocol —
only agents that master the FULL toolkit can survive."

**[Show before/after comparison]**

"Before training: random actions, security hits zero by turn 30.
After training: strategic countermeasures, 100% security maintained."

---

## SLIDE 4: Why This Matters (2:15 - 2:45)

"This isn't just a game. This environment directly trains the capabilities
needed for SCALABLE AI OVERSIGHT:

- Theory of Mind — modeling hidden states of other agents
- Deception Detection — separating real signals from noise
- Strategic Planning — multi-step reasoning under uncertainty
- Adaptive Response — evolving your strategy as the adversary evolves

This is exactly what we need to build trustworthy AI systems.

We're targeting the FLEET AI Scalable Oversight sub-theme
because ARGUS IS an oversight agent — monitoring, analyzing,
and explaining deceptive behavior in multi-agent systems."

---

## SLIDE 5: Close (2:45 - 3:00)

**[Show HF Space link and QR code]**

"The Panopticon Protocol v3.
7 mechanics. 6 phases. 5 difficulty levels.
One question: Can your AI catch the spy?

Try it live on our HuggingFace Space.
Thank you."

---

## Q&A PREP (2 minutes)

**Expected Questions:**

**Q: How is this different from Mafia/Werewolf games?**
A: Those are static. Our adversary has ADAPTIVE MEMORY — it evolves counter-tactics
based on your past games. Plus our 7-mechanic stack creates emergent complexity
that simple social deduction games don't have.

**Q: Can an LLM actually learn to play this?**
A: Yes — we demonstrate this with TRL training. The key insight is that the observation
space is structured text (JSON), so an LLM can naturally reason about it.
We show clear reward improvement with Qwen 0.5B fine-tuned via LoRA+PPO.

**Q: What's the observation/action space?**
A: Observation is a JSON object with worker states, active leaks, canary traps,
and game metrics. Action is a 3-part choice: ActionType (8 options),
Target (worker/dept/channel), SubAction (7 modifiers).
For RL we flatten to a 136-dim vector with MultiDiscrete(8,8,7) actions.

**Q: How does grading work?**
A: 5-dimension programmatic grading: Security (30%), Revenue (25%),
Intelligence (20%), Adaptability (15%), Efficiency (10%).
Each level has pass/fail thresholds calibrated to the difficulty.

**Q: What theme does this target?**
A: Multi-Agent Interactions (Theme 1) with the Fleet AI Scalable Oversight
sub-theme. ARGUS is literally an AI oversight agent monitoring other agents.
