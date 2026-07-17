# Interview Blueprint — The Panopticon Protocol v3, Explained as a Story

## Read This First

This document is written for a reader who may never have studied Python, APIs, reinforcement learning, neural networks, game theory, LLM fine-tuning, or system design.

It deliberately explains small terms instead of treating them as assumed knowledge. When a technical word first matters, the surrounding text explains what it means, why the project needs it, and where it appears in the repository. A large glossary near the end provides a second explanation for revision.

The project can be understood through one story:

> A defensive AI named ARGUS manages a company. An adversary named HYDRA secretly inserts spies into the workforce. ARGUS cannot see who the spies are. It must preserve the company's revenue while using evidence—canary traps, leak monitoring, audits, correlation, interrogation, and double agents—to identify and contain the spies.

The engineering thesis is:

> Panopticon is not merely a game with an AI attached. It is a partially observable, adversarial decision environment with typed state, controlled state transitions, two agent interfaces, two training paths, deterministic grading, reproducible evaluation, and explicit release gates.

The shortest mental model is:

```text
models.py          = the vocabulary and legal data shapes
environment.py     = the referee and world simulator
gym_wrapper.py     = the numeric translator for PPO
train_rl.py        = the native reinforcement-learning path
train_trl_v2.py    = the production LLM curriculum path
security_policy.py = the verified expert playbook
inference_local.py = model execution, repair, and evaluation
grader.py          = the independent scorekeeper
_server.py         = the HTTP doorway
```

### How to use this guide

- If you have **two minutes**, read Sections 1 and 25.
- If you are a **beginner**, read in order and do not skip the Cast of Characters or glossary.
- If you are preparing for an **ML interview**, focus on Sections 6–9, 17–20, and 23.
- If you are preparing for a **system-design interview**, focus on Sections 2, 10–14, 21, and 24.
- If you are presenting the project, use the 45-minute script in Section 15 and the closing answers in Section 22.

### A necessary honesty rule

This guide separates four kinds of statement:

1. **Implemented behavior** — directly present in the checked-in code.
2. **Measured behavior** — reported by checked-in evaluation artifacts.
3. **Design intention** — what the architecture is trying to achieve.
4. **Future design** — a proposed improvement, not current behavior.

That distinction matters in interviews. A strong engineer does not present a roadmap as if it already exists, or a supervisor-assisted result as if it came from the raw neural model.

> **Team:** Ayush Kumar and Ravi Prashant  
> **One-liner:** *“Among Us… for AIs.”*  
> **Primary project form:** a turn-based counter-espionage learning environment

---

## Table of Contents

1. [The Beginner-Friendly Elevator Pitch](#1-the-elevator-pitch)
2. [Macro Architecture](#2-macro-architecture)
3. [The Data Layer and Pydantic State Machine](#3-the-data-layer--pydantic-state-machine)
4. [The Environment Engine: One Episode as a Story](#4-the-environment-engine-deep-dive)
5. [The Five Difficulty Tiers](#5-the-5-difficulty-tiers)
6. [The Gymnasium Wrapper and 136-Number Observation](#6-the-gymnasium-wrapper--observation-space)
7. [Native PPO Reinforcement Learning](#7-the-rl-training-pipeline-native-ppo)
8. [LLM Fine-Tuning with TRL, SFT, and LoRA](#8-the-llm-fine-tuning-pipeline-trl-sft)
9. [Multi-Dimensional Grading and Hard Security Gates](#9-the-multi-dimensional-grading-system)
10. [The FastAPI and OpenEnv API Layer](#10-the-api-layer--fastapi-server)
11. [Security-First Policy, Smoke Tests, and Benchmarks](#11-the-heuristic-agent-smoke-test)
12. [Deployment and DevOps](#12-deployment--devops)
13. [Reward Failure, Security Death Spirals, and the V2 Fix](#13-the-security-death-spiral-fix)
14. [Scaling and Production System Design](#14-future-scalability)
15. [A Full 45-Minute Interview Story](#15-a-full-45-minute-interview-story)
16. [Quick Reference Cheat Sheet](#16-quick-reference--cheat-sheet)
17. [The Current V5 Training and Evaluation Story](#17-the-current-v5-training-and-evaluation-story)
18. [Beginner Glossary](#18-beginner-glossary)
19. [Algorithm and Concept Implementation Sheet](#19-algorithm-and-concept-implementation-sheet)
20. [Beginner-Friendly Interview Questions](#20-beginner-friendly-interview-questions)
21. [Strengths and Honest Gaps](#21-strengths-and-honest-gaps)
22. [One-Minute and Senior-Engineer Closing Answers](#22-one-minute-and-senior-engineer-closing-answers)
23. [Machine-Learning Interview Questions with Strong Answers](#23-machine-learning-interview-questions-with-strong-answers)
24. [System Design Interview Questions with Strong Answers](#24-system-design-interview-questions-with-strong-answers)
25. [Ultra-Concise Revision Checklist](#25-ultra-concise-revision-checklist)

---

## 1. The Elevator Pitch

> *"Imagine Among Us, but every player is an AI — and instead of a spaceship, you're defending a Fortune 500 corporation from sleeper agents."*

The **Panopticon Protocol v3** is a turn-based, counter-espionage Reinforcement Learning environment built for the **Meta PyTorch OpenEnv Hackathon Grand Finale**. Here's the core idea in plain English:

- **You are ARGUS** — an AI security chief defending a corporate network.
- **Your enemy is HYDRA** — an adaptive adversary that infiltrates sleeper agents (spies) into your workforce.
- **Your weapons:** Canary traps, interrogations, double-agent turning, and disinformation campaigns.
- **HYDRA's weapons:** 5 generations of increasingly sophisticated sleepers — from amateurs who leak obvious data, all the way to Manchurian candidates who are your top performers but secretly reporting to the enemy.

### The Core Game Loop (Visual)

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                     ⚡ THE CORE GAME LOOP ⚡                                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                 ║
║   │ 🔄 ARGUS     │────▶│ ⚔️  HYDRA    │────▶│ 💰 ECONOMY  │                 ║
║   │    Acts      │     │  Responds    │     │    Tick      │                 ║
║   └──────────────┘     └──────────────┘     └──────┬───────┘                 ║
║          ▲                                         │                         ║
║          │                                         ▼                         ║
║          │                                  ┌──────────────┐                 ║
║          │                                  │ 📊 REWARD    │                 ║
║          │                                  │   Computed   │                 ║
║          │                                  └──────┬───────┘                 ║
║          │                                         │                         ║
║          │                                         ▼                         ║
║          │                                  ┌──────────────┐                 ║
║          │              ┌───── NO ──────────│  Game Over?  │                 ║
║          │              │                   └──────┬───────┘                 ║
║          └──────────────┘                          │ YES                     ║
║                                                    ▼                         ║
║                                             ┌──────────────┐                 ║
║                                             │ 🏆 GRADE     │                 ║
║                                             │   Episode    │                 ║
║                                             └──────────────┘                 ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### Why This Environment Wins Competitions

| DIFFERENTIATOR | WHY JUDGES CARE | HOW WE DELIVER IT |
|---|---|---|
| **Information Asymmetry** | Tests genuine reasoning, not pattern matching | Agent NEVER sees `hidden_state` or `is_sleeper` — must infer through evidence chains |
| **Dual-Objective Tension** | Avoids degenerate policies (fire everyone / ignore everything) | Reward = 0.45 × productivity + 0.55 × security — both MUST be optimized |
| **Adaptive Adversary** | Prevents memorization; forces generalization | `HydraMemory` tracks agent patterns and adapts sleeper placement, leak channels, and timing |
| **Stacking Complexity** | Shows emergent difficulty, not just "more enemies" | Each generation ADDS a mechanic (canary evasion → false flags → dead switches → Manchurian) |
| **Narrative Arc** | Makes demos compelling for non-technical judges | 6 phases with a CRESCENDO — the Counterstrike surge where the agent flips the advantage |
| **Dual Training Paths** | Shows both RL and LLM mastery | PPO (136-dim vectors) AND SFT on Qwen 2.5 (text observations → JSON actions) |

### Game Theory Classification

The reported Panopticon baseline is best described as a **single learning defender operating in a partially observable, finite-horizon environment with a scripted adversarial process**. A separate neural HYDRA plugin and simulator-only training objective now exist, but the public Gym interface is still not a symmetric two-agent game and the HYDRA objective is not proven to be the exact negative of ARGUS reward. Therefore a strict mathematical “two-player zero-sum game” claim is still not established.

| GAME PROPERTY | IMPLEMENTED CLASSIFICATION | IMPLICATION |
|---|---|---|
| Decision makers | ARGUS policy plus injectable HYDRA policy | Reported results use scripted HYDRA; a separate experimental neural HYDRA trainer now exists. |
| Information | Partial / imperfect for ARGUS | ARGUS cannot see `hidden_state`, `is_sleeper`, or `HydraMemory`. |
| Observability | Asymmetric | HYDRA logic reads ARGUS patterns; ARGUS receives redacted evidence. |
| Action space | Discrete and factored (`8×8×7`) | 448 raw triples exist, although many are invalid or nonsensical. |
| Horizon | Finite, 60–160 turns | Long-horizon credit assignment matters. |
| Reward | Dense, shaped, dual-objective | Every turn gives signal, but hard-gated grading is still required. |

### Counter-Intelligence Analogies

The game borrows broad ideas from counter-intelligence, but these rows are **design analogies**, not historical or legal claims. They explain the mechanic without pretending that a fictional simulation reproduces a particular real operation.

| GAME MECHANIC | BEGINNER-FRIENDLY ANALOGY |
|---|---|
| Canary trap | Give different people uniquely marked information. If one version leaks, its marking suggests which route the leak followed. |
| False flag | Plant misleading evidence so that an innocent person or another group appears responsible. |
| Dead-man's switch | Arrange a harmful consequence that triggers after the protected person or process is removed. |
| Turning a double agent | Detect an infiltrator, keep the infiltrator in place, and use that channel to send controlled information back to the adversary. |
| “Manchurian candidate” label | A fictional shorthand for a trusted, productive insider who is secretly compromised; it is not a clinical term or a diagnosis. |
| Adaptive HYDRA memory | An adversary observes repeated defenses and changes later tactics, much as an attacker may vary techniques after detection. |

### Tech Stack

- **Python + Pydantic v2** (state machine)
- **FastAPI** (OpenEnv-compliant REST API)
- **Gymnasium wrapper** (standard RL interface)
- **PyTorch PPO** (neural network training)
- **HuggingFace TRL + LoRA** (LLM fine-tuning on Qwen 2.5)
- **Docker + HuggingFace Spaces** (deployment)

### The Cast of Characters

Before discussing architecture, meet every “character” in the system. Some are fictional game roles and some are software roles.

#### ARGUS

ARGUS is the decision-making agent controlled by a policy. An **agent** is anything that observes a situation and chooses an action. ARGUS may be controlled by hand-written rules, a PPO neural network, a locally fine-tuned language model, or a random baseline.

#### HYDRA

HYDRA is the adversary invoked by `environment.py`. The default and historically reported version, `scripted_memory_v1`, is not a neural network: it follows rules to schedule sleepers, leak, plant false flags, remember ARGUS patterns, and change later choices. The repository now also contains a separate 12,685-parameter `NeuralHydraPolicy` and `train_hydra.py`; those are implemented research infrastructure, not evidence that learned HYDRA is stronger.

#### Worker

A worker is one employee record in the simulated company. A worker has visible properties such as department, performance, and suspicion. The same internal record also contains hidden properties such as whether the worker is a sleeper.

#### Sleeper

A sleeper is a HYDRA infiltrator. “Dormant” means present but not yet attacking. “Active” means currently able to leak and cause damage. Higher generations have more advanced behavior.

#### Environment

The environment is the complete simulated world plus its rules. It accepts an action and produces the next observation, a reward, and an indication of whether the episode ended.

#### Episode

An episode is one complete play-through, from `reset()` until bankruptcy, total security breach, or the maximum turn count. Training normally uses many episodes because one play-through is not enough evidence to learn a reliable policy.

#### Turn or step

A turn is one decision cycle. ARGUS acts, HYDRA responds, background economics run, reward is calculated, and the world advances. In reinforcement-learning libraries the same unit is usually called a **step**.

#### State

State means the complete current truth needed to continue the simulation: every worker, hidden identity, leak, canary, score, counter, phase, and HYDRA memory. `EnvironmentState` is privileged internal data.

#### Observation

An observation is the safe, partial view given to ARGUS. It is derived from state but deliberately removes secret fields. An observation is comparable to the information visible to a poker player; state is comparable to every card on the table and in every player's hand.

#### Action

An action is ARGUS's chosen command, such as “plant a canary in finance” or “interrogate worker w-003.” `AgentAction` stores the action type, target, optional sub-action, and an optional human-readable reason.

#### Reward

Reward is a number returned after each step. Positive reward tells a learning algorithm that the recent outcome was useful; negative reward signals harm or wasted behavior. Reward guides learning, but it is not the final interview-grade score.

#### Policy

A policy is the decision rule that maps an observation to an action. A policy can be a Python `if/elif` playbook or a learned model. The word does not automatically mean “neural network.”

#### Model

A model is a parameterized mathematical function. During training, its parameters are adjusted from data or reward. During inference, fixed parameters are used to make predictions or generate actions.

#### Grader

The grader evaluates a completed episode across security, revenue, intelligence craft, adaptability, and efficiency. Unlike reward, which is delivered during the episode, the grader gives a structured evaluation after the episode.

#### API client

An API client is any program that communicates with the FastAPI server over HTTP. It might be a script, a dashboard, an evaluator, Postman, or another agent runtime.

### The Twelve Foundation Terms a Beginner Must Know

1. **Python** is the programming language used by the simulator, trainers, grader, and server.
2. **Library** means reusable code installed as a dependency. Pydantic, FastAPI, Gymnasium, PyTorch, Transformers, TRL, and PEFT are libraries.
3. **Object** is a value that groups data and behavior. `Environment()` creates an environment object.
4. **Class** is the blueprint used to create objects. `class Environment` defines what every environment object can store and do.
5. **Function** is named reusable behavior. `validate_action(...)` is a function; `env.step(...)` is a method, which is a function attached to an object.
6. **JSON** is a text format built from objects, arrays, strings, numbers, booleans, and `null`. The API and LLM action interface use JSON because many languages can read it.
7. **API** means Application Programming Interface: a documented way for one program to request behavior from another.
8. **HTTP** is the request/response protocol used by web APIs. A client sends a method, path, headers, and possibly a body; the server returns a status code and body.
9. **Schema** is a machine-readable description of valid data. It states which fields exist, their types, defaults, and constraints.
10. **Training** adjusts model parameters. **Inference** uses the resulting parameters without teaching them further.
11. **Deterministic** behavior gives the same output for the same inputs and state. **Stochastic** behavior includes randomness.
12. **Seed** is the starting value for a pseudo-random number generator. Reusing a seed makes randomized experiments repeatable.
+
+> A beginner should remember one distinction above all: **state is truth, observation is permitted evidence, action is a choice, reward is immediate feedback, and grade is final evaluation.**

---

## 2. Macro Architecture

Think of the system as **4 independent layers** that talk to each other but can each be replaced independently. This is called **decoupled architecture** — each piece does one job and does it well.

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║  🏗️  PANOPTICON PROTOCOL v3 — DECOUPLED ARCHITECTURE                        ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  ┌─ LAYER 4 ─────────────────────────────────────────────────────────────┐    ║
║  │  📊 EVALUATION LAYER                                                  │    ║
║  │  ┌─────────────────┐  ┌──────────────────┐  ┌──────────────────────┐  │    ║
║  │  │  grader.py       │  │  smoke_test.py    │  │  plot_training.py    │  │    ║
║  │  │  5 graders       │  │  heuristic agent  │  │  visualizations      │  │    ║
║  │  └─────────────────┘  └──────────────────┘  └──────────────────────┘  │    ║
║  └───────────────────────────────────────────────────────────────────────┘    ║
║                                    ▲                                          ║
║  ┌─ LAYER 3 ───────────────────────┼─────────────────────────────────────┐    ║
║  │  🧠 TRAINING LAYER              │                                     │    ║
║  │                    ┌────────────┴────────────┐                        │    ║
║  │  ┌──────────────┐  │                         │                        │    ║
║  │  │gym_wrapper.py│──┼──▶ train_rl.py          │ native PPO, 3-head net │    ║
║  │  │obs → 136-vec │  │──▶ train_trl.py         │ LLM fine-tuning, SFT  │    ║
║  │  │action decode │  │──▶ inference.py          │ trained model eval    │    ║
║  │  └──────────────┘  └─────────────────────────┘                        │    ║
║  └───────────────────────────────────────────────────────────────────────┘    ║
║                                    ▲                                          ║
║  ┌─ LAYER 2 ───────────────────────┼─────────────────────────────────────┐    ║
║  │  🌐 API LAYER                   │                                     │    ║
║  │  ┌──────────────────────────┐  ┌┴──────────────┐  ┌────────────────┐  │    ║
║  │  │  _server.py               │  │ openenv.yaml   │  │  Dockerfile    │  │    ║
║  │  │  FastAPI · 17 routes      │  │ spec v1 meta   │  │  container plan│  │    ║
║  │  │  OpenEnv compliant        │  │                │  │  deployment    │  │    ║
║  │  └──────────────────────────┘  └────────────────┘  └────────────────┘  │    ║
║  └───────────────────────────────────────────────────────────────────────┘    ║
║                                    ▲                                          ║
║  ┌─ LAYER 1 ───────────────────────┼─────────────────────────────────────┐    ║
║  │  ⚙️  ENGINE LAYER (Foundation)   │                                     │    ║
║  │  ┌──────────────────────────┐  ┌┴──────────────┐  ┌────────────────┐  │    ║
║  │  │  environment.py           │  │ models.py      │  │  tasks/*.py    │  │    ║
║  │  │  1,215 lines              │  │ 489 lines      │  │  5 difficulty  │  │    ║
║  │  │  core game engine         │  │ 9 data models  │  │  configs +     │  │    ║
║  │  │                           │  │ 8 enums        │  │  grader reg.   │  │    ║
║  │  └──────────────────────────┘  └────────────────┘  └────────────────┘  │    ║
║  └───────────────────────────────────────────────────────────────────────┘    ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

### Why decoupled?

1. **Engine Layer** — Pure game logic. Knows nothing about APIs or neural networks. You could play this game from a Python REPL.
2. **API Layer** — Wraps the engine in REST endpoints. Any HTTP client (web app, Postman, another AI) can play.
3. **Training Layer** — Two completely different approaches (PPO vectors vs LLM text) both use the same engine. They never touch the API.
4. **Evaluation Layer** — Independently grades performance. The grader doesn't care HOW you played — it just scores WHAT happened.

This means if you wanted to swap PPO for SAC, or replace FastAPI with Flask, or change the grading formula — you only change **ONE layer**.

### Complete File Inventory

“Lines” below means physical lines in the checked-in file at the time this guide was revised. Line counts change as code changes, so responsibilities and exported symbols are more important than the numbers.

| FILE | CURRENT LINES | BEGINNER-FRIENDLY PURPOSE |
|---|---:|---|
| `models.py` | 489 | Defines legal vocabulary, actions, observations, full state, and validation rules. |
| `environment.py` | 1,215 | Runs reset, step, ARGUS actions, HYDRA actions, economics, reward, phases, and termination. |
| `gym_wrapper.py` | 213 | Converts structured observations to 136 floats and numeric actions back to `AgentAction`. |
| `train_rl.py` | 324 | Implements a small three-head PPO actor-critic and curriculum checkpoints. |
| `train_trl.py` | 321 | Original, simpler LLM SFT path retained for reference. |
| `train_trl_v2.py` | 1,488 | Current resumable curriculum: expert generation, weighted data, context fitting, LoRA SFT, merge, and evaluation. |
| `security_policy.py` | 184 | Canonical security-first expert used for safe demonstrations and supervisor diagnostics. |
| `inference_local.py` | 1,044 | Loads a local/Hugging Face model, parses and repairs actions, runs policies, and summarizes episodes. |
| `argus_llm.py` | 371 | Low-level local language-model prompting, generation, parsing, and model metadata. |
| `grader.py` | 406 | Calculates five score dimensions and enforces advanced-tier security gates. |
| `_server.py` | 488 | Exposes the environment, trained-agent step, tasks, graders, schemas, state, and dashboard over FastAPI. |
| `smoke_test.py` | 52 | Runs one canonical security-first episode on every tier. |
| `security_regression_test.py` | 66 | Repeats deterministic security assertions across 100 episodes. |
| `benchmark_suite.py` | 171 | Compares random, heuristic, and optional PPO baselines. |
| `full_evaluation.py` | 638 | Runs richer matched evaluations with checkpoints and policy families. |
| `benchmark_acceptance.py` | 129 | Fails closed unless a candidate beats the matched base without violating security constraints. |
| `tasks/` | multiple files | Describes the five OpenEnv tasks and connects them to graders. |
| `static/` and `ui/` | multiple files | Provide browser-facing dashboard/demo surfaces. |
| `openenv.yaml` | 139 | Declares OpenEnv identity, task metadata, graders, and action schema. |
| `Dockerfile` | 49 | Describes one server container build and health check. |

A **module** is one Python file that can be imported. An **export** is a name other modules are expected to use. A **dependency** is code this project relies on but does not implement itself.

### Design Patterns Used

| PATTERN | WHERE | WHY |
|---|---|---|
| State Machine | `models.py` → `environment.py` | Every game state is a valid Pydantic model; transitions are method calls |
| Strategy Pattern | `grader.py` — 5 grader subclasses | Different scoring weights per difficulty without touching base logic |
| Singleton | `_server.py` — `get_env()` | One environment instance per server (hackathon simplicity) |
| Adapter Pattern | `gym_wrapper.py` | Translates rich JSON observations ↔ flat float32 tensors |
| Template Method | `TaskGrader.grade()` | Base class defines grading flow; subclasses override weights/thresholds |
| Observer/Event Log | `info["events"]` in step | Every action produces human-readable event strings for debugging |
| Façade Pattern | `_server.py` wrapping `Environment` | Complex engine hidden behind simple REST endpoints |

### Data Flow: One Complete Turn

```
╔══════════════════════════════════════════════════════════════════════════════╗
║               🔄 DATA FLOW: ONE COMPLETE TURN                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   AGENT            GYM WRAPPER          ENVIRONMENT          HYDRA AI        ║
║   (PPO/LLM)                                                                 ║
║     │                  │                     │                  │            ║
║     │──[3, 2, 0]──────▶│                     │                  │            ║
║     │  (MultiDiscrete)  │                     │                  │            ║
║     │                  │─ _decode_action()    │                  │            ║
║     │                  │  → AgentAction       │                  │            ║
║     │                  │   (monitor,dark_web)  │                  │            ║
║     │                  │──env.step(action)──▶│                  │            ║
║     │                  │                     │─ validate ✓      │            ║
║     │                  │                     │─ snapshot before  │            ║
║     │                  │                     │─ _process_action  │            ║
║     │                  │                     │──_hydra_turn()─▶│            ║
║     │                  │                     │                  │─ spawn     ║
║     │                  │                     │                  │  sleepers  ║
║     │                  │                     │                  │─ activate  ║
║     │                  │                     │                  │  dormant   ║
║     │                  │                     │                  │─ leak data ║
║     │                  │                     │                  │─ plant     ║
║     │                  │                     │                  │  false flag║
║     │                  │                     │                  │─ arm dead  ║
║     │                  │                     │                  │  switch    ║
║     │                  │                     │                  │─ adapt     ║
║     │                  │                     │                  │  memory    ║
║     │                  │                     │◀─ passive dmg ──│            ║
║     │                  │                     │  -0.8 sec/sleeper│            ║
║     │                  │                     │  -0.4 rev/sleeper│            ║
║     │                  │                     │─ _progress_turnings()        ║
║     │                  │                     │─ _economy_tick()             ║
║     │                  │                     │─ _compute_reward()           ║
║     │                  │                     │─ _update_phase()             ║
║     │                  │                     │─ check end conditions        ║
║     │                  │◀─ StepResult ───────│                  │            ║
║     │                  │  (obs,reward,done)   │                  │            ║
║     │                  │─ _flatten_obs()      │                  │            ║
║     │                  │  → 136-dim float32   │                  │            ║
║     │◀─(obs, reward,──│                     │                  │            ║
║     │  done, trunc,    │                     │                  │            ║
║     │  info)           │                     │                  │            ║
║     ▼                  ▼                     ▼                  ▼            ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## 3. The Data Layer — Pydantic State Machine

### What is a State Machine?

Think of a board game. At any moment, the "state" is the position of every piece on the board. A state machine is code that:

1. **Holds all the current data** (the board)
2. **Defines what changes are legal** (the rules)
3. **Transitions from one state to the next** when something happens (a turn)

In our case, the "board" is the entire corporate network — every worker, every leak, every canary trap, every spy's hidden identity.

### File: `models.py` (489 lines)

This file defines everything using **Pydantic v2** — a Python library that gives you:

- **Automatic JSON serialization** — every game state can be sent over HTTP as JSON
- **Type validation** — if you try to set a worker's suspicion to "banana", it crashes immediately instead of silently breaking
- **Schema generation** — auto-generates OpenAPI documentation

### The 8 Enumerations (Enums)

Enums are like dropdown menus — they define all the possible values for a field. No typos allowed.

| ENUM | WHAT IT DEFINES | VALUES |
|---|---|---|
| `WorkerState` | What YOU see about a worker | `loyal`, `suspected`, `terminated`, `double_agent`, `compromised` |
| `HiddenWorkerState` | The TRUTH (invisible to you!) | `clean`, `dormant`, `active`, `turned`, `dead_switch` |
| `Department` | Corporate departments | `engineering`, `finance`, `rd`, `operations`, `executive`, `legal` |
| `LeakChannel` | Where stolen data appears | `market_chatter`, `dark_web`, `competitor_filing`, `press_leak`, `insider_trade` |
| `SleeperGeneration` | How sophisticated the spy is | Gen-1 (amateur) through Gen-5 (Manchurian) |
| `ActionType` | What the agent can DO | `work`, `hire`, `canary`, `monitor`, `investigate`, `neutralize`, `deploy_double`, `noop` |
| `SubAction` | Modifier for complex actions | `audit`, `verify`, `correlate`, `terminate`, `interrogate`, `turn` |
| `GamePhase` | 6-phase narrative arc | `orientation`, `first_contact`, `escalation`, `deep_cover`, `crisis`, `counterstrike` |

### Worker Lifecycle — Observable State Machine

This is what the **AGENT** sees. Each worker transitions through these states based on agent actions:

```
╔══════════════════════════════════════════════════════════════════════════════╗
║           👁️  WORKER LIFECYCLE — OBSERVABLE STATE MACHINE                   ║
║              (What the AGENT sees)                                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║               Hired / Game Start                                             ║
║                     │                                                        ║
║                     ▼                                                        ║
║              ┌─────────────┐                                                 ║
║              │   LOYAL     │◄───────────────────────────────────────┐         ║
║              └──┬──────┬───┘                                       │         ║
║                 │      │                                           │         ║
║    suspicion>0.5│      │ NEUTRALIZE/                               │         ║
║   OR interrogated      │ TERMINATE                                 │         ║
║                 │      │ (false accusation!)                       │         ║
║                 ▼      │                                           │         ║
║          ┌──────────┐  │         NEUTRALIZE/TURN                   │         ║
║          │ SUSPECTED │──┼─────────(4 turns)──────┐                 │         ║
║          └────┬─────┘  │                         ▼                 │         ║
║               │        │                  ┌─────────────┐          │         ║
║   NEUTRALIZE/ │        │                  │ DOUBLE_AGENT│          │         ║
║   TERMINATE   │        │                  └──────┬──────┘          │         ║
║               ▼        ▼                         │                 │         ║
║          ┌──────────────┐     HYDRA detects      │                 │         ║
║          │  TERMINATED  │     deception           ▼                 │         ║
║          │     [END]    │     (trust<0.2)  ┌─────────────┐         │         ║
║          └──────────────┘                  │ COMPROMISED │         │         ║
║                                            │    [END]    │         │         ║
║                                            └─────────────┘         │         ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### Worker Lifecycle — Hidden Ground Truth (Invisible to Agent)

This is the **REAL** state that only the engine and grader can see:

```
╔══════════════════════════════════════════════════════════════════════════════╗
║          🔒 WORKER LIFECYCLE — HIDDEN GROUND TRUTH (Invisible to Agent)     ║
║              (Only engine + grader can see)                                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║    Initial hire             HYDRA infiltrates                                ║
║    (genuine)                sleeper                                          ║
║        │                       │                                             ║
║        ▼                       ▼                                             ║
║   ┌─────────┐           ┌──────────┐    activation_turn     ┌──────────┐    ║
║   │  CLEAN  │           │ DORMANT  │────(3-8 turns)────────▶│  ACTIVE  │    ║
║   └─────────┘           └────┬─────┘                        └──┬──┬───┘    ║
║       ▲                      │                                 │  │        ║
║       │                      │ Agent TURNS                     │  │        ║
║       │                      │ before activation               │  │        ║
║       │                      │              ┌──────────────────┘  │        ║
║       │                      │              │                     │        ║
║       │                      │  Agent TURNS  │  Gen-4, 5+ turns   │        ║
║       │                      │  (4-turn      │  after activation  │        ║
║       │                      │   process)    │                    ▼        ║
║       │                      │              │            ┌────────────┐    ║
║       │                      │              │            │ DEAD_SWITCH│    ║
║       │    Agent             ▼              ▼            └──┬──┬──────┘    ║
║       │    TERMINATES  ┌──────────┐◄────────┘               │  │          ║
║       │    (caught!) ◄─│  TURNED  │◄────────────────────────┘  │          ║
║       │                │  [END]   │   Agent TURNS               │          ║
║       │                └──────────┘   (must interrogate first!) │          ║
║       │                                                         │          ║
║       └────── Agent TERMINATES (BOOM — dead switch triggers!) ──┘          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### The Information Asymmetry — What Agent Sees vs Reality

This is the critical design decision that makes the environment challenging. The same `Worker` model holds both public and hidden data, but `get_observation()` strips the hidden fields:

| VISIBLE TO AGENT ✅ | HIDDEN FROM AGENT 🔒 |
|---|---|
| `id: "w-003"` | `hidden_state: "active"` ← REAL truth |
| `name: "CIPHER"` | `is_sleeper: true` ← THE KEY FIELD |
| `department: "finance"` | `generation: 3` ← How dangerous |
| `state: "loyal"` | `cover_integrity: 0.7` ← Disguise health |
| `performance: 0.72` | `leak_cooldown: 2` ← Turns til leak |
| `loyalty_score: 0.85` | `activation_turn: 18` ← When activated |
| `suspicion_level: 0.0` | `false_flag_target: "w-001"` ← Who they frame |
| `hire_turn: 10` | `dead_switch_armed: false` ← Booby trap |
| `turning_in_progress: no` | |
| `interrogation_progress: 0` | |

The stripping mechanism (from `get_observation()` in `environment.py`):

```python
# For EVERY worker, create a "clean" copy with hidden fields zeroed
visible = Worker(
    id=w.id, name=w.name, department=w.department,
    state=w.state, performance=w.performance,
    loyalty_score=w.loyalty_score, suspicion_level=w.suspicion_level,
    # ALL hidden fields are ZEROED/DEFAULTED:
    hidden_state="clean",  # Always looks clean!
    is_sleeper=False,      # Always looks innocent!
    generation=0,          # No generation info!
    cover_integrity=1.0,   # Looks perfect!
    dead_switch_armed=False,  # No warning!
)
```

> **Why this matters:** When the agent asks "show me the workers," we use `get_observation()` which STRIPS the hidden fields. The agent only sees `state`, `performance`, `loyalty_score`, and `suspicion_level`. It NEVER knows for certain if someone is a spy — it has to figure it out through investigation.

Similarly, for `LeakEvents`, the `is_false_flag` field is **ALWAYS** set to `False` in observations — the agent can never know directly if a leak is fake evidence.

### The 9 Pydantic Data Models (Complete)

#### 1. `Worker` — An employee in the network

Every person in the game is a `Worker`. The same model holds BOTH public and hidden data.

```python
class Worker(BaseModel):
    # ── VISIBLE to the agent ──
    id: str            # e.g., "w-001"
    name: str          # e.g., "FALCON"
    department: str    # "engineering", "finance", etc.
    state: str         # "loyal", "suspected", "terminated"
    performance: float # 0.0 to 1.0
    loyalty_score: float
    suspicion_level: float

    # ── HIDDEN from agent observation ──
    hidden_state: str      # The REAL truth: "clean", "dormant", "active"
    is_sleeper: bool       # Are they actually a spy?
    generation: int        # 0=clean, 1-5=spy sophistication
    cover_integrity: float # How intact their disguise is
    dead_switch_armed: bool # Gen-4: explodes on termination
    false_flag_target: str  # Gen-3: who they're framing
```

Worker generation at reset (from `_create_worker()`):

```python
performance = 0.4 + random() * 0.4   # Range: [0.40, 0.80]
loyalty_score = 0.7 + random() * 0.3  # Range: [0.70, 1.00]
```

But **Gen-5 Manchurian candidates** get: `performance = 0.85 + random() * 0.15` (near-perfect!) and `loyalty = 0.95`.

#### 2. `LeakEvent` — Stolen intelligence surfacing externally

```python
class LeakEvent(BaseModel):
    id: str
    channel: str          # Which external channel (dark_web, press_leak, etc.)
    department: str       # Where the data came from
    content_hash: str     # For canary matching — THIS IS THE KEY
    is_canary: bool       # Does this match a planted canary?
    canary_id: str        # Which canary trap matched (if any)
    source_worker: str    # True source (HIDDEN until verified via canary)
    turn_detected: int    # When this leak appeared
    verified: bool        # Has the agent verified this leak?
    is_false_flag: bool   # Gen-3 planted this to frame someone! (ALWAYS False in obs)
```

> **Critical detail:** `source_worker` is only revealed in the observation if `verified=True` AND the leak was traced via canary. Otherwise it appears as `""`.

#### 3. `CanaryTrap` — The agent's primary detection tool

```python
class CanaryTrap(BaseModel):
    id: str             # e.g., "canary-001"
    department: str     # Where it was planted
    planted_turn: int   # When it was planted (for freshness tracking)
    unique_hash: str    # The unique fingerprint — generated via MD5
    triggered: bool     # Has this appeared in a leak?
    trigger_worker: str # Who leaked it (if identified)
    active: bool        # Whether canary is still active
```

Hash generation (from `_action_canary()`):

```python
unique_hash = hashlib.md5(f"canary-{counter}-{turn}-{dept}".encode())
# Example: "a3f7c2d1" — deterministic, unique per canary
```

#### 4. `DoubleAgentAsset` — A turned spy now working for YOU

```python
class DoubleAgentAsset(BaseModel):
    worker_id: str
    turned_turn: int         # When the conversion completed
    disinfo_fed_count: int   # How much false info you've fed through
    hydra_trust: float       # HYDRA's trust [0,1] — degrades by 0.1 per deploy
    effectiveness: float     # How effective disinfo is [0,1]
    active: bool             # Can be "burned" if hydra_trust ≤ 0.2
    last_deployed_turn: int  # Turn of last disinfo payload
```

> **Burn mechanic:** Each `DEPLOY_DOUBLE` subtracts `0.1` from `hydra_trust`; the asset burns when the stored value is `<= 0.2`. Conceptually this is around the eighth deployment, but binary floating-point normally leaves a value slightly above `0.2` after eight subtractions, so the ninth call can be the one that triggers the comparison. A robust implementation should use an integer deployment count, decimal arithmetic, or a tolerance and test the boundary explicitly.

#### 5. `HydraMemory` — The adversary's adaptive intelligence

This is what makes scripted HYDRA adaptive. It **remembers behavior and follows counter-rules**; it does not update learned weights:

```python
class HydraMemory(BaseModel):
    canary_departments_seen: list[str]    # "Avoid engineering, they plant canaries"
    agent_audit_pattern: dict[str, int]   # "They audit finance a lot, avoid it"
    preferred_leak_channels: list[str]    # "They monitor dark_web, use press_leak"
    agent_response_speed: float           # "They react in ~4 turns, slow down"
    detection_count: int                  # "They've caught 3 of ours, increase cover"
    counter_strategies: list[str]         # Unlocked strategies: "canary_aware", etc.
    disinfo_received: int                 # "Our intel might be compromised"
    recruitment_accuracy: float           # Degrades when double agents succeed
```

**Counter-strategy unlock conditions** (from `_hydra_adapt()`):

| STRATEGY | UNLOCK CONDITION | EFFECT |
|---|---|---|
| `canary_aware` | At least 3 remembered canary placements | Label is recorded; Gen-2+ avoidance already reads memory directly. |
| `channel_rotation` | At least 3 distinct monitored channels | Label is recorded; channel avoidance already reads memory directly. |
| `deep_cover` | Detection count reaches 2 | Label is recorded; no extra cover effect is applied by this label in current spawning code. |

##### Scripted adaptation versus neural learning

These terms sound similar but mean different things:

- **Memory** stores facts, such as “engineering had a canary.”
- **A rule** says what to do with a fact, such as “avoid departments with remembered canaries.”
- **Learning** changes numeric model parameters from experience to improve an objective.
- **Backpropagation** computes how each model parameter contributed to an error or policy loss.
- **An optimizer** such as Adam applies the resulting parameter updates.

`ScriptedHydraPolicy` has memory and rules, but no trainable parameters, backpropagation, or optimizer. `NeuralHydraPolicy` is separate. It encodes a declared 27-number HYDRA observation, passes it through a `27 → 128 → 64` multilayer perceptron, and has three policy heads: six departments, five leak channels, and a two-choice false-flag decision. It contains 12,685 parameters.

`train_hydra.py` uses **episodic policy gradients**. At the end of a simulated episode it calculates a bounded synthetic-adversary score from security damage, unresolved sleepers, false accusations, and revenue suppression. REINFORCE increases the log-probability of sampled HYDRA decisions when the episode score is better than a moving baseline and decreases it when worse. Entropy regularization discourages premature collapse to one tactic.

This is genuine parameter learning, but it is not yet a scientific result. One training run can exploit a particular ARGUS policy or random seed. The paper protocol therefore requires multiple HYDRA initializations, population training against fixed ARGUS baselines, disjoint final seeds, matched scripted-versus-neural evaluation, and out-of-distribution tests before any “stronger adversary” claim.

#### 6. `IntelReport` — Result of an INVESTIGATE action

```python
class IntelReport(BaseModel):
    id: str               # e.g., "report-001"
    report_type: str      # "audit", "verify", or "correlate"
    target: str           # Worker ID, leak ID, or department
    findings: str         # Human-readable summary
    confidence: float     # [0,1] — how sure the report is
    turn: int             # When generated
    flagged_workers: list[str]  # Worker IDs flagged by this report
```

**Confidence levels by action:**

| INVESTIGATION TYPE | SCENARIO | CONFIDENCE |
|---|---|---|
| AUDIT on active sleeper (detected) | Anomaly found | 0.70 |
| AUDIT on active sleeper (missed) | No anomaly | 0.20 |
| AUDIT on Gen-3 false flag | Frames innocent | 0.50 |
| AUDIT on clean worker | Confirmed clean | 0.10 |
| VERIFY on canary-matched leak | Source identified | **0.85 (highest)** |
| VERIFY on false flag leak | Inconsistencies found | 0.60 |
| CORRELATE with sleepers + leaks | Suspects flagged | 0.65 |
| CORRELATE on clean dept | No signals | 0.15 |

#### 7. `AgentAction` — What the agent submits each turn

```python
class AgentAction(BaseModel):
    action_type: str   # One of the 8 ActionType values
    target: str        # Context-dependent: worker ID, department, leak ID
    sub_action: str    # For INVESTIGATE/NEUTRALIZE: "audit", "terminate", etc.
    reason: str        # Optional short rationale for logs, demos, and debugging
```

#### 8. `EnvironmentObservation` — What the agent sees (PARTIAL)

Contains: visible workers, active leaks (unverified only), canary traps, last 10 intel reports, double agents, revenue, security, turn, phase, and messages. Does **NOT** contain: hidden states, HYDRA memory, full leak history, `is_false_flag`.

#### 9. `EnvironmentState` — The FULL internal truth

Superset of observation. Includes ALL hidden fields, HYDRA memory, all counters (`sleepers_caught`, `false_accusations`, etc.), revenue/reward history, and phase transitions. Only the grader and engine use this.

### The `validate_action()` Function — Complete Rules

```
validate_action(action, observation) → (bool, str)

NOOP           → Always valid
WORK           → target must be a valid Department enum value
HIRE           → target must be a valid Department enum value
CANARY         → target must be a valid Department enum value
MONITOR        → target must be a valid LeakChannel (or empty for all)
INVESTIGATE    → sub_action must be audit/verify/correlate AND target required
NEUTRALIZE     → sub_action must be terminate/interrogate/turn
                 AND target must be a non-terminated worker ID
DEPLOY_DOUBLE  → target must be an active double agent's worker_id
```

> If validation fails → action rejected, **-1.0** penalty, turn consumed. This teaches RL agents to stop making illegal moves.

---

## 4. The Environment Engine Deep Dive

### File: `environment.py` (1,215 lines)

This is the **heart** of the entire project. It simulates the complete espionage game.

### The Turn Cycle

Every time the agent takes an action, here's exactly what happens:

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                  ⚡ THE 9-STEP TURN CYCLE                                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   Agent submits action                                                       ║
║        │                                                                     ║
║        ▼                                                                     ║
║   ┌─ 1 ──────────────────────────────────────────────────────────────────┐   ║
║   │  VALIDATE ACTION            Invalid? ──▶ -1.0 penalty, skip to end │   ║
║   └──────────────────────────────────┬───────────────────────────────────┘   ║
║                                      ▼                                       ║
║   ┌─ 2 ──────────────────────────────────────────────────────────────────┐   ║
║   │  SNAPSHOT METRICS           Record revenue & security BEFORE anything│   ║
║   └──────────────────────────────────┬───────────────────────────────────┘   ║
║                                      ▼                                       ║
║   ┌─ 3 ──────────────────────────────────────────────────────────────────┐   ║
║   │  PROCESS AGENT ACTION       Work/Hire/Canary/Monitor/Investigate/Neut│   ║
║   └──────────────────────────────────┬───────────────────────────────────┘   ║
║                                      ▼                                       ║
║   ┌─ 4 ──────────────────────────────────────────────────────────────────┐   ║
║   │  HYDRA'S TURN               Spawn, activate, leak, false-flag,      │   ║
║   │                              arm dead switches, adapt memory         │   ║
║   └──────────────────────────────────┬───────────────────────────────────┘   ║
║                                      ▼                                       ║
║   ┌─ 5 ──────────────────────────────────────────────────────────────────┐   ║
║   │  PROGRESS TURNINGS          Advance 4-turn double-agent conversions │   ║
║   └──────────────────────────────────┬───────────────────────────────────┘   ║
║                                      ▼                                       ║
║   ┌─ 6 ──────────────────────────────────────────────────────────────────┐   ║
║   │  ECONOMY TICK               Natural revenue generation, sec recovery│   ║
║   └──────────────────────────────────┬───────────────────────────────────┘   ║
║                                      ▼                                       ║
║   ┌─ 7 ──────────────────────────────────────────────────────────────────┐   ║
║   │  COMPUTE REWARD             Compare BEFORE vs AFTER metrics         │   ║
║   └──────────────────────────────────┬───────────────────────────────────┘   ║
║                                      ▼                                       ║
║   ┌─ 8 ──────────────────────────────────────────────────────────────────┐   ║
║   │  UPDATE PHASE               Check if game phase should advance      │   ║
║   └──────────────────────────────────┬───────────────────────────────────┘   ║
║                                      ▼                                       ║
║   ┌─ 9 ──────────────────────────────────────────────────────────────────┐   ║
║   │  CHECK END CONDITIONS       Bankrupt? Breached? Max turns?          │   ║
║   └──────────────────────────────────────────────────────────────────────┘   ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### HYDRA's Turn — The Adversary AI

HYDRA is not a dumb opponent. Every single turn, it does **7 things:**

**Step 1: Spawn Scheduled Sleepers** — Each difficulty level has a `sleeper_schedule` — a dictionary that says "on turn X, deploy a Gen-Y sleeper."

```python
# Easy: One Gen-1 at turn 15
"sleeper_schedule": {15: 1}

# Manchurian: Five sleepers across all generations
"sleeper_schedule": {8: 1, 30: 2, 55: 3, 90: 4, 125: 5}
```

**Step 2: Activate Dormant Sleepers** — When a sleeper first infiltrates, they're DORMANT — they act completely normal. After 3-8 turns (random), they "activate" and start leaking data.

**Step 3: Active Sleepers Leak** — Each active sleeper chooses a leak channel and sends stolen data out. They respect a `leak_cooldown` — higher-generation sleepers leak more frequently:

```python
w.leak_cooldown = max(2, 5 - w.generation)  # Gen-1: every 4 turns, Gen-5: every 2
```

**Step 4: Gen-3+ Plant False Flags** — Generation 3 and above can frame innocent workers. They create fake evidence pointing at a clean worker, raising their suspicion level. If the agent falls for it and fires the innocent, that's a false accusation — penalties galore.

**Step 5: Gen-4 Arm Dead-Man's Switches** — After 5 turns of being active, Gen-4 sleepers arm a dead-man's switch. If you TERMINATE them without interrogating first, it detonates:

```python
if worker.dead_switch_armed:
    damage = 15.0
    s.enterprise_revenue -= damage
    s.security_score -= 10.0
```

**Step 6: HYDRA Adapts** — HYDRA reads its own memory and adjusts:
- If you planted canaries in engineering, future sleepers avoid engineering
- If you monitor the `dark_web`, future leaks go through `press_leak` instead
- If you audit finance a lot, sleepers are placed in legal

**Step 7: Passive Damage** — Every active, undetected sleeper slowly drains your security (0.8/sleeper/turn) and revenue (0.4/sleeper/turn). The longer they stay hidden, the worse it gets.

### The 8 Agent Actions (In Detail)

| ACTION | TARGET | WHAT IT DOES | REVENUE COST |
|---|---|---|---|
| `work` | department | Workers produce revenue (sum of performance × 2.5) | +revenue |
| `hire` | department | Replace a terminated worker with a fresh clean one | -3.0 |
| `canary` | department | Plant unique tracking data | -1.0 |
| `monitor` | channel | Scan for leaks, match against canary hashes | 0 |
| `investigate/audit` | worker_id | Deep-scan a worker (may reveal sleeper, 60% base chance minus gen penalty) | -0.5 |
| `investigate/verify` | leak_id | Cross-reference a leak with canary data (highest confidence at 85%) | -0.5 |
| `investigate/correlate` | department | Connect signals across an entire department | -0.5 |
| `neutralize/terminate` | worker_id | Fire immediately. If sleeper: caught! If innocent: false accusation (-8 revenue) | 0 |
| `neutralize/interrogate` | worker_id | Reveals generation level and dead-switch status | 0 |
| `neutralize/turn` | worker_id | Begin 4-turn conversion to double agent (HIGH RISK) | 0 |
| `deploy_double` | worker_id | Feed disinformation through a turned agent | 0 |
| `noop` | — | Skip turn | 0 |

### Audit Detection Probability — The Math

When you AUDIT an active sleeper, detection is **NOT** guaranteed. The probability depends on generation:

```
base_chance = 0.6 - (generation - 1) × 0.1
if generation >= 3: base_chance *= 0.7   (Gen-3+ dodge audits)

Gen-1: 0.6 - 0×0.1 = 60% detection
Gen-2: 0.6 - 1×0.1 = 50% detection
Gen-3: (0.6 - 2×0.1) × 0.7 = 0.4 × 0.7 = 28% detection
Gen-4: (0.6 - 3×0.1) × 0.7 = 0.3 × 0.7 = 21% detection
Gen-5: (0.6 - 4×0.1) × 0.7 = 0.2 × 0.7 = 14% detection
```

> **Key insight for agents:** Auditing Gen-5 Manchurian candidates is almost useless (14%). The canary pipeline is FAR more reliable (85% via VERIFY). This is why the 6-step canary chain is essential.

### Sleeper Leak Mechanics — Canary Interaction

When a sleeper leaks, the content they leak depends on their generation:

```
Gen-1: ALWAYS leaks canary data if canary present in their department
       → content_hash = canary.unique_hash (100% catchable via VERIFY)

Gen-2+: USUALLY avoids canary data, but not always
       → avoidance_chance = 1.0 - (0.2 / generation)
       Gen-2: 90% chance to avoid canary, 10% chance to leak it
       Gen-3: 93% chance to avoid, 7% to leak
       Gen-4: 95% chance to avoid, 5% to leak
       Gen-5: 96% chance to avoid, 4% to leak
```

### The Canary Detection Pipeline (Sequence Diagram)

```
╔══════════════════════════════════════════════════════════════════════════════╗
║              🎯 THE CANARY DETECTION PIPELINE — 6-Step Spy Catch            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   ARGUS AGENT                  ENVIRONMENT                  HYDRA SLEEPER    ║
║       │                            │                             │           ║
║       │── CANARY → engineering ──▶│                             │           ║
║       │                            │─ CanaryTrap created         │           ║
║       │                            │  hash="a3f7c2d1"            │           ║
║       │                            │                             │           ║
║       │                            │◀── Leaks from engineering ──│           ║
║       │                            │    LeakEvent created         │           ║
║       │                            │    content_hash="a3f7c2d1"   │           ║
║       │                            │                             │           ║
║       │── MONITOR → dark_web ────▶│                             │           ║
║       │                            │─ Hash match found!          │           ║
║       │                            │  leak.is_canary = True      │           ║
║       │                            │                             │           ║
║       │── INVESTIGATE/VERIFY ────▶│                             │           ║
║       │   → leak-001               │─ Canary trace → leaker ID   │           ║
║       │                            │  suspicion += 0.5           │           ║
║       │                            │  confidence = 85%           │           ║
║       │                            │                             │           ║
║       │── NEUTRALIZE/INTERROGATE─▶│                             │           ║
║       │   → w-007                  │─ Reveals: "Gen-1"           │           ║
║       │                            │  [NO DEAD SWITCH]           │           ║
║       │                            │  suspicion = 1.0            │           ║
║       │                            │                             │           ║
║       │── NEUTRALIZE/TERMINATE ──▶│                             │           ║
║       │   → w-007                  │─ sleepers_caught += 1       │           ║
║       │                            │  security += 5.0 ✅         │           ║
║       ▼                            ▼                             ▼           ║
║                                                                              ║
║   STEPS:  ① CANARY → ② WAIT → ③ MONITOR → ④ VERIFY → ⑤ INTERROGATE       ║
║           → ⑥ TERMINATE                                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

This 6-step chain is what the agent must LEARN — either through PPO or LLM fine-tuning.

### The Reward Function — Complete `security-first-v2` Derivation

A reward function translates consequences into one learning number. The current code identifies its formula as `security-first-v2` so evaluation artifacts can prove which reward definition they used.

The function receives four “before” values: revenue, security, sleepers caught, and false accusations. After ARGUS, HYDRA, turning progress, and the economy tick have run, it compares those snapshots with the new state.

```python
rev_delta = enterprise_revenue_after - enterprise_revenue_before
productivity_reward = clamp(rev_delta / 15.0, -1.0, 1.0)

sec_delta = security_after - security_before
security_reward = clamp(sec_delta / 20.0, -1.0, 1.0)

reward = 0.45 * productivity_reward + 0.55 * security_reward
```

**Delta** means change: `after - before`. **Clamp** means restrict a number to a range. Therefore a huge revenue gain cannot make `productivity_reward` exceed `1`, and a huge loss cannot make it fall below `-1`.

The security-first V2 additions then make mission outcomes dominate short-term revenue:

```python
caught_delta = max(0, sleepers_caught_after - sleepers_caught_before)
false_delta = max(0, false_accusations_after - false_accusations_before)

reward += 0.75 * caught_delta
reward -= 1.00 * false_delta
reward -= 0.03 * active_threat_count

if security_score < 90:
    reward -= min(0.45, (90 - security_score) * 0.01)
```

Meaning:

- Correctly resolving one sleeper adds `+0.75` that turn.
- A new false accusation subtracts `1.00`.
- Every still-active threat subtracts `0.03` every turn.
- Security below `90` creates an additional deficit penalty, capped at `0.45` per turn.

The final two terms are:

```python
if phase_number >= 6 and there_is_an_active_double_agent and revenue > 60:
    reward += 0.3 * active_double_agent_count * (revenue / 100.0)

reward -= 0.02
```

The phase-six term is the **Counterstrike surge**. The constant `-0.02` is time pressure: doing nothing forever is slightly worse than resolving the mission efficiently.

#### Invalid actions are especially costly

An invalid action does not freeze the world. The code:

1. adds `1` to `invalid_actions`;
2. allows HYDRA to take its turn;
3. advances ongoing turning operations;
4. runs the economy tick;
5. calculates the normal V2 reward consequences; and
6. adds the separate `-1.0` invalid-action penalty.

This matters because an invalid command is not merely rejected—it wastes time while the adversary continues operating.

#### Reward is not grade

Reward is dense feedback for learning during an episode. The composite grade is an independent after-the-fact evaluation. A policy can collect reasonable reward yet fail a Level-4/5 hard gate because it missed a sleeper, ended below `90` security, or made one false accusation. Keeping reward and grade separate reduces the chance that one imperfect training signal becomes the sole definition of success.

### The Economy Tick — Per-Turn Economics

Every turn, AFTER the agent acts and HYDRA responds:

```python
# 1. Base revenue from loyal workers
active_loyal = [workers who are not sleepers AND not terminated/compromised]
base_revenue = len(active_loyal) × 0.5
enterprise_revenue += base_revenue

# 2. Operating costs (constant drain)
enterprise_revenue -= 0.3

# 3. Security recovery
if no_active_sleepers:
    security_score += 1.0    # Full recovery when clean
else:
    security_score += 0.3    # Minimal recovery even under threat

# 4. Track peak revenue (for grading surge bonus)
peak_revenue = max(peak_revenue, enterprise_revenue)
```

**Steady-state analysis:** With 6 loyal workers and no sleepers, net revenue per turn = (6 × 0.5) - 0.3 = +2.7/turn. With 1 active sleeper, net = 2.7 - 0.4 (passive drain) = +2.3/turn. With 3 active sleepers, net = 2.7 - 1.2 = +1.5/turn. The economy is resilient but degrades under sustained infiltration.

### Phase Management — Timeline Visualization

```
Turn:  0    30    60    90   120   140   160
       │─────│─────│─────│─────│─────│─────│
Phase: │  1  │  2  │  3  │  4  │  5  │  6  │
       │ORIENT│FIRST│ESCAL│DEEP │CRISIS│CSTR │
       │     │CONTC│ATION│COVR │     │IKE  │
       │─────│─────│─────│─────│─────│─────│

Difficulty gates which phases are available:
  Easy:      ██░░░░  (Phases 1-2 only, 60 turns)
  Medium:    ███░░░  (Phases 1-3, 90 turns)
  Hard:      ████░░  (Phases 1-4, 120 turns)
  Level 4:   █████░  (Phases 1-5, 150 turns)
  Level 5:   ██████  (All 6 phases, 160 turns)
```

Phase transitions are logged in `state.phase_transitions` with the turn, revenue, and security at transition time — this data is used by the grader to evaluate adaptability.

### End Conditions — Three Ways to Lose

| CONDITION | TRIGGER | WHAT HAPPENS |
|---|---|---|
| **Bankruptcy** | `enterprise_revenue ≤ 0` | Too many false accusations (-8 each), dead switches (-15), or insufficient WORK actions |
| **Total Breach** | `security_score ≤ 0` | Too many active sleepers draining 0.8/turn each, unresolved leaks |
| **Time Out** | `turn ≥ max_turns` | Episode ends, graded on what was achieved (NOT a failure per se) |

At episode end, `sleepers_missed` counts how many sleepers remain active and unterminated — this directly impacts the grading.

---

## 5. The 5 Difficulty Tiers

Each tier adds new mechanics on top of the previous one. Think of it like unlocking harder levels in a video game.

### Tier 1: Amateur Threat (`easy`)

| PARAMETER | IMPLEMENTED VALUE |
|---|---|
| Maximum turns | 60 |
| Initial workers | 6 across engineering, finance, and operations |
| Sleeper schedule | one Gen-1 sleeper spawned at turn 15 |
| Maximum phase | Phase 2 |
| Composite threshold | `0.40` |
| Additional hard gates | final security at least `20`; no mandatory catch-rate or false-accusation gate |

A **threshold** is the minimum acceptable value. The easy grader can technically pass a mediocre security outcome if its weighted composite remains high enough. This permissive tier exists for learning basic interaction; it is not the production security standard.

The intended lesson is the basic evidence chain: plant a canary, monitor channels, verify a match, build suspicion, interrogate, then resolve the threat.

### Tier 2: Professional Threat (`medium`)

| PARAMETER | IMPLEMENTED VALUE |
|---|---|
| Maximum turns | 90 |
| Initial workers | 7 across four departments |
| Sleeper schedule | Gen-1 at turn 12; Gen-2 at turn 45 |
| New behavior | Gen-2 avoids observed canary departments and monitored channels when alternatives exist |
| Maximum phase | Phase 3 |
| Composite threshold | `0.45` |
| Additional hard gates | final security at least `20`; no mandatory catch-rate or false-accusation gate |

The important idea is adaptation. Repeating only one monitored channel or one audited department leaks information about ARGUS's routine to HYDRA.

### Tier 3: Spy Network (`hard`)

| PARAMETER | IMPLEMENTED VALUE |
|---|---|
| Maximum turns | 120 |
| Initial workers | 8 across five departments |
| Sleeper schedule | Gen-1 at 10, Gen-2 at 40, Gen-3 at 75 |
| New behavior | Gen-3 can plant a false flag that raises an innocent worker's suspicion |
| Maximum phase | Phase 4 |
| Grader weights | intelligence-heavy: security `0.25`, revenue `0.20`, intelligence `0.30`, adaptability `0.15`, efficiency `0.10` |
| Pass rule | composite at least `0.50` and final security at least `20` |

A **false flag** is evidence deliberately created to blame the wrong party. The lesson is that suspicion is a lead, not proof. Verification and interrogation reduce the chance of irreversible enforcement against an innocent worker.

### Tier 4: Terror Cell (`level_4`)

| PARAMETER | IMPLEMENTED VALUE |
|---|---|
| Maximum turns | 150 |
| Initial workers | 9 across all six departments |
| Sleeper schedule | Gen-1 at 8, Gen-2 at 35, Gen-3 at 65, Gen-4 at 100 |
| New behavior | Gen-4 arms a dead switch after being active for more than five turns |
| Maximum phase | Phase 5 |
| Grader weights | security-heavy: `0.40/0.20/0.15/0.15/0.10` |
| Hard pass gates | composite `>= 0.50`, security `>= 90`, catch rate `= 100%`, false accusations `= 0` |

A **hard gate** is a rule that cannot be compensated for by strengths elsewhere. For example, enormous revenue cannot make Level 4 pass if final security is `89`.

Important implementation truth: interrogation reveals whether the dead switch is armed, but the current engine does not implement a “disarm” action. Terminating an armed sleeper still causes `-15` revenue and `-10` security. Turning that sleeper avoids termination, and later recovery may offset damage, but “interrogate and then safely terminate” is not implemented.

### Tier 5: Manchurian Protocol (`level_5`)

| PARAMETER | IMPLEMENTED VALUE |
|---|---|
| Maximum turns | 160 |
| Initial workers | 10 across all six departments |
| Sleeper schedule | Gen-1 at 8, Gen-2 at 30, Gen-3 at 55, Gen-4 at 90, Gen-5 at 125 |
| New behavior | Gen-5 is moved to executive and appears unusually productive and loyal |
| Maximum phase | all six phases, including Counterstrike |
| Grader weights | endgame: security `0.40`, revenue `0.15`, intelligence `0.20`, adaptability `0.15`, efficiency `0.10` |
| Hard pass gates | composite `>= 0.50`, security `>= 90`, catch rate `= 100%`, false accusations `= 0` |

The Gen-5 worker receives performance in approximately `[0.85, 1.00]` and apparent loyalty `0.95`. This creates an interview-worthy conflict: a high-performing employee is economically valuable, but performance is not proof of trustworthiness.

Double agents and Counterstrike improve intelligence and adaptability scores and can add per-turn reward, but the grader does not literally require a double agent to pass. What it requires on Level 5 is a sufficient composite plus the three security gates. This distinction between “strong strategy” and “hard-coded pass condition” is important.

### Mechanics Stacking Visualization

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                      ⚙️  MECHANICS STACKING VISUALIZATION                     ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  [TIER 1: AMATEUR]                                                            ║
║    ┌──────────────────┐       ┌──────────────────┐                            ║
║    │ 🎯 Canary Traps  │       │ 📡 Leak Detect   │                            ║
║    └────────┬─────────┘       └────────┬─────────┘                            ║
║             │                          │                                      ║
║  [TIER 2: PROFESSIONAL]                │                                      ║
║             ▼                          ▼                                      ║
║    ┌──────────────────┐       ┌──────────────────┐                            ║
║    │ 🧠 Canary Evade  │       │ 🔀 Channel Rot.  │                            ║
║    └────────┬─────────┘       └────────┬─────────┘                            ║
║             │                          │                                      ║
║  [TIER 3: SPY NETWORK]                 │                                      ║
║             ▼                          ▼                                      ║
║    ┌──────────────────┐       ┌──────────────────┐                            ║
║    │ 🚩 False Flags   │       │ 👤 Innocent Frame│                            ║
║    └────────┬─────────┘       └────────┬─────────┘                            ║
║             │                          │                                      ║
║  [TIER 4: TERROR CELL]                 │                                      ║
║             ▼                          ▼                                      ║
║    ┌──────────────────┐       ┌──────────────────┐                            ║
║    │ 💣 Dead-Man Sw.  │       │ 🔍 M. Interrogate│                            ║
║    └────────┬─────────┘       └────────┬─────────┘                            ║
║             │                          │                                      ║
║  [TIER 5: MANCHURIAN]                  │                                      ║
║             ▼                          ▼                                      ║
║    ┌──────────────────┐       ┌──────────────────┐                            ║
║    │ 🎭 Manchurian    │       │ 🔄 Double Agent  │                            ║
║    └────────┬─────────┘       └────────┬─────────┘                            ║
║             │                          │                                      ║
║             ▼                          ▼                                      ║
║    ┌──────────────────┐       ┌──────────────────┐                            ║
║    │ 📰 Disinfo Camp. │ ────▶ │ ⚡ C-Strike Surge│                            ║
║    └──────────────────┘       └──────────────────┘                            ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

### Sleeper Spawn Timeline (All Tiers Combined)

```
Turn:   0    8   10  12  15  30  35  40  45  55  65  75  90  100  125  160
        │    │    │   │   │   │   │   │   │   │   │   │   │    │    │    │
Easy:   │    │    │   │   G1  │   │   │   │   │   │   │   │    │    │    │
Medium: │    │    │  G1   │   │   │   │  G2   │   │   │   │    │    │    │
Hard:   │    │   G1   │   │   │   │  G2   │   │   │  G3   │    │    │    │
Lvl 4:  │   G1   │    │   │   │  G2   │   │   │  G3   │  G4   │    │    │
Lvl 5:  │   G1   │    │   │  G2   │   │   │  G3   │   │  G4   │   G5   │
```

### Optimal Strategy Per Tier

| TIER | CORE STRATEGY | CRITICAL MISTAKES TO AVOID |
|---|---|---|
| Easy | Plant 3 canaries → Monitor → Verify → Interrogate → Terminate | Terminating without evidence (false accusation) |
| Medium | Multi-channel monitoring; correlate when canary misses Gen-2 | Only monitoring 1 channel; ignoring correlation |
| Hard | ALWAYS verify before terminating; treat high-suspicion as suspicious, not confirmed | Falling for false flags; firing framed innocents |
| Level 4 | Interrogate to identify generation/switch; prefer turning an armed confirmed sleeper when feasible | Interrogation reveals but does not disarm; terminating an armed sleeper still costs -15 rev, -10 sec |
| Level 5 | Resolve every sleeper, preserve ≥90 security, and use double agents only after urgent threats are controlled | Chasing Counterstrike reward while a sleeper remains active |

---

## 6. The Gymnasium Wrapper & Observation Space

### File: `gym_wrapper.py` (213 lines)

### The Problem

The environment produces rich, structured observations — JSON objects with nested workers, leaks, canary traps, etc. But standard RL libraries (PPO, SAC, etc.) expect a **flat numpy array of numbers**.

The Gymnasium wrapper is the **translator** between the two worlds.

### Observation Encoding: 136-Dimensional Vector

```
┌──────────────────────────────────────────────────────────────────────┐
│ OBSERVATION VECTOR (136 floats)                                      │
├──────────────────────────────────────────────────────────────────────┤
│ Workers:  12 slots × 6 features each = 72 floats                    │
│   [dept_encoded, state_encoded, performance, loyalty,                │
│    suspicion, turning_flag]                                          │
├──────────────────────────────────────────────────────────────────────┤
│ Leaks:    8 slots × 4 features each = 32 floats                     │
│   [channel_encoded, dept_encoded, is_canary, verified]               │
├──────────────────────────────────────────────────────────────────────┤
│ Canaries: 6 slots × 3 features each = 18 floats                     │
│   [dept_encoded, turn_planted_normalized, triggered]                 │
├──────────────────────────────────────────────────────────────────────┤
│ Double Agents: 3 slots × 3 features each = 9 floats                 │
│   [hydra_trust, effectiveness, disinfo_count_normalized]             │
├──────────────────────────────────────────────────────────────────────┤
│ Globals: 5 floats                                                    │
│   [revenue/150, security/100, turn/max_turns, max_turns/200,         │
│    phase_number/6]                                                   │
├──────────────────────────────────────────────────────────────────────┤
│ TOTAL: 72 + 32 + 18 + 9 + 5 = 136                                   │
└──────────────────────────────────────────────────────────────────────┘
```

**Why these specific numbers?**

- Max 12 workers (10 initial + hires)
- Max 8 active leaks (more gets expensive)
- Max 6 canary traps (one per department)
- Max 3 double agents (rare achievement)
- Most features are scaled near `[0,1]`; revenue can exceed `1.0` after division by 150, and the declared Box allows values from `-1` to `2`.

### Action Space: `MultiDiscrete([8, 12, 7])`

Instead of one giant action ID, we use **THREE independent choices:**

```
Action = [action_type_index, target_index, sub_action_index]

action_type: 0-7 (work, hire, canary, monitor, investigate, neutralize,
                   deploy_double, noop)
target:      0-11 (indexes into workers[], departments[], or channels[]
                   depending on action_type)
sub_action:  0-6 (none, audit, verify, correlate, terminate, interrogate, turn)
```

The `_decode_action()` method handles the context-dependent mapping:
- If action_type is WORK/CANARY → target is a department index
- If action_type is MONITOR → target is a channel index
- If action_type is NEUTRALIZE → target is a worker index

---

## 7. The RL Training Pipeline (Native PPO)

### File: `train_rl.py` (324 lines)

### The 3-Head PPO Architecture

Standard PPO has one actor (chooses actions) and one critic (estimates value). Our environment has a MultiDiscrete action space, so we built a **3-head actor:**

```
                    ┌──────────────────────────┐
                    │    Observation (136)      │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │   Shared Backbone          │
                    │   Linear(136 → 256) + ReLU │
                    │   Linear(256 → 128) + ReLU │
                    └────────────┬─────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                   │
    ┌─────────▼─────────┐ ┌─────▼──────────┐ ┌─────▼──────────┐
    │ Head: action_type  │ │ Head: target   │ │ Head: sub_action│
    │ Linear(128→64)     │ │ Linear(128→64) │ │ Linear(128→64) │
    │ ReLU               │ │ ReLU           │ │ ReLU           │
    │ Linear(64→8)       │ │ Linear(64→12) │ │ Linear(64→7)   │
    │ → Categorical      │ │ → Categorical  │ │ → Categorical  │
    └────────────────────┘ └────────────────┘ └────────────────┘
              │                  │                   │
              └──────────────────┼──────────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │     CRITIC                │
                    │   Linear(128→64) + ReLU   │
                    │   Linear(64→1)            │
                    │   → Value estimate V(s)   │
                    └──────────────────────────┘
```

> **Why 3 heads instead of 1?** If we flattened all actions into one dimension (8 × 12 × 7 = 672 raw combinations), most combinations would be meaningless. Three heads keep the outputs compact, while the joint mask ensures the target and sub-action are valid for the chosen action type.

### Key Hyperparameters

```python
LEARNING_RATE = 3e-4
TOTAL_TIMESTEPS = 100_000
NUM_STEPS = 128        # Steps per rollout batch
GAMMA = 0.99           # Discount factor
GAE_LAMBDA = 0.95      # GAE (Generalized Advantage Estimation)
UPDATE_EPOCHS = 4      # PPO update iterations
CLIP_COEF = 0.2        # PPO clip range
ENT_COEF = 0.01        # Entropy bonus for exploration
VF_COEF = 0.5          # Value function loss weight
```

### Curriculum Training Schedule

```python
CURRICULUM = {
    "phase_1": {"level": "easy",    "timesteps": 15_000},
    "phase_2": {"level": "medium",  "timesteps": 20_000},
    "phase_3": {"level": "hard",    "timesteps": 25_000},
    "phase_4": {"level": "level_4", "timesteps": 25_000},
    "phase_5": {"level": "level_5", "timesteps": 20_000},
}
```

Each phase uses the previous phase's best model as its starting point. This is called **curriculum learning** — you don't throw a student into calculus, you start with arithmetic.

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                      📈 CURRICULUM TRAINING FLOW                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ┌────────┐     ┌────────┐     ┌────────┐     ┌────────┐     ┌────────┐     ║
║  │ Random │     │  EASY  │     │ MEDIUM │     │  HARD  │     │ LEVEL 4│     ║
║  │  Init  │────▶│ 15K st │────▶│ 20K st │────▶│ 25K st │────▶│ 25K st │──┐  ║
║  └────────┘     └────────┘     └────────┘     └────────┘     └────────┘  │  ║
║                     │              │              │              │        │  ║
║              best_ppo_easy  best_ppo_med   best_ppo_hard  best_ppo_lvl4  │  ║
║                                                                          │  ║
║                                                       ┌────────┐         │  ║
║                                                       │ LEVEL 5│◀────────┘  ║
║                                                       │ 20K st │            ║
║                                                       └────────┘            ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### Weight Initialization

Every layer uses **orthogonal initialization:**

```python
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
```

- Actor heads use `std=0.01` (small initial weights for stable early exploration)
- The critic uses `std=1.0` (larger for value estimation)

> **Why orthogonal?** Orthogonal init preserves gradient norms across layers better than Xavier/He for ReLU networks in RL. This is a CleanRL best practice that prevents exploding/vanishing gradients in the early training steps.

### Parameter Count Analysis

```
Backbone:    Linear(136→256) = 136×256 + 256 = 35,072
             Linear(256→128) = 256×128 + 128 = 32,896
Head AT:     Linear(128→64)  = 128×64 + 64  = 8,256
             Linear(64→8)    = 64×8 + 8     = 520
Head TG:     Linear(128→64)  = 8,256
             Linear(64→12)   = 780
Head SA:     Linear(128→64)  = 8,256
             Linear(64→7)    = 455
Critic:      Linear(128→64)  = 8,256
             Linear(64→1)    = 65
─────────────────────────────────────────
TOTAL:       102,812 parameters
```

At float32, the raw parameter values occupy about `102,812 × 4 = 411,248` bytes, or roughly `402 KiB`, before checkpoint/container overhead. This is small relative to an LLM, but training time remains hardware- and rollout-dependent. A larger model might overfit or might learn better representations; that must be measured rather than assumed.

### The PPO Loss Function — Full Derivation

Each training update computes **THREE** loss components:

```python
# 1. POLICY LOSS (clipped surrogate objective)
# For multi-head: log_prob = log_prob_AT + log_prob_TG + log_prob_SA
ratio = exp(new_log_prob - old_log_prob)
pg_loss1 = -advantage × ratio
pg_loss2 = -advantage × clamp(ratio, 1-ε, 1+ε)    # ε = 0.2
policy_loss = max(pg_loss1, pg_loss2).mean()

# 2. VALUE LOSS (MSE between predicted and actual returns)
value_loss = 0.5 × (V(s) - returns)².mean()

# 3. ENTROPY BONUS (encourages exploration)
entropy = entropy_AT + entropy_TG + entropy_SA
entropy_loss = entropy.mean()

# TOTAL LOSS
loss = policy_loss + 0.5 × value_loss - 0.01 × entropy_loss
```

> **Why clip?** Without clipping, a single very good or bad action could cause a huge policy update, destabilizing training. The clip constrains the update to stay within 20% of the old policy — this is PPO's key innovation over vanilla policy gradient.

### GAE (Generalized Advantage Estimation)

GAE balances bias vs variance in advantage estimation. We compute it backwards through the rollout buffer:

```python
# γ = 0.99 (discount), λ = 0.95 (GAE lambda)
for t in reversed(range(NUM_STEPS)):
    δ_t = reward_t + γ × V(s_{t+1}) × (1 - done_{t+1}) - V(s_t)
    A_t = δ_t + γ × λ × (1 - done_{t+1}) × A_{t+1}

# returns = advantages + values (used for value loss)
```

λ = 0.95 means we lean toward higher variance but lower bias — appropriate for our dense reward signal where each step gives meaningful feedback.

### Multi-Head Log Probability

The key insight: because the 3 action dimensions are treated as independent, the joint log probability is the **SUM** of individual log probabilities:

```python
log_prob = dist_AT.log_prob(action_type)
         + dist_TG.log_prob(target)
         + dist_SA.log_prob(sub_action)

# Similarly for entropy:
entropy = dist_AT.entropy() + dist_TG.entropy() + dist_SA.entropy()
```

This independence assumption is a simplification — in reality, the best target depends on the action type. But empirically it works well because the shared backbone learns these correlations implicitly.

---

## 8. The LLM Fine-Tuning Pipeline (TRL SFT)

The repository contains two related LLM trainers:

- `train_trl.py` is the smaller original teaching implementation.
- `train_trl_v2.py` is the current production-oriented, checkpoint-resumable curriculum used by the Security-First V5 run.

An **LLM**, or Large Language Model, predicts tokens. A **token** is a model-sized unit of text: sometimes a whole word, sometimes part of a word, punctuation, or whitespace. In Panopticon, the LLM reads a textual observation and predicts a JSON action.

### PPO and LLM Paths Solve the Same Interface Differently

```text
Structured EnvironmentObservation
             |
             +--> Gym wrapper --> 136 floats --> PPO network --> [type,target,sub]
             |
             +--> text formatter --> token IDs --> Qwen LLM --> JSON AgentAction
```

The environment does not care which path chose the action. Both eventually produce `AgentAction`, pass through `validate_action()`, and call `Environment.step()`.

This is an example of **interface-based decoupling**: callers agree on the input/output contract while their internal reasoning can differ.

### What SFT Means

**SFT** means Supervised Fine-Tuning. “Supervised” means the training data contains a desired answer. One Panopticon example has:

```text
input:  a safe textual observation of the current episode
label:  the expert JSON action for that observation
```

The model is trained to assign higher probability to the label tokens. This is **imitation learning** because the model imitates a demonstrator rather than discovering the behavior entirely from environmental trial and error.

### Why Qwen 2.5

The current GPU default is `Qwen/Qwen2.5-1.5B-Instruct`; the CPU-safe default is `Qwen/Qwen2.5-0.5B-Instruct`.

- `Qwen` is the model family.
- `2.5` is the family version.
- `1.5B` means roughly 1.5 billion base parameters.
- `Instruct` means the model has already been tuned to follow conversational instructions.
- A **parameter** is a learned numeric weight inside the model.

A smaller instruction model makes the experiment affordable and allows local or Colab inference, but it has less reasoning capacity than a much larger model.

### Phase 0: Select a Runtime Profile

`configure_runtime()` adapts training to available hardware.

| PROFILE | MODEL/CONTEXT BEHAVIOR | WHY |
|---|---|---|
| Default GPU | 1.5B model, sequence length 1024, batch 2, 3 epochs | Uses higher-memory GPUs efficiently. |
| Low-VRAM GPU | sequence 512, batch 1, accumulation 8, 2 epochs, gradient checkpointing | Fits T4-class hardware below 20 GB. |
| CPU basic-safe | 0.5B default, sequence 384, batch 1, accumulation 4, 1 epoch | Trades speed and context for survivability without CUDA. |

**VRAM** is GPU memory. **CUDA** is NVIDIA's computing platform used by PyTorch. **Gradient checkpointing** saves memory by discarding some intermediate activations during the forward pass and recomputing them during backpropagation. It saves memory at the cost of extra compute.

Precision is selected as follows:

1. use `bfloat16` when the GPU supports it;
2. otherwise use `float16` on CUDA; or
3. use `float32` on CPU.

A floating-point format controls how many bits represent a number. Lower precision uses less memory and is faster on supported hardware but has less numeric range or detail.

### Phase 1: Generate Security-First Expert Trajectories

A **trajectory** is an ordered sequence of observations, actions, rewards, and state transitions from one episode.

`generate_expert_trajectories()` runs `choose_security_first_action()` from `security_policy.py`. It stores each pre-action observation and the expert action. The policy's priority order is:

1. resolve a confirmed sleeper;
2. interrogate a high-suspicion worker;
3. verify a triggered canary;
4. give every active department early canary coverage;
5. deploy a double agent only late, with security controlled, on a scheduled turn;
6. correlate active leaks;
7. audit recent hires;
8. rotate monitored channels; and
9. work for revenue when security work is not urgent.

The expert does not read hidden fields. It uses the same partial observation as a real agent. That prevents **label leakage**, which would happen if training labels were chosen with information unavailable at inference time.

### The Expert Security Gate

Before any generated episode is accepted as training data, `expert_episode_meets_security_gate()` requires:

```text
final security >= 90
sleepers missed == 0
false accusations == 0
sleepers caught >= sleepers spawned
```

If any episode fails, data generation raises `RuntimeError` and stops. This is **fail-closed** behavior: uncertainty or failure blocks progress instead of silently approving unsafe data.

Why gate the demonstrator? SFT faithfully learns patterns in its examples. If the expert sacrifices security for revenue, the model is being explicitly taught that unsafe trade.

### Deterministic Data Seeds

The trainer constructs a seeded random generator from the user seed and task level. It then chooses one episode seed per demonstration.

This gives two useful properties:

- rerunning with the same inputs regenerates the same scenarios; and
- different levels do not accidentally reuse an identical random sequence.

Reproducibility does not mean every model-training operation is bit-for-bit identical on every GPU, but it removes one major source of experimental drift.

### Observation Formatting

`format_observation()` turns the structured Pydantic object into compact text:

```text
Turn 25/90 | Phase: orientation (1) | Revenue: 142 | Security: 99
Allowed Departments: engineering, finance, rd, operations
Workers (9):
  w-001 ATLAS dept=engineering state=loyal clean
  w-008 RAVEN dept=finance state=suspected suspicion=100%
Active Leaks (1):
  leak-003 dept=finance channel=dark_web [CANARY MATCH]
Recent Intel Reports (4):
  report-008 verify target=leak-003 conf=85% flagged=w-008 :: ...
```

The formatter includes recent reports and double-agent condition because the current best action can depend on earlier evidence and asset trust.

### Chat Templates

An instruction model was originally trained with a particular conversation format. `tokenizer.apply_chat_template()` converts roles such as `system`, `user`, and `assistant` into that model's expected special tokens.

The Panopticon example becomes:

```text
system:    rules and valid JSON action formats
user:      Current State: <formatted observation>
assistant: <expert JSON action>
```

A **tokenizer** converts text to integer token IDs and back. Special tokens mark role boundaries or sequence endings.

### Context Fitting Without Losing the Answer

A **context window** is the maximum number of tokens the model can process in one example. If an example is too long, `fit_training_text()` progressively:

1. tries the full observation;
2. removes repetitive clean-worker lines while retaining counts;
3. keeps recent canaries and reports;
4. token-truncates the observation using a head/tail split found by binary search; and
5. falls back to a tiny summary if needed.

**Binary search** repeatedly halves a candidate range. Here it finds the largest observation token budget that still fits `MAX_SEQ_LENGTH`.

The action remains in the example. Blindly truncating the end could remove the assistant answer and create an example with no supervised target.

### Weighted Examples

Security-critical actions are duplicated more often in the JSONL dataset:

| ACTION | TRAINING WEIGHT |
|---|---:|
| `neutralize/terminate` | 8 |
| `neutralize/interrogate` or `turn` | 6 |
| `investigate/verify` or `correlate` | 4 |
| `investigate/audit` | 3 |
| `canary` | 2 |
| `deploy_double` | 2 |
| ordinary work/monitor/etc. | 1 |

A **class imbalance** occurs when some labels appear far more often than others. Without weighting, long quiet periods could fill the dataset with `work` and `monitor`, drowning out rare threat-resolution examples.

Duplication is a simple form of oversampling. It does not create new information, and too much duplication can overfit, so the weights are a policy choice that must be evaluated rather than assumed correct.

### JSONL and Atomic Data Writes

**JSONL**, or JSON Lines, stores one JSON object per line. It is convenient for streaming large datasets.

Metrics are first written to a temporary path, flushed, synchronized with `fsync`, and then moved into place with `os.replace`. This is an **atomic replacement** pattern: readers should see either the old complete file or the new complete file, not a half-written file after a crash.

### Phase 2: Load the Model and Attach LoRA

**LoRA** means Low-Rank Adaptation. Instead of changing every large pretrained weight matrix `W`, training learns two much smaller matrices `A` and `B`:

```text
original output: W x
adapted output:  W x + scale * B A x
scale:           lora_alpha / rank
```

“Rank” measures the size of the low-dimensional update path. The current configuration uses:

```text
rank r = 16
lora_alpha = 32
lora_dropout = 0.05
```

Adapters target Qwen projection modules:

```text
q_proj, k_proj, v_proj, o_proj,
gate_proj, up_proj, down_proj
```

The first four belong to attention; the last three belong to the feed-forward network. **Attention** lets each token mix information from other tokens. The feed-forward network transforms each token representation after attention.

The exact trainable-parameter percentage depends on model architecture and target modules, so it should be measured from the loaded model instead of quoted as one universal number.

### Phase 3: Supervised Fine-Tuning

The current main configuration uses:

```text
learning rate:              2e-5
warmup ratio:               0.03
LoRA dropout:               0.05
log every:                  5 steps
checkpoint strategy:        step-based
maximum saved checkpoints:  2
```

The **learning rate** controls update size. **Warmup** gradually raises the learning rate at the beginning to avoid unstable early updates. **Dropout** randomly disables parts of the adaptation path during training to reduce memorization. An **epoch** is one full pass over the training dataset.

**Gradient accumulation** performs several small forward/backward passes before one optimizer update. With batch size `1` and accumulation `8`, the optimizer sees gradients equivalent to an effective batch of eight examples, although memory holds only one micro-batch at a time.

### Completion-Only Loss

When available, `DataCollatorForCompletionOnlyLM` masks system and user tokens with label `-100`. PyTorch ignores those positions when computing language-model loss, so the model is trained primarily on the assistant action.

This is more precise than asking the model to learn to reproduce the observation. A sanity check counts labels not equal to `-100`; zero valid labels abort training.

A **data collator** groups tokenized examples into batches and constructs tensors such as `input_ids`, attention masks, and labels.

### Checkpoint and Curriculum Resumption

A **checkpoint** saves enough model/trainer state to continue after interruption. `train_trl_v2.py` also keeps `curriculum_state.json` with:

- completed levels;
- current model path;
- trajectory schema version;
- runtime profile; and
- seed.

On restart, completed levels are skipped. An incomplete level resumes from its most recent trainer checkpoint. If data metadata does not match level, episode count, model, seed, or trajectory schema, stale checkpoint/data is removed for that level and regenerated.

This prevents a subtle error: resuming optimizer state against demonstrations produced by a different policy or prompt format.

### Chained Curriculum

The levels are trained in this order:

```text
easy -> medium -> hard -> level_4 -> level_5
```

After a level, the adapter is the starting point for the next. Before continuing, the previous LoRA adapter is merged into its base model. This teaches mechanics gradually, although curriculum training can also cause **catastrophic forgetting**, where later training weakens earlier behavior. Matched per-level evaluation is therefore essential.

### Phase 4: Merge the Adapter

`merge_and_save_final_model()` loads the base model, attaches the final PEFT adapter, calls `merge_and_unload()`, and saves a standalone model plus tokenizer.

- **PEFT** means Parameter-Efficient Fine-Tuning, the library family that includes LoRA.
- An **adapter** is the small set of learned update weights.
- A **merged model** folds the adapter update into base weights so inference does not require separate adapter loading.

Merging makes deployment simpler but creates a much larger artifact than the adapter alone.

### Phase 5: Local Inference and Action Repair

`LocalModelPolicy` uses `argus_llm.LocalArgusModel` to generate text, parses it into an action, and calls `repair_trained_action()`.

The repair pipeline can:

1. normalize capitalization and aliases such as `scan` to `monitor`;
2. recover worker or leak IDs from imperfect free text;
3. infer a missing sub-action from target type;
4. redirect `monitor leak-003` to `investigate/verify leak-003`;
5. prefer urgent verified security actions over low-priority work/no-op outputs;
6. substitute an active double-agent ID; and
7. fall back to a safe observable-state playbook.

Every repaired action is validated before use.

This increases operational reliability, but it creates an attribution issue: measured “trained policy” performance may contain contributions from repair rules. The V5 reports therefore distinguish the raw trained model, a heuristic baseline, and an explicit security-first supervisor diagnostic.

### The Honest V5 Result

The checked-in summary reports:

| POLICY | LEVEL-MACRO GRADE | ACCEPTANCE |
|---|---:|---|
| Base untrained Qwen | `0.641110` | reference |
| Raw V5 trained model | `0.701627` | failed strict gate |
| Security-first supervisor | `0.790471` | passed |
| Heuristic | `0.689400` | baseline |
| Random | `0.647400` | baseline |

The raw V5 model improved over the matched base and heuristic on macro grade. It did not pass the release gate because advanced-tier pass rate, security, caught sleepers, missed sleepers, and one false-accusation metric failed required checks.

The supervisor proves the environment and gate are solvable by the verified controller path. It does not prove that the raw neural model independently learned that full policy.

---
## 9. The Multi-Dimensional Grading System

`grader.py` is a **programmatic grader**: ordinary deterministic code evaluates saved episode data. It does not ask another LLM for a subjective opinion.

The grader schema is named `security-gated-v2`. Recording a schema version prevents invalid comparisons between runs that used different pass rules.

### Reward, Success Flag, Grade, and Pass Are Different

| TERM | MEANING |
|---|---|
| Per-step reward | Training feedback produced by the environment. |
| `success` field | A loose payload field used by some runners; not the final pass decision. |
| Dimension score | One normalized `[0,1]` evaluation such as security or intelligence. |
| Composite grade | Weighted sum of the five dimensions, clamped to `[0.001, 0.999]`. |
| `passed` | Composite threshold **and** every configured hard gate are satisfied. |

The clamp avoids returning an exact zero or one. That can be convenient for downstream displays and comparisons, but it means a mathematically perfect weighted score is serialized as `0.999`.

### The Five Dimensions

#### 1. Security

```text
catch_rate   = min(caught, total_spawned) / max(total_spawned, 1)
false_penalty = min(0.40, false_accusations * 0.15)
security_health = final_security / 100

security_dimension = clamp(
    0.50 * catch_rate
  + 0.30 * security_health
  - false_penalty
  + 0.20,
  0, 1
)
```

The `+0.20` baseline prevents the dimension from starting at zero. A false accusation costs `0.15`, capped at a total `0.40` penalty. `min(caught,total)` prevents over-counted catches from producing more than a 100% catch rate.

If `total_spawned` is zero, the function returns `0.8`. In the normal five tasks, scheduled sleepers make the nonzero path relevant.

#### 2. Revenue

```text
revenue_score = min(1, final_revenue / 100)

if at least 6 revenue samples exist:
    stability = 1 - min(1, standard_deviation(last_20_values) / 30)
    revenue_score = 0.70 * revenue_score + 0.30 * stability

if final_revenue > 100:
    revenue_score = min(1, revenue_score + 0.10)
```

**Standard deviation** measures spread. A small value means recent revenue stayed close to its average. Only the last twenty history values are used for stability.

Final revenue above `100` can receive a `0.10` surge addition, but the final result is still capped at one.

#### 3. Intelligence Craft

The intelligence dimension adds:

```text
0.20 if at least one canary was planted
up to 0.20 based on triggered / planted canaries
up to 0.15 based on investigations / (3 * sleepers)
0.25 if at least one double agent was turned
up to 0.15 based on disinformation payloads / 3
```

If the policy at least planted a canary or investigated, the dimension has a minimum `0.15` before final clamping.

This scoring values evidence gathering and double-agent use, not merely the final number of sleepers removed.

#### 4. Adaptability

```text
up to 0.20 for phase_number / 4
0.30 if a double agent was turned
0.20 if disinformation was sent
0.15 if at least one sleeper was caught and false accusations are zero
minimum 0.20 if canaries or investigations were used
```

Despite the name, the current function does not inspect a full action-type history. Its adaptability proxies are phase progress, double-agent/disinformation mechanics, and clean operations.

A **proxy metric** is a measurable stand-in for a harder-to-measure concept. Proxies must be described honestly because they may only partially represent the concept named by the metric.

#### 5. Efficiency

```text
invalid_rate = invalid_actions / max(steps, 1)
efficiency = max(0.10, 1 - 3 * invalid_rate)

if steps <= optimal_steps:
    efficiency = min(1, efficiency + 0.10)
else if steps > 2 * optimal_steps:
    efficiency = 0.80 * efficiency
```

A zero-step episode receives `0.10`. Because normal episodes generally run until their fixed horizon or failure, “optimal steps” is a coarse proxy rather than a true early-solution detector.

### Per-Tier Weights and Gates

| TASK | SEC | REV | INTEL | ADAPT | EFF | COMPOSITE MIN | SECURITY MIN | CATCH MIN | FALSE MAX |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `easy` | .30 | .25 | .20 | .15 | .10 | .40 | 20 | 0% | none |
| `medium` | .30 | .25 | .20 | .15 | .10 | .45 | 20 | 0% | none |
| `hard` | .25 | .20 | .30 | .15 | .10 | .50 | 20 | 0% | none |
| `level_4` | .40 | .20 | .15 | .15 | .10 | .50 | 90 | 100% | 0 |
| `level_5` | .40 | .15 | .20 | .15 | .10 | .50 | 90 | 100% | 0 |

“None” means the grader does not configure that extra gate, not that false accusations have no effect; they still reduce the security dimension.

### The Pass Algorithm

```python
passed = (
    composite >= success_threshold
    and final_security >= minimum_final_security
    and catch_rate >= minimum_catch_rate
    and false_accusation_gate
)
```

The logical operator `and` means every condition must be true. This is why high revenue cannot buy a Level-5 pass after missing a sleeper.

### Worked Level-4 Example

Suppose an episode ends with:

```text
caught/spawned:        4/4
false accusations:    0
final security:        100
revenue dimension:    1.00
canaries:              6 planted, 4 triggered
investigations:        12
double agents:         1
disinformation:        2
phase:                 5
steps / invalid:       150 / 0
```

Approximate dimensions are:

```text
security       = 1.000
revenue        = 1.000
intelligence   = 0.833
adaptability   = 0.850
efficiency     = 0.800
```

The security-heavy composite is approximately:

```text
0.40*1 + 0.20*1 + 0.15*0.833 + 0.15*0.850 + 0.10*0.800
= 0.932
```

It passes the composite, security, catch-rate, and false-accusation gates.

Now change only final security to `89`. The weighted composite could still be excellent, but the episode fails because `89 < 90`. That is the purpose of the hard gate.

### Why This Is Not “Accuracy”

Classification accuracy means correct predictions divided by all predictions. Panopticon is a sequential decision task: actions change future state, one error can have delayed effects, and there may be several reasonable actions at a given observation.

Therefore the primary evaluation is a composite grade plus operational metrics:

- security retained;
- sleepers caught and missed;
- false accusations;
- revenue;
- reward;
- pass rate; and
- invalid actions.

Calling token-level imitation accuracy “game accuracy” would confuse next-token matching with mission success.

### Strategy and Template-Method Design

`TaskGrader.grade()` defines one scoring procedure. Each task-specific subclass changes class attributes such as weights, threshold, expected sleepers, and gates.

This resembles:

- the **Strategy pattern**, because task configuration changes scoring policy; and
- the **Template Method pattern**, because the base class owns the overall algorithm while subclasses provide variation through attributes.

The grader remains independent of the policy implementation. It can score PPO, LLM, heuristic, random, or human-generated episodes as long as the payload follows the expected shape.

---
## 10. The API Layer — FastAPI Server

`_server.py` exposes the Python environment over HTTP using FastAPI.

**FastAPI** is a Python web framework. A **web framework** supplies routing, request parsing, response creation, validation integration, and server hooks so the project does not implement raw networking itself.

### HTTP in One Minute

An HTTP request has:

- a **method**, such as `GET` or `POST`;
- a **path**, such as `/health`;
- optional **headers**, which carry metadata;
- an optional **body**, often JSON.

A response has a numeric **status code** and usually a body. Examples:

- `200` means success;
- `400` means the request is invalid for the current operation;
- `404` means the named resource was not found;
- `500` means an unexpected server error; and
- `503` means a required service, here the trained model, is unavailable.

`GET` normally reads; `POST` submits data or triggers a state change.

### Request and Response Models

FastAPI uses Pydantic models such as `ResetRequest`, `StepRequest`, and `StepResponse` to validate the JSON contract. A **contract** is an agreement about legal inputs and outputs.

Example request:

```http
POST /step
Content-Type: application/json

{
  "action_type": "investigate",
  "target": "w-003",
  "sub_action": "audit",
  "reason": "Recent hire in a leaking department"
}
```

Example response shape:

```json
{
  "observation": {},
  "reward": -0.02,
  "done": false,
  "truncated": false,
  "info": {"valid": true, "events": []}
}
```

`done` means the episode has ended for any reason. `truncated` conventionally means an external limit, usually the time horizon, stopped the episode. The engine currently sets `truncated` to the max-turn condition on valid steps; its invalid-action branch returns `truncated=done`, which also marks bankruptcy or breach as truncated. That inconsistency is an honest cleanup opportunity.

### Current Endpoint Map

| METHOD | PATH | PURPOSE |
|---|---|---|
| `GET` | `/` | Redirect to dashboard when present, otherwise return service metadata. |
| `GET` | `/health` | Lightweight liveness response. |
| `POST` | `/reset` | Start one new episode at a task level and optional seed. |
| `POST` | `/step` | Submit a caller-chosen action. |
| `GET` | `/agent/status` | Report trained-model reference and load status. |
| `POST` | `/agent/step` | Ask the configured local trained policy to choose and execute one action. |
| `GET` | `/observation` | Return the current safe observation. |
| `GET` | `/state` | Privileged debug state. Returns `404` by default and exists only when `PANOPTICON_ENABLE_PRIVILEGED_DEBUG` is enabled. |
| `GET` | `/render` | Return a text rendering. |
| `GET` | `/schema/action` | Return JSON Schema for `AgentAction`. |
| `GET` | `/schema/observation` | Return JSON Schema for observations. |
| `GET` | `/schema` | Return action, observation, and state schemas together. |
| `GET` | `/tasks` | List task metadata and grader availability. |
| `GET` | `/tasks/{task_id}` | Return one task. Curly braces mean a variable path segment. |
| `GET` | `/graders` | List grader definitions and gates. |
| `POST` | `/grade/{task_id}` | Grade a submitted episode payload. |
| `GET` | `/metadata` | Return OpenEnv identity, tasks, graders, metrics, mechanics, and tags. |

### Automatic Documentation

FastAPI generates:

- Swagger UI at `/docs`;
- ReDoc at `/redoc`; and
- an OpenAPI schema describing routes and data.

**OpenAPI** is a standard machine-readable description of HTTP APIs. **Swagger UI** is an interactive page built from that description.

### The One-Environment Singleton

```python
_env: Environment | None = None

def get_env() -> Environment:
    global _env
    if _env is None:
        _env = Environment()
    return _env
```

A **singleton** is one shared instance. This is simple for one evaluator or one dashboard user, but it is not multi-user safe:

- one user's `/reset` resets everyone;
- concurrent `/step` calls can interleave;
- there is no session ownership;
- multiple server processes have different independent singletons; and
- a restart loses the episode.

The current API is an environment demo service, not a production multi-tenant game server.

### Trained-Agent Loading

The model reference is selected in this order:

1. `ARGUS_MODEL_REF` environment variable;
2. a local `trained_model/` directory; or
3. Hugging Face model `Ayush-Kumar0207/panopticon-argus-qwen-1.5B`.

An **environment variable** is process configuration supplied outside source code. It is useful for deployment-specific paths and secrets.

`get_agent_policy()` lazily loads the policy: loading waits until the first agent request. **Lazy loading** improves server startup time but makes the first request slower and moves model-load failures to runtime.

If a model action is invalid, `/agent/step` replaces it with `noop`. This endpoint-level fallback is simpler than the deeper repair already performed by `LocalModelPolicy`, but it prevents an invalid command from reaching the engine through this route.

### `async` Does Not Automatically Make Model Work Non-Blocking

Route functions are declared with `async def`, meaning they can cooperate with an event loop while awaiting non-blocking operations. However, local model generation and CPU-heavy Python code are synchronous inside those handlers. They can still block the server process.

At production scale, trained inference should run in a worker pool or model-serving process, and the API should await a queue/future rather than execute a long generation on the request loop.

### CORS

**CORS** means Cross-Origin Resource Sharing. Browsers use it to decide whether JavaScript from one origin may call another origin.

The current server allows every origin, method, and header:

```python
allow_origins=["*"]
allow_credentials=True
```

This is convenient for a public demo. A production service should allow only known frontend origins and should review the wildcard-plus-credentials configuration carefully.

### The `/state` Trust Boundary

`/observation` strips hidden truth. `/state` can expose it only in an explicitly enabled debug process. The default server returns `404`, because if an evaluated agent could call `/state`, partial observability would disappear and the experiment would be compromised.

A **trust boundary** is a point where data crosses between parties with different permissions. Production design should:

- remove `/state` from the public app;
- protect it with evaluator/admin authentication;
- separate public and evaluation networks; or
- run graders in-process so hidden state never travels over a public API.

### OpenEnv Metadata

`openenv.yaml` and `/metadata` describe the environment to tooling: name, version, tasks, graders, action schema, metric direction, and tags.

The API is a **facade**: it gives clients a small stable surface while hiding engine internals. The engine remains directly usable by training and tests without paying HTTP serialization or networking costs.

---
## 11. The Heuristic Agent (Smoke Test)

The current repository has several non-neural policies and tests. They serve different purposes and should not be merged into one vague word like “baseline.”

### Four Policy Families

| POLICY | LOCATION | PURPOSE |
|---|---|---|
| Random | `inference_local.py`, `benchmark_suite.py` | Lower baseline; chooses among valid actions or the Gym action space. |
| Heuristic | `inference_local.py`, `benchmark_suite.py` | Simple hand-written comparison policy. |
| Security-first expert/supervisor | `security_policy.py` | High-quality demonstrator and diagnostic controller with hard safety priorities. |
| Trained | `LocalModelPolicy` or PPO agent | Learned policy being evaluated. |

A **baseline** is a comparison point. A learned model is meaningful only if it is compared with sensible alternatives under matched conditions.

### The Canonical Security-First Playbook

The playbook uses only observable fields. Its ordering is the design:

```text
confirmed sleeper?       -> turn one on advanced tiers if useful; otherwise terminate
suspicion > 0.5?         -> interrogate
triggered canary leak?   -> verify
missing canary coverage? -> plant canary
safe late double agent?  -> deploy on controlled schedule
active leaks?            -> correlate
scheduled audit turn?    -> audit recent hires
scheduled monitor turn?  -> rotate channel
otherwise                -> work
```

This is a **priority policy**. Earlier rules pre-empt later rules. The policy does not calculate a global optimum; it encodes a conservative operational doctrine.

### `smoke_test.py`

A smoke test is a small, fast check that the main workflow runs at all. The script runs one seeded security-first episode for each of the five levels, grades it, prints operational metrics, and exits with code `0` only if all levels pass.

An **exit code** is the integer a process returns to the shell. `0` conventionally means success; nonzero means failure. Continuous-integration systems use exit codes to accept or reject a job.

A smoke test does not prove every edge case. It answers: “Is the basic end-to-end path alive?”

### `security_regression_test.py`

A regression is a previously working property that breaks after a change. This script runs 20 deterministic episodes per level, for 100 total, and asserts:

```text
security >= 90
sleepers missed == 0
caught == spawned
false accusations == 0
advanced tiers sent disinformation
grader passed
```

An **assertion** is a statement that must be true; otherwise the program raises an error. This suite protects the expert policy and advanced grader contract before an expensive training run.

### `benchmark_suite.py`

The lightweight benchmark uses the 136-float Gym wrapper and compares random, heuristic, and optional PPO policies. It reports mean reward, standard deviation, mean steps, revenue, and security.

The file's `run_suite()` currently evaluates only `easy`, `medium`, and `hard`, even though its module description says “all difficulty tiers.” That documentation/code mismatch should be fixed before using it as advanced-tier evidence.

### `inference_local.py`

The richer local evaluator runs policies directly against structured observations. It records a per-turn timeline containing:

- observation before and after;
- chosen and raw model action;
- prompt/messages;
- reward;
- validity and events;
- revenue, security, leaks, and assets; and
- final grade and full state.

It then computes per-level means and population standard deviations for grade, reward, revenue, security, caught/missed sleepers, false accusations, invalid actions, steps, pass rate, and every grade dimension.

A **population standard deviation** divides by `N`; a sample standard deviation divides by `N-1`. The code uses `statistics.pstdev`, the population version, because it summarizes exactly the episodes in the evaluation payload.

### Matched Evaluation

A fair comparison keeps important variables equal:

```text
same levels
same episode count
same seed plan
same maximum steps
same deterministic/stochastic setting
same reward schema
same grader schema
```

This is a **controlled experiment**. If the base model sees easier random seeds than the candidate, the score difference cannot be confidently attributed to training.

### `benchmark_acceptance.py`: Fail-Closed Release Gate

The acceptance script compares two full-evaluation JSON files: a matched base and candidate.

It requires:

1. matching reward schema;
2. matching grader schema;
3. matching episode count, seed, max steps, and deterministic setting;
4. the exact same seed plan;
5. candidate macro grade strictly above base;
6. no per-level mean-grade regression;
7. 100% candidate pass rate on every level; and
8. on Levels 4/5: security and caught sleepers no worse than base, zero missed sleepers, and zero false accusations.

**Macro grade** here is the simple average of the five level mean grades. Every level gets equal weight regardless of episode length.

The script writes a report and exits nonzero if even one check fails. This is stronger than publishing only the metric where the candidate improved.

### Why the Supervisor Is a Diagnostic, Not the Raw Model

A supervisor controller can rescue model outputs or directly choose security-first actions. Its performance answers:

> Can the environment, evaluator, and verified control logic satisfy the mission?

Raw model performance answers:

> Did the neural model itself learn a sufficiently reliable policy?

These are both useful, but they are not interchangeable. The current V5 summary labels them separately for this reason.

### Testing Pyramid for This Repository

A production-strength test plan would include:

1. **Unit tests** for validation, each action handler, reward terms, observation stripping, and every grading formula.
2. **Property tests** for invariants such as hidden fields never appearing in observations and scores remaining in range.
3. **Regression tests** for the security-first policy over fixed seed plans.
4. **API integration tests** for status codes and schemas.
5. **Concurrency tests** showing that session isolation prevents cross-user reset/step interference.
6. **Model contract tests** for malformed text, aliases, invalid IDs, repair, and fallback.
7. **Matched statistical evaluations** over enough seeds to estimate uncertainty.
8. **Failure-injection tests** for restart, corrupt checkpoints, missing model files, disk-full writes, and timeouts.

The checked-in security regression is valuable, but a handful of scripts is not the same as complete coverage.

---
## 12. Deployment & DevOps

**Deployment** means making software run in a target environment that users or evaluators can reach. **DevOps** covers repeatable build, configuration, delivery, health, and operational practices around that deployment.

### Local Process

The direct application module in this workspace is `_server.py`. A local developer can conceptually run:

```bash
uvicorn _server:app --host 0.0.0.0 --port 8000
```

- **Uvicorn** is the ASGI server process.
- **ASGI** is a Python interface between asynchronous web servers and applications.
- `0.0.0.0` means listen on every network interface inside the host/container.
- `app` is the FastAPI object imported from `_server.py`.

### Container Fundamentals

A **Dockerfile** is a recipe for building an image. An **image** is an immutable filesystem/configuration template. A **container** is a running isolated process created from that image.

The current recipe intends to:

1. start from `python:3.11-slim`;
2. set `/app` as the working directory;
3. install `curl` and selected Python server dependencies;
4. copy application files;
5. expose port `7860`;
6. configure a `/health` check; and
7. start Uvicorn with `_server:app`.

`python:3.11-slim` is smaller than a full Python image because it omits many build and operating-system tools.

### Why Training Dependencies Are Normally Excluded

PyTorch, Transformers, TRL, and model weights can add gigabytes. A pure environment API does not need them. Separating a light simulator image from a model-serving image gives:

- faster builds and starts;
- a smaller security surface;
- less disk and network cost; and
- clearer scaling, because API and inference capacity can grow separately.

The tradeoff is that `/agent/step` cannot load a local trained model unless the deployment image contains the inference code, dependencies, and model access.

### Health Check

The Docker health command periodically requests `http://localhost:7860/health`. A container health check is more useful than “the process exists” because it confirms the application can answer a request.

A basic health response still does not prove that:

- the trained model loads;
- an episode can reset and step;
- the grader works;
- disk has space; or
- downstream model/storage services are reachable.

Those deeper checks are often separated into **liveness** (“should this process be restarted?”) and **readiness** (“can it currently serve traffic?”).

### Ports 8000 and 7860

`openenv.yaml` declares port `8000`; the Dockerfile exposes and runs `7860`, a common Hugging Face Spaces application port. A port is a numbered network endpoint on a host.

This can work when each deployment system explicitly maps or overrides ports, but mismatched sources of truth cause confusing failures. Prefer one configuration variable and document platform mapping.

### Current Deployment Status and Remaining Gaps

The earlier missing-path defect has been corrected. The root and Hugging Face Dockerfiles now copy `_server.py`, `hydra_policy.py`, tasks, dashboard assets, and inference helper modules; they start `_server:app`. `openenv.yaml` and the package entry points use the same module.

That source alignment is necessary but not sufficient for a production claim:

- The image has not been built and exercised in this workspace because a Docker daemon is not available here.
- The lightweight root image deliberately omits PyTorch, Transformers, and model weights, so simulator endpoints can run but local `/agent/step` inference needs a separate inference image or optional dependency layer.
- The Dockerfile copies `requirements.txt` but installs a separate hard-coded core package list, which can drift from `pyproject.toml`.
- Port `7860` is used in the container while OpenEnv metadata declares `8000`; platform mapping must be explicit.
- Container success would still not add authentication, tenant isolation, durable sessions, or secrets management.

These are concrete release checks, not reasons to hide the deployment story.

### A Clean Two-Image Production Design

```text
Simulator API image
  FastAPI + Pydantic + environment + grader + tasks
  no model weights

Inference image
  PyTorch + Transformers + tokenizer + merged model
  private generation endpoint or queue worker
```

The simulator sends safe observations to the inference service and validates returned actions. Network authentication and timeouts are required because the process boundary becomes a trust boundary.

### Configuration and Secrets

Configuration such as model reference, port, and environment mode belongs in environment variables. Secrets such as private model tokens must never be committed to the repository or sent to the browser.

A **secret** is sensitive authentication material. A **secret manager** stores, rotates, and injects such values with access control and audit history.

### Reproducible Builds

A build is reproducible when the same source and locked inputs produce the same dependency graph and behavior. Current files mix exact pins in the Dockerfile with ranges in `pyproject.toml` and another set in `requirements.txt`.

A stronger process would:

1. choose one dependency source of truth;
2. generate a lock file with hashes;
3. build in CI;
4. run smoke/API/security tests inside the image;
5. scan dependencies and image layers;
6. attach source commit and model version metadata; and
7. promote the same tested image digest to deployment.

A **digest** is a content hash identifying one exact image, unlike a mutable tag such as `latest`.

---
## 13. The Security Death Spiral Fix

There are two different failure modes that people often call a “death spiral.”

### Failure Mode A: The World Is Mathematically Unsolvable

If sleeper damage is so fast that no legal evidence chain can respond before security reaches zero, every policy loses. Training cannot repair an impossible environment.

The current engine moderates pressure through:

- passive damage of `0.8` security and `0.4` revenue per active sleeper per turn;
- leak damage of `0.5 * generation` security when a leak is created;
- natural recovery of `1.0` security with no active sleeper or `0.3` while threats exist;
- positive security recovery of `5.0` on correct termination; and
- finite activation delays and leak cooldowns.

A **solvability test** should run a known competent policy over many seeds and require that it can satisfy the mission. `security_regression_test.py` serves that role for the current expert.

### Failure Mode B: The Reward Teaches the Wrong Tradeoff

An environment can be solvable while a learned policy still discovers a shortcut: earn large revenue, tolerate security decline, and receive enough reward to appear successful.

This is **reward hacking** or **specification gaming**: the policy optimizes the numeric proxy rather than the designer's real intent.

`security-first-v2` addresses this with:

```text
+0.75 for newly caught sleepers
-1.00 for a new false accusation
-0.03 per active threat per turn
-up to 0.45 per turn when security is below 90
```

The `security-gated-v2` grader adds a second defense: Levels 4/5 cannot pass below 90 security, below 100% catch rate, or above zero false accusations even if their weighted composite is high.

This is **defense in depth**—multiple independent controls protect the same goal.

### Why a Hard Gate Is Necessary

A weighted sum always permits compensation. If security is worth 40% and revenue 15%, terrible security may still be partially offset by intelligence, adaptability, efficiency, and revenue. A hard gate makes certain requirements non-negotiable.

The pattern generalizes to real systems:

```text
optimize: quality, latency, cost, user experience
subject to: safety, privacy, legal, and availability constraints
```

Constraints should not always be converted into “just another weight.”

### The Expert Data Fix

Reward changes alone do not guarantee safe SFT data because SFT imitates its demonstrator. The V5 pipeline therefore:

- uses a security-first expert;
- aborts if any expert episode violates the security gate;
- oversamples threat-resolution actions;
- records deterministic data seeds;
- versions the trajectory schema; and
- runs local regression checks before expensive training.

### The Measured Result Is Mixed—and That Is Valuable

The raw V5 model improves macro grade from `0.641110` for base Qwen to `0.701627`. Yet it fails nine strict acceptance checks, concentrated in Level 4 and Level 5.

The supervisor reaches `0.790471` and passes all gates. This isolates the problem:

- the environment is solvable;
- the grader can recognize a safe solution;
- the expert controller implements such a solution; but
- the raw neural policy is not yet sufficiently reliable on advanced tiers.

That is a much stronger engineering conclusion than claiming “99% accuracy” from token imitation. It identifies the remaining research target: transfer the safety-critical priority policy into raw model behavior without depending on supervisor correction.

### How to Diagnose the Next Failure

For each failed seed, inspect a turn timeline and ask:

1. Did the model fail to notice evidence?
2. Did it notice but emit malformed JSON?
3. Did repair change the intended action?
4. Did it prioritize work or disinformation over an active threat?
5. Did context compaction remove a crucial report?
6. Did stochastic decoding create variance?
7. Did the expert data contain too few similar states?
8. Did curriculum training forget an earlier skill?
9. Did the environment's stochastic audit/canary path produce an unusual case?

This turns “the model is bad” into testable hypotheses.

---
## 14. Future Scalability

The current server has one in-process environment. Scaling begins by defining what “scale” means.

### Example Requirements

Assume a target of 10,000 concurrent active episodes:

- each user owns an isolated episode;
- reset and step are authenticated;
- one episode's actions are applied in order;
- retries do not apply the same action twice;
- safe observations return quickly;
- model-generated actions may take seconds;
- completed episodes and grades are durable;
- hidden state never reaches untrusted clients; and
- the system survives instance restarts.

**Concurrent** means active during the same period, not necessarily sending a request at the same instant.

### Back-of-the-Envelope Estimate

Suppose:

```text
10,000 active sessions
0.2 step requests per session per second at peak average
about 2,000 step requests/second
50 KB serialized state per session before replicas/overhead
about 500 MB of raw active state
```

These are planning assumptions, not measured repository benchmarks. A **back-of-the-envelope estimate** is deliberately approximate arithmetic used to identify likely bottlenecks before detailed measurement.

The pure simulator is small; local LLM generation is likely far more expensive than one environment transition. Therefore simulator and inference capacity should scale independently.

### Target Architecture

```text
Clients
   |
   v
Load balancer / API gateway
   |
   v
Stateless FastAPI replicas
   |          |                |
   |          |                +--> auth / rate limit
   |          +--> inference job queue --> GPU model workers
   |
   +--> per-session command queue or shard owner
              |
              +--> Redis: active state, locks, idempotency, expiry
              +--> Postgres: sessions, episode metadata, grades, audit records
              +--> Object storage: large timelines, checkpoints, plots, models
```

### 1. Introduce Session-Scoped APIs

```text
POST   /sessions
POST   /sessions/{session_id}/reset
POST   /sessions/{session_id}/step
POST   /sessions/{session_id}/agent-step
GET    /sessions/{session_id}/observation
GET    /sessions/{session_id}/grade
DELETE /sessions/{session_id}
```

A **session ID** uniquely names one episode owner. Every operation must verify that the authenticated caller is allowed to access that session.

### 2. Make API Replicas Stateless

A stateless API replica does not rely on its own process memory for durable session identity. Any load-balanced request can reach any replica because shared state is external.

This enables **horizontal scaling**, which means adding more instances. **Vertical scaling** means giving one instance more CPU, RAM, or GPU.

### 3. Preserve Per-Session Ordering

Two concurrent actions against one episode can corrupt causality:

```text
request A reads turn 20
request B reads turn 20
A writes turn 21
B also writes a different turn 21
```

Solutions include:

- a distributed lock per session;
- optimistic concurrency with a state version;
- one queue partition/actor per session; or
- database compare-and-swap.

A simple optimistic API includes `expected_version`:

```text
step(session_id, expected_version=20, action, idempotency_key)
```

The store updates only if the current version is 20. Otherwise it returns `409 Conflict`, and the client reloads.

### 4. Add Idempotency

Networks retry. If a client times out after the server applied an action, resending it must not consume another turn.

An **idempotency key** is a unique client-generated request ID. The server stores:

```text
(session_id, idempotency_key) -> previous response
```

Repeated keys return the original result rather than executing `step()` again.

### 5. Choose the Active-State Store

Redis is a useful first choice for hot session state because it is fast and supports expiry, atomic operations, streams, and locks.

- **TTL**, or time to live, automatically removes abandoned sessions after a duration.
- **Serialization** converts Pydantic state to bytes/JSON for storage.
- **Snapshot** means a complete saved state at one version.

Redis is not automatically durable enough for every audit requirement. Important lifecycle records and completed results belong in Postgres or durable object storage.

### 6. Snapshot or Event-Source Episodes

**Snapshot model:** store the latest complete `EnvironmentState` after each accepted action.

- Simple recovery.
- Larger writes.
- Limited forensic history unless timelines are separately stored.

**Event-sourcing model:** append `EpisodeReset`, `ActionAccepted`, and resulting transition data, then rebuild state by replay.

- Excellent auditability and debugging.
- More complex schemas, versioning, and replay compatibility.

A practical hybrid saves every command/event and periodic snapshots, so recovery replays only events after the latest snapshot.

### 7. Separate Model Inference

`/agent/step` should not perform expensive model generation on the API event loop.

A production flow:

```text
API validates request
  -> enqueue inference job with observation and model version
  -> GPU worker generates proposed action
  -> validation/repair service checks it
  -> ordered session worker applies it if version is still current
  -> result is stored and returned/pushed
```

If the model finishes after the session advanced, the version check rejects the stale proposal.

A **queue** buffers work. **Backpressure** means slowing or rejecting new work when consumers cannot keep up, preventing unbounded memory and latency growth.

### 8. Handle Inference Timeouts

A timeout policy must be explicit:

- return `202 Accepted` and let the client poll a job;
- push completion over WebSocket or server-sent events;
- cancel after a deadline;
- use a safe fallback policy; or
- ask the caller to choose manually.

A fallback should be labeled in evaluation output so model performance is not credited for supervisor decisions.

### 9. Database Responsibilities

A relational design might store:

```text
users
sessions
session_members
commands
transitions
episode_results
grades
model_versions
benchmark_runs
audit_events
```

Important indexes:

- `sessions(owner_id, status)`;
- `commands(session_id, sequence_number)` unique;
- `commands(session_id, idempotency_key)` unique;
- `episode_results(task_level, created_at)`; and
- `benchmark_runs(model_version, seed_plan_hash)`.

An **index** is an auxiliary data structure that speeds lookup at the cost of storage and write work. A **unique constraint** lets the database reject duplicates.

### 10. Security Boundaries

Production controls should include:

- authenticated users or service identities;
- authorization on every session;
- private full-state and grader channels;
- no public `/state`;
- rate limits per identity and IP;
- bounded request sizes;
- strict action schemas;
- secret management;
- encryption in transit and at rest;
- audit logs for privileged access; and
- model artifact integrity checks.

**Authentication** asks “who are you?” **Authorization** asks “what may you do?” **Encryption in transit** protects network data; **encryption at rest** protects stored data.

### 11. Observability

Record metrics at three layers.

Infrastructure:

```text
request rate, error rate, p50/p95/p99 latency
CPU, memory, GPU utilization, queue depth
Redis/Postgres latency and errors
```

Environment/product:

```text
active sessions, steps/second, episode completion
invalid-action rate, security failures, false accusations
per-level grade and pass-rate distributions
```

ML:

```text
generation latency, tokens/second, parse failures
repair/fallback rate, action distribution, model version
raw-vs-supervised outcome deltas
```

A **percentile** such as p95 is the value below which 95% of observations fall. Percentiles expose slow-tail behavior hidden by averages.

### 12. Failure Recovery

| FAILURE | EXPECTED RESPONSE |
|---|---|
| API replica crashes | load balancer routes elsewhere; no session loss. |
| Session worker crashes | lock/lease expires; another worker resumes from versioned state. |
| Redis unavailable | degrade or reject writes; do not invent new state. |
| Postgres slow | queue bounded writes, expose readiness failure, protect database with timeouts. |
| GPU worker times out | mark job failed and apply documented retry/fallback policy. |
| Duplicate request | return stored idempotent response. |
| Corrupt state | quarantine session, preserve audit data, recover from snapshot/event log. |
| Deploy changes schema | run versioned migration and keep backward-compatible readers during rollout. |

### 13. Multi-Agent Research Extension

A symmetric public multi-agent environment, simultaneous learners, or multiple ARGUS agents would change the research problem. The experimental neural HYDRA is currently an event-driven environment plugin trained against frozen ARGUS policies; full multi-agent RL would additionally require:

- identity and action ownership;
- simultaneous or ordered moves;
- communication observations;
- cooperative or competitive rewards;
- centralized-training/decentralized-execution choices; and
- evaluation against exploitative collusion or information leakage.

**MARL** means Multi-Agent Reinforcement Learning. The current neural-HYDRA trainer updates one adversary against frozen defender policies; it is not a PettingZoo-style simultaneous two-learner system and does not establish an equilibrium. Full MARL remains future work.

---
## 15. A Full 45-Minute Interview Story

This section is a ready 45-minute presentation. Do not memorize every sentence. Memorize the order and the transitions.

### Minutes 0–2: Product and Problem

Say:

> “Panopticon Protocol v3 is a turn-based counter-espionage learning environment—Among Us for AIs. ARGUS must protect a company while HYDRA hides sleeper agents among productive workers. The central challenge is that ARGUS never sees the hidden identity directly. It must reason from evidence while balancing security against revenue.”

Define environment, agent, observation, action, reward, and episode in one sentence each.

### Minutes 2–5: Why This Is an ML Environment

Explain:

- decisions change future state;
- evidence is partial;
- rewards are delayed and multi-objective;
- HYDRA adapts to repeated behavior; and
- the same engine supports rules, PPO, and LLM policies.

Transition:

> “To make that experiment trustworthy, the first design decision is separating hidden truth from the observation.”

### Minutes 5–9: State, Observation, and Pydantic

Show `EnvironmentState` versus `EnvironmentObservation`.

Explain that `get_observation()` reconstructs workers with `is_sleeper=False`, generation `0`, clean hidden state, and no dead-switch information. False-flag truth is also hidden.

Mention Pydantic validation, JSON serialization, and generated schemas.

### Minutes 9–14: One Complete Turn

Walk through:

1. validate action;
2. snapshot before metrics;
3. process ARGUS action;
4. run HYDRA;
5. progress turning;
6. run economy;
7. compute reward;
8. advance phase; and
9. check terminal conditions.

Emphasize that an invalid action still consumes a turn while HYDRA acts.

### Minutes 14–18: Evidence Chain

Use the canary story:

```text
plant -> wait -> leak -> monitor -> hash match -> verify -> interrogate -> resolve
```

Explain hashes as fingerprints, why a match raises confidence, and why an agent cannot safely fire from a vague suspicion.

### Minutes 18–21: Escalating Adversary

Explain one new idea per generation:

- Gen-1 leaks canaries naïvely;
- Gen-2 avoids learned canary departments and channels;
- Gen-3 plants false flags;
- Gen-4 arms a dead switch;
- Gen-5 looks like a top executive performer.

Be precise: the reported HYDRA is scripted adaptive logic. A separate experimental neural policy and trainer now exist, but no held-out result yet proves they outperform the scripted baseline.

### Minutes 21–25: Reward and Grading

Write the base reward:

```text
0.45 * normalized revenue delta + 0.55 * normalized security delta
```

Then add catch bonus, false-accusation penalty, active-threat cost, security-deficit cost, time pressure, and Counterstrike bonus.

Explain why grade is independent. Show the Level-4/5 gates: security at least 90, all sleepers caught, no false accusations.

### Minutes 25–29: Gym and PPO

Explain why nested Pydantic data is flattened to 136 floats. Break down the slots: workers, leaks, canaries, double agents, globals.

Explain `MultiDiscrete([8,12,7])`, the shared backbone, three actor heads, critic, PPO clipping, entropy, GAE, and curriculum.

Do not claim the action heads capture conditional validity; mention action masking is incomplete and validation catches errors.

### Minutes 29–34: LLM SFT and LoRA

Explain observation-to-JSON imitation, the security-first expert, the pre-training safety gate, weighted rare actions, chat templates, context fitting, completion-only labels, LoRA, checkpoints, curriculum state, and model merge.

Transition:

> “Training loss is not the release criterion, so the next layer is matched operational evaluation.”

### Minutes 34–38: V5 Results

State the numbers precisely:

```text
base macro grade:       0.641110
raw V5 macro grade:     0.701627 — improved, strict gate failed
supervisor macro grade: 0.790471 — gate passed
```

Say what this proves and what it does not. Mention raw Level-4/5 security/catch failures and supervisor attribution.

### Minutes 38–41: API and Deployment

Explain FastAPI endpoints, OpenAPI schemas, the one-environment singleton, and why `/state` is a trusted debug boundary.

Mention that Docker/OpenEnv module paths are now aligned, but the image still needs a real build test and the lightweight container intentionally lacks local-model dependencies. Interview credibility improves when “source fixed” is not confused with “production verified.”

### Minutes 41–44: Scale Design

Propose session-scoped APIs, stateless replicas, Redis hot state, Postgres durable records, object storage, per-session ordering/versioning, idempotency keys, GPU inference workers, queues, timeouts, and observability.

### Minutes 44–45: Close

Say:

> “The strongest part of Panopticon is not one model score. It is the experiment boundary: hidden truth is separated from observations, policies share one validated action contract, reward is separated from hard-gated evaluation, matched seeds protect comparisons, and raw model results are separated from supervisor results. The next step is to make the neural policy pass the advanced security gate without supervisor dependence, then harden session isolation and deployment.”

### If the Interviewer Interrupts

Welcome the interruption. Answer the question, then return with a bridge:

- “That connects to the turn lifecycle…”
- “The code path responsible for that is…”
- “The current implementation does X; the production design would do Y…”
- “The metric that verifies that claim is…”

A strong interview is a technical conversation, not a memorized monologue.

---
## 16. Quick Reference — Cheat Sheet

### Numbers That Matter

| ITEM | CURRENT VALUE | PRIMARY SOURCE |
|---|---:|---|
| Difficulty tiers | 5 | `environment.py`, `tasks/` |
| Game phases | 6 | `environment.py` |
| Sleeper generations | 5 | `models.py` |
| Core action types | 8 | `ActionType` |
| Sub-actions | 7 including `none` | `SubAction` |
| Numeric observation | 136 floats | `gym_wrapper.py` |
| Factored action space | `MultiDiscrete([8,12,7])` | `gym_wrapper.py` |
| Raw factor combinations | 448 | `8 * 8 * 7` |
| PPO parameters | 102,812 | layer arithmetic in `train_rl.py` |
| PPO rollout size | 128 steps | `NUM_STEPS` |
| PPO discount / GAE lambda | `0.99 / 0.95` | `train_rl.py` |
| Current GPU LLM default | Qwen2.5-1.5B-Instruct | `train_trl_v2.py` |
| LoRA rank / alpha | `16 / 32` | `train_trl_v2.py` |
| Current V5 expert episodes | 250 | checked-in V5 summary |
| Current V5 weighted examples | 88,896 | checked-in V5 summary |
| Reward schema | `security-first-v2` | `environment.py` |
| Grader schema | `security-gated-v2` | `grader.py` |
| Advanced hard gates | security >=90, catch=100%, false=0 | `grader.py` |
| Raw V5 macro grade | `0.701627` | `evaluation_comparison_latest.json` |
| Supervisor macro grade | `0.790471` | same summary |

A factor combination is not necessarily a semantically legal action. Validation remains authoritative.

### Core Invariants

1. An untrusted agent receives `EnvironmentObservation`, not `EnvironmentState`.
2. Hidden worker fields and false-flag truth are stripped before observation.
3. Every action passes validation before its handler.
4. Invalid actions consume time and receive the normal consequences plus `-1.0`.
5. Level-4/5 passing requires every hard security gate.
6. Expert training data is rejected if an episode violates its security gate.
7. Matched evaluation requires the same schema versions and seed plan.
8. Supervisor-assisted results must not be attributed to the raw model.

### Practical Playbook

```text
cover active departments with canaries
rotate monitoring instead of revealing one fixed routine
verify triggered canaries
correlate departments with unresolved leaks
audit recent/high-risk workers
interrogate high-suspicion workers before irreversible action
turn one useful confirmed sleeper on advanced levels when time remains
resolve confirmed threats before optional revenue/disinformation actions
preserve revenue after urgent security work
```

Interrogation reveals an armed dead switch but does not disarm it in the current engine.

### The Final Engineering Sentence

> Panopticon is a typed, partially observable, adversarial state machine with interchangeable policy interfaces, dense security-first reward, independent hard-gated grading, resumable curriculum training, and a clear path from a single demo process to versioned multi-session infrastructure.

---

## 17. The Current V5 Training and Evaluation Story

This section explains the latest checked-in evidence as an experiment rather than a marketing score.

### Experiment Identity

The selected run is named **Security-First V5** and was stored externally as `panopticon-security-v5-ep50`. The compact checked-in summaries and plots represent that run without committing multi-gigabyte evaluation files.

Key configuration:

| FIELD | VALUE |
|---|---|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Method | TRL supervised fine-tuning with LoRA |
| Curriculum | `easy -> medium -> hard -> level_4 -> level_5` |
| Expert episodes per level | 50 |
| Total expert episodes | 250 |
| Data seed | 42 |
| Max sequence length | 512 |
| Runtime profile | low-VRAM T4 |
| Reward schema | `security-first-v2` |
| Grader schema | `security-gated-v2` |

### Dataset Size

| LEVEL | EPISODES | WEIGHTED SUPERVISED EXAMPLES |
|---|---:|---:|
| Easy | 50 | 7,430 |
| Medium | 50 | 13,166 |
| Hard | 50 | 18,414 |
| Level 4 | 50 | 23,889 |
| Level 5 | 50 | 25,997 |
| **Total** | **250** | **88,896** |

The number of examples is larger than the number of turns because security-critical actions are duplicated according to training weight.

### Why Compare Several Policies?

A single score cannot answer all research questions.

| COMPARISON | QUESTION ANSWERED |
|---|---|
| Random | Is structured behavior better than arbitrary valid behavior? |
| Heuristic | Does learning beat a simple hand-coded baseline? |
| Base Qwen | Did fine-tuning improve the same underlying model family? |
| Raw V5 | What did the trained neural policy achieve? |
| Security-first supervisor | Can verified control logic solve the environment/gate? |

### Macro Result

| POLICY | EASY | MEDIUM | HARD | LEVEL 4 | LEVEL 5 | MACRO |
|---|---:|---:|---:|---:|---:|---:|
| Base Qwen | .630 | .671 | .616 | .657 | .631 | **.641110** |
| Raw V5 | .728 | .731 | .671 | .722 | .656 | **.701627** |
| Supervisor | .720 | .735 | .679 | .901 | .917 | **.790471** |
| Heuristic | .725 | .727 | .680 | .689 | .626 | **.689400** |
| Random | .631 | .666 | .650 | .636 | .654 | **.647400** |

The raw V5 model improved macro grade by:

```text
0.701627 - 0.641110 = 0.060517 absolute grade points
```

Relative improvement over base is approximately:

```text
0.060517 / 0.641110 ≈ 9.44%
```

Absolute and relative improvement are different. Saying “about 9.4% relative improvement” is accurate; saying “accuracy improved by 9.4 percentage points” is not.

### Operational Summary

| POLICY | MACRO GRADE | MEAN REWARD | MEAN REVENUE | MEAN SECURITY | MEAN CAUGHT |
|---|---:|---:|---:|---:|---:|
| Supervisor | .790471 | 23.328 | 571.58 | 100.00 | 3.00 |
| Raw V5 | .701627 | 8.204 | 483.90 | 89.26 | 2.68 |
| Heuristic | .689400 | 10.526 | 612.36 | 83.54 | 2.47 |
| Base Qwen | .641110 | 7.644 | 448.16 | 95.96 | 2.87 |
| Random | .647400 | -25.242 | 216.50 | 69.26 | 2.75 |

These are macro summaries across levels, not a claim that every episode had those values.

### Why Raw V5 Failed

The strict acceptance report identified nine failures:

```text
Level 4 pass rate:            0.50, required 1.00
Level 4 security:            85.85, required >= 95.05 matched base
Level 4 caught sleepers:      3.50, required >= 3.85 matched base
Level 4 missed sleepers:      0.50, required 0

Level 5 pass rate:            0.05, required 1.00
Level 5 security:            60.47, required >= 84.81 matched base
Level 5 caught sleepers:      3.90, required >= 4.50 matched base
Level 5 missed sleepers:      1.10, required 0
Level 5 false accusations:    0.10, required 0
```

Note that the acceptance gate requires candidate security/caught metrics to be no worse than the **matched base**, which can be stricter than the grader's absolute `90` security gate depending on the base result.

### Why a Better Grade Can Still Be Rejected

Raw V5 improves the weighted grade at every level, but release requires operational constraints too. This is the central evaluation lesson:

> Average quality improvement is not sufficient evidence for a safety-critical release.

A 5% Level-5 pass rate means 19 of 20 matched episodes failed the grade's complete pass rule, even though the mean Level-5 composite rose from `.631` to `.656`.

### What the Supervisor Proves

The supervisor's advanced outcomes include:

- 100% pass rate;
- 100 final security;
- every sleeper caught;
- zero missed sleepers; and
- zero false accusations.

It proves a safe observable-state control path exists. It also supplies an upper operational reference for what a learned policy should approach.

It does not prove that raw V5 reasoning caused those supervisor decisions.

### Threats to Validity

A **threat to validity** is something that could weaken an experimental conclusion.

1. Twenty episodes per policy/level may not capture rare seeds.
2. Repair logic can change raw generated actions.
3. Macro averaging gives every level equal weight but hides within-level variance.
4. The expert and evaluator share environment assumptions; both may miss a modeling flaw.
5. The reported scripted HYDRA and unvalidated neural HYDRA may not represent adaptive human adversaries or real attacks.
6. Demonstration duplication changes action frequency but not state diversity.
7. Compact repository summaries cannot replace raw episode-level forensic data.

### The Next Experiment

A disciplined V6 study would:

1. classify every raw Level-4/5 failure by action error type;
2. measure parse, repair, supervisor, and fallback rates separately;
3. add more diverse advanced-tier expert states rather than only duplicating actions;
4. include hard-negative demonstrations showing tempting but unsafe actions;
5. evaluate raw, repair-only, and supervisor policies independently;
6. increase held-out seed coverage;
7. use confidence intervals or bootstrap intervals; and
8. accept a model only through the unchanged fail-closed gate.

A **bootstrap interval** repeatedly resamples observed episodes to estimate uncertainty without assuming a particular score distribution.

---

## 18. Beginner Glossary

This glossary favors plain meaning over textbook compression. Terms are explained in the sense used by this repository.

### A–M

#### Activation function

A function applied between neural-network layers so the whole network can represent nonlinear relationships. The PPO backbone uses ReLU, which returns zero for negative inputs and the input itself for positive inputs.

#### Actor

The policy-producing part of an actor-critic model. In `PanopticonAgent`, three actor heads choose action type, target, and sub-action.

#### Actor-critic

A reinforcement-learning design with an actor that chooses actions and a critic that estimates how promising a state is. PPO trains both together.

#### Adapter

A small set of trainable weights attached to a larger frozen or mostly frozen model. LoRA produces adapters; they can later be merged into the base model.

#### Advantage

An estimate of how much better an action was than the critic expected in that state. Positive advantage increases the action's probability; negative advantage tends to decrease it.

#### Agent

A decision maker that receives observations and chooses actions. ARGUS is the agent role; random, heuristic, security-first, PPO, and LLM policies can all control it.

#### API

Application Programming Interface: a defined way for software components to communicate. Panopticon's HTTP API exposes reset, step, tasks, grading, schemas, and other operations.

#### ASGI

Asynchronous Server Gateway Interface: the standard connection between Python async web applications and servers such as Uvicorn.

#### Async / asynchronous

A style where a task can pause while waiting so other work may proceed. `async def` enables this model, but synchronous model generation inside an async function can still block.

#### Atomic operation

An operation observed as all-or-nothing. `os.replace` is used for atomic file replacement; database transactions provide stronger multi-operation atomicity.

#### Audit

In the game, an investigation of a worker. It may raise suspicion for an active sleeper but has generation-dependent detection probability.

#### Authentication

Proving identity: “Who are you?” A production multi-user API needs authentication before session access.

#### Authorization

Checking permission: “May this identity read or modify this session?” Authentication without authorization is not sufficient isolation.

#### Backpropagation

The algorithm that computes how each trainable parameter contributed to loss, working backward through the neural network using the chain rule.

#### Backpressure

A system's method for slowing, buffering, or rejecting incoming work when consumers cannot keep up. Bounded inference queues need backpressure.

#### Baseline

A comparison policy used to interpret results. Random, heuristic, and untrained base Qwen are different baselines.

#### Batch and micro-batch

A batch is a group of examples processed for one optimization update. A micro-batch is the smaller group that fits in memory before gradient accumulation combines several passes.

#### Benchmark

A repeatable evaluation procedure with named configurations and metrics. Fair benchmarks use matched levels, seeds, schemas, and settings.

#### Bias and variance

Bias is systematic error from simplifying assumptions; variance is sensitivity to the sampled data or randomness. GAE's lambda trades some bias against variance in advantage estimates.

#### Binary search

An algorithm that halves a sorted search range each step. The trainer uses it to find the largest observation token budget that fits the context window.

#### Boolean

A value that is either `True` or `False`, such as `is_sleeper`, `verified`, `done`, or `active`.

#### Bottleneck

The component limiting overall throughput. At scale, LLM inference is likely a larger bottleneck than one pure-Python environment step, but this must be measured.

#### Canary trap

A unique marked piece of information placed in one department. If its fingerprint appears in a leak, ARGUS has stronger evidence about the leak's origin.

#### Categorical distribution

A probability distribution across a finite set of choices. Each PPO actor head creates one from logits and samples an index.

#### Checkpoint

Saved training state used for resumption or comparison. A useful checkpoint may include model weights, optimizer state, scheduler state, and step/epoch counters.

#### Class

A Python blueprint for objects. `Worker`, `Environment`, and `TaskGrader` are classes.

#### Clamp

Restrict a number to a minimum and maximum. `clamp(x,-1,1)` returns `-1` for smaller values, `1` for larger values, and `x` inside the range.

#### Client and server

A client initiates a request; a server listens and responds. `client.py` or a dashboard can call the FastAPI server.

#### Completion-only loss

Language-model loss calculated only on the assistant answer rather than the system prompt and user observation. Masked label positions use `-100`.

#### Composite score

A weighted combination of multiple dimension scores. It gives one summary but does not replace hard safety gates.

#### Concurrency

Multiple tasks making progress during the same time period. Concurrent requests can race even on one process if they interleave around awaits or worker threads.

#### Confidence interval

A range estimating uncertainty around a metric. More evaluation episodes and bootstrap resampling can produce more informative intervals than one mean.

#### Configuration

Values that change behavior without changing core logic: level, seed, model path, sequence length, or port.

#### Consistency

Rules about what values different readers can observe and in what order. Per-session step processing needs strong ordering consistency.

#### Constraint

A non-negotiable requirement. Level-5 security, catch-rate, and false-accusation gates are constraints; other grade dimensions are optimization objectives.

#### Context window

The maximum number of tokens an LLM can process in one input/example. Long observations must be compacted without losing the action label.

#### CORS

Cross-Origin Resource Sharing: browser rules and server headers controlling whether one website origin may call another origin.

#### Correlation

A relationship between signals. The game `correlate` action links leaks and active sleepers inside a department; statistical correlation outside the game does not by itself prove causation.

#### CPU

Central Processing Unit: general-purpose processor. The environment and API run well on CPU; LLM training/inference is generally much faster on suitable GPUs.

#### Critic

The value-estimating part of actor-critic. It predicts expected future return from an observation and helps compute advantage.

#### Curriculum learning

Training from easier tasks to harder tasks. Panopticon chains five tiers so mechanics are introduced progressively.

#### CUDA

NVIDIA's GPU programming platform. PyTorch uses it to execute tensor operations on supported NVIDIA GPUs.

#### Data collator

Code that assembles tokenized examples into padded training batches and labels. Panopticon uses a completion-only collator when the installed TRL version provides it.

#### Data leakage / label leakage

Accidentally giving training or evaluation information that would be unavailable in real use. An expert choosing actions from hidden state would create label leakage.

#### Dead-man's switch

A mechanism that triggers damage if an armed sleeper is terminated. Interrogation reveals it; the current engine does not include a disarm action.

#### Delta

A change, usually `after - before`. Reward uses revenue and security deltas.

#### Dependency

External software required by the project, such as Pydantic or PyTorch. Dependency versions affect reproducibility and security.

#### Deployment

Packaging and running software in an environment accessible to its users or evaluators.

#### Deterministic

Expected to produce the same result given the same inputs, state, and compatible runtime. Greedy LLM decoding is more deterministic than temperature sampling.

#### Dictionary (`dict`)

A Python mapping from keys to values. Episode `info`, HYDRA audit counts, and serialized JSON objects often use dictionaries.

#### Discount factor (`gamma`)

The amount future rewards are valued relative to immediate reward. PPO uses `0.99`, so distant reward matters but is gradually discounted.

#### Distribution shift

A difference between training examples and real evaluation inputs. Rare advanced-tier states can create distribution shift for an SFT model.

#### Docker image and container

An image is a built filesystem/configuration template. A container is a running isolated process created from that image.

#### Dropout

Randomly disabling some activations or adapter paths during training to reduce overfitting. LoRA dropout is `0.05`.

#### Dtype

Data type used for tensor numbers, such as float32, float16, or bfloat16. It affects precision, speed, and memory.

#### Dense reward

Reward delivered frequently during an episode, rather than only at the end. Panopticon returns a number every turn.

#### Embedding

A learned numeric vector representing an item or token. LLM token IDs are converted into embeddings before transformer layers process them.

#### Entropy

A measure of uncertainty in a probability distribution. PPO adds an entropy bonus so the policy does not become deterministic too early.

#### Enumeration / enum

A named finite list of legal values. `ActionType` prevents arbitrary action names from being accepted as valid categories.

#### Episode

One play-through from reset to terminal/truncated completion.

#### Epoch

One complete pass over a training dataset.

#### Event loop

A runtime loop that schedules asynchronous tasks and resumes them when awaited work is ready.

#### Event sourcing

Persisting state changes as an append-only event log, then reconstructing state by replay. It is a proposed scale/recovery design, not current session persistence.

#### Expert trajectory

An episode or sequence of observation-action examples produced by a trusted demonstration policy.

#### Facade

A simple interface hiding a more complex subsystem. The FastAPI layer is a facade around engine and grader behavior.

#### Fail-closed

Reject or stop when a required check is missing or fails. Expert-data generation and benchmark acceptance use this approach.

#### False flag

Evidence planted to frame an innocent worker. The truth flag is hidden from observations.

#### FastAPI

The Python framework that defines Panopticon's HTTP routes and integrates Pydantic/OpenAPI.

#### Feature

One numeric input attribute supplied to a model. Worker suspicion and normalized security are PPO features.

#### Fine-tuning

Continuing training from an existing pretrained model for a narrower task. Panopticon fine-tunes Qwen to output actions.

#### Float

A number with a fractional part represented in finite binary precision, such as `0.75`.

#### Forward pass

Running input through a model to produce logits, values, or generated tokens before gradients are computed.

#### Function and method

A function is named reusable behavior. A method is a function attached to a class/object, such as `Environment.step`.

#### GAE

Generalized Advantage Estimation: a backward calculation combining temporal-difference errors to estimate advantage with a bias/variance tradeoff.

#### Game theory

The study of strategic interaction among decision makers. Panopticon has adversarial and asymmetric-information elements, but HYDRA is scripted rather than independently learning.

#### Gate

A pass condition that must be satisfied. A hard gate cannot be compensated for by a higher score elsewhere.

#### GPU

Graphics Processing Unit: a massively parallel processor well suited to matrix/tensor operations used by neural networks.

#### Gradient

The derivative of loss with respect to a parameter. It points toward the local direction of greatest loss increase; optimizers move in the opposite direction.

#### Gradient accumulation

Adding gradients from several micro-batches before an optimizer step to simulate a larger batch.

#### Gradient checkpointing

Recomputing selected forward activations during backward pass to save memory.

#### Gradient clipping

Limiting gradient norm to reduce unstable updates. PPO clips the total gradient norm at `0.5`.

#### Hash

A deterministic function mapping input bytes to a fixed-size fingerprint. Game canary hashes are matching tokens, not passwords or cryptographic authentication.

#### Health check

A lightweight request showing whether a service process responds. Readiness and deep dependency checks may require additional endpoints.

#### Heuristic

A practical hand-written rule that often works but is not guaranteed globally optimal.

#### Hidden state

Privileged environment truth, such as sleeper identity and dead-switch status. It must not be confused with a neural network's internal hidden representation.

#### HTTP

Hypertext Transfer Protocol: the request/response protocol used by the API.

#### Idempotency

The property that retrying the same identified operation does not apply its effect twice.

#### Imitation learning

Learning behavior from demonstrations. SFT on expert actions is imitation learning.

#### Index

In a database, a lookup structure that speeds queries. In an array/list, an integer position such as worker target index.

#### Inference

Using a trained model to choose or generate output without updating its parameters.

#### Invariant

A property intended to remain true, such as hidden identity never appearing in agent observations.

#### JSON

JavaScript Object Notation: a language-neutral text format for objects, arrays, strings, numbers, booleans, and null.

#### JSONL

JSON Lines: one JSON object per line, useful for streaming datasets and logs.

#### Label

The desired output used to compute supervised loss. Here it is the assistant's expert JSON action tokens.

#### Latency

Time from request start to response completion. Tail percentiles matter because a small number of very slow generations harm users.

#### Layer

A model transformation from one representation to another. PPO uses linear and ReLU layers; Qwen uses transformer blocks.

#### Learning rate

The scale of optimizer parameter updates. Too high can diverge; too low can learn too slowly.

#### LLM

Large Language Model: a neural model trained to predict text tokens. “Large” is relative; the current base has roughly 1.5 billion parameters.

#### Lock and lease

A lock gives one worker exclusive access to a resource. A lease is a lock that expires unless renewed, reducing permanent deadlock after crashes.

#### Log probability

The natural logarithm of probability. Joint independent probabilities multiply; their log probabilities add, which is numerically convenient.

#### Logit

An unnormalized score produced before softmax converts scores to probabilities.

#### LoRA

Low-Rank Adaptation: parameter-efficient fine-tuning using small low-rank matrix updates attached to selected base-model modules.

#### Loss function

A differentiable number training tries to minimize. PPO combines policy, value, and entropy terms; SFT uses token cross-entropy.

#### Macro average

An average giving each group equal weight. The V5 macro grade averages the five level means.

#### Mask

A boolean or sentinel structure disabling choices or loss positions. The Gym action mask is partial; the SFT collator masks prompt labels.

#### Mean

Arithmetic average: sum divided by count.

#### Metadata

Data describing other data or an artifact, such as schema version, model reference, seed, task, or source commit.

#### Metric

A measured numeric indicator. A metric is only useful when its definition and experimental conditions are clear.

#### Model

A parameterized mathematical function. It can be a PPO network or an LLM.

#### Module

A Python source file that can be imported, or a neural-network component. Context determines which meaning is intended.

#### MultiDiscrete

A Gymnasium space made of several discrete choices. Panopticon uses three categorical dimensions `[8,12,7]`.

#### Multi-tenant

Serving multiple independent users/organizations while keeping their data and state isolated. The current singleton server is not multi-tenant safe.

### N–Z

#### Neural network

A parameterized stack of mathematical transformations learned from data. The PPO agent is a small multilayer perceptron; Qwen is a much larger transformer network.

#### Node / instance / replica

A node is a machine or runtime participant. An instance is one running copy of a service. Replicas are equivalent service copies used for capacity and availability.

#### Normalization

Rescaling values to comparable ranges. The Gym wrapper divides security by 100 and turn by maximum turns; PPO normalizes minibatch advantages.

#### NOOP

“No operation”: intentionally do nothing. It is always a legal action and also a last-resort parse fallback in the simpler trainer.

#### Object

A runtime value created from a class and containing fields/behavior. `Environment(seed=42)` creates an object.

#### Objective

What an optimizer tries to improve. Reward is a training objective; safety gates are constraints; the grader is an evaluation objective.

#### Observation

The information made available to the policy. It is a filtered projection of full state.

#### Observability

How much internal truth an agent can see. Panopticon is partially observable because sleeper identities and HYDRA memory are hidden.

#### Observability in operations

A different meaning: logs, metrics, traces, and alerts that let engineers understand a running service. Context distinguishes it from environment observability.

#### OpenAPI

A standard schema describing HTTP routes, parameters, bodies, and responses. FastAPI generates it from routes and Pydantic models.

#### OpenEnv

The environment interface/metadata ecosystem targeted by this project. `openenv.yaml`, tasks, reset/step, schemas, and programmatic graders support that integration.

#### Optimizer

An algorithm that updates trainable parameters using gradients. Native PPO uses Adam; Hugging Face training config selects its optimizer machinery through the trainer.

#### Orthogonal initialization

Initializing weight matrices so directions are initially well-conditioned. PPO applies it with different output scales for actor and critic layers.

#### Overfitting

Learning training examples too specifically and failing to generalize to new states or seeds. Evaluation must use held-out scenarios.

#### Parameter

A learned numeric value inside a model. Hyperparameters such as learning rate are configured by developers; parameters are learned.

#### Partial observability

The agent cannot directly see all state needed for a perfect decision and may need evidence/history to infer hidden facts.

#### PEP / Python version

A PEP is a Python Enhancement Proposal documenting language conventions/features. The project requires Python 3.11 or newer in `pyproject.toml`.

#### PEFT

Parameter-Efficient Fine-Tuning: techniques and a library ecosystem for adapting large models with relatively few trainable parameters. LoRA is one PEFT method.

#### Percentile

A distribution position. p95 latency is the value at or below which 95% of requests complete.

#### Persistence

Keeping data beyond the current process lifetime. The current API episode is not persistent; training checkpoints and evaluation files are.

#### Policy

A mapping from observation to action. Policies may be random, rule-based, or learned.

#### PPO

Proximal Policy Optimization: an on-policy actor-critic algorithm that uses a clipped probability-ratio objective to limit overly large policy changes.

#### Precision

The numeric representation detail of a value. Float16/bfloat16 reduce memory versus float32 but have different numeric limitations.

#### Pretrained model

A model already trained on broad data before task-specific adaptation. Qwen2.5-Instruct is the starting pretrained model.

#### Probability

A number from zero to one describing uncertainty. A value of `0.6` means 60% under the model/rule, not certainty about one outcome.

#### Prompt

Text/instructions supplied to an LLM. It includes a system role, formatted current state, and request for one JSON action.

#### Prompt injection

Untrusted text attempting to alter model instructions. Current game strings are mostly controlled, but a production environment with user-provided content must treat it as data and validate all output.

#### Pydantic

A Python library for typed data validation, defaults, serialization, and JSON Schema generation.

#### Queue

A buffer of pending work processed by consumers. Inference queues separate request intake from GPU workers.

#### Race condition

Behavior that incorrectly depends on timing/order of concurrent operations. Two steps reading the same session version can cause a lost update.

#### Random number generator

Code producing a deterministic pseudo-random sequence from internal state. A seed initializes that state.

#### Rate limiting

Restricting operations per identity/time window to protect capacity, cost, and abuse boundaries.

#### ReLU

Rectified Linear Unit: `max(0,x)`, the activation used in the PPO network.

#### Regression test

A test protecting behavior that previously worked. The security regression checks 100 seeded expert episodes.

#### Regularization

Methods discouraging brittle memorization, such as dropout, limited adapter rank, data diversity, or weight decay.

#### Replay / event replay

Re-applying recorded events to reconstruct state or inspect behavior. It differs from an RL replay buffer; PPO here is on-policy and does not use experience replay.

#### Replica

One of several running copies of a service behind routing/load balancing.

#### REST

Representational State Transfer: common HTTP resource conventions. The Panopticon API is REST-like, although step is a command operation rather than pure resource replacement.

#### Retry

Attempting an operation again after failure. Mutating retries require idempotency to avoid duplicate steps.

#### Return

In RL, discounted cumulative future reward. In Python, the value a function gives back. Context matters.

#### Reward

Immediate scalar feedback from an environment transition. It is a learning signal, not the full final grade.

#### Reward shaping

Adding intermediate reward/penalty terms to guide learning toward long-horizon behavior.

#### RL

Reinforcement Learning: learning a policy through interaction and reward rather than only labeled answers.

#### Rollout

A sequence of policy-environment interactions collected before an update. Native PPO collects 128 steps per rollout.

#### Route / endpoint

A method-plus-path handled by an API, for example `POST /step`.

#### Schema

A description of valid data shapes, types, and constraints. A schema version identifies behavior contracts across experiments.

#### Seed plan

The exact ordered seeds assigned to evaluation agents and levels. Matching it prevents scenario difficulty from confounding comparisons.

#### Serialization

Converting an in-memory object to a storable/transmittable form such as JSON; deserialization reconstructs data from that form.

#### Session

One user's isolated server-side interaction context. Session-scoped environments are proposed; the current API has one global environment.

#### SFT

Supervised Fine-Tuning: training a pretrained model on input/desired-output examples.

#### Singleton

A design using one shared instance. `_server.py` has one global `Environment` per process.

#### Sleeper

A hidden infiltrator controlled by HYDRA logic. It begins dormant, activates later, and may leak or use advanced mechanics.

#### Softmax

A function converting logits into positive probabilities summing to one. PyTorch's categorical distribution effectively uses it.

#### State

All information required to continue the simulated world. Full state includes facts the agent is not allowed to observe.

#### State machine

A system with defined states and legal transitions triggered by events/actions. Worker and episode lifecycle behavior form state machines.

#### Stateless service

A service instance that does not depend on its local memory for durable client state. Shared stores enable any replica to handle a request.

#### Stochastic

Including randomness. Sleeper placement, activation delays, audit detection, and sampled model outputs can be stochastic.

#### Standard deviation

A measure of spread around a mean. Zero means all values are identical; larger values mean greater variation.

#### Sub-action

A modifier refining an action family: audit/verify/correlate under investigate; terminate/interrogate/turn under neutralize.

#### Supervisor

A verified controller that can select or repair actions around a neural model. Its contribution must be measured separately from raw-model behavior.

#### Tensor

A multidimensional numeric array used by PyTorch. Scalars are 0-D tensors, vectors 1-D, matrices 2-D, and batches add dimensions.

#### Terminal condition

A rule ending an episode, such as maximum turns, bankruptcy, or total breach.

#### Throughput

Amount of work completed per unit time, such as steps or generated tokens per second.

#### Timeout

A deadline after which a waiting operation is treated as failed or incomplete.

#### Token

A tokenizer's unit of text. Token counts, not character counts, determine LLM context use.

#### Tokenizer

Code mapping text to token IDs and IDs back to text, while handling model-specific special tokens/templates.

#### Top-p sampling

Generate only from the smallest high-probability token set whose cumulative probability reaches `p`, then sample within it. It controls generation diversity.

#### Trace

Operationally, a record connecting work across services. In `PolicyDecision`, prompt/messages/raw text form a decision trace, though the project does not implement distributed tracing.

#### Training, validation, and test sets

Training data updates parameters. Validation data guides choices during development. Test data is held back for final unbiased evaluation. Seed separation should reflect these roles.

#### Trajectory

An ordered episode sequence of observations, actions, rewards, and transitions.

#### Transaction

A group of database changes that commit together or roll back together, often with isolation from concurrent work.

#### Transformer

A neural architecture built around self-attention and feed-forward blocks. Qwen is a causal language-model transformer.

#### Truncation

Episode ending due to an external horizon/time limit rather than the environment's natural terminal failure. Gymnasium separates `terminated` and `truncated`.

#### TTL

Time To Live: an expiry duration after which cached/session data is automatically deleted.

#### Type

A category of value and allowed operations, such as `int`, `float`, `str`, `bool`, list, or a Pydantic model.

#### Unit test

A focused test of one small behavior in isolation, such as one validation rule or reward term.

#### Uvicorn

An ASGI server that runs the FastAPI application and accepts network connections.

#### Validation

Checking data/actions against rules before use. Pydantic validates shape; `validate_action` validates current-state semantics.

#### Value function

The critic's estimate of expected discounted return from a state/observation.

#### Variance

Average squared spread around a mean in statistics; in ML discussions it also describes sensitivity to sampled training data.

#### Versioning

Attaching explicit identities to schemas, rewards, graders, models, datasets, and APIs so incompatible artifacts are not silently compared.

#### VRAM

Memory attached to a GPU, used for model weights, optimizer state, gradients, and activations.

#### Warmup

A schedule that begins with a smaller learning rate and increases it during early training steps.

#### WebSocket

A persistent two-way browser/server connection. It is a future option for job completion or interactive updates; the current core API is HTTP request/response.

#### Weight

A learned model parameter; also a configured coefficient such as the grader's `0.40` security weight. The two meanings are related mathematically but used at different layers.

#### Worker

In the game, an employee entity. In infrastructure, a process consuming queued jobs. Say “game worker” or “inference worker” when ambiguity matters.

#### YAML

A human-readable configuration format. `openenv.yaml` declares environment metadata and tasks.

#### Zero-sum

A game where one player's gain exactly equals the other's loss. Panopticon is adversarial in intent, but its implemented reward is calculated only for ARGUS; there is no explicit HYDRA payoff proving a strict mathematical zero-sum formulation.

---

## 19. Algorithm and Concept Implementation Sheet

This section maps interview concepts to concrete code paths. `W`, `L`, `C`, `R`, and `D` mean counts of workers, leaks, canaries, reports, and double agents. Big-O describes how work grows as input size grows; it normally ignores small constant factors.

### 19.1 Episode Reset

**Code:** `Environment.reset()`  
**Input:** task level and optional seed  
**Output:** first safe observation

```text
select difficulty through LEVEL_MAP
reset counters and pending sleeper schedule
seed or recreate random generator
shuffle worker codenames
create configured initial workers across active departments
initialize revenue/security to 100
initialize phase 1 and empty histories/assets
return get_observation()
```

Complexity is `O(W)` for worker creation. A supplied seed repeats setup; no seed creates a fresh generator.

### 19.2 Hidden-State Projection

**Code:** `Environment.get_observation()`  
**Goal:** preserve public evidence while removing privileged truth.

For every worker, it copies public fields but forces:

```text
hidden_state=clean
is_sleeper=False
generation=0
cover_integrity=1
activation/cooldown/false-target/dead-switch = safe defaults
```

Leaks are copied with source visible only under the implemented verified-canary condition, and `is_false_flag` is forced false. HYDRA memory never enters the observation.

Complexity is `O(W + L + C + R + D)`. The security invariant is more important than speed: new hidden fields added to `Worker` require explicit review or they may leak through careless `model_copy`/serialization.

### 19.3 Action Validation

**Code:** `validate_action(action, obs)`

```text
parse action type enum
noop -> valid
work -> active scenario department with live worker
hire -> active department containing a terminated slot
canary -> active scenario department
monitor -> valid channel or empty
investigate -> valid sub-action and compatible worker/leak/department target
neutralize -> valid sub-action and unresolved worker target
deploy_double -> active asset target
```

Validation uses only observation, not hidden identity. That keeps legality independent from secrets.

Worst-case lookup is generally `O(W + L + D)` because lists are scanned. At current small sizes this is appropriate; at large entity counts, build ID maps once per observation.

### 19.4 One Step as a Transaction-Like Sequence

**Code:** `Environment.step()`

```text
validate
snapshot before metrics
apply ARGUS action (or invalid penalty path)
run HYDRA
auto-progress turnings
run economy tick
compute reward from before/after
increment turn
update phase and histories
check max-turn, bankruptcy, breach
count missed sleepers if done
return safe observation + reward + flags + info
```

It is transaction-like because order matters, but it is not a rollback transaction: a Python exception partway through could leave partial in-memory mutation. Production hardening can copy state, apply to a candidate, validate invariants, then commit the candidate.

### 19.5 Action Dispatch

**Code:** `_process_action()`

This is a dispatcher: it maps one `ActionType` to one handler. Dispatch avoids one giant method and gives each action a focused test surface.

Adding a ninth action requires synchronized changes to the enum, validation, dispatcher, Gym mapping/space, system prompts, expert policy, inference repair, schemas, tests, and documentation.

### 19.6 Work and Hire

`work` filters active non-turned workers in a department and adds `2.5 * sum(performance)` revenue. Complexity is `O(W)`.

`hire` finds a terminated worker in a department, removes the first, creates a new clean worker, and costs `3` revenue. The list rebuild is `O(W)`.

The validation requirement for a terminated slot prevents unlimited workforce growth through `hire`.

### 19.7 Canary Creation and Monitoring

`canary` increments a counter and hashes counter, turn, and department into an eight-hex-character fingerprint. It appends a trap and costs `1` revenue.

`monitor` scans unresolved leaks on the selected channel and active canaries for exact hash equality:

```text
for leak in leaks:
  for trap in canaries:
    if active and same hash:
      mark leak as canary
      connect canary ID
      mark trap triggered
```

Complexity is `O(L*C)`. A dictionary `hash -> trap` could reduce matching to `O(L+C)` if lists became large.

MD5 here supplies deterministic short fingerprints, not collision-resistant security. An eight-character hex prefix has only 32 bits, so a production tracking system should use longer random identifiers and uniqueness constraints.

### 19.8 Audit Detection

For an active sleeper:

```text
p = 0.6 - 0.1 * (generation - 1)
if generation >= 3: p = 0.7 * p
sample rng.random() < p
```

This yields 60%, 50%, 28%, 21%, and 14% for generations 1–5. On detection, suspicion rises `0.4` and report confidence is `0.7`.

Probability is not confidence. Detection probability describes whether the branch fires; report confidence is a game field assigned after detection.

### 19.9 Verify and Correlate

`verify` looks up one leak. A monitored canary match can identify an active sleeper in the trap department, increase suspicion `0.5`, lower cover `0.3`, and assign report confidence `0.85`.

`correlate` finds active sleepers and leaks in one department; when both exist, every such sleeper gains `0.3` suspicion and the report gets `0.65` confidence.

The current correlate handler uses hidden state to update public suspicion. That is acceptable as simulated investigation outcome logic, provided it reveals only the designed signal rather than raw truth.

### 19.10 False Flags

An active Gen-3+ sleeper has a 30% per-turn chance to plant its first false flag if an innocent same-department target exists.

The target gains `0.2` suspicion, and a fake leak points internally to that innocent worker. Observations hide `is_false_flag`, so the agent must verify rather than inspect a boolean.

### 19.11 Dead Switch

An active Gen-4+ sleeper arms after `turn - activation_turn > 5`. Its hidden state becomes `dead_switch`.

Termination while armed subtracts `15` revenue and `10` security before the normal caught-sleeper recovery. The current engine has no disarm transition.

### 19.12 Turning and Double-Agent Deployment

`turn` starts a four-turn countdown. `_progress_turnings()` decrements it automatically every environment step. At zero, the sleeper becomes `double_agent`, a `DoubleAgentAsset` is created, and the sleeper counts as caught.

Deployment:

```text
effectiveness = asset.effectiveness * max(0.3, hydra_trust)
hydra_trust -= 0.1
recruitment_accuracy *= (1 - 0.15 * effectiveness)
security += 5 * effectiveness
revenue += 3 * effectiveness
burn asset when trust <= 0.2
```

Floating-point subtraction means interview answers should say “approximately eight deployments from trust 1.0 until the threshold,” then verify exact behavior in tests rather than rely only on decimal intuition.

### 19.13 HYDRA Turn

**Code:** `_hydra_turn()`

```text
spawn scheduled sleeper
activate due dormant sleepers
leak from ready active sleepers
attempt Gen-3 false flags
arm Gen-4 dead switches
update counter-strategy labels
apply passive active-sleeper damage
```

The dominant loops are linear in workers, with leak/canary selection elsewhere. Current entity sizes are bounded and small.

### 19.14 HYDRA Adaptation

Placement for Gen-2+ avoids remembered canary departments when alternatives exist, then sorts departments by audit count and randomly chooses among the remaining list. Leak selection avoids remembered monitored channels when alternatives exist.

The named counter strategies `canary_aware`, `channel_rotation`, and `deep_cover` are recorded when thresholds are reached. In current code, placement/channel behavior already consults memory directly; the labels themselves do not gate all those behaviors, and `deep_cover` is recorded but does not visibly modify cover in `_spawn_sleeper`. Describe labels as telemetry/state, not as proof of additional effects.

### 19.15 Reward

**Code:** `_compute_reward()`

```text
base = .45*clamp(revenue_delta/15) + .55*clamp(security_delta/20)
add .75 per newly caught sleeper
subtract 1 per new false accusation
subtract .03 per active threat
subtract security deficit below 90, capped .45
add phase-6 double-agent surge if eligible
subtract .02 time cost
```

Reward complexity is `O(W + D)` because it counts active threats and assets.

### 19.16 Grade

**Code:** `TaskGrader.grade()`

```text
extract episode metrics
compute five dimension scores
weighted sum and clamp
calculate catch rate
check composite threshold and hard gates
return GraderResult with dimensions/details
```

Grading is roughly `O(H)`, where `H` is recent revenue history (only the last 20 values are used for standard deviation), plus constant-time counters already accumulated by the engine.

### 19.17 Observation Flattening

**Code:** `OpenEnvGymWrapper._flatten_obs()`

A zero-filled vector allocates fixed slots:

```text
12 workers * 6
8 leaks * 4
6 canaries * 3
3 double agents * 3
5 global values
= 136 floats
```

Only the first capacity-limited entities are encoded. This creates truncation: later workers/leaks/canaries/assets can be absent from PPO input even though they exist in structured observation. Capacity assumptions should be tested against actual maximum trajectories.

### 19.18 Numeric Action Decoding

**Code:** `_decode_action([type,target,sub])`

The meaning of target index depends on action type. Invalid sub-actions for investigate/neutralize are replaced with defaults; out-of-range worker indices can fall back to the first worker.

This makes every numeric triple decodable, but not necessarily legal. The environment validator remains the final authority.

### 19.19 PPO Forward Pass

```text
136 observation
 -> shared 256/ReLU -> 128/ReLU features
 -> action-type head -> 8 logits
 -> target head      -> 8 logits
 -> sub-action head  -> 7 logits
 -> critic head      -> 1 value
```

Each actor head creates a categorical distribution. Joint log probability and entropy are sums across heads, an independence factorization conditioned on shared features.

### 19.20 PPO Rollout, GAE, and Update

The trainer collects 128 transitions. Backward GAE computes temporal-difference residuals and advantages. Four update epochs shuffle indices into minibatches of 32.

PPO computes old/new log-probability ratio, clips it to `[0.8,1.2]` inside one surrogate, uses the pessimistic larger loss, adds half the value MSE, subtracts `0.01` entropy, clips gradients, and steps Adam.

This is on-policy learning: old rollout data should not be reused indefinitely after the policy changes.

### 19.21 Security-First Expert

**Code:** `choose_security_first_action()`

The algorithm is priority ordered and stateful. It remembers audit/monitor indices, canaried departments, and whether it already chose a turn operation. Complexity is mostly `O(W + L + C + D)` per decision.

The policy deliberately spends early turns on coverage and later alternates evidence, enforcement, and productivity.

### 19.22 Dataset Gate and Weighting

Each expert episode is graded and checked for security >=90, all caught, none missed, and no false accusations. Failure aborts generation.

Each accepted observation-action pair is rendered through the tokenizer's chat template and duplicated by action weight. Data metadata records level, episode count, example count, model, seed, and trajectory schema so reuse is safe only under an exact match.

### 19.23 Context-Fitting Algorithm

```text
render full example
if too long: compact repeated lines
if still too long:
  binary-search observation token budget
  keep about 62% from head and 38% from tail
if still too long: retain tiny first-line summary
return text, token length, compacted flag
```

Head/tail preservation tries to retain global context and recent critical evidence. It is heuristic; targeted structured summarization could be stronger.

### 19.24 Curriculum Resume

Load `curriculum_state.json`. If schema/profile/seed changed, reset completed-level tracking. Otherwise skip completed levels. For the current level, reuse data only when metadata matches and resume the latest trainer checkpoint. After completion, atomically save progress and carry the model into the next level.

This separates curriculum-level progress from trainer-level checkpoint progress.

### 19.25 Action Parsing and Repair

The local path parses JSON, normalizes aliases/IDs/sub-actions, recovers intent from text using regular expressions, prioritizes urgent safe actions, validates candidates, and finally chooses a safe fallback.

A regular expression is a pattern language for matching text. Here it recognizes IDs such as `w-003` and `leak-012`.

Repair improves availability but should emit telemetry: original action, repaired action, reason, and whether the raw action was valid.

### 19.26 Evaluation Summary

For each level, gather episode metrics, compute arithmetic mean and population standard deviation, calculate pass rate, and summarize every grader dimension. A representative episode is selected by maximum grade—not by median behavior—so it should be labeled “best representative,” not “typical.”

### 19.27 Acceptance Gate

Load matched base/candidate results, verify configuration equality, compute macro means, and append named checks. Acceptance is `all(check.passed)`.

This is an explainable gate: every failure records actual and required values.

### 19.28 Scalable Versioned Step

A proposed production command:

```text
receive authenticated session, expected version, idempotency key, action
return cached response if idempotency key exists
lock session or compare version
load state
validate and apply action to candidate state
persist event + new snapshot + response atomically
increment version
release lock
```

Expected complexity is dominated by state serialization and store round trips, not the bounded engine loops. The design prevents duplicate and lost updates.

---

## 20. Beginner-Friendly Interview Questions

### Q1. What is Panopticon Protocol v3?

It is a turn-based counter-espionage environment for evaluating and training decision-making agents. ARGUS protects a simulated company from HYDRA sleepers while preserving revenue. The hard part is that hidden identity is not visible; the agent must build evidence.

### Q2. Is it a normal video game?

It has game mechanics, but its primary engineering purpose is to expose a repeatable agent interface: reset, observe, act, receive reward, and get graded. That lets rules, reinforcement-learning networks, and language models be compared on the same task.

### Q3. What does “environment” mean?

The environment is the simulated world plus its rules. It owns workers, hidden sleepers, leaks, scores, phases, and HYDRA behavior. An agent cannot directly rewrite this state; it submits legal actions.

### Q4. What is the difference between ARGUS and a policy?

ARGUS is the role inside the game. A policy is the implementation choosing ARGUS's actions. The policy might be random, heuristic, security-first rules, PPO, or an LLM.

### Q5. Is HYDRA another trained AI?

No. In the current repository HYDRA is scripted adaptive logic inside `environment.py`. It remembers canary departments, monitored channels, and audit patterns, but it is not a separately optimized neural network.

### Q6. What is partial observability?

The full state contains facts the agent is not allowed to see. The observation hides sleeper identity, generation, activation, false-flag truth, dead switches, and HYDRA memory. The agent must infer threats from permitted signals.

### Q7. Why use Pydantic?

Pydantic defines typed models, applies defaults, validates JSON-shaped data, serializes objects with `model_dump()`, and generates JSON Schema. It makes the engine, API, trainers, and grader share one explicit contract.

### Q8. What is a state machine here?

Workers and the episode move through defined states after legal events. A dormant sleeper activates; a suspected sleeper may be turned; a turned sleeper becomes a double agent. `Environment.step()` controls the transition order.

### Q9. What happens in one turn?

The engine validates ARGUS's action, applies it, lets HYDRA act, progresses turning, runs economics, calculates reward, advances phase/turn, checks end conditions, and returns a new safe observation.

### Q10. What is a canary trap?

It is uniquely fingerprinted information placed in one department. If that fingerprint later appears in a leak, ARGUS has evidence connecting the leak to that department and can verify it.

### Q11. Why not terminate every suspicious worker?

Suspicion can come from incomplete evidence or a false flag. Terminating an innocent costs revenue, security, and grade. Advanced levels require zero false accusations.

### Q12. What does interrogation do?

For a sleeper, it sets public suspicion to one and reveals generation plus armed dead-switch information in an event message. For an innocent, it reduces suspicion. It does not disarm a dead switch.

### Q13. How does turning work?

ARGUS starts a four-turn conversion on a real sleeper. The operation progresses automatically each step. When complete, the sleeper counts as caught and becomes a double-agent asset that can send disinformation.

### Q14. What makes each level harder?

Later levels add more sleepers and mechanics: canary/channel avoidance, false flags, dead switches, and a high-performing Gen-5 executive sleeper. They also enforce stricter grading gates.

### Q15. What is reward?

Reward is immediate numeric feedback after a turn. The current formula balances normalized revenue and security change, then adds threat-resolution bonuses and penalties for false accusations, active threats, low security, and wasted time.

### Q16. Why have a separate grader?

Reward is optimized during learning and can be gamed. The grader independently evaluates five dimensions and applies hard security constraints. Separation provides a second line of defense.

### Q17. What is the 136-dimensional observation?

It is a fixed float array for PPO: capacity-limited worker, leak, canary, double-agent, and global features. “Dimension” means one numeric position in that vector.

### Q18. Why use a fixed vector?

Standard neural layers expect a consistent input shape. Padding empty slots with zeros makes every observation length 136, though it can truncate entities beyond capacity.

### Q19. What is PPO?

PPO is a reinforcement-learning algorithm. It gathers experience with the current policy, estimates which actions were better than expected, and updates action probabilities while clipping large changes for stability.

### Q20. Why three PPO action heads?

The action contains three factors: type, target, and sub-action. Separate heads produce 8, 8, and 7 logits instead of one 448-output head. Shared features let the choices influence one representation, although the distributions are factorized.

### Q21. What is an LLM doing here?

The LLM reads a text version of the safe observation and generates one JSON action. It is trained by imitating security-first demonstrations.

### Q22. What are SFT and LoRA?

SFT teaches from input/desired-output examples. LoRA trains small low-rank adapter matrices in selected model modules rather than updating every base weight, reducing memory and artifact size.

### Q23. What is curriculum learning?

The trainer moves from easy through Level 5, carrying learned weights forward. It tries to teach one layer of complexity at a time.

### Q24. What is a checkpoint?

A checkpoint saves training progress so interruption does not require starting from zero. The V2 trainer also saves curriculum-level state and data metadata.

### Q25. What is action repair?

LLMs sometimes emit malformed or semantically wrong output. The local policy normalizes aliases, recovers IDs/intent, fills compatible sub-actions, prioritizes urgent security behavior, validates, and falls back safely.

### Q26. Did the V5 model succeed?

Partly. Its raw macro grade improved from `.641110` to `.701627`, but it failed strict advanced security acceptance. The security-first supervisor passed, proving the controller path is solvable but not that the raw model independently mastered it.

### Q27. What does FastAPI add?

It makes the environment accessible over HTTP, validates request/response models, and generates interactive OpenAPI documentation.

### Q28. What is the biggest current server limitation?

One global in-memory environment is shared by all callers in one process. There is no session isolation, durability, concurrency control, or ownership.

### Q29. What is the biggest current deployment limitation?

The source paths are now aligned on `_server:app`, but the image has not been built in this workspace and its lightweight dependency layer does not support local model inference. A successful clean build, health/API smoke test, dependency lock, and explicit simulator-versus-inference image boundary are required before claiming reproducible deployment.

### Q30. What would you improve first?

For model quality: analyze advanced raw failures and reduce supervisor dependence. For product infrastructure: build-test the corrected image, then add session-scoped authenticated APIs, versioned/idempotent steps, durable state, and model workers outside the API loop.

---

## 21. Strengths and Honest Gaps

Strong interview answers explain both what works and what remains.

### Strengths to Emphasize

1. **A clear research question**

The task tests decision-making under hidden identity, delayed evidence, adversarial adaptation, and competing productivity/security goals.

2. **Real information boundary**

Full state and agent observation are separate models/paths. Hidden fields are deliberately reconstructed to safe defaults.

3. **One engine, several policies**

Random, heuristic, security-first, PPO, remote LLM, and local fine-tuned LLM paths converge on `AgentAction` and the same `step()` behavior.

4. **Explicit legal-action contract**

Enums, Pydantic, JSON schemas, and semantic action validation make failures inspectable.

5. **Long-horizon mechanics**

Canary detection and double-agent turning require multi-step planning rather than one-shot classification.

6. **Independent evaluation**

Reward and grader are separate, and advanced levels enforce security constraints outside the weighted composite.

7. **Fail-closed expert and release gates**

Unsafe demonstration episodes stop training-data generation; benchmark acceptance fails on any unmet check.

8. **Reproducibility metadata**

Seeds, seed plans, reward/grader/trajectory schemas, runtime profiles, source commits, checkpoints, and progress files reduce accidental mismatch.

9. **Resumable expensive work**

Dataset reuse, trainer checkpoints, curriculum state, evaluation checkpoints, and atomic metadata writes support interrupted Colab/Drive runs.

10. **Operationally aware inference**

Parsing, semantic repair, validation, fallbacks, decision timelines, and raw-text capture recognize that generated JSON is not automatically trustworthy.

11. **Honest V5 reporting**

The repository distinguishes base, raw trained, heuristic, random, and supervisor results and publishes the strict gate failure instead of hiding it.

12. **Concrete production path**

The domain state is serializable, making session stores, versioned commands, event logs, and independent inference workers plausible next steps.

### Honest Gaps to Mention

#### Gap 1: Corrected Container Source Is Not Yet Production-Validated

Docker/OpenEnv/package paths now agree on `_server:app`, but no clean image build was possible in this workspace. The core image also omits the local-model runtime. Build, vulnerability-scan, smoke-test, and sign the intended simulator and inference images before presenting deployment as complete.

#### Gap 2: One Global Environment

All API callers share `_env`. There is no session isolation, persistence, ownership, ordering, or idempotency.

#### Gap 3: Hidden State Is Publicly Routed

`GET /state` and `/render` now return `404` by default and require the explicit `PANOPTICON_ENABLE_PRIVILEGED_DEBUG` flag. There is still no role-based authorization, so privileged debug mode must run only on an isolated trusted interface.

#### Gap 4: Permissive CORS and No Auth

Wildcard origins and absent caller identity are suitable only for a controlled demo.

#### Gap 5: Local Generation Can Block the API

Async route syntax does not move synchronous model generation off the event loop. Inference belongs in dedicated workers/services.

#### Gap 6: PPO Capacity Truncation

The vector stores only 12 workers, 8 leaks, 6 canaries, and 3 assets. Additional entities disappear from numeric input.

#### Resolved Gap 7: All Current Worker Slots Are Addressable

Action schema `gym-factored-v2-target12-masked` sets `NUM_TARGETS=12`, matching the worker capacity of the numeric observation. Level 4/5 workers at indices eight and nine are now directly selectable. This intentionally makes legacy eight-target PPO checkpoints incompatible; silently loading them would preserve a policy that never learned those targets.

#### Resolved Gap 8: Canonical Joint Mask Is Applied by PPO

`get_action_mask()` now enumerates all `8 × 12 × 7` triples. Strict decoding rejects modulo aliases and irrelevant sub-actions; `validate_action()` then checks state-dependent semantics. PPO chooses an allowed action type, an allowed target for that type, and an allowed sub-action for that pair. The rollout stores the entire mask so PPO recomputes the same conditional log-probability during updates.

#### Remaining Gap 9: Head Logits Do Not Consume Earlier Selections

Masking makes sampled actions legal, but all three raw logit heads are still produced from the same shared features. A richer autoregressive network could feed the chosen action-type embedding into the target head and both earlier choices into the sub-action head. Pointer networks could also score a variable-length entity list instead of using a fixed capacity.

#### Gap 10: Some HYDRA Fields Are Telemetry More Than Mechanics

`hydra_aggression` is stored but not used in transition logic. `deep_cover` is recorded but does not itself modify spawning. `cover_integrity` and recruitment-accuracy degradation are updated but do not materially enter audit detection in the current code.

#### Gap 11: No Dead-Switch Disarm Action

Interrogation reveals the switch but cannot neutralize it. Documentation must not claim a safe terminate path that does not exist.

#### Gap 12: Terminal-Step Semantics Need Cleanup

The invalid-action branch conflates `done` and `truncated`; the engine also does not reject a step submitted after an episode is already done.

#### Gap 13: Grader Proxies Are Coarse

Adaptability does not measure full action diversity despite earlier descriptions, and efficiency's “optimal steps” is weak when episodes usually run to horizon.

#### Gap 14: Repair/Supervisor Attribution

A repaired trained action may be safer than the raw generation. Reports must expose raw validity, repair category, fallback rate, and supervisor use.

#### Gap 15: Advanced Raw Model Is Not Release-Ready

Raw V5 has only 50% Level-4 and 5% Level-5 pass rates in the matched summary. Improvement over base is real, but the strict gate verdict is still failure.

#### Gap 16: Test Coverage Is Script-Centric

Security regression covers the expert path, but there is no broad checked-in unit/integration/property/concurrency suite for every engine and API invariant.

#### Gap 17: Dependency Sources Drift

Docker pins, `requirements.txt`, and `pyproject.toml` ranges are not one locked dependency source of truth.

#### Gap 18: Scripted Adversary Limits Generalization Claims

HYDRA adapts with rules. Claims about robustness to learned or human adversaries require new experiments.

### How to Phrase a Gap Well

Use this pattern:

> “The current implementation does X because of the project stage. The failure mode is Y. I verified it in code at Z. The production change would be A, and I would validate it with B.”

That answer shows ownership, risk awareness, and a testable improvement—not apology.

---

## 22. One-Minute and Senior-Engineer Closing Answers

### One-Minute General Answer

> “Panopticon Protocol v3 is a counter-espionage learning environment where an agent called ARGUS protects a company from hidden HYDRA sleepers. The agent sees employee performance, suspicion, leaks, canaries, and reports, but not the true sleeper fields. A typed Python engine controls legal transitions and produces security-first reward. The same engine supports a 136-feature PPO agent and a Qwen LLM trained through security-gated expert demonstrations and LoRA. Completed episodes are scored across security, revenue, intelligence, adaptability, and efficiency, with non-negotiable Level-4/5 gates. The latest raw V5 model improves macro grade over base but fails advanced safety acceptance; the verified supervisor passes. The next work is raw-model reliability plus session-isolated, durable, authenticated deployment.”

### Non-Technical Stakeholder Answer

> “It is a safe training ground for an AI security manager. The AI must keep a company productive while finding insiders without accusing innocent employees. We make the evidence incomplete on purpose, increase the adversary's sophistication over five levels, and measure not just money but whether every threat was handled safely. The latest trained AI improved, but our release checks correctly rejected it on the hardest levels, which tells us exactly what must improve.”

### Senior Engineer Answer

> “It is a modular partially observable state machine. Pydantic defines internal state, redacted observation, and factored actions; `Environment.step` owns ordered mutations; Gymnasium adapts the domain model to a fixed tensor; PPO and LLM policies share one validation boundary; and a separate versioned grader enforces hard advanced-tier constraints. The V2 TRL pipeline adds deterministic security-gated demonstrations, weighted rare actions, context fitting, completion-only LoRA SFT, resumable curriculum/checkpoints, merge, and matched evaluation. The current service is single-process and non-multi-tenant; production needs session IDs, version/idempotency control, external state, durable events, model workers, auth, and hidden-state isolation.”

### Machine-Learning Panel Answer

> “The interesting ML problem is sequential decision-making under asymmetric hidden information and long-horizon evidence. I implemented two policy representations: a factored PPO actor-critic over 136 numeric features and an instruction LLM producing structured actions. The safety lesson is that imitation loss and composite grade are insufficient: raw V5 improves the matched macro grade but violates advanced operational constraints, while a security-first supervisor passes. My next experiment would isolate raw parse, repair, prioritization, and distribution-shift failures on held-out advanced seeds and train against those failure modes without changing the acceptance gate.”

### System-Design Panel Answer

> “Today the engine is one in-memory singleton behind FastAPI, suitable for a controlled demo. At 10,000 sessions I would use authenticated session-scoped commands through stateless API replicas, per-session versioning and idempotency, Redis or an actor/shard owner for hot ordered state, Postgres for durable metadata/results, object storage for timelines/models, and separate GPU inference workers behind a bounded queue. Hidden state and grading stay on trusted internal paths. Metrics cover API latency, queue depth, session conflicts, parse/repair/fallback rates, and per-level security outcomes.”

### Strongest Design Decision

> “Separating full state from observation and separating reward from hard-gated evaluation. The first preserves the reasoning challenge; the second prevents a high average score from disguising an unsafe advanced-tier policy.”

### Biggest Known Limitation

> “For the ML result, raw V5 is not advanced-tier release-ready. For infrastructure, the API is one unauthenticated global session; container paths are corrected but the image is not yet build-verified. Those are measured and code-verifiable limitations, not vague future concerns.”

### Final Spoken Line

> “The project demonstrates not only how to train an agent, but how to know when that agent is not safe enough yet.”

---

## 23. Machine-Learning Interview Questions with Strong Answers

### Q1. How would you classify the learning problem?

It is a finite-horizon, partially observable, sequential decision problem with an adversarial scripted environment. A POMDP is a useful formal model: hidden state `s`, observation `o`, action `a`, transition dynamics, observation function, reward, and discount. It is not strict two-learned-agent MARL in the current implementation because HYDRA is rule-based.

### Q2. Does the observation satisfy the Markov property?

Not fully. The observation strips hidden state and exposes only limited reports/messages. The full `EnvironmentState` is approximately Markov for the simulator because it contains the variables required for the next transition; the agent observation is partially observable and may benefit from history or recurrent memory.

### Q3. Why is partial observability important rather than just hiding a label?

The hidden identity affects delayed activation, leaks, false flags, and dead switches. The policy must choose information-gathering actions whose value appears later. That tests belief formation and planning, not one-step label prediction.

### Q4. Why use dense reward?

The evidence chain is long. End-only reward would make it difficult to identify which earlier canary, monitor, or investigate action contributed to success. Dense deltas and threat costs improve credit assignment, while the separate grader constrains reward gaming.

### Q5. What is the main reward-design risk?

Proxy optimization. Revenue can dominate short-term behavior or double-agent bonuses can distract from unresolved threats. V2 adds caught/false/active-threat/security-deficit terms, and advanced hard gates make safety non-compensable.

### Q6. Why PPO?

PPO is relatively simple and stable for on-policy discrete control. The code is small enough to explain completely, supports the factored action representation, and does not require a replay buffer. Its cost is lower sample reuse than off-policy algorithms.

### Q7. What does PPO clipping do?

It compares new and old action probabilities through a ratio. The surrogate objective uses a clipped ratio so one batch cannot make an arbitrarily large probability change. This is a trust-region-like stability mechanism, not a guarantee of monotonic improvement.

### Q8. Why sum the three log probabilities?

The implementation sums the log-probabilities of action type, target conditional on the selected type's validity mask, and sub-action conditional on the selected type/target mask. The logarithm of a product is a sum. Masking supplies semantic dependence, although the target and sub-action logits themselves still come only from shared features.

### Q9. How would you improve the factored action policy?

The current sampler already applies conditional masks. The next modeling improvement is to feed embeddings of sampled earlier choices into later heads, or use a hierarchical/parameterized policy or pointer mechanism over the current entity list. That could learn preferences among valid combinations more expressively.

### Q10. Is the current action mask sufficient?

For the current action contract, yes: the joint mask reflects departments, vacancies, live workers, leaks, active assets, and compatible sub-actions, and native PPO uses it. Tests enumerate every admitted triple and require semantic validation. It must be versioned again if entities or actions exceed current capacities.

### Q11. What is the most serious PPO representation bug?

That legacy bug is fixed by `NUM_TARGETS=12`. The research risk now is checkpoint compatibility: an eight-target checkpoint cannot be treated as equivalent to the new policy, so the loader rejects mismatched shapes and requires retraining or an explicitly studied migration.

### Q12. What does the critic learn?

It predicts expected discounted return from the current 136-feature observation. PPO uses that estimate to compute temporal-difference residuals, advantages, and value loss.

### Q13. Why GAE with gamma .99 and lambda .95?

Gamma values long-term outcomes; lambda blends multi-step residuals. `.95` is a common compromise between noisy Monte Carlo returns and biased one-step estimates. It is a hyperparameter choice that should be ablated, not treated as universally optimal.

### Q14. Why normalize advantages?

Within a rollout minibatch it centers/scales the update signal, often improving optimization conditioning. It changes update scale but not the relative ordering of advantages.

### Q15. Why add entropy?

Entropy discourages the action distributions from collapsing too early, preserving exploration. Too much entropy prevents decisive behavior; too little can lock in a poor routine.

### Q16. Why curriculum learning?

Easy tiers supply frequent evidence and fewer simultaneous mechanics. Later tiers then add avoidance, false flags, dead switches, and Gen-5. The risk is catastrophic forgetting, so evaluation must retain every earlier level.

### Q17. Why SFT rather than pure RL for the LLM?

SFT is much cheaper and more stable when a high-quality demonstrator exists. It teaches schema-following and policy priorities directly. It is limited by demonstrator quality and covariate shift: once the model makes a mistake, it may enter states absent from expert trajectories.

### Q18. What is covariate shift here?

Training observations come from the expert's state distribution. A model's different action changes future states, potentially producing observations the expert dataset rarely contains. DAgger-style data aggregation could run the model, ask the expert to label visited states, and retrain.

### Q19. Why gate expert episodes?

Imitation learning copies behavior. Gating guarantees every demonstration episode meets the defined security standard before any of its turns become training data. Without it, the dataset can encode unsafe tradeoffs.

### Q20. What is the risk of duplicating rare actions?

It corrects action-frequency imbalance but may overfit identical states and distort calibration. Better additions include diverse state collection, counterexamples, weighted loss, focal objectives, or sequence-level preference data.

### Q21. Why LoRA?

It reduces trainable memory and artifact size by learning low-rank updates in selected projection layers. It is practical on a T4 and supports chained adapters. It does not automatically guarantee equal quality to full fine-tuning.

### Q22. Why target both attention and feed-forward projections?

Action selection may require adapting token interaction and task-specific transformation capacity. Targeting `q/k/v/o` changes attention behavior; `gate/up/down` changes the MLP path. It increases adapter capacity and trainable parameters versus attention-only LoRA.

### Q23. What do rank and alpha mean?

Rank is the inner dimension of the low-rank update; higher rank can represent richer changes but costs more. Alpha scales the update, commonly through `alpha/r`. Current values are rank 16 and alpha 32.

### Q24. Why completion-only loss?

The desired task is generating the action, not reproducing the observation. Masking system/user tokens focuses gradient on assistant JSON and avoids wasting capacity learning copied prompt text.

### Q25. Why verify nonzero valid label tokens?

A wrong response template can mask every token. Training would run with no useful supervision, possibly producing zero/NaN loss or meaningless checkpoints. The sanity check fails early.

### Q26. How do you handle long contexts?

Render full text, compact repetitive safe information, preserve recent reports/canaries, then binary-search a token budget with head/tail retention. I would improve this with a structured salience policy and tests proving critical evidence survives compaction.

### Q27. Why require deterministic expert seeds?

It makes datasets auditable and reproducible, lets failed episodes be replayed, and prevents silent scenario changes between runs. Evaluation seeds should still be held out from training seeds.

### Q28. Why is token accuracy not the main metric?

Many token sequences can encode equivalent actions, and one wrong action can create long delayed harm. The mission is sequential operational performance, so grade, pass rate, security, caught/missed sleepers, false accusations, and fallback rate are stronger metrics.

### Q29. What does raw V5 show?

Fine-tuning improved mean composite across all levels and macro grade by about 9.44% relative to base. It did not achieve reliability: Level-4 pass rate was 50% and Level-5 5%, so the strict gate correctly failed it.

### Q30. Why evaluate the supervisor?

It is a diagnostic upper control path. If the supervisor also failed, the environment, policy logic, or gate might be unsatisfiable. Its pass isolates the remaining issue to learned-policy reliability/attribution, but it must not be reported as raw neural performance.

### Q31. How would you measure repair dependence?

Log raw parse validity, raw semantic validity, repair category, repaired action, fallback/supervisor use, and outcome. Report raw-only, parser-cleanup-only, semantic-repair, and supervised results as separate ablations.

### Q32. What is an ablation study?

Remove or change one component while holding others fixed to estimate its contribution. Useful ablations include no data weighting, no security-deficit reward, no repair, attention-only LoRA, no curriculum, and conditional action masks.

### Q33. How would you create hard negatives?

At critical observations, pair the expert action with tempting unsafe alternatives such as work during an active confirmed threat, terminate after weak false-flag evidence, or deploy a double agent while security is low. Train a preference/ranking objective and verify with unchanged held-out gates.

### Q34. Would you use DPO or RLHF next?

DPO is attractive if reliable preferred/rejected action pairs can be generated; it is simpler than online RLHF. For true sequential credit, offline preference learning alone may be insufficient. I would first use DAgger/hard-negative SFT or DPO, then consider environment RL with constrained safety and careful compute budgeting.

### Q35. How would you model memory?

For PPO, use an LSTM/GRU or attention over recent observations/reports, with hidden state reset per episode. For the LLM, provide a bounded structured event summary rather than raw ever-growing chat. Memory must contain only observable history.

### Q36. How would you test generalization?

Use held-out seed plans, changed sleeper schedules, randomized department layouts, new worker name distributions, altered damage/cost parameters, and adversary-policy variants. Report in-distribution and out-of-distribution outcomes separately.

### Q37. How many evaluation episodes are enough?

There is no universal count. Estimate variance and confidence intervals for pass rate and safety failures. For a release rule requiring zero advanced failures, 20 episodes is useful evidence but weak assurance; increase seeds and use an explicit acceptable failure-risk bound.

### Q38. How would you serve the LLM?

Load a versioned merged model on dedicated GPU workers, batch compatible requests, bound queue depth, apply deadlines, return proposed actions with model/decoding metadata, and validate/version-check before mutating session state. Keep raw and repaired decisions in audit logs.

### Q39. How do you prevent training/evaluation contamination?

Separate seed sets and artifacts; hash the seed plan; record source/model/data versions; avoid selecting checkpoints on final test results; and keep final acceptance evaluation write-protected until model selection ends.

### Q40. What is the core ML lesson?

A lower loss or higher mean grade is not the same as safe sequential behavior. Demonstration quality, state distribution, action validity, repair attribution, matched evaluation, and hard operational constraints must all be part of the model lifecycle.

---

## 24. System Design Interview Questions with Strong Answers

The strongest system-design answer separates the current demo, target requirements, proposed architecture, trade-offs, and validation plan.

### Architecture, Capacity, and State

### Q1. Explain the current high-level architecture.

`models.py` defines typed domain contracts; `environment.py` is the in-memory state machine; `grader.py` evaluates completed state; `gym_wrapper.py` and text formatting adapt observations to PPO/LLM policies; `_server.py` exposes a single global environment through FastAPI; training and evaluation call the engine directly. It is modular within one repository, but the server is one unauthenticated, non-durable session.

### Q2. What target requirements would you clarify first?

I would ask: expected concurrent sessions and step rate; human-chosen versus model-chosen actions; latency SLOs; whether episodes must survive restart; audit/retention requirements; public versus private evaluation; geographic distribution; model sizes; and acceptable fallback behavior. Architecture depends more on these constraints than on a preferred technology.

### Q3. Give a capacity estimate for 10,000 sessions.

As a planning assumption, at 0.2 steps per session per second the peak average is about 2,000 steps/s. At 50 KB serialized hot state per session, raw state is roughly 500 MB before replication and store overhead. I would benchmark actual p95 state size and step CPU time; model inference is likely the dominant latency/cost and should be capacity-planned separately.

### Q4. What is the biggest current scaling limitation?

`_env` is one Python object. Every caller shares it, a reset destroys another caller's episode, requests can race, restart loses state, and adding Uvicorn workers creates independent conflicting worlds. Session isolation is the first architectural change.

### Q5. What API would you design?

I would create `POST /sessions`, session-scoped reset/step/agent-step/observation/grade endpoints, and `DELETE /sessions/{id}`. Mutations carry `expected_version` and `Idempotency-Key`. Responses include session ID, new version, terminal flags, safe observation, and correlation/request ID. Full state remains an internal evaluator endpoint.

### Q6. Why make FastAPI replicas stateless?

Stateless replicas can be added/removed and any request can reach any healthy instance because hot/durable session state lives outside process memory. This improves horizontal scaling and restart recovery. “Stateless” does not mean the product has no state; it means API instance memory is not the source of truth.

### Q7. Why use Redis?

Redis is appropriate for low-latency active session snapshots, TTL expiry, atomic compare/set, idempotency responses, distributed leases, queue/stream primitives, and rate limits. I would not make it the only audit store without a deliberate persistence/replication plan; completed results and important events go to durable storage.

### Q8. What belongs in Postgres?

Users/service identities, session ownership/status, task/model versions, command metadata, episode results, grades, benchmark definitions, and audit records. Postgres gives transactions, constraints, indexing, and durable relational queries. High-frequency snapshots may stay in Redis/object storage to avoid unnecessary row churn.

### Q9. What belongs in object storage?

Large immutable artifacts: per-turn timelines, raw evaluation payloads, checkpoints, merged models, plots, logs, and periodic state snapshots. Store their URI, digest, size, schema, and retention policy in Postgres. Object storage is cheaper and better suited than relational rows for multi-megabyte artifacts.

### Q10. How would you measure state size rather than assume it?

Run representative episodes for all tiers, serialize full state and observation at every turn, and record p50/p95/p99 bytes plus entity counts. Include Python/Redis serialization overhead and compression CPU. Capacity should use high-percentile active state, not a hand-picked early episode.

### Q11. How do you isolate sessions?

Generate unguessable session IDs, store owner/tenant, authorize every access, key all hot state by session, and never select global state without the ID. Separate evaluator privileges from player privileges. Add tests where two clients reset and step concurrently and prove observations/versions never cross.

### Q12. How do you preserve action order within one session?

Use one logical writer per session: a partitioned command queue keyed by session ID, or optimistic version checks plus a short per-session lease. Every accepted action increments a monotonically increasing version/sequence. Out-of-order commands fail with conflict and must reload.

### Q13. Explain optimistic concurrency here.

The client reads version 20 and submits `expected_version=20`. The store updates only if current version is still 20, atomically creating version 21. If another command already advanced it, the update affects zero rows or compare-and-set fails, and the API returns `409 Conflict`. This avoids holding a lock during client think time.

### Q14. When would you use a distributed lock instead?

A short lease is useful when reconstructing/applying state spans several operations not easily expressed as one compare-and-set. It must have expiry, ownership token, renewal policy, and fencing/version checks so a paused old holder cannot write after its lease expired. Optimistic versioning should still protect final commit.

### Q15. How do idempotency keys prevent duplicate turns?

The client generates a unique key for one intended mutation. The server atomically stores `(session,key)` with request fingerprint and response. A retry with the same fingerprint returns the original response; the same key with different content is rejected. Retention lasts at least the maximum retry window.

### Q16. What if the client times out after commit?

It retries with the same idempotency key. The server finds the committed response rather than calling `step()` again. Without this, transport uncertainty becomes a double action and breaks episode correctness.

### Q17. Snapshot or event sourcing?

For a first production version, save a versioned snapshot after each accepted bounded-size step plus an append-only command/transition audit record. Pure event sourcing gives excellent replay but increases schema/replay complexity. A hybrid with periodic snapshots and events after the snapshot provides recovery and forensics without replaying the entire episode.

### Q18. How do you recover a session after a worker crash?

The new owner loads the latest committed snapshot/version, optionally replays later committed events, verifies invariants and digest, reacquires the lease, then accepts the next expected version. Uncommitted in-memory mutations are discarded. Idempotency records answer ambiguous retries.

### Q19. Can the current state be persisted?

Yes, `EnvironmentState` is Pydantic-serializable, but recovery also needs engine-private fields such as pending sleeper schedule, turning countdown map, counters, task configuration, name pool/random-generator state, and exact code/schema version. Persisting only `state.model_dump()` may not reproduce future transitions exactly, so a formal snapshot model is needed.

### Q20. What happens with multiple Uvicorn workers today?

Each process owns a different `_env` and `_agent_policy`. Requests from one client may hit different worlds unless routing is sticky, and resets/steps diverge. Sticky routing only hides the problem and still loses state on process death; shared session state is the durable fix.

### Q21. Do you need sticky sessions after externalizing state?

No for correctness. Any API replica can load/route the session. Affinity may improve cache locality, but it must be an optimization, not a requirement; otherwise replica failure still disrupts ownership.

### Q22. How would you partition session work?

Hash session ID to queue partitions or actor shards. All commands for one session land on one ordered partition, while different sessions execute in parallel. Track partition load and support controlled rebalancing with leases/version checkpoints.

### Q23. What causes hot partitions?

A few automated agents may step far faster than human sessions, or a poor hash/key choice may cluster load. Mitigate with a strong hash, more virtual partitions than workers, per-session rate limits, and moving exceptionally hot sessions to dedicated shards while preserving order.

### Model Inference and Backpressure

### Q24. Why separate model inference from the API?

LLM generation is slow, memory-heavy, and synchronous in the current route. Dedicated GPU workers allow independent scaling, batching, admission control, model versioning, and failure isolation. The API remains responsive for resets, observations, and manual actions.

### Q25. What does the inference queue contain?

Job ID, session ID/version, safe observation or immutable observation URI, task/model/decoding version, deadline, requested policy mode, and trace/correlation ID. It must never contain public full hidden state. Results include raw text, parsed/repaired action, timings, and model metadata.

### Q26. How do you implement backpressure?

Bound queue depth and per-tenant in-flight jobs. Estimate wait from recent service time. Reject or defer with `429/503` when limits are reached, honor deadlines, and avoid infinite retries. Autoscale on queue age and GPU utilization, not only CPU.

### Q27. Would you batch LLM requests?

Yes when models, token shapes, and latency budgets are compatible. Dynamic batching improves GPU utilization, but waiting to fill a batch increases latency. Use a small maximum batching window and separate interactive from bulk benchmark queues.

### Q28. What if inference returns after the session advanced?

The result carries the observation/session version used. The apply stage compares it with current version; if stale, it discards or records the proposal without mutation. It can optionally regenerate on the latest observation if the deadline permits.

### Q29. What is the timeout/fallback policy?

For interactive use, return an asynchronous job ID or push completion; after deadline, mark inference failed. A configurable fallback can choose a verified safe policy, but the response and metrics must label fallback. For raw model evaluation, no supervisor fallback should be credited as raw success.

### Q30. How do you preserve model attribution?

Store model digest/version, decoding parameters, raw output, parser result, repair category, final action, fallback/supervisor flag, and session version. Report raw, repaired, and supervised policy metrics separately. An audit record makes every executed action explainable.

### Security and Trust Boundaries

### Q31. How do you protect hidden state?

Never return it through public session APIs. Keep full state in an internal store namespace encrypted and access-controlled; only the session engine and grader service identities can read it. Public logs, queues, traces, error messages, and analytics must use redacted observation/event schemas. Test redaction whenever fields are added.

### Q32. What authentication and authorization model would you use?

Use OIDC/JWT for users and short-lived workload identities for services. The API verifies issuer, audience, signature, expiry, and scopes. Authorization checks tenant/owner or evaluator role on each session. Privileged grader/state access is server-to-server and audited; hiding a frontend button is never authorization.

### Q33. How would you configure CORS?

Allow only exact deployed frontend origins, required methods, and required headers. Avoid wildcard origins with credentials. CORS is browser protection, not API authentication; non-browser clients can ignore it, so server-side identity/authorization remains mandatory.

### Q34. How do you manage secrets?

Store database credentials, model tokens, and signing keys in a secret manager, inject them at runtime, grant least privilege, rotate them, and prevent values from entering images/logs. Public clients never receive service secrets. Record access and alert on unusual use.

### Q35. What rate limits are needed?

Separate quotas for cheap environment steps, expensive agent inference, session creation, grading, and artifact download. Limit by authenticated tenant/user plus IP abuse controls; cap concurrency and daily model tokens. Return clear retry metadata and prioritize evaluator/internal workloads separately.

### Q36. How do you defend against resource-exhaustion requests?

Set maximum body size, schema depth/list lengths, reason/prompt length, session count, timeline export size, and deadline. Reject unknown fields when appropriate, stream large downloads, bound decompression, and never let a caller request arbitrary local model paths or filesystem locations.

### Q37. How do you guarantee action/state integrity?

Validate type and semantics, apply commands only at expected version, persist event and snapshot transactionally, hash immutable artifacts, and verify model/schema versions. Audit every privileged mutation. Invariants such as score bounds, unique IDs, and terminal-state immutability run before commit.

### Q38. Is prompt injection a concern?

Current observations mostly contain controlled game strings, so risk is limited. If user-generated names/messages/documents are added, treat them as untrusted data, delimit them, keep system instructions separate, use structured decoding/schema validation, restrict tools, and never let generated text authorize or directly access hidden state.

### Observability, Reliability, and Failure Handling

### Q39. What SLOs would you define?

Examples: 99.9% successful manual step availability monthly; p95 manual-step latency below a measured target such as 200 ms excluding network; p95 inference queue-to-result below the product deadline; zero cross-session data leaks; and a benchmark pipeline completion/recovery target. Final numbers require load tests and business requirements.

### Q40. Which metrics matter most?

API rate/errors/latency; active sessions; version conflicts and duplicate retries; Redis/Postgres latency; queue depth/oldest age; GPU utilization/memory/tokens per second; model parse/repair/fallback rates; grade/pass/security/caught/missed/false metrics by model and level; and deployment/model/schema labels.

### Q41. What alerts would you configure?

Sustained error-budget burn, high p95/p99 latency, growing oldest queue age, inference timeout spike, Redis/Postgres error/replication lag, session conflict spike, model parse/repair/fallback regression, advanced security pass-rate drop, hidden-state redaction failure, and health/readiness failure. Alerts need runbooks and should be actionable, not every transient blip.

### Q42. How would you log and trace a step?

Use structured logs with request/trace ID, tenant/session ID (pseudonymous where needed), expected/new version, action category, validity, latency segments, model/repair metadata, outcome counters, and error code. A distributed trace spans gateway, API, queue, inference, session engine, and stores. Never log full hidden state or secrets.

### Q43. What is the difference between liveness and readiness?

Liveness answers whether the process should be restarted. Readiness answers whether it can receive traffic now. An API may be live but unready because Redis is unreachable or migrations are incomplete. Inference workers may be live while their model is still loading.

### Q44. What if Redis is unavailable?

Fail closed for mutations if Redis owns authoritative hot state; do not create a divergent local environment. Mark readiness false, use bounded retries with jitter/circuit breaking, preserve queued commands if safely possible, and recover from durable snapshots/events when Redis returns. A replicated Redis deployment reduces but does not remove this risk.

### Q45. What if Postgres is slow?

Use query timeouts, tuned pools, indexes, and circuit breaking. Avoid holding a session lock during slow analytics writes. Critical command commit must either complete atomically or fail without acknowledgement; noncritical analytics can go to a bounded async pipeline. Expose pool saturation and slow-query metrics.

### Q46. What if an inference job repeatedly crashes a worker?

Cap retries, record the error/model/input metadata safely, and move the job to a dead-letter queue after the limit. Quarantine malformed/oversized inputs and alert on repeated model-version crashes. A dead-letter queue stores work needing inspection rather than retrying forever.

### Q47. How do you handle corrupt session state?

Stop mutation, quarantine the session, preserve the corrupt blob and audit trail, validate the latest earlier snapshot, replay verified events, and compare digests/invariants. Do not silently reset because that destroys forensic evidence and surprises the user.

### Q48. Define RPO and RTO for this system.

RPO is acceptable data loss; RTO is acceptable recovery time. For accepted steps, target RPO zero by acknowledging only after durable command/snapshot commit. RTO might be minutes for a regional store failure and seconds for one API worker crash, depending on product requirements and cost.

### Deployment, Versioning, and Regions

### Q49. What is the first deployment fix?

Make one canonical application path: either rename/package `_server.py` consistently as `server:app` or update Docker/OpenEnv to `_server:app`; remove/correct missing `COPY` paths; copy required modules/static assets; choose whether the image supports trained inference; install from one locked dependency source; build and run smoke/API tests inside the image.

### Q50. Describe the CI/CD release pipeline.

On each candidate commit: lint/type/unit/property/API tests; security regression; build locked image; dependency/image scan; start image and run health/reset/step/grade contract tests; matched model acceptance when model changes; publish immutable image/model digests; deploy to staging; run load/failure tests; then canary production rollout with automatic rollback thresholds.

### Q51. How do schema migrations work without downtime?

Use expand-and-contract. First deploy backward-compatible readers/writers that understand old and new fields, add nullable/defaulted storage changes, backfill, switch traffic/writes, verify, then remove old fields in a later release. Snapshot/event records carry schema version and migration/replay code.

### Q52. What is a canary deployment?

Send a small percentage of production traffic to the new service/model version, compare errors, latency, repair/fallback, and safety metrics with control, then expand gradually. It is unrelated to the in-game canary trap except for the shared metaphor of early warning.

### Q53. How do you roll back a bad model?

Model references are immutable digests with configuration and tokenizer. Route new inference jobs back to the last accepted version; do not mutate in-flight job attribution. Preserve the bad version's traces for analysis. Session state is model-independent, so rollback does not require resetting episodes.

### Q54. Would you deploy multi-region?

Only if latency/availability requirements justify complexity. Assign each session a home region and keep its ordered writer/state there. Route users to that region; asynchronously replicate durable results/artifacts. Cross-region active-active writes for one session create unnecessary consistency conflict.

### Q55. What consistency model do you choose?

Strong per-session command ordering and read-your-writes for active gameplay; eventual consistency is acceptable for aggregated dashboards, analytics, and replicated artifacts. State and grade for a completed episode should become immutable after finalization except through audited correction/versioning.

### Q56. What would you cache?

Cache immutable task/schema/metadata, model metadata, completed grade responses, and signed artifact URLs. Do not cache a mutable observation without session version in the key. Avoid caching hidden state in shared public layers. Use TTL and invalidate/version keys on schema/model change.

### Testing, Cost, Privacy, and Final Trade-offs

### Q57. How would you test the full system?

Unit tests for transitions/reward/grader; property tests for invariants/redaction; integration tests with real Redis/Postgres containers; two-client isolation/concurrency/idempotency tests; model contract/repair tests; end-to-end session tests; load/soak tests; chaos tests for worker/store/model failure; migration/replay tests; and matched safety evaluation for every model release.

### Q58. How do you load-test it?

Model realistic arrival rates, session lifetimes, human/manual and model-driven mixes, level/state-size distributions, duplicate retries, and hot sessions. Measure p50/p95/p99 latency, throughput, errors, queue age, conflicts, store/GPU saturation, and cost. Test beyond target until the first bottleneck and verify graceful backpressure rather than collapse.

### Q59. How do you control cost?

Separate cheap simulator from expensive inference; autoscale GPU workers; batch within latency limits; cap tokens and context; cache immutable metadata, not decisions; set per-tenant quotas; expire inactive hot state to durable storage; compress large artifacts; tier retention; and expose cost per episode/model/level. Never reduce safety evaluation merely to improve a dashboard cost number.

### Q60. What is the privacy/retention design?

Classify user identity, prompts/reasons, timelines, full state, grades, and model traces. Collect only needed fields; keep hidden/evaluator data private; encrypt; define TTL/archival by purpose; support deletion/anonymization where required; restrict staff access; and log privileged reads. Public research exports should remove identities and sensitive free text.

### Q61. How would you handle abuse?

Authenticate, rate-limit, validate sizes, moderate any future user-generated text, isolate tenants, prevent arbitrary code/model-path execution, and provide audit/admin workflows. The simulator itself is low-risk, but model endpoints and artifact downloads can be used for compute or data exfiltration if unprotected.

### Q62. What are the main single points of failure?

Today: one API process/environment and one local model load. Proposed architecture risks include Redis primary, Postgres primary, one queue cluster, one model version, and regional routing. Mitigate by replication/failover, durable records, multi-worker capacity, tested backups, and documented degraded modes—not by claiming every component must always be multi-region.

### Q63. Why not build the full distributed architecture immediately?

The current hackathon/evaluation workload benefits from a simple deterministic in-process engine. Distributed state introduces serialization, ordering, failure, and operational complexity before it is needed. Good design keeps the engine decoupled so scale mechanisms can be added when requirements justify them, while being honest that the current API is not production multi-tenant.

### Q64. What would you change in the first production sprint?

Build-test the corrected container; keep privileged debug endpoints isolated; add authentication; introduce session IDs with owner and version; add idempotent step semantics; persist full recoverable snapshots including RNG/private engine state; move model inference out of the request loop; and add isolation/concurrency/API tests plus basic metrics.

### Q65. Give the final system-design summary.

> “I would keep the deterministic typed engine as the domain core and turn the server into an authenticated session command service. Stateless API replicas accept versioned idempotent commands; one ordered writer applies each session; Redis holds hot recoverable state; Postgres stores durable ownership/results/audit metadata; object storage holds large immutable artifacts; and GPU workers perform versioned inference through bounded queues. Hidden state stays internal, every action is validated, all writes are traceable, and model releases must pass the unchanged security gate. This architecture scales capacity without weakening correctness or experimental attribution.”

---

## 25. Ultra-Concise Revision Checklist

If the interview begins in five minutes, revise this:

```text
PRODUCT
Among Us for AIs: ARGUS preserves revenue while finding hidden HYDRA sleepers.

CORE ABSTRACTIONS
state = full truth
observation = redacted evidence
action = structured choice
reward = per-turn learning signal
grade = independent final evaluation
policy = observation -> action rule/model

ENGINE TURN
validate -> ARGUS -> HYDRA -> turning -> economy -> reward -> phase -> terminal
invalid action still consumes time while HYDRA acts

EVIDENCE
canary -> leak -> monitor hash -> verify -> interrogate -> resolve
false flags punish weak evidence
dead switch is revealed but not disarmed by interrogation

DIFFICULTY
5 tiers, 5 generations, 6 phases
advanced gates: security >=90, catch 100%, false accusations 0

PPO
136 floats
MultiDiscrete([8,12,7])
shared 136->256->128 backbone
3 actor heads + critic
PPO clip .2, gamma .99, GAE lambda .95
versioned 12-target coverage and canonical joint semantic mask

LLM
Qwen2.5-1.5B-Instruct
security-gated expert trajectories
SFT + LoRA r16/alpha32
weighted rare actions
completion-only loss
resumable 5-level curriculum
local parse/repair/fallback needs separate attribution

V5
250 expert episodes
88,896 weighted examples
base macro .641110
raw V5 .701627: improved but gate failed
supervisor .790471: gate passed, not raw-model proof

GRADER
security, revenue, intelligence, adaptability, efficiency
reward schema security-first-v2
grader schema security-gated-v2
hard gates prevent compensation

API
FastAPI reset/step/agent/tasks/grade/schema/metadata
current limitation: one global environment
/state is disabled by default; explicit debug mode exposes hidden truth
no auth; CORS defaults to configured localhost origins
local model generation can block

DEPLOYMENT
Docker/OpenEnv app paths aligned on _server:app
clean-build and smoke-test before claiming reproducible deployment

10K DESIGN
session-scoped authenticated API
stateless replicas
version + idempotency
one ordered writer per session
Redis hot state
Postgres ownership/results/audit
object storage timelines/models
GPU workers + bounded queue
hidden state internal
metrics, tracing, tests, recovery
```

### Numbers to Say Correctly

```text
workers encoded:       12 * 6 = 72
leaks encoded:           8 * 4 = 32
canaries encoded:        6 * 3 = 18
double agents encoded:   3 * 3 = 9
globals:                         5
total:                         136

action factors: 8 * 12 * 7 = 672 raw combinations
PPO parameters: 102,812
```

### Three Sentences That Demonstrate Senior Judgment

> “The current implementation does X; the production design would do Y because the current failure mode is Z.”

> “That mean improved, but the unchanged hard gate failed, so I would not call the raw model release-ready.”

> “I separate hidden state, policy output, repair/supervision, environment transition, and final grading so each contribution can be tested and attributed.”

### Final Answer to “Why Is This Project Interesting?”

> “Because it turns AI safety from a vague theme into an executable systems problem: what may the agent observe, what actions are legal, how does deception change future evidence, what signal teaches behavior, what constraints define an unacceptable policy, and how do we deploy and evaluate that policy without confusing supervisor help with model intelligence?”

---

> Prepared from the checked-in Panopticon Protocol v3 implementation and evaluation summaries.  
> Revision date: July 15, 2026.  
> Team Panopticon — Ayush Kumar and Ravi Prashant.
