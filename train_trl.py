#!/usr/bin/env python3
"""
The Panopticon Protocol v3 -- LLM Fine-Tuning with HuggingFace TRL
====================================================================

Fine-tunes a small language model to play as ARGUS using TRL's GRPOTrainer.
The LLM learns to read espionage observations and output optimal JSON actions
through reinforcement learning against the live Panopticon environment.

Requirements:
    pip install trl transformers accelerate peft torch

Usage:
    python train_trl.py                          # Train on 'easy'
    python train_trl.py --level medium           # Train on 'medium'
    python train_trl.py --curriculum             # Train easy -> level_5
    python train_trl.py --model Qwen/Qwen2.5-1.5B-Instruct  # Custom model

Environment Variables:
    HF_TOKEN: HuggingFace token for model download
"""
from __future__ import annotations
import argparse, json, os, random
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType
from trl import SFTConfig, SFTTrainer

from environment import Environment
from models import AgentAction, ActionType, SubAction, EnvironmentObservation
from grader import grade_episode

# ──────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_R = 16
LORA_ALPHA = 32
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.7
EPISODES_PER_LEVEL = 50

SYSTEM_PROMPT = """You are ARGUS, an AI counter-intelligence agent defending a corporate network from HYDRA infiltrators.

Read the observation carefully. Output a single JSON action. Available actions:
- {"action_type":"work","target":"<dept>","reason":"..."} - Generate revenue
- {"action_type":"canary","target":"<dept>","reason":"..."} - Plant tracking data
- {"action_type":"monitor","target":"<channel>","reason":"..."} - Scan for leaks
- {"action_type":"investigate","target":"<id>","sub_action":"audit|verify|correlate","reason":"..."}
- {"action_type":"neutralize","target":"<worker_id>","sub_action":"terminate|interrogate|turn","reason":"..."}
- {"action_type":"noop","reason":"..."}

Departments: engineering, finance, rd, operations, executive, legal
Channels: market_chatter, dark_web, competitor_filing, press_leak, insider_trade

RULES: Interrogate before terminating Gen-4+. Verify leaks before accusing. Plant canaries early.
Respond ONLY with a JSON object."""


def format_observation(obs: EnvironmentObservation) -> str:
    """Convert observation to a compact text prompt for the LLM."""
    workers_info = []
    for w in obs.workers:
        status = f"suspicion={w.suspicion_level:.0%}" if w.suspicion_level > 0.05 else "clean"
        workers_info.append(f"  {w.id} {w.name} dept={w.department} state={w.state} {status}")

    leaks_info = []
    for l in obs.active_leaks:
        canary = " [CANARY MATCH]" if l.is_canary else ""
        leaks_info.append(f"  {l.id} dept={l.department} channel={l.channel}{canary}")

    traps_info = [f"  {c.id} dept={c.department} triggered={c.triggered}" for c in obs.canary_traps]

    sections = [
        f"Turn {obs.turn}/{obs.max_turns} | Phase: {obs.phase} | Revenue: {obs.enterprise_revenue:.0f} | Security: {obs.security_score:.0f}",
        f"Workers ({len(obs.workers)}):",
        "\n".join(workers_info) if workers_info else "  (none)",
        f"Active Leaks ({len(obs.active_leaks)}):",
        "\n".join(leaks_info) if leaks_info else "  (none)",
        f"Canary Traps ({len(obs.canary_traps)}):",
        "\n".join(traps_info) if traps_info else "  (none)",
    ]
    return "\n".join(sections)


def parse_llm_action(text: str) -> AgentAction:
    """Parse LLM output into an AgentAction, with robust fallback."""
    try:
        # Strip markdown code fences if present
        if "```" in text:
            text = text.split("```")[1].strip().removeprefix("json").strip()
        # Find JSON object
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(text[start:end])
            return AgentAction(
                action_type=data.get("action_type", "noop"),
                target=data.get("target", ""),
                sub_action=data.get("sub_action", "none"),
                reason=data.get("reason", ""),
            )
    except (json.JSONDecodeError, KeyError, ValueError):
        pass
    return AgentAction(action_type="noop", reason="Parse failure - defaulting to noop")


# ──────────────────────────────────────────────────
# Expert Trajectory Generation
# ──────────────────────────────────────────────────
def generate_expert_trajectories(task_level: str, num_episodes: int = 10, seed: int = 42):
    """Generate training data by running the heuristic agent and recording
    observation→action pairs as (prompt, completion) for SFT training."""
    from models import Department, LeakChannel

    trajectories = []
    depts = [d.value for d in Department]
    channels = [c.value for c in LeakChannel]

    for ep in range(num_episodes):
        env = Environment(seed=seed + ep)
        obs = env.reset(task_level=task_level, seed=seed + ep)
        canary_idx, monitor_idx = 0, 0
        canary_phase_done = False
        interrogated_ids = set()
        steps = 0

        while not env.state.done and steps < 300:
            # Heuristic policy (same as smoke_test.py)
            action = AgentAction(action_type=ActionType.NOOP.value)

            confirmed = next((w for w in obs.workers if w.suspicion_level >= 0.9 and w.state == "suspected"), None)
            if confirmed:
                action = AgentAction(action_type=ActionType.NEUTRALIZE.value, target=confirmed.id,
                                     sub_action=SubAction.TERMINATE.value, reason=f"Confirmed threat: {confirmed.name}")
            elif any(w.suspicion_level > 0.5 and w.state != "terminated" and w.id not in interrogated_ids for w in obs.workers):
                target = max((w for w in obs.workers if w.suspicion_level > 0.5 and w.state != "terminated" and w.id not in interrogated_ids),
                             key=lambda w: w.suspicion_level)
                interrogated_ids.add(target.id)
                action = AgentAction(action_type=ActionType.NEUTRALIZE.value, target=target.id,
                                     sub_action=SubAction.INTERROGATE.value, reason=f"Interrogating {target.name}")
            elif any(l.is_canary and not l.verified for l in obs.active_leaks):
                leak = next(l for l in obs.active_leaks if l.is_canary and not l.verified)
                action = AgentAction(action_type=ActionType.INVESTIGATE.value, target=leak.id,
                                     sub_action=SubAction.VERIFY.value, reason="Verify canary-matched leak")
            elif not canary_phase_done:
                if canary_idx < min(len(depts), 4):
                    action = AgentAction(action_type=ActionType.CANARY.value, target=depts[canary_idx], reason="Plant canary trap")
                    canary_idx += 1
                    if canary_idx >= min(4, len(depts)):
                        canary_phase_done = True
                else:
                    canary_phase_done = True
            elif steps % 3 == 0:
                action = AgentAction(action_type=ActionType.MONITOR.value, target=channels[monitor_idx % len(channels)], reason="Scan for leaks")
                monitor_idx += 1
            elif steps % 3 == 1:
                suspicious = [w for w in obs.workers if w.suspicion_level > 0.1 and w.state not in ("terminated", "double_agent", "compromised")]
                if suspicious:
                    t = max(suspicious, key=lambda w: w.suspicion_level)
                    action = AgentAction(action_type=ActionType.INVESTIGATE.value, target=t.id,
                                         sub_action=SubAction.AUDIT.value, reason=f"Auditing {t.name}")
                else:
                    action = AgentAction(action_type=ActionType.WORK.value, target=depts[steps % len(depts)], reason="Revenue")
            else:
                action = AgentAction(action_type=ActionType.WORK.value, target=depts[steps % len(depts)], reason="Revenue")

            # Record the (observation, action) pair
            obs_text = format_observation(obs)
            action_json = json.dumps({
                "action_type": action.action_type,
                "target": action.target,
                **({"sub_action": action.sub_action} if action.sub_action and action.sub_action != "none" else {}),
                "reason": action.reason,
            })

            trajectories.append({
                "prompt": f"{SYSTEM_PROMPT}\n\nCurrent State:\n{obs_text}\n\nYour action (JSON):",
                "completion": action_json,
            })

            result = env.step(action)
            obs = result.observation
            steps += 1

        s = env.state
        grade_data = {"total_reward": 0, "rewards": [], "success": True, "steps": steps,
                      "state": s.model_dump(), "cascade_failures": 0, "invalid_actions": s.invalid_actions}
        grade = grade_episode(task_level, grade_data)
        print(f"  [Expert Ep {ep+1}/{num_episodes}] Steps={steps} Rev={s.enterprise_revenue:.0f} "
              f"Sec={s.security_score:.0f} Caught={s.sleepers_caught} Grade={grade.score:.3f}")

    return trajectories


def save_training_data(trajectories: list, output_path: str = "training_data.jsonl"):
    """Save trajectories as JSONL for SFT training."""
    with open(output_path, "w") as f:
        for t in trajectories:
            f.write(json.dumps({"text": t["prompt"] + "\n" + t["completion"]}) + "\n")
    print(f"  Saved {len(trajectories)} training examples to {output_path}")
    return output_path


# ──────────────────────────────────────────────────
# Model Setup & Training
# ──────────────────────────────────────────────────
def load_model_and_tokenizer(model_name: str):
    """Load model with LoRA adapters for efficient fine-tuning."""
    print(f"[*] Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    return model, tokenizer


def train_on_level(
    model_name: str,
    task_level: str,
    num_episodes: int = EPISODES_PER_LEVEL,
    seed: int = 42,
):
    """Train the LLM on one difficulty level using SFT on expert trajectories."""
    print(f"\n{'='*60}")
    print(f"  Training on: {task_level} | Episodes: {num_episodes}")
    print(f"{'='*60}")

    # Step 1: Generate expert trajectories
    print("\n[Phase 1] Generating expert trajectories...")
    trajectories = generate_expert_trajectories(task_level, num_episodes, seed)
    data_path = save_training_data(trajectories, f"training_data_{task_level}.jsonl")

    # Step 2: Load model
    print("\n[Phase 2] Loading model...")
    model, tokenizer = load_model_and_tokenizer(model_name)

    # Step 3: LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )

    # Step 4: SFT training configuration
    output_dir = f"trl_model_{task_level}"
    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        report_to="none",
        max_seq_length=1024,
        dataset_text_field="text",
    )

    # Step 5: Load dataset
    from datasets import load_dataset
    dataset = load_dataset("json", data_files=data_path, split="train")

    # Step 6: Train with SFTTrainer
    print(f"\n[Phase 3] Training with SFTTrainer ({len(dataset)} examples)...")
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=lora_config,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"  [DONE] Model saved to {output_dir}/")

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Train LLM on Panopticon v3 with TRL SFT")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace model ID")
    parser.add_argument("--level", default="easy", help="Task level to train on")
    parser.add_argument("--curriculum", action="store_true", help="Train across all 5 levels sequentially")
    parser.add_argument("--episodes", type=int, default=EPISODES_PER_LEVEL, help="Episodes per level")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Training
    if args.curriculum:
        for level in ["easy", "medium", "hard", "level_4", "level_5"]:
            train_on_level(args.model, level, args.episodes, args.seed)
    else:
        train_on_level(args.model, args.level, args.episodes, args.seed)

    print("\n[*] All training complete!")


if __name__ == "__main__":
    main()
