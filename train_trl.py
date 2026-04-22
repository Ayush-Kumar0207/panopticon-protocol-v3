#!/usr/bin/env python3
"""
The Panopticon Protocol v3 -- LLM Fine-Tuning with HuggingFace TRL
====================================================================

Fine-tunes a small language model to play as ARGUS using TRL's PPOTrainer.
The LLM learns to read espionage observations and output optimal JSON actions
through reinforcement learning against the live Panopticon environment.

Requirements:
    pip install trl transformers accelerate peft bitsandbytes torch

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
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead

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
STEPS_PER_PPO_BATCH = 8  # collect this many steps before PPO update

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
# Model Setup
# ──────────────────────────────────────────────────
def load_model_and_tokenizer(model_name: str, device: str = "auto"):
    """Load model with LoRA adapters for efficient fine-tuning."""
    print(f"[*] Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA config for parameter-efficient training
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )

    # Load model with value head for PPO
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device,
        trust_remote_code=True,
        peft_config=lora_config,
    )

    return model, tokenizer


# ──────────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────────
def train_on_level(
    model,
    tokenizer,
    ppo_trainer: PPOTrainer,
    task_level: str,
    num_episodes: int = EPISODES_PER_LEVEL,
    seed: int = 42,
):
    """Train the LLM on one difficulty level using PPO."""
    print(f"\n{'='*60}")
    print(f"  Training on: {task_level} | Episodes: {num_episodes}")
    print(f"{'='*60}")

    device = model.pretrained_model.device if hasattr(model, "pretrained_model") else "cpu"
    reward_history = []
    best_reward = float("-inf")

    for episode in range(num_episodes):
        env = Environment(seed=seed + episode)
        obs = env.reset(task_level=task_level, seed=seed + episode)

        episode_queries = []
        episode_responses = []
        episode_rewards = []
        step_count = 0

        while not env.state.done and step_count < 300:
            # Format observation as text prompt
            obs_text = format_observation(obs)
            prompt = f"{SYSTEM_PROMPT}\n\nCurrent State:\n{obs_text}\n\nYour action (JSON):"

            # Tokenize
            input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=1024)
            input_ids = input_ids.to(device)

            # Generate response
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )

            # Extract only the generated tokens
            response_ids = output_ids[0][input_ids.shape[1]:]
            response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

            # Parse action
            action = parse_llm_action(response_text)

            # Step environment
            rev_before, sec_before = env.state.enterprise_revenue, env.state.security_score
            result = env.step(action)
            obs = result.observation
            reward = result.reward

            # Store for PPO batch
            episode_queries.append(input_ids.squeeze(0))
            episode_responses.append(response_ids)
            episode_rewards.append(reward)
            step_count += 1

            # PPO update when we have enough steps
            if len(episode_queries) >= STEPS_PER_PPO_BATCH:
                reward_tensors = [torch.tensor([r], dtype=torch.float32) for r in episode_rewards[-STEPS_PER_PPO_BATCH:]]
                try:
                    stats = ppo_trainer.step(
                        episode_queries[-STEPS_PER_PPO_BATCH:],
                        episode_responses[-STEPS_PER_PPO_BATCH:],
                        reward_tensors,
                    )
                except Exception as e:
                    # PPO step can fail with shape mismatches on early batches
                    pass

        total_reward = sum(episode_rewards)
        reward_history.append(total_reward)

        s = env.state
        if total_reward > best_reward:
            best_reward = total_reward
            # Save best checkpoint
            model.save_pretrained(f"best_trl_{task_level}")
            tokenizer.save_pretrained(f"best_trl_{task_level}")

        if (episode + 1) % 5 == 0 or episode == 0:
            avg = sum(reward_history[-10:]) / min(len(reward_history), 10)
            print(
                f"  [Ep {episode+1:3d}/{num_episodes}] "
                f"Reward={total_reward:7.2f} Avg10={avg:7.2f} "
                f"Rev={s.enterprise_revenue:.0f} Sec={s.security_score:.0f} "
                f"Caught={s.sleepers_caught} Steps={step_count}"
            )

    # Save final model
    model.save_pretrained(f"final_trl_{task_level}")
    tokenizer.save_pretrained(f"final_trl_{task_level}")
    print(f"  [DONE] Best reward: {best_reward:.2f} | Saved: final_trl_{task_level}/")

    return reward_history


def main():
    parser = argparse.ArgumentParser(description="Train LLM on Panopticon v3 with TRL PPO")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace model ID")
    parser.add_argument("--level", default="easy", help="Task level to train on")
    parser.add_argument("--curriculum", action="store_true", help="Train across all 5 levels sequentially")
    parser.add_argument("--episodes", type=int, default=EPISODES_PER_LEVEL, help="Episodes per level")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model)

    # PPO configuration
    ppo_config = PPOConfig(
        model_name=args.model,
        batch_size=STEPS_PER_PPO_BATCH,
        mini_batch_size=min(4, STEPS_PER_PPO_BATCH),
        learning_rate=1.41e-5,
        log_with=None,  # Set to "wandb" for experiment tracking
        ppo_epochs=4,
        gradient_accumulation_steps=1,
        optimize_cuda_cache=True if torch.cuda.is_available() else False,
    )

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        tokenizer=tokenizer,
    )

    # Training
    all_rewards = {}
    if args.curriculum:
        for level in ["easy", "medium", "hard", "level_4", "level_5"]:
            rewards = train_on_level(model, tokenizer, ppo_trainer, level, args.episodes, args.seed)
            all_rewards[level] = rewards
    else:
        rewards = train_on_level(model, tokenizer, ppo_trainer, args.level, args.episodes, args.seed)
        all_rewards[args.level] = rewards

    # Save reward history for plotting
    with open("trl_reward_history.json", "w") as f:
        json.dump(all_rewards, f, indent=2)
    print(f"\nReward history saved to trl_reward_history.json")


if __name__ == "__main__":
    main()
