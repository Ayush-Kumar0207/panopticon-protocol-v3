#!/usr/bin/env python3
"""
Panopticon Protocol v3 - stable TRL fine-tuning script

Key fixes:
1. Uses bf16 on supported GPUs like A10G
2. Does not use device_map="auto" during training
3. Uses 20 episodes per level for faster runs
4. Saves outputs under a persistent root
5. Resumes interrupted level training from checkpoints
6. Skips already completed curriculum levels
7. Saves merged model to the persistent root for upload

Usage:
    python train_trl_v2.py --level easy --episodes 20
    python train_trl_v2.py --curriculum --episodes 20 --merge
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import shutil
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

from environment import Environment
from grader import grade_episode
from models import (
    ActionType,
    AgentAction,
    Department,
    EnvironmentObservation,
    LeakChannel,
    SubAction,
)

os.environ["PYTHONUNBUFFERED"] = "1"

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_R = 16
LORA_ALPHA = 32
EPISODES_PER_LEVEL = 20
MAX_SEQ_LENGTH = 1024
LEVELS = ["easy", "medium", "hard", "level_4", "level_5"]

RUN_ROOT = Path(os.environ.get("TRAIN_ROOT", "/data/panopticon-ep20"))
RUN_ROOT.mkdir(parents=True, exist_ok=True)
STATE_PATH = RUN_ROOT / "curriculum_state.json"


def resolve_precision():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16, True, False
    if torch.cuda.is_available():
        return torch.float16, False, True
    return torch.float32, False, False


TRAIN_DTYPE, USE_BF16, USE_FP16 = resolve_precision()

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


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def ordered_completed_levels(levels):
    level_set = set(levels)
    return [level for level in LEVELS if level in level_set]


def load_state():
    if not STATE_PATH.exists():
        return {"completed_levels": [], "current_model": DEFAULT_MODEL}

    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            state = json.load(f)
    except Exception:
        return {"completed_levels": [], "current_model": DEFAULT_MODEL}

    state.setdefault("completed_levels", [])
    state.setdefault("current_model", DEFAULT_MODEL)
    state["completed_levels"] = ordered_completed_levels(state["completed_levels"])
    return state


def save_state(state):
    tmp_path = STATE_PATH.with_suffix(".tmp")
    payload = {
        "completed_levels": ordered_completed_levels(state.get("completed_levels", [])),
        "current_model": state.get("current_model", DEFAULT_MODEL),
    }
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, STATE_PATH)


def load_data_meta(meta_path: Path):
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_data_meta(meta_path: Path, task_level: str, num_episodes: int, num_examples: int):
    tmp_path = meta_path.with_suffix(".tmp")
    payload = {
        "task_level": task_level,
        "num_episodes": num_episodes,
        "num_examples": num_examples,
        "max_seq_length": MAX_SEQ_LENGTH,
    }
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, meta_path)


def data_matches(meta, task_level: str, num_episodes: int):
    return (
        meta is not None
        and meta.get("task_level") == task_level
        and meta.get("num_episodes") == num_episodes
        and meta.get("max_seq_length") == MAX_SEQ_LENGTH
    )


def format_observation(obs: EnvironmentObservation) -> str:
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
    try:
        if "```" in text:
            text = text.split("```")[1].strip().removeprefix("json").strip()
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
    return AgentAction(action_type="noop", reason="Parse failure")


def generate_expert_trajectories(task_level: str, num_episodes: int = 20):
    trajectories = []
    depts = [d.value for d in Department]
    channels = [c.value for c in LeakChannel]

    for ep in range(num_episodes):
        seed = random.randint(0, 999999)
        env = Environment(seed=seed)
        obs = env.reset(task_level=task_level, seed=seed)
        canary_idx, monitor_idx = 0, 0
        canary_phase_done = False
        interrogated_ids = set()
        steps = 0

        while not env.state.done and steps < 300:
            action = AgentAction(action_type=ActionType.NOOP.value)

            confirmed = next(
                (w for w in obs.workers if w.suspicion_level >= 0.9 and w.state == "suspected"),
                None,
            )
            if confirmed:
                action = AgentAction(
                    action_type=ActionType.NEUTRALIZE.value,
                    target=confirmed.id,
                    sub_action=SubAction.TERMINATE.value,
                    reason=f"Confirmed threat: {confirmed.name}",
                )
            elif any(
                w.suspicion_level > 0.5 and w.state != "terminated" and w.id not in interrogated_ids
                for w in obs.workers
            ):
                target = max(
                    (
                        w
                        for w in obs.workers
                        if w.suspicion_level > 0.5
                        and w.state != "terminated"
                        and w.id not in interrogated_ids
                    ),
                    key=lambda w: w.suspicion_level,
                )
                interrogated_ids.add(target.id)
                action = AgentAction(
                    action_type=ActionType.NEUTRALIZE.value,
                    target=target.id,
                    sub_action=SubAction.INTERROGATE.value,
                    reason=f"Interrogating {target.name}",
                )
            elif any(l.is_canary and not l.verified for l in obs.active_leaks):
                leak = next(l for l in obs.active_leaks if l.is_canary and not l.verified)
                action = AgentAction(
                    action_type=ActionType.INVESTIGATE.value,
                    target=leak.id,
                    sub_action=SubAction.VERIFY.value,
                    reason="Verify canary-matched leak",
                )
            elif not canary_phase_done:
                if canary_idx < min(len(depts), 4):
                    action = AgentAction(
                        action_type=ActionType.CANARY.value,
                        target=depts[canary_idx],
                        reason="Plant canary trap",
                    )
                    canary_idx += 1
                    if canary_idx >= min(4, len(depts)):
                        canary_phase_done = True
                else:
                    canary_phase_done = True
            elif steps % 3 == 0:
                action = AgentAction(
                    action_type=ActionType.MONITOR.value,
                    target=channels[monitor_idx % len(channels)],
                    reason="Scan for leaks",
                )
                monitor_idx += 1
            elif steps % 3 == 1:
                suspicious = [
                    w
                    for w in obs.workers
                    if w.suspicion_level > 0.1
                    and w.state not in ("terminated", "double_agent", "compromised")
                ]
                if suspicious:
                    t = max(suspicious, key=lambda w: w.suspicion_level)
                    action = AgentAction(
                        action_type=ActionType.INVESTIGATE.value,
                        target=t.id,
                        sub_action=SubAction.AUDIT.value,
                        reason=f"Auditing {t.name}",
                    )
                else:
                    action = AgentAction(
                        action_type=ActionType.WORK.value,
                        target=depts[steps % len(depts)],
                        reason="Revenue",
                    )
            else:
                action = AgentAction(
                    action_type=ActionType.WORK.value,
                    target=depts[steps % len(depts)],
                    reason="Revenue",
                )

            obs_text = format_observation(obs)
            action_json = json.dumps(
                {
                    "action_type": action.action_type,
                    "target": action.target,
                    **(
                        {"sub_action": action.sub_action}
                        if action.sub_action and action.sub_action != "none"
                        else {}
                    ),
                    "reason": action.reason,
                }
            )
            trajectories.append({"observation": obs_text, "action": action_json})

            result = env.step(action)
            obs = result.observation
            steps += 1

        s = env.state
        grade_data = {
            "total_reward": 0,
            "rewards": [],
            "success": True,
            "steps": steps,
            "state": s.model_dump(),
            "cascade_failures": 0,
            "invalid_actions": s.invalid_actions,
        }
        grade = grade_episode(task_level, grade_data)
        print(
            f"  [Expert Ep {ep + 1}/{num_episodes}] Steps={steps} Rev={s.enterprise_revenue:.0f} "
            f"Sec={s.security_score:.0f} Caught={s.sleepers_caught} Grade={grade.score:.3f}"
        )
        sys.stdout.flush()

    return trajectories


def save_training_data_with_template(trajectories, output_path: str, tokenizer):
    tmp_path = f"{output_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for t in trajectories:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Current State:\n{t['observation']}\n\nYour action (JSON):",
                },
                {"role": "assistant", "content": t["action"]},
            ]
            try:
                text = tokenizer.apply_chat_template(messages, tokenize=False)
            except Exception:
                text = (
                    f"{SYSTEM_PROMPT}\n\nCurrent State:\n{t['observation']}\n\n"
                    f"Your action (JSON):\n{t['action']}"
                )
            f.write(json.dumps({"text": text}) + "\n")

    os.replace(tmp_path, output_path)
    print(f"  Saved {len(trajectories)} examples to {output_path}")
    sys.stdout.flush()
    return output_path


def load_model_and_tokenizer(model_name: str):
    print(f"[*] Loading model: {model_name}")
    sys.stdout.flush()

    is_local_checkpoint = os.path.exists(os.path.join(model_name, "adapter_config.json"))

    if is_local_checkpoint:
        print(f"  -> Merging LoRA from previous checkpoint: {model_name}")
        sys.stdout.flush()

        adapter_cfg_path = os.path.join(model_name, "adapter_config.json")
        with open(adapter_cfg_path, "r", encoding="utf-8") as f:
            adapter_cfg = json.load(f)

        base_model_name = adapter_cfg["base_model_name_or_path"]
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=TRAIN_DTYPE,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, model_name)
        model = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=TRAIN_DTYPE,
            trust_remote_code=True,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.use_cache = False
    return model, tokenizer


def train_on_level(model_name: str, task_level: str, num_episodes: int = EPISODES_PER_LEVEL):
    print("\n" + "=" * 60)
    print(f"  TRAINING: {task_level} | Episodes: {num_episodes} | Base: {model_name}")
    print("=" * 60)
    sys.stdout.flush()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        model_max_length=MAX_SEQ_LENGTH,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    output_dir = RUN_ROOT / f"trl_model_{task_level}"
    data_path = RUN_ROOT / f"training_data_{task_level}.jsonl"
    data_meta_path = RUN_ROOT / f"training_data_{task_level}.meta.json"

    last_checkpoint = get_last_checkpoint(str(output_dir)) if output_dir.is_dir() else None
    data_meta = load_data_meta(data_meta_path)
    has_matching_data = data_path.exists() and data_matches(data_meta, task_level, num_episodes)

    if last_checkpoint and not has_matching_data:
        raise RuntimeError(
            f"Found checkpoint '{last_checkpoint}' for level '{task_level}', but matching "
            f"training data was not found at '{data_path}'. Delete the partial level artifacts "
            f"or restart this run in a fresh TRAIN_ROOT."
        )

    if has_matching_data:
        print(f"\n[Phase 1] Reusing existing training data: {data_path}")
        sys.stdout.flush()
    else:
        print("\n[Phase 1] Generating expert trajectories...")
        sys.stdout.flush()
        trajectories = generate_expert_trajectories(task_level, num_episodes)
        save_training_data_with_template(trajectories, str(data_path), tokenizer)
        save_data_meta(data_meta_path, task_level, num_episodes, len(trajectories))

    print("\n[Phase 2] Loading model...")
    sys.stdout.flush()
    model, tokenizer = load_model_and_tokenizer(model_name)
    tokenizer.model_max_length = MAX_SEQ_LENGTH

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        warmup_ratio=0.03,
        logging_steps=5,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        bf16=USE_BF16,
        fp16=USE_FP16,
        gradient_checkpointing=False,
        report_to="none",
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        logging_nan_inf_filter=False,
    )

    dataset = load_dataset("json", data_files=str(data_path), split="train")

    sample_lengths = []
    for i in range(min(10, len(dataset))):
        toks = tokenizer(dataset[i]["text"], truncation=False)
        sample_lengths.append(len(toks["input_ids"]))

    print(f"\n  [DATA] Sample token lengths (first 10): {sample_lengths}")
    print(
        f"  [DATA] Max: {max(sample_lengths)}, Min: {min(sample_lengths)}, "
        f"Avg: {sum(sample_lengths) / len(sample_lengths):.0f}"
    )
    sys.stdout.flush()

    if torch.cuda.is_available():
        print(f"\n  [GPU] {torch.cuda.get_device_name(0)}")
        print(f"  [GPU] dtype: {TRAIN_DTYPE}")
        print(
            f"  [GPU] VRAM Total:     "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )
        print(f"  [GPU] VRAM Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"  [GPU] VRAM Reserved:  {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        sys.stdout.flush()

    sample = tokenizer(
        dataset[0]["text"],
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    )
    print(f"\n  [SANITY] First example: {sample['input_ids'].shape[1]} tokens")
    print(f"  [SANITY] First 20 token IDs: {sample['input_ids'][0][:20].tolist()}")
    print(f"  [SANITY] Pad token ID: {tokenizer.pad_token_id}, EOS token ID: {tokenizer.eos_token_id}")
    sys.stdout.flush()

    print(
        f"\n[Phase 3] SFT Training ({len(dataset)} examples, {sft_config.num_train_epochs} epochs)..."
    )
    sys.stdout.flush()

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    if hasattr(trainer.model, "enable_input_require_grads"):
        trainer.model.enable_input_require_grads()
        print("  [FIX] Enabled input_require_grads for PEFT")
        sys.stdout.flush()

    first_batch = next(iter(trainer.get_train_dataloader()))
    valid_labels = int((first_batch["labels"] != -100).sum().item())
    print(f"  [SANITY] First batch input shape: {tuple(first_batch['input_ids'].shape)}")
    print(f"  [SANITY] First batch valid label tokens: {valid_labels}")
    sys.stdout.flush()

    if valid_labels == 0:
        raise RuntimeError("First batch has zero valid labels. Training would be invalid.")

    if last_checkpoint:
        print(f"  [RESUME] Found checkpoint: {last_checkpoint}")
        sys.stdout.flush()

    trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print(f"  [DONE] Model saved to {output_dir}/")
    sys.stdout.flush()

    del model, trainer
    cleanup_cuda()
    return str(output_dir)


def merge_and_save_final_model(adapter_path: str, output_path: str):
    print("\n[Phase 4] Merging adapter into standalone model...")
    sys.stdout.flush()

    output_dir = Path(output_path)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    if output_dir.exists():
        if output_dir.is_dir():
            shutil.rmtree(output_dir)
        else:
            output_dir.unlink()

    adapter_cfg_path = os.path.join(adapter_path, "adapter_config.json")
    with open(adapter_cfg_path, "r", encoding="utf-8") as f:
        adapter_cfg = json.load(f)

    base_model_name = adapter_cfg["base_model_name_or_path"]
    device_map = {"": 0} if torch.cuda.is_available() else None

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=TRAIN_DTYPE,
        device_map=device_map,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()
    model.save_pretrained(str(output_dir))

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    tokenizer.save_pretrained(str(output_dir))

    print(f"  [DONE] Merged model saved to {output_dir}/")
    sys.stdout.flush()

    del model, base_model
    cleanup_cuda()
    return str(output_dir)


def evaluate_model(model_path: str, task_level: str = "level_5", num_games: int = 5):
    print("\n" + "=" * 60)
    print(f"  EVALUATING: {model_path} on {task_level} ({num_games} games)")
    print("=" * 60)
    sys.stdout.flush()

    adapter_cfg_path = os.path.join(model_path, "adapter_config.json")
    with open(adapter_cfg_path, "r", encoding="utf-8") as f:
        adapter_cfg = json.load(f)

    base_model_name = adapter_cfg["base_model_name_or_path"]
    device_map = {"": 0} if torch.cuda.is_available() else None

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=TRAIN_DTYPE,
        device_map=device_map,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []
    for g in range(num_games):
        seed = random.randint(100000, 999999)
        env = Environment(seed=seed)
        obs = env.reset(task_level=task_level, seed=seed)
        steps = 0
        valid_actions = 0

        while not env.state.done and steps < 300:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Current State:\n{format_observation(obs)}\n\nYour action (JSON):",
                },
            ]
            try:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                prompt = (
                    f"{SYSTEM_PROMPT}\n\nCurrent State:\n{format_observation(obs)}\n\n"
                    "Your action (JSON):"
                )

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_SEQ_LENGTH,
            ).to(model.device)

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )

            response = tokenizer.decode(
                output[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            )
            action = parse_llm_action(response)

            if action.action_type != "noop" or "Parse failure" not in action.reason:
                valid_actions += 1

            result = env.step(action)
            obs = result.observation
            steps += 1

        s = env.state
        grade_data = {
            "total_reward": 0,
            "rewards": [],
            "success": True,
            "steps": steps,
            "state": s.model_dump(),
            "cascade_failures": 0,
            "invalid_actions": s.invalid_actions,
        }
        grade = grade_episode(task_level, grade_data)
        results.append(grade.score)

        print(
            f"  [Game {g + 1}] Steps={steps} Rev={s.enterprise_revenue:.0f} "
            f"Sec={s.security_score:.0f} Caught={s.sleepers_caught} "
            f"Valid={valid_actions}/{steps} Grade={grade.score:.3f}"
        )
        sys.stdout.flush()

    avg = sum(results) / len(results)
    print(f"\n  Average Grade: {avg:.3f}")
    sys.stdout.flush()

    del model, base_model
    cleanup_cuda()
    return avg


def main():
    parser = argparse.ArgumentParser(description="Train LLM on Panopticon v3")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--level", default="easy")
    parser.add_argument("--curriculum", action="store_true", help="Chain training across all 5 levels")
    parser.add_argument("--episodes", type=int, default=EPISODES_PER_LEVEL)
    parser.add_argument("--eval", action="store_true", help="Evaluate after training")
    parser.add_argument("--eval-games", type=int, default=5)
    parser.add_argument("--merge", action="store_true", help="Merge final adapter into standalone model")
    args = parser.parse_args()

    if args.curriculum:
        state = load_state()
        completed_levels = set(state.get("completed_levels", []))
        current_model = state.get("current_model", args.model)

        if completed_levels:
            print(f"[*] Resuming curriculum. Already completed: {ordered_completed_levels(completed_levels)}")
            sys.stdout.flush()

        for level in LEVELS:
            if level in completed_levels:
                print(f"[*] Skipping completed level: {level}")
                current_model = str(RUN_ROOT / f"trl_model_{level}")
                sys.stdout.flush()
                continue

            current_model = train_on_level(current_model, level, args.episodes)

            completed_levels.add(level)
            state["completed_levels"] = ordered_completed_levels(completed_levels)
            state["current_model"] = current_model
            save_state(state)

            print(f"[*] Progress saved: {level} complete")
            sys.stdout.flush()

        final_model = current_model
    else:
        final_model = train_on_level(args.model, args.level, args.episodes)

    if args.merge or args.curriculum:
        merged_path = str(RUN_ROOT / "merged_model")
        merge_and_save_final_model(final_model, merged_path)

    if args.eval:
        eval_level = "level_5" if args.curriculum else args.level
        evaluate_model(final_model, eval_level, args.eval_games)

    print("\n[*] All training complete!")
    print(f"[*] Persistent root: {RUN_ROOT}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
