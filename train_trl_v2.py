#!/usr/bin/env python3
"""
Panopticon Protocol v3 - stable TRL fine-tuning script

Key fixes:
1. Uses bf16 on supported GPUs like A10G
2. Does not use device_map="auto" during training
3. Defaults to 20 episodes per level locally, but supports higher counts via --episodes
4. Saves outputs under a persistent root
5. Resumes interrupted level training from checkpoints
6. Skips already completed curriculum levels
7. Saves merged model to the persistent root for upload

Usage:
    python train_trl_v2.py --level easy --episodes 20
    python train_trl_v2.py --curriculum --episodes 20 --merge
    python train_trl_v2.py --curriculum --episodes 50 --merge
"""

from __future__ import annotations

import argparse
import inspect
import gc
import json
import os
import random
import shutil
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

try:
    from trl import DataCollatorForCompletionOnlyLM
except ImportError:  # pragma: no cover - depends on the installed TRL build
    DataCollatorForCompletionOnlyLM = None

from environment import Environment
from grader import grade_episode
from models import (
    ActionType,
    AgentAction,
    Department,
    EnvironmentObservation,
    SubAction,
)
from security_policy import (
    choose_security_first_action,
    expert_episode_meets_security_gate,
    new_security_expert_state,
)

os.environ["PYTHONUNBUFFERED"] = "1"


def disable_incompatible_torchao():
    try:
        torchao_version = version("torchao")
    except PackageNotFoundError:
        return

    def parse_parts(raw_version):
        parts = []
        for token in raw_version.replace("-", ".").split("."):
            if token.isdigit():
                parts.append(int(token))
            else:
                break
        return tuple(parts)

    if parse_parts(torchao_version) >= (0, 16, 0):
        return

    try:
        import peft.import_utils as peft_import_utils

        peft_import_utils.is_torchao_available = lambda: False
    except Exception:
        pass

    try:
        import peft.tuners.lora.torchao as peft_torchao

        peft_torchao.is_torchao_available = lambda: False
    except Exception:
        pass

    print(
        f"[*] Disabled incompatible torchao {torchao_version}. "
        "PEFT LoRA training does not require torchao for this notebook run."
    )
    sys.stdout.flush()


disable_incompatible_torchao()

GPU_DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
CPU_BASIC_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_MODEL = GPU_DEFAULT_MODEL
LORA_R = 16
LORA_ALPHA = 32
EPISODES_PER_LEVEL = 20  # Default/fallback only; production GPU runs can override with --episodes 50
DEFAULT_MAX_SEQ_LENGTH = 1024
CPU_BASIC_MAX_SEQ_LENGTH = 384
LOW_VRAM_GPU_MAX_SEQ_LENGTH = 512
DEFAULT_TRAIN_EPOCHS = 3
CPU_BASIC_TRAIN_EPOCHS = 1
LOW_VRAM_GPU_TRAIN_EPOCHS = 2
DEFAULT_BATCH_SIZE = 2
CPU_BASIC_BATCH_SIZE = 1
LOW_VRAM_GPU_BATCH_SIZE = 1
DEFAULT_GRAD_ACCUM = 4
CPU_BASIC_GRAD_ACCUM = 4
LOW_VRAM_GPU_GRAD_ACCUM = 8
DEFAULT_SAVE_STEPS = 50
CPU_BASIC_SAVE_STEPS = 5
LOW_VRAM_GPU_SAVE_STEPS = 25
LOW_VRAM_GPU_THRESHOLD_GB = 20
TRAJECTORY_SCHEMA_VERSION = "curriculum-expert-v5-security-first"
LEVELS = ["easy", "medium", "hard", "level_4", "level_5"]

# Default local/persistent root; HF worker Spaces override this via ENV TRAIN_ROOT.
RUN_ROOT = Path(os.environ.get("TRAIN_ROOT", "/data/panopticon-ep20"))
RUN_ROOT.mkdir(parents=True, exist_ok=True)
STATE_PATH = RUN_ROOT / "curriculum_state.json"
EVENTS_PATH = RUN_ROOT / "training_events.jsonl"
MAX_SEQ_LENGTH = DEFAULT_MAX_SEQ_LENGTH
NUM_TRAIN_EPOCHS = DEFAULT_TRAIN_EPOCHS
PER_DEVICE_TRAIN_BATCH_SIZE = DEFAULT_BATCH_SIZE
GRADIENT_ACCUMULATION_STEPS = DEFAULT_GRAD_ACCUM
SAVE_STEPS = DEFAULT_SAVE_STEPS
GRADIENT_CHECKPOINTING = False
CPU_BASIC_SAFE_MODE = False


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _format_log_value(value):
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def log_event(event: str, *, echo: bool = True, **fields):
    payload = {"event": event, "time_unix": round(time.time(), 3), **fields}
    line = json.dumps(payload, sort_keys=True, default=_json_default)
    if echo:
        print("[TRAIN_EVENT] " + line)
    try:
        with open(EVENTS_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError as exc:
        if echo:
            print(f"[WARN] Could not append training event log: {exc}")
    sys.stdout.flush()


class CurriculumProgressCallback(TrainerCallback):
    def __init__(self, task_level: str, level_index: int, total_levels: int):
        self.task_level = task_level
        self.level_index = max(1, level_index)
        self.total_levels = max(1, total_levels)

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        max_steps = max(1, int(state.max_steps or 1))
        step = int(state.global_step or 0)
        level_pct = min(100.0, 100.0 * step / max_steps)
        curriculum_pct = min(
            100.0,
            100.0 * ((self.level_index - 1) + (level_pct / 100.0)) / self.total_levels,
        )
        print(
            "[TRAIN_PROGRESS] "
            f"level={self.task_level} "
            f"step={step}/{max_steps} "
            f"level_pct={level_pct:.1f}% "
            f"curriculum_pct={curriculum_pct:.1f}% "
            f"epoch={_format_log_value(state.epoch)} "
            f"loss={_format_log_value(logs.get('loss'))} "
            f"grad_norm={_format_log_value(logs.get('grad_norm'))} "
            f"lr={_format_log_value(logs.get('learning_rate'))}"
        )
        log_event(
            "train_progress",
            level=self.task_level,
            step=step,
            max_steps=max_steps,
            level_pct=round(level_pct, 3),
            curriculum_pct=round(curriculum_pct, 3),
            epoch=state.epoch,
            loss=logs.get("loss"),
            grad_norm=logs.get("grad_norm"),
            learning_rate=logs.get("learning_rate"),
        )
        sys.stdout.flush()


def resolve_precision():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16, True, False
    if torch.cuda.is_available():
        return torch.float16, False, True
    return torch.float32, False, False


TRAIN_DTYPE, USE_BF16, USE_FP16 = resolve_precision()


def is_low_vram_gpu():
    if not torch.cuda.is_available():
        return False
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return total_gb < LOW_VRAM_GPU_THRESHOLD_GB


def runtime_profile_name(running_on_cpu, low_vram_gpu):
    if running_on_cpu or CPU_BASIC_SAFE_MODE:
        return "cpu-basic-safe"
    if low_vram_gpu:
        return "low-vram-gpu-safe"
    return "default-gpu"


def current_runtime_profile_name():
    return runtime_profile_name(not torch.cuda.is_available(), is_low_vram_gpu())


def configure_runtime(args):
    global TRAIN_DTYPE
    global USE_BF16
    global USE_FP16
    global MAX_SEQ_LENGTH
    global NUM_TRAIN_EPOCHS
    global PER_DEVICE_TRAIN_BATCH_SIZE
    global GRADIENT_ACCUMULATION_STEPS
    global SAVE_STEPS
    global GRADIENT_CHECKPOINTING
    global CPU_BASIC_SAFE_MODE

    TRAIN_DTYPE, USE_BF16, USE_FP16 = resolve_precision()
    running_on_cpu = not torch.cuda.is_available()
    low_vram_gpu = is_low_vram_gpu()
    CPU_BASIC_SAFE_MODE = running_on_cpu or args.cpu_basic_safe

    if CPU_BASIC_SAFE_MODE:
        if args.model == GPU_DEFAULT_MODEL:
            args.model = CPU_BASIC_MODEL
        MAX_SEQ_LENGTH = args.max_seq_length or CPU_BASIC_MAX_SEQ_LENGTH
        NUM_TRAIN_EPOCHS = args.epochs or CPU_BASIC_TRAIN_EPOCHS
        PER_DEVICE_TRAIN_BATCH_SIZE = CPU_BASIC_BATCH_SIZE
        GRADIENT_ACCUMULATION_STEPS = CPU_BASIC_GRAD_ACCUM
        SAVE_STEPS = CPU_BASIC_SAVE_STEPS
        GRADIENT_CHECKPOINTING = True
    elif low_vram_gpu:
        MAX_SEQ_LENGTH = args.max_seq_length or LOW_VRAM_GPU_MAX_SEQ_LENGTH
        NUM_TRAIN_EPOCHS = args.epochs or LOW_VRAM_GPU_TRAIN_EPOCHS
        PER_DEVICE_TRAIN_BATCH_SIZE = LOW_VRAM_GPU_BATCH_SIZE
        GRADIENT_ACCUMULATION_STEPS = LOW_VRAM_GPU_GRAD_ACCUM
        SAVE_STEPS = LOW_VRAM_GPU_SAVE_STEPS
        GRADIENT_CHECKPOINTING = True
    else:
        MAX_SEQ_LENGTH = args.max_seq_length or DEFAULT_MAX_SEQ_LENGTH
        NUM_TRAIN_EPOCHS = args.epochs or DEFAULT_TRAIN_EPOCHS
        PER_DEVICE_TRAIN_BATCH_SIZE = DEFAULT_BATCH_SIZE
        GRADIENT_ACCUMULATION_STEPS = DEFAULT_GRAD_ACCUM
        SAVE_STEPS = DEFAULT_SAVE_STEPS
        GRADIENT_CHECKPOINTING = False

    print(
        "[*] Runtime profile: "
        f"{runtime_profile_name(running_on_cpu, low_vram_gpu)} | "
        f"device={'cuda' if torch.cuda.is_available() else 'cpu'} | "
        f"model={args.model}"
    )
    print(
        f"[*] Training config: seq={MAX_SEQ_LENGTH} | epochs={NUM_TRAIN_EPOCHS} | "
        f"batch={PER_DEVICE_TRAIN_BATCH_SIZE} | accum={GRADIENT_ACCUMULATION_STEPS} | "
        f"grad_ckpt={GRADIENT_CHECKPOINTING}"
    )
    if CPU_BASIC_SAFE_MODE:
        print("[*] CPU-safe note: truncated contexts, 1 epoch, and rapid checkpoints are enabled.")
    elif low_vram_gpu:
        print("[*] Low-VRAM GPU note: using seq=512, batch=1, grad checkpointing, and slower accumulation for T4-class cards.")
    sys.stdout.flush()
    return args

SYSTEM_PROMPT = """You are ARGUS, an AI counter-intelligence agent defending a corporate network from HYDRA infiltrators.

Read the observation carefully. Output a single JSON action. Available actions:
- {"action_type":"work","target":"<dept>","reason":"..."} - Generate revenue
- {"action_type":"canary","target":"<dept>","reason":"..."} - Plant tracking data
- {"action_type":"monitor","target":"<channel>","reason":"..."} - Scan for leaks
- {"action_type":"investigate","target":"<id>","sub_action":"audit|verify|correlate","reason":"..."}
- {"action_type":"neutralize","target":"<worker_id>","sub_action":"terminate|interrogate|turn","reason":"..."}
- {"action_type":"deploy_double","target":"<worker_id>","reason":"..."} - Feed disinformation through an active double agent
- {"action_type":"noop","reason":"..."}

Departments: engineering, finance, rd, operations, executive, legal
Channels: market_chatter, dark_web, competitor_filing, press_leak, insider_trade

PRIORITIES: Confirmed threats outrank all revenue and double-agent actions. Keep security at or above 90. Catch every sleeper with zero false accusations. Verify leaks before accusing. Plant canaries early. Interrogate uncertain suspects. Turn at most one confirmed sleeper in advanced levels, then terminate the remaining confirmed threats. Deploy a double agent only when no confirmed/high-suspicion threat is waiting and security is at least 95.
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
        return {
            "completed_levels": [],
            "current_model": DEFAULT_MODEL,
            "trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION,
            "runtime_profile": "",
            "seed": None,
        }

    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            state = json.load(f)
    except Exception:
        return {
            "completed_levels": [],
            "current_model": DEFAULT_MODEL,
            "trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION,
            "runtime_profile": "",
            "seed": None,
        }

    state.setdefault("completed_levels", [])
    state.setdefault("current_model", DEFAULT_MODEL)
    state.setdefault("trajectory_schema_version", "")
    state.setdefault("runtime_profile", "")
    state.setdefault("seed", None)
    state["completed_levels"] = ordered_completed_levels(state["completed_levels"])
    return state


def save_state(state):
    tmp_path = STATE_PATH.with_suffix(".tmp")
    payload = {
        "completed_levels": ordered_completed_levels(state.get("completed_levels", [])),
        "current_model": state.get("current_model", DEFAULT_MODEL),
        "trajectory_schema_version": state.get("trajectory_schema_version", TRAJECTORY_SCHEMA_VERSION),
        "runtime_profile": state.get("runtime_profile", ""),
        "seed": state.get("seed"),
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


def save_data_meta(
    meta_path: Path,
    task_level: str,
    num_episodes: int,
    num_examples: int,
    model_name: str,
    seed: int,
):
    tmp_path = meta_path.with_suffix(".tmp")
    payload = {
        "task_level": task_level,
        "num_episodes": num_episodes,
        "num_examples": num_examples,
        "max_seq_length": MAX_SEQ_LENGTH,
        "trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION,
        "model_name": model_name,
        "seed": seed,
        "runtime_profile": current_runtime_profile_name(),
    }
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, meta_path)


def data_matches(meta, task_level: str, num_episodes: int, model_name: str, seed: int):
    return (
        meta is not None
        and meta.get("task_level") == task_level
        and meta.get("num_episodes") == num_episodes
        and meta.get("max_seq_length") == MAX_SEQ_LENGTH
        and meta.get("trajectory_schema_version") == TRAJECTORY_SCHEMA_VERSION
        and meta.get("model_name") == model_name
        and meta.get("seed") == seed
        and meta.get("runtime_profile") == current_runtime_profile_name()
    )


def active_departments_from_observation(obs: EnvironmentObservation) -> list[str]:
    departments: list[str] = []
    for worker in obs.workers:
        if worker.department not in departments:
            departments.append(worker.department)
    for trap in obs.canary_traps:
        if trap.department not in departments:
            departments.append(trap.department)
    if departments:
        return departments
    return [dept.value for dept in Department]


def format_observation(obs: EnvironmentObservation) -> str:
    workers_info = []
    for w in obs.workers:
        status = f"suspicion={w.suspicion_level:.0%}" if w.suspicion_level > 0.05 else "clean"
        if w.turning_in_progress:
            status += f" turning={w.interrogation_progress}/4"
        workers_info.append(f"  {w.id} {w.name} dept={w.department} state={w.state} {status}")

    leaks_info = []
    for l in obs.active_leaks:
        canary = " [CANARY MATCH]" if l.is_canary else ""
        leaks_info.append(f"  {l.id} dept={l.department} channel={l.channel}{canary}")

    traps_info = [f"  {c.id} dept={c.department} triggered={c.triggered}" for c in obs.canary_traps]
    intel_info = []
    for report in obs.intel_reports[-4:]:
        flagged = ",".join(report.flagged_workers) if report.flagged_workers else "none"
        findings = report.findings.replace("\n", " ").strip() or "no findings"
        intel_info.append(
            f"  {report.id} {report.report_type} target={report.target} "
            f"conf={report.confidence:.0%} flagged={flagged} :: {findings}"
        )
    da_info = []
    for asset in obs.double_agents:
        da_info.append(
            f"  {asset.worker_id} active={asset.active} trust={asset.hydra_trust:.0%} "
            f"eff={asset.effectiveness:.0%} disinfo={asset.disinfo_fed_count}"
        )
    active_departments = active_departments_from_observation(obs)

    sections = [
        f"Turn {obs.turn}/{obs.max_turns} | Phase: {obs.phase} ({obs.phase_number}) | Revenue: {obs.enterprise_revenue:.0f} | Security: {obs.security_score:.0f}",
        f"Allowed Departments: {', '.join(active_departments)}",
        f"Workers ({len(obs.workers)}):",
        "\n".join(workers_info) if workers_info else "  (none)",
        f"Active Leaks ({len(obs.active_leaks)}):",
        "\n".join(leaks_info) if leaks_info else "  (none)",
        f"Canary Traps ({len(obs.canary_traps)}):",
        "\n".join(traps_info) if traps_info else "  (none)",
        f"Recent Intel Reports ({len(obs.intel_reports)}):",
        "\n".join(intel_info) if intel_info else "  (none)",
        f"Active Double Agents ({len(obs.double_agents)}):",
        "\n".join(da_info) if da_info else "  (none)",
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


def _choose_curriculum_expert_action(
    obs: EnvironmentObservation,
    task_level: str,
    expert_state: dict,
) -> AgentAction:
    return choose_security_first_action(obs, task_level, expert_state)


def generate_expert_trajectories(task_level: str, num_episodes: int = 20, seed: int = 42):
    trajectories = []
    episode_metrics = []
    seed_rng = random.Random(f"{seed}:{task_level}")

    for ep in range(num_episodes):
        episode_seed = seed_rng.randint(0, 999999)
        env = Environment(seed=episode_seed)
        obs = env.reset(task_level=task_level, seed=episode_seed)
        expert_state = new_security_expert_state()
        steps = 0
        rewards = []

        while not env.state.done and steps < 300:
            action = _choose_curriculum_expert_action(obs, task_level, expert_state)

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
            rewards.append(result.reward)
            steps += 1
            post_state = env.state
            log_event(
                "expert_step",
                echo=False,
                level=task_level,
                episode=ep + 1,
                episodes=num_episodes,
                turn=post_state.turn,
                max_turns=post_state.max_turns,
                phase=post_state.phase,
                phase_number=post_state.phase_number,
                action_type=action.action_type,
                target=action.target,
                sub_action=action.sub_action or "none",
                valid_action=bool(result.info.get("valid", True)),
                reward=round(result.reward, 6),
                total_reward=round(post_state.total_reward, 6),
                revenue=round(post_state.enterprise_revenue, 3),
                security=round(post_state.security_score, 3),
                sleepers_caught=post_state.sleepers_caught,
                sleepers_spawned=post_state.total_sleepers_spawned,
                sleepers_missed=post_state.sleepers_missed,
                false_accusations=post_state.false_accusations,
                invalid_actions=post_state.invalid_actions,
                active_leaks=len(obs.active_leaks),
                canary_traps=len(post_state.canary_traps),
                triggered_canaries=sum(1 for trap in post_state.canary_traps if trap.triggered),
                double_agents=len(post_state.double_agents),
                active_double_agents=sum(1 for asset in post_state.double_agents if asset.active),
                event_count=len(result.info.get("events", [])),
            )

        s = env.state
        grade_data = {
            "total_reward": sum(rewards),
            "rewards": rewards,
            "success": s.security_score > 20 and s.enterprise_revenue > 20,
            "steps": steps,
            "state": s.model_dump(),
            "cascade_failures": 0,
            "invalid_actions": s.invalid_actions,
        }
        grade = grade_episode(task_level, grade_data)
        episode_metrics.append(
            {
                "task_level": task_level,
                "episode": ep + 1,
                "seed": episode_seed,
                "steps": steps,
                "total_reward": sum(rewards),
                "revenue": s.enterprise_revenue,
                "security": s.security_score,
                "sleepers_caught": s.sleepers_caught,
                "sleepers_spawned": s.total_sleepers_spawned,
                "sleepers_missed": s.sleepers_missed,
                "false_accusations": s.false_accusations,
                "invalid_actions": s.invalid_actions,
                "double_agents": len(s.double_agents),
                "active_double_agents": sum(1 for asset in s.double_agents if asset.active),
                "canary_traps": len(s.canary_traps),
                "triggered_canaries": sum(1 for trap in s.canary_traps if trap.triggered),
                "active_leaks": len(obs.active_leaks),
                "grade": grade.score,
                "passed": grade.passed,
            }
        )
        if not expert_episode_meets_security_gate(episode_metrics[-1]):
            raise RuntimeError(
                "Security-first expert gate failed before training: "
                f"level={task_level} seed={episode_seed} security={s.security_score:.1f} "
                f"caught={s.sleepers_caught}/{s.total_sleepers_spawned} "
                f"missed={s.sleepers_missed} false={s.false_accusations}"
            )
        progress_pct = 100.0 * (ep + 1) / max(1, num_episodes)
        print(
            f"  [Expert Ep {ep + 1}/{num_episodes}] Steps={steps} Reward={sum(rewards):.2f} "
            f"Rev={s.enterprise_revenue:.0f} Sec={s.security_score:.0f} "
            f"Caught={s.sleepers_caught} DAs={len(s.double_agents)} Grade={grade.score:.3f}"
        )
        print(
            f"  [DATA_PROGRESS] level={task_level} episode={ep + 1}/{num_episodes} "
            f"pct={progress_pct:.1f}% examples_so_far={len(trajectories)}"
        )
        log_event(
            "expert_episode",
            level=task_level,
            episode=ep + 1,
            episodes=num_episodes,
            pct=round(progress_pct, 3),
            seed=episode_seed,
            steps=steps,
            reward=round(sum(rewards), 6),
            revenue=s.enterprise_revenue,
            security=s.security_score,
            sleepers_caught=s.sleepers_caught,
            sleepers_spawned=s.total_sleepers_spawned,
            sleepers_missed=s.sleepers_missed,
            false_accusations=s.false_accusations,
            invalid_actions=s.invalid_actions,
            double_agents=len(s.double_agents),
            grade=grade.score,
            passed=grade.passed,
            trajectory_examples_so_far=len(trajectories),
        )
        sys.stdout.flush()

    return trajectories, episode_metrics


def save_episode_metrics(metrics, output_path: str):
    tmp_path = f"{output_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp_path, output_path)
    print(f"  Saved {len(metrics)} expert episode metrics to {output_path}")
    sys.stdout.flush()
    return output_path


def trajectory_training_weight(trajectory: dict) -> int:
    try:
        action = json.loads(trajectory["action"])
    except (KeyError, json.JSONDecodeError, TypeError):
        return 1

    action_type = action.get("action_type", "")
    sub_action = action.get("sub_action", "")
    if action_type == ActionType.DEPLOY_DOUBLE.value:
        return 2
    if action_type == ActionType.NEUTRALIZE.value:
        if sub_action == SubAction.TERMINATE.value:
            return 8
        if sub_action in {SubAction.INTERROGATE.value, SubAction.TURN.value}:
            return 6
    if action_type == ActionType.INVESTIGATE.value:
        return 4 if sub_action in {SubAction.VERIFY.value, SubAction.CORRELATE.value} else 3
    if action_type == ActionType.CANARY.value:
        return 2
    return 1


def render_training_text(tokenizer, observation: str, action: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Current State:\n{observation}\n\nYour action (JSON):",
        },
        {"role": "assistant", "content": action},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False)
    except Exception:
        return (
            f"{SYSTEM_PROMPT}\n\nCurrent State:\n{observation}\n\n"
            f"Your action (JSON):\n{action}"
        )


def compact_observation_lines(observation: str) -> str:
    lines = observation.splitlines()
    kept: list[str] = []
    clean_workers: list[str] = []
    canary_lines: list[str] = []
    report_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if line.startswith("  ") and " state=loyal clean" in line:
            clean_workers.append(line)
            continue
        if line.startswith("  canary-"):
            canary_lines.append(line)
            continue
        if line.startswith("  report-"):
            report_lines.append(line)
            continue
        kept.append(line)

    if clean_workers:
        kept.append(f"  clean loyal workers omitted: {len(clean_workers)}")
    if canary_lines:
        kept.append("Recent Canary Traps:")
        kept.extend(canary_lines[-6:])
    if report_lines:
        kept.append("Recent Intel Reports:")
        kept.extend(report_lines[-4:])

    return "\n".join(kept)


def truncate_observation_tokens(tokenizer, observation: str, token_budget: int) -> str:
    token_budget = max(32, token_budget)
    token_ids = tokenizer.encode(observation, add_special_tokens=False)
    if len(token_ids) <= token_budget:
        return observation

    separator = "\n[... compacted for training context ...]\n"
    separator_tokens = tokenizer.encode(separator, add_special_tokens=False)
    available = max(16, token_budget - len(separator_tokens))
    head_budget = max(8, int(available * 0.62))
    tail_budget = max(8, available - head_budget)
    head = tokenizer.decode(token_ids[:head_budget], skip_special_tokens=False)
    tail = tokenizer.decode(token_ids[-tail_budget:], skip_special_tokens=False)
    return head.rstrip() + separator + tail.lstrip()


def fit_training_text(tokenizer, observation: str, action: str) -> tuple[str, int, bool]:
    text = render_training_text(tokenizer, observation, action)
    length = len(tokenizer(text, truncation=False)["input_ids"])
    if length <= MAX_SEQ_LENGTH:
        return text, length, False

    compact = compact_observation_lines(observation)
    text = render_training_text(tokenizer, compact, action)
    length = len(tokenizer(text, truncation=False)["input_ids"])
    if length <= MAX_SEQ_LENGTH:
        return text, length, True

    obs_ids = tokenizer.encode(compact, add_special_tokens=False)
    low = 32
    high = max(low, len(obs_ids))
    best_text = text
    best_length = length

    while low <= high:
        mid = (low + high) // 2
        candidate_obs = truncate_observation_tokens(tokenizer, compact, mid)
        candidate_text = render_training_text(tokenizer, candidate_obs, action)
        candidate_length = len(tokenizer(candidate_text, truncation=False)["input_ids"])
        if candidate_length <= MAX_SEQ_LENGTH:
            best_text = candidate_text
            best_length = candidate_length
            low = mid + 1
        else:
            high = mid - 1

    if best_length > MAX_SEQ_LENGTH:
        tiny_obs = "\n".join(compact.splitlines()[:4]) or "State compacted for training."
        best_text = render_training_text(tokenizer, tiny_obs, action)
        best_length = len(tokenizer(best_text, truncation=False)["input_ids"])

    return best_text, best_length, True


def save_training_data_with_template(trajectories, output_path: str, tokenizer, task_level: str):
    tmp_path = f"{output_path}.tmp"
    written = 0
    action_counts: dict[str, int] = {}
    weighted_action_counts: dict[str, int] = {}
    token_lengths: list[int] = []
    compacted_examples = 0
    with open(tmp_path, "w", encoding="utf-8") as f:
        for t in trajectories:
            try:
                action_payload = json.loads(t["action"])
                action_key = action_payload.get("action_type", "unknown")
                sub_action = action_payload.get("sub_action", "none")
                if sub_action and sub_action != "none":
                    action_key = f"{action_key}/{sub_action}"
            except (KeyError, TypeError, json.JSONDecodeError):
                action_key = "unknown"
            weight = trajectory_training_weight(t)
            action_counts[action_key] = action_counts.get(action_key, 0) + 1
            weighted_action_counts[action_key] = weighted_action_counts.get(action_key, 0) + weight

            text, token_length, was_compacted = fit_training_text(tokenizer, t["observation"], t["action"])
            token_lengths.append(token_length)
            if was_compacted:
                compacted_examples += 1
            for _ in range(weight):
                f.write(json.dumps({"text": text}) + "\n")
                written += 1

    os.replace(tmp_path, output_path)
    print(
        f"  Saved {written} weighted examples to {output_path} "
        f"from {len(trajectories)} trajectory steps"
    )
    print(f"  [DATA_ACTIONS] raw={json.dumps(action_counts, sort_keys=True)}")
    print(f"  [DATA_ACTIONS_WEIGHTED] weighted={json.dumps(weighted_action_counts, sort_keys=True)}")
    if token_lengths:
        print(
            "  [DATA_COMPACT] "
            f"compacted={compacted_examples}/{len(token_lengths)} "
            f"token_min={min(token_lengths)} token_max={max(token_lengths)} "
            f"token_avg={sum(token_lengths) / len(token_lengths):.0f}"
        )
    log_event(
        "dataset_written",
        level=task_level,
        output_path=output_path,
        raw_examples=len(trajectories),
        weighted_examples=written,
        compacted_examples=compacted_examples,
        prompt_token_min=min(token_lengths) if token_lengths else None,
        prompt_token_max=max(token_lengths) if token_lengths else None,
        prompt_token_avg=round(sum(token_lengths) / len(token_lengths), 3) if token_lengths else None,
        action_counts=action_counts,
        weighted_action_counts=weighted_action_counts,
    )
    sys.stdout.flush()
    return written


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
            low_cpu_mem_usage=True,
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
            low_cpu_mem_usage=True,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.use_cache = False
    return model, tokenizer


def build_completion_data_collator(tokenizer):
    if DataCollatorForCompletionOnlyLM is None:
        print("  [WARN] TRL DataCollatorForCompletionOnlyLM unavailable; training full text.")
        sys.stdout.flush()
        return None

    response_template = "<|im_start|>assistant\n"
    try:
        marker = "__ARGUS_RESPONSE_MARKER__"
        rendered = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "user"},
                {"role": "assistant", "content": marker},
            ],
            tokenize=False,
        )
        marker_idx = rendered.find(marker)
        if marker_idx >= 0:
            prefix = rendered[:marker_idx]
            for candidate in (
                "<|im_start|>assistant",
                "<|start_header_id|>assistant<|end_header_id|>",
                "assistant",
            ):
                candidate_idx = prefix.rfind(candidate)
                if candidate_idx >= 0:
                    response_template = prefix[candidate_idx:]
                    break
    except Exception:
        pass

    print(f"  [DATA] Assistant-only response template: {response_template!r}")
    sys.stdout.flush()
    return DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )


def train_on_level(
    model_name: str,
    task_level: str,
    num_episodes: int = EPISODES_PER_LEVEL,
    level_index: int = 1,
    total_levels: int = 1,
    seed: int = 42,
):
    level_started_at = time.time()
    print("\n" + "=" * 60)
    print(f"  TRAINING: {task_level} | Episodes: {num_episodes} | Base: {model_name}")
    print("=" * 60)
    sys.stdout.flush()
    log_event(
        "level_start",
        level=task_level,
        level_index=level_index,
        total_levels=total_levels,
        episodes=num_episodes,
        base_model=model_name,
        run_root=str(RUN_ROOT),
        max_seq_length=MAX_SEQ_LENGTH,
        epochs=NUM_TRAIN_EPOCHS,
        batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        runtime_profile=current_runtime_profile_name(),
    )

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
    metrics_path = RUN_ROOT / f"expert_metrics_{task_level}.json"

    last_checkpoint = get_last_checkpoint(str(output_dir)) if output_dir.is_dir() else None
    data_meta = load_data_meta(data_meta_path)
    has_matching_data = data_path.exists() and data_matches(
        data_meta,
        task_level,
        num_episodes,
        model_name,
        seed,
    )

    if last_checkpoint and not has_matching_data:
        print(
            f"\n[Phase 0] Found stale checkpoint/data for {task_level}. "
            "Resetting this level so fresh expert trajectories can be generated."
        )
        sys.stdout.flush()
        shutil.rmtree(output_dir, ignore_errors=True)
        last_checkpoint = None

    if has_matching_data:
        print(f"\n[Phase 1] Reusing existing training data: {data_path}")
        log_event(
            "dataset_reused",
            level=task_level,
            data_path=str(data_path),
            meta=data_meta,
        )
        sys.stdout.flush()
    else:
        print("\n[Phase 1] Generating expert trajectories...")
        sys.stdout.flush()
        trajectories, episode_metrics = generate_expert_trajectories(task_level, num_episodes, seed)
        written_examples = save_training_data_with_template(trajectories, str(data_path), tokenizer, task_level)
        save_episode_metrics(episode_metrics, str(metrics_path))
        save_data_meta(data_meta_path, task_level, num_episodes, written_examples, model_name, seed)
        log_event(
            "expert_generation_complete",
            level=task_level,
            episodes=num_episodes,
            raw_examples=len(trajectories),
            weighted_examples=written_examples,
            metrics_path=str(metrics_path),
            data_path=str(data_path),
        )

    print("\n[Phase 2] Loading model...")
    sys.stdout.flush()
    log_event("model_load_start", level=task_level, model=model_name)
    model, tokenizer = load_model_and_tokenizer(model_name)
    tokenizer.model_max_length = MAX_SEQ_LENGTH
    log_event("model_load_complete", level=task_level, model=model_name, train_dtype=str(TRAIN_DTYPE))

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

    sft_config_kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": NUM_TRAIN_EPOCHS,
        "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": 2e-5,
        "warmup_ratio": 0.03,
        "logging_steps": 5,
        "save_strategy": "steps",
        "save_steps": SAVE_STEPS,
        "save_total_limit": 2,
        "bf16": USE_BF16,
        "fp16": USE_FP16,
        "gradient_checkpointing": GRADIENT_CHECKPOINTING,
        "report_to": "none",
        "dataset_text_field": "text",
        "logging_nan_inf_filter": False,
    }

    sft_config_params = inspect.signature(SFTConfig).parameters
    trainer_accepts_max_seq_length = "max_seq_length" in inspect.signature(SFTTrainer.__init__).parameters

    if "max_seq_length" in sft_config_params:
        sft_config_kwargs["max_seq_length"] = MAX_SEQ_LENGTH

    sft_config = SFTConfig(**sft_config_kwargs)

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
    log_event(
        "token_stats",
        level=task_level,
        sample_count=len(sample_lengths),
        token_min=min(sample_lengths),
        token_max=max(sample_lengths),
        token_avg=round(sum(sample_lengths) / len(sample_lengths), 3),
        token_samples=sample_lengths,
    )
    sys.stdout.flush()

    if torch.cuda.is_available():
        gpu_total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_allocated_gb = torch.cuda.memory_allocated() / 1024**3
        gpu_reserved_gb = torch.cuda.memory_reserved() / 1024**3
        print(f"\n  [GPU] {torch.cuda.get_device_name(0)}")
        print(f"  [GPU] dtype: {TRAIN_DTYPE}")
        print(
            f"  [GPU] VRAM Total:     "
            f"{gpu_total_gb:.1f} GB"
        )
        print(f"  [GPU] VRAM Allocated: {gpu_allocated_gb:.2f} GB")
        print(f"  [GPU] VRAM Reserved:  {gpu_reserved_gb:.2f} GB")
        log_event(
            "gpu_snapshot",
            level=task_level,
            gpu=torch.cuda.get_device_name(0),
            train_dtype=str(TRAIN_DTYPE),
            vram_total_gb=round(gpu_total_gb, 3),
            vram_allocated_gb=round(gpu_allocated_gb, 3),
            vram_reserved_gb=round(gpu_reserved_gb, 3),
        )
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
    log_event(
        "train_start",
        level=task_level,
        examples=len(dataset),
        epochs=float(sft_config.num_train_epochs),
        learning_rate=sft_config.learning_rate,
        max_seq_length=MAX_SEQ_LENGTH,
        output_dir=str(output_dir),
    )
    sys.stdout.flush()

    trainer_kwargs = {
        "model": model,
        "args": sft_config,
        "train_dataset": dataset,
        "peft_config": lora_config,
    }

    data_collator = build_completion_data_collator(tokenizer)
    if data_collator is not None:
        trainer_kwargs["data_collator"] = data_collator

    trainer_params = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer

    if "dataset_text_field" in trainer_params and "dataset_text_field" not in sft_config_params:
        trainer_kwargs["dataset_text_field"] = "text"

    if trainer_accepts_max_seq_length and "max_seq_length" not in sft_config_params:
        trainer_kwargs["max_seq_length"] = MAX_SEQ_LENGTH

    trainer = SFTTrainer(**trainer_kwargs)
    trainer.add_callback(CurriculumProgressCallback(task_level, level_index, total_levels))

    if hasattr(trainer.model, "enable_input_require_grads"):
        trainer.model.enable_input_require_grads()
        print("  [FIX] Enabled input_require_grads for PEFT")
        sys.stdout.flush()

    first_batch = next(iter(trainer.get_train_dataloader()))
    valid_labels = int((first_batch["labels"] != -100).sum().item())
    print(f"  [SANITY] First batch input shape: {tuple(first_batch['input_ids'].shape)}")
    print(f"  [SANITY] First batch valid label tokens: {valid_labels}")
    log_event(
        "train_sanity",
        level=task_level,
        first_batch_shape=list(first_batch["input_ids"].shape),
        valid_label_tokens=valid_labels,
        max_steps=int(trainer.state.max_steps or 0),
    )
    sys.stdout.flush()

    if valid_labels == 0:
        raise RuntimeError("First batch has zero valid labels. Training would be invalid.")

    if last_checkpoint:
        print(f"  [RESUME] Found checkpoint: {last_checkpoint}")
        sys.stdout.flush()

    train_output = trainer.train(resume_from_checkpoint=last_checkpoint)
    log_event(
        "train_complete",
        level=task_level,
        metrics=getattr(train_output, "metrics", {}),
        runtime_sec=round(time.time() - level_started_at, 3),
    )
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print(f"  [DONE] Model saved to {output_dir}/")
    log_event(
        "level_complete",
        level=task_level,
        output_dir=str(output_dir),
        runtime_sec=round(time.time() - level_started_at, 3),
    )
    sys.stdout.flush()

    del model, trainer
    cleanup_cuda()
    return str(output_dir)


def merge_and_save_final_model(adapter_path: str, output_path: str):
    merge_started_at = time.time()
    print("\n[Phase 4] Merging adapter into standalone model...")
    sys.stdout.flush()
    log_event("merge_start", adapter_path=adapter_path, output_path=output_path)

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
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()
    model.save_pretrained(str(output_dir))

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    tokenizer.save_pretrained(str(output_dir))

    print(f"  [DONE] Merged model saved to {output_dir}/")
    log_event(
        "merge_complete",
        adapter_path=adapter_path,
        output_path=str(output_dir),
        runtime_sec=round(time.time() - merge_started_at, 3),
    )
    sys.stdout.flush()

    del model, base_model
    cleanup_cuda()
    return str(output_dir)


def evaluate_model(model_path: str, task_level: str = "level_5", num_games: int = 5):
    eval_started_at = time.time()
    print("\n" + "=" * 60)
    print(f"  EVALUATING: {model_path} on {task_level} ({num_games} games)")
    print("=" * 60)
    sys.stdout.flush()
    log_event("inline_eval_start", model_path=model_path, level=task_level, games=num_games)

    from inference_local import LocalModelPolicy, run_episode

    policy = LocalModelPolicy(model_path, deterministic=True)
    results = []
    try:
        for g in range(num_games):
            seed = random.randint(100000, 999999)
            episode = run_episode(
                policy,
                task_level=task_level,
                seed=seed,
                max_steps=300,
                verbose=False,
            )
            results.append(episode["grade"]["score"])
            final_state = episode["final_state"]
            invalid_actions = final_state.get("invalid_actions", 0)

            print(
                f"  [Game {g + 1}] Steps={episode['steps']} "
                f"Rev={final_state['enterprise_revenue']:.0f} "
                f"Sec={final_state['security_score']:.0f} "
                f"Caught={final_state['sleepers_caught']} "
                f"Invalid={invalid_actions} Grade={episode['grade']['score']:.3f}"
            )
            log_event(
                "inline_eval_game",
                model_path=model_path,
                level=task_level,
                game=g + 1,
                games=num_games,
                seed=seed,
                steps=episode["steps"],
                reward=episode["total_reward"],
                grade=episode["grade"]["score"],
                passed=episode["grade"]["passed"],
                revenue=final_state["enterprise_revenue"],
                security=final_state["security_score"],
                sleepers_caught=final_state["sleepers_caught"],
                sleepers_spawned=final_state["total_sleepers_spawned"],
                invalid_actions=invalid_actions,
            )
            sys.stdout.flush()
    finally:
        policy.close()

    avg = sum(results) / len(results)
    print(f"\n  Average Grade: {avg:.3f}")
    log_event(
        "inline_eval_complete",
        model_path=model_path,
        level=task_level,
        games=num_games,
        average_grade=avg,
        runtime_sec=round(time.time() - eval_started_at, 3),
    )
    sys.stdout.flush()
    return avg


def main():
    run_started_at = time.time()
    parser = argparse.ArgumentParser(description="Train LLM on Panopticon v3")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--level", default="easy")
    parser.add_argument("--curriculum", action="store_true", help="Chain training across all 5 levels")
    parser.add_argument("--episodes", type=int, default=EPISODES_PER_LEVEL)
    parser.add_argument("--seed", type=int, default=42, help="Deterministic expert-data seed")
    parser.add_argument("--epochs", type=int, help="Override SFT epochs for this run")
    parser.add_argument("--max-seq-length", type=int, help="Override tokenizer/training sequence length")
    parser.add_argument(
        "--cpu-basic-safe",
        action="store_true",
        help="Use a lighter runtime profile suitable for CPU basic / low-memory environments",
    )
    parser.add_argument("--eval", action="store_true", help="Evaluate after training")
    parser.add_argument("--eval-games", type=int, default=5)
    parser.add_argument("--merge", action="store_true", help="Merge final adapter into standalone model")
    args = parser.parse_args()
    args = configure_runtime(args)
    runtime_profile = current_runtime_profile_name()
    requested_levels = LEVELS if args.curriculum else [args.level]
    print(f"[*] Structured event log: {EVENTS_PATH}")
    sys.stdout.flush()
    log_event(
        "run_start",
        curriculum=bool(args.curriculum),
        requested_levels=requested_levels,
        episodes=args.episodes,
        model=args.model,
        run_root=str(RUN_ROOT),
        events_path=str(EVENTS_PATH),
        runtime_profile=runtime_profile,
        max_seq_length=MAX_SEQ_LENGTH,
        epochs=NUM_TRAIN_EPOCHS,
        batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        seed=args.seed,
        merge=bool(args.merge or args.curriculum),
        eval=bool(args.eval),
    )

    if args.curriculum:
        state = load_state()
        if (
            state.get("trajectory_schema_version") != TRAJECTORY_SCHEMA_VERSION
            or state.get("runtime_profile") != runtime_profile
            or state.get("seed") != args.seed
        ):
            print(
                "[*] Curriculum state was produced by an older expert/runtime profile. "
                "Resetting completed-level tracking so fresh-compatible data is regenerated."
            )
            sys.stdout.flush()
            state = {
                "completed_levels": [],
                "current_model": args.model,
                "trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION,
                "runtime_profile": runtime_profile,
                "seed": args.seed,
            }
            save_state(state)

        completed_levels = set(state.get("completed_levels", []))
        current_model = state.get("current_model", args.model)

        if completed_levels:
            print(f"[*] Resuming curriculum. Already completed: {ordered_completed_levels(completed_levels)}")
            log_event(
                "curriculum_resume",
                completed_levels=ordered_completed_levels(completed_levels),
                current_model=current_model,
            )
            sys.stdout.flush()

        for level_index, level in enumerate(LEVELS, start=1):
            if level in completed_levels:
                print(f"[*] Skipping completed level: {level}")
                current_model = str(RUN_ROOT / f"trl_model_{level}")
                log_event(
                    "level_skipped",
                    level=level,
                    level_index=level_index,
                    total_levels=len(LEVELS),
                    curriculum_pct=round(100.0 * level_index / len(LEVELS), 3),
                    current_model=current_model,
                )
                sys.stdout.flush()
                continue

            current_model = train_on_level(
                current_model,
                level,
                args.episodes,
                level_index=level_index,
                total_levels=len(LEVELS),
                seed=args.seed,
            )

            completed_levels.add(level)
            state["completed_levels"] = ordered_completed_levels(completed_levels)
            state["current_model"] = current_model
            state["trajectory_schema_version"] = TRAJECTORY_SCHEMA_VERSION
            state["runtime_profile"] = runtime_profile
            state["seed"] = args.seed
            save_state(state)

            print(f"[*] Progress saved: {level} complete")
            log_event(
                "curriculum_progress",
                completed_levels=ordered_completed_levels(completed_levels),
                completed_count=len(completed_levels),
                total_levels=len(LEVELS),
                curriculum_pct=round(100.0 * len(completed_levels) / len(LEVELS), 3),
                current_model=current_model,
            )
            sys.stdout.flush()

        final_model = current_model
    else:
        final_model = train_on_level(
            args.model,
            args.level,
            args.episodes,
            level_index=1,
            total_levels=1,
            seed=args.seed,
        )

    if args.merge or args.curriculum:
        merged_path = str(RUN_ROOT / "merged_model")
        final_model = merge_and_save_final_model(final_model, merged_path)

    if args.eval:
        eval_level = "level_5" if args.curriculum else args.level
        evaluate_model(final_model, eval_level, args.eval_games)

    print("\n[*] All training complete!")
    print(f"[*] Persistent root: {RUN_ROOT}")
    log_event(
        "run_complete",
        final_model=final_model,
        run_root=str(RUN_ROOT),
        runtime_sec=round(time.time() - run_started_at, 3),
    )
    sys.stdout.flush()


if __name__ == "__main__":
    main()
