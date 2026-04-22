#!/usr/bin/env python3
"""
The Panopticon Protocol v3 — LLM Inference Script
====================================================

Connects an LLM agent to the Panopticon environment.
The LLM plays as ARGUS — the corporate counter-intelligence AI.

Environment Variables:
    API_BASE_URL, MODEL_NAME, HF_TOKEN, TASK_LEVEL, ENV_SEED
"""
from __future__ import annotations
import asyncio, json, os, sys
from openai import AsyncOpenAI
from client import LocalClient, StepResult
from models import ActionType, SubAction, AgentAction, EnvironmentObservation

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY")
TASK_LEVEL = os.environ.get("TASK_LEVEL", "all")
MAX_STEPS = int(os.environ.get("MAX_STEPS", "200"))

SYSTEM_PROMPT = """You are ARGUS, an elite AI counter-intelligence agent defending a corporate network
against HYDRA — an adaptive adversary that infiltrates sleeper agents into the workforce.

YOUR MISSION: Detect and neutralize sleeper agents while maintaining enterprise revenue and security.

AVAILABLE ACTIONS (respond with JSON):

PRODUCTIVITY:
- work: Process projects in a department → {"action_type": "work", "target": "<department>"}
- hire: Replace a terminated worker → {"action_type": "hire", "target": "<department>"}

INTELLIGENCE:
- canary: Plant a canary trap (unique tracking data) → {"action_type": "canary", "target": "<department>"}
- monitor: Scan leak channels for canary matches → {"action_type": "monitor", "target": "<channel>"}
- investigate: Deep investigation with sub-actions:
  - audit: Audit a worker → {"action_type": "investigate", "target": "<worker_id>", "sub_action": "audit"}
  - verify: Cross-ref a leak with canary data → {"action_type": "investigate", "target": "<leak_id>", "sub_action": "verify"}
  - correlate: Cross-dept signal analysis → {"action_type": "investigate", "target": "<department>", "sub_action": "correlate"}

ENFORCEMENT:
- neutralize: Act on a suspect with sub-actions:
  - terminate: Immediately fire (CAUTION: dead-man's switch on Gen-4!) → {"action_type": "neutralize", "target": "<worker_id>", "sub_action": "terminate"}
  - interrogate: Gather intel first (reveals generation) → {"action_type": "neutralize", "target": "<worker_id>", "sub_action": "interrogate"}
  - turn: Convert sleeper to double agent (4 turns, HIGH RISK, HIGH REWARD) → {"action_type": "neutralize", "target": "<worker_id>", "sub_action": "turn"}
- deploy_double: Feed disinformation through a turned asset → {"action_type": "deploy_double", "target": "<worker_id>"}

META:
- noop: Skip turn → {"action_type": "noop"}

DEPARTMENTS: engineering, finance, rd, operations, executive, legal
CHANNELS: market_chatter, dark_web, competitor_filing, press_leak, insider_trade

STRATEGY GUIDE:
1. Plant canaries in departments early (cover your departments)
2. Monitor channels regularly for canary matches
3. When a canary triggers, VERIFY the leak to trace the source
4. AUDIT suspicious workers for confirmation
5. INTERROGATE before TERMINATING (especially Gen-4+ with dead switches!)
6. Consider TURNING sleepers into double agents for long-term advantage
7. Deploy double agents to FEED DISINFORMATION back to HYDRA
8. Keep WORK actions flowing to maintain revenue

CRITICAL: Gen-3+ sleepers plant FALSE FLAGS framing innocent workers!
Always VERIFY before acting. A false accusation damages your score severely.

Respond with a single JSON action object. Include a "reason" field explaining your thinking.
"""


async def get_llm_action(client: AsyncOpenAI, obs: EnvironmentObservation, history: list) -> AgentAction:
    """Query LLM for next action based on current observation."""
    obs_text = json.dumps(obs.model_dump(), indent=2, default=str)
    history.append({
        "role": "user",
        "content": f"Current state (Turn {obs.turn}/{obs.max_turns}, Phase: {obs.phase}):\n{obs_text}\n\nChoose your action (JSON):",
    })

    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history[-12:],
            temperature=0.15, max_tokens=384,
        )
        text = response.choices[0].message.content.strip()
        history.append({"role": "assistant", "content": text})

        # Parse JSON from response
        if "```" in text:
            text = text.split("```")[1].strip().removeprefix("json").strip()
        data = json.loads(text)

        return AgentAction(
            action_type=data.get("action_type", "noop"),
            target=data.get("target", ""),
            sub_action=data.get("sub_action", "none"),
            reason=data.get("reason", ""),
        )
    except Exception as e:
        return AgentAction(action_type=ActionType.NOOP.value, reason=f"Parse error: {e}")


async def run_episode(task_level: str):
    """Run a single episode with LLM agent."""
    llm = AsyncOpenAI(api_key=HF_TOKEN or "dummy", base_url=API_BASE_URL)
    env = LocalClient()
    obs = env.reset(task_level=task_level)
    history, rewards = [], []
    print(f"[START] task={task_level} env=panopticon-v3 model={MODEL_NAME}")

    for step in range(MAX_STEPS):
        action = await get_llm_action(llm, obs, history)
        result = env.step(action)
        obs, rewards = result.observation, rewards + [result.reward]

        action_json = json.dumps({
            "action_type": action.action_type,
            "target": action.target,
            "sub_action": action.sub_action,
        })
        print(
            f"[STEP] step={step+1} action={action_json} "
            f"reward={result.reward:.2f} revenue={obs.enterprise_revenue:.0f} "
            f"security={obs.security_score:.0f} done={result.done}"
        )
        if result.done:
            break

    score = sum(rewards)
    reward_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={obs.security_score > 20} steps={len(rewards)} "
        f"score={score:.3f} revenue={obs.enterprise_revenue:.0f} "
        f"security={obs.security_score:.0f} rewards={reward_str}"
    )


async def main():
    levels = ["easy", "medium", "hard", "level_4", "level_5"] if TASK_LEVEL == "all" else [TASK_LEVEL]
    for level in levels:
        try:
            await run_episode(level)
        except Exception as e:
            print(f"[END] success=false steps=0 score=0.000 rewards=0.00 error={e}")

if __name__ == "__main__":
    asyncio.run(main())
