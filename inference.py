#!/usr/bin/env python3
"""
OpenEnv Starter Kit — LLM Inference Script
=============================================

Connects an LLM agent to the environment. Logs results in the
required OpenEnv format.

Environment Variables:
    API_BASE_URL, MODEL_NAME, HF_TOKEN, TASK_LEVEL, ENV_SEED
"""
from __future__ import annotations
import asyncio, json, os, sys
from openai import AsyncOpenAI
from client import LocalClient, StepResult
from models import ActionType, AgentAction, EnvironmentObservation

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY")
TASK_LEVEL = os.environ.get("TASK_LEVEL", "all")
MAX_STEPS = int(os.environ.get("MAX_STEPS", "100"))

# TODO: Write your system prompt describing the problem domain
SYSTEM_PROMPT = """You are an expert agent. Analyze the current state and choose the optimal action.

AVAILABLE ACTIONS (respond with JSON):
- inspect: Get detailed info about a target entity
- process: Process a target entity to complete its task
- repair: Fix a failed entity
- wait: Skip this turn
- noop: Do nothing

Response format: {"action_type": "...", "target": "...", "reason": "..."}
"""

async def get_llm_action(client: AsyncOpenAI, obs: EnvironmentObservation, history: list) -> AgentAction:
    obs_text = json.dumps(obs.model_dump(), indent=2, default=str)
    history.append({"role": "user", "content": f"Current state:\n{obs_text}\n\nChoose your action (JSON):"})

    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME, messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history[-10:],
            temperature=0.1, max_tokens=256,
        )
        text = response.choices[0].message.content.strip()
        history.append({"role": "assistant", "content": text})

        # Parse JSON from response
        if "```" in text:
            text = text.split("```")[1].strip().removeprefix("json").strip()
        data = json.loads(text)
        return AgentAction(
            action_type=ActionType(data.get("action_type", "noop")),
            target=data.get("target", ""),
            reason=data.get("reason", ""),
        )
    except Exception as e:
        return AgentAction(action_type=ActionType.NOOP, reason=f"Parse error: {e}")

async def run_episode(task_level: str):
    llm = AsyncOpenAI(api_key=HF_TOKEN or "dummy", base_url=API_BASE_URL)
    env = LocalClient()
    obs = env.reset(task_level=task_level)
    history, rewards = [], []
    print(f"[START] task={task_level} env=openenv-starter model={MODEL_NAME}")

    for step in range(MAX_STEPS):
        action = await get_llm_action(llm, obs, history)
        result = env.step(action)
        obs, rewards = result.observation, rewards + [result.reward]
        action_json = json.dumps({"action_type": action.action_type, "target": action.target})
        print(f"[STEP] step={step+1} action={action_json} reward={result.reward:.2f} done={result.done}")
        if result.done:
            break

    success = len(obs.tasks) == 0
    score = sum(rewards)
    reward_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={len(rewards)} score={score:.3f} rewards={reward_str}")

async def main():
    levels = ["easy", "medium", "hard", "level_4", "level_5"] if TASK_LEVEL == "all" else [TASK_LEVEL]
    for level in levels:
        try:
            await run_episode(level)
        except Exception as e:
            print(f"[END] success=false steps=0 score=0.000 rewards=0.00 error={e}")

if __name__ == "__main__":
    asyncio.run(main())
