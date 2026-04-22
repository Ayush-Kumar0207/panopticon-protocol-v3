"""
The Panopticon Protocol v3 — HTTP Client
==========================================
Connects to the environment server for inference and testing.
"""
from __future__ import annotations
import asyncio
from dataclasses import dataclass
from typing import Any
import httpx
from models import AgentAction, EnvironmentObservation, ActionType, SubAction


@dataclass
class StepResult:
    observation: EnvironmentObservation
    reward: float
    done: bool
    truncated: bool
    info: dict


class EnvClient:
    """Async HTTP client for the Panopticon Protocol server."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0):
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=timeout)

    async def reset(self, task_level: str = "easy", seed: int | None = None) -> EnvironmentObservation:
        payload = {"task_level": task_level}
        if seed is not None:
            payload["seed"] = seed
        resp = await self._client.post("/reset", json=payload)
        resp.raise_for_status()
        return EnvironmentObservation(**resp.json()["observation"])

    async def step(self, action: AgentAction) -> StepResult:
        payload = {
            "action_type": action.action_type,
            "target": action.target,
            "sub_action": action.sub_action,
            "reason": action.reason,
        }
        resp = await self._client.post("/step", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return StepResult(
            observation=EnvironmentObservation(**data["observation"]),
            reward=data["reward"], done=data["done"],
            truncated=data["truncated"], info=data["info"],
        )

    async def close(self):
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()


class LocalClient:
    """Direct (non-HTTP) client for local testing."""

    def __init__(self, seed: int | None = None):
        from environment import Environment
        self._env = Environment(seed=seed)

    def reset(self, task_level: str = "easy", seed: int | None = None) -> EnvironmentObservation:
        return self._env.reset(task_level=task_level, seed=seed)

    def step(self, action: AgentAction) -> StepResult:
        result = self._env.step(action)
        return StepResult(
            observation=result.observation, reward=result.reward,
            done=result.done, truncated=result.truncated, info=result.info,
        )
