"""Typed HYDRA policy boundary with the reproducible scripted baseline.

HYDRA's historical "adaptive memory" is stateful rule-based logic, not learned
parameters. This module makes that fact explicit and creates a stable interface
for learned adversaries without forcing PyTorch into the core environment.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


HYDRA_POLICY_SCHEMA_VERSION = "hydra-policy-v1"


@dataclass(frozen=True)
class HydraPolicyObservation:
    """Information HYDRA is allowed to use when choosing synthetic attacks."""

    turn: int
    max_turns: int
    phase_number: int
    enterprise_revenue: float
    security_score: float
    hydra_aggression: float
    departments_active: tuple[str, ...]
    canary_departments_seen: tuple[str, ...]
    agent_audit_pattern: dict[str, int] = field(default_factory=dict)
    monitored_channels: tuple[str, ...] = ()
    detection_count: int = 0
    disinfo_received: int = 0
    recruitment_accuracy: float = 1.0
    active_sleepers: int = 0
    dormant_sleepers: int = 0


@runtime_checkable
class HydraPolicy(Protocol):
    """Event-driven policy used by the environment's synthetic adversary."""

    policy_name: str

    def reset(self) -> None: ...

    def choose_spawn_department(
        self,
        observation: HydraPolicyObservation,
        generation: int,
        candidates: list[str],
        rng: random.Random,
    ) -> str: ...

    def choose_leak_channel(
        self,
        observation: HydraPolicyObservation,
        generation: int,
        candidates: list[str],
        rng: random.Random,
    ) -> str: ...

    def should_plant_false_flag(
        self,
        observation: HydraPolicyObservation,
        generation: int,
        rng: random.Random,
    ) -> bool: ...


class ScriptedHydraPolicy:
    """The original memory-aware rules, preserved as an auditable baseline."""

    policy_name = "scripted_memory_v1"

    def reset(self) -> None:
        return None

    def choose_spawn_department(
        self,
        observation: HydraPolicyObservation,
        generation: int,
        candidates: list[str],
        rng: random.Random,
    ) -> str:
        choices = list(candidates)
        if observation.canary_departments_seen and generation >= 2:
            safe = [dept for dept in choices if dept not in observation.canary_departments_seen]
            if safe:
                choices = safe
        if observation.agent_audit_pattern and generation >= 2:
            choices.sort(key=lambda dept: observation.agent_audit_pattern.get(dept, 0))
        if not choices:
            raise ValueError("HYDRA has no valid department candidate")
        return rng.choice(choices)

    def choose_leak_channel(
        self,
        observation: HydraPolicyObservation,
        generation: int,
        candidates: list[str],
        rng: random.Random,
    ) -> str:
        choices = list(candidates)
        if observation.monitored_channels and generation >= 2:
            safe = [channel for channel in choices if channel not in observation.monitored_channels]
            if safe:
                choices = safe
        if not choices:
            raise ValueError("HYDRA has no valid leak-channel candidate")
        return rng.choice(choices)

    def should_plant_false_flag(
        self,
        observation: HydraPolicyObservation,
        generation: int,
        rng: random.Random,
    ) -> bool:
        return generation >= 3 and rng.random() < 0.3
