"""
OpenEnv Starter Kit — Environment Core
========================================

The main simulation engine. Implements reset() and step().

TODO: Replace the example logic with your problem domain.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Literal

from models import (
    EntityState, EntityTier, ActionType,
    Entity, Task, Relationship,
    AgentAction, EnvironmentObservation, EnvironmentState,
    validate_action,
)


# =============================================================================
# STEP RESULT
# =============================================================================

@dataclass
class StepResult:
    """Result from environment.step()."""
    observation: EnvironmentObservation
    reward: float
    done: bool
    truncated: bool
    info: dict


# =============================================================================
# CONSTANTS — TODO: Tune for your problem
# =============================================================================

INVALID_ACTION_PENALTY = -0.5
VICTORY_BONUS = 50.0
TIME_PRESSURE_PENALTY = -0.1
MAX_TURNS = {"easy": 30, "medium": 50, "hard": 100, "level_4": 60, "level_5": 80}


# =============================================================================
# ENVIRONMENT
# =============================================================================

class Environment:
    """
    OpenEnv-compliant environment.

    Provides:
      - reset(task_level, seed) -> EnvironmentObservation
      - step(action) -> StepResult
      - get_observation() -> EnvironmentObservation
      - state property -> EnvironmentState

    TODO: Implement your domain-specific scenario generation and step logic.
    """

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)
        self._state = EnvironmentState()
        self._prev_penalty = 0.0
        self._task_level = "easy"
        self._cascade_failures = 0
        self._invalid_actions = 0

    @property
    def state(self) -> EnvironmentState:
        return self._state

    def reset(
        self,
        task_level: str = "easy",
        seed: int | None = None,
    ) -> EnvironmentObservation:
        """Reset environment to a new episode. TODO: Generate your scenarios."""
        if seed is not None:
            self._rng = random.Random(seed)

        self._task_level = task_level
        self._cascade_failures = 0
        self._invalid_actions = 0
        max_turns = MAX_TURNS.get(task_level, 50)

        # ──── TODO: Replace this with your scenario generation ────
        if task_level == "easy":
            entities = [
                Entity(id="entity-1", name="Alpha", tier=EntityTier.LOW),
                Entity(id="entity-2", name="Beta", tier=EntityTier.LOW),
                Entity(id="entity-3", name="Gamma", tier=EntityTier.LOW),
            ]
            tasks = [Task(id="task-1", description="Process entity-1", target_entities=["entity-1"])]
            relationships = []

        elif task_level == "medium":
            entities = [
                Entity(id="entity-1", name="Alpha", tier=EntityTier.HIGH),
                Entity(id="entity-2", name="Beta", tier=EntityTier.MEDIUM),
                Entity(id="entity-3", name="Gamma", tier=EntityTier.MEDIUM),
                Entity(id="entity-4", name="Delta", tier=EntityTier.LOW),
                Entity(id="entity-5", name="Epsilon", tier=EntityTier.LOW),
            ]
            tasks = [
                Task(id="task-1", description="Process entity-1", priority=9.0, target_entities=["entity-1"]),
                Task(id="task-2", description="Process entity-4", priority=5.0, target_entities=["entity-4"]),
            ]
            relationships = [
                Relationship(source="entity-2", target="entity-1", relationship_type="depends_on"),
                Relationship(source="entity-3", target="entity-1", relationship_type="depends_on"),
            ]

        elif task_level == "hard":
            entities = [Entity(id=f"entity-{i}", name=f"Node-{i}", tier=EntityTier(min(3, 1 + i // 4))) for i in range(1, 11)]
            tasks = [Task(id=f"task-{i}", description=f"Process entity-{i}", target_entities=[f"entity-{i}"]) for i in range(1, 6)]
            relationships = [Relationship(source=f"entity-{i+1}", target=f"entity-{i}") for i in range(1, 5)]

        else:
            # level_4, level_5, or any new level — extend here
            entities = [Entity(id=f"entity-{i}", name=f"Node-{i}") for i in range(1, 8)]
            tasks = [Task(id=f"task-{i}", description=f"Process entity-{i}", target_entities=[f"entity-{i}"]) for i in range(1, 4)]
            relationships = []
        # ──── END TODO ────

        self._state = EnvironmentState(
            entities=entities,
            tasks=tasks,
            relationships=relationships,
            turn=0,
            max_turns=max_turns,
            total_reward=0.0,
            done=False,
        )
        self._prev_penalty = self._compute_penalty()
        return self.get_observation()

    def step(self, action: AgentAction) -> StepResult:
        """Execute one step. TODO: Implement your domain-specific action effects."""
        s = self._state
        info: dict = {"valid": True}

        # Validate action
        obs = self.get_observation()
        valid, reason = validate_action(action, obs)
        if not valid:
            self._invalid_actions += 1
            s.turn += 1
            info["valid"] = False
            info["reason"] = reason
            done = s.turn >= s.max_turns
            s.done = done
            return StepResult(
                observation=self.get_observation(),
                reward=INVALID_ACTION_PENALTY,
                done=done,
                truncated=done and not self._all_tasks_complete(),
                info=info,
            )

        # ──── TODO: Implement your action effects ────
        if action.action_type == ActionType.INSPECT:
            info["inspected"] = action.target

        elif action.action_type == ActionType.PROCESS:
            entity = self._get_entity(action.target)
            if entity and entity.state == EntityState.ACTIVE:
                entity.state = EntityState.PROCESSING
                # Mark associated task as complete
                for task in s.tasks:
                    if action.target in task.target_entities:
                        task.completed = True

        elif action.action_type == ActionType.REPAIR:
            entity = self._get_entity(action.target)
            if entity and entity.state == EntityState.FAILED:
                entity.state = EntityState.ACTIVE

        elif action.action_type == ActionType.WAIT:
            pass  # No-op, just advances the turn

        elif action.action_type == ActionType.NOOP:
            pass
        # ──── END TODO ────

        # Advance processing entities back to active
        for entity in s.entities:
            if entity.state == EntityState.PROCESSING:
                entity.state = EntityState.ACTIVE

        # Compute reward
        current_penalty = self._compute_penalty()
        reward = self._prev_penalty - current_penalty + TIME_PRESSURE_PENALTY
        self._prev_penalty = current_penalty

        # Check victory
        if self._all_tasks_complete():
            reward += VICTORY_BONUS
            s.done = True

        s.turn += 1
        s.total_reward += reward

        # Check turn limit
        if s.turn >= s.max_turns:
            s.done = True

        return StepResult(
            observation=self.get_observation(),
            reward=reward,
            done=s.done,
            truncated=s.turn >= s.max_turns and not self._all_tasks_complete(),
            info=info,
        )

    def get_observation(self) -> EnvironmentObservation:
        """Generate agent-visible observation from internal state."""
        s = self._state
        return EnvironmentObservation(
            entities=copy.deepcopy(s.entities),
            tasks=[t for t in s.tasks if not t.completed],
            relationships=copy.deepcopy(s.relationships),
            turn=s.turn,
            max_turns=s.max_turns,
            messages=[],
        )

    def render(self) -> str:
        """Human-readable state summary."""
        s = self._state
        lines = [f"Turn {s.turn}/{s.max_turns} | Reward: {s.total_reward:.2f}"]
        for e in s.entities:
            lines.append(f"  {e.name}: {e.state}")
        remaining = sum(1 for t in s.tasks if not t.completed)
        lines.append(f"  Tasks remaining: {remaining}/{len(s.tasks)}")
        return "\n".join(lines)

    # ── Private helpers ──

    def _get_entity(self, entity_id: str) -> Entity | None:
        return next((e for e in self._state.entities if e.id == entity_id), None)

    def _all_tasks_complete(self) -> bool:
        return all(t.completed for t in self._state.tasks)

    def _compute_penalty(self) -> float:
        """Compute current state penalty. TODO: Customize for your scoring."""
        penalty = 0.0
        for task in self._state.tasks:
            if not task.completed:
                penalty += task.priority
        for entity in self._state.entities:
            if entity.state == EntityState.FAILED:
                penalty += 5.0 * (4 - entity.tier)
        return penalty
