"""
Stub OpenEnv task for E2E validation.
A simple toggle environment: PROCESS active → SUCCESS.
"""

from __future__ import annotations
import random, copy
from dataclasses import dataclass
from models import (
    EntityState, EntityTier, ActionType,
    Entity, Task, Relationship,
    AgentAction, EnvironmentObservation, EnvironmentState,
    validate_action
)

@dataclass
class StepResult:
    observation: EnvironmentObservation
    reward: float
    done: bool
    truncated: bool
    info: dict

class Environment:
    """Minimal OpenEnv task for verification."""
    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)
        self._state = EnvironmentState()

    @property
    def state(self) -> EnvironmentState: return self._state

    def reset(self, task_level: str = "easy", seed: int | None = None) -> EnvironmentObservation:
        if seed is not None: self._rng = random.Random(seed)
        
        # Grid/Toggle state: 1 entity, 1 task
        entities = [Entity(id="e1", name="StubNode", state=EntityState.INACTIVE, tier=EntityTier.LOW)]
        tasks = [Task(id="t1", description="Enable e1", target_entities=["e1"])]
        
        self._state = EnvironmentState(entities=entities, tasks=tasks, turn=0, max_turns=10)
        return self.get_observation()

    def step(self, action: AgentAction) -> StepResult:
        s = self._state
        info = {"valid": True}
        reward = -0.1 # Step penalty
        
        if action.action_type == ActionType.PROCESS and action.target == "e1":
            target = s.entities[0]
            if target.state == EntityState.INACTIVE:
                target.state = EntityState.ACTIVE
                s.tasks[0].completed = True
                reward += 10.0
        
        s.turn += 1
        done = s.turn >= s.max_turns or s.tasks[0].completed
        s.done = done
        return StepResult(self.get_observation(), reward, done, s.turn >= s.max_turns, info)

    def get_observation(self) -> EnvironmentObservation:
        s = self._state
        return EnvironmentObservation(
            entities=copy.deepcopy(s.entities),
            tasks=[t for t in s.tasks if not t.completed],
            relationships=[],
            turn=s.turn, max_turns=s.max_turns
        )

    def render(self) -> str:
        s = self._state
        return f"Turn {s.turn} | Node: {s.entities[0].state}"
