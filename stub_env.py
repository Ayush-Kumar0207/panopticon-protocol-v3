"""
Stub Panopticon environment for E2E pipeline verification.
Minimal: 2 workers, 1 sleeper, basic canary flow.
"""

from __future__ import annotations
import random, copy
from dataclasses import dataclass
from models import (
    WorkerState, HiddenWorkerState, Department, ActionType, SubAction,
    Worker, LeakEvent, CanaryTrap, DoubleAgentAsset, HydraMemory,
    AgentAction, EnvironmentObservation, EnvironmentState,
    validate_action,
)


@dataclass
class StepResult:
    observation: EnvironmentObservation
    reward: float
    done: bool
    truncated: bool
    info: dict


class Environment:
    """Minimal Panopticon stub for pipeline verification."""

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)
        self._state = EnvironmentState()

    @property
    def state(self) -> EnvironmentState:
        return self._state

    def reset(self, task_level: str = "easy", seed: int | None = None) -> EnvironmentObservation:
        if seed is not None:
            self._rng = random.Random(seed)

        workers = [
            Worker(id="w-001", name="ATLAS", department="engineering",
                   state=WorkerState.LOYAL.value),
            Worker(id="w-002", name="BEACON", department="finance",
                   state=WorkerState.LOYAL.value,
                   is_sleeper=True, generation=1,
                   hidden_state=HiddenWorkerState.SLEEPER_ACTIVE.value),
        ]

        self._state = EnvironmentState(
            workers=workers, leaks=[], canary_traps=[], double_agents=[],
            hydra_memory=HydraMemory(),
            enterprise_revenue=100.0, security_score=100.0,
            turn=0, max_turns=15,
            difficulty="amateur", departments_active=["engineering", "finance"],
            total_sleepers_spawned=1,
        )
        return self.get_observation()

    def step(self, action: AgentAction) -> StepResult:
        s = self._state
        info = {"valid": True, "events": []}
        reward = -0.1  # Step cost

        at = ActionType(action.action_type) if action.action_type in [a.value for a in ActionType] else ActionType.NOOP

        if at == ActionType.WORK:
            s.enterprise_revenue += 2.0
            reward += 0.2

        elif at == ActionType.CANARY:
            trap = CanaryTrap(id=f"c-{len(s.canary_traps)+1}", department=action.target,
                              planted_turn=s.turn, unique_hash="stub-hash")
            s.canary_traps.append(trap)
            s.canaries_planted += 1

        elif at == ActionType.NEUTRALIZE:
            sa = SubAction(action.sub_action) if action.sub_action in [x.value for x in SubAction] else SubAction.TERMINATE
            worker = next((w for w in s.workers if w.id == action.target), None)
            if worker and worker.is_sleeper:
                if sa == SubAction.TERMINATE:
                    worker.state = WorkerState.TERMINATED.value
                    s.sleepers_caught += 1
                    reward += 5.0
                    s.security_score = min(100, s.security_score + 10)

        # Passive damage from active sleepers
        active = [w for w in s.workers if w.is_sleeper and w.state != WorkerState.TERMINATED.value]
        if active:
            s.security_score -= 2.0
            s.enterprise_revenue -= 1.0

        s.turn += 1
        done = s.turn >= s.max_turns or s.sleepers_caught >= 1
        s.done = done
        s.total_reward += reward
        return StepResult(self.get_observation(), reward, done, s.turn >= s.max_turns, info)

    def get_observation(self) -> EnvironmentObservation:
        s = self._state
        visible_workers = []
        for w in s.workers:
            vis = Worker(id=w.id, name=w.name, department=w.department, state=w.state,
                         performance=w.performance, loyalty_score=w.loyalty_score,
                         suspicion_level=w.suspicion_level)
            visible_workers.append(vis)
        return EnvironmentObservation(
            workers=visible_workers,
            canary_traps=copy.deepcopy(s.canary_traps),
            double_agents=[],
            enterprise_revenue=s.enterprise_revenue,
            security_score=s.security_score,
            turn=s.turn, max_turns=s.max_turns,
            entities=[{"id": w.id, "name": w.name, "state": w.state} for w in visible_workers],
            tasks=[{"id": "defend", "description": "Defend network", "completed": s.sleepers_caught > 0}],
        )

    def render(self) -> str:
        s = self._state
        return f"StubPanopticon Turn {s.turn} | Rev: {s.enterprise_revenue:.0f} | Sec: {s.security_score:.0f}"