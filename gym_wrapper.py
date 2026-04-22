"""
The Panopticon Protocol v3 — Gymnasium Wrapper
================================================
Standardized Gymnasium interface for PPO/CleanRL training.
Encodes the espionage observation into a fixed-size float32 tensor
and decodes MultiDiscrete([8, 8, 4]) actions back to AgentAction.
"""

from __future__ import annotations
from typing import Any, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from environment import Environment
from models import (
    ActionType, SubAction, Department, LeakChannel,
    AgentAction, EnvironmentObservation, WorkerState,
)

# ── CAPACITY CONSTANTS ──
MAX_WORKERS = 12
MAX_LEAKS = 8
MAX_CANARIES = 6
MAX_DOUBLE_AGENTS = 3

WORKER_FEATS = 6    # dept_enc, state_enc, performance, loyalty, suspicion, turning_flag
LEAK_FEATS = 4      # channel_enc, dept_enc, is_canary, verified
CANARY_FEATS = 3    # dept_enc, turn_planted_norm, triggered
DA_FEATS = 3        # trust, effectiveness, disinfo_count_norm
GLOBAL_FEATS = 5    # revenue_norm, security_norm, turn_norm, max_turns_norm, phase_norm

# TOTAL: 12*6 + 8*4 + 6*3 + 3*3 + 5 = 72 + 32 + 18 + 9 + 5 = 136
OBS_SIZE = (MAX_WORKERS * WORKER_FEATS) + (MAX_LEAKS * LEAK_FEATS) + \
           (MAX_CANARIES * CANARY_FEATS) + (MAX_DOUBLE_AGENTS * DA_FEATS) + GLOBAL_FEATS

# Action space sizes
NUM_ACTION_TYPES = len(ActionType)       # 8
NUM_TARGETS = 8                          # Workers 0-7, or department/channel index
NUM_SUB_ACTIONS = len(SubAction)         # 7

# Department and channel encoding maps
DEPT_INDEX = {d.value: i / len(Department) for i, d in enumerate(Department)}
CHANNEL_INDEX = {c.value: i / len(LeakChannel) for i, c in enumerate(LeakChannel)}
STATE_INDEX = {s.value: i / len(WorkerState) for i, s in enumerate(WorkerState)}

# Target mapping: indices 0-7 can represent worker indices, departments, or channels
DEPT_LIST = [d.value for d in Department]
CHANNEL_LIST = [c.value for c in LeakChannel]


class OpenEnvGymWrapper(gym.Env):
    """
    Panopticon Protocol v3 — Gymnasium adapter.
    Observation: Box(136,) float32
    Action: MultiDiscrete([8, 8, 7])
    """

    def __init__(self, task_level: str = "medium", seed: Optional[int] = None):
        super().__init__()
        self._env = Environment(seed=seed)
        self._task_level = task_level
        self._last_obs: Optional[EnvironmentObservation] = None

        # ── Spaces ──
        self.observation_space = spaces.Box(
            low=-1.0, high=2.0, shape=(OBS_SIZE,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([NUM_ACTION_TYPES, NUM_TARGETS, NUM_SUB_ACTIONS])

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._last_obs = self._env.reset(task_level=self._task_level, seed=seed)
        return self._flatten_obs(self._last_obs), {"level": self._task_level}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        agent_action = self._decode_action(action)
        result = self._env.step(agent_action)
        self._last_obs = result.observation
        return (
            self._flatten_obs(result.observation),
            float(result.reward),
            result.done,
            result.truncated,
            result.info,
        )

    def _decode_action(self, action: np.ndarray) -> AgentAction:
        """Convert MultiDiscrete indices to an AgentAction."""
        idx_type, idx_target, idx_sub = int(action[0]), int(action[1]), int(action[2])

        action_types = list(ActionType)
        at = action_types[idx_type % len(action_types)]

        sub_actions = list(SubAction)
        sa = sub_actions[idx_sub % len(sub_actions)]

        # Resolve target based on action type
        target = ""
        if at in (ActionType.WORK, ActionType.HIRE, ActionType.CANARY):
            # Target is a department
            target = DEPT_LIST[idx_target % len(DEPT_LIST)]
        elif at == ActionType.MONITOR:
            # Target is a leak channel
            target = CHANNEL_LIST[idx_target % len(CHANNEL_LIST)]
        elif at in (ActionType.INVESTIGATE, ActionType.NEUTRALIZE, ActionType.DEPLOY_DOUBLE):
            # Target is a worker by index
            if self._last_obs and idx_target < len(self._last_obs.workers):
                worker = self._last_obs.workers[idx_target]
                target = worker.id
            elif self._last_obs and self._last_obs.workers:
                target = self._last_obs.workers[0].id

            # For INVESTIGATE, also resolve target as department for CORRELATE
            if at == ActionType.INVESTIGATE and sa == SubAction.CORRELATE:
                target = DEPT_LIST[idx_target % len(DEPT_LIST)]
            elif at == ActionType.INVESTIGATE and sa == SubAction.VERIFY:
                # Target should be a leak ID
                if self._last_obs and idx_target < len(self._last_obs.active_leaks):
                    target = self._last_obs.active_leaks[idx_target].id
                elif self._last_obs and self._last_obs.active_leaks:
                    target = self._last_obs.active_leaks[0].id

        # Default sub-action based on action type
        if at == ActionType.INVESTIGATE and sa not in (SubAction.AUDIT, SubAction.VERIFY, SubAction.CORRELATE):
            sa = SubAction.AUDIT
        elif at == ActionType.NEUTRALIZE and sa not in (SubAction.TERMINATE, SubAction.INTERROGATE, SubAction.TURN):
            sa = SubAction.TERMINATE

        return AgentAction(
            action_type=at.value,
            target=target,
            sub_action=sa.value,
            reason="RL Policy Step",
        )

    def _flatten_obs(self, obs: EnvironmentObservation) -> np.ndarray:
        """Convert structured observation to flat float32 tensor."""
        vec = np.zeros(OBS_SIZE, dtype=np.float32)
        offset = 0

        # ── Encode Workers ──
        for i, w in enumerate(obs.workers[:MAX_WORKERS]):
            base = offset + (i * WORKER_FEATS)
            vec[base + 0] = DEPT_INDEX.get(w.department, 0.0)
            vec[base + 1] = STATE_INDEX.get(w.state, 0.0)
            vec[base + 2] = w.performance
            vec[base + 3] = w.loyalty_score
            vec[base + 4] = w.suspicion_level
            vec[base + 5] = 1.0 if w.turning_in_progress else 0.0
        offset += MAX_WORKERS * WORKER_FEATS

        # ── Encode Leaks ──
        for i, leak in enumerate(obs.active_leaks[:MAX_LEAKS]):
            base = offset + (i * LEAK_FEATS)
            vec[base + 0] = CHANNEL_INDEX.get(leak.channel, 0.0)
            vec[base + 1] = DEPT_INDEX.get(leak.department, 0.0)
            vec[base + 2] = 1.0 if leak.is_canary else 0.0
            vec[base + 3] = 1.0 if leak.verified else 0.0
        offset += MAX_LEAKS * LEAK_FEATS

        # ── Encode Canaries ──
        for i, trap in enumerate(obs.canary_traps[:MAX_CANARIES]):
            base = offset + (i * CANARY_FEATS)
            vec[base + 0] = DEPT_INDEX.get(trap.department, 0.0)
            vec[base + 1] = trap.planted_turn / max(obs.max_turns, 1)
            vec[base + 2] = 1.0 if trap.triggered else 0.0
        offset += MAX_CANARIES * CANARY_FEATS

        # ── Encode Double Agents ──
        for i, da in enumerate(obs.double_agents[:MAX_DOUBLE_AGENTS]):
            base = offset + (i * DA_FEATS)
            vec[base + 0] = da.hydra_trust
            vec[base + 1] = da.effectiveness
            vec[base + 2] = min(1.0, da.disinfo_fed_count / 5.0)
        offset += MAX_DOUBLE_AGENTS * DA_FEATS

        # ── Global Features ──
        vec[offset + 0] = obs.enterprise_revenue / 150.0
        vec[offset + 1] = obs.security_score / 100.0
        vec[offset + 2] = obs.turn / max(obs.max_turns, 1)
        vec[offset + 3] = obs.max_turns / 200.0
        vec[offset + 4] = obs.phase_number / 6.0

        return vec

    def get_action_mask(self) -> np.ndarray:
        """Boolean mask of valid actions (for masked PPO)."""
        mask = np.ones((NUM_ACTION_TYPES, NUM_TARGETS, NUM_SUB_ACTIONS), dtype=bool)

        if not self._last_obs:
            return mask

        # Disable DEPLOY_DOUBLE if no double agents
        if not self._last_obs.double_agents:
            da_idx = list(ActionType).index(ActionType.DEPLOY_DOUBLE)
            mask[da_idx, :, :] = False

        return mask
