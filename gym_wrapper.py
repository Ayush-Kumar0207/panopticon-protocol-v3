"""
Elite OpenEnv Gymnasium Wrapper — Rugged RL Edition
===================================================
Standardized Gymnasium interface for universal OpenEnv tasks.
Optimized for TorchRL/CleanRL workflows.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from environment import Environment
from models import ActionType, AgentAction, EnvironmentObservation

# CAPACITY CONSTANTS (Adjust these if problem complexity exceeds limits)
MAX_ENTITIES = 20
MAX_TASKS = 10
ENTITY_FEATS = 5  # tier, state, score, exists_flag, normalized_id
TASK_FEATS = 3    # priority, completed_flag, exists_flag

# TOTAL SIZE: 20*5 + 10*3 + 2 (turn, max_turns) = 132
OBS_SIZE = (MAX_ENTITIES * ENTITY_FEATS) + (MAX_TASKS * TASK_FEATS) + 2

class OpenEnvGymWrapper(gym.Env):
    """
    Universal adapter for OpenEnv Pydantic models to fixed-size numpy tensors.
    """
    def __init__(self, task_level: str = "medium", seed: Optional[int] = None):
        super().__init__()
        self._env = Environment(seed=seed)
        self._task_level = task_level
        self._last_obs: Optional[EnvironmentObservation] = None

        # ── Spaces ───────────────────────────────────────────────────────────
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32
        )
        
        # Action: [ActionTypeIndex, EntityIndex]
        self.action_space = spaces.MultiDiscrete([len(ActionType), MAX_ENTITIES])

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Standard Gymnasium Reset."""
        super().reset(seed=seed)
        self._last_obs = self._env.reset(task_level=self._task_level, seed=seed)
        return self._flatten_obs(self._last_obs), {"level": self._task_level}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Standard Gymnasium Step."""
        idx_type, idx_entity = action
        
        # Map indices back to Pydantic models
        action_types = list(ActionType)
        at = action_types[idx_type]
        
        target_id = ""
        if idx_entity < len(self._last_obs.entities):
            target_id = self._last_obs.entities[idx_entity].id

        agent_action = AgentAction(
            action_type=at,
            target=target_id,
            reason="Adaptive RL Step"
        )
        
        result = self._env.step(agent_action)
        self._last_obs = result.observation
        
        return (
            self._flatten_obs(result.observation),
            float(result.reward),
            result.done,
            result.truncated,
            result.info
        )

    def _flatten_obs(self, obs: EnvironmentObservation) -> np.ndarray:
        """Converts structured Pydantic models to a flat float32 vector."""
        vec = np.zeros(OBS_SIZE, dtype=np.float32)
        offset = 0

        # Encode Entities
        for i, entity in enumerate(obs.entities[:MAX_ENTITIES]):
            base = offset + (i * ENTITY_FEATS)
            vec[base + 0] = float(entity.tier) / 3.0
            vec[base + 1] = 1.0 if entity.state == "active" else 0.0 # Standardize
            vec[base + 2] = entity.score / 100.0
            vec[base + 3] = 1.0 # exists_flag
            vec[base + 4] = i / MAX_ENTITIES # normalized_pos
        offset += MAX_ENTITIES * ENTITY_FEATS

        # Encode Tasks
        for i, task in enumerate(obs.tasks[:MAX_TASKS]):
            base = offset + (i * TASK_FEATS)
            vec[base + 0] = task.priority / 10.0
            vec[base + 1] = 1.0 if task.completed else 0.0
            vec[base + 2] = 1.0 # exists_flag
        offset += MAX_TASKS * TASK_FEATS

        # Global Features
        vec[offset + 0] = float(obs.turn) / float(obs.max_turns)
        vec[offset + 1] = float(obs.max_turns) / 100.0
        
        return vec

    def get_action_mask(self) -> np.ndarray:
        """
        Returns a boolean mask of valid (action_type, entity) pairs.
        Shape: (ActionTypeCount, MAX_ENTITIES)
        """
        mask = np.zeros((len(ActionType), MAX_ENTITIES), dtype=bool)
        if not self._last_obs:
            return mask

        num_entities = len(self._last_obs.entities)
        for t_idx, at in enumerate(list(ActionType)):
            for e_idx in range(MAX_ENTITIES):
                if e_idx < num_entities:
                    mask[t_idx, e_idx] = True # Simplistic: all existing entities valid
                elif at == ActionType.NOOP or at == ActionType.WAIT:
                    mask[t_idx, e_idx] = True # Non-target actions always valid
        
        return mask
