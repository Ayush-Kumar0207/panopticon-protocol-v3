"""
The Panopticon Protocol v3 — Benchmark Suite
==============================================
Headless evaluation engine comparing Random, Heuristic, and RL agents
across all difficulty tiers.
"""

from __future__ import annotations
import os
import numpy as np
import torch
from typing import Dict, Optional
from gym_wrapper import OpenEnvGymWrapper, OBS_SIZE, NUM_ACTION_TYPES, NUM_TARGETS, NUM_SUB_ACTIONS
from models import ActionType, SubAction, Department, LeakChannel


class BaseAgent:
    def act(self, obs: np.ndarray, env: OpenEnvGymWrapper) -> np.ndarray:
        raise NotImplementedError


class RandomAgent(BaseAgent):
    def act(self, obs: np.ndarray, env: OpenEnvGymWrapper) -> np.ndarray:
        return env.action_space.sample()


class HeuristicAgent(BaseAgent):
    """ARGUS heuristic: canary → monitor → investigate → neutralize cycle."""

    def __init__(self):
        self._turn = 0
        self._depts = [d.value for d in Department]
        self._channels = [c.value for c in LeakChannel]

    def act(self, obs: np.ndarray, env: OpenEnvGymWrapper) -> np.ndarray:
        self._turn += 1
        last = env._last_obs

        if not last:
            return np.array([list(ActionType).index(ActionType.NOOP), 0, 0])

        # Phase 1: Plant canaries (first 4 turns)
        if self._turn <= 4:
            dept_idx = (self._turn - 1) % len(self._depts)
            return np.array([
                list(ActionType).index(ActionType.CANARY),
                dept_idx,
                list(SubAction).index(SubAction.NONE),
            ])

        # Check for high-suspicion workers to neutralize
        for i, w in enumerate(last.workers):
            if w.suspicion_level > 0.7 and w.state not in ("terminated", "double_agent"):
                return np.array([
                    list(ActionType).index(ActionType.NEUTRALIZE),
                    i,
                    list(SubAction).index(SubAction.TERMINATE),
                ])

        # Monitor channels
        if self._turn % 3 == 0:
            ch_idx = (self._turn // 3) % len(self._channels)
            return np.array([
                list(ActionType).index(ActionType.MONITOR),
                ch_idx,
                list(SubAction).index(SubAction.NONE),
            ])

        # Work for revenue
        if self._turn % 3 == 1:
            dept_idx = self._turn % len(self._depts)
            return np.array([
                list(ActionType).index(ActionType.WORK),
                dept_idx,
                list(SubAction).index(SubAction.NONE),
            ])

        # Audit the most suspicious worker
        if last.workers:
            most_sus = max(enumerate(last.workers), key=lambda x: x[1].suspicion_level)
            if most_sus[1].suspicion_level > 0.15:
                return np.array([
                    list(ActionType).index(ActionType.INVESTIGATE),
                    most_sus[0],
                    list(SubAction).index(SubAction.AUDIT),
                ])

        # Default: work
        return np.array([
            list(ActionType).index(ActionType.WORK),
            0,
            list(SubAction).index(SubAction.NONE),
        ])


class RLAgent(BaseAgent):
    def __init__(self, model_path: str):
        from train_rl import PanopticonAgent
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PanopticonAgent(obs_dim=OBS_SIZE).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def act(self, obs: np.ndarray, env: OpenEnvGymWrapper) -> np.ndarray:
        obs_tensor = torch.Tensor(obs).to(self.device)
        with torch.no_grad():
            action, _, _, _ = self.model.get_action_and_value(obs_tensor)
        return action.cpu().numpy().astype(int)


def evaluate(agent: BaseAgent, task_level: str, num_episodes: int = 10) -> Dict:
    env = OpenEnvGymWrapper(task_level=task_level)
    scores, steps_list, revenues, securities = [], [], [], []

    for _ in range(num_episodes):
        if isinstance(agent, HeuristicAgent):
            agent._turn = 0
        obs, _ = env.reset()
        done, truncated, ep_reward, ep_steps = False, False, 0.0, 0

        while not (done or truncated):
            action = agent.act(obs, env)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            ep_steps += 1

        scores.append(ep_reward)
        steps_list.append(ep_steps)

        if env._last_obs:
            revenues.append(env._last_obs.enterprise_revenue)
            securities.append(env._last_obs.security_score)

    return {
        "mean_score": np.mean(scores),
        "std_score": np.std(scores),
        "mean_steps": np.mean(steps_list),
        "mean_revenue": np.mean(revenues) if revenues else 0,
        "mean_security": np.mean(securities) if securities else 0,
    }


def run_suite(model_path: Optional[str] = None):
    levels = ["easy", "medium", "hard"]
    agents = {"Random": RandomAgent(), "Heuristic": HeuristicAgent()}

    if model_path and os.path.exists(model_path):
        agents["RL"] = RLAgent(model_path)

    print(f"\n{'='*75}")
    print(f"  👁️  PANOPTICON v3 — Benchmark Suite")
    print(f"{'='*75}")
    print(f"{'AGENT':<12} | {'LEVEL':<8} | {'SCORE (±STD)':<15} | {'REV':>5} | {'SEC':>5} | {'STEPS':>5}")
    print(f"{'-'*75}")

    for agent_name, agent in agents.items():
        for level in levels:
            res = evaluate(agent, level)
            print(
                f"{agent_name:<12} | {level:<8} | "
                f"{res['mean_score']:>6.1f} ± {res['std_score']:<5.1f} | "
                f"{res['mean_revenue']:>5.0f} | "
                f"{res['mean_security']:>5.0f} | "
                f"{res['mean_steps']:>5.1f}"
            )
    print(f"{'='*75}\n")


if __name__ == "__main__":
    latest_model = "best_ppo_medium.pt"
    run_suite(latest_model if os.path.exists(latest_model) else None)
