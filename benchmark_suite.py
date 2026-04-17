"""
Elite OpenEnv Benchmark Suite — Rugged RL Edition
=================================================
Headless evaluation engine for comparing agent architectures.
Outputs rigorous performance metrics directly to terminal.
"""

from __future__ import annotations
import time
import torch
import numpy as np
from typing import List, Dict, Type
from gym_wrapper import OpenEnvGymWrapper
from train_rl import Agent

class BaseAgent:
    def act(self, obs: np.ndarray, env: OpenEnvGymWrapper) -> np.ndarray:
        raise NotImplementedError

class RandomAgent(BaseAgent):
    def act(self, obs: np.ndarray, env: OpenEnvGymWrapper) -> np.ndarray:
        return env.action_space.sample()

class HeuristicAgent(BaseAgent):
    def act(self, obs: np.ndarray, env: OpenEnvGymWrapper) -> np.ndarray:
        # Simple heuristic: find first uncompleted task and its target
        last_obs = env._last_obs
        if not last_obs or not last_obs.tasks:
            return np.array([4, 0]) # NOOP
        
        # Priority mapping
        for task in sorted(last_obs.tasks, key=lambda x: x.priority, reverse=True):
            if not task.completed and task.target_entities:
                target_id = task.target_entities[0]
                target_idx = next((i for i, e in enumerate(last_obs.entities) if e.id == target_id), 0)
                return np.array([1, target_idx]) # PROCESS
        return np.array([4, 0])

class RLAgent(BaseAgent):
    def __init__(self, model_path: str, obs_dim: int, act_dim: int):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Agent(obs_dim, act_dim).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def act(self, obs: np.ndarray, env: OpenEnvGymWrapper) -> np.ndarray:
        obs_tensor = torch.Tensor(obs).to(self.device)
        with torch.no_grad():
            action, _, _, _ = self.model.get_action_and_value(obs_tensor)
        
        # Convert flattened back to [Type, Entity]
        val = action.item()
        return np.array([val // 20, val % 20])

def evaluate(agent: BaseAgent, task_level: str, num_episodes: int = 20) -> Dict:
    env = OpenEnvGymWrapper(task_level=task_level)
    scores, steps, successes = [], [], []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done, truncated, ep_reward, ep_steps = False, False, 0.0, 0
        while not (done or truncated):
            action = agent.act(obs, env)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            ep_steps += 1
        
        scores.append(ep_reward)
        steps.append(ep_steps)
        successes.append(1.0 if (env._last_obs and all(t.completed for t in env._env.state.tasks)) else 0.0)

    return {
        "mean_score": np.mean(scores),
        "std_score": np.std(scores),
        "mean_steps": np.mean(steps),
        "success_rate": np.mean(successes) * 100
    }

def run_suite(model_path: Optional[str] = None):
    levels = ["easy", "medium", "hard"]
    agents = {"Random": RandomAgent(), "Heuristic": HeuristicAgent()}
    
    if model_path and os.path.exists(model_path):
        env = OpenEnvGymWrapper()
        agents["RL"] = RLAgent(model_path, env.observation_space.shape[0], env.action_space.nvec.prod())

    print(f"\n{'='*60}")
    print(f"{'AGENT':<12} | {'LEVEL':<8} | {'SCORE (±STD)':<15} | {'SR %':<6} | {'STEPS'}")
    print(f"{'-'*60}")

    for agent_name, agent in agents.items():
        for level in levels:
            res = evaluate(agent, level)
            print(f"{agent_name:<12} | {level:<8} | {res['mean_score']:>6.1f} ± {res['std_score']:<5.1f} | {res['success_rate']:>5.1f}% | {res['mean_steps']:>4.1f}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    import os
    latest_model = "ppomodel_medium.pt"
    run_suite(latest_model if os.path.exists(latest_model) else None)
