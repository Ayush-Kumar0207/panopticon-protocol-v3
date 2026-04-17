"""
Elite OpenEnv Training Suite — Rugged RL Edition
================================================
Headless PyTorch PPO implementation based on CleanRL conventions.
Optimized for rapid adaptation during hackathon sprints.
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from gym_wrapper import OpenEnvGymWrapper

# HYPERPARAMETERS
LEARNING_RATE = 2.5e-4
TOTAL_TIMESTEPS = 100000
NUM_ENVS = 4
NUM_STEPS = 128
GAMMA = 0.99
GAE_LAMBDA = 0.95
UPDATE_EPOCHS = 4
NORM_ADV = True
CLIP_COEF = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, obs_shape, action_shape):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_shape), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

def train(task_level: str = "medium", checkpoint_path: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = OpenEnvGymWrapper(task_level=task_level)
    
    obs_shape = env.observation_space.shape[0]
    # Simple flattening for MultiDiscrete for this draft
    action_shape = env.action_space.nvec.prod() 

    agent = Agent(obs_shape, action_shape).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)

    # RESUME FROM CHECKPOINT
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"[*] Resuming from checkpoint: {checkpoint_path}")
        agent.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # STORAGE (Simplified for single-env draft)
    obs = torch.zeros((NUM_STEPS, obs_shape)).to(device)
    actions = torch.zeros(NUM_STEPS).to(device)
    logprobs = torch.zeros(NUM_STEPS).to(device)
    rewards = torch.zeros(NUM_STEPS).to(device)
    dones = torch.zeros(NUM_STEPS).to(device)
    values = torch.zeros(NUM_STEPS).to(device)

    # START TRAINING
    next_obs, _ = env.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(1).to(device)

    print(f"[*] Training on {task_level} using {device}...")
    best_mean_reward = -float("inf")
    
    for global_step in range(0, TOTAL_TIMESTEPS, NUM_STEPS):
        for step in range(0, NUM_STEPS):
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            
            actions[step] = action
            logprobs[step] = logprob

            # Convert flattened action index back to MultiDiscrete
            # [Type, Entity]
            act_type = action.item() // 20 
            act_ent = action.item() % 20
            
            next_obs, reward, terminated, truncated, info = env.step([act_type, act_ent])
            rewards[step] = torch.tensor(reward).to(device)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.tensor(terminated or truncated).to(device)

            if next_done:
                next_obs, _ = env.reset()
                next_obs = torch.Tensor(next_obs).to(device)

        # BOOTSTRAPPING & GAE
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(NUM_STEPS)):
                if t == NUM_STEPS - 1:
                    nextnonterminal = 1.0 - next_done.float()
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + GAMMA * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
            returns = advantages + values

        # OPTIMIZE
        b_obs = obs
        b_logprobs = logprobs
        b_actions = actions
        b_advantages = advantages
        b_returns = returns
        b_values = values

        inds = np.arange(NUM_STEPS)
        for epoch in range(UPDATE_EPOCHS):
            np.random.shuffle(inds)
            for start in range(0, NUM_STEPS, 32):
                end = start + 32
                mb_inds = inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds].long())
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                if NORM_ADV:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ENT_COEF * entropy_loss + v_loss * VF_COEF

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                optimizer.step()

        if global_step % 1280 == 0:
            current_mean_reward = rewards.mean().item()
            print(f"[Step {global_step}] Reward: {current_mean_reward:.2f} Loss: {loss.item():.4f}")
            
            # BEST MODEL CHECKPOINTING
            if current_mean_reward > best_mean_reward:
                best_mean_reward = current_mean_reward
                torch.save(agent.state_dict(), f"best_ppo_{task_level}.pt")
                print(f"  [SAVED] New best model: best_ppo_{task_level}.pt (Reward: {best_mean_reward:.2f})")

    # SAVE FINAL
    torch.save(agent.state_dict(), f"final_ppo_{task_level}.pt")
    print(f"[*] Training Complete. Model saved as final_ppo_{task_level}.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=str, default="medium", help="Task difficulty level")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pt checkpoint")
    args = parser.parse_args()
    
    try:
        train(task_level=args.level, checkpoint_path=args.checkpoint)
    except KeyboardInterrupt:
        print("\n[!] Training interrupted by user.")
