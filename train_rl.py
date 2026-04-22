"""
The Panopticon Protocol v3 — PPO Training Suite
=================================================
Phase-scheduled curriculum PPO with multi-head actor for
MultiDiscrete([8, 8, 7]) action space.

Supports:
  - 6-phase curriculum auto-escalation
  - 3 separate actor heads (action_type, target, sub_action)
  - Reward normalization for dual-objective balance
  - Model checkpointing with best-model selection
  - Resume from checkpoint
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from gym_wrapper import OpenEnvGymWrapper, OBS_SIZE, NUM_ACTION_TYPES, NUM_TARGETS, NUM_SUB_ACTIONS

# ── HYPERPARAMETERS ──
LEARNING_RATE = 3e-4
TOTAL_TIMESTEPS = 100_000
NUM_STEPS = 128
GAMMA = 0.99
GAE_LAMBDA = 0.95
UPDATE_EPOCHS = 4
NORM_ADV = True
CLIP_COEF = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5

# ── CURRICULUM SCHEDULE ──
CURRICULUM = {
    "phase_1": {"level": "easy",    "episodes": 500,  "timesteps": 15_000},
    "phase_2": {"level": "medium",  "episodes": 500,  "timesteps": 20_000},
    "phase_3": {"level": "hard",    "episodes": 1000, "timesteps": 25_000},
    "phase_4": {"level": "level_4", "episodes": 1000, "timesteps": 25_000},
    "phase_5": {"level": "level_5", "episodes": 500,  "timesteps": 20_000},
}


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PanopticonAgent(nn.Module):
    """
    3-head PPO agent for MultiDiscrete action space.
    Shared feature backbone → 3 separate actor heads + critic.
    """

    def __init__(self, obs_dim: int = OBS_SIZE):
        super().__init__()

        # Shared feature backbone
        self.backbone = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 128)),
            nn.ReLU(),
        )

        # 3 Actor heads (one per action dimension)
        self.head_action_type = nn.Sequential(
            layer_init(nn.Linear(128, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, NUM_ACTION_TYPES), std=0.01),
        )
        self.head_target = nn.Sequential(
            layer_init(nn.Linear(128, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, NUM_TARGETS), std=0.01),
        )
        self.head_sub_action = nn.Sequential(
            layer_init(nn.Linear(128, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, NUM_SUB_ACTIONS), std=0.01),
        )

        # Critic
        self.critic = nn.Sequential(
            layer_init(nn.Linear(128, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def get_value(self, x):
        features = self.backbone(x)
        return self.critic(features)

    def get_action_and_value(self, x, action=None):
        features = self.backbone(x)

        # 3 independent distributions
        logits_at = self.head_action_type(features)
        logits_tg = self.head_target(features)
        logits_sa = self.head_sub_action(features)

        dist_at = Categorical(logits=logits_at)
        dist_tg = Categorical(logits=logits_tg)
        dist_sa = Categorical(logits=logits_sa)

        if action is None:
            act_at = dist_at.sample()
            act_tg = dist_tg.sample()
            act_sa = dist_sa.sample()
            action = torch.stack([act_at, act_tg, act_sa], dim=-1)
        else:
            act_at = action[..., 0].long()
            act_tg = action[..., 1].long()
            act_sa = action[..., 2].long()

        # Sum log probs and entropies across 3 heads
        log_prob = dist_at.log_prob(act_at) + dist_tg.log_prob(act_tg) + dist_sa.log_prob(act_sa)
        entropy = dist_at.entropy() + dist_tg.entropy() + dist_sa.entropy()
        value = self.critic(features)

        return action, log_prob, entropy, value


# For backward compatibility
Agent = PanopticonAgent


def train(task_level: str = "medium", checkpoint_path: str = None,
          total_timesteps: int = None):
    """Train PPO on Panopticon Protocol."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = OpenEnvGymWrapper(task_level=task_level)

    agent = PanopticonAgent(obs_dim=OBS_SIZE).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)

    # Resume from checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"[*] Resuming from checkpoint: {checkpoint_path}")
        agent.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))

    timesteps = total_timesteps or TOTAL_TIMESTEPS

    # Storage
    obs_buf = torch.zeros((NUM_STEPS, OBS_SIZE)).to(device)
    actions_buf = torch.zeros((NUM_STEPS, 3)).to(device)
    logprobs_buf = torch.zeros(NUM_STEPS).to(device)
    rewards_buf = torch.zeros(NUM_STEPS).to(device)
    dones_buf = torch.zeros(NUM_STEPS).to(device)
    values_buf = torch.zeros(NUM_STEPS).to(device)

    next_obs, _ = env.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(1).to(device)

    print(f"[*] Training Panopticon v3 | Level: {task_level} | Device: {device}")
    best_mean_reward = -float("inf")
    episode_rewards = []
    current_ep_reward = 0.0

    for global_step in range(0, timesteps, NUM_STEPS):
        for step in range(NUM_STEPS):
            obs_buf[step] = next_obs
            dones_buf[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values_buf[step] = value.flatten()

            actions_buf[step] = action
            logprobs_buf[step] = logprob

            # Convert to numpy for env
            action_np = action.cpu().numpy().astype(int)
            next_obs_np, reward, terminated, truncated, info = env.step(action_np)

            rewards_buf[step] = torch.tensor(reward).to(device)
            next_obs = torch.Tensor(next_obs_np).to(device)
            next_done = torch.tensor(float(terminated or truncated)).to(device)

            current_ep_reward += reward
            if terminated or truncated:
                episode_rewards.append(current_ep_reward)
                current_ep_reward = 0.0
                next_obs_np, _ = env.reset()
                next_obs = torch.Tensor(next_obs_np).to(device)

        # ── GAE Bootstrapping ──
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards_buf).to(device)
            lastgaelam = 0
            for t in reversed(range(NUM_STEPS)):
                if t == NUM_STEPS - 1:
                    nextnonterminal = 1.0 - next_done.float()
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buf[t + 1]
                    nextvalues = values_buf[t + 1]
                delta = rewards_buf[t] + GAMMA * nextvalues * nextnonterminal - values_buf[t]
                advantages[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
            returns = advantages + values_buf

        # ── PPO Update ──
        b_obs = obs_buf
        b_logprobs = logprobs_buf
        b_actions = actions_buf
        b_advantages = advantages
        b_returns = returns

        inds = np.arange(NUM_STEPS)
        for epoch in range(UPDATE_EPOCHS):
            np.random.shuffle(inds)
            for start in range(0, NUM_STEPS, 32):
                end = start + 32
                mb_inds = inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                if NORM_ADV:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss (clipped)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # Entropy bonus
                entropy_loss = entropy.mean()

                loss = pg_loss - ENT_COEF * entropy_loss + v_loss * VF_COEF

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                optimizer.step()

        # ── Logging ──
        if global_step % (NUM_STEPS * 5) == 0:
            mean_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0.0
            print(
                f"[Step {global_step:>6d}] "
                f"Episodes: {len(episode_rewards):>4d} | "
                f"Mean Reward (last 10): {mean_reward:>7.2f} | "
                f"Loss: {loss.item():.4f}"
            )

            if mean_reward > best_mean_reward and len(episode_rewards) >= 5:
                best_mean_reward = mean_reward
                torch.save(agent.state_dict(), f"best_ppo_{task_level}.pt")
                print(f"  [SAVED] New best: best_ppo_{task_level}.pt (Reward: {best_mean_reward:.2f})")

    # Save final
    torch.save(agent.state_dict(), f"final_ppo_{task_level}.pt")
    print(f"[*] Training Complete. Saved: final_ppo_{task_level}.pt")


def train_curriculum(checkpoint_path: str = None):
    """Run full curriculum training across all difficulty phases."""
    print("=" * 60)
    print("  PANOPTICON v3 — Curriculum Training")
    print("=" * 60)

    current_checkpoint = checkpoint_path

    for phase_name, config in CURRICULUM.items():
        level = config["level"]
        timesteps = config["timesteps"]

        print(f"\n{'─' * 40}")
        print(f"  {phase_name.upper()}: {level} ({timesteps} timesteps)")
        print(f"{'─' * 40}")

        train(
            task_level=level,
            checkpoint_path=current_checkpoint,
            total_timesteps=timesteps,
        )

        # Use this phase's best model as next phase's starting point
        best_path = f"best_ppo_{level}.pt"
        if os.path.exists(best_path):
            current_checkpoint = best_path
        else:
            current_checkpoint = f"final_ppo_{level}.pt"

    print("\n" + "=" * 60)
    print("  CURRICULUM COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Panopticon v3 PPO Training")
    parser.add_argument("--level", type=str, default="medium", help="Task difficulty level")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pt checkpoint")
    parser.add_argument("--curriculum", action="store_true", help="Run full curriculum training")
    parser.add_argument("--timesteps", type=int, default=None, help="Override total timesteps")
    args = parser.parse_args()

    try:
        if args.curriculum:
            train_curriculum(checkpoint_path=args.checkpoint)
        else:
            train(
                task_level=args.level,
                checkpoint_path=args.checkpoint,
                total_timesteps=args.timesteps,
            )
    except KeyboardInterrupt:
        print("\n[!] Training interrupted by user.")
