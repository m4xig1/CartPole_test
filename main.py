import gymnasium as gym

# import gym
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from collections import defaultdict


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.fc(x)


tmp_env = gym.make("CartPole-v1", render_mode="rgb_array")
env = gym.wrappers.RecordVideo(
    env=tmp_env,
    video_folder="./videos_0_0/",
    name_prefix="test-video",
    episode_trigger=lambda x: x % 50 == 0,
)
env.action_space.seed(42)


# Config
lr = 0.01
gamma = 0.99  # Discount factor
lambda_status_quo = 0  # Penalty for changing actions
lambda_novelty = 0  # Reward for exploring new state-action pairs

policy_net = PolicyNetwork(env.observation_space.shape[0], env.action_space.n)
optimizer = Adam(policy_net.parameters(), lr=lr)

# Track novelty
state_action_counts = defaultdict(int)


def compute_modified_reward(state, action, prev_action, base_reward):
    """
    Modify the reward function to include status quo bias and novelty seeking.

    Args:
        state (tuple): Current state (discretized for novelty tracking).
        action (int): Current action.
        prev_action (int): Previous action.
        base_reward (float): Original environment reward.

    Returns:
        float: Modified reward.
    """
    # Status quo bias: penalize for changing actions
    status_quo_penalty = -lambda_status_quo if action != prev_action else 0

    # Novelty seeking: reward less-visited state-action pairs
    state_action_key = (tuple(state), action)
    novelty_bonus = lambda_novelty / (1 + state_action_counts[state_action_key])
    state_action_counts[state_action_key] += 1

    return base_reward + status_quo_penalty + novelty_bonus


def train_agent(num_episodes=500):
    for episode in range(num_episodes):
        state, info = env.reset()
        prev_action = None
        rewards = []
        log_probs = []

        for t in range(200):  # Limit to 200 steps per episode
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action_probs = policy_net(state_tensor)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample().item()

            next_state, base_reward, done, _, _ = env.step(action)

            # Ensure state is converted to tuple for tracking novelty
            state_tuple = tuple(state) if isinstance(state, np.ndarray) else state

            reward = compute_modified_reward(
                state_tuple, action, prev_action, base_reward
            )

            # Store log-probabilities and rewards
            log_probs.append(action_dist.log_prob(torch.tensor(action)))
            rewards.append(reward)

            # Update state and action
            state = next_state
            prev_action = action

            if done:
                break

        discounted_rewards = []
        cumulative_reward = 0
        for r in reversed(rewards):
            cumulative_reward = r + gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)

        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)

        # Normalize rewards
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
            discounted_rewards.std() + 1e-8
        )

        loss = -torch.sum(torch.stack(log_probs) * discounted_rewards)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % 50 == 0:
            print(f"Episode {episode}, Total Reward: {sum(rewards):.2f}")


train_agent(num_episodes=500)

env.close()
