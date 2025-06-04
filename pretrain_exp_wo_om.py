import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
from pettingzoo.classic import texas_holdem_no_limit_v6
from torch.distributions import Categorical

# === Constants ===
NUM_PLAYERS = 2
OBSERVATION_SPACE_SIZE = 54  # no dummy opponent action
ACTION_SPACE_SIZE = 5
HIDDEN_LAYER_SIZE = 32


# === Model Definition ===
class PolicyWithValue(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, act_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.policy_head(x), self.value_head(x)


# === Agent ===
class BaselineAgent:
    def __init__(self, alpha=0.01, gamma=0.99):
        self.gamma = gamma
        self.policy = PolicyWithValue(OBSERVATION_SPACE_SIZE, ACTION_SPACE_SIZE, HIDDEN_LAYER_SIZE)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=alpha)

    def get_action(self, obs, mask):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits, value = self.policy(obs_tensor)

        probs = torch.softmax(logits, dim=-1).squeeze()
        mask_tensor = torch.tensor(mask, dtype=torch.float32)
        masked_probs = probs * mask_tensor
        masked_probs /= masked_probs.sum() + 1e-8

        dist = Categorical(masked_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value.squeeze()

    def update(self, log_probs, values, rewards):
        returns, G = [], 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        values = torch.stack(values)
        log_probs = torch.stack(log_probs)
        advantages = returns - values.detach()
        entropy = -(log_probs.exp() * log_probs).sum()
        policy_loss = -(log_probs * advantages).sum()
        value_loss = F.mse_loss(values, returns)
        total_loss = policy_loss + value_loss - 0.01 * entropy

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()


# === Training Loop ===
def pretrain_agent(episodes=10000, save_path="medium_agent_2.pth"):
    env = texas_holdem_no_limit_v6.env(render_mode=None, num_players=NUM_PLAYERS)
    agent = BaselineAgent()

    for ep in range(1, episodes + 1):
        env.reset()
        log_probs, values, rewards = [], [], []

        for name in env.agent_iter():
            obs, rew, term, trunc, _ = env.last()
            if term or trunc:
                env.step(None)
                continue

            mask = obs["action_mask"]
            state = obs["observation"]

            if name == "player_0":
                action, log_prob, value = agent.get_action(state, mask)
                log_probs.append(log_prob)
                values.append(value)
                rewards.append(rew)
            else:
                valid = [i for i, m in enumerate(mask) if m == 1]
                action = np.random.choice(valid) if valid else 0

            env.step(action)

        if log_probs:
            agent.update(log_probs, values, rewards)

        if ep % 100 == 0:
            print(f"[Ep {ep}] Final Reward = {sum(rewards):.2f}")

    torch.save(agent.policy.state_dict(), save_path)
    print(f"Medium opponent (54D) saved to {save_path}")


if __name__ == "__main__":
    pretrain_agent()
