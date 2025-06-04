import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch import optim
import numpy as np
from collections import deque

class Policy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, act_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.output(x)

class CategoricalMasked(Categorical):
    def __init__(self, logits, mask):
        logits = logits.squeeze(0).clone()
        mask_tensor = torch.tensor(mask, dtype=torch.bool)

        if not mask_tensor.any():
            raise ValueError(f"[FATAL] No legal actions: {mask_tensor}")

        logits[~mask_tensor] = -1e10

        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logits = torch.nan_to_num(logits, nan=-1e10, posinf=1e10, neginf=-1e10)

        super().__init__(logits=logits)


class Agent:
    def __init__(self, obs_dim, act_dim, alpha=1e-2, gamma=0.99, hidden_dim=32):
        self.gamma = gamma
        self.policy = Policy(obs_dim, act_dim, hidden_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=alpha)

    def get_action(self, state, mask):
        state = torch.from_numpy(state).float().unsqueeze(0)
        logits = self.policy(state)
        mask_tensor = mask.clone().detach() if isinstance(mask, torch.Tensor) else torch.tensor(mask, dtype=torch.bool)

        if not any(mask):
            # Fallback: always pick 'fold'
            action = 0
            return action, torch.tensor(0.0), torch.tensor(0.0)

        m = CategoricalMasked(logits, mask_tensor)
        action = m.sample()
        return action.item(), m.log_prob(action), torch.tensor(0.0)


    def update(self, log_probs, rewards):
        returns = deque()
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.appendleft(G)
        returns = torch.tensor(list(returns), dtype=torch.float32)

        log_probs = torch.stack(log_probs)
        eps = np.finfo(np.float32).eps.item()
        returns = (returns - returns.mean()) / (returns.std() + eps)

        loss = -(log_probs * returns).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class StrongOpponent:
    def __init__(self, obs_dim, act_dim):
        self.agent = Agent(obs_dim, act_dim)

    def get_action(self, state, mask):
        return self.agent.get_action(state, mask)

    def update(self, log_probs, rewards):
        self.agent.update(log_probs, rewards)
