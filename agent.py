import numpy as np
import torch
from torch import optim
from collections import deque
from constants import DEFAULT_AGENT_NAME, CPU
from distribution import CategoricalMasked
from policy import Policy


class Agent:
    def __init__(self, alpha: float, gamma: float, theta: Policy, name=DEFAULT_AGENT_NAME) -> None:
        self._name = name

        self._alpha = alpha
        self._gamma = gamma
        self._theta = theta

        self._optimizer = optim.AdamW(self._theta.parameters(), lr=self._alpha)

    def get_name(self) -> str:
        return self._name

    def get_action(self, state, mask):
        state = torch.from_numpy(state).float().unsqueeze(0).to(CPU)
        logits = self._theta(state)
        m = CategoricalMasked(logits, torch.tensor(mask, dtype=torch.bool))
        action = m.sample()
        return action.item(), m.log_prob(action)

    def REINFORCE(self, log_probs, rewards) -> None:
        returns = deque()
        for R_t in reversed(rewards):
            G_t = returns[0] if len(returns) > 0 else 0
            returns.appendleft(self._gamma * G_t + R_t)
        returns = torch.tensor(returns)

        eps = np.finfo(np.float32).eps.item()
        if len(list(returns.size())) > 1:
            returns = (returns - returns.mean()) / (returns.std() + eps)

        policy_loss = []
        for log_prob, G_t in zip(log_probs, returns):
            policy_loss.append(-log_prob * G_t)
        policy_loss = torch.cat(policy_loss).sum()

        self._optimizer.zero_grad()
        policy_loss.backward()
        self._optimizer.step()
