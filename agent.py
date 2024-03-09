import numpy as np
import torch
from torch import optim
from collections import deque
from constants import DEFAULT_AGENT_NAME
from policy import Policy


class Agent:
    def __init__(self, alpha: float, gamma: float, theta: Policy, name=DEFAULT_AGENT_NAME) -> None:
        self._name = name

        self._alpha = alpha
        self._gamma = gamma
        self._theta = theta

        self._optimizer = optim.Adam(self._theta.parameters(), lr=self._alpha)

    def get_name(self):
        return self._name

    def get_action(self, state, mask):
        return self._theta.act(state, mask)

    def REINFORCE(self, log_probs, rewards) -> None:
        returns = deque()
        for t in reversed(range(len(rewards))):
            G_t = returns[0] if len(returns) > 0 else 0
            returns.appendleft(self._gamma * G_t + rewards[t])
        eps = np.finfo(np.float32).eps.item()
        returns = torch.tensor(returns)
        if len(list(returns.size())) > 1:
            returns = (returns - returns.mean()) / (returns.std() + eps)

        policy_loss = []
        for log_prob, G_t in zip(log_probs, returns):
            policy_loss.append(-log_prob * G_t)
        policy_loss = torch.cat(policy_loss).sum()

        self._optimizer.zero_grad()
        policy_loss.backward()
        self._optimizer.step()
