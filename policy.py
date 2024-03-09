import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from constants import CPU


class CategoricalMasked(Categorical):
    def __init__(self, logits: torch.Tensor, mask: torch.Tensor) -> None:
        ninf = torch.tensor(torch.finfo(logits.dtype).min, dtype=logits.dtype)
        logits = torch.where(mask, logits, ninf)
        super(CategoricalMasked, self).__init__(logits=logits)


class Policy(nn.Module):
    def __init__(self, observation_space_size: int, action_space_size: int, hidden_layer_size: int) -> None:
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(observation_space_size, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size * 2)
        self.fc3 = nn.Linear(hidden_layer_size * 2, action_space_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def act(self, state, mask: list[int]):
        state = torch.from_numpy(state).float().unsqueeze(0).to(CPU)
        logits = self.forward(state).cpu()
        m = CategoricalMasked(logits, torch.tensor(mask, dtype=torch.bool))
        action = m.sample()
        return action.item(), m.log_prob(action)
