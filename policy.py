import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical

from constants import OBSERVATION_SPACE_SIZE, ACTION_SPACE_SIZE


class Policy(nn.Module):
    def __init__(self) -> None:
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(OBSERVATION_SPACE_SIZE, 128)
        self.fc2 = nn.Linear(128, ACTION_SPACE_SIZE)

    def forward(self, x: int) -> Tensor:
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def act(self, state) -> tuple:
        # TODO: Make .to(device) programmatic
        state = torch.from_numpy(state).float().unsqueeze(0).to("cpu")
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return (action.item(), m.log_prob(action))
