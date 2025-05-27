import torch
import torch.nn as nn
import torch.nn.functional as F

class OpponentModel(nn.Module):
    def __init__(self, obs_dim, hidden_dim, act_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, act_dim)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.output(x), dim=-1)
