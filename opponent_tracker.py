import torch
import torch.nn as nn
import torch.optim as optim

class OpponentTracker:
    def __init__(self, model, lr=1e-3, buffer_size=100):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.buffer = []
        self.buffer_size = buffer_size
        self.loss_fn = nn.NLLLoss()

    def observe(self, obs, action):
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        action_tensor = torch.tensor(action, dtype=torch.long)
        self.buffer.append((obs_tensor, action_tensor))
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def train_step(self):
        if len(self.buffer) < 10:
            return 
        obs_batch, action_batch = zip(*self.buffer)
        obs_batch = torch.stack(obs_batch)
        action_batch = torch.tensor(action_batch)

        self.optimizer.zero_grad()
        log_probs = self.model(obs_batch)
        loss = self.loss_fn(log_probs, action_batch)
        loss.backward()
        self.optimizer.step()
