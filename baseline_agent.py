import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

OBSERVATION_SPACE_SIZE = 54
ACTION_SPACE_SIZE = 5
HIDDEN_LAYER_SIZE = 32

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
        sum_probs = masked_probs.sum()

        if sum_probs.item() == 0 or torch.isnan(sum_probs):
            valid_indices = (mask_tensor == 1).nonzero(as_tuple=True)[0]
            masked_probs = torch.zeros_like(mask_tensor)
            masked_probs[valid_indices] = 1.0 / len(valid_indices)
        else:
            masked_probs /= sum_probs

        # === Bluffing logic (Monte Carlo override) ===
        try:
            hole_cards, community_cards = decode_cards(obs)
            bluff_score = estimate_bluff_score(hole_cards, community_cards)

            if bluff_score < 0.9:
                aggr_actions = [i for i in [2, 3, 4] if mask[i] == 1]
                if aggr_actions:
                    probs = torch.tensor([bluff_score**(4 - i) for i in aggr_actions])
                    probs /= probs.sum()
                    action = aggr_actions[torch.multinomial(probs, 1).item()]
                    if bluff_score > 0.85:
                        print(f"[Bluff Override] score={bluff_score:.2f}, action={action}")
                    return action, torch.log(masked_probs[action]), value.squeeze()
        except Exception:
            pass

        action_dist = torch.distributions.Categorical(masked_probs)
        action = action_dist.sample()
        return action.item(), action_dist.log_prob(action), value.squeeze()


    def update(self, log_probs, values, rewards):
        # Implement this as in your main script
        pass
