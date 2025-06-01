import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
from collections import Counter
from pettingzoo.classic import texas_holdem_no_limit_v6

from opponent_model import OpponentModel
from opponent_tracker import OpponentTracker
from poker_score import decode_cards, estimate_bluff_score
from deception_reward import compute_deception_reward

# === Constants ===
OBSERVATION_SPACE_SIZE = 54
ACTION_SPACE_SIZE = 5
HIDDEN_LAYER_SIZE = 32

# === Network Definition ===
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
        loss = policy_loss + value_loss - 0.01 * entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# === Training Loop ===
def train_bluffing_baseline(episodes=10000):
    env = texas_holdem_no_limit_v6.env(render_mode="ansi", num_players=2)
    agent = BaselineAgent()
    opponent = BaselineAgent()
    opponent_model = OpponentModel(OBSERVATION_SPACE_SIZE, 32, ACTION_SPACE_SIZE)
    tracker = OpponentTracker(opponent_model)

    for ep in range(1, episodes + 1):
        env.reset()
        log_probs, values = [], []
        actions_this_game, states_this_game = [], []
        episode_reward = 0

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
                actions_this_game.append(action)
                states_this_game.append(state)
                if action != 0:
                    rew += 0.1
                episode_reward += rew
            else:
                action, _, _ = opponent.get_action(state, mask)

            if name == "player_1":
                tracker.observe(state, action)
            if name == "player_0":
                tracker.train_step()

            env.step(action)

        # === Deception-aware shaping ===
        if log_probs:
            deception_bonus = sum(
                compute_deception_reward(obs=state, action=act, final_reward=episode_reward)
                for state, act in zip(states_this_game, actions_this_game)
            )
            if deception_bonus > 0:
                print(f"[Ep {ep}] Deception Bonus: +{deception_bonus:.2f}")
            episode_reward += deception_bonus
            agent.update(log_probs, values, [episode_reward] * len(log_probs))

        # === Logging ===
        if ep % 1000 == 0:
            action_summary = Counter(a for a in actions_this_game if a != 0)
            if action_summary:
                print(f"Ep {ep}: Reward = {episode_reward:.1f}, Actions = {action_summary}")

if __name__ == "__main__":
    print("Training bluffing agent...")
    train_bluffing_baseline(episodes=10000)
