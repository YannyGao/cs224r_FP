# i just copy paste a huge part of the code because i dont wanna change the orirginal one. 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
from collections import deque, Counter
from pettingzoo.classic import texas_holdem_no_limit_v6
# And these are for opponent modeling
from opponent_model import OpponentModel
from opponent_tracker import OpponentTracker

# Constants
OBSERVATION_SPACE_SIZE = 54
ACTION_SPACE_SIZE = 5
HIDDEN_LAYER_SIZE = 32
CPU = "cpu"


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
    def __init__(self, alpha=0.01, gamma=0.99, opponent_model=None):
        self.gamma = gamma
        self.opponent_model = opponent_model
        self.policy = PolicyWithValue(OBSERVATION_SPACE_SIZE, ACTION_SPACE_SIZE, HIDDEN_LAYER_SIZE)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=alpha)

    def get_action(self, obs, mask, opponent_model=None, opponent_obs=None):
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

        # predict opponent's likely action
        if opponent_model and opponent_obs is not None:
            opponent_obs_tensor = torch.tensor(opponent_obs, dtype=torch.float32).unsqueeze(0)
            opponent_pred = opponent_model(opponent_obs_tensor).squeeze()
            likely_action = torch.argmax(opponent_pred).item()
            
            #  if opponent is likely to fold (say action 0), consider bluffing - idk but this is part of the strategy that we should experiment? 
            if likely_action == 0 and mask[2] == 1:
                return 2, torch.log(masked_probs[2]), value.squeeze()

        action_dist = torch.distributions.Categorical(masked_probs)
        action = action_dist.sample()
        return action.item(), action_dist.log_prob(action), value.squeeze()

    def update(self, log_probs, values, rewards):
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)

        values = torch.stack(values)
        log_probs = torch.stack(log_probs)

        advantages = returns - values.detach()
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum()

        policy_loss = -(log_probs * advantages).sum()
        value_loss = F.mse_loss(values, returns)
        total_loss = policy_loss + value_loss - 0.01 * entropy

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

def train_bluffing_baseline(episodes=10000):
    env = texas_holdem_no_limit_v6.env(render_mode="ansi", num_players=2)
    agent = BaselineAgent()
    opponent = BaselineAgent()
    opponent_model = OpponentModel(OBSERVATION_SPACE_SIZE, 32, ACTION_SPACE_SIZE)
    tracker = OpponentTracker(opponent_model)

    for ep in range(1, episodes + 1):
        env.reset()
        log_probs, rewards, values = [], [], []
        episode_reward = 0
        actions_this_game = []

        for name in env.agent_iter():
            obs, rew, term, trunc, _ = env.last()
            if term or trunc:
                action = None
            else:
                mask = obs["action_mask"]
                state = obs["observation"]
                #print( mask)
                if name == "player_0":
                    opponent_state = obs["observation"]
                    action, log_prob, value = agent.get_action(state, mask, opponent_model, opponent_state)
                    log_probs.append(log_prob)
                    values.append(value)
                    actions_this_game.append(action)
                    # So like encourage aggression... folding...
                    if action != 0:
                        rew += 0.1
                else:
                    action, _, _ = opponent.get_action(state, mask)
                if name == "player_1":
                    tracker.observe(state, action)
                if name == "player_0":
                    tracker.train_step()  #only update on our own turn to decouple training

            env.step(action)
            loss = tracker.train_step()
            if loss and ep % 1000 == 0:
                print(f"Episode {ep}: Opponent model loss = {loss:.4f}")

            if name == "player_0":
                episode_reward += rew

        if log_probs:
            final_rewards = [episode_reward] * len(log_probs)
            agent.update(log_probs, values, final_rewards)

        if ep % 1000 == 0:
            print(f"Episode {ep}: Final Reward = {episode_reward}, Actions: {Counter(actions_this_game)}")

if __name__ == "__main__":
    print("started")
    train_bluffing_baseline(episodes=10000)
 
'''
Result is like this:
Episode 1000: Final Reward = 1, Actions: Counter()
Episode 2000: Final Reward = 100.3, Actions: Counter({1: 3})
Episode 3000: Final Reward = 1, Actions: Counter()
Episode 4000: Final Reward = 1, Actions: Counter()
Episode 5000: Final Reward = 2.1, Actions: Counter({3: 1})
Episode 6000: Final Reward = 100.1, Actions: Counter({1: 1})
Episode 7000: Final Reward = -2, Actions: Counter({0: 1})
Episode 8000: Final Reward = 4.1, Actions: Counter({4: 1})
Episode 9000: Final Reward = 1, Actions: Counter()
Episode 10000: Final Reward = 100.1, Actions: Counter({4: 1})
'''