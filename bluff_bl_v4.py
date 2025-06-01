import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
from collections import Counter, defaultdict
from pettingzoo.classic import texas_holdem_no_limit_v6
from torch.utils.tensorboard import SummaryWriter
import os

from opponent_model import OpponentModel
from opponent_tracker import OpponentTracker
from poker_score import decode_cards, estimate_bluff_score
from deception_reward import compute_deception_reward

# === Constants ===
OBSERVATION_SPACE_SIZE = 54 + 1  # Adding the bluff score
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

# === Agent Definition ===
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

# === Opponent Definitions ===
class WeakOpponent:
    def get_action(self, obs, mask):
        valid_actions = [i for i, m in enumerate(mask) if m == 1]
        action = random.choice(valid_actions) if valid_actions else 0
        return action, torch.tensor(0.0), torch.tensor(0.0)

class MediumOpponent:
    def __init__(self, base_agent):
        self.agent = base_agent

    def get_action(self, obs, mask):
        return self.agent.get_action(obs, mask)

class StrongOpponent:
    def __init__(self, base_agent):
        self.agent = base_agent

    def get_action(self, obs, mask):
        return self.agent.get_action(obs, mask)

# === Adaptive Curriculum ===
trained_medium_agent = BaselineAgent()
trained_strong_agent = BaselineAgent()

def adaptive_opponent_selection(ep, win_loss_stats):
    if win_loss_stats["WeakOpponent"]["games"] < 200:
        return WeakOpponent()

    weak_wr = win_loss_stats["WeakOpponent"]["wins"] / max(1, win_loss_stats["WeakOpponent"]["games"])
    medium_wr = win_loss_stats["MediumOpponent"]["wins"] / max(1, win_loss_stats["MediumOpponent"]["games"])

    if weak_wr > 0.75 and win_loss_stats["MediumOpponent"]["games"] < 200:
        return MediumOpponent(trained_medium_agent)

    if medium_wr > 0.72 and win_loss_stats["StrongOpponent"]["games"] < 200:
        return StrongOpponent(trained_strong_agent)

    if medium_wr > 0.60:
        return MediumOpponent(trained_medium_agent)

    return WeakOpponent()

# === Training Loop ===
def train_bluffing_baseline(episodes=10000):
    print("[TensorBoard] Logging to:", os.path.abspath("runs/bluffing_baseline"))

    env = texas_holdem_no_limit_v6.env(render_mode="ansi", num_players=2)
    agent = BaselineAgent()
    opponent_model = OpponentModel(OBSERVATION_SPACE_SIZE, 32, ACTION_SPACE_SIZE)
    tracker = OpponentTracker(opponent_model)
    win_loss_stats = defaultdict(lambda: {"wins": 0, "losses": 0, "games": 0})
    writer = SummaryWriter(log_dir="runs/bluffing_baseline")

    for ep in range(1, episodes + 1):
        opponent = adaptive_opponent_selection(ep, win_loss_stats)
        env.reset()
        log_probs, values = [], []
        actions_this_game, states_this_game = [], []
        step_rewards, deception_rewards = [], []

        for name in env.agent_iter():
            obs, rew, term, trunc, _ = env.last()
            if term or trunc:
                env.step(None)
                continue

            mask = obs["action_mask"]
            state = obs["observation"]
            hole_cards, community_cards = decode_cards(state)
            bluff_score = estimate_bluff_score(hole_cards, community_cards)
            state_with_bluff = np.concatenate([state, [bluff_score]])

            if name == "player_0":
                action, log_prob, value = agent.get_action(state_with_bluff, mask)
                log_probs.append(log_prob)
                values.append(value)
                actions_this_game.append(action)
                states_this_game.append(state_with_bluff)

                if bluff_score < 0.25 and action in [3, 4]:
                    rew += 0.2

                step_rewards.append(rew)
                deception_rewards.append(
                    compute_deception_reward(obs=state_with_bluff, action=action, final_reward=rew)
                )
            else:
                action, _, _ = opponent.get_action(state_with_bluff, mask)

            if name == "player_1":
                tracker.observe(state_with_bluff, action)
            if name == "player_0":
                tracker.train_step()

            env.step(action)

        # === Deception-aware shaping + Update ===
        if log_probs:
            total_deception = sum(deception_rewards)
            if total_deception > 0:
                print(f"[Ep {ep}] Deception Bonus: +{total_deception:.2f}")
            combined_rewards = [r + d for r, d in zip(step_rewards, deception_rewards)]
            agent.update(log_probs, values, combined_rewards)
            if isinstance(opponent, MediumOpponent):
                opponent.agent.update(log_probs, values, combined_rewards)

            if isinstance(opponent, StrongOpponent):
                opponent.agent.update(log_probs, values, combined_rewards)


        # === Win/loss tracking using true rewards ===
        opp_name = type(opponent).__name__
        win_loss_stats[opp_name]["games"] += 1
        final_rewards = env.rewards
        r0 = final_rewards.get("player_0", 0)
        r1 = final_rewards.get("player_1", 0)
        if r0 > r1:
            win_loss_stats[opp_name]["wins"] += 1
        else:
            win_loss_stats[opp_name]["losses"] += 1

        # === TensorBoard Logging ===
        total_reward = sum(step_rewards) + sum(deception_rewards)
        writer.add_scalar("Reward/Total", total_reward, ep)
        writer.add_scalar("Deception/Bonus", total_deception, ep)
        if states_this_game:
            bluff_avg = np.mean([s[-1] for s in states_this_game])
            writer.add_scalar("Bluff/AverageScore", bluff_avg, ep)
        for name, stats in win_loss_stats.items():
            if stats["games"] > 0:
                winrate = stats["wins"] / stats["games"]
                writer.add_scalar(f"WinRate/{name}", winrate, ep)

        # === Console Logging ===
        if ep % 1000 == 0:
            action_summary = Counter(a for a in actions_this_game if a != 0)
            print(f"Ep {ep}: Reward = {total_reward:.1f}, Actions = {action_summary}")
            print(f"[Stats @ Ep {ep}]")
            for opp, stats in win_loss_stats.items():
                w, l, g = stats["wins"], stats["losses"], stats["games"]
                winrate = w / g if g > 0 else 0.0
                print(f"{opp}: {w}W-{l}L ({winrate:.2%} win rate over {g} games)")

    writer.close()

if __name__ == "__main__":
    print("Training bluffing agent with adaptive curriculum...")
    train_bluffing_baseline(episodes=10000)
