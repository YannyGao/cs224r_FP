import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch import optim
from collections import deque, Counter, defaultdict
from pettingzoo.classic import texas_holdem_no_limit_v6
from treys import Deck, Evaluator, Card
from torch.utils.tensorboard import SummaryWriter
from deception_reward import compute_deception_reward
from poker_score import decode_cards
from strong_oppo import CategoricalMasked, Agent as StrongAgent

# === Constants ===
NUM_PLAYERS = 2
OBSERVATION_SPACE_SIZE = 54
ACTION_SPACE_SIZE = 5
HIDDEN_LAYER_SIZE = 32
evaluator = Evaluator()


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
        m = CategoricalMasked(logits, mask)
        action = m.sample()
        return action.item(), m.log_prob(action), value.squeeze()

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
        loss = -(log_probs * advantages).sum() + F.mse_loss(values, returns) - 0.01 * entropy
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class WeakOpponent:
    def get_action(self, obs, mask):
        valid = [i for i, m in enumerate(mask) if m == 1]
        return (random.choice(valid) if valid else 0), torch.tensor(0.0), torch.tensor(0.0), 0


class MediumOpponent:
    def __init__(self, base_agent):
        self.agent = base_agent

    def get_action(self, obs, mask):
        return self.agent.get_action(obs, mask)


class StrongOpponent:
    def __init__(self, base_agent):
        self.agent = base_agent

    def get_action(self, obs, mask):
        action, log_prob, _ = self.agent.get_action(obs, mask)
        return action, log_prob, torch.tensor(0.0), 0


def estimate_bluff_score(hole_cards, community_cards, num_simulations=20):
    wins = 0
    for _ in range(num_simulations):
        deck = Deck()
        used = set(hole_cards + community_cards)
        for c in used:
            deck.cards.remove(c)
        board = community_cards + deck.draw(5 - len(community_cards))
        opponent = deck.draw(2)
        if evaluator.evaluate(board, hole_cards) < evaluator.evaluate(board, opponent):
            wins += 1
    return wins / num_simulations


# === Curriculum Logic ===
trained_medium_agent = BaselineAgent()
trained_medium_agent.policy.load_state_dict(torch.load("medium_agent_2.pth"))
trained_medium_agent.policy.eval()

trained_strong_agent = StrongOpponent(StrongAgent(obs_dim=OBSERVATION_SPACE_SIZE, act_dim=ACTION_SPACE_SIZE))
used_strong_opponent = False

def adaptive_opponent_selection(ep, win_loss_stats):
    global used_strong_opponent

    if used_strong_opponent:
        return trained_strong_agent

    if win_loss_stats["WeakOpponent"]["games"] < 200:
        return WeakOpponent()

    weak_wr = win_loss_stats["WeakOpponent"]["wins"] / max(1, win_loss_stats["WeakOpponent"]["games"])
    medium_wr = win_loss_stats["MediumOpponent"]["wins"] / max(1, win_loss_stats["MediumOpponent"]["games"])

    if weak_wr > 0.7 and win_loss_stats["MediumOpponent"]["games"] < 200:
        return MediumOpponent(trained_medium_agent)

    if medium_wr > 0.65 and win_loss_stats["StrongOpponent"]["games"] < 200:
        used_strong_opponent = True
        return trained_strong_agent

    if medium_wr > 0.6:
        return MediumOpponent(trained_medium_agent)

    return WeakOpponent()


def train_bluffing_baseline(episodes=10000):
    env = texas_holdem_no_limit_v6.env(render_mode="ansi", num_players=NUM_PLAYERS)
    agent = BaselineAgent()
    writer = SummaryWriter("runs/with_tiers_no_modeling")

    win_loss_stats = defaultdict(lambda: {"wins": 0, "losses": 0, "games": 0})
    total_bluff_attempts, total_successful_bluffs, all_bluff_scores = 0, 0, []

    for ep in range(1, episodes + 1):
        opponent = adaptive_opponent_selection(ep, win_loss_stats)
        opponent_type = type(opponent).__name__
        env.reset()

        log_probs, values, rewards, deception_rewards = [], [], [], []
        bluff_scores, actions = [], []
        cumulative_reward = {"player_0": 0, "player_1": 0}

        for name in env.agent_iter():
            obs, rew, term, trunc, _ = env.last()
            mask, state = obs["action_mask"], obs["observation"]
            cumulative_reward[name] += rew
            if term or trunc:
                env.step(None)
                continue

            if name == "player_0":
                action, log_prob, value = agent.get_action(state, mask)
                hole_cards, community_cards = decode_cards(state)
                bluff_score = estimate_bluff_score(hole_cards, community_cards)
                bluff_scores.append(bluff_score)
                deception_rewards.append(compute_deception_reward(bluff_score, action, bluff_reward=1))
                log_probs.append(log_prob)
                values.append(value)
                actions.append(action)
            else:
                action, *_ = opponent.get_action(state, mask)

            env.step(action)
            rewards.append(rew)

        combined_rewards = [r + d for r, d in zip(rewards, deception_rewards)]
        final_reward = cumulative_reward["player_0"]
        won = final_reward > cumulative_reward["player_1"]

        successful_bluff = any(bs > 0.8 and a in [2, 3, 4] for a, bs in zip(actions, bluff_scores))
        if won and successful_bluff:
            combined_rewards[-1] += 0.2 * final_reward
            total_successful_bluffs += 1

        if log_probs:
            agent.update(log_probs, values, combined_rewards)

        total_bluff_attempts += sum(bs > 0.8 and a in [2, 3, 4] for a, bs in zip(actions, bluff_scores))
        all_bluff_scores.extend(bluff_scores)

        # Logging
        writer.add_scalar("Reward/Total", sum(combined_rewards), ep)
        writer.add_scalar("Deception/Bonus", sum(deception_rewards), ep)
        writer.add_scalar("Bluff/Success", successful_bluff, ep)
        writer.add_scalar("Bluff/AverageScore", np.mean(bluff_scores) if bluff_scores else 0.0, ep)
        writer.add_scalar(f"WinRate/{opponent_type}", int(won), ep)
        win_loss_stats[opponent_type]["games"] += 1
        win_loss_stats[opponent_type]["wins" if won else "losses"] += 1

        if ep % 1000 == 0:
            print(f"[Ep {ep}] Reward: {sum(combined_rewards):.2f} | {opponent_type} | Bluffs: {total_successful_bluffs}/{total_bluff_attempts}")
    
    print("\n=== Final Win Rates by Opponent Type ===")
    for opp_name, stats in win_loss_stats.items():
        games = stats["games"]
        wins = stats["wins"]
        if games > 0:
            winrate = wins / games * 100
            print(f"{opp_name}: {wins}W-{stats['losses']}L ({winrate:.2f}% win rate over {games} games)")
        else:
            print(f"{opp_name}: No games played.")

    print("\n=== Experiment Complete ===")
    if total_bluff_attempts:
        print(f"Bluff Success Rate: {total_successful_bluffs / total_bluff_attempts * 100:.2f}%")
    if all_bluff_scores:
        print(f"Average Bluff Score: {np.mean(all_bluff_scores):.3f}")
    writer.close()


if __name__ == "__main__":
    print("Started training with weak→medium→strong opponents, no modeling...")
    train_bluffing_baseline(episodes=50000)
