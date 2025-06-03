# i just copy paste a huge part of the code because i dont wanna change the orirginal one. 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
from collections import deque, Counter
from pettingzoo.classic import texas_holdem_no_limit_v6
from deception_reward import compute_deception_reward
from poker_score import decode_cards, estimate_bluff_score
from collections import Counter, defaultdict
# And these are for opponent modeling
from opponent_tracker_simp import OpponentModel, OpponentTracker
from treys import Deck, Evaluator, Card
import random
from strong_oppo import Agent as StrongAgent


from torch.utils.tensorboard import SummaryWriter

import os

# Constants
NUM_PLAYERS = 2
OBSERVATION_SPACE_SIZE = 54 + NUM_PLAYERS - 1
ACTION_SPACE_SIZE = 5
HIDDEN_LAYER_SIZE = 32
CPU = "cpu"


# PettingZoo action map: Action Index	Meaning
            # 0	Fold
            # 1	Call / Check
            # 2	Raise Half Pot
            # 3	Raise Pot
            # 4	All-In

# Evaluate poker hands
evaluator = Evaluator()


def estimate_bluff_score(hole_cards, community_cards, num_simulations=20):
    wins = 0
    for _ in range(num_simulations):
        deck = Deck()
        used = set(hole_cards + community_cards)
        for c in used:
            deck.cards.remove(c)

        remaining = 5 - len(community_cards)
        board = community_cards + deck.draw(remaining)
        opponent = deck.draw(2)

        my_score = evaluator.evaluate(board, hole_cards)
        opp_score = evaluator.evaluate(board, opponent)
        if my_score < opp_score:
            wins += 1
    return wins / num_simulations

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
        if opponent_model is not None and opponent_obs is not None:
            if opponent_obs.dim() == 2:
                opp_tensor = opponent_obs.unsqueeze(0)  # (1, seq_len, input_dim)
            else:
                opp_tensor = opponent_obs  # already with batch dim
        
        
            opp_pred = torch.softmax(opponent_model(opp_tensor)['action_logits'], dim=-1).squeeze()
            if not isinstance(obs, torch.Tensor):
                obs = torch.tensor(obs, dtype=torch.float32)

            likely_opp_action = torch.argmax(opp_pred)  # tensor scalar
            likely_opp_action = likely_opp_action.unsqueeze(0)  # shape [1]

            obs = torch.cat((obs, likely_opp_action), dim=0)
        else:
            obs = torch.tensor(obs, dtype=torch.float32)  # Ensure it's a torch tensor
            zero = torch.tensor([0.0], dtype=torch.float32)  # 1D tensor with 0
            obs = torch.cat((obs, zero), dim=0)
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

      
        
            # # If opponent likely to fold and we can raise, bluff
            # if likely_opp_action == 0:  # Opponent fold
            #     bluffable_actions = [a for a in [2, 3, 4] if mask[a] == 1]
            #     if bluffable_actions:
            #         probs = torch.tensor([0.4, 0.3, 0.3])[:len(bluffable_actions)]
            #         probs /= probs.sum()
            #         chosen = torch.multinomial(probs, 1).item()
            #         action = bluffable_actions[chosen]
            #         print(f"[Bluff based on opponent folding] Predicted={likely_opp_action} → Bluff action {action}")
            #         return action, torch.log(masked_probs[action]), value.squeeze()

            # # If opponent likely to be aggressive, fold if allowed
            # if likely_opp_action in [2, 3, 4] and mask[0] == 1:
            #     print(f"[Avoid Aggressive Opponent] Predicted={likely_opp_action} → FOLD")
            #     return 0, torch.log(masked_probs[0]), value.squeeze()

   

        # --- Evaluator-based bluff override ---
        # try:
        #     hole_cards, community_cards = decode_cards(obs)
        #     bluff_score = estimate_bluff_score(hole_cards, community_cards)

        #     if bluff_score > 0.8:
        #         aggr_actions = [i for i in [4, 3, 2] if mask[i] == 1]
        #         if aggr_actions:
        #             probs = torch.tensor([bluff_score**(4 - i) for i in aggr_actions])
        #             probs /= probs.sum()
        #             choice = torch.multinomial(probs, 1).item()
        #             action = aggr_actions[choice]
        #             print(f"[Bluff Override] bluff_score={bluff_score:.2f} → sampled aggressive action {action}")
        #             return action, torch.log(masked_probs[action]), value.squeeze()
        # except Exception as e:
        #     print(f"[Bluff Eval Error] {e}")

        # --- Default policy sampling ---
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
trained_medium_agent.policy.load_state_dict(torch.load("medium_agent.pth"))
trained_medium_agent.policy.eval() 

strong_inner_agent = StrongAgent(obs_dim=54, act_dim=5)
trained_strong_agent = StrongOpponent(strong_inner_agent)

def adaptive_opponent_selection(ep, win_loss_stats):
    if win_loss_stats["WeakOpponent"]["games"] < 200:
        return WeakOpponent()

    weak_wr = win_loss_stats["WeakOpponent"]["wins"] / max(1, win_loss_stats["WeakOpponent"]["games"])
    medium_wr = win_loss_stats["MediumOpponent"]["wins"] / max(1, win_loss_stats["MediumOpponent"]["games"])

    if weak_wr > 0.70 and win_loss_stats["MediumOpponent"]["games"] < 200:
        return MediumOpponent(trained_medium_agent)

    if medium_wr > 0.70 and win_loss_stats["StrongOpponent"]["games"] < 200:
        return StrongOpponent(trained_strong_agent)

    if medium_wr > 0.50:
        return MediumOpponent(trained_medium_agent)

    return WeakOpponent()

def train_bluffing_baseline(episodes=10000):
    env = texas_holdem_no_limit_v6.env(render_mode="ansi", num_players=NUM_PLAYERS)
    agent = BaselineAgent()
    
    opponent_model = OpponentModel(OBSERVATION_SPACE_SIZE - NUM_PLAYERS + 1, 32, ACTION_SPACE_SIZE)
    tracker = OpponentTracker(opponent_model)
    win_loss_stats = defaultdict(lambda: {"wins": 0, "losses": 0, "games": 0})

    writer = SummaryWriter(log_dir="runs/bluffing_baseline")
    num_succ_bluffs = 0
    for ep in range(1, episodes + 1):
        opponent = adaptive_opponent_selection(ep, win_loss_stats)
        env.reset()

        log_probs, values = [], []
        step_rewards, deception_rewards = [], []
        actions_this_game, states_this_game = [], []
        opponent_obs_history = []
        bluff_scores = []

        cumulative_reward = {"player_0": 0, "player_1": 0}

        for name in env.agent_iter():
            obs, rew, term, trunc, _ = env.last()
            mask = obs["action_mask"]
            state = obs["observation"]
            cumulative_reward[name] += rew
            
            if term or trunc:
                env.step(None)
                continue
            if name == "player_0":
                # Opponent observation history
                if opponent_obs_history:
                    history_len = 10
                    recent_obs = opponent_obs_history[-history_len:]
                    if len(recent_obs) < history_len:
                        padding = [torch.zeros_like(torch.tensor(recent_obs[0]))] * (history_len - len(recent_obs))
                        recent_obs = padding + recent_obs
                    opponent_obs_seq = torch.stack([torch.tensor(o, dtype=torch.float32) for o in recent_obs])
                else:
                    opponent_obs_seq = None

                
                action, log_prob, value = agent.get_action(state, mask, opponent_model, opponent_obs_seq)

                # if bluff_score > 0.8 and action in [3, 4]:
                #     rew += 1  # bluffing bonus
                # if action != 0:
                #     rew += 0.1  # aggression reward
                hole_cards, community_cards = decode_cards(state)
                bluff_score = estimate_bluff_score(hole_cards, community_cards)
                step_rewards.append(rew)
                deception_rewards.append(compute_deception_reward(bluff_score=bluff_score, action=action, bluff_reward=0.5))
                log_probs.append(log_prob)
                values.append(value)
                actions_this_game.append(action)
                states_this_game.append(state)
                bluff_scores.append(bluff_score)

            else:
                action, _, _ = opponent.get_action(state, mask)
                opponent_obs_history.append(state)

                if name == "player_1":
                    tracker.observe(state, action)

       
            env.step(action)

        # Final training + logging
        
        # Check if final reward was positive (agent won the game)
        total_deception = sum(deception_rewards)
        # if total_deception > 0:
        #         print(f"[Ep {ep}] Deception Bonus: +{total_deception:.2f}")
        combined_rewards = [r + d for r, d in zip(step_rewards, deception_rewards)]
        final_reward = cumulative_reward["player_0"]
        won_game = final_reward > 0

        successful_bluff = False
        for a, bs in zip(actions_this_game, bluff_scores):
            if bs > 0.8 and a in [2, 3, 4]:  # aggressive action
                successful_bluff = True
                num_succ_bluffs += 1
                break
            
        final_bluff_bonus = 0.0
        if won_game and successful_bluff:
            final_bluff_bonus = 0.2*final_reward  # tune this value
            print(f"[Ep {ep}] Successful bluff detected. Extra bonus: +{final_bluff_bonus}")

        # Add final bluff bonus to last step's reward
        if combined_rewards:
            combined_rewards[-1] += final_bluff_bonus

        if log_probs:
            
            agent.update(log_probs, values, combined_rewards)
            # print(combined_rewards[:8])
            # if isinstance(opponent, (MediumOpponent, StrongOpponent)):
            #     detached_log_probs = [lp.detach() for lp in log_probs]
            #     detached_values = [v.detach() for v in values]
            #     detached_rewards = [r for r in combined_rewards]  # rewards are scalars, no need to detach
            #     opponent.agent.update(detached_log_probs, detached_values, detached_rewards)
            pass

        # Opponent model training at end of episode
        batch = tracker.build_batch()
        if batch:
            loss = tracker.train_step(batch)
            if ep % 1000 == 0:
                print(f"Episode {ep}: Opponent model loss = {loss['total_loss']:.4f}")
        tracker.reset()

        # # Optional: log cumulative reward
        # print(f"[Ep {ep}] Final cumulative reward: Player 0: {cumulative_reward['player_0']}, Player 1: {cumulative_reward['player_1']}")


        # === Win/loss tracking using true rewards ===
        opp_name = type(opponent).__name__
        win_loss_stats[opp_name]["games"] += 1
        r0 = cumulative_reward["player_0"]
        r1 = cumulative_reward["player_1"]

        if r0 > r1:
            win_loss_stats[opp_name]["wins"] += 1
        else:
            win_loss_stats[opp_name]["losses"] += 1

        # === TensorBoard Logging ===
        total_reward = sum(step_rewards) + sum(deception_rewards)
        # if total_reward > 0:
        #     print(ep)
      
        writer.add_scalar("Reward/Total", total_reward, ep)
        writer.add_scalar("Deception/Bonus", total_deception, ep)
        writer.add_scalar("Successful Bluff",successful_bluff , ep)
        if states_this_game:
            bluff_avg = np.mean(bluff_score)
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
    print("started")
    train_bluffing_baseline(episodes=50000)
 
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