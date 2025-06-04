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
from strong_oppo import StrongOpponent, CategoricalMasked
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
strong_opponent = StrongOpponent(obs_dim=OBSERVATION_SPACE_SIZE, act_dim=ACTION_SPACE_SIZE)


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
        likely_opp_action = None

        # Add predicted opponent action if available
        if opponent_model is not None and opponent_obs is not None:
            opp_tensor = opponent_obs.unsqueeze(0) if opponent_obs.dim() == 2 else opponent_obs
            opp_pred = torch.softmax(opponent_model(opp_tensor)['action_logits'], dim=-1).squeeze()
            likely_opp_action = torch.argmax(opp_pred).item()
            obs = torch.tensor(obs, dtype=torch.float32)
            obs = torch.cat((obs, torch.tensor([likely_opp_action], dtype=torch.float32)))
        else:
            obs = torch.tensor(obs, dtype=torch.float32)
            obs = torch.cat((obs, torch.tensor([0.0])))

        # Get logits and value
        obs_tensor = obs.unsqueeze(0)
        logits, value = self.policy(obs_tensor)

        # Safe sampling with CategoricalMasked
        m = CategoricalMasked(logits, mask)
        action = m.sample()

        # Optional debug: catch illegal samples
        if not mask[action.item()]:
            print(f"[WARNING] Sampled illegal action: {action.item()}")

        return action.item(), m.log_prob(action), value.squeeze(), likely_opp_action



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
        return action, torch.tensor(0.0), torch.tensor(0.0), 0

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
        return action, log_prob, torch.tensor(0.0)
    
    def update(self, log_probs, rewards):
        self.agent.update(log_probs, rewards)

        #return self.agent.get_action(obs, mask)
        
# === Adaptive Curriculum ===
trained_medium_agent = BaselineAgent()
trained_medium_agent.policy.load_state_dict(torch.load("medium_agent.pth"))
trained_medium_agent.policy.eval() 

strong_inner_agent = StrongAgent(obs_dim=54, act_dim=5)
trained_strong_agent = StrongOpponent(strong_inner_agent)

used_strong_opponent = False

def adaptive_opponent_selection(ep, win_loss_stats):
    global used_strong_opponent

    # if used_strong_opponent:
    #     return trained_strong_agent

    # if win_loss_stats["WeakOpponent"]["games"] < 200:
    #     return WeakOpponent()

    # weak_wr = win_loss_stats["WeakOpponent"]["wins"] / max(1, win_loss_stats["WeakOpponent"]["games"])
    # medium_wr = win_loss_stats["MediumOpponent"]["wins"] / max(1, win_loss_stats["MediumOpponent"]["games"])

    # if weak_wr > 0.7 and win_loss_stats["MediumOpponent"]["games"] < 200:
    #     return MediumOpponent(trained_medium_agent)

    # if medium_wr > 0.65 and win_loss_stats["StrongOpponent"]["games"] < 200:
    used_strong_opponent = True
    return trained_strong_agent

    # if medium_wr > 0.6:
    #return MediumOpponent(trained_medium_agent)

    #return WeakOpponent()

def train_bluffing_baseline(episodes=10000):
    all_bluff_scores = []
    env = texas_holdem_no_limit_v6.env(render_mode="ansi", num_players=NUM_PLAYERS)
    agent = BaselineAgent()
    
    opponent_model = OpponentModel(OBSERVATION_SPACE_SIZE - NUM_PLAYERS + 1, 32, ACTION_SPACE_SIZE)
    tracker = OpponentTracker(opponent_model)
    win_loss_stats = defaultdict(lambda: {"wins": 0, "losses": 0, "games": 0})

    writer = SummaryWriter(log_dir="runs/bluffing_baseline")
    total_bluff_attempts = 0
    total_successful_bluffs = 0

    num_succ_bluffs = 0
    accuracies = []
    for ep in range(1, episodes + 1):
        opponent = adaptive_opponent_selection(ep, win_loss_stats)
       
        env.reset()

        log_probs, values = [], []
        step_rewards, deception_rewards = [], []
        actions_this_game, states_this_game = [], []
        opponent_obs_history = []
        bluff_scores = []
        pred_actions = []
        actual_actions = []
        opponent_log_probs, opponent_rewards = [], []

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

                
                action, log_prob, value, likely_opp_action = agent.get_action(state, mask, opponent_model, opponent_obs_seq)
                pred_actions.append(likely_opp_action)

                # if bluff_score > 0.8 and action in [3, 4]:
                #     rew += 1  # bluffing bonus
                # if action != 0:
                #     rew += 0.1  # aggression reward
                hole_cards, community_cards = decode_cards(state)
                bluff_score = estimate_bluff_score(hole_cards, community_cards)
                step_rewards.append(rew)
                deception_rewards.append(compute_deception_reward(bluff_score=bluff_score, action=action, bluff_reward=1))
                log_probs.append(log_prob)
                values.append(value)
                actions_this_game.append(int(action))
                states_this_game.append(state)
                bluff_scores.append(bluff_score)
                all_bluff_scores.extend(bluff_scores)

            else:
                output = opponent.get_action(state, mask)
                if (len(output) == 3):
                    action, _, _ = output
                    
                else:
                    action, _, _,_ = output
                opponent_obs_history.append(state)

                if name == "player_1":
                    tracker.observe(state, action)
                    if isinstance(opponent, StrongOpponent):
                        _, log_prob, _ = opponent.get_action(state, mask)
                        opponent_log_probs.append(log_prob)
                        opponent_rewards.append(rew)  # or same combined_rewards logic if deception applies

                actual_actions.append(action)

       
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
            # print(f"[Ep {ep}] Successful bluff detected. Extra bonus: +{final_bluff_bonus}")

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
            if isinstance(opponent, StrongOpponent) and opponent_log_probs:
                opponent.update(opponent_log_probs, opponent_rewards)

            pass

        # Opponent model training at end of episode
        batch = tracker.build_batch()
        pred_actions = np.array(pred_actions)
        actual_actions = np.array(actual_actions)

        min_len = min(len(pred_actions), len(actual_actions))
        accuracy = np.mean(np.array(pred_actions[:min_len]) == np.array(actual_actions[:min_len]))
        accuracies.append(accuracy)
        
        
        if batch:
            loss = tracker.train_step(batch)
            if ep % 1000 == 0:
                print(f"Episode {ep}: Opponent model loss = {loss['total_loss']:.4f}")
                print(len(accuracies))
                if len(accuracies) > 0:
                    print(f"mean accuracy {np.mean([a for a in accuracies if not np.isnan(a)])}")                
       
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
            bluff_avg = np.mean(bluff_scores) if bluff_scores else 0.0
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
        bluff_attempts = sum(1 for a, bs in zip(actions_this_game, bluff_scores) if bs > 0.8 and a in [2, 3, 4])
        successful_bluffs = int(successful_bluff)
        percent_successful_bluff = (successful_bluffs / bluff_attempts * 100) if bluff_attempts > 0 else 0.0
        total_bluff_attempts += bluff_attempts
        total_successful_bluffs += successful_bluffs

    if total_bluff_attempts > 0:
        overall_bluff_success_rate = total_successful_bluffs / total_bluff_attempts * 100
        print(f"\n=== Overall Bluff Stats ===")
        print(f"Total Bluff Attempts: {total_bluff_attempts}")
        print(f"Total Successful Bluffs: {total_successful_bluffs}")
        print(f"Overall Bluff Success Rate: {overall_bluff_success_rate:.2f}%")
    else:
        print("\nNo bluff attempts recorded.")

    if all_bluff_scores:
        avg_bluff_score = np.mean(all_bluff_scores)
        print(f"\n=== Bluff Statistics ===")
        print(f"Average Bluff Score Across All Steps: {avg_bluff_score:.3f}")
    else:
        print("\nNo bluff scores recorded.")

    writer.close()
if __name__ == "__main__":
    print("started")
    train_bluffing_baseline(episodes=10000)
 