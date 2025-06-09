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
from opponent_tracker_simp import OpponentModel, OpponentTracker
# And these are for opponent modeling

from treys import Deck, Evaluator, Card
import random
from strong_oppo import Agent as StrongAgent

from stats import *


from torch.utils.tensorboard import SummaryWriter
from strong_oppo import StrongOpponent, CategoricalMasked
import os

# Constants
NUM_PLAYERS = 2
OBSERVATION_SPACE_SIZE = 54 
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

from sentence_transformers import SentenceTransformer

feedback_encoder = SentenceTransformer("all-MiniLM-L6-v2")  # Or another




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
        
        self.policy = PolicyWithValue(OBSERVATION_SPACE_SIZE, ACTION_SPACE_SIZE, HIDDEN_LAYER_SIZE)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=alpha)

    def get_action(self, obs, mask, opponent_model=None, opponent_obs=None, feedback_embedding=None):
        obs = torch.tensor(obs, dtype=torch.float32)


        logits, value = self.policy(obs.unsqueeze(0))
        
        m = CategoricalMasked(logits, mask)
        action = m.sample()

        return action.item(), m.log_prob(action), value.squeeze(), 0



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
        
class WeakOpponent:
    def get_action(self, obs, mask):
        valid_actions = [i for i, m in enumerate(mask) if m == 1]
        action = random.choice(valid_actions) if valid_actions else 0
        return action, torch.tensor(0.0), torch.tensor(0.0), 0

class MediumOpponent:
    def __init__(self, base_agent):
        self.agent = base_agent

    def get_action(self, obs, mask):
        action, log_prob, _ = self.agent.get_action(obs, mask)
        return action, log_prob, torch.tensor(0.0), 0
    
    

class StrongOpponent:
    def __init__(self, base_agent):
        self.agent = base_agent

    def get_action(self, obs, mask):
        action, log_prob, _ = self.agent.get_action(obs, mask)
        return action, log_prob, torch.tensor(0.0),0
    
    
    def update(self, log_probs, rewards):
        self.agent.update(log_probs, rewards)


# === Adaptive Curriculum ===
trained_medium_agent = BaselineAgent()
trained_strong_agent = BaselineAgent()

def adaptive_opponent_selection(ep, win_loss_stats):
    if win_loss_stats["WeakOpponent"]["games"] < 200:
        return WeakOpponent()

    weak_wr = win_loss_stats["WeakOpponent"]["wins"] / max(1, win_loss_stats["WeakOpponent"]["games"])
    medium_wr = win_loss_stats["MediumOpponent"]["wins"] / max(1, win_loss_stats["MediumOpponent"]["games"])

    if weak_wr > 0.70 and win_loss_stats["MediumOpponent"]["games"] < 200:
        return MediumOpponent(trained_medium_agent)

    if medium_wr > 0.65 and win_loss_stats["StrongOpponent"]["games"] < 200:
        return StrongOpponent(trained_strong_agent)

    if medium_wr > 0.60:
        return MediumOpponent(trained_medium_agent)

    return WeakOpponent()



def train_bluffing_baseline(episodes=10000):
    from stats import (
    tokenize_feedback,
    init_stats_counters,
    update_stats_on_hand,
    compute_stats,
    extract_stage_action_counts,
    # add others if needed
)

def train_bluffing_baseline(episodes=10000, bluff_reward=2):
    main_agent_stats = init_stats_counters()
    opponent_stats = init_stats_counters()
    previous_feedback, stats_before, stats_after = None, None, None
    
   
   

    all_bluff_scores = []
    env = texas_holdem_no_limit_v6.env(render_mode="ansi", num_players=NUM_PLAYERS)
    agent = BaselineAgent()
  
    
    
    
    win_loss_stats = defaultdict(lambda: {"wins": 0, "losses": 0, "games": 0})
    writer = SummaryWriter(log_dir="runs/agent_wo_opp")

    total_bluff_attempts = 0
    total_successful_bluffs = 0
    num_succ_bluffs = 0
    accuracies = []
    accuracies_by_type = defaultdict(list)
    opponent_models = {
        "WeakOpponent": OpponentModel(54, 32, ACTION_SPACE_SIZE),
        "StrongOpponent": OpponentModel(54, 32, ACTION_SPACE_SIZE),
        "MediumOpponent": OpponentModel(54, 32, ACTION_SPACE_SIZE)
    }
    stats_counters_by_opponent = {
        opp_type: init_stats_counters() for opp_type in opponent_models.keys()
    }

    for ep in range(1, episodes + 1):
        print(f"ok {ep}")
       
        
        opponent = adaptive_opponent_selection(ep, win_loss_stats)
        opponent_type = type(opponent).__name__
        
       
        
       
        current_stats = stats_counters_by_opponent[opponent_type]
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

        # Track actions by stage and player for proper stats calculation
        agent_actions_by_stage = {"preflop": [], "flop": [], "turn": [], "river": []}
        opponent_actions_by_stage = {"preflop": [], "flop": [], "turn": [], "river": []}
        
        # Track all actions for each player (for AFq calculation)
        agent_all_actions = []
        opponent_all_actions = []
        
        # Track game progression
        went_to_flop = False
        went_to_showdown = False
        last_stage_seen = "preflop"

        for name in env.agent_iter():
            obs, rew, term, trunc, _ = env.last()
            mask = obs["action_mask"]
            state = obs["observation"]
            cumulative_reward[name] += rew
            
            # Get current stage with consistent naming
            stage = get_stage_from_obs(state)
            if stage == "pre-flop":  # Fix inconsistent naming
                stage = "preflop"
            
            # Track game progression
            if stage in ["flop", "turn", "river"]:
                went_to_flop = True
            last_stage_seen = stage
            
            if term or trunc:
                env.step(None)
                continue
            
            if name == "player_0":  # Main agent
                # Opponent observation history for model
                
                
                action, log_prob, value, likely_opp_action = agent.get_action(
                    state, mask
                )
                pred_actions.append(likely_opp_action)

                hole_cards, community_cards = decode_cards(state)
                bluff_score = estimate_bluff_score(hole_cards, community_cards)
                bluff_scores.append(bluff_score)
                all_bluff_scores.extend(bluff_scores)

                step_rewards.append(rew)
                deception_rewards.append(compute_deception_reward(bluff_score=bluff_score, action=action, bluff_reward=bluff_reward))
                log_probs.append(log_prob)
                values.append(value)
                actions_this_game.append(int(action))
                states_this_game.append(state)

                # Track actions by stage for stats
                if stage in agent_actions_by_stage:
                    agent_actions_by_stage[stage].append(int(action))
                agent_all_actions.append(int(action))

            else:  # Opponent (player_1)
                mask_clean = [bool(m) for m in mask]
                action, _, _, *_ = opponent.get_action(state, mask_clean)
                opponent_obs_history.append(state)
                
                # Track opponent actions by stage for stats
                if stage in opponent_actions_by_stage:
                    opponent_actions_by_stage[stage].append(int(action))
                opponent_all_actions.append(int(action))

            player = 0 if name == "player_0" else 1
           
            
            if player == 1: 
                if isinstance(opponent, StrongOpponent):
                    _, log_prob, _ = opponent.get_action(state, mask)
                    opponent_log_probs.append(log_prob)
                    opponent_rewards.append(rew)

                actual_actions.append(action)

            if mask[action] == 0:
                print(f"[ILLEGAL] action {action} with mask {mask}, {opponent_type}, {name}")
            
            env.step(action)

        # === After episode ends ===
        
        # Detect if hand went to showdown (reached river and didn't fold)
        went_to_showdown = (last_stage_seen == "river" and 
                           (not agent_all_actions or agent_all_actions[-1] != FOLD) and
                           (not opponent_all_actions or opponent_all_actions[-1] != FOLD))
        
        
        # Update stats for main agent
        update_stats_on_hand_corrected(
            stats=main_agent_stats,
            preflop_actions=agent_actions_by_stage["preflop"],
            actions=agent_all_actions,
            went_to_showdown=went_to_showdown
        )
        
        # Update stats for opponent
        update_stats_on_hand_corrected(
            stats=opponent_stats,
            preflop_actions=opponent_actions_by_stage["preflop"], 
            actions=opponent_all_actions,
            went_to_showdown=went_to_showdown
        )
        
        # Update opponent-specific stats
        update_stats_on_hand_corrected(
            stats=current_stats,
            preflop_actions=opponent_actions_by_stage["preflop"],
            actions=opponent_all_actions,
            went_to_showdown=went_to_showdown
        )
        if ep != 1:
            stats_before = main_agent_agg_stats
        
        # Compute aggregated stats for feedback
        
        main_agent_agg_stats = compute_stats(main_agent_stats)
        print(main_agent_agg_stats)
        opponent_agg_stats = compute_stats(opponent_stats)
       
        stats_after = main_agent_agg_stats
        total_deception = sum(deception_rewards)
        combined_rewards = [r + d for r, d in zip(step_rewards, deception_rewards)]
        final_reward = cumulative_reward["player_0"]
        won_game = final_reward > 0
        
        

        successful_bluff = any(bs > 0.8 and a in [2, 3, 4] for a, bs in zip(actions_this_game, bluff_scores))
        if won_game and successful_bluff:
            final_bluff_bonus = 0.2 * final_reward
            if combined_rewards:
                combined_rewards[-1] += final_bluff_bonus

        if log_probs:
            agent.update(log_probs, values, combined_rewards)

        
        

        # Update win/loss stats
        win_loss_stats[opponent_type]["games"] += 1
        r0, r1 = cumulative_reward["player_0"], cumulative_reward["player_1"]
        if r0 > r1:
            win_loss_stats[opponent_type]["wins"] += 1
        else:
            win_loss_stats[opponent_type]["losses"] += 1

        # TensorBoard logging
        total_reward = sum(step_rewards) + sum(deception_rewards)
        writer.add_scalar("Reward/Total", total_reward, ep)
        writer.add_scalar("Deception/Bonus", total_deception, ep)
        writer.add_scalar("Successful Bluff", successful_bluff, ep)

        if states_this_game:
            bluff_avg = np.mean(bluff_scores) if bluff_scores else 0.0
            writer.add_scalar("Bluff/AverageScore", bluff_avg, ep)

        for name, stats in win_loss_stats.items():
            if stats["games"] > 0:
                winrate = stats["wins"] / stats["games"]
                writer.add_scalar(f"WinRate/{name}", winrate, ep)

        # Log poker stats to TensorBoard
        writer.add_scalar("Stats/Agent_VPIP", main_agent_agg_stats["VPIP"], ep)
        writer.add_scalar("Stats/Agent_PFR", main_agent_agg_stats["PFR"], ep)
        writer.add_scalar("Stats/Agent_AFq", main_agent_agg_stats["AFq"], ep)
        writer.add_scalar("Stats/Agent_WTSD", main_agent_agg_stats["WTSD"], ep)
        # if ep != 1 and adherence_score:
        #     writer.add_scalar("Rewards/Adherence_Awards", adherence_score, ep)
        
        
        writer.add_scalar("Stats/Opponent_VPIP", opponent_agg_stats["VPIP"], ep)
        writer.add_scalar("Stats/Opponent_PFR", opponent_agg_stats["PFR"], ep)
        writer.add_scalar("Stats/Opponent_AFq", opponent_agg_stats["AFq"], ep)
        writer.add_scalar("Stats/Opponent_WTSD", opponent_agg_stats["WTSD"], ep)

        if ep % 1000 == 0:
            action_summary = Counter(a for a in actions_this_game if a != 0)
            print(f"Ep {ep}: Reward = {total_reward:.1f}, Actions = {action_summary}")
            print(f"[Stats @ Ep {ep}]")
            print(f"Agent Stats: VPIP={main_agent_agg_stats['VPIP']:.1f}%, PFR={main_agent_agg_stats['PFR']:.1f}%, AFq={main_agent_agg_stats['AFq']:.1f}%")
            print(f"Opponent Stats: VPIP={opponent_agg_stats['VPIP']:.1f}%, PFR={opponent_agg_stats['PFR']:.1f}%, AFq={opponent_agg_stats['AFq']:.1f}%")
            
            try: 
                os.makedirs(f"agent_wo_rew", exist_ok=True)
                torch.save(agent.policy.state_dict(), f"agent_wo_rew/main_agent_ep{ep}.pt")
                if opponent_type == "StrongAgent":
                    torch.save(opponent.policy.state_dict(), f"agent_wo_rew/opponent_ep{ep}.pt")
            except:
                print("coudn't save")

            for opp, stats in win_loss_stats.items():
                w, l, g = stats["wins"], stats["losses"], stats["games"]
                winrate = w / g if g > 0 else 0.0
                print(f"{opp}: {w}W-{l}L ({winrate:.2%} win rate over {g} games)")

        bluff_attempts = sum(1 for a, bs in zip(actions_this_game, bluff_scores) if bs > 0.8 and a in [2, 3, 4])
        successful_bluffs = int(successful_bluff)
        percent_successful_bluff = (successful_bluffs / bluff_attempts * 100) if bluff_attempts > 0 else 0.0
        total_bluff_attempts += bluff_attempts
        total_successful_bluffs += successful_bluffs

       
        
        
   
    writer.close()

def update_stats_on_hand_corrected(stats, preflop_actions, actions, went_to_showdown=False):
    """
    Corrected version that properly handles the showdown parameter.
    """
    stats["hands_dealt"] += 1

    # VPIP: hands played preflop = count of non-fold preflop actions
    if any(a != FOLD for a in preflop_actions):
        stats["hands_played_preflop"] += 1

    # PFR: hands raised preflop (actions 2,3,4) 
    if any(a in [RAISE_HALF_POT, RAISE_POT, ALL_IN] for a in preflop_actions):
        stats["hands_raised_preflop"] += 1

    # AFq: aggressive action frequency = raise / (call+fold+raise)
    for a in actions:
        stats["total_actions"] += 1
        if a in [RAISE_HALF_POT, RAISE_POT, ALL_IN]:
            stats["aggressive_actions"] += 1
        elif a in [FOLD, CALL_CHECK]:
            stats["called_or_folded_actions"] += 1

    # WTSD: hands went to showdown / hands went to flop
    # Only count as went to flop if we have postflop actions or community cards were dealt
    if len(actions) > len(preflop_actions) or went_to_showdown:
        stats["hands_went_to_flop"] += 1
        
        if went_to_showdown:
            stats["hands_went_to_showdown"] += 1
import argparse 
def parse_args():
    parser = argparse.ArgumentParser(description='Train poker agent with configurable bluff reward')
    parser.add_argument('--br', type=float, default=1.0,
                       help='Reward multiplier for successful bluffs (default: 1.0)')
    return parser.parse_args()

if __name__ == "__main__":
    print("started")
    args = parse_args()
    print(f"Starting training with bluff reward: {args.br}")
    train_bluffing_baseline(episodes=10000, bluff_reward=args.br)
