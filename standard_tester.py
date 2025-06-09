#!/usr/bin/env python3
"""
Evaluation script for trained poker agent against strong opponent.
Loads checkpoints and runs evaluation for 1000 episodes with fixed bluff reward of 2.0.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
from collections import deque, Counter, defaultdict
from pettingzoo.classic import texas_holdem_no_limit_v6
from deception_reward import compute_deception_reward
from poker_score import decode_cards, estimate_bluff_score
from opponent_tracker_simp import OpponentModel, OpponentTracker
from treys import Deck, Evaluator, Card
import random
from strong_oppo import Agent as StrongAgent, StrongOpponent, CategoricalMasked
from stats import *
from torch.utils.tensorboard import SummaryWriter
import argparse
import os

# Constants
NUM_PLAYERS = 2
OBSERVATION_SPACE_SIZE = 54 + NUM_PLAYERS - 1 
ACTION_SPACE_SIZE = 5
HIDDEN_LAYER_SIZE = 32
BLUFF_REWARD = 2.0  # Fixed bluff reward for evaluation

# Action constants
FOLD = 0
CALL_CHECK = 1
RAISE_HALF_POT = 2
RAISE_POT = 3
ALL_IN = 4

# Evaluator for poker hands
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

class EvalAgent:
    """Evaluation version of the trained agent"""
    def __init__(self, checkpoint_path, alpha=0.01, gamma=0.99):
        self.gamma = gamma
        self.policy = PolicyWithValue(OBSERVATION_SPACE_SIZE, ACTION_SPACE_SIZE, HIDDEN_LAYER_SIZE)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=alpha)
        
        # Load checkpoint
        if os.path.exists(checkpoint_path):
            self.policy.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
            print(f"Loaded agent checkpoint from {checkpoint_path}")
        else:
            raise FileNotFoundError(f"Agent checkpoint not found: {checkpoint_path}")

    def get_action(self, obs, mask, opponent_model=None, opponent_obs=None):
        obs = torch.tensor(obs, dtype=torch.float32)

        if opponent_model is not None and opponent_obs is not None:
            opp_tensor = opponent_obs.unsqueeze(0) if opponent_obs.dim() == 2 else opponent_obs
            opp_pred = torch.softmax(opponent_model(opp_tensor)['action_logits'], dim=-1).squeeze()
            likely_opp_action = torch.argmax(opp_pred).item()
        else:
            likely_opp_action = 0

        obs = torch.cat((obs, torch.tensor([likely_opp_action], dtype=torch.float32)))
        
        with torch.no_grad():  # No gradients needed during evaluation
            logits, value = self.policy(obs.unsqueeze(0))
        
        m = CategoricalMasked(logits, mask)
        action = m.sample()

        return action.item(), m.log_prob(action), value.squeeze(), likely_opp_action

class EvalStrongOpponent:
    """Strong opponent that continues learning during evaluation"""
    def __init__(self, checkpoint_path=None):
        self.agent = StrongOpponent(obs_dim=OBSERVATION_SPACE_SIZE, act_dim=ACTION_SPACE_SIZE)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.agent.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
            print(f"Loaded strong opponent checkpoint from {checkpoint_path}")
        else:
            print("No strong opponent checkpoint provided or found, using default initialization")

    def get_action(self, obs, mask):
        return self.agent.get_action(obs, mask)
    
    def update(self, log_probs, rewards):
        """Allow opponent to continue learning"""
        if hasattr(self.agent, 'update') and log_probs:
            self.agent.update(log_probs, rewards)

def evaluate_agent(agent_checkpoint, opponent_checkpoint=None, episodes=1000):
    """
    Evaluate trained agent against strong opponent for specified episodes.
    """
    print(f"Starting evaluation for {episodes} episodes with bluff reward {BLUFF_REWARD}")
    
    # Initialize environment and agents
    env = texas_holdem_no_limit_v6.env(render_mode="ansi", num_players=NUM_PLAYERS)
    eval_agent = EvalAgent(agent_checkpoint)
    strong_opponent = EvalStrongOpponent(opponent_checkpoint)
    
    # Initialize opponent modeling
    opponent_model = OpponentModel(54, 32, ACTION_SPACE_SIZE)
    tracker = OpponentTracker(opponent_model)
    
    # Initialize statistics tracking
    main_agent_stats = init_stats_counters()
    opponent_stats = init_stats_counters()
    
    # Results tracking
    win_loss_stats = {"wins": 0, "losses": 0, "draws": 0}
    total_rewards = []
    total_deception_rewards = []
    bluff_success_rate = []
    
    # TensorBoard logging
    writer = SummaryWriter(log_dir=f"runs/evaluation_br_{BLUFF_REWARD}")
    
    print("Starting evaluation...")
    
    for ep in range(1, episodes + 1):
        env.reset()
        
        # Episode tracking
        log_probs, values = [], []
        step_rewards, deception_rewards = [], []
        actions_this_game, states_this_game = [], []
        opponent_obs_history = []
        opponent_log_probs, opponent_rewards = [], []
        bluff_scores = []
        
        cumulative_reward = {"player_0": 0, "player_1": 0}
        
        # Track actions by stage for stats
        agent_actions_by_stage = {"preflop": [], "flop": [], "turn": [], "river": []}
        opponent_actions_by_stage = {"preflop": [], "flop": [], "turn": [], "river": []}
        agent_all_actions = []
        opponent_all_actions = []
        
        # Game progression tracking
        went_to_showdown = False
        last_stage_seen = "preflop"
        
        for name in env.agent_iter():
            obs, rew, term, trunc, _ = env.last()
            mask = obs["action_mask"]
            state = obs["observation"]
            cumulative_reward[name] += rew
            
            # Get current stage
            stage = get_stage_from_obs(state)
            if stage == "pre-flop":
                stage = "preflop"
            
            # Track game progression
            if stage in ["flop", "turn", "river"]:
                went_to_flop = True
            last_stage_seen = stage
            
            if term or trunc:
                env.step(None)
                continue
            
            if name == "player_0":  # Main agent
                # Prepare opponent observation history
                if opponent_obs_history:
                    history_len = 10
                    recent_obs = opponent_obs_history[-history_len:]
                    if len(recent_obs) < history_len:
                        padding = [torch.zeros_like(torch.tensor(recent_obs[0]))] * (history_len - len(recent_obs))
                        recent_obs = padding + recent_obs
                    opponent_obs_seq = torch.stack([torch.tensor(o, dtype=torch.float32) for o in recent_obs])
                else:
                    opponent_obs_seq = None
                
                action, log_prob, value, likely_opp_action = eval_agent.get_action(
                    state, mask, opponent_model, opponent_obs_seq
                )
                
                # Calculate bluff score and deception reward
                hole_cards, community_cards = decode_cards(state)
                bluff_score = estimate_bluff_score(hole_cards, community_cards)
                bluff_scores.append(bluff_score)
                
                deception_reward = compute_deception_reward(
                    bluff_score=bluff_score, 
                    action=action, 
                    bluff_reward=BLUFF_REWARD
                )
                
                step_rewards.append(rew)
                deception_rewards.append(deception_reward)
                log_probs.append(log_prob)
                values.append(value)
                actions_this_game.append(int(action))
                states_this_game.append(state)
                
                # Track actions by stage
                if stage in agent_actions_by_stage:
                    agent_actions_by_stage[stage].append(int(action))
                agent_all_actions.append(int(action))
                
            else:  # Strong opponent (player_1)
                mask_clean = [bool(m) for m in mask]
                action, log_prob, _ = strong_opponent.get_action(state, mask_clean)
                
                opponent_obs_history.append(state)
                opponent_log_probs.append(log_prob)
                opponent_rewards.append(rew)
                
                # Track opponent actions by stage
                if stage in opponent_actions_by_stage:
                    opponent_actions_by_stage[stage].append(int(action))
                opponent_all_actions.append(int(action))
            
            # Track for opponent modeling
            player = 0 if name == "player_0" else 1
            tracker.observe(state, action, player)
            
            if mask[action] == 0:
                print(f"[ILLEGAL ACTION] Player {name} attempted action {action} with mask {mask}")
            
            env.step(action)
        
        # === End of episode processing ===
        
        # Detect showdown
        went_to_showdown = (last_stage_seen == "river" and 
                           (not agent_all_actions or agent_all_actions[-1] != FOLD) and
                           (not opponent_all_actions or opponent_all_actions[-1] != FOLD))
        
        # Update statistics
        update_stats_on_hand_corrected(
            stats=main_agent_stats,
            preflop_actions=agent_actions_by_stage["preflop"],
            actions=agent_all_actions,
            went_to_showdown=went_to_showdown
        )
        
        update_stats_on_hand_corrected(
            stats=opponent_stats,
            preflop_actions=opponent_actions_by_stage["preflop"],
            actions=opponent_all_actions,
            went_to_showdown=went_to_showdown
        )
        
        # Update opponent model
        batch = tracker.build_batch()
        if batch:
            loss = tracker.train_step(batch)
        tracker.reset()
        
        # Allow strong opponent to continue learning
        if opponent_log_probs:
            strong_opponent.update(opponent_log_probs, opponent_rewards)
        
        # Calculate episode results
        agent_reward = cumulative_reward["player_0"]
        opponent_reward = cumulative_reward["player_1"]
        
        if agent_reward > opponent_reward:
            win_loss_stats["wins"] += 1
        elif agent_reward < opponent_reward:
            win_loss_stats["losses"] += 1
        else:
            win_loss_stats["draws"] += 1
        
        # Track rewards and bluff performance
        total_episode_reward = sum(step_rewards) + sum(deception_rewards)
        total_rewards.append(total_episode_reward)
        total_deception_rewards.append(sum(deception_rewards))
        
        # Calculate bluff success
        bluff_attempts = sum(1 for a, bs in zip(actions_this_game, bluff_scores) 
                           if bs > 0.8 and a in [RAISE_HALF_POT, RAISE_POT, ALL_IN])
        successful_bluff = any(bs > 0.8 and a in [RAISE_HALF_POT, RAISE_POT, ALL_IN] 
                              for a, bs in zip(actions_this_game, bluff_scores)) and agent_reward > 0
        
        bluff_success_rate.append(1.0 if successful_bluff and bluff_attempts > 0 else 0.0)
        
        # Logging to TensorBoard
        writer.add_scalar("Eval/Episode_Reward", total_episode_reward, ep)
        writer.add_scalar("Eval/Agent_Game_Reward", agent_reward, ep)
        writer.add_scalar("Eval/Deception_Reward", sum(deception_rewards), ep)
        writer.add_scalar("Eval/Bluff_Success", successful_bluff, ep)
        
        # Compute and log stats
        if ep % 100 == 0:  # Every 100 episodes
            main_agent_agg_stats = compute_stats(main_agent_stats)
            opponent_agg_stats = compute_stats(opponent_stats)
            
            # Log poker statistics
            writer.add_scalar("Eval/Agent_VPIP", main_agent_agg_stats["VPIP"], ep)
            writer.add_scalar("Eval/Agent_PFR", main_agent_agg_stats["PFR"], ep)
            writer.add_scalar("Eval/Agent_AFq", main_agent_agg_stats["AFq"], ep)
            writer.add_scalar("Eval/Agent_WTSD", main_agent_agg_stats["WTSD"], ep)
            
            writer.add_scalar("Eval/Opponent_VPIP", opponent_agg_stats["VPIP"], ep)
            writer.add_scalar("Eval/Opponent_PFR", opponent_agg_stats["PFR"], ep)
            writer.add_scalar("Eval/Opponent_AFq", opponent_agg_stats["AFq"], ep)
            writer.add_scalar("Eval/Opponent_WTSD", opponent_agg_stats["WTSD"], ep)
            
            # Win rate
            total_games = sum(win_loss_stats.values())
            win_rate = win_loss_stats["wins"] / total_games if total_games > 0 else 0.0
            writer.add_scalar("Eval/Win_Rate", win_rate, ep)
            
            # Print progress
            avg_reward = np.mean(total_rewards[-100:]) if len(total_rewards) >= 100 else np.mean(total_rewards)
            avg_bluff_success = np.mean(bluff_success_rate[-100:]) if len(bluff_success_rate) >= 100 else np.mean(bluff_success_rate)
            
            print(f"Episode {ep}/{episodes}")
            print(f"  Win Rate: {win_rate:.1%} ({win_loss_stats['wins']}W-{win_loss_stats['losses']}L-{win_loss_stats['draws']}D)")
            print(f"  Avg Reward (last 100): {avg_reward:.2f}")
            print(f"  Avg Bluff Success (last 100): {avg_bluff_success:.1%}")
            print(f"  Agent Stats: VPIP={main_agent_agg_stats['VPIP']:.1f}%, PFR={main_agent_agg_stats['PFR']:.1f}%, AFq={main_agent_agg_stats['AFq']:.1f}%")
            print(f"  Opponent Stats: VPIP={opponent_agg_stats['VPIP']:.1f}%, PFR={opponent_agg_stats['PFR']:.1f}%, AFq={opponent_agg_stats['AFq']:.1f}%")
            print()
    
    # Final results
    total_games = sum(win_loss_stats.values())
    final_win_rate = win_loss_stats["wins"] / total_games if total_games > 0 else 0.0
    final_avg_reward = np.mean(total_rewards)
    final_avg_deception = np.mean(total_deception_rewards)
    final_bluff_success = np.mean(bluff_success_rate)
    
    print("=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Episodes: {episodes}")
    print(f"Final Win Rate: {final_win_rate:.1%} ({win_loss_stats['wins']}W-{win_loss_stats['losses']}L-{win_loss_stats['draws']}D)")
    print(f"Average Episode Reward: {final_avg_reward:.2f}")
    print(f"Average Deception Reward: {final_avg_deception:.2f}")
    print(f"Bluff Success Rate: {final_bluff_success:.1%}")
    
    # Compute final aggregate stats
    final_agent_stats = compute_stats(main_agent_stats)
    final_opponent_stats = compute_stats(opponent_stats)
    
    print(f"\nFinal Agent Stats:")
    print(f"  VPIP: {final_agent_stats['VPIP']:.1f}%")
    print(f"  PFR: {final_agent_stats['PFR']:.1f}%")
    print(f"  AFq: {final_agent_stats['AFq']:.1f}%")
    print(f"  WTSD: {final_agent_stats['WTSD']:.1f}%")
    
    print(f"\nFinal Opponent Stats:")
    print(f"  VPIP: {final_opponent_stats['VPIP']:.1f}%")
    print(f"  PFR: {final_opponent_stats['PFR']:.1f}%")
    print(f"  AFq: {final_opponent_stats['AFq']:.1f}%")
    print(f"  WTSD: {final_opponent_stats['WTSD']:.1f}%")
    
    writer.close()
    return {
        "win_rate": final_win_rate,
        "avg_reward": final_avg_reward,
        "avg_deception_reward": final_avg_deception,
        "bluff_success_rate": final_bluff_success,
        "agent_stats": final_agent_stats,
        "opponent_stats": final_opponent_stats,
        "win_loss_stats": win_loss_stats
    }

def update_stats_on_hand_corrected(stats, preflop_actions, actions, went_to_showdown=False):
    """Update statistics for a completed hand"""
    stats["hands_dealt"] += 1

    # VPIP: hands played preflop
    if any(a != FOLD for a in preflop_actions):
        stats["hands_played_preflop"] += 1

    # PFR: hands raised preflop
    if any(a in [RAISE_HALF_POT, RAISE_POT, ALL_IN] for a in preflop_actions):
        stats["hands_raised_preflop"] += 1

    # AFq: aggressive action frequency
    for a in actions:
        stats["total_actions"] += 1
        if a in [RAISE_HALF_POT, RAISE_POT, ALL_IN]:
            stats["aggressive_actions"] += 1
        elif a in [FOLD, CALL_CHECK]:
            stats["called_or_folded_actions"] += 1

    # WTSD: went to showdown
    if len(actions) > len(preflop_actions) or went_to_showdown:
        stats["hands_went_to_flop"] += 1
        
        if went_to_showdown:
            stats["hands_went_to_showdown"] += 1

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate trained poker agent against strong opponent')
    parser.add_argument('--agent_checkpoint', type=str, required=True,
                       help='Path to the trained agent checkpoint (.pt file)')
    parser.add_argument('--opponent_checkpoint', type=str, default=None,
                       help='Path to the strong opponent checkpoint (.pt file)')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of episodes to evaluate (default: 1000)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    print(f"Agent checkpoint: {args.agent_checkpoint}")
    print(f"Opponent checkpoint: {args.opponent_checkpoint}")
    print(f"Episodes: {args.episodes}")
    print(f"Fixed bluff reward: {BLUFF_REWARD}")
    
    results = evaluate_agent(
        agent_checkpoint=args.agent_checkpoint,
        opponent_checkpoint=args.opponent_checkpoint,
        episodes=args.episodes
    )
    
    print("\nEvaluation completed successfully!")
    
    
# python eval_script.py --agent_checkpoint path/to/main_agent_ep1000.pt --opponent_checkpoint path/to/opponent_ep1000.pt --episodes 2000