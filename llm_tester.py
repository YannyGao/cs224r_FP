# Testing script for poker agents - focused on StrongOpponent evaluation
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

from stats import *

from torch.utils.tensorboard import SummaryWriter
from strong_oppo import StrongOpponent, CategoricalMasked
import os

# Constants
NUM_PLAYERS = 2
OBSERVATION_SPACE_SIZE = 54 + NUM_PLAYERS - 1 + 384
ACTION_SPACE_SIZE = 5
HIDDEN_LAYER_SIZE = 32
CPU = "cpu"

# Evaluate poker hands
evaluator = Evaluator()

from sentence_transformers import SentenceTransformer
feedback_encoder = SentenceTransformer("all-MiniLM-L6-v2")

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

    def get_action(self, obs, mask, opponent_model=None, opponent_obs=None, feedback_embedding=None):
        obs = torch.tensor(obs, dtype=torch.float32)

        if opponent_model is not None and opponent_obs is not None:
            opp_tensor = opponent_obs.unsqueeze(0) if opponent_obs.dim() == 2 else opponent_obs
            opp_pred = torch.softmax(opponent_model(opp_tensor)['action_logits'], dim=-1).squeeze()
            likely_opp_action = torch.argmax(opp_pred).item()
        else:
            likely_opp_action = 0

        obs = torch.cat((obs, torch.tensor([likely_opp_action], dtype=torch.float32)))
            
        if feedback_embedding is None:
            feedback_embedding = torch.zeros(384)
        obs = torch.cat((obs, feedback_embedding.float()))
        logits, value = self.policy(obs.unsqueeze(0))
        
        m = CategoricalMasked(logits, mask)
        action = m.sample()

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

    def load_weights(self, checkpoint_path):
        """Load weights from checkpoint"""
        try:
            self.policy.load_state_dict(torch.load(checkpoint_path, map_location=CPU))
            print(f"Successfully loaded main agent weights from {checkpoint_path}")
        except Exception as e:
            print(f"Failed to load main agent weights: {e}")

class StrongOpponent:
    def __init__(self, base_agent):
        self.agent = base_agent

    def get_action(self, obs, mask):
        action, log_prob, value = self.agent.get_action(obs, mask)
        return action, log_prob, value, 0
    
    def update(self, log_probs, values, rewards):
        self.agent.update(log_probs, values, rewards)

    def load_weights(self, checkpoint_path):
        """Load weights from checkpoint"""
        try:
            self.agent.policy.load_state_dict(torch.load(checkpoint_path, map_location=CPU))
            print(f"Successfully loaded strong opponent weights from {checkpoint_path}")
        except Exception as e:
            print(f"Failed to load strong opponent weights: {e}")

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

def test_against_strong_opponent(episodes=1000, 
                                main_agent_checkpoint=None, 
                                strong_opponent_checkpoint=None,
                                continue_training=True):
    """
    Test the main agent against StrongOpponent for specified episodes
    
    Args:
        episodes: Number of episodes to test
        main_agent_checkpoint: Path to main agent checkpoint
        strong_opponent_checkpoint: Path to strong opponent checkpoint
        continue_training: Whether to continue updating the strong opponent during testing
    """
    
    print(f"Starting testing for {episodes} episodes against StrongOpponent")
    print(f"Continue training: {continue_training}")
    
    # Initialize stats
    main_agent_stats = init_stats_counters()
    opponent_stats = init_stats_counters()
    
    # Initialize environment and agents
    env = texas_holdem_no_limit_v6.env(render_mode="ansi", num_players=NUM_PLAYERS)
    agent = BaselineAgent()
    
    # Load main agent weights if provided
    if main_agent_checkpoint:
        agent.load_weights(main_agent_checkpoint)
    
    # Initialize strong opponent
    strong_inner_agent = StrongAgent(obs_dim=54, act_dim=5)
    opponent = StrongOpponent(strong_inner_agent)
    
    # Load strong opponent weights if provided
    if strong_opponent_checkpoint:
        opponent.load_weights(strong_opponent_checkpoint)
    
    # Initialize opponent model and tracker
    opponent_model = OpponentModel(54, 32, ACTION_SPACE_SIZE)
    tracker = OpponentTracker(opponent_model)
    
    # TensorBoard logging
    writer = SummaryWriter(log_dir="runs/testing_vs_strong")
    
    # Statistics tracking
    win_loss_stats = {"wins": 0, "losses": 0, "games": 0}
    accuracies = []
    all_bluff_scores = []
    total_bluff_attempts = 0
    total_successful_bluffs = 0
    
    cached_feedback_embedding = None
    
    for ep in range(1, episodes + 1):
        if ep % 100 == 0:
            print(f"Testing episode {ep}/{episodes}")
            
        feedback_embedding = cached_feedback_embedding
        
        env.reset()

        # Episode tracking variables
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
                if opponent_obs_history:
                    history_len = 10
                    recent_obs = opponent_obs_history[-history_len:]
                    if len(recent_obs) < history_len:
                        padding = [torch.zeros_like(torch.tensor(recent_obs[0]))] * (history_len - len(recent_obs))
                        recent_obs = padding + recent_obs
                    opponent_obs_seq = torch.stack([torch.tensor(o, dtype=torch.float32) for o in recent_obs])
                else:
                    opponent_obs_seq = None
                
                action, log_prob, value, likely_opp_action = agent.get_action(
                    state, mask, opponent_model, opponent_obs_seq, feedback_embedding=feedback_embedding
                )
                pred_actions.append(likely_opp_action)

                hole_cards, community_cards = decode_cards(state)
                bluff_score = estimate_bluff_score(hole_cards, community_cards)
                bluff_scores.append(bluff_score)
                all_bluff_scores.extend(bluff_scores)

                step_rewards.append(rew)
                deception_rewards.append(compute_deception_reward(bluff_score=bluff_score, action=action, bluff_reward=1))
                log_probs.append(log_prob)
                values.append(value)
                actions_this_game.append(int(action))
                states_this_game.append(state)

                # Track actions by stage for stats
                if stage in agent_actions_by_stage:
                    agent_actions_by_stage[stage].append(int(action))
                agent_all_actions.append(int(action))

            else:  # Strong Opponent (player_1)
                mask_clean = [bool(m) for m in mask]
                action, log_prob, value, _ = opponent.get_action(state, mask_clean)
                opponent_obs_history.append(state)
                
                # Track opponent actions by stage for stats
                if stage in opponent_actions_by_stage:
                    opponent_actions_by_stage[stage].append(int(action))
                opponent_all_actions.append(int(action))
                
                # Store for opponent training if continue_training is True
                if continue_training:
                    opponent_log_probs.append(log_prob)
                    opponent_rewards.append(rew)

                actual_actions.append(action)

            player = 0 if name == "player_0" else 1
            tracker.observe(state, action, player)

            if mask[action] == 0:
                print(f"[ILLEGAL] action {action} with mask {mask}, StrongOpponent, {name}")
            
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
        
        # Compute aggregated stats for feedback
        main_agent_agg_stats = compute_stats(main_agent_stats)
        opponent_agg_stats = compute_stats(opponent_stats)
       
        total_deception = sum(deception_rewards)
        combined_rewards = [r + d for r, d in zip(step_rewards, deception_rewards)]
        final_reward = cumulative_reward["player_0"]
        won_game = final_reward > 0

        successful_bluff = any(bs > 0.8 and a in [2, 3, 4] for a, bs in zip(actions_this_game, bluff_scores))
        if won_game and successful_bluff:
            final_bluff_bonus = 0.2 * final_reward
            if combined_rewards:
                combined_rewards[-1] += final_bluff_bonus

        # Update main agent (always update during testing for learning)
        if log_probs:
            agent.update(log_probs, values, combined_rewards)

        # Update strong opponent only if continue_training is True
        if continue_training and opponent_log_probs:
            opponent_combined_rewards = [r for r in opponent_rewards]  # No deception rewards for opponent
            opponent.update(opponent_log_probs, [value], opponent_combined_rewards)

        # Train opponent model
        batch = tracker.build_batch()
        pred_actions = np.array(pred_actions)
        actual_actions = np.array(actual_actions)
        min_len = min(len(pred_actions), len(actual_actions))
        accuracy = np.mean(pred_actions[:min_len] == actual_actions[:min_len]) if min_len > 0 else 0.0

        if not np.isnan(accuracy):
            accuracies.append(accuracy)
            writer.add_scalar("OpponentModel/Accuracy_StrongOpponent", accuracy, ep)

        if batch:
            loss = tracker.train_step(batch)

        tracker.reset()

        # Update win/loss stats
        win_loss_stats["games"] += 1
        r0, r1 = cumulative_reward["player_0"], cumulative_reward["player_1"]
        if r0 > r1:
            win_loss_stats["wins"] += 1
        else:
            win_loss_stats["losses"] += 1

        # TensorBoard logging
        total_reward = sum(step_rewards) + sum(deception_rewards)
        writer.add_scalar("Reward/Total", total_reward, ep)
        writer.add_scalar("Deception/Bonus", total_deception, ep)
        writer.add_scalar("Test/Successful_Bluff", successful_bluff, ep)

        if states_this_game:
            bluff_avg = np.mean(bluff_scores) if bluff_scores else 0.0
            writer.add_scalar("Bluff/AverageScore", bluff_avg, ep)

        if win_loss_stats["games"] > 0:
            winrate = win_loss_stats["wins"] / win_loss_stats["games"]
            writer.add_scalar("Test/WinRate_vs_StrongOpponent", winrate, ep)

        # Log poker stats to TensorBoard
        writer.add_scalar("Stats/Agent_VPIP", main_agent_agg_stats["VPIP"], ep)
        writer.add_scalar("Stats/Agent_PFR", main_agent_agg_stats["PFR"], ep)
        writer.add_scalar("Stats/Agent_AFq", main_agent_agg_stats["AFq"], ep)
        writer.add_scalar("Stats/Agent_WTSD", main_agent_agg_stats["WTSD"], ep)
        
        writer.add_scalar("Stats/Opponent_VPIP", opponent_agg_stats["VPIP"], ep)
        writer.add_scalar("Stats/Opponent_PFR", opponent_agg_stats["PFR"], ep)
        writer.add_scalar("Stats/Opponent_AFq", opponent_agg_stats["AFq"], ep)
        writer.add_scalar("Stats/Opponent_WTSD", opponent_agg_stats["WTSD"], ep)

        # Progress reporting
        if ep % 100 == 0:
            action_summary = Counter(a for a in actions_this_game if a != 0)
            print(f"\n=== Testing Progress: Episode {ep}/{episodes} ===")
            print(f"Reward = {total_reward:.1f}, Actions = {action_summary}")
            print(f"Agent Stats: VPIP={main_agent_agg_stats['VPIP']:.1f}%, PFR={main_agent_agg_stats['PFR']:.1f}%, AFq={main_agent_agg_stats['AFq']:.1f}%")
            print(f"Opponent Stats: VPIP={opponent_agg_stats['VPIP']:.1f}%, PFR={opponent_agg_stats['PFR']:.1f}%, AFq={opponent_agg_stats['AFq']:.1f}%")
            
            w, l, g = win_loss_stats["wins"], win_loss_stats["losses"], win_loss_stats["games"]
            winrate = w / g if g > 0 else 0.0
            print(f"Win Rate vs StrongOpponent: {w}W-{l}L ({winrate:.2%} over {g} games)")
            
            if accuracies:
                mean_accuracy = np.mean(accuracies[-100:])  # Last 100 episodes accuracy
                print(f"Opponent Model Accuracy (last 100): {mean_accuracy:.3f}")
            
            # Save checkpoints every 200 episodes during testing
            if ep % 200 == 0:
                try:
                    os.makedirs("test_checkpoints", exist_ok=True)
                    torch.save(agent.policy.state_dict(), f"test_checkpoints/main_agent_test_ep{ep}.pt")
                    if continue_training:
                        torch.save(opponent.agent.policy.state_dict(), f"test_checkpoints/strong_opponent_test_ep{ep}.pt")
                    print(f"Saved test checkpoints at episode {ep}")
                except Exception as e:
                    print(f"Couldn't save test checkpoints: {e}")

        # Bluff statistics
        bluff_attempts = sum(1 for a, bs in zip(actions_this_game, bluff_scores) if bs > 0.8 and a in [2, 3, 4])
        successful_bluffs = int(successful_bluff)
        total_bluff_attempts += bluff_attempts
        total_successful_bluffs += successful_bluffs

        # Generate feedback (less frequently during testing)
        if ep % 50 == 0:  # Every 50 episodes instead of every episode
            feedback_text = get_gpt_feedback(main_agent_agg_stats, opponent_agg_stats)
            print(f"GPT Feedback (ep {ep}): {feedback_text}")
            writer.add_text("Feedback/GPT_Response", feedback_text, ep)
            cached_feedback_embedding = tokenize_feedback(feedback_encoder, feedback_text)
   
    # Final summary
    print(f"\n=== TESTING COMPLETE ===")
    print(f"Episodes: {episodes}")
    print(f"Continue Training: {continue_training}")
    w, l, g = win_loss_stats["wins"], win_loss_stats["losses"], win_loss_stats["games"]
    winrate = w / g if g > 0 else 0.0
    print(f"Final Win Rate vs StrongOpponent: {w}W-{l}L ({winrate:.2%})")
    
    if accuracies:
        mean_accuracy = np.mean(accuracies)
        print(f"Average Opponent Model Accuracy: {mean_accuracy:.3f}")
    
    bluff_success_rate = (total_successful_bluffs / total_bluff_attempts * 100) if total_bluff_attempts > 0 else 0.0
    print(f"Bluff Success Rate: {total_successful_bluffs}/{total_bluff_attempts} ({bluff_success_rate:.1f}%)")
    
    print(f"Final Agent Stats: VPIP={main_agent_agg_stats['VPIP']:.1f}%, PFR={main_agent_agg_stats['PFR']:.1f}%, AFq={main_agent_agg_stats['AFq']:.1f}%")
    
    writer.close()
    
    return {
        "win_rate": winrate,
        "total_games": g,
        "agent_stats": main_agent_agg_stats,
        "opponent_stats": opponent_agg_stats,
        "bluff_success_rate": bluff_success_rate,
        "opponent_model_accuracy": np.mean(accuracies) if accuracies else 0.0
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test poker agents against StrongOpponent")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes to test (default: 1000)")
    parser.add_argument("--main-agent-checkpoint", type=str, default=None, 
                       help="Path to main agent checkpoint (e.g., checkpoints_8/main_agent_ep5000.pt)")
    parser.add_argument("--strong-opponent-checkpoint", type=str, default=None,
                       help="Path to strong opponent checkpoint (e.g., checkpoints_8/opponent_ep5000.pt)")
    parser.add_argument("--continue-training", action="store_true", default=True,
                       help="Continue updating strong opponent during testing (default: True)")
    parser.add_argument("--no-continue-training", dest="continue_training", action="store_false",
                       help="Freeze strong opponent during testing")
    
    args = parser.parse_args()
    
    print("Starting Testing Script")
    print(f"Episodes: {args.episodes}")
    print(f"Main Agent Checkpoint: {args.main_agent_checkpoint}")
    print(f"Strong Opponent Checkpoint: {args.strong_opponent_checkpoint}")
    print(f"Continue Training: {args.continue_training}")
    
    # Run testing
    results = test_against_strong_opponent(
        episodes=args.episodes,
        main_agent_checkpoint=args.main_agent_checkpoint,
        strong_opponent_checkpoint=args.strong_opponent_checkpoint,
        continue_training=args.continue_training
    )
    
    print("\n=== FINAL RESULTS ===")
    for key, value in results.items():
        print(f"{key}: {value}")
        
        
# python test_script.py --main-agent-checkpoint checkpoints_8/main_agent_ep5000.pt --strong-opponent-checkpoint checkpoints_8/opponent_ep5000.pt