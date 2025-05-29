import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
from typing import List, Tuple, Optional, Dict
import math
from temporal_opponent_model import TemporalOpponentModel
class TemporalOpponentTracker:
    """
    Enhanced opponent tracker with temporal sequence management
    """
    def __init__(self, model: TemporalOpponentModel, lr: float = 1e-3, 
                 sequence_length: int = 20, buffer_size: int = 1000):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.sequence_length = sequence_length
        self.buffer_size = buffer_size
        
        # Sequence buffer for temporal data
        self.sequence_buffer = deque(maxlen=buffer_size)
        self.current_sequence = []
        
        # Loss functions
        self.action_loss_fn = nn.NLLLoss()
        self.regression_loss_fn = nn.MSELoss()
        
        # Training statistics
        self.training_stats = {
            'total_loss': [],
            'action_loss': [],
            'hand_strength_loss': [],
            'aggression_loss': [],
            'bluff_tendency_loss': []
        }
        
    def observe(self, obs: np.ndarray, action: int, 
                hand_strength: Optional[float] = None,
                pot_size: float = 0, bet_amount: float = 0,
                game_phase: str = "preflop"):
        """
        Observe opponent action and add to current sequence
        """
        # Create enriched observation with additional context
        enriched_obs = self._enrich_observation(obs, pot_size, bet_amount, game_phase)
        
        observation_data = {
            'obs': torch.tensor(enriched_obs, dtype=torch.float32),
            'action': action,
            'hand_strength': hand_strength,
            'pot_size': pot_size,
            'bet_amount': bet_amount,
            'normalized_bet': bet_amount / max(pot_size, 1.0),
            'game_phase': game_phase,
            'timestamp': len(self.current_sequence)
        }
        
        self.current_sequence.append(observation_data)
        
        # If sequence is complete, add to buffer
        if len(self.current_sequence) >= self.sequence_length:
            self.sequence_buffer.append(self.current_sequence.copy())
            # Keep sliding window
            self.current_sequence = self.current_sequence[1:]
    
    def _enrich_observation(self, obs: np.ndarray, pot_size: float, 
                           bet_amount: float, game_phase: str) -> np.ndarray:
        """
        Add contextual features to observation
        """
        # Game phase encoding
        phase_encoding = {
            'preflop': [1, 0, 0, 0],
            'flop': [0, 1, 0, 0],
            'turn': [0, 0, 1, 0],
            'river': [0, 0, 0, 1]
        }
        
        phase_vec = phase_encoding.get(game_phase, [0, 0, 0, 0])
        
        # Betting context
        normalized_bet = bet_amount / max(pot_size, 1.0)
        pot_odds = bet_amount / max(pot_size + bet_amount, 1.0)
        
        # Combine original observation with enriched features
        enriched = np.concatenate([
            obs,
            phase_vec,
            [normalized_bet, pot_odds]
        ])
        
        return enriched
    
    def end_hand(self, final_outcome: Optional[float] = None):
        """
        Call this at the end of each hand to finalize the sequence
        """
        if len(self.current_sequence) > 0:
            # Add final outcome information if available
            if final_outcome is not None:
                for obs_data in self.current_sequence:
                    obs_data['final_outcome'] = final_outcome
            
            # Add sequence to buffer even if not full length
            self.sequence_buffer.append(self.current_sequence.copy())
            self.current_sequence = []
    
    def _compute_labels(self, sequence: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Compute training labels from sequence data
        """
        seq_len = len(sequence)
        
        # Action labels
        actions = torch.tensor([obs['action'] for obs in sequence], dtype=torch.long)
        
        # Hand strength labels (if available)
        hand_strengths = []
        valid_hand_indices = []
        for i, obs in enumerate(sequence):
            if obs['hand_strength'] is not None:
                hand_strengths.append(obs['hand_strength'])
                valid_hand_indices.append(i)
        
        hand_strength_labels = torch.tensor(hand_strengths, dtype=torch.float32) if hand_strengths else None
        
        # Aggression labels (derived from betting behavior)
        aggression_labels = []
        for obs in sequence:
            # Compute aggression based on action and bet size
            action = obs['action']
            normalized_bet = obs['normalized_bet']
            
            if action == 0:  # Fold
                aggression = 0.0
            elif action == 1:  # Call
                aggression = 0.3
            else:  # Raise variants
                aggression = 0.5 + 0.5 * min(normalized_bet, 1.0)
            
            aggression_labels.append(aggression)
        
        aggression_labels = torch.tensor(aggression_labels, dtype=torch.float32)
        
        # Bluff tendency labels (heuristic based on outcome vs betting behavior)
        bluff_labels = []
        for obs in sequence:
            # Simple heuristic: aggressive betting with poor outcome suggests bluffing
            action_aggression = max(0, obs['action'] - 1) / 3.0  # Normalize action to [0,1]
            final_outcome = obs.get('final_outcome', 0)
            
            # If aggressive action but poor outcome, likely a bluff
            if action_aggression > 0.5 and final_outcome < 0:
                bluff_tendency = 0.7
            elif action_aggression < 0.3:
                bluff_tendency = 0.2
            else:
                bluff_tendency = 0.4
                
            bluff_labels.append(bluff_tendency)
        
        bluff_labels = torch.tensor(bluff_labels, dtype=torch.float32)
        
        return {
            'actions': actions,
            'hand_strengths': hand_strength_labels,
            'hand_strength_indices': valid_hand_indices,
            'aggression': aggression_labels,
            'bluff_tendency': bluff_labels
        }
    
    def train_step(self, batch_size: int = 8) -> Optional[Dict[str, float]]:
        """
        Train the temporal opponent model
        """
        if len(self.sequence_buffer) < batch_size:
            return None
        
        # Sample batch of sequences
        batch_sequences = np.random.choice(len(self.sequence_buffer), batch_size, replace=False)
        
        total_loss = 0
        losses = {'action': 0, 'hand_strength': 0, 'aggression': 0, 'bluff_tendency': 0}
        
        self.optimizer.zero_grad()
        
        for seq_idx in batch_sequences:
            sequence = self.sequence_buffer[seq_idx]
            
            # Prepare input sequence
            obs_sequence = torch.stack([obs['obs'] for obs in sequence]).unsqueeze(0)  # [1, seq_len, obs_dim]
            
            # Get model predictions
            predictions = self.model(obs_sequence)
            
            # Compute labels
            labels = self._compute_labels(sequence)
            
            # Action loss (classification)
            action_loss = self.action_loss_fn(
                predictions['action_logits'].repeat(len(labels['actions']), 1),
                labels['actions']
            )
            losses['action'] += action_loss.item()
            total_loss += action_loss
            
            # Hand strength loss (regression, if available)
            if labels['hand_strengths'] is not None and len(labels['hand_strengths']) > 0:
                hand_strength_loss = self.regression_loss_fn(
                    predictions['hand_strength'].repeat(len(labels['hand_strengths']), 1).squeeze(),
                    labels['hand_strengths']
                )
                losses['hand_strength'] += hand_strength_loss.item()
                total_loss += 0.5 * hand_strength_loss
            
            # Aggression loss (regression)
            aggression_loss = self.regression_loss_fn(
                predictions['aggression'].repeat(len(labels['aggression']), 1).squeeze(),
                labels['aggression']
            )
            losses['aggression'] += aggression_loss.item()
            total_loss += 0.3 * aggression_loss
            
            # Bluff tendency loss (regression)
            bluff_loss = self.regression_loss_fn(
                predictions['bluff_tendency'].repeat(len(labels['bluff_tendency']), 1).squeeze(),
                labels['bluff_tendency']
            )
            losses['bluff_tendency'] += bluff_loss.item()
            total_loss += 0.2 * bluff_loss
        
        # Backpropagation
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Gradient clipping
        self.optimizer.step()
        
        # Update training statistics
        avg_losses = {key: value / batch_size for key, value in losses.items()}
        avg_losses['total'] = total_loss.item() / batch_size
        
        for key, value in avg_losses.items():
            if key in self.training_stats:
                self.training_stats[key].append(value)
            else:
                self.training_stats[key] = [value]
        
        return avg_losses
    
    def get_recent_behavior_summary(self, lookback: int = 50) -> Dict[str, float]:
        """
        Get recent behavioral statistics
        """
        if not self.sequence_buffer:
            return {}
        
        recent_actions = []
        recent_bets = []
        
        # Look at recent sequences
        for sequence in list(self.sequence_buffer)[-lookback:]:
            for obs in sequence:
                recent_actions.append(obs['action'])
                recent_bets.append(obs['normalized_bet'])
        
        if not recent_actions:
            return {}
        
        return {
            'fold_rate': recent_actions.count(0) / len(recent_actions),
            'call_rate': recent_actions.count(1) / len(recent_actions),
            'raise_rate': sum(1 for a in recent_actions if a >= 2) / len(recent_actions),
            'avg_aggression': sum(max(0, a-1) for a in recent_actions) / len(recent_actions),
            'avg_bet_size': np.mean(recent_bets) if recent_bets else 0,
            'total_observations': len(recent_actions)
        }
    
    def predict_next_action(self, current_sequence: List[np.ndarray], 
                           return_confidence: bool = False) -> Dict:
        """
        Predict opponent's next action given current sequence
        """
        if len(current_sequence) == 0:
            return {'action': 1, 'confidence': 0.0}  # Default to call with low confidence
        
        self.model.eval()
        with torch.no_grad():
            # Pad sequence if too short
            padded_sequence = current_sequence.copy()
            while len(padded_sequence) < self.sequence_length:
                padded_sequence.insert(0, np.zeros_like(current_sequence[0]))
            
            # Take last sequence_length observations
            sequence_tensor = torch.stack([
                torch.tensor(obs, dtype=torch.float32) 
                for obs in padded_sequence[-self.sequence_length:]
            ]).unsqueeze(0)
            
            predictions = self.model(sequence_tensor, return_attention=True)
            
            # Get most likely action
            action_probs = torch.exp(predictions['action_logits'])
            predicted_action = torch.argmax(action_probs).item()
            confidence = torch.max(action_probs).item()
            
            result = {
                'action': predicted_action,
                'action_probs': action_probs.squeeze().tolist(),
                'hand_strength': predictions['hand_strength'].item(),
                'aggression': predictions['aggression'].item(),
                'bluff_tendency': predictions['bluff_tendency'].item(),
                'model_confidence': predictions['confidence'].item()
            }
            
            if return_confidence:
                result['prediction_confidence'] = confidence
                result['attention_weights'] = predictions['attention_weights']
        
        self.model.train()
        return result