import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
from typing import List, Tuple, Optional, Dict
import math
from transformer import PositionalEncoding, MultiHeadAttention
class TemporalOpponentModel(nn.Module):
    """
    Advanced opponent model with LSTM + Attention for temporal pattern recognition
    """
    def __init__(self, obs_dim: int, hidden_dim: int, act_dim: int, 
                 num_layers: int = 2, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.act_dim = act_dim
        self.num_layers = num_layers
        
        # Input embedding
        self.input_embedding = nn.Linear(obs_dim, hidden_dim)
        
        # LSTM for sequential modeling
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Attention mechanism
        self.attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.attention_norm = nn.LayerNorm(hidden_dim)
        
        # Positional encoding for attention
        self.pos_encoding = PositionalEncoding(hidden_dim)
        
        # Output heads for multi-task learning
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, act_dim)
        )
        
        self.hand_strength_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.aggression_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.bluff_tendency_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Confidence head - how confident the model is in its predictions
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, obs_sequence, return_attention=False):
        """
        Args:
            obs_sequence: [batch_size, seq_len, obs_dim]
            return_attention: whether to return attention weights
        
        Returns:
            Dict of predictions and optionally attention weights
        """
        batch_size, seq_len, _ = obs_sequence.shape
        
        # Input embedding
        embedded = self.input_embedding(obs_sequence)  # [batch, seq_len, hidden_dim]
        
        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(embedded)  # [batch, seq_len, hidden_dim]
        
        # Add positional encoding for attention
        lstm_out_transposed = lstm_out.transpose(0, 1)  # [seq_len, batch, hidden_dim]
        lstm_out_with_pos = self.pos_encoding(lstm_out_transposed)
        lstm_out_with_pos = lstm_out_with_pos.transpose(0, 1)  # [batch, seq_len, hidden_dim]
        
        # Self-attention
        attended_out, attention_weights = self.attention(
            lstm_out_with_pos, lstm_out_with_pos, lstm_out_with_pos
        )
        
        # Residual connection and layer norm
        attended_out = self.attention_norm(attended_out + lstm_out)
        
        # Use the last timestep for predictions
        final_hidden = attended_out[:, -1, :]  # [batch, hidden_dim]
        
        # Multi-task outputs
        predictions = {
            'action_logits': F.log_softmax(self.action_head(final_hidden), dim=-1),
            'hand_strength': torch.sigmoid(self.hand_strength_head(final_hidden)),
            'aggression': torch.sigmoid(self.aggression_head(final_hidden)),
            'bluff_tendency': torch.sigmoid(self.bluff_tendency_head(final_hidden)),
            'confidence': torch.sigmoid(self.confidence_head(final_hidden))
        }
        
        if return_attention:
            predictions['attention_weights'] = attention_weights
            
        return predictions