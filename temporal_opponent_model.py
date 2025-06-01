import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import PositionalEncoding, MultiHeadAttention

class OpponentModel(nn.Module):
    """
    Transformer-only opponent model - simpler and more effective for poker
    """
    def __init__(self, obs_dim: int, hidden_dim: int, act_dim: int, 
                 num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.act_dim = act_dim
        
        # Input projection
        self.input_proj = nn.Linear(obs_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim)
        
        # Stack of transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Output heads - what we want to predict about opponent
        self.action_head = nn.Linear(hidden_dim, act_dim)          # What will they do?
        self.aggression_head = nn.Linear(hidden_dim, 1)            # How aggressive are they?
        self.bluff_tendency_head = nn.Linear(hidden_dim, 1)        # Do they bluff often?
        self.confidence_head = nn.Linear(hidden_dim, 1)            # How confident is our prediction?
        
    def forward(self, obs_sequence):
        """
        Args:
            obs_sequence: [batch_size, seq_len, obs_dim] - opponent's recent actions
        Returns:
            Dictionary of predictions about opponent's next move
        """
        print(obs_sequence.shape)
        batch_size, seq_len, _ = obs_sequence.shape
        
        # Project to hidden dimension
        x = self.input_proj(obs_sequence)  # [batch, seq_len, hidden_dim]
        
        # Add positional encoding
        x = x.transpose(0, 1)  # [seq_len, batch, hidden_dim]
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # [batch, seq_len, hidden_dim]
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Use last timestep for prediction (most recent context)
        final_hidden = x[:, -1, :]  # [batch, hidden_dim]
        
        # Make predictions
        predictions = {
            'action_probs': F.softmax(self.action_head(final_hidden), dim=-1),
            'aggression': torch.sigmoid(self.aggression_head(final_hidden)),
            'bluff_tendency': torch.sigmoid(self.bluff_tendency_head(final_hidden)),
            'confidence': torch.sigmoid(self.confidence_head(final_hidden))
        }
        
        return predictions


class TransformerBlock(nn.Module):
    """Single transformer block with self-attention + feed-forward"""
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        
        self.attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        # Self-attention with residual connection
        attended, _ = self.attention(x, x, x)
        x = self.norm1(x + attended)
        
        # Feed-forward with residual connection  
        x = self.norm2(x + self.ffn(x))
        
        return x