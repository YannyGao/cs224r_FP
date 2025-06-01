import torch
import torch.nn as nn
import torch.nn.functional as F

class OpponentModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_actions, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Heads
        self.action_head = nn.Linear(hidden_dim, num_actions)  # logits, no activation
        self.hand_strength_head = nn.Linear(hidden_dim, 1)
        self.aggression_head = nn.Linear(hidden_dim, 1)
        self.bluff_tendency_head = nn.Linear(hidden_dim, 1)
        self.confidence_head = nn.Linear(hidden_dim, 1)

    def forward(self, x, src_key_padding_mask=None):
        """
        x: Tensor of shape (batch_size, seq_len, input_dim)
        src_key_padding_mask: Optional mask for padding tokens (batch_size, seq_len), bool tensor
        """
        print(x.shape)
        x = self.input_proj(x)  # (batch, seq_len, hidden_dim)
        print(x.shape)
        # x = x.transpose(0, 1)
        print(x.shape)
        enc_out = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)  # (seq_len, batch, hidden_dim)
        
        last_hidden = enc_out[:, -1, :] # (batch, hidden_dim)
        print(last_hidden.shape)
        action_logits = self.action_head(last_hidden)  # (batch, num_actions)
        hand_strength = torch.sigmoid(self.hand_strength_head(last_hidden)).squeeze(-1)  # (batch,)
        aggression = torch.sigmoid(self.aggression_head(last_hidden)).squeeze(-1)
        bluff_tendency = torch.sigmoid(self.bluff_tendency_head(last_hidden)).squeeze(-1)
        confidence = torch.sigmoid(self.confidence_head(last_hidden)).squeeze(-1)

        return {
            'action_logits': action_logits,
            'hand_strength': hand_strength,
            'aggression': aggression,
            'bluff_tendency': bluff_tendency,
            'confidence': confidence
        }


class OpponentTracker:
    def __init__(self, model, device=None, lr=1e-3):
        self.device = device or torch.device('cpu')
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Losses
        self.action_loss_fn = nn.CrossEntropyLoss()
        self.regression_loss_fn = nn.MSELoss()
        self.trajectory = []
    def observe(self, obs, action, hand_strength=0.5, aggression=0.5, bluff_tendency=0.5, confidence=0.5):
        # Save observation and labels (you can update these if you have better heuristics)
        self.trajectory.append({
            "observation": torch.tensor(obs, dtype=torch.float),
            "action": torch.tensor(action, dtype=torch.long),
            "hand_strength": torch.tensor(hand_strength, dtype=torch.float),
            "aggression": torch.tensor(aggression, dtype=torch.float),
            "bluff_tendency": torch.tensor(bluff_tendency, dtype=torch.float),
            "confidence": torch.tensor(confidence, dtype=torch.float),
        })

    def build_batch(self):
        if len(self.trajectory) < 2:
            return None  # Not enough data to build a sequence

        # Simplified fixed-length batch construction
        seq_len = len(self.trajectory)
        input_dim = self.trajectory[0]['observation'].shape[0]

        observations = torch.stack([step['observation'] for step in self.trajectory]).unsqueeze(0)  # (1, seq_len, input_dim)
        actions = torch.tensor([step['action'] for step in self.trajectory[-1:]])  # only last action matters
        hand_strength = torch.tensor([step['hand_strength'] for step in self.trajectory[-1:]])
        aggression = torch.tensor([step['aggression'] for step in self.trajectory[-1:]])
        bluff_tendency = torch.tensor([step['bluff_tendency'] for step in self.trajectory[-1:]])
        confidence = torch.tensor([step['confidence'] for step in self.trajectory[-1:]])
        padding_mask = torch.zeros((1, seq_len), dtype=torch.bool)

        return {
            'observations': observations,
            'actions': actions,
            'hand_strength': hand_strength,
            'aggression': aggression,
            'bluff_tendency': bluff_tendency,
            'confidence': confidence,
            'padding_mask': padding_mask
        }

    def reset(self):
        self.trajectory = []

    def train_step(self, batch):
        """
        batch is a dict containing:
          - 'observations': tensor (batch, seq_len, input_dim)
          - 'actions': tensor (batch,) long, ground truth action labels
          - 'hand_strength': tensor (batch,) float
          - 'aggression': tensor (batch,) float
          - 'bluff_tendency': tensor (batch,) float
          - 'confidence': tensor (batch,) float
          - 'padding_mask': optional bool tensor (batch, seq_len), True for padding tokens
        """
        self.model.train()
        obs = batch['observations'].to(self.device)
        actions = batch['actions'].to(self.device)
        hand_strength = batch['hand_strength'].to(self.device)
        aggression = batch['aggression'].to(self.device)
        bluff_tendency = batch['bluff_tendency'].to(self.device)
        confidence = batch['confidence'].to(self.device)

        padding_mask = batch.get('padding_mask', None)
        if padding_mask is not None:
            padding_mask = padding_mask.to(self.device)

        preds = self.model(obs, src_key_padding_mask=padding_mask)

        loss_action = self.action_loss_fn(preds['action_logits'], actions)
        loss_hand = self.regression_loss_fn(preds['hand_strength'], hand_strength)
        loss_aggr = self.regression_loss_fn(preds['aggression'], aggression)
        loss_bluff = self.regression_loss_fn(preds['bluff_tendency'], bluff_tendency)
        loss_conf = self.regression_loss_fn(preds['confidence'], confidence)

        total_loss = loss_action + loss_hand + loss_aggr + loss_bluff + loss_conf

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'action_loss': loss_action.item(),
            'hand_strength_loss': loss_hand.item(),
            'aggression_loss': loss_aggr.item(),
            'bluff_loss': loss_bluff.item(),
            'confidence_loss': loss_conf.item()
        }

    def predict(self, observations, padding_mask=None):
        """
        observations: tensor (batch, seq_len, input_dim)
        padding_mask: optional bool tensor (batch, seq_len)
        Returns dict of predictions (same format as forward)
        """
        self.model.eval()
        with torch.no_grad():
            obs = observations.to(self.device)
            mask = padding_mask.to(self.device) if padding_mask is not None else None
            preds = self.model(obs, src_key_padding_mask=mask)
        return preds
