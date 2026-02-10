"""
Transformer Policy 1 - Placeholder for Radio-Cortex
Custom transformer-based policy architecture (to be implemented).
"""

import torch
import torch.nn as nn


class TransformerPolicy1(nn.Module):
    """
    Transformer-based policy architecture variant 1.
    
    Placeholder: currently wraps a simple MLP with the same interface
    as ActorCritic. Replace internals with actual transformer logic later.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, device: str = 'cpu'):
        super().__init__()
        self.device = device

        # --- Placeholder MLP (replace with transformer later) ---
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )
        self.actor_logstd_head = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, state: torch.Tensor):
        features = self.feature_net(state)
        action_mean = self.actor_mean(features)
        logstd = torch.clamp(self.actor_logstd_head(features), -2, 1)
        value = self.critic(features)
        return action_mean, logstd, value

    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        action_mean, logstd, _ = self.forward(state)
        if deterministic:
            return action_mean, None, None
        action_std = torch.exp(logstd)
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy

    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor):
        action_mean, logstd, value = self.forward(state)
        action_std = torch.exp(logstd)
        dist = torch.distributions.Normal(action_mean, action_std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return value, log_prob, entropy
