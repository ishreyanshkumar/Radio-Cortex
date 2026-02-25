"""
Neural Network Architectures for Radio-Cortex
"""

import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO
    Actor: Policy network (state -> action)
    Critic: Value network (state -> value)
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()        
        # Shared feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Actor heads (policy)
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Output normalized actions
        )
        
        # State-dependent standard deviation for dynamic entropy
        self.actor_logstd_head = nn.Linear(hidden_dim, action_dim)
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, state):
        features = self.feature_net(state)
        action_mean = self.actor_mean(features)
        logstd = self.actor_logstd_head(features)
        # Optional: clamp logstd to avoid extreme values
        logstd = torch.clamp(logstd, -2, 1)
        value = self.critic(features)
        return action_mean, logstd, value
    
    def get_action(self, state, deterministic=False):
        """Sample action from policy"""
        action_mean, logstd, _ = self.forward(state)
        
        if deterministic:
            return action_mean, None, None
        
        action_std = torch.exp(logstd)
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return action, log_prob, entropy
    
    def evaluate_actions(self, state, action):
        """Evaluate log probability and entropy of actions"""
        action_mean, logstd, value = self.forward(state)
        action_std = torch.exp(logstd)
        dist = torch.distributions.Normal(action_mean, action_std)
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return value, log_prob, entropy
