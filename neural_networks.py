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
        
        # Actor head (policy)
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Output normalized actions
        )
        
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, state):
        features = self.feature_net(state)
        action_mean = self.actor_mean(features)
        value = self.critic(features)
        return action_mean, value
    
    def get_action(self, state, deterministic=False):
        """Sample action from policy"""
        action_mean, _ = self.forward(state)
        
        if deterministic:
            return action_mean
        
        action_std = torch.exp(self.actor_logstd)
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob
    
    def evaluate_actions(self, state, action):
        """Evaluate log probability and entropy of actions"""
        action_mean, value = self.forward(state)
        action_std = torch.exp(self.actor_logstd)
        dist = torch.distributions.Normal(action_mean, action_std)
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return value, log_prob, entropy



