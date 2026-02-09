"""
BDH Policy Adapter for Radio-Cortex
"""

import torch
import torch.nn as nn
import importlib
from typing import Optional

class BDHPolicy(nn.Module):
    """
    Minimal adapter that allows a `bdh.BDH` model to be used as a policy.

    This wrapper converts the input state vector into a short sequence of
    discrete token ids which are embedded via the BDH model's embedding
    layer; the pooled embedding is then projected to the action space.

    This implementation is intended for inference/compatibility: it provides
    `get_action(state, deterministic=True)` and `evaluate_actions(...)`
    with simple deterministic/stochastic behavior so it can replace
    `ActorCritic` in the rest of the codebase.
    """

    def __init__(self, state_dim: int, action_dim: int, bdh_config: Optional[object] = None, device: str = 'cpu'):
        super().__init__()
        try:
            # Try importing from current directory (Radio-Cortex)
            import bdh as bdh_mod
        except ImportError:
             try:
                 # Fallback if package structure differs
                 bdh_mod = importlib.import_module('Radio-Cortex.bdh')
             except Exception as e:
                 raise ImportError("Could not import bdh module: " + str(e))

        # instantiate BDH core model
        cfg = bdh_config or bdh_mod.BDHConfig()
        self.device = device
        self.bdh = bdh_mod.BDH(cfg).to(device)

        # small adapter: project pooled BDH embedding to action dim
        emb_dim = cfg.n_embd
        self.pool_proj = nn.Linear(emb_dim, action_dim)

        # state -> token logits: map state to small sequence of token ids
        self.seq_len = 8
        self.vocab_size = getattr(cfg, 'vocab_size', 256)
        # Using a simple linear layer to map state to token logits
        self.state_to_tokens = nn.Linear(state_dim, self.seq_len * self.vocab_size)
        
        # State-dependent standard deviation for exploration (Dynamic Entropy)
        # Project from pooled BDH embedding
        self.logstd_proj = nn.Linear(emb_dim, action_dim)
        
        # Value head (critic) for explained variance
        self.value_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2),
            nn.ReLU(),
            nn.Linear(emb_dim // 2, 1)
        )

    def forward(self, state: torch.Tensor):
        # state: (B, state_dim)
        B = state.size(0)
        logits = self.state_to_tokens(state)
        logits = logits.view(B, self.seq_len, self.vocab_size)
        # pick most likely token per position (deterministic adapter)
        token_ids = torch.argmax(logits, dim=-1).to(torch.long)

        # use BDH embedding to produce features
        emb = self.bdh.embed(token_ids.to(self.bdh.embed.weight.device))
        # emb: (B, seq_len, emb_dim) -> pool
        pooled = emb.mean(dim=1)
        action_mean = self.pool_proj(pooled)
        logstd = self.logstd_proj(pooled)
        # clamp logstd similarly to ActorCritic
        logstd = torch.clamp(logstd, -2, 1)
        
        # produce real value estimate
        value = self.value_head(pooled)
        return action_mean, logstd, value

    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        """Return an action tensor (or (action, log_prob)) like ActorCritic.get_action"""
        action_mean, logstd, _ = self.forward(state)
        if deterministic:
            return action_mean, None, None
        
        # Stochasticity using state-dependent logstd
        action_std = torch.exp(logstd)
        dist = torch.distributions.Normal(action_mean, action_std)
        
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return action, log_prob, entropy

    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor):
        action_mean, logstd, value = self.forward(state)
        
        # Use state-dependent logstd
        action_std = torch.exp(logstd)
        dist = torch.distributions.Normal(action_mean, action_std)
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return value, log_prob, entropy
