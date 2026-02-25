"""
Universal Transformer Policy - EFFICIENT Baseline
Demonstrates Parameter Efficiency via Weight Sharing across steps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class UniversalPolicy(nn.Module):
    """
    Universal Transformer.
    Recurrently applies the SAME transformer block T times (depth-wise).
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, 
                 n_layer: int = 4, n_head: int = 4, context_len: int = 64, 
                 dropout: float = 0.1, device: str = 'cpu'):
        super().__init__()
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.n_layer = n_layer # Depth of recursion
        self.context_len = context_len
        
        # 1. Embeddings
        self.state_embed = nn.Linear(state_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, context_len, hidden_dim))
        # Universal Transformer also typically adds a "step" embedding for depth
        self.step_embed = nn.Embedding(n_layer, hidden_dim)
        
        self.drop = nn.Dropout(dropout)

        # 2. Shared Block
        self.shared_block = UniversalBlock(hidden_dim, n_head, context_len, dropout)
        
        self.ln_f = nn.LayerNorm(hidden_dim)

        # 3. Heads
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        self.critic = nn.Linear(hidden_dim, 1)

        self.apply(self._init_weights)
        
        self.history_buffer = []

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def forward(self, states, timesteps=None):
        if states.dim() == 2:
            states = states.unsqueeze(1)
            
        B, T, _ = states.size()
        
        token_embeddings = self.state_embed(states)
        
        if T > self.context_len:
             token_embeddings = token_embeddings[:, -self.context_len:, :]
             T = self.context_len
             
        position_embeddings = self.pos_embed[:, :T, :]
        x = self.drop(token_embeddings + position_embeddings)
        
        # Recurrent application of shared block
        for i in range(self.n_layer):
            # Add step embedding (optional but recommended for Universal)
            # x = x + self.step_embed(torch.tensor(i, device=self.device))
            x = self.shared_block(x)
            
        x = self.ln_f(x)
        
        logits = self.actor_mean(x)
        values = self.critic(x)
        
        return logits, self.actor_logstd, values

    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        # Update history buffer (Simple logic similar to GPT2Policy)
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        if len(self.history_buffer) == 0:
             self.history_buffer = state
        else:
             self.history_buffer = torch.cat([self.history_buffer, state], dim=0)
        
        if self.history_buffer.size(0) > self.context_len:
            self.history_buffer = self.history_buffer[-self.context_len:, :]
            
        context = self.history_buffer.unsqueeze(0)
        action_mean, logstd, _ = self.forward(context)
        action_mean = action_mean[:, -1, :]
        
        if deterministic:
            return action_mean, None, None
            
        action_std = torch.exp(logstd)
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return action, log_prob, entropy

    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor):
        # Universal Transformer with PPO:
        # We process inputs as single-step sequences to evaluate the shared-weight mechanism.
        if state.dim() == 2: state = state.unsqueeze(1)
        action_mean, logstd, value = self.forward(state)
        action_mean = action_mean[:, -1, :]
        value = value[:, -1, :]
        
        action_std = torch.exp(logstd)
        dist = torch.distributions.Normal(action_mean, action_std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return value, log_prob, entropy

    def reset_memory(self):
        self.history_buffer = []


class UniversalBlock(nn.Module):
    def __init__(self, hidden_dim, n_head, context_len, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.attn = CausalSelfAttention(hidden_dim, n_head, context_len, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class CausalSelfAttention(nn.Module):
    # Reusing from GPT2
    def __init__(self, hidden_dim, n_head, context_len, dropout):
        super().__init__()
        assert hidden_dim % n_head == 0
        self.c_attn = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.c_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.hidden_dim = hidden_dim
        self.register_buffer("bias", torch.tril(torch.ones(context_len, context_len))
                                     .view(1, 1, context_len, context_len))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(self.hidden_dim, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))
