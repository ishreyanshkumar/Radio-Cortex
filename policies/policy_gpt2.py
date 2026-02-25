"""
Transformer Policy 1 - Standard Decoder-Only Transformer (GPT-2 Style)
Serves as the "Standard" baseline to demonstrate O(T^2) complexity and fixed-window limitations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GPT2Policy(nn.Module):
    """
    Standard Decoder-Only Transformer (Decision Transformer / GPT-2 Style).
    
    Architecture:
    - Embedding: Linear projection of state + Learned Positional Embedding
    - Blocks: Causal Self-Attention + MLP
    - Context: Fixed window of history (e.g., 64 steps)
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, 
                 n_layer: int = 4, n_head: int = 4, context_len: int = 64, 
                 dropout: float = 0.1, device: str = 'cpu'):
        super().__init__()
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.context_len = context_len

        # 1. Embeddings
        self.state_embed = nn.Linear(state_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, context_len, hidden_dim))
        self.drop = nn.Dropout(dropout)

        # 2. Transformer Blocks
        self.blocks = nn.ModuleList([
            Block(hidden_dim, n_head, context_len, dropout) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(hidden_dim)

        # 3. Heads
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_logstd = nn.Linear(hidden_dim, action_dim)  # State-dependent exploration
        self.critic = nn.Linear(hidden_dim, 1)

        self.apply(self._init_weights)

        # Runtime buffer for context (not a model parameter)
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
        """
        states: (B, T, state_dim) or (B, state_dim)
        """
        # Handle 2D input (B, D) from standard PPO
        if states.dim() == 2:
            states = states.unsqueeze(1) # (B, 1, D)

        B, T, _ = states.size()
        
        # Embed states
        token_embeddings = self.state_embed(states) # (B, T, H)
        
        # Add positional embeddings (broadcast across batch)
        # If T < context_len, we take the last T positions or just 0..T?
        # Standard approach: 0..T for simplicity in RL, or use timestep index if available.
        # Here we just use 0..T since we slide the window.
        if T > self.context_len:
             # This should ideally be handled by truncating input before forward
             token_embeddings = token_embeddings[:, -self.context_len:, :]
             T = self.context_len

        position_embeddings = self.pos_embed[:, :T, :] # (1, T, H)
        x = self.drop(token_embeddings + position_embeddings)

        # Transformer Blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        
        # We only care about the prediction from the LAST token (current state)
        # But during training we might want all. For now, let's return all.
        logits = self.actor_mean(x)
        logstd = self.actor_logstd(x)
        logstd = torch.clamp(logstd, -2, 1)
        values = self.critic(x)

        return logits, logstd, values

    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        """
        Inference Interface.
        Manages the sliding window context buffer efficiently.
        """
        # Update history buffer
        if state.dim() == 1:
            state = state.unsqueeze(0) # (1, state_dim)
        
        # If this is a new episode (how do we know? usually manual reset or check internal logic)
        # For simple interface compatibility, we just append.
        # Ideally user calls reset_memory() between episodes.
        
        if len(self.history_buffer) == 0:
             self.history_buffer = state
        else:
             self.history_buffer = torch.cat([self.history_buffer, state], dim=0)
        
        # Truncate to context length
        if self.history_buffer.size(0) > self.context_len:
            self.history_buffer = self.history_buffer[-self.context_len:, :]
            
        # Add batch dim for forward
        context = self.history_buffer.unsqueeze(0) # (1, T, state_dim)
        
        # Forward pass
        action_mean, logstd, _ = self.forward(context)
        
        # Take last step
        action_mean = action_mean[:, -1, :] # (1, action_dim)
        logstd = logstd[:, -1, :]           # (1, action_dim)
        
        if deterministic:
            return action_mean, None, None
            
        action_std = torch.exp(logstd)
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return action, log_prob, entropy

    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor):
        # Standard PPO updates often pass uncorrelated batches of states.
        # For a Transformer baseline in this setting, we treat the input 'state' 
        # as a sequence of length 1 (using the immediate observation).
        # This allows the Transformer architecture to be evaluated for its 
        # spatial reasoning capabilities even if temporal range is limited during update.
        
        if state.dim() == 2:
            state = state.unsqueeze(1) # (B, 1, D)
            
        action_mean, logstd, value = self.forward(state)
        
        # If we have sequence, we might want to align actions.
        # Assuming PPO passed aligned inputs.
        
        action_mean = action_mean[:, -1, :] # Take last if sequence
        logstd = logstd[:, -1, :]           # Match
        value = value[:, -1, :]
        
        action_std = torch.exp(logstd)
        dist = torch.distributions.Normal(action_mean, action_std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return value, log_prob, entropy
        
    def reset_memory(self):
        self.history_buffer = []


class Block(nn.Module):
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
    def __init__(self, hidden_dim, n_head, context_len, dropout):
        super().__init__()
        assert hidden_dim % n_head == 0
        self.c_attn = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.c_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.hidden_dim = hidden_dim
        
        # Causal mask (lower triangular)
        self.register_buffer("bias", torch.tril(torch.ones(context_len, context_len))
                                     .view(1, 1, context_len, context_len))

    def forward(self, x):
        B, T, C = x.size() # batch, sequence length, embedding dimensionality
        
        # Q, K, V
        q, k, v  = self.c_attn(x).split(self.hidden_dim, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # Causal Attention
        # (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # Mask future tokens
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        
        return self.resid_dropout(self.c_proj(y))
