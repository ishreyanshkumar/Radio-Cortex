"""
Linear Transformer Policy - FAST Baseline
Demonstrates O(T) complexity using Kernel-based Attention (Katharopoulos et al.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LinearPolicy(nn.Module):
    """
    Linear Transformer (e.g. Performer / Katharopoulos).
    Uses kernel feature map approximation for Attention.
    Attn(Q, K, V) = (phi(Q) @ (phi(K)^T @ V)) / (phi(Q) @ (phi(K)^T @ 1))
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, 
                 n_layer: int = 4, n_head: int = 4, context_len: int = 64, 
                 dropout: float = 0.1, device: str = 'cpu'):
        super().__init__()
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # 1. Embeddings
        self.state_embed = nn.Linear(state_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, context_len, hidden_dim))
        self.drop = nn.Dropout(dropout)

        # 2. Transformer Blocks
        self.blocks = nn.ModuleList([
            LinearBlock(hidden_dim, n_head, dropout) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(hidden_dim)

        # 3. Heads
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_logstd = nn.Linear(hidden_dim, action_dim)  # State-dependent exploration
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
        # Handle 2D input
        if states.dim() == 2:
            states = states.unsqueeze(1)

        B, T, _ = states.size()
        
        # Simple learnable pos embedding (fixed window style)
        # For a truly linear transformer, we usually want infinite context, 
        # but for this baseline we stick to a window to keep comparison fair on inputs.
        # Alternatively, we could implement RNN-mode inference.
        
        token_embeddings = self.state_embed(states)
        
        # Pos embeddings
        if T > self.pos_embed.size(1):
             # Truncate or fail. For baseline, we assume context_len handles it.
             token_embeddings = token_embeddings[:, -self.pos_embed.size(1):, :]
             T = self.pos_embed.size(1)
             
        position_embeddings = self.pos_embed[:, :T, :]
        x = self.drop(token_embeddings + position_embeddings)

        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        
        logits = self.actor_mean(x)
        logstd = self.actor_logstd(x)
        logstd = torch.clamp(logstd, -2, 1)
        values = self.critic(x)

        return logits, logstd, values

    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        if state.dim() == 1: state = state.unsqueeze(0)
        
        # Recurrent Inference for Linear Transformer
        if state.dim() == 2: state = state.unsqueeze(1) # (1, 1, D)
        
        # Use history buffer for sequence context
        if len(self.history_buffer) == 0:
             self.history_buffer = state
        else:
             self.history_buffer = torch.cat([self.history_buffer, state], dim=1)
        
        # Limit context to avoid infinite memory growth during long episodes
        if self.history_buffer.size(1) > 1024: 
            self.history_buffer = self.history_buffer[:, -1024:, :]
            
        context = self.history_buffer
        
        # Forward pass on full context (O(T) but safe/correct for baseline)
        logits, logstd, _ = self.forward(context)
        
        # Take last step
        action_mean = logits[:, -1, :] 
        logstd_last = logstd[:, -1, :]
        
        if deterministic:
            return action_mean, None, None
            
        action_std = torch.exp(logstd_last)
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy

    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor):
        if state.dim() == 2: state = state.unsqueeze(1)
        
        action_mean, logstd, value = self.forward(state)
        
        # Select last step if sequence, or squeeze if not
        if action_mean.size(1) > 1:
            action_mean = action_mean[:, -1, :]
            logstd = logstd[:, -1, :]
            value = value[:, -1, :]
        else:
            action_mean = action_mean.squeeze(1)
            logstd = logstd.squeeze(1)
            value = value.squeeze(1)
        
        action_std = torch.exp(logstd)
        dist = torch.distributions.Normal(action_mean, action_std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return value, log_prob, entropy

    def reset_memory(self):
        self.history_buffer = []


class LinearBlock(nn.Module):
    def __init__(self, hidden_dim, n_head, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.attn = LinearAttention(hidden_dim, n_head, dropout)
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

class LinearAttention(nn.Module):
    """
    Katharopoulos et al. Linear Attention
    phi(x) = elu(x) + 1
    """
    def __init__(self, hidden_dim, n_head, dropout):
        super().__init__()
        self.n_head = n_head
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // n_head
        
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)
        
    def feature_map(self, x):
        # phi(x) = elu(x) + 1
        return F.elu(x) + 1.0

    def forward(self, x):
        B, T, C = x.size()
        
        q, k, v = self.qkv(x).split(self.hidden_dim, dim=2)
        
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, H, T, D)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        Q = self.feature_map(q)
        K = self.feature_map(k)
        
        # Causal Linear Attention
        # Output_i = (Sum_j<=i Q_i K_j^T V_j) / (Sum_j<=i Q_i K_j^T)
        
        # (B, H, T, D) -> (B, H, T, D)
        
        # Compute K * V^T for each step: (B, H, T, D, 1) * (B, H, T, 1, D) -> (B, H, T, D, D)
        KV = torch.einsum("bhtd,bhte->bhtde", K, v)
        K_sum = K # (B, H, T, D)
        
        # Cumsum along time
        KV_cum = torch.cumsum(KV, dim=2) # (B, H, T, D, D)
        K_cum = torch.cumsum(K, dim=2)   # (B, H, T, D)
        
        # Q @ S
        # Q is (B, H, T, D). Querying against (B, H, T, D, D)
        # We want Q_t @ S_t. (1xD) @ (DxD) -> (1xD)
        
        # (B, H, T, 1, D) @ (B, H, T, D, D) -> (B, H, T, 1, D)
        num = torch.einsum("bhtd,bhtde->bhte", Q, KV_cum)
        
        # Q @ Z
        # (B, H, T, D) dot (B, H, T, D) -> (B, H, T, 1)
        den = torch.einsum("bhtd,bhtd->bht", Q, K_cum).unsqueeze(-1)
        
        out = num / (den + 1e-6)
        
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out(out)
