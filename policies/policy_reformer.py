"""
Reformer Policy - LONG CONTEXT Baseline
Demonstrates handling of long sequences via buckets/LSH approximation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ReformerPolicy(nn.Module):
    """
    Reformer-Style Policy.
    Uses LSH (Locality Sensitive Hashing) or Bucketed Attention to handle long contexts.
    For this baseline, we implement a simplified Bucketed Attention.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, 
                 n_layer: int = 4, n_head: int = 4, context_len: int = 128, 
                 bucket_size: int = 32, dropout: float = 0.1, device: str = 'cpu'):
        super().__init__()
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.context_len = context_len
        self.bucket_size = bucket_size
        
        # 1. Embeddings
        self.state_embed = nn.Linear(state_dim, hidden_dim)
        # Reformer uses Axial Positional Encodings typically, but learned is fine for baseline
        self.pos_embed = nn.Parameter(torch.zeros(1, context_len, hidden_dim)) 
        self.drop = nn.Dropout(dropout)

        # 2. Transformer Blocks
        self.blocks = nn.ModuleList([
            ReformerBlock(hidden_dim, n_head, bucket_size, dropout) for _ in range(n_layer)
        ])
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
        
        # Ensure T is divisible by bucket_size (pad if needed)
        # For simplicity in this baseline, we might just truncate or pad
        pad_len = (self.bucket_size - (T % self.bucket_size)) % self.bucket_size
        if pad_len > 0:
            states = F.pad(states, (0, 0, 0, pad_len)) # Pad time dim
            
        B, T_padded, _ = states.size()
            
        token_embeddings = self.state_embed(states)
        
        # Pos embeddings (handle padding)
        if T_padded > self.context_len:
             # Truncate to context len if too long
             token_embeddings = token_embeddings[:, :self.context_len, :]
             T_padded = self.context_len
             
        # Naive pos embedding
        # Ideally we slice based on actual steps
        position_embeddings = self.pos_embed[:, :T_padded, :]
        x = self.drop(token_embeddings + position_embeddings)

        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        
        # Slice back to original T if we padded? 
        # Actually for RL next step prediction, the last token is what matters.
        # If we padded, the last token of ORIGINAL input is at index T-1.
        # The padding is at the end.
        if pad_len > 0:
            x = x[:, :T, :]
            
        logits = self.actor_mean(x)
        values = self.critic(x)

        return logits, self.actor_logstd, values

    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        if state.dim() == 1: state = state.unsqueeze(0)
        
        # Update history
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

class ReformerBlock(nn.Module):
    def __init__(self, hidden_dim, n_head, bucket_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.attn = LSHSelfAttention(hidden_dim, n_head, bucket_size, dropout)
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

class LSHSelfAttention(nn.Module):
    """
    Simplified LSH Attention (Bucketed).
    We chunk the sequence into buckets and attend only within bucket + previous bucket.
    This reduces complexity from O(T^2) to O(T * bucket_size).
    """
    def __init__(self, hidden_dim, n_head, bucket_size=32, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.bucket_size = bucket_size
        self.head_dim = hidden_dim // n_head
        
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x):
        B, T, C = x.size()
        
        # 1. Project QKV
        q, k, v = self.qkv(x).split(self.hidden_dim, dim=2)
        
        # 2. Reshape to buckets: (B, num_buckets, bucket_size, dims)
        # We assume T is divisible by bucket_size (padded in policy)
        assert T % self.bucket_size == 0, f"T={T} must be divisible by bucket_size={self.bucket_size}"
        num_buckets = T // self.bucket_size
        
        # Reshape Q, K, V
        # (B, num_buckets, bucket_size, n_head, head_dim)
        def bucketize(tensor):
            return tensor.view(B, num_buckets, self.bucket_size, self.n_head, self.head_dim)
            
        qb = bucketize(q)
        kb = bucketize(k)
        vb = bucketize(v)
        
        # 3. Attend within buckets (and look back one bucket)
        # We iterate over buckets. In a real efficient implementation this is vectorized 
        # by chunking [bucket_i-1, bucket_i].
        
        # Let's vectorize it:
        # Concatenate K, V with shifted version for lookback
        # K_lookback: (B, num_buckets, 2*bucket_size, n_head, head_dim)
        # We roll K and V to get previous bucket.
        
        # This implementation attends to Current + Previous bucket (Lookback 1).
        # This provides a causal window of size 2*bucket_size.
        
        # Construct Key/Value chunks: [Prev_Bucket, Curr_Bucket]
        # Pad beginning with zeros for first bucket
        kb_prev = torch.roll(kb, shifts=1, dims=1)
        kb_prev[:, 0, ...] = 0 # Zero out wrapped around last bucket
        
        vb_prev = torch.roll(vb, shifts=1, dims=1)
        vb_prev[:, 0, ...] = 0
            
        # Concat along bucket_size dimension
        kb_concat = torch.cat([kb_prev, kb], dim=2) # (B, NB, 2*BS, NH, HD)
        vb_concat = torch.cat([vb_prev, vb], dim=2)
        
        # Attention
        # Q: (B, NB, BS, NH, HD)
        # K: (B, NB, 2*BS, NH, HD)
        
        # Transpose for matmul: (B, NB, NH, BS, HD)
        qb_t = qb.permute(0, 1, 3, 2, 4) 
        kb_t = kb_concat.permute(0, 1, 3, 2, 4)
        vb_t = vb_concat.permute(0, 1, 3, 2, 4)
        
        # Score: (B, NB, NH, BS, 2*BS)
        scores = torch.matmul(qb_t, kb_t.transpose(-1, -2)) * self.scale
        
        # Masking
        # Causal mask for the 2*BS size
        # Create mask (BS, 2*BS)
        
        i_idx = torch.arange(self.bucket_size, device=x.device).unsqueeze(1) # (BS, 1)
        j_idx = torch.arange(2 * self.bucket_size, device=x.device).unsqueeze(0) # (1, 2*BS)
        
        # Valid if j < BS (prev bucket, always valid) OR j - BS <= i (current bucket causal)
        mask = (j_idx < self.bucket_size) | ((j_idx - self.bucket_size) <= i_idx)
        
        # Apply mask
        scores = scores.masked_fill(~mask, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.drop(attn)
        
        # Out: (B, NB, NH, BS, 2*BS) @ (B, NB, NH, 2*BS, HD) -> (B, NB, NH, BS, HD)
        out_b = torch.matmul(attn, vb_t)
        
        # Reshape back to (B, T, C)
        out = out_b.permute(0, 1, 3, 2, 4).reshape(B, T, C)
        
        return self.out(out)
