"""
Transformer Policy 2 - Transformer-XL (Recurrent Memory)
Serves as the "Memory" baseline (GTrXL) to demonstrate explicit KV-caching vs BDH state.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TrXLPolicy(nn.Module):
    """
    Recurrent Transformer (Transformer-XL / GTrXL Style).
    
    Architecture:
    - Relative Positional Encodings
    - Segment-Level Recurrence (KV Cache)
    - Memory: Maintains state across steps
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, 
                 n_layer: int = 4, n_head: int = 4, mem_len: int = 64, 
                 dropout: float = 0.1, device: str = 'cpu'):
        super().__init__()
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.mem_len = mem_len
        self.n_layer = n_layer

        # 1. Embeddings
        self.state_embed = nn.Linear(state_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)

        # 2. Transformer Blocks
        self.blocks = nn.ModuleList([
            MemTransformerBlock(hidden_dim, n_head, dropout) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(hidden_dim)

        # 3. Heads
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        self.critic = nn.Linear(hidden_dim, 1)

        self.apply(self._init_weights)

        # Memory buffer (List of Tensors, one per layer)
        self.memory = None
        self.reset_memory()

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def reset_memory(self):
        # Initialize memory as None (will be created on first forward)
        # or zeros. None is safer to detect start.
        self.memory = [None] * self.n_layer

    def forward(self, states, mems=None):
        """
        states: (T, B, state_dim) -- Note: XL usually prefers T-first
        mems: List of (M, B, H) tensors from previous segment
        """
        # Handle 2D input (B, D) from standard PPO
        if states.dim() == 2:
            states = states.unsqueeze(1) # (B, 1, D)

        # Ensure T-first for XL logic, or handle B-first carefully.
        # Let's stick to B-first (Batch, Time, Dim) for consistency with PyTorch
        B, T, _ = states.size()
        
        x = self.state_embed(states)
        x = self.drop(x)
        
        new_mems = []
        
        for i, block in enumerate(self.blocks):
            mem = mems[i] if mems else None
            x, new_mem = block(x, mem=mem)
            new_mems.append(new_mem.detach()) # Stop gradient flow to deep history (TBPTT)
            
        x = self.ln_f(x)
        
        logits = self.actor_mean(x)
        values = self.critic(x)
        
        if mems is None:
             return logits, self.actor_logstd, values
             
        return logits, self.actor_logstd, values, new_mems

    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        """
        Inference Interface (Step-by-Step).
        """
        if state.dim() == 1:
            state = state.unsqueeze(0).unsqueeze(0) # (1, 1, D)
        elif state.dim() == 2:
            state = state.unsqueeze(1) # (B, 1, D)
            
        # Use and update internal memory
        # We assume self.memory is a list (even if contents are None), so forward returns 4 values
        action_mean, logstd, _, new_mems = self.forward(state, self.memory)
        
        # Simple FIFO memory update
        # In real XL, we keep M tokens. Here we just swap.
        # The block logic handles the concatenation [Mem, Input].
        # We need to keep the output mems which represent the cached KV of THIS step
        # to stand in for the previous context for the NEXT step.
        self.memory = new_mems
        
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
        # We treat the update as stateless (memory=None) to be compatible with 
        # standard PPO batches which are not temporally ordered.
        # This evaluates the spatial attention mechanism.
        
        if state.dim() == 2:
            state = state.unsqueeze(1)
            
        # Pass None memory -> behaves like standard Transformer locally
        # Returns 3 values now
        action_mean, logstd, value = self.forward(state, mems=None)
        
        action_mean = action_mean[:, -1, :]
        value = value[:, -1, :]
        
        action_std = torch.exp(logstd)
        dist = torch.distributions.Normal(action_mean, action_std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return value, log_prob, entropy


class MemTransformerBlock(nn.Module):
    def __init__(self, hidden_dim, n_head, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.attn = RelPartialLearnableSelfAttention(hidden_dim, n_head, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, mem=None):
        # x: (B, T, H)
        # mem: (B, M, H)
        
        # Create context by concatenating memory and input (along time)
        # Context = [Memory, Input]
        if mem is not None:
            # We must ensure mem and x match batch size, which they should.
            c = torch.cat([mem, x], dim=1)
        else:
            c = x
            
        # Attention with relative pos
        # We only want output for the 'x' part (queries), but keys/values come from 'c'.
        attn_out = self.attn(queries=self.ln1(x), context=self.ln1(c))
        
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        
        # New memory is just x (or a window of C).
        # For simple recurrence, we pass 'x' as the memory for input to next step.
        return x, x.detach()

class RelPartialLearnableSelfAttention(nn.Module):
    # Simplified Relative Attention
    def __init__(self, hidden_dim, n_head, dropout):
        super().__init__()
        self.n_head = n_head
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // n_head
        
        self.qkv_net = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.drop = nn.Dropout(dropout)
        self.o_net = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        self.scale = 1 / math.sqrt(self.head_dim)

    def forward(self, queries, context):
        B, T, H = queries.shape
        _, S, _ = context.shape # S = M + T
        
        # Generate Q, K, V
        # Ideally logic is distinct for context but simplified here
        # We project queries from 'queries', K/V from 'context'
        
        q = self.qkv_net(queries).chunk(3, dim=-1)[0] # Only need Q part
        k, v = self.qkv_net(context).chunk(3, dim=-1)[1:] # Need K, V parts
        
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.n_head, self.head_dim).transpose(1, 2)
        
        # Content score
        content_score = torch.matmul(q, k.transpose(-1, -2))
        
        # (Simplified: ignoring explicit relative pos bias term for brevity. 
        # In full XL this adds R matrix. Here we rely on causal masking over context)
        
        # Causal Masking:
        # We need to mask positions where query sees future.
        # Since Context = [Mem, Input], and Query = Input
        # position i in Query corresponds to position M+i in Context.
        # It can attend to anything < M+i.
        
        # Matrix shape: (T, S) i.e. (T, M+T)
        # The top-left (T x M) is full attention (attending to past).
        # The right (T x T) is lower triangular causal.
        
        M = S - T
        mask = torch.ones(T, S, device=queries.device)
        mask_causal = torch.tril(torch.ones(T, T, device=queries.device))
        mask[:, M:] = mask_causal
        
        content_score = content_score * self.scale
        content_score = content_score.masked_fill(mask == 0, float('-inf'))
        
        probs = F.softmax(content_score, dim=-1)
        probs = self.drop(probs)
        
        out = torch.matmul(probs, v)
        out = out.transpose(1, 2).reshape(B, T, H)
        
        return self.o_net(out)
