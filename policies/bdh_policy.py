"""
BDH Policy Adapter for Radio-Cortex
Wraps the official immutable bdh.py
Maps temporal frame-stacking to the BDH sequence dimension (T).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

from . import bdh as bdh_mod

class BDHPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, bdh_config: Optional[object] = None, device: str = 'cpu', env_config: Optional[object] = None):
        super().__init__()

        # --- Layout ---
        self.num_cells = getattr(env_config, 'num_cells', 3) if env_config else 3
        self.cell_features = 16   # Base features per cell
        self.n_stack = 3          # Temporal frames from env
        self.cell_actions = 3     # TxPower, CIO, TTT

        # The full network state per frame
        self.frame_dim = self.num_cells * self.cell_features 

        if bdh_config is None:
            cfg = bdh_mod.BDHConfig(
                n_layer=4,
                n_embd=128,
                n_head=4,
                dropout=0.0, # Disable dropout for PPO
                mlp_internal_dim_multiplier=128, # Match official BDH width
                vocab_size=256
            )
        else:
            cfg = bdh_config

        self.device = device
        self.bdh = bdh_mod.BDH(cfg).to(device)
        emb_dim = cfg.n_embd

        # --- Temporal Frame Encoder ---
        # Encodes the entire network state of ONE timestep into the BDH dimension
        self.frame_encoder = nn.Sequential(
            nn.Linear(self.frame_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim)
        )

        # --- Action & Value Heads ---
        self.action_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, self.num_cells * self.cell_actions)
        )
        self.logstd_head = nn.Parameter(torch.full((1, self.num_cells * self.cell_actions), -0.5))

        self.value_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, 1)
        )
        
        self.apply(self._init_weights)
        self._apply_specialized_init()

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            if module.weight is not None:
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def _apply_specialized_init(self):
        # Action head should start with near-zero weights (gain=0.01) to prevent action saturation
        if hasattr(self, 'action_head') and isinstance(self.action_head[-1], nn.Linear):
            nn.init.orthogonal_(self.action_head[-1].weight, gain=0.01)
        # Value head should start with gain=1.0
        if hasattr(self, 'value_head') and isinstance(self.value_head[-1], nn.Linear):
            nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0)

    # ── Tokenization ──────────────────────────────────────────────
    def _process_state(self, state: torch.Tensor) -> torch.Tensor:
        """Reshape stacked state → (B, T, frame_dim) and encode."""
        B = state.size(0)
        T = self.n_stack
        
        # Reshape [B, T * frame_dim] -> [B, T, frame_dim]
        x = state.view(B, T, self.frame_dim)
        return self.frame_encoder(x)  # (B, T, emb_dim)

    # ── BDH Transformer Stack ─────────────────────────────────────
    def _bdh_layer_stack(self, x_input: torch.Tensor) -> torch.Tensor:
        """
        Runs the BDH logic using the official layers.
        x_input is (B, T, D) where T is time.
        """
        C = self.bdh.config
        B, T, D = x_input.size()
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh

        x = x_input.unsqueeze(1)  # (B, 1, T, D)
        x = self.bdh.ln(x)

        for level in range(C.n_layer):
            x_latent = x @ self.bdh.encoder
            x_sparse = F.relu(x_latent)

            # Uses official causal attention over T!
            yKV = self.bdh.attn(
                Q=x_sparse,
                K=x_sparse,
                V=x
            )
            yKV = self.bdh.ln(yKV)

            y_latent = yKV @ self.bdh.encoder_v
            y_sparse = F.relu(y_latent)
            
            xy_sparse = x_sparse * y_sparse
            xy_sparse = self.bdh.drop(xy_sparse)

            yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.bdh.decoder
            
            y = self.bdh.ln(yMLP)
            x = self.bdh.ln(x + y)

        return x.squeeze(1)  # (B, T, D)

    # ── Forward ───────────────────────────────────────────────────
    def _forward_common(self, state: torch.Tensor):
        B = state.size(0)

        # 1. Tokenize temporal frames: T=3
        frame_tokens = self._process_state(state)  # (B, T, D)

        # 2. BDH Processing over Time
        context = self._bdh_layer_stack(frame_tokens)  # (B, T, D)
        
        # 3. We only care about the LATEST timestep to make our current action
        latest_context = context[:, -1, :]  # (B, D)
        
        # 4. Decode cell actions
        flat_mean = self.action_head(latest_context)   # (B, M*3)
        flat_logstd = self.logstd_head.expand(B, -1)   # (B, M*3)
        flat_logstd = torch.clamp(flat_logstd, -5, 2.0)

        # 5. Global value
        value = self.value_head(latest_context)        # (B, 1)

        return flat_mean, flat_logstd, value

    def forward(self, state: torch.Tensor):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return self._forward_common(state)

    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        action_mean, logstd, value = self._forward_common(state)

        action_std = torch.exp(logstd)
        dist = torch.distributions.Normal(action_mean, action_std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return value, log_prob, entropy

    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        if state.dim() == 1:
            state = state.unsqueeze(0)

        action_mean, logstd, value = self._forward_common(state)

        if deterministic:
            return action_mean, None, None

        action_std = torch.exp(logstd)
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return action, log_prob, entropy

    def reset_memory(self):
        pass