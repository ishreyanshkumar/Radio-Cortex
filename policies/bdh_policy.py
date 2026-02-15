"""
BDH Policy Adapter for Radio-Cortex — Cell-Centric Architecture

Processes a sequence of 'Enriched Cell Tokens' (12 features each).
Produces cell-level actions only (TxPower, SchedulerWeight).
State and action spaces are SCALE-INVARIANT: the model sees only cells,
regardless of whether there are 20 or 2,000 UEs in the simulation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from typing import Optional

from . import bdh as bdh_mod


class BDHPolicy(nn.Module):
    """
    Cell-Centric BDH Policy.

    State:  (B, num_cells * 12)  — Enriched Cell Tokens
    Action: (B, num_cells * 5)   — [TxPower, SchedulerWeight, Hysteresis, MacDelay, MaxHarq] per cell
    """

    def __init__(self, state_dim: int, action_dim: int, bdh_config: Optional[object] = None, device: str = 'cpu', env_config: Optional[object] = None):
        super().__init__()

        # --- Layout ---
        self.cell_features = 12   # Must match env features_per_cell
        self.cell_actions = 5     # TxPower, SchedulerWeight, Hysteresis, MacDelay, MaxHarq
        self.num_cells = getattr(env_config, 'num_cells', 3) if env_config else 3

        # --- BDH Core ---
        if bdh_config is None:
            cfg = bdh_mod.BDHConfig(
                n_layer=4,
                n_embd=256,
                n_head=4,
                mlp_internal_dim_multiplier=32,
                vocab_size=256
            )
        else:
            cfg = bdh_config

        self.device = device
        self.bdh = bdh_mod.BDH(cfg).to(device)
        self.state = None
        emb_dim = cfg.n_embd

        # --- Cell Encoder ---
        self.cell_encoder = nn.Sequential(
            nn.Linear(self.cell_features, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )

        # --- Cell Action Head ---
        self.cell_action_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, self.cell_actions)
        )
        self.cell_logstd_head = nn.Linear(emb_dim, self.cell_actions)

        # --- Global Value Head ---
        self.value_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, 1)
        )

    # ── Tokenization ──────────────────────────────────────────────

    def _process_state(self, state: torch.Tensor) -> torch.Tensor:
        """Reshape flat state → (B, num_cells, cell_features) and encode."""
        B = state.size(0)
        M = self.num_cells
        # Reshape: [B, M*12] → [B, M, 12]
        x = state.view(B, M, self.cell_features)
        return self.cell_encoder(x)  # (B, M, emb_dim)

    # ── BDH Transformer Stack ─────────────────────────────────────

    def _bdh_block(self, x: torch.Tensor) -> torch.Tensor:
        """Single BDH layer block (checkpointing-compatible)."""
        C = self.bdh.config
        B, _, T, D = x.size()
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh

        x_latent = x @ self.bdh.encoder
        x_sparse = F.relu(x_latent)

        # Scale-Free Bidirectional Attention (cells talk to cells)
        yKV = self.bdh.attn(Q=x_sparse, K=x_sparse, V=x, causal=False)
        yKV = self.bdh.ln(yKV)

        y_latent = yKV @ self.bdh.encoder_v
        y_sparse = F.relu(y_latent)

        xy_sparse = x_sparse * y_sparse
        xy_sparse = self.bdh.drop(xy_sparse)

        yMLP = torch.einsum('bhtn,hnd->btd', xy_sparse, self.bdh.decoder.view(nh, N, D)).unsqueeze(1)

        y = self.bdh.ln(yMLP)
        return self.bdh.ln(x + y)

    def _bdh_layer_stack(self, x_input: torch.Tensor) -> torch.Tensor:
        """Run the BDH layer stack. x_input: (B, T, D)"""
        C = self.bdh.config
        x = x_input.unsqueeze(1)  # (B, 1, T, D)
        x = self.bdh.ln(x)

        for _level in range(C.n_layer):
            if self.training and x.requires_grad:
                x = checkpoint.checkpoint(self._bdh_block, x, use_reentrant=False)
            else:
                x = self._bdh_block(x)

        return x.squeeze(1)  # (B, T, D)

    # ── Forward ───────────────────────────────────────────────────

    def _forward_common(self, state: torch.Tensor):
        B = state.size(0)

        # 1. Tokenize cells
        cell_tokens = self._process_state(state)  # (B, M, D)

        # 2. Contextualize (cells attend to each other)
        context = self._bdh_layer_stack(cell_tokens)  # (B, M, D)

        # 3. Decode cell actions
        action_mean = self.cell_action_head(context)   # (B, M, 2)
        logstd = self.cell_logstd_head(context)        # (B, M, 2)

        flat_mean = action_mean.reshape(B, -1)         # (B, M*2)
        flat_logstd = logstd.reshape(B, -1)
        flat_logstd = torch.clamp(flat_logstd, -2, 1)

        # 4. Global value from mean-pooled context
        global_pool = context.mean(dim=1)              # (B, D)
        value = self.value_head(global_pool)            # (B, 1)

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
        self.state = None
