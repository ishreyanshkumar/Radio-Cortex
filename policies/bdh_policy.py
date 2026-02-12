"""
BDH Policy Adapter for Radio-Cortex
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
from typing import Optional
import math

# --- MONKEY PATCH START ---
# We must patch the Attention.forward method to remove the causal mask (.tril)
# so that Cells and UEs can see each other (Bidirectional Attention).

def bidirectional_attention_forward(self, Q, K, V):
    # Standard checks from original code
    assert self.freqs.dtype == torch.float32
    assert K is Q
    _, _, T, _ = Q.size()

    # Re-implement RoPE logic from bdh.py
    r_phases = (
        torch.arange(0, T, device=self.freqs.device, dtype=self.freqs.dtype)
        .view(1, 1, -1, 1)
    ) * self.freqs
    
    # We call the static method from the class (self.__class__) 
    # or rely on the instance method if bound correctly.
    # Safe way: use the original class's static method
    QR = self.rope(r_phases, Q) 
    KR = QR

    # --- THE FIX: No .tril(diagonal=-1) ---
    # Original: scores = (QR @ KR.mT).tril(diagonal=-1)
    # New (Bidirectional):
    scores = (QR @ KR.mT)
    
    return scores @ V

# Apply the patch immediately after importing bdh
from . import bdh as bdh_mod
bdh_mod.Attention.forward = bidirectional_attention_forward
# --- MONKEY PATCH END ---

class BDHPolicy(nn.Module):
    """
    Minimal adapter that allows a `bdh.BDH` model to be used as a policy.

    This wrapper converts the input state vector into a sequence of UEs and Cells (Scale-Free)
    which are embedded via shared encoders and processed by the BDH core model.
    It produces actions for both Cells and UEs using separate heads.
    """

    def __init__(self, state_dim: int, action_dim: int, bdh_config: Optional[object] = None, device: str = 'cpu', env_config: Optional[dict] = None):
        super().__init__()
        from . import bdh as bdh_mod
        
        # --- Config & Dimensions ---
        # Defaults based on oran-congestion-scenario
        self.ue_features = 12 
        self.num_ues = 20
        self.ue_actions = 1
        
        self.num_cells = 3
        self.cell_features = 5
        self.cell_actions = 7
        
        if env_config:
            self.num_ues = getattr(env_config, 'numUes', self.num_ues) or self.num_ues
            self.num_cells = getattr(env_config, 'numCells', self.num_cells) or self.num_cells
            
        # Instantiate BDH core model
        # bdh.py is immutable, we use it as a component store
        # Restore original config but keep a sane multiplier for this hardware
        # Original: multiplier=128 (Huge). 
        # Compromise: multiplier=64 + Checkpointing should fit.
        # Layer count=6 (Original)
        if bdh_config is None:
            cfg = bdh_mod.BDHConfig(
                n_layer=6, 
                n_embd=256, 
                n_head=4, 
                mlp_internal_dim_multiplier=128, # Restored to 128 (Paper Default)
                vocab_size=256 
            )
        else:
            cfg = bdh_config
            
        self.device = device
        self.bdh = bdh_mod.BDH(cfg).to(device)
        self.state = None 
        
        emb_dim = cfg.n_embd
        
        # --- Heterogeneous Scale-Free Components ---
        
        # 1. Encoders (State -> Latent)
        self.ue_encoder = nn.Sequential(
            nn.Linear(self.ue_features, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )
        self.cell_encoder = nn.Sequential(
            nn.Linear(self.cell_features, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )
        
        # 2. Action Heads (Latent -> Action)
        self.ue_action_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, self.ue_actions)
        )
        self.cell_action_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, self.cell_actions)
        )
        
        # Logstd heads
        self.ue_logstd_head = nn.Linear(emb_dim, self.ue_actions)
        self.cell_logstd_head = nn.Linear(emb_dim, self.cell_actions)
        
        # Value head (Global)
        self.value_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, 1)
        )

    def _process_state_to_tokens(self, state):
        """
        Convert flat state -> (UE_Tokens, Cell_Tokens)
        State Layout: [UE1..N, Cell1..M]
        """
        B = state.size(0)
        
        # Slicing indices
        end_ue = self.num_ues * self.ue_features
        end_cell = end_ue + self.num_cells * self.cell_features
        
        limit = state.shape[1]
        
        # Extract UE Part
        if limit < end_ue:
            actual_ues = limit // self.ue_features
            ue_part = state[:, :actual_ues * self.ue_features]
            ue_tokens = ue_part.view(B, actual_ues, self.ue_features)
            cell_tokens = torch.zeros(B, self.num_cells, self.cell_features, device=state.device)
        else:
            ue_part = state[:, :end_ue]
            ue_tokens = ue_part.view(B, self.num_ues, self.ue_features)
            
            # Extract Cell Part
            if limit < end_cell:
                 cell_part = state[:, end_ue:limit]
                 actual_cells = (limit - end_ue) // self.cell_features
                 if actual_cells > 0:
                      cell_tokens = cell_part[:, :actual_cells * self.cell_features].view(B, actual_cells, self.cell_features)
                      if actual_cells < self.num_cells:
                           pad = torch.zeros(B, self.num_cells - actual_cells, self.cell_features, device=state.device)
                           cell_tokens = torch.cat([cell_tokens, pad], dim=1)
                 else:
                      cell_tokens = torch.zeros(B, self.num_cells, self.cell_features, device=state.device)
            else:
                 cell_part = state[:, end_ue:end_cell]
                 cell_tokens = cell_part.view(B, self.num_cells, self.cell_features)
        
        # Encode
        ue_emb = self.ue_encoder(ue_tokens)       # (B, N_UE, D)
        cell_emb = self.cell_encoder(cell_tokens) # (B, N_Cell, D)
        
        return ue_emb, cell_emb

    def _bdh_block(self, x):
        """Single BDH layer block for checkpointing."""
        C = self.bdh.config
        B, _, T, D = x.size()
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh

        x_latent = x @ self.bdh.encoder
        x_sparse = F.relu(x_latent) 
        
        yKV = self.bdh.attn(Q=x_sparse, K=x_sparse, V=x)
        yKV = self.bdh.ln(yKV)
        
        y_latent = yKV @ self.bdh.encoder_v
        y_sparse = F.relu(y_latent)
        
        xy_sparse = x_sparse * y_sparse
        xy_sparse = self.bdh.drop(xy_sparse)
        
        yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.bdh.decoder
        
        y = self.bdh.ln(yMLP)
        out = self.bdh.ln(x + y)
        return out

    def _bdh_layer_stack(self, x_input, states=None):
        """
        Run the BDH layer stack manually with checkpointing.
        x_input: (B, T, D)
        """
        import torch.utils.checkpoint as checkpoint
        
        C = self.bdh.config
        
        x = x_input.unsqueeze(1) # (B, 1, T, D)
        x = self.bdh.ln(x)
        
        for level in range(C.n_layer):
            # Gradient Checkpointing: Trades compute for memory
            # Calculates forward pass 2x, but stores only input for backward
            if self.training and x.requires_grad:
                x = checkpoint.checkpoint(self._bdh_block, x, use_reentrant=False)
            else:
                x = self._bdh_block(x)
            
        return x.squeeze(1)

    def _forward_common(self, state):
        B = state.size(0)
        # 1. Tokenize
        ue_emb, cell_emb = self._process_state_to_tokens(state)
        
        # 2. Concat: [Cells, UEs] Order
        all_tokens = torch.cat([cell_emb, ue_emb], dim=1) # (B, N_Cell + N_UE, D)
        
        # 3. Process
        output_emb = self._bdh_layer_stack(all_tokens)
        
        # 4. Split
        # Cells were first
        out_cell = output_emb[:, :self.num_cells, :]
        out_ue = output_emb[:, self.num_cells:, :]
        
        # 5. Decode Actions
        act_cell = self.cell_action_head(out_cell)  # (B, 3, 7)
        logstd_cell = self.cell_logstd_head(out_cell)
        
        act_ue = self.ue_action_head(out_ue)        # (B, 20, 1)
        logstd_ue = self.ue_logstd_head(out_ue)
        
        # 6. Concat for Flat Action Vector: [CellActions, UEActions]
        flat_cell = act_cell.reshape(B, -1)
        flat_logstd_cell = logstd_cell.reshape(B, -1)
        
        flat_ue = act_ue.reshape(B, -1)
        flat_logstd_ue = logstd_ue.reshape(B, -1)
        
        action_mean = torch.cat([flat_cell, flat_ue], dim=1)
        logstd = torch.cat([flat_logstd_cell, flat_logstd_ue], dim=1)
        logstd = torch.clamp(logstd, -2, 1)
        
        # 7. Value (Mean over all entities)
        global_pool = output_emb.mean(dim=1)
        value = self.value_head(global_pool)
        
        return action_mean, logstd, value

    def forward(self, state: torch.Tensor):
        if state.dim() == 1: state = state.unsqueeze(0)
        return self._forward_common(state)

    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor):
        if state.dim() == 1: state = state.unsqueeze(0)
        action_mean, logstd, value = self._forward_common(state)
        
        action_std = torch.exp(logstd)
        dist = torch.distributions.Normal(action_mean, action_std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return value, log_prob, entropy

    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        if state.dim() == 1: state = state.unsqueeze(0)
        
        # Use Parallel forward for inference
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
