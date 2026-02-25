"""
Radio-Cortex Policy Architectures

This package contains all neural network policy definitions for the RL agent.
Supports multiple architectures: MLP, BDH Transformer, and custom variants.
"""

from .neural_networks import ActorCritic
from .bdh_policy import BDHPolicy

def get_policy(model_type, num_ues, num_cells, hidden_dim=256, device='cpu', env_config=None):
    """
    Factory function to retrieve and initialize a policy by name.
    
    Args:
        model_type: String identifier (bdh, gpt2, trxl, linear, universal, reformer, nn)
        num_ues: Number of User Equipments (used to derive state dim)
        num_cells: Number of cells (used to derive state/action dim)
        hidden_dim: Hidden layer size (for MLP)
        device: 'cpu' or 'cuda'
        env_config: Optional configuration object for specialized policies (e.g. BDH)
        
    Returns:
        Initialized nn.Module policy
    """
    # Calculate dimensions based on the system architecture
    # State: num_cells * 16 (features) * 3 (temporal frames)
    state_dim = num_cells * 16 * 3 
    # Action: num_cells * 3 (TxPower, CIO, TTT)
    action_dim = num_cells * 3
        
    if model_type == 'bdh':
        from .bdh_policy import BDHPolicy
        return BDHPolicy(state_dim, action_dim, device=device, env_config=env_config).to(device)
    elif model_type == 'gpt2':
        from .policy_gpt2 import GPT2Policy
        return GPT2Policy(state_dim, action_dim, device=device).to(device)
    elif model_type == 'trxl':
        from .policy_trxl import TrXLPolicy
        return TrXLPolicy(state_dim, action_dim, device=device).to(device)
    elif model_type == 'linear':
        from .policy_linear import LinearPolicy
        return LinearPolicy(state_dim, action_dim, device=device).to(device)
    elif model_type == 'universal':
        from .policy_universal import UniversalPolicy
        return UniversalPolicy(state_dim, action_dim, device=device).to(device)
    elif model_type == 'reformer':
        from .policy_reformer import ReformerPolicy
        return ReformerPolicy(state_dim, action_dim, device=device).to(device)
    else:
        # Default to Neural Network (MLP)
        from .neural_networks import ActorCritic
        return ActorCritic(state_dim, action_dim, hidden_dim).to(device)

__all__ = ['ActorCritic', 'BDHPolicy', 'get_policy']
