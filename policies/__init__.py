"""
Radio-Cortex Policy Architectures

This package contains all neural network policy definitions for the RL agent.
Supports multiple architectures: MLP, BDH Transformer, and custom variants.
"""

from .neural_networks import ActorCritic
from .bdh_policy import BDHPolicy


def get_policy(model_type: str, num_ues: int, num_cells: int, device: str = 'cpu', **kwargs):
    """
    Factory function: returns the correct policy instance for a given model_type string.
    
    Args:
        model_type: One of 'bdh', 'nn', 'gpt2', 'trxl', 'linear', 'universal', 'reformer'.
        num_ues:    Number of UEs (used for some architectures).
        num_cells:  Number of cells.
        device:     Compute device string (e.g., 'cpu', 'cuda').
    """
    # Standard dimensions shared across all policies
    state_dim  = num_cells * 16 * 3   # 16 features * 3 stacked frames
    action_dim = num_cells * 3        # 3 actions per cell

    if model_type == 'bdh':
        env_config = kwargs.get('env_config', None)
        return BDHPolicy(state_dim, action_dim, device=device, env_config=env_config)
    elif model_type == 'gpt2':
        from .policy_gpt2 import GPT2Policy
        return GPT2Policy(state_dim, action_dim, device=device)
    elif model_type == 'trxl':
        from .policy_trxl import TrXLPolicy
        return TrXLPolicy(state_dim, action_dim, device=device)
    elif model_type == 'linear':
        from .policy_linear import LinearPolicy
        return LinearPolicy(state_dim, action_dim, device=device)
    elif model_type == 'universal':
        from .policy_universal import UniversalPolicy
        return UniversalPolicy(state_dim, action_dim, device=device)
    elif model_type == 'reformer':
        from .policy_reformer import ReformerPolicy
        return ReformerPolicy(state_dim, action_dim, device=device)
    else:  # 'nn' or any unknown type → default MLP
        return ActorCritic(state_dim, action_dim)


__all__ = ['ActorCritic', 'BDHPolicy', 'get_policy']
