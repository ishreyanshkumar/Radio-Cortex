"""
Radio-Cortex Vectorized Environment Wrapper

Provides parallel environment execution using:
- SubprocVecEnv: Run multiple ns-3 simulations in separate processes
- VecNormalize: Normalize observations and rewards for stable training

Each parallel environment uses isolated Kafka topics (e.g., e2_kpm_stream_0, e2_kpm_stream_1)
to prevent message crosstalk between simulations.
"""

import copy
import numpy as np
import traceback
from typing import Callable, List, Optional, Dict, Any

# Use stable-baselines3 vectorized environment utilities
try:
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
    from stable_baselines3.common.vec_env.base_vec_env import VecEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("[vec_env_wrapper] Warning: stable-baselines3 not installed. VecEnv features disabled.")

from oran_ns3_env import ORANns3Env, NS3Config


def make_env(config: NS3Config, env_id: int, seed: Optional[int] = None) -> Callable[[], ORANns3Env]:
    """
    Factory function to create an environment with isolated Kafka topics.
    
    Args:
        config: Base NS3Config to copy
        env_id: Environment index for topic isolation
        seed: Optional random seed base
        
    Returns:
        Callable that creates the configured environment
    """
    def _init() -> ORANns3Env:
        # Deep copy config to avoid shared state
        cfg = copy.deepcopy(config)
        try:
            # Set unique topic suffix for this environment
            cfg.topic_suffix = f"_{env_id}"
            # Assuming kpm_interval_ms might be set in the base config,
            # if not, it would default to NS3Config's default or be set here.
            # For now, we'll just ensure topic_suffix is set.
            
            env = ORANns3Env(cfg)
            if seed is not None:
                env.reset(seed=seed + env_id) # Use env_id as rank for seeding
            else:
                # Use config seed + env_id to ensure diversity
                base_seed = getattr(config, 'seed', 42)
                env.reset(seed=base_seed + env_id)
            return env
        except Exception as e:
            # Fatal error during initialization
            fail_msg = f"Worker {env_id} failed to initialize: {str(e)}\n{traceback.format_exc()}"
            print(fail_msg)
            with open(f"worker_failure_{env_id}.log", "w") as f:
                f.write(fail_msg)
            raise # Re-raise the exception after logging
    
    return _init


def make_vec_env(
    config: NS3Config,
    n_envs: int = 4,
    normalize_obs: bool = True,
    normalize_reward: bool = True,
    use_subprocess: bool = True,
    seed: Optional[int] = None, # Added seed parameter
) -> VecEnv:
    """
    Create a vectorized environment with parallel execution and normalization.
    
    Args:
        config: Base NS3Config for all environments
        n_envs: Number of parallel environments
        normalize_obs: Whether to normalize observations
        normalize_reward: Whether to normalize rewards
        use_subprocess: Use SubprocVecEnv (True) or DummyVecEnv (False)
        
    Returns:
        VecNormalize-wrapped vectorized environment
    """
    if not SB3_AVAILABLE:
        raise ImportError(
            "stable-baselines3 is required for vectorized environments. "
            "Install with: pip install stable-baselines3"
        )
    
    # Create environment factory functions with unique topic suffixes
    env_fns = [make_env(config, i, seed) for i in range(n_envs)]
    
    # Create vectorized environment
    if use_subprocess and n_envs > 1:
        print(f"[VecEnv] Creating SubprocVecEnv with {n_envs} parallel environments")
        # Use 'spawn' to avoid CUDA initialization deadlocks in forked processes
        vec_env = SubprocVecEnv(env_fns, start_method='spawn')
    else:
        print(f"[VecEnv] Creating DummyVecEnv with {n_envs} sequential environments")
        vec_env = DummyVecEnv(env_fns)
    
    # Wrap with VecNormalize for observation/reward normalization
    if normalize_obs or normalize_reward:
        print(f"[VecEnv] Adding VecNormalize (obs={normalize_obs}, reward={normalize_reward})")
        vec_env = VecNormalize(
            vec_env,
            norm_obs=normalize_obs,
            norm_reward=normalize_reward,
            clip_obs=10.0,       # Clip normalized observations
            clip_reward=10.0,   # Clip normalized rewards
            gamma=0.99,          # Discount for running reward std
        )
    
    return vec_env


def save_vec_normalize(vec_env: VecEnv, path: str) -> None:
    """
    Save VecNormalize statistics to a file.
    
    Args:
        vec_env: The vectorized environment (must contain VecNormalize)
        path: Path to save normalization statistics
    """
    if not SB3_AVAILABLE:
        return
    
    # Extract VecNormalize from wrapper chain
    env = vec_env
    while env is not None:
        if isinstance(env, VecNormalize):
            env.save(path)
            print(f"[VecEnv] Saved VecNormalize stats to {path}")
            return
        env = getattr(env, 'venv', None)
    print("[VecEnv] Warning: No VecNormalize found in environment chain")


def load_vec_normalize(vec_env: VecEnv, path: str) -> None:
    """
    Load VecNormalize statistics from a file.
    
    Args:
        vec_env: The vectorized environment (must contain VecNormalize)
        path: Path to load normalization statistics from
    """
    if not SB3_AVAILABLE:
        return
    
    import os
    if not os.path.exists(path):
        print(f"[VecEnv] Warning: VecNormalize stats file not found: {path}")
        return
    
    # Extract VecNormalize from wrapper chain
    env = vec_env
    while env is not None:
        if isinstance(env, VecNormalize):
            # Load from file using stable-baselines3's method
            loaded = VecNormalize.load(path, vec_env.venv)
            # Copy statistics
            env.obs_rms = loaded.obs_rms
            env.ret_rms = loaded.ret_rms
            print(f"[VecEnv] Loaded VecNormalize stats from {path}")
            return
        env = getattr(env, 'venv', None)
    print("[VecEnv] Warning: No VecNormalize found in environment chain")


# ============================================================================
# Utility for non-vectorized single environment (for backward compatibility)
# ============================================================================

def create_single_env(config: NS3Config, env_id: int = 0) -> ORANns3Env:
    """
    Create a single environment with topic isolation support.
    
    Args:
        config: NS3Config
        env_id: Environment ID for topic suffix (default 0)
        
    Returns:
        ORANns3Env instance
    """
    cfg = copy.deepcopy(config)
    cfg.topic_suffix = f"_{env_id}" if env_id != 0 else ""
    return ORANns3Env(cfg)


if __name__ == "__main__":
    # Simple test
    print("Testing vec_env_wrapper...")
    
    if SB3_AVAILABLE:
        config = NS3Config(num_ues=5, num_cells=2, sim_time=2.0)
        print(f"Base config: {config}")
        
        # Test factory function
        factory = make_env(config, 0)
        print(f"Factory created for env_id=0, topic_suffix will be '_0'")
        
        print("\n✓ vec_env_wrapper module loaded successfully")
    else:
        print("stable-baselines3 not available, skipping vector env tests")
