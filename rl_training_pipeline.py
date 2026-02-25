"""
Radio-Cortex RL Training Pipeline
Integrates O-RAN ns-3 environment with standard RL algorithms
Supports PPO, SAC, TD3 for RAN congestion control
"""

import torch
import torch.nn as nn
import numpy as np
import os
from typing import Dict, List, Optional, Tuple
import gymnasium as gym
import time
from collections import deque
import wandb
import json
from pathlib import Path
from datetime import datetime
import dataclasses
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, SpinnerColumn
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.console import Console, Group
from rich.columns import Columns

# OPTIMIZATION: Prevent PyTorch from starving ns-3 of CPU cycles
torch.set_num_threads(1)


# ============================================================================
# Neural Network Architectures
# ============================================================================

# ============================================================================
# Neural Network Architectures
# ============================================================================

from policies.neural_networks import ActorCritic
from policies.bdh_policy import BDHPolicy


# ============================================================================
# Replay Buffer
# ============================================================================

class ReplayBuffer:
    """Experience replay buffer for off-policy algorithms"""
    
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones)).unsqueeze(1)
        )
    
    def __len__(self):
        return len(self.buffer)


# ============================================================================
# PPO Trainer
# ============================================================================

class PPOTrainer:
    """
    Proximal Policy Optimization trainer
    Stable, robust algorithm for RAN control
    """
    
    def __init__(
        self,        env: gym.Env,
        hidden_dim: int = 256,
        lr: float = 8e-5, # Updated for Stability/Speed balance
        gamma: float = 0.98, # Short-term focus for stability
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.1, # Aligned with radio_cortex_complete (Conservative)
        vf_coef: float = 0.5,
        ent_coef: float = 0.03, # Boosted exploration
        max_grad_norm: float = 0.5,
        device: Optional[str] = None,
        checkpoint_dir: str = 'models',
        checkpoint_interval: int = 5,
        model_type: str = 'bdh',
        lr_scheduler_gamma: float = 0.999, # Default decay per update
        target_kl: float = 0.05,
        ns3_config: Optional[object] = None,
        # Entropy Annealing
        ent_coef_start: float = 0.03,
        ent_coef_end: float = 0.005,
        ent_decay_fraction: float = 0.8
    ):
        self.hyperparams = {
            'hidden_dim': hidden_dim,
            'lr': lr,
            'gamma': gamma,
            'gae_lambda': gae_lambda,
            'clip_epsilon': clip_epsilon,
            'vf_coef': vf_coef,
            'ent_coef': ent_coef_start, # Start with initial value
            'max_grad_norm': max_grad_norm,
            'model_type': model_type,
            'lr_scheduler_gamma': lr_scheduler_gamma,
            'ent_coef_start': ent_coef_start,
            'ent_coef_end': ent_coef_end,
            'ent_decay_fraction': ent_decay_fraction
        }
        self.env = env
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.model_type = model_type
        self.target_kl = target_kl
        
        # Entropy Annealing State
        self.ent_coef = ent_coef
        self.ent_coef_start = ent_coef
        self.ent_coef_end = ent_coef_end
        self.ent_decay_fraction = ent_decay_fraction
        self.ns3_config = ns3_config
        
        # Detect vectorized environment
        self.is_vec_env = hasattr(env, 'num_envs')
        self.n_envs = env.num_envs if self.is_vec_env else 1
        if self.is_vec_env:
            print(f"[PPOTrainer] Detected VecEnv with {self.n_envs} parallel environments")
        
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        # Initialize networks based on model choice
        print(f"[PPOTrainer] Initializing Policy: {model_type.upper()}")
        
        if model_type == 'bdh':
            from policies.bdh_policy import BDHPolicy
            # Priority: Explicit config > Env config > None
            env_config = self.ns3_config
            if env_config is None:
                 # Try to extract from env wrapper (fallback)
                 env_config = getattr(env.envs[0], 'config', None) if hasattr(env, 'envs') and len(env.envs) > 0 else getattr(env, 'config', None)
            
            self.policy = BDHPolicy(state_dim, action_dim, device=device, env_config=env_config).to(device)
        elif model_type == 'gpt2':
            from policies.policy_gpt2 import GPT2Policy
            self.policy = GPT2Policy(state_dim, action_dim, device=device).to(device)
        elif model_type == 'trxl':
            from policies.policy_trxl import TrXLPolicy
            self.policy = TrXLPolicy(state_dim, action_dim, device=device).to(device)
        elif model_type == 'linear':
            from policies.policy_linear import LinearPolicy
            self.policy = LinearPolicy(state_dim, action_dim, device=device).to(device)
        elif model_type == 'universal':
            from policies.policy_universal import UniversalPolicy
            self.policy = UniversalPolicy(state_dim, action_dim, device=device).to(device)
        elif model_type == 'reformer':
            from policies.policy_reformer import ReformerPolicy
            self.policy = ReformerPolicy(state_dim, action_dim, device=device).to(device)
        else:
            # Default to Neural Network (MLP)
            self.policy = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
            
        # ── Live Interpretability Logger (BDH only) ──
        self.interp_logger = None
        if model_type == 'bdh':
            try:
                from interpretability.live_logger import InterpretabilityLogger
                _num_cells = getattr(self.ns3_config, 'num_cells', 3) if self.ns3_config else 3
                _log_dir = os.environ.get('RADIO_CORTEX_LOG_DIR', 'logs')
                self.interp_logger = InterpretabilityLogger(
                    self.policy, log_dir=_log_dir, num_cells=_num_cells
                )
            except Exception as e:
                print(f"[PPOTrainer] Interpretability logger init failed (non-fatal): {e}")

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        # Decay LR by gamma every step (approximating 0.999 per update)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=lr_scheduler_gamma)
        
        # Mixed Precision (AMP) support - targeting bfloat16 for stability on 16GB cards
        self.enable_amp = device == 'cuda'
        self.amp_dtype = torch.bfloat16 if self.enable_amp and torch.cuda.is_bf16_supported() else torch.float16
        # bfloat16 typically doesn't need scaling, but GradScaler handles enabled=False gracefully
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.enable_amp and self.amp_dtype == torch.float16))
        
        if self.enable_amp:
             print(f"[PPOTrainer] Mixed Precision enabled. Using dtype: {self.amp_dtype}")
        
        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        
        # Tracking
        self.total_steps = 0
        self.episode_rewards = []
        
        # Logging
        self.action_history = []
        self.telemetry_dir = os.environ.get('RADIO_CORTEX_LOG_DIR', 'logs')
        Path(self.telemetry_dir).mkdir(parents=True, exist_ok=True)
        self.log_file = os.path.join(self.telemetry_dir, "action_logs.jsonl")
        
        # Checkpointing
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        Path(checkpoint_dir).mkdir(exist_ok=True)
        
        # Environment Tracking
        self._last_obs = None
        self.max_checkpoints = 3 # Keep only last 3 checkpoints to save disk room
        
        # Initialize log file
            
    def compute_gae(self, rewards, values, dones, next_values):
        """Generalized Advantage Estimation (Vectorized for parallel environments)"""
        # Ensure inputs are tensors of shape (num_steps,) or (num_steps, n_envs)
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        if not isinstance(values, torch.Tensor):
            values = torch.FloatTensor(np.array(values)).to(self.device)
        if not isinstance(dones, torch.Tensor):
            dones = torch.FloatTensor(np.array(dones)).to(self.device)
        if not isinstance(next_values, torch.Tensor):
            next_values = torch.tensor(next_values, dtype=torch.float32).to(self.device)
        
        # Handle both single-env (1D) and vec-env (2D) shapes
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(-1)
            values = values.unsqueeze(-1)
            dones = dones.unsqueeze(-1)
            next_values = next_values.unsqueeze(-1) if next_values.dim() == 0 else next_values.unsqueeze(-1)
            squeeze_output = True
        else:
            squeeze_output = False
        
        num_steps, n_envs = rewards.shape
        advantages = torch.zeros((num_steps, n_envs), device=self.device)
        last_gae = torch.zeros(n_envs, device=self.device)
        
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_val = next_values
            else:
                next_val = values[t + 1]
            
            non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_val * non_terminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * non_terminal * last_gae
            advantages[t] = last_gae
        
        if squeeze_output:
            advantages = advantages.squeeze(-1)
        
        return advantages
    
    def collect_rollout(self, num_steps: int, progress: Optional[Progress] = None, task_id = None, on_step=None):
        """Collect experience from environment"""
        self.policy.eval() # Ensure deterministic behavior (disable dropout/batchnorm updates)
        # Dispatch to VecEnv-specific method if using vectorized environment
        if self.is_vec_env:
            return self.collect_rollout_vec(num_steps, progress, task_id, on_step)
        
        states, actions, rewards, dones, values, log_probs = [], [], [], [], [], []
        
        state, _ = self.env.reset()
        # print(f"[debug] collect_rollout start: num_steps={num_steps}, state_shape={np.shape(state)}")
        
        for step_i in range(num_steps):
            if progress and task_id is not None:
                progress.update(task_id, advance=1)
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.enable_amp, dtype=self.amp_dtype):
                    action, log_prob, entropy = self.policy.get_action(state_tensor)
                    _, _, value = self.policy(state_tensor)
            
            # Denormalize action to environment's action space
            action_np = action.float().cpu().numpy()[0]
            action_denorm = self._denormalize_action(action_np)
            
            next_state, reward, terminated, truncated, info = self.env.step(action_denorm)
            
            # ── Interpretability data collection (per step) ──
            if getattr(self, 'interp_logger', None):
                try:
                    _e2 = info.get('e2_metrics')
                    _e2_dict = _e2 if isinstance(_e2, dict) else ({'ue_metrics': _e2.ue_metrics, 'cell_metrics': _e2.cell_metrics} if _e2 and hasattr(_e2, 'ue_metrics') else {})
                    self.interp_logger.collect_step(state, _e2_dict)
                except Exception:
                    pass
            
            # Live UI Callback
            if on_step:
                on_step(info, reward, entropy=entropy.item() if entropy is not None else None)
            
            # Console Logging for User Verification
            if step_i % 10 == 0:
                e2_metrics = info.get('e2_metrics')
                avg_tput = 0.0
                if e2_metrics:
                    ue_metrics = e2_metrics['ue_metrics'] if isinstance(e2_metrics, dict) else e2_metrics.ue_metrics
                    if ue_metrics:
                        avg_tput = np.mean([m['throughput'] for m in ue_metrics.values()])
                        avg_delay = np.mean([m['delay'] for m in ue_metrics.values()])
                        avg_loss = np.mean([m['packet_loss'] for m in ue_metrics.values()])
                else:
                    avg_tput, avg_delay, avg_loss = 0.0, 0.0, 0.0
                
                # Format cell-centric action summary
                actions_per_cell = 3 # TxPower, CIO, TTT
                num_cells_log = len(action_denorm) // actions_per_cell
                if num_cells_log > 0:
                    c0 = action_denorm[:actions_per_cell]
                    # print(f"\n[Step {self.total_steps}] 🤖 Cell 0: TxΔ={c0[0]:.2f} | HO_Sens={c0[1]:.2f}")
                
                # print(f"[{self.total_steps}] Reward={reward:.3f} | Tput={avg_tput * self.env.config.num_ues:.2f} Mbps | Delay={avg_delay:.0f}ms | Loss={avg_loss*100:.1f}%", flush=True)

            # File logging
            try:
                e2_metrics = info.get('e2_metrics')
                log_entry = {
                    'step': self.total_steps,
                    'timestamp': datetime.now().isoformat(),
                    'reward': float(reward),
                    'action': [float(x) for x in action_denorm.tolist()],
                    'metrics': {
                        'ue': (e2_metrics['ue_metrics'] if isinstance(e2_metrics, dict) else e2_metrics.ue_metrics) if e2_metrics else {},
                        'cell': (e2_metrics['cell_metrics'] if isinstance(e2_metrics, dict) else e2_metrics.cell_metrics) if e2_metrics else {}
                    }
                }
                # Use custom default to handle numpy/torch types
                def _json_default(o):
                    try:
                        import numpy as _np
                        import torch as _torch
                        if isinstance(o, (_np.floating, _np.integer)):
                            return o.item()
                        if isinstance(o, _np.ndarray):
                            return o.tolist()
                        if isinstance(o, _torch.Tensor):
                            return o.detach().cpu().numpy().tolist()
                    except Exception:
                        pass
                    return str(o)

                with open(self.log_file, 'a') as f:
                    f.write(json.dumps(log_entry, default=_json_default) + "\n")
            except Exception as e:
                print(f"Logging error: {e}")

            done = terminated or truncated
            
            states.append(state)
            actions.append(action_np)
            rewards.append(reward)
            dones.append(done)
            values.append(value.item())
            log_probs.append(log_prob.item())
            
            state = next_state
            self.total_steps += 1
            
            if done:
                print(f"[debug] rollout early done at step {step_i}, total_steps={self.total_steps}")
                state, _ = self.env.reset()
        
        # Get value of final state for GAE
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.cuda.amp.autocast(enabled=self.enable_amp, dtype=self.amp_dtype):
                _, _, next_value = self.policy(state_tensor)
            next_value = next_value.item()
        
        # Compute advantages
        advantages = self.compute_gae(rewards, values, dones, next_value)
        returns = advantages + torch.tensor(values).to(self.device)
        
        return {
            'states': torch.FloatTensor(np.array(states)),
            'actions': torch.FloatTensor(np.array(actions)),
            'log_probs': torch.FloatTensor(np.array(log_probs)),
            'values': torch.FloatTensor(np.array(values)),
            'returns': returns,
            'advantages': advantages,
        }
    
    def collect_rollout_vec(self, num_steps: int, progress: Optional[Progress] = None, task_id = None, on_step=None):
        """Collect experience from vectorized environment (parallel envs)"""
        self.policy.eval() # Ensure deterministic behavior
        n_envs = self.n_envs
        
        # Storage: [num_steps, n_envs, ...]
        all_states = []
        all_actions = []
        all_rewards = []
        all_dones = []
        all_values = []
        all_log_probs = []
        
        # VecEnv reset returns just observations (n_envs, obs_dim)
        # Avoid redundant resets between rollouts for speed
        if self._last_obs is None:
            states = self.env.reset()
        else:
            states = self._last_obs
        
        for step_i in range(num_steps):
            if progress and task_id is not None:
                progress.update(task_id, advance=n_envs)  # Update by number of parallel steps
            
            # states: (n_envs, state_dim) -> tensor (n_envs, state_dim)
            states_tensor = torch.FloatTensor(states).to(self.device)
            
            with torch.no_grad():
                # Get actions for all envs at once using same precision as update
                with torch.cuda.amp.autocast(enabled=self.enable_amp, dtype=self.amp_dtype):
                    actions, log_probs, entropy = self.policy.get_action(states_tensor)
                    _, _, values = self.policy(states_tensor)
            
            # Convert to numpy: (n_envs, action_dim)
            actions_np = actions.float().cpu().numpy()
            
            # Denormalize actions for each env
            actions_denorm = np.array([self._denormalize_action(a) for a in actions_np])
            
            # Step all environments: returns (n_envs, ...) arrays
            next_states, rewards_arr, dones_arr, infos = self.env.step(actions_denorm)
            
            # ── Interpretability data collection (first env only for efficiency) ──
            if getattr(self, 'interp_logger', None) and len(infos) > 0:
                try:
                    _e2 = infos[0].get('e2_metrics', {})
                    _e2_dict = _e2 if isinstance(_e2, dict) else ({'ue_metrics': _e2.ue_metrics, 'cell_metrics': _e2.cell_metrics} if _e2 and hasattr(_e2, 'ue_metrics') else {})
                    self.interp_logger.collect_step(states[0], _e2_dict)
                except Exception:
                    pass
            
            # Live UI Callback
            if on_step:
                on_step(infos, rewards_arr, entropy=entropy.mean().item() if entropy is not None else None)
            
            # Progress is visible in the live LIVE row — no terminal print needed
            
            # Store batch data
            all_states.append(states)
            all_actions.append(actions_np)
            all_rewards.append(rewards_arr)
            all_dones.append(dones_arr)
            all_values.append(values.float().cpu().numpy().flatten())
            all_log_probs.append(log_probs.float().cpu().numpy().flatten())
            
            states = next_states
            self._last_obs = states # Store for next rollout cycle
            self.total_steps += n_envs
        
        # Convert collected lists to tensors of shape (num_steps, n_envs, ...)
        all_states_tensor = torch.FloatTensor(np.array(all_states)).to(self.device)
        all_actions_tensor = torch.FloatTensor(np.array(all_actions)).to(self.device)
        all_rewards_tensor = torch.FloatTensor(np.array(all_rewards)).to(self.device)
        all_dones_tensor = torch.FloatTensor(np.array(all_dones)).to(self.device)
        all_values_tensor = torch.FloatTensor(np.array(all_values)).to(self.device)
        all_log_probs_tensor = torch.FloatTensor(np.array(all_log_probs)).to(self.device)
        
        # Get value of final states for GAE
        with torch.no_grad():
            states_tensor = torch.FloatTensor(states).to(self.device)
            with torch.cuda.amp.autocast(enabled=self.enable_amp, dtype=self.amp_dtype):
                _, _, next_values = self.policy(states_tensor)
            next_values = next_values.squeeze(-1) # (n_envs,)
        
        # Compute advantages across all environments at once (vectorized)
        # Resulting shape: (num_steps, n_envs)
        advantages = self.compute_gae(
            all_rewards_tensor, 
            all_values_tensor, 
            all_dones_tensor, 
            next_values
        )
        returns = advantages + all_values_tensor
        
        return {
            'states': all_states_tensor.reshape(-1, all_states_tensor.size(-1)),
            'actions': all_actions_tensor.reshape(-1, all_actions_tensor.size(-1)),
            'log_probs': all_log_probs_tensor.flatten(),
            'values': all_values_tensor.flatten(),
            'returns': returns.flatten(),
            'advantages': advantages.flatten(),
        }
    
    def update_policy(self, rollout: Dict, num_epochs: int = 20, batch_size: int = 64):
        """Update policy using PPO objective - Optimized for sample efficiency"""
        self.policy.train() # Enable gradient updates and dropout (if any)
        states = rollout['states'].to(self.device)
        actions = rollout['actions'].to(self.device)
        old_log_probs = rollout['log_probs'].to(self.device)
        old_values = rollout['values'].to(self.device)
        returns = rollout['returns'].to(self.device)
        advantages = rollout['advantages'].to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        dataset_size = states.shape[0]
        
        for epoch in range(num_epochs):
            indices = torch.randperm(dataset_size)
            
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                idx = indices[start:end]
                
                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_old_values = old_values[idx]
                batch_returns = returns[idx]
                batch_advantages = advantages[idx]
                
                # Evaluate actions with mixed precision (if enabled)
                with torch.cuda.amp.autocast(enabled=self.enable_amp, dtype=self.amp_dtype):
                    values, log_probs, entropy = self.policy.evaluate_actions(
                        batch_states, batch_actions
                    )
                    
                    # PPO objective
                    ratio = torch.exp(log_probs - batch_old_log_probs)
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                    
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # Value function clipping (Anchor new predictions to rollout values)
                    v_loss_unclipped = (values.squeeze() - batch_returns) ** 2
                    v_clipped = batch_old_values + torch.clamp(
                        values.squeeze() - batch_old_values,
                        -self.clip_epsilon,
                        self.clip_epsilon
                    )
                    v_loss_clipped = (v_clipped - batch_returns) ** 2
                    value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                    
                    entropy_loss = -entropy.mean()
                    
                    loss = (
                        policy_loss +
                        self.vf_coef * value_loss +
                        self.ent_coef * entropy_loss
                    )
                    
                    # Calculate approximate KL divergence for monitoring + early stop
                    with torch.no_grad():
                        log_ratio = log_probs - batch_old_log_probs
                        approx_kl = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).item()
                
                # Check KL BEFORE applying the gradient step
                if self.target_kl is not None and approx_kl > self.target_kl * 1.5:
                    # Soft-clip the policy loss if KL diverges.
                    # CRITICAL FIX: Only train the Value network when KL goes over target. If we keep entropy_loss, it will skyrocket.
                    loss = self.vf_coef * value_loss
                    
                # Optimization step with GradScaler
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            
            # Epoch-level early stopping is removed to ensure Critic fully trains.
            # We rely on PPO's clip_epsilon and the mini-batch soft-clip above for policy safety.
        # Update learning rate
        self.scheduler.step()
        
        # Calculate Explained Variance in mini-batches to avoid OOM
        with torch.no_grad():
            v_pred_list = []
            ev_batch_size = batch_size * 2 # Slightly larger batch for inference
            for start in range(0, dataset_size, ev_batch_size):
                end = min(start + ev_batch_size, dataset_size)
                with torch.cuda.amp.autocast(enabled=self.enable_amp, dtype=self.amp_dtype):
                    batch_v = self.policy(states[start:end])[2].squeeze()
                v_pred_list.append(batch_v)
            
            v_pred = torch.cat(v_pred_list)
            y_true = returns
            var_y = torch.var(y_true)
            explained_var = 1.0 - torch.var(y_true - v_pred) / (var_y + 1e-8)
            explained_var = explained_var.item()

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': -entropy_loss.item(),
            'explained_variance': explained_var,
            'approx_kl': approx_kl
        }
    
    def _denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Convert normalized action [-1, 1] to environment action space"""
        # Clip action to valid range [-1, 1]
        action = np.clip(action, -1.0, 1.0)
        low = self.env.action_space.low
        high = self.env.action_space.high
        return low + (action + 1.0) * 0.5 * (high - low)
    
    def train(self, total_timesteps: int, rollout_steps: int = 20, log_interval: int = 1, batch_size: int = 64, checkpoint_interval: int = None, num_epochs: int = 20):
        """Main training loop
        
        Args:
            checkpoint_interval: Override instance checkpoint_interval (None = use default)
        """
        if checkpoint_interval is not None:
            self.checkpoint_interval = checkpoint_interval
        
        # Calculate update frequency based on total samples per rollout
        samples_per_update = rollout_steps * self.n_envs
        
        # Resumption logic: Determine how many updates are actually needed
        start_update = self.total_steps // samples_per_update
        total_updates = total_timesteps // samples_per_update
        num_updates = max(0, total_updates - start_update)
        
        if num_updates == 0 and self.total_steps < total_timesteps:
            # If we have less than one full update remaining, run one final update
            num_updates = 1
        
        if num_updates == 0:
            print(f"INFO: Target total timesteps ({total_timesteps:,}) already reached (currently at {self.total_steps:,} steps).")
            return
        
        print(f"Resuming PPO training from Update {start_update} ({self.total_steps:,} steps)")
        print(f"Running {num_updates} more updates to reach {total_timesteps:,} steps")
        print(f"Device: {self.device} | Mixed Precision (AMP): {self.enable_amp} ({self.amp_dtype})")
        
        # Add training loop params
        self.hyperparams.update({
            'total_timesteps': total_timesteps,
            'rollout_steps': rollout_steps,
            'batch_size': batch_size,
            'n_envs': self.n_envs
        })
        
        update = 0
        # Rich UI Setup
        console = Console()
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            expand=True
        )
        
        main_task = progress.add_task("Training", total=total_timesteps)
        
        # Live metrics storage: {env_id: {'tput': x, 'loss': y, 'ue_metrics': {...}}}
        live_env_metrics = {}
        live_step_stats = {'reward': 0.0, 'entropy': 0.0, 'steps': 0}
        current_rollout_rewards = []
        
        # We need a reference to the live object for the callback
        live_display = None

        # Callback to update live metrics from env info
        def on_step_callback(infos, rewards=None, entropy=None):
            # Debug: Print type and content of first info
            # if isinstance(infos, (list, tuple)) and len(infos) > 0:
            #    print(f"\n[DEBUG] Info type: {type(infos)}, Len: {len(infos)}")
            #    if 'e2_metrics' in infos[0]:
            #        print(f" [DEBUG] e2_metrics found. Type: {type(infos[0]['e2_metrics'])}")
            
            # Handle both single and parallel envs
            if isinstance(infos, dict): # Single env
                 infos = [infos]
            elif isinstance(infos, tuple):
                 infos = list(infos)
            
            # --- Live Per-Step Stats (separate from frozen PPO snapshot) ---
            if rewards is not None:
                if np.isscalar(rewards):
                    current_rollout_rewards.append(rewards)
                else:
                    current_rollout_rewards.extend(rewards)
            
            # nonlocal current_stats  <-- Removed undefined reference
            
            # Update live step stats — shown in real-time LIVE row
            if current_rollout_rewards:
                live_step_stats['reward'] = float(np.mean(current_rollout_rewards))
            live_step_stats['steps'] = self.total_steps
            if entropy is not None:
                live_step_stats['entropy'] = float(entropy)

            # ---------------------------
            # Optimization: Only perform complex UI metrics calculations at 10Hz
            # This saves massive CPU time during rollouts.
            if not hasattr(self, '_last_ui_update'):
                self._last_ui_update = 0
            
            now = time.time()
            if now - self._last_ui_update < 0.1: # 10Hz limit for metric processing
                return # Skip heavy processing
            
            self._last_ui_update = now # Mark update as happening
            # ---------------------------

            for env_i, info in enumerate(infos):
                e2 = info.get('e2_metrics')
                if e2:
                    # Handle both E2Message objects and plain dicts (from SubprocVecEnv)
                    if hasattr(e2, 'ue_metrics'):
                        ue_kpms_raw = e2.ue_metrics
                        cell_kpms_raw = e2.cell_metrics
                    elif isinstance(e2, dict):
                        ue_kpms_raw = e2.get('ue_metrics', {})
                        cell_kpms_raw = e2.get('cell_metrics', {})
                    else:
                        continue
                    
                    ue_kpms = list(ue_kpms_raw.values()) if ue_kpms_raw else []
                    cell_kpms = list(cell_kpms_raw.values()) if cell_kpms_raw else []
                    
                    tput = np.mean([m['throughput'] for m in ue_kpms]) if ue_kpms else 0.0
                    delay = np.mean([m['delay'] for m in ue_kpms]) if ue_kpms else 0.0
                    loss = np.mean([m['packet_loss'] for m in ue_kpms]) if ue_kpms else 0.0
                    sinr = np.mean([m['sinr'] for m in ue_kpms]) if ue_kpms else -10.0
                    rsrp = np.mean([m['rsrp'] for m in ue_kpms]) if ue_kpms else -140.0
                    
                    queue = np.mean([c['queue_length'] for c in cell_kpms]) if cell_kpms else 0.0
                    rb = np.mean([c['rb_utilization'] for c in cell_kpms]) if cell_kpms else 0.0
                    tx_dbm = np.mean([c.get('tx_power', 23.0) for c in cell_kpms]) if cell_kpms else 23.0
                    
                    live_env_metrics[env_i] = {
                        'tput': tput, 'delay': delay, 'loss': loss, 
                        'sinr': sinr, 'rsrp': rsrp, 'queue': queue, 'rb': rb, 'power': tx_dbm,
                        'success': info.get('z_success', 0.0)
                    }
            
            # Force refresh of the live display
            if live_display:
                live_display.update(make_layout())

        # Stats display table (Updated for Convergence Metrics)
        def create_stats_table(current_metrics=None):
            table = Table(show_header=True, header_style="bold magenta", expand=True)
            table.add_column("Upd", justify="center")
            table.add_column("Steps", justify="center")
            table.add_column("Reward", justify="center")
            table.add_column("Trend", justify="center")
            table.add_column("Expl Var", justify="center")
            table.add_column("Pol Loss", justify="center")
            table.add_column("Val Loss", justify="center")
            table.add_column("KL", justify="center")
            table.add_column("Entropy", justify="center")
            table.add_column("LR", justify="center")

            # Live values (10Hz) — reward and entropy update every step
            live_rew = live_step_stats.get('reward', 0.0)
            live_ent = live_step_stats.get('entropy', 0.0)
            live_steps = live_step_stats.get('steps', 0)

            # PPO snapshot values — persist from last update, shown as dim until refreshed
            if current_metrics:
                trend = current_metrics.get('trend', '→')
                trend_str = f"[green]{trend}[/]" if trend == '↗' else f"[red]{trend}[/]" if trend == '↘' else trend
                expl_var = current_metrics.get('explained_variance', 0.0)
                ev_color = "green" if expl_var > 0.8 else "yellow" if expl_var > 0.4 else "red"
                kl = current_metrics.get('approx_kl', 0.0)
                kl_color = "green" if kl < 0.05 else "yellow" if kl < 0.15 else "red"
                pol_loss = f"{current_metrics.get('policy_loss', 0.0):.4f}"
                val_loss = f"{current_metrics.get('value_loss', 0.0):.4f}"
                upd_str = str(current_metrics.get('update', '-'))
                lr_str = f"{self.scheduler.get_last_lr()[0]:.2e}"
            else:
                trend_str = "→"
                ev_color, kl_color = "dim", "dim"
                expl_var, kl = 0.0, 0.0
                pol_loss, val_loss = "—", "—"
                upd_str, lr_str = "—", "—"

            # Single merged row: live Reward + Entropy, frozen PPO diagnostics
            table.add_row(
                upd_str,
                f"[cyan]{live_steps:,}[/]",
                f"[bold green]{live_rew:.3f}[/]" if live_rew > 0 else f"[bold red]{live_rew:.3f}[/]",
                trend_str,
                f"[{ev_color}]{expl_var:.3f}[/]",
                pol_loss,
                val_loss,
                f"[{kl_color}]{kl:.4f}[/]",
                f"[cyan]{live_ent:.4f}[/]",
                lr_str,
            )
            return Panel(table, title="[bold blue]RL Training Progress[/]", border_style="blue", expand=True)

        # Per-Env Metrics Table
        def create_env_metrics_table():
            table = Table(show_header=True, header_style="bold green", expand=True)
            table.add_column("Env ID", justify="center")
            table.add_column("Succ", justify="right")      # New: Success Rate
            table.add_column("Activity", justify="center") # Heartbeat indicator
            table.add_column("Tput (Mbps)", justify="right")
            table.add_column("Delay (ms)", justify="right")
            table.add_column("Loss (%)", justify="right")
            table.add_column("SINR (dB)", justify="right")
            table.add_column("Queue", justify="right")
            table.add_column("RB Util", justify="right")
            table.add_column("TX (dBm)", justify="right")
            
            # Simple alternating heartbeat
            heartbeat = "●" if int(time.time() * 2) % 2 == 0 else "○"
            
            for env_id in sorted(live_env_metrics.keys()):
                m = live_env_metrics[env_id]
                table.add_row(
                    str(env_id),
                    f"{m.get('success', 0)*100:.0f}%",
                    f"[bold green]{heartbeat}[/]" if m.get('tput', 0) > 0 else "[dim]idling[/]",
                    f"{m.get('tput', 0):.2f}",
                    f"{m.get('delay', 0):.1f}",
                    f"{m.get('loss', 0)*100:.1f}",
                    f"{m.get('sinr', -10):.1f}",
                    f"{m.get('queue', 0):.1f}",
                    f"{m.get('rb', 0)*100:.1f}%",
                    f"{m.get('power', 23.0):.1f}"
                )
            if not live_env_metrics:
                 table.add_row("-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-")
                 
            return Panel(table, title="[bold green]Live Environment Metrics[/]", border_style="green", expand=True)

        # UE Metrics Grid
        def create_ue_grid():
            tables = []
            for env_id in sorted(live_env_metrics.keys()):
                m = live_env_metrics[env_id]
                if 'ue_metrics' not in m: continue
                
                ue_data = m['ue_metrics']
                table = Table(title=f"Env {env_id} UEs", show_header=True, header_style="bold cyan", expand=True, box=None)
                table.add_column("UE", justify="right", style="cyan", width=3)
                table.add_column("Cell", justify="right", style="magenta", width=3)
                table.add_column("Tput", justify="right", style="green")
                table.add_column("Delay", justify="right", style="yellow")
                table.add_column("Loss", justify="right", style="red")
                
                # Limit to first 5 UEs to save space
                displayed_ues = sorted(ue_data.keys())[:5]
                
                for ue_id in displayed_ues:
                    ud = ue_data[ue_id]
                    table.add_row(
                        str(ue_id),
                        str(ud['cell']),
                        f"{ud['tput']:.1f}",
                        f"{ud['delay']:.0f}",
                        f"{ud['loss']*100:.0f}%"
                    )
                
                # Add a row indicating more UEs if truncated
                if len(ue_data) > 5:
                    table.add_row("..", "..", "..", "..", "..")
                    
                tables.append(Panel(table, border_style="white", expand=True))
            
            # Optimization: If many envs, only show UE grid if reasonably small
            # Rendering 12+ tables per frame is a massive CPU sink.
            if len(live_env_metrics) > 4:
                return Panel("(UE Detail Grid Hidden for Performance — showing 12+ Envs)", style="dim italic")

            if not tables:
                return Panel("Waiting for UE data...", style="dim")
                
            # Create a 2-column grid to hold the panels
            grid = Table.grid(expand=True)
            grid.add_column(ratio=1)
            grid.add_column(ratio=1)
            
            # Add panels in rows of 2
            for i in range(0, len(tables), 2):
                row_panels = tables[i:i+2]
                if len(row_panels) == 1:
                    row_panels.append("") # Padding
                grid.add_row(*row_panels)
                
            return grid

        update = 0
        current_stats = None
        live_step_stats = {'reward': 0.0, 'entropy': 0.0, 'steps': 0}  # Per-step live tracking
        reward_history = [0.0] # For trend calculation
        self.consistent_level_2_counter = 0 # Track mastery for early stopping
        
        # Helper to generate the full layout
        def make_layout():
            return Group(
                progress, # Progress bar at top
                create_stats_table(current_stats),
                create_env_metrics_table(),
                create_ue_grid()
            )
            
        with Live(make_layout(), console=console, refresh_per_second=10) as live:
            live_display = live # Set reference for callback
            for current_update_count in range(num_updates):
                # Calculate absolute update number for logs/UI
                abs_update = start_update + current_update_count + 1
                
                # Reset storage for live reward tracking per update
                current_rollout_rewards = []
                
                # Collect rollout with callback
                rollout = self.collect_rollout(rollout_steps, progress=progress, task_id=main_task, on_step=on_step_callback)
            
                # Update policy
                metrics = self.update_policy(rollout, batch_size=batch_size, num_epochs=num_epochs)
                
                # Linear decay of entropy coefficient
                fraction = min(1.0, self.total_steps / (total_timesteps * self.ent_decay_fraction))
                self.ent_coef = self.ent_coef_start + fraction * (self.ent_coef_end - self.ent_coef_start)

                # ── Periodic Interpretability Analysis ──
                if getattr(self, 'interp_logger', None):
                    try:
                        self.interp_logger.run_periodic_analysis(
                            rollout.get('states'), update + 1, self.total_steps
                        )
                    except Exception:
                        pass
                
                # Update Dashboard Stats (Final for this update)
                avg_reward = rollout['returns'].mean().item()
                
                # Calculate Trend: compare with previous update average
                prev_avg = reward_history[-1]
                if avg_reward > prev_avg * 1.05: # > 5% improvement
                    trend = '↗'
                elif avg_reward < prev_avg * 0.95: # > 5% drop
                    trend = '↘'
                else:
                    trend = '→'
                
                reward_history.append(avg_reward)
                if len(reward_history) > 10: reward_history.pop(0)

                current_stats = {
                    'update': abs_update,
                    'steps': self.total_steps,
                    'reward': avg_reward,
                    'trend': trend,
                    'explained_variance': metrics['explained_variance'],
                    'policy_loss': metrics['policy_loss'],
                    'value_loss': metrics['value_loss'],
                    'approx_kl': metrics['approx_kl'],
                    'entropy': metrics['entropy']
                }
                
                # ── Persist per-update metrics to disk for convergence graphs ──
                try:
                    _log_path = os.path.join(self.telemetry_dir, 'training_log.jsonl')
                    with open(_log_path, 'a') as _lf:
                        _lf.write(json.dumps(current_stats) + '\n')
                except Exception:
                    pass

                # Force refresh
                live.update(make_layout())

                # -----------------------------------------
                
                # Periodic checkpointing
                if self.checkpoint_interval > 0 and abs_update % self.checkpoint_interval == 0:
                    checkpoint_path = str(Path(self.checkpoint_dir) / f'radio_cortex_upd_{abs_update}.pt')
                    self.save(checkpoint_path)
                    
                    # --- ADDED: Save scalars for periodic checkpoints ---
                    if self.is_vec_env:
                        try:
                            from vec_env_wrapper import save_vec_normalize
                            scalar_path = str(Path(self.checkpoint_dir) / f'vec_normalize_upd_{abs_update}.pkl')
                            save_vec_normalize(self.env, scalar_path)
                        except ImportError:
                            pass
                    # ----------------------------------------------------
                    
                    self._rotate_checkpoints()
        
        print("\n✓ Training complete")


    def save(self, path: str):
        """Save trained model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
            'hyperparams': getattr(self, 'hyperparams', {})
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_steps = checkpoint['total_steps']
        print(f"Model loaded from {path}")

    def _rotate_checkpoints(self):
        """Keep only the most recent checkpoints to save disk space"""
        try:
            checkpoints = sorted(
                Path(self.checkpoint_dir).glob("radio_cortex_upd_*.pt"),
                key=os.path.getmtime
            )
            if len(checkpoints) > self.max_checkpoints:
                # Delete oldest
                for i in range(len(checkpoints) - self.max_checkpoints):
                    os.remove(checkpoints[i])
                    print(f"[Disk Cleanup] Removed old checkpoint: {checkpoints[i]}")
        except Exception as e:
            print(f"[Disk Cleanup] Warning: Rotation failed: {e}")


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_policy(
    env: gym.Env,
    policy: ActorCritic,
    num_episodes: int = 10,
    device: str = 'cpu',
    render: bool = False
) -> Dict:
    """Evaluate trained policy"""
    episode_rewards = []
    episode_metrics = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_data = {
            'throughputs': [],
        
            'delays': [],
            'losses': []
        }
        
        done = False
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            with torch.no_grad():
                action = policy.get_action(state_tensor, deterministic=True)
                if isinstance(action, tuple):
                    action = action[0]
            
            action_np = action.cpu().numpy()[0]
            
            # Denormalize
            low = env.action_space.low
            high = env.action_space.high
            action_denorm = low + (action_np + 1.0) * 0.5 * (high - low)
            
            state, reward, terminated, truncated, info = env.step(action_denorm)
            done = terminated or truncated
            
            episode_reward += reward
            
            # Collect metrics from info
            if 'e2_metrics' in info:
                e2 = info['e2_metrics']
                # Handle both dict (from serialized env) and object
                ue_metrics = e2['ue_metrics'] if isinstance(e2, dict) else e2.ue_metrics
                
                episode_data['throughputs'].extend([m['throughput'] for m in ue_metrics.values()])
                episode_data['delays'].extend([m['delay'] for m in ue_metrics.values()])
                episode_data['losses'].extend([m['packet_loss'] for m in ue_metrics.values()])
            
            if render:
                env.render()
        
        episode_rewards.append(episode_reward)
        episode_metrics.append({
            'avg_throughput': np.mean(episode_data['throughputs']) if episode_data['throughputs'] else 0,
            'avg_delay': np.mean(episode_data['delays']) if episode_data['delays'] else 0,
            'avg_loss': np.mean(episode_data['losses']) if episode_data['losses'] else 0,
        })
        
        print(f"Episode {episode + 1}: Reward = {episode_reward:.3f}")
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_throughput': np.mean([m['avg_throughput'] for m in episode_metrics]),
        'mean_delay': np.mean([m['avg_delay'] for m in episode_metrics]),
        'mean_loss': np.mean([m['avg_loss'] for m in episode_metrics]),
    }


