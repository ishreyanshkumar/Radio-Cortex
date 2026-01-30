"""
Radio-Cortex RL Training Pipeline
Integrates O-RAN ns-3 environment with standard RL algorithms
Supports PPO, SAC, TD3 for RAN congestion control
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
import gymnasium as gym
from collections import deque
import wandb
import json
from pathlib import Path
from datetime import datetime
import dataclasses
import torch


# ============================================================================
# Neural Network Architectures
# ============================================================================

from neural_networks import ActorCritic, SACAgent


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
        self,
        env: gym.Env,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.env = env
        self.device = device
        
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        # Initialize networks
        self.policy = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
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
        self.log_file = "action_logs.jsonl"
        
        # Initialize log file
        with open(self.log_file, 'w') as f:
            pass # Clear file
            
    def compute_gae(self, rewards, values, dones, next_value):
        """Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        return torch.tensor(advantages, dtype=torch.float32)
    
    def collect_rollout(self, num_steps: int):
        """Collect experience from environment"""
        states, actions, rewards, dones, values, log_probs = [], [], [], [], [], []
        
        state, _ = self.env.reset()
        print(f"[debug] collect_rollout start: num_steps={num_steps}, state_shape={np.shape(state)}")
        
        for step_i in range(num_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action, log_prob = self.policy.get_action(state_tensor)
                _, value = self.policy(state_tensor)
            
            # Denormalize action to environment's action space
            action_np = action.cpu().numpy()[0]
            action_denorm = self._denormalize_action(action_np)
            
            next_state, reward, terminated, truncated, info = self.env.step(action_denorm)
            
            # Console Logging for User Verification
            if step_i % 10 == 0:
                e2_metrics = info.get('e2_metrics')
                avg_tput = 0.0
                if e2_metrics and e2_metrics.ue_metrics:
                    avg_tput = np.mean([m['throughput'] for m in e2_metrics.ue_metrics.values()])
                
                # Format action for display (first 3 dims)
                action_str = f"[{', '.join(f'{x:.2f}' for x in action_denorm[:3])}...]"
                print(f"[{self.total_steps}] Action={action_str} | Reward={reward:.3f} | Tput={avg_tput:.2f} Mbps", flush=True)

            # File logging
            try:
                e2_metrics = info.get('e2_metrics')
                log_entry = {
                    'step': self.total_steps,
                    'timestamp': datetime.now().isoformat(),
                    'reward': float(reward),
                    'action': [float(x) for x in action_denorm.tolist()],
                    'metrics': {
                        'ue': e2_metrics.ue_metrics if e2_metrics else {},
                        'cell': e2_metrics.cell_metrics if e2_metrics else {}
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
            _, next_value = self.policy(state_tensor)
            next_value = next_value.item()
        
        # Compute advantages
        advantages = self.compute_gae(rewards, values, dones, next_value)
        returns = advantages + torch.tensor(values)
        
        return {
            'states': torch.FloatTensor(np.array(states)),
            'actions': torch.FloatTensor(np.array(actions)),
            'log_probs': torch.FloatTensor(np.array(log_probs)),
            'returns': returns,
            'advantages': advantages,
        }
    
    def update_policy(self, rollout: Dict, num_epochs: int = 4, batch_size: int = 64):
        """Update policy using PPO objective"""
        states = rollout['states'].to(self.device)
        actions = rollout['actions'].to(self.device)
        old_log_probs = rollout['log_probs'].to(self.device)
        returns = rollout['returns'].to(self.device)
        advantages = rollout['advantages'].to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        dataset_size = states.shape[0]
        
        print(f"[debug] update_policy: dataset_size={dataset_size}, num_epochs={num_epochs}, batch_size={batch_size}")
        for epoch in range(num_epochs):
            indices = torch.randperm(dataset_size)
            print(f"[debug] update_policy: epoch {epoch+1}/{num_epochs}")
            
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                idx = indices[start:end]
                print(f"[debug]   batch rows {start}-{end} (actual {len(idx)})")
                
                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_returns = returns[idx]
                batch_advantages = advantages[idx]
                
                # Evaluate actions
                values, log_probs, entropy = self.policy.evaluate_actions(
                    batch_states, batch_actions
                )
                
                # PPO objective
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.MSELoss()(values.squeeze(), batch_returns)
                entropy_loss = -entropy.mean()
                
                loss = (
                    policy_loss +
                    self.vf_coef * value_loss +
                    self.ent_coef * entropy_loss
                )
                
                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': -entropy_loss.item()
        }
    
    def _denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Convert normalized action [-1, 1] to environment action space"""
        # Clip action to valid range [-1, 1]
        action = np.clip(action, -1.0, 1.0)
        low = self.env.action_space.low
        high = self.env.action_space.high
        return low + (action + 1.0) * 0.5 * (high - low)
    
    def train(self, total_timesteps: int, rollout_steps: int = 20, log_interval: int = 1):
        """Main training loop"""
        num_updates = total_timesteps // rollout_steps
        
        print(f"Starting PPO training for {total_timesteps} timesteps")
        print(f"Device: {self.device}")
        
        for update in range(num_updates):
            # Collect rollout
            print(f"[debug] train: starting rollout {update+1}/{num_updates}")
            rollout = self.collect_rollout(rollout_steps)
            
            # Update policy
            print(f"[debug] train: starting policy update for rollout {update+1}")
            metrics = self.update_policy(rollout)
            print(f"[debug] train: completed policy update for rollout {update+1}")
            # Logging
            if update % log_interval == 0:
                avg_reward = rollout['returns'].mean().item()
                print(f"\nUpdate {update}/{num_updates}")
                print(f"  Total steps: {self.total_steps}")
                print(f"  Avg return: {avg_reward:.3f}")
                print(f"  Policy loss: {metrics['policy_loss']:.4f}")
                print(f"  Value loss: {metrics['value_loss']:.4f}")
                print(f"  Entropy: {metrics['entropy']:.4f}")
        
        print("\n✓ Training complete")
    
    def save(self, path: str):
        """Save trained model"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_steps = checkpoint['total_steps']
        print(f"Model loaded from {path}")


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
                episode_data['throughputs'].extend([m['throughput'] for m in e2.ue_metrics.values()])
                episode_data['delays'].extend([m['delay'] for m in e2.ue_metrics.values()])
                episode_data['losses'].extend([m['packet_loss'] for m in e2.ue_metrics.values()])
            
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


# ============================================================================
# Main Training Script
# ============================================================================

if __name__ == "__main__":
    from oran_ns3_env import create_oran_env, NS3Config
    
    # Create environment
    config = NS3Config(
        num_ues=20,
        num_cells=3,
        sim_time=5.0,  # 5 seconds per episode
        kpm_interval_ms=100,
        seed=42
    )
    
    env = create_oran_env(config)
    
    # Create trainer
    trainer = PPOTrainer(
        env=env,
        hidden_dim=256,
        lr=3e-4,
        gamma=0.99,
        clip_epsilon=0.2,
    )
    
    # Train
    trainer.train(
        total_timesteps=50000,  # 50k steps
        rollout_steps=2048,
        log_interval=5
    )
    
    # Save model
    save_dir = Path("models")
    save_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trainer.save(f"models/radio_cortex_ppo_{timestamp}.pt")
    
    # Evaluate
    print("\n" + "="*50)
    print("Evaluating trained policy...")
    print("="*50)
    
    results = evaluate_policy(
        env=env,
        policy=trainer.policy,
        num_episodes=10,
        device=trainer.device,
        render=True
    )
    
    print(f"\nEvaluation Results:")
    print(f"  Mean reward: {results['mean_reward']:.3f} ± {results['std_reward']:.3f}")
    print(f"  Mean throughput: {results['mean_throughput']:.2f} Mbps")
    print(f"  Mean delay: {results['mean_delay']:.2f} ms")
    print(f"  Mean packet loss: {results['mean_loss']:.4f}")
    
    env.close()