"""Quick train script: tiny PPO on a mock ORAN env for fast verification
This trains a small model in seconds (no ns-3, no Kafka).
"""
import time
import numpy as np
import torch
import gymnasium as gym
from rl_training_pipeline import PPOTrainer
from neural_networks import ActorCritic
from oran_ns3_env import NS3Config

class MockORANEnv(gym.Env):
    def __init__(self, num_ues=5, num_cells=2, kpm_interval_ms=100, sim_time=2.0):
        super().__init__()
        self.config = NS3Config(num_ues=num_ues, num_cells=num_cells, sim_time=sim_time, kpm_interval_ms=kpm_interval_ms)
        self.state_dim = num_ues * 7 + num_cells * 3
        self.action_dim = num_cells * 6
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)
        self.step_count = 0
        self.max_steps = int(sim_time * 1000 / kpm_interval_ms)

    def reset(self, seed=None, options=None):
        self.step_count = 0
        state = np.zeros(self.state_dim, dtype=np.float32)
        return state, {}

    def step(self, action):
        # Simple dynamics: higher first half of action increases throughput
        self.step_count += 1
        # Build fake ue metrics
        ue_metrics = {}
        for ue in range(self.config.num_ues):
            # correlate throughput to mean action value
            tput = max(0.0, (action.mean() + 1.0) * 5.0 + np.random.randn() * 0.5)
            delay = max(0.0, 50.0 - tput + np.random.randn() * 2.0)
            loss = float(max(0.0, 0.01 * (1.0 / (tput + 1e-3))))
            sinr = float(10.0 * (tput / (tput + 5.0)) + np.random.randn() * 0.5)
            ue_metrics[ue] = {'throughput': tput, 'delay': delay, 'packet_loss': loss, 'sinr': sinr, 'rsrp': -100.0, 'rsrq': -10.0, 'ul_rbs': 1.0}

        cell_metrics = {c: {'queue_length': 0, 'rb_utilization': 0.5, 'tx_power': 23.0, 'num_connected_ues': self.config.num_ues // self.config.num_cells} for c in range(self.config.num_cells)}

        # state: flatten some representative numbers
        state = []
        for ue in range(self.config.num_ues):
            m = ue_metrics[ue]
            state.extend([m['throughput']/100.0, m['delay']/1000.0, m['packet_loss'], (m['sinr']+10)/40.0, 0.0, 0.0, 0.0])
        for c in range(self.config.num_cells):
            cm = cell_metrics[c]
            state.extend([cm['queue_length']/1000.0, cm['rb_utilization'], (cm['tx_power']-10)/36.0])
        state = np.array(state, dtype=np.float32)

        # reward: encourage throughput, penalize delay
        avg_tput = np.mean([m['throughput'] for m in ue_metrics.values()])
        avg_delay = np.mean([m['delay'] for m in ue_metrics.values()])
        avg_sinr = np.mean([m['sinr'] for m in ue_metrics.values()])
        reward = float(np.log(avg_tput + 1e-6) - 0.01 * avg_delay + 0.05 * avg_sinr)

        terminated = self.step_count >= self.max_steps
        truncated = False
        info = {'e2_metrics': type('E', (), {'ue_metrics': ue_metrics, 'cell_metrics': cell_metrics})()}
        return state, reward, terminated, truncated, info

    def render(self, mode='human'):
        pass

    def close(self):
        pass


if __name__ == '__main__':
    # tiny quick training
    env = MockORANEnv(num_ues=5, num_cells=2, kpm_interval_ms=100, sim_time=2.0)
    trainer = PPOTrainer(env=env, hidden_dim=64, lr=1e-3, clip_epsilon=0.2)

    start = time.time()
    trainer.train(total_timesteps=2000, rollout_steps=200, log_interval=1)
    elapsed = time.time() - start
    print(f"Quick training finished in {elapsed:.2f}s")

    # quick eval
    from rl_training_pipeline import evaluate_policy
    results = evaluate_policy(env=env, policy=trainer.policy, num_episodes=3, device=trainer.device)
    print('Eval:', results)

    # save small model
    import os
    os.makedirs('models', exist_ok=True)
    trainer.save('models/quick_radio.pt')
    env.close()
