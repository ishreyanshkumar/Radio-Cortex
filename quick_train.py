"""Quick train script: tiny PPO on a mock ORAN env for fast verification.
This trains a small model in seconds (no ns-3, no Kafka).
Updated to match real environment observation and action spaces.
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
        
        # Matches ORANns3Env
        self.state_dim = num_ues * 12 + num_cells * 5
        self.action_dim = num_cells * 7 + num_ues
        
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)
        self.step_count = 0
        self.max_steps = int(sim_time * 1000 / kpm_interval_ms)

    def reset(self, seed=None, options=None):
        self.step_count = 0
        state = np.zeros(self.state_dim, dtype=np.float32)
        return state, {}

    def step(self, action):
        self.step_count += 1
        
        # Build fake ue metrics
        ue_metrics = {}
        for ue in range(self.config.num_ues):
            # correlate throughput to mean action value
            tput = max(0.0, (action.mean() + 1.0) * 5.0 + np.random.randn() * 0.5)
            delay = max(0.0, 50.0 - tput + np.random.randn() * 2.0)
            loss = float(max(0.0, 0.01 * (1.0 / (tput + 1e-3))))
            sinr = float(10.0 * (tput / (tput + 5.0)) + np.random.randn() * 0.5)
            
            ue_metrics[ue] = {
                'throughput': tput, 
                'delay': delay, 
                'packet_loss': loss, 
                'sinr': sinr, 
                'rsrp': -100.0, 
                'rsrq': -10.0, 
                'ul_rbs': 1.0,
                'rb_allocated': 10,
                'cqi': 12,
                'rsrp_var': 0.1,
                'rsrq_var': 0.1,
                'buffer_occupancy': 500,
                'serving_cell': 0,
                'handover_attempts': 0,
                'handover_successes': 0
            }

        cell_metrics = {c: {
            'queue_length': 10, 
            'rb_utilization': 0.5, 
            'tx_power': 23.0, 
            'num_connected_ues': self.config.num_ues // self.config.num_cells,
            'cell_load': 0.5,
            'avg_rb_request': 20.0
        } for c in range(self.config.num_cells)}

        # state: matches ORANns3Env._extract_state
        state = []
        for ue in range(self.config.num_ues):
            m = ue_metrics[ue]
            state.extend([
                m['throughput']/100.0, 
                m['delay']/1000.0, 
                m['packet_loss'], 
                (m['sinr']+10)/40.0,
                (m['rsrp']+140.0)/100.0,
                (m['rsrq']+20.0)/20.0,
                m['ul_rbs']/100.0,
                m['rb_allocated']/100.0,
                m['cqi']/15.0,
                m['rsrp_var']/50.0,
                m['rsrq_var']/50.0,
                m['buffer_occupancy']/10000.0
            ])
        for c in range(self.config.num_cells):
            cm = cell_metrics[c]
            state.extend([
                cm['queue_length']/1000.0, 
                cm['rb_utilization'], 
                (cm['tx_power']-10)/36.0,
                cm['cell_load'],
                cm['avg_rb_request']/100.0
            ])
        state = np.array(state, dtype=np.float32)

        # reward: encourage throughput, penalize delay
        tputs = [m['throughput'] for m in ue_metrics.values()]
        delays = [m['delay'] for m in ue_metrics.values()]
        sinrs = [m['sinr'] for m in ue_metrics.values()]
        
        sum_log_tput = np.sum(np.log(np.array(tputs) + 1e-6))
        avg_delay = np.mean(delays)
        avg_sinr = np.mean(sinrs)
        
        reward = float(sum_log_tput - 0.1 * avg_delay + 0.05 * avg_sinr + 0.5)

        terminated = self.step_count >= self.max_steps
        truncated = False
        
        # Mocking the E2Message structure
        class MockE2Msg:
            def __init__(self, ue_m, cell_m):
                self.ue_metrics = ue_m
                self.cell_metrics = cell_m
        
        info = {'e2_metrics': MockE2Msg(ue_metrics, cell_metrics)}
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

    # save small model
    import os
    os.makedirs('models', exist_ok=True)
    trainer.save('models/quick_radio.pt')
    env.close()
    print("Model saved to models/quick_radio.pt")
