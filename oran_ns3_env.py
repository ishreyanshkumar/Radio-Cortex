"""
Radio-Cortex O-RAN ns-3 Environment
A Gym-compatible environment for RL-based RAN congestion control
integrating with ns-O-RAN simulation platform.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import subprocess
import json
import socket
import time
import sys
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class SchedulerType(Enum):
    """MAC Scheduler algorithms"""
    PROPORTIONAL_FAIR = 0
    ROUND_ROBIN = 1
    MAX_THROUGHPUT = 2

@dataclass
class NS3Config:
    """ns-3 simulation configuration"""
    num_ues: int = 20
    num_cells: int = 3
    sim_time: float = 10.0  # seconds
    seed: int = 42
    e2_port: int = 36421
    kpm_interval_ms: int = 10  # E2SM-KPM reporting interval


@dataclass
class E2Message:
    """E2 interface message format"""
    timestamp: float
    ue_metrics: Dict[int, Dict]  # {ue_id: {throughput, delay, loss, sinr}}
    cell_metrics: Dict[int, Dict]  # {cell_id: {queue_len, rb_util, power}}


class NS3Interface:
    """
    Interface to ns-O-RAN simulation via E2 protocol
    Handles E2SM-KPM (monitoring) and E2SM-RC (control)
    """
    
    def __init__(self, config: NS3Config):
        self.config = config
        self.ns3_process = None
        self.e2_socket = None
        self.current_step = 0
        
    def start_simulation(self):
        """Launch ns-3 simulation with O-RAN E2 interface enabled"""
        # Resolve ns3 script path
        ns3_path = 'ns3'
        if not os.path.exists(ns3_path):
            if os.path.exists(os.path.join('..', 'ns3')):
                ns3_path = os.path.join('..', 'ns3')
            elif os.path.exists(os.path.join('..', '..', 'ns3')): # Handle scratch/Radio-Cortex case
                ns3_path = os.path.join('..', '..', 'ns3')
        
        ns3_cmd = [
            sys.executable, ns3_path, 'run',
            f'scratch/oran-congestion-scenario',
            '--',
            f'--numUes={self.config.num_ues}',
            f'--numCells={self.config.num_cells}',
            f'--simTime={self.config.sim_time}',
            f'--seed={self.config.seed}',
            f'--e2Port={self.config.e2_port}',
            f'--kpmInterval={self.config.kpm_interval_ms}',
            '--enableE2=true'
        ]
        
        # Start ns-3 in subprocess
        self.ns3_process = subprocess.Popen(
            ns3_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
        # Connect to E2 interface (SCTP socket) with retry
        self._connect_e2()
        
    def _connect_e2(self):
        """Establish E2 connection with ns-3 simulation with retry logic"""
        self.e2_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        max_retries = 150  # 300 seconds total wait for build
        retry_interval = 2
        
        print(f"Waiting for ns-3 E2 interface on port {self.config.e2_port} (timeout: {max_retries*retry_interval}s)...")
        
        for i in range(max_retries):
            try:
                self.e2_socket.connect(('localhost', self.config.e2_port))
                print(f"✓ Connected to ns-3 E2 interface on port {self.config.e2_port}")
                return
            except ConnectionRefusedError:
                if i < max_retries - 1:
                    time.sleep(retry_interval)
                    # Check if process is still alive
                    if self.ns3_process.poll() is not None:
                        stdout, stderr = self.ns3_process.communicate()
                        raise RuntimeError(f"ns-3 process died unexpectedly.\nSTDOUT: {stdout.decode()}\nSTDERR: {stderr.decode()}")
                else:
                    raise
            except Exception as e:
                print(f"✗ Failed to connect to E2 interface: {e}")
                raise
    
    def receive_kpm_report(self) -> E2Message:
        """
        Receive E2SM-KPM report from ns-3
        Returns current network state metrics
        """
        try:
            # Receive E2AP message (simplified - real impl uses ASN.1 encoding)
            # Loop to handle partial reads/stream buffering
            buffer = ""
            while True:
                chunk = self.e2_socket.recv(4096).decode()
                if not chunk:
                    raise ConnectionError("Socket connection closed")
                buffer += chunk
                try:
                    # Try to parse JSON from the buffer
                    # Note: This is a simple implementation assuming one JSON object per packet
                    # or that recv gets the full JSON. In TCP streams, we might get partials.
                    # A robust implementation would use a delimiter or length prefix.
                    # For now, let's assume ns-3 sends one full JSON string which might be fragmented
                    # But Python's json.loads picks it up if it's valid.
                    
                    # Hack: The C++ side sends "}{" if multiple messages are concatenated quickly
                    # We might need to handle stream delimiters. 
                    # Let's try to find the first complete JSON object.
                    
                    # Brute force: find matching braces
                    depth = 0
                    start_idx = buffer.find('{')
                    if start_idx == -1:
                        if len(buffer) > 8192: buffer = "" # Clear garbage
                        continue

                    for i, char in enumerate(buffer[start_idx:], start_idx):
                        if char == '{': depth +=1
                        elif char == '}': depth -=1
                        
                        if depth == 0:
                            # Found complete object
                            json_str = buffer[start_idx:i+1]
                            kpm_data = json.loads(json_str)
                            # Keep the rest of the buffer for next time? 
                            # For gym step(), we just need one fresh state.
                            # It's better to discard old buffer to behave like a sample hold.
                            return self._parse_kpm(kpm_data)
                            
                except json.JSONDecodeError:
                    continue # Wait for more data
        except Exception as e:
            # print(f"Error receiving KPM: {e}")
            return self._get_default_metrics()
            
    def _parse_kpm(self, kpm_data):
        # Parse KPM metrics
        ue_metrics = {}
        for ue_id in range(self.config.num_ues):
            ue_metrics[ue_id] = {
                'throughput': kpm_data.get(f'ue_{ue_id}_tput', 0.0),  # Mbps
                'delay': kpm_data.get(f'ue_{ue_id}_delay', 0.0),  # ms
                'packet_loss': kpm_data.get(f'ue_{ue_id}_loss', 0.0),  # ratio
                'sinr': kpm_data.get(f'ue_{ue_id}_sinr', 0.0),  # dB
                'rb_allocated': kpm_data.get(f'ue_{ue_id}_rbs', 0),
            }
        
        cell_metrics = {}
        for cell_id in range(self.config.num_cells):
            cell_metrics[cell_id] = {
                'queue_length': kpm_data.get(f'cell_{cell_id}_queue', 0),
                'rb_utilization': kpm_data.get(f'cell_{cell_id}_rb_util', 0.0),
                'tx_power': kpm_data.get(f'cell_{cell_id}_power', 23.0),  # dBm
                'num_connected_ues': kpm_data.get(f'cell_{cell_id}_ues', 0),
            }
        
        return E2Message(
            timestamp=time.time(),
            ue_metrics=ue_metrics,
            cell_metrics=cell_metrics
        )
    
    def send_rc_control(self, actions: Dict):
        """
        Send E2SM-RC control message to ns-3
        Applies RL agent's actions to the RAN
        """
        rc_message = {
            'type': 'E2SM_RC',
            'actions': actions,
            'timestamp': time.time()
        }
        
        try:
            self.e2_socket.send(json.dumps(rc_message).encode())
        except Exception as e:
            # print(f"Error sending RC control: {e}")
            pass # Suppress send error to continue training loop if simulation ended
    
    def _get_default_metrics(self) -> E2Message:
        """Fallback metrics if E2 connection fails"""
        return E2Message(
            timestamp=time.time(),
            ue_metrics={i: {'throughput': 0, 'delay': 0, 'packet_loss': 0, 'sinr': -10, 'rb_allocated': 0} 
                       for i in range(self.config.num_ues)},
            cell_metrics={i: {'queue_length': 0, 'rb_utilization': 0, 'tx_power': 23, 'num_connected_ues': 0}
                         for i in range(self.config.num_cells)}
        )
    
    def stop_simulation(self):
        """Clean shutdown of ns-3 and E2 connection"""
        if self.e2_socket:
            self.e2_socket.close()
        if self.ns3_process:
            self.ns3_process.terminate()
            self.ns3_process.wait()


class ORANns3Env(gym.Env):
    """
    OpenAI Gym environment for O-RAN RAN Intelligent Controller (RIC)
    
    State Space: Network KPMs from E2SM-KPM
    Action Space: RAN control parameters via E2SM-RC
    Reward: Network performance (throughput, latency, fairness)
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, config: Optional[NS3Config] = None):
        super().__init__()
        
        self.config = config or NS3Config()
        self.ns3 = NS3Interface(self.config)
        
        # State space: flattened network metrics
        # [per-UE: throughput, delay, loss, sinr] + [per-cell: queue, rb_util, power]
        state_dim = (
            self.config.num_ues * 4 +  # UE metrics
            self.config.num_cells * 3   # Cell metrics
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )
        
        # Action space: per-cell control parameters (from Table 1)
        # [tx_power, scheduler_type, max_harq_tx, handover_hysteresis]
        self.action_space = spaces.Box(
            low=np.array([
                10.0,  # TxPower min (dBm)
                0.0,   # SchedulerType (discrete, normalized)
                1.0,   # MaxHarqTx min
                0.0,   # Hysteresis min (dB)
            ] * self.config.num_cells),
            high=np.array([
                46.0,  # TxPower max
                2.0,   # SchedulerType max
                8.0,   # MaxHarqTx max
                6.0,   # Hysteresis max
            ] * self.config.num_cells),
            dtype=np.float32
        )
        
        # Episode tracking
        self.current_step = 0
        self.max_steps = int(self.config.sim_time * 1000 / self.config.kpm_interval_ms)
        self.episode_metrics = []
        
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        """Reset environment and start new ns-3 simulation episode"""
        super().reset(seed=seed)
        
        if seed is not None:
            self.config.seed = seed
        
        # Stop previous simulation if running
        if hasattr(self, 'ns3') and self.ns3.ns3_process:
            self.ns3.stop_simulation()
        
        # Start fresh ns-3 simulation
        self.ns3 = NS3Interface(self.config)
        self.ns3.start_simulation()
        
        # Get initial state
        time.sleep(0.1)  # Wait for first KPM report
        e2_msg = self.ns3.receive_kpm_report()
        state = self._extract_state(e2_msg)
        
        self.current_step = 0
        self.episode_metrics = []
        
        info = {
            'episode': 0,
            'seed': self.config.seed
        }
        
        return state, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one timestep of RL control loop
        
        Args:
            action: RAN control parameters [tx_power, scheduler, harq, hysteresis] per cell
            
        Returns:
            observation: Network state
            reward: Performance metric
            terminated: Episode complete
            truncated: Max steps reached
            info: Additional metrics
        """
        # Parse action into control parameters
        rc_actions = self._parse_action(action)
        
        # Send E2SM-RC control to ns-3
        self.ns3.send_rc_control(rc_actions)
        
        # Wait for next KPM interval
        time.sleep(self.config.kpm_interval_ms / 1000.0)
        
        # Receive new state from E2SM-KPM
        e2_msg = self.ns3.receive_kpm_report()
        next_state = self._extract_state(e2_msg)
        
        # Calculate reward
        reward = self._compute_reward(e2_msg)
        
        # Track metrics
        self.episode_metrics.append({
            'step': self.current_step,
            'reward': reward,
            'avg_throughput': np.mean([m['throughput'] for m in e2_msg.ue_metrics.values()]),
            'avg_delay': np.mean([m['delay'] for m in e2_msg.ue_metrics.values()]),
            'avg_loss': np.mean([m['packet_loss'] for m in e2_msg.ue_metrics.values()]),
        })
        
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        info = {
            'step': self.current_step,
            'e2_metrics': e2_msg,
            'actions_applied': rc_actions
        }
        
        return next_state, reward, terminated, truncated, info
    
    def _extract_state(self, e2_msg: E2Message) -> np.ndarray:
        """Convert E2 KPM message to RL state vector"""
        state = []
        
        # UE metrics
        for ue_id in range(self.config.num_ues):
            ue = e2_msg.ue_metrics.get(ue_id, {})
            state.extend([
                ue.get('throughput', 0.0) / 100.0,  # Normalize to ~[0,1]
                ue.get('delay', 0.0) / 1000.0,      # Normalize ms
                ue.get('packet_loss', 0.0),         # Already ratio
                (ue.get('sinr', 0.0) + 10) / 40.0,  # Normalize SINR [-10,30]dB
            ])
        
        # Cell metrics
        for cell_id in range(self.config.num_cells):
            cell = e2_msg.cell_metrics.get(cell_id, {})
            state.extend([
                cell.get('queue_length', 0) / 1000.0,  # Normalize queue
                cell.get('rb_utilization', 0.0),       # Already ratio
                (cell.get('tx_power', 23.0) - 10) / 36.0,  # Normalize power
            ])
        
        return np.array(state, dtype=np.float32)
    
    def _parse_action(self, action: np.ndarray) -> Dict:
        """Convert RL action vector to E2SM-RC control parameters"""
        rc_actions = {}
        
        for cell_id in range(self.config.num_cells):
            idx = cell_id * 4
            rc_actions[f'cell_{cell_id}'] = {
                'TxPower': float(action[idx]),
                'SchedulerType': SchedulerType(int(action[idx + 1])).name,
                'MaxHarqTx': int(action[idx + 2]),
                'Hysteresis': float(action[idx + 3]),
            }
        
        return rc_actions
    
    def _compute_reward(self, e2_msg: E2Message) -> float:
        """
        Reward function optimizing for:
        1. High aggregate throughput
        2. Low latency
        3. Low packet loss
        4. Fairness (Jain's index)
        """
        throughputs = [m['throughput'] for m in e2_msg.ue_metrics.values()]
        delays = [m['delay'] for m in e2_msg.ue_metrics.values()]
        losses = [m['packet_loss'] for m in e2_msg.ue_metrics.values()]
        
        # Throughput reward (higher is better)
        avg_tput = np.mean(throughputs)
        tput_reward = avg_tput / 10.0  # Normalize
        
        # Latency penalty (lower is better)
        avg_delay = np.mean(delays)
        delay_penalty = -avg_delay / 100.0
        
        # Packet loss penalty (catastrophic if high)
        avg_loss = np.mean(losses)
        loss_penalty = -100.0 * avg_loss
        
        # Fairness (Jain's index)
        if sum(throughputs) > 0:
            fairness = (sum(throughputs) ** 2) / (len(throughputs) * sum([t**2 for t in throughputs]))
        else:
            fairness = 0
        fairness_reward = fairness
        
        # Combined reward
        reward = (
            1.0 * tput_reward +
            0.5 * delay_penalty +
            2.0 * loss_penalty +
            0.3 * fairness_reward
        )
        
        return reward
    
    def render(self, mode='human'):
        """Visualize current network state"""
        if not self.episode_metrics:
            print("No metrics to render yet")
            return
        
        latest = self.episode_metrics[-1]
        print(f"\n=== Step {latest['step']} ===")
        print(f"Reward: {latest['reward']:.3f}")
        print(f"Avg Throughput: {latest['avg_throughput']:.2f} Mbps")
        print(f"Avg Delay: {latest['avg_delay']:.2f} ms")
        print(f"Avg Loss: {latest['avg_loss']:.4f}")
    
    def close(self):
        """Cleanup ns-3 simulation"""
        self.ns3.stop_simulation()


# ============================================================================
# RL Training Integration
# ============================================================================

def create_oran_env(config: Optional[NS3Config] = None) -> ORANns3Env:
    """Factory function for creating O-RAN environment"""
    return ORANns3Env(config)


if __name__ == "__main__":
    # Demo: Create environment and run random policy
    print("Creating O-RAN ns-3 Environment...")
    
    config = NS3Config(
        num_ues=20,
        num_cells=3,
        sim_time=10.0,
        kpm_interval_ms=100  # 100ms for demo
    )
    
    env = create_oran_env(config)
    
    print(f"State space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Run one episode with random actions
    obs, info = env.reset()
    print(f"\nInitial state shape: {obs.shape}")
    
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step {step}: Reward = {reward:.3f}")
        
        if terminated or truncated:
            break
    
    env.close()
    print("\n✓ Environment test complete")

