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
    seed: int = 42
    e2_port: int = 36421
    kpm_interval_ms: int = 10  # E2SM-KPM reporting interval
    scenario: str = "flash_crowd"  # Scenario to run


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
        self.kafka_consumer = None
        self.kafka_producer = None
        self.current_step = 0
        self.last_kpm_ts = None
        
    def start_simulation(self):
        """Launch ns-3 simulation with O-RAN E2 interface enabled"""
        # Resolve ns3 script path
        ns3_path = 'ns3'
        if not os.path.exists(ns3_path):
            # Check for standard nested structure
            nested_path = os.path.join('ns-allinone-3.46.1', 'ns-3.46.1', 'ns3')
            if os.path.exists(nested_path):
                ns3_path = nested_path
            elif os.path.exists(os.path.join('..', 'ns3')):
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
            f'--kpmInterval={self.config.kpm_interval_ms}',
            f'--kpmInterval={self.config.kpm_interval_ms}',
            '--enableE2=true',
            f'--scenario={self.config.scenario}'
        ]
        
        # Start ns-3 in subprocess
        # Native Kafka support in ns-3, no adapter needed.
        # Redirect output to file for debugging
        self.ns3_log_file = open("ns3.log", "w")
        self.ns3_process = subprocess.Popen(
            ns3_cmd,
            stdout=self.ns3_log_file,
            stderr=subprocess.STDOUT,
        )
        
        time.sleep(2) # Give ns-3 time to initialize
        
        # Connect to Kafka
        self._connect_kafka()
        
    def _connect_kafka(self):
        """Establish Kafka connections"""
        from kafka import KafkaConsumer, KafkaProducer
        import socket

        try:
            print("Connecting to Kafka...")
            self.kafka_consumer = KafkaConsumer(
                'e2_kpm_stream',
                bootstrap_servers=['localhost:9092'],
                auto_offset_reset='latest',
                enable_auto_commit=False,
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                consumer_timeout_ms=30000  # Non-blocking check
            )

            # Ensure we only consume NEW messages from this point onward
            self.kafka_consumer.poll(timeout_ms=30000)
            partitions = self.kafka_consumer.assignment()
            if partitions:
                self.kafka_consumer.seek_to_end(*partitions)
            self.last_kpm_ts = None
            
            self.kafka_producer = KafkaProducer(
                bootstrap_servers=['localhost:9092'],
                value_serializer=lambda x: json.dumps(x).encode('utf-8')
            )
            print("✓ Connected to Kafka")
            
        except Exception as e:
            print(f"✗ Failed to connect to Kafka: {e}")
            raise
    
    def receive_kpm_report(self, wait_for_new: bool = True, max_wait_s: Optional[float] = None) -> E2Message:
        """
        Receive E2SM-KPM report from Kafka ('e2_kpm_stream')
        Returns current network state metrics
        """
        try:
            # Poll for new messages
            # We want the LATEST message for the current step
            if max_wait_s is None:
                max_wait_s = max(1.0, (self.config.kpm_interval_ms / 1000.0) * 2)

            deadline = time.time() + max_wait_s
            last_record = None

            while True:
                records = self.kafka_consumer.poll(timeout_ms=30000)
                if records:
                    for partition, messages in records.items():
                        if messages:
                            candidate = messages[-1]
                            if self.last_kpm_ts is None or candidate.timestamp > self.last_kpm_ts:
                                last_record = candidate

                    if last_record is not None:
                        break

                if not wait_for_new or time.time() >= deadline:
                    break
            
            if not records:
                # No data yet, return defaults or wait?
                # For training, we need data.
                print("No KPM data received, returning default metrics")
                
                return self._get_default_metrics()
            
            if last_record:
                self.last_kpm_ts = last_record.timestamp
                kpm_data = last_record.value
                # DEBUG: Print keys from first few reports to verify JSON structure
                if getattr(self, '_debug_kpm_count', 0) < 5:
                    #print(f"DEBUG: Received KPM keys: {list(kpm_data.keys())} Sample: {kpm_data}")
                    self._debug_kpm_count = getattr(self, '_debug_kpm_count', 0) + 1
                return self._parse_kpm(kpm_data)
            else:
                return self._get_default_metrics()
            
        except Exception as e:
            print(f"Error receiving KPM: {e}")
            return self._get_default_metrics()
            
    def _parse_kpm(self, kpm_data):
        # Parse KPM metrics
        ue_metrics = {}
      #  print(kpm_data.keys())
        for ue_id in range(self.config.num_ues):
            ue_metrics[ue_id] = {
                'throughput': kpm_data.get(f'ue_{ue_id}_tput', 0.0),  # Mbps
                'delay': kpm_data.get(f'ue_{ue_id}_delay', 0.0),  # ms
                'packet_loss': kpm_data.get(f'ue_{ue_id}_loss', 0.0),  # ratio
                'sinr': kpm_data.get(f'ue_{ue_id}_sinr', 0.0),  # dB
                'rsrp': kpm_data.get(f'ue_{ue_id}_rsrp', -140.0),  # dBm
                'rsrq': kpm_data.get(f'ue_{ue_id}_rsrq', -20.0),  # dB
                'ul_rbs': kpm_data.get(f'ue_{ue_id}_ul_rbs', 0.0),  # avg RBs
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
        Send E2SM-RC control message to Kafka ('e2_rc_control')
        Applies RL agent's actions to the RAN
        """
        rc_message = {
            'type': 'E2SM_RC',
            'actions': actions,
            'timestamp': time.time()
        }
        
        try:
            self.kafka_producer.send('e2_rc_control', rc_message)
            self.kafka_producer.flush()
        except Exception as e:
            # print(f"Error sending RC control: {e}")
            pass 
    
    def _get_default_metrics(self) -> E2Message:
        """Fallback metrics if E2 connection fails"""
        return E2Message(
            timestamp=time.time(),
            ue_metrics={i: {'throughput': 0, 'delay': 0, 'packet_loss': 0, 'sinr': -10,
                            'rsrp': -140, 'rsrq': -20, 'ul_rbs': 0.0, 'rb_allocated': 0}
                       for i in range(self.config.num_ues)},
            cell_metrics={i: {'queue_length': 0, 'rb_utilization': 0, 'tx_power': 23, 'num_connected_ues': 0}
                         for i in range(self.config.num_cells)}
        )
    
    def stop_simulation(self):
        """Clean shutdown of ns-3 and Kafka connection"""
        if self.kafka_consumer:
            self.kafka_consumer.close()
        if self.kafka_producer:
            self.kafka_producer.close()
            
        if hasattr(self, 'adapter_process') and self.adapter_process:
            self.adapter_process.terminate()
            self.adapter_process.wait()
            
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
        # [per-UE: throughput, delay, loss, sinr, rsrp, rsrq, ul_rbs] + [per-cell: queue, rb_util, power]
        state_dim = (
            self.config.num_ues * 7 +  # UE metrics
            self.config.num_cells * 3   # Cell metrics
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )
        
        # Action space: per-cell control parameters
        # [tx_power, scheduler_type, max_harq_tx, handover_hysteresis, mac_ch_delay, noise_figure]
        self.action_space = spaces.Box(
            low=np.array([
                10.0,  # TxPower min (dBm)
                0.0,   # SchedulerType (discrete, normalized)
                1.0,   # MaxHarqTx min
                0.0,   # Hysteresis min (dB)
                0.0,   # MacChDelay min (TTIs)
                0.0,   # NoiseFigure min (dB)
            ] * self.config.num_cells),
            high=np.array([
                46.0,  # TxPower max
                2.0,   # SchedulerType max
                8.0,   # MaxHarqTx max
                6.0,   # Hysteresis max
                10.0,  # MacChDelay max (TTIs)
                10.0,  # NoiseFigure max (dB)
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
        e2_msg = self.ns3.receive_kpm_report(wait_for_new=False)
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

        # Receive new state from E2SM-KPM (synchronized to newest Kafka record)
        e2_msg = self.ns3.receive_kpm_report(wait_for_new=True, max_wait_s=(self.config.kpm_interval_ms / 1000.0) * 2)
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

        # If ns-3 simulation has ended, terminate the episode immediately
        if self.ns3.ns3_process and self.ns3.ns3_process.poll() is not None:
            terminated = True
        truncated = False
        
        ns3_finished = bool(self.ns3.ns3_process and self.ns3.ns3_process.poll() is not None)
        info = {
            'step': self.current_step,
            'e2_metrics': e2_msg,
            'actions_applied': rc_actions,
            'ns3_finished': ns3_finished
        }
        
        return next_state, reward, terminated, truncated, info
    
    def _extract_state(self, e2_msg: E2Message) -> np.ndarray:
        """Convert E2 KPM message to RL state vector"""
        state = []
        
        # UE metrics
        for ue_id in range(self.config.num_ues):
            ue = e2_msg.ue_metrics.get(ue_id, {})

            #print(ue.keys())
            state.extend([
                ue.get('throughput', 0.0) / 100.0,  # Normalize to ~[0,1]
                ue.get('delay', 0.0) / 1000.0,      # Normalize ms
                ue.get('packet_loss', 0.0),         # Already ratio
                (ue.get('sinr', 0.0) + 10) / 40.0,  # Normalize SINR [-10,30]dB
                (ue.get('rsrp', -140.0) + 140.0) / 100.0,  # RSRP [-140,-40] dBm
                (ue.get('rsrq', -20.0) + 20.0) / 20.0,     # RSRQ [-20,0] dB
                ue.get('ul_rbs', 0.0) / 100.0,       # Normalize avg UL RBs
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
            idx = cell_id * 6
            rc_actions[f'cell_{cell_id}'] = {
                'TxPower': float(action[idx]),
                'SchedulerType': SchedulerType(int(action[idx + 1])).name,
                'MaxHarqTx': int(action[idx + 2]),
                'Hysteresis': float(action[idx + 3]),
                'MacChDelay': float(action[idx + 4]),
                'NoiseFigure': float(action[idx + 5]),
            }
        return rc_actions
    
    def _compute_reward(self, e2_msg: E2Message) -> float:
        """
        Compute reward based on network performance
        Maximize Throughput and Fairness, Minimize Delay
        Adding SINR component to ensure gradient even without traffic
        """
        if not e2_msg.ue_metrics:
            return 0.0
            
        for m in e2_msg.ue_metrics.values():
            print(m)
        tputs = [m['throughput'] for m in e2_msg.ue_metrics.values()]
        delays = [m['delay'] for m in e2_msg.ue_metrics.values()]
        sinrs = [m['sinr'] for m in e2_msg.ue_metrics.values()]
        
        sum_log_tput = np.sum(np.log(np.array(tputs) + 1e-6)) # Proportional Fairness
        avg_delay = np.mean(delays)
        avg_sinr = np.mean(sinrs)
        
        # Fairness (Jain's index)
        if sum(tputs) > 0:
            fairness = (sum(tputs) ** 2) / (len(tputs) * sum(np.array(tputs) ** 2))
        else:
            fairness = 1.0
            
        # Reward components
        # 1. Throughput (Log utility)
        # 2. Delay penalty
        # 3. SINR bonus (0.05 * SINR_dB) -> e.g. 20dB -> +1.0
        # 4. Fairness bonus
        
        reward = sum_log_tput - (0.1 * avg_delay) + (0.05 * avg_sinr) + (0.5 * fairness)
        print(f"Reward components: Throughput={sum_log_tput:.3f}, Delay={avg_delay:.3f}, SINR={avg_sinr:.3f}, Fairness={fairness:.3f}, Total Reward={reward:.3f}")
        return float(reward)
    
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

