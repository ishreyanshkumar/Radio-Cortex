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
import random
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
    kpm_interval_ms: int = 100  # E2SM-KPM reporting interval
    system_bandwidth_mhz: float = 10.0 # System Bandwidth
    # Scenario to run (12 available: flash_crowd, mobility_storm, traffic_burst, handover_ping_pong, 
    # sleepy_campus, ambulance, adversarial, commuter_rush, mixed_reality, urban_canyon, iot_tsunami, spectrum_crunch)
    scenario: str = "flash_crowd"
    # Multi-scenario training: if set, reset() will randomly pick from this list each episode
    scenarios: Optional[List[str]] = None 


@dataclass
class E2Message:
    """E2 interface message format"""
    timestamp: float
    ue_metrics: Dict[int, Dict]  # {ue_id: {throughput, delay, loss, sinr}}
    cell_metrics: Dict[int, Dict]  # {cell_id: {queue_len, rb_util, power}}


class RewardEngine:
    """
    Hybrid Reward Engine for Radio-Cortex O-RAN.

    Combines:
      UE Utility  — Throughput (α-fairness log), Delay (linear + SLA barrier),
                     Packet Loss (IQX exponential), Spectral Efficiency (Shannon)
      Network     — Energy efficiency, Load balancing, Queue congestion, Action smoothing

    Safety:
      Stage 1 — per-component clipping (prevents any single term from dominating)
      Stage 2 — total reward clipped to [-10, +2]

    Diagnostics:
      Returns (reward, breakdown_dict) for logging and evaluation.
    """

    def __init__(self, config: NS3Config):
        self.config = config

        # ── Weights ──────────────────────────────────────────────
        self.W_TPUT      = 1.0      # Throughput (Log Utility)
        self.W_DELAY_LIN = 0.5      # Linear Delay Penalty
        self.W_DELAY_BAR = 5.0      # Quadratic SLA Barrier
        self.W_LOSS      = 2.0      # Packet Loss (IQX)
        self.W_SE        = 0.05     # Spectral Efficiency (keep-alive signal)
        self.W_ENERGY    = 0.5      # Energy Efficiency
        self.W_LOAD      = 1.0      # Load Balancing
        self.W_QUEUE     = 0.3      # Queue Congestion (NEW)
        self.W_SMOOTH    = 0.05     # Action Smoothing

        # ── Thresholds / Normalizers ─────────────────────────────
        self.T_MAX     = 100.0      # Max Throughput (Mbps)
        self.D_MAX     = 100.0      # Normalizing Delay (ms)
        self.D_SLA     = 50.0       # SLA Threshold (ms) — barrier kicks in after this
        self.BETA_LOSS = 5.0        # IQX Sensitivity parameter
        self.EPSILON   = 1e-6       # Safe log
        self.Q_MAX     = 1000.0     # Queue length normalizer (NEW)

        # ── Clip bounds (Stage 1 — per component) ───────────────
        self.CLIP_TPUT   = (-0.5, 5.0)    # log(1+x) for x≥0 is ≥0, but allow small neg for numerical safety
        self.CLIP_DELAY  = (-5.0, 0.0)    # Delay is ALWAYS a penalty (≤0)
        self.CLIP_LOSS   = (-5.0, 0.0)    # Loss is ALWAYS a penalty (≤0)
        self.CLIP_SE     = (0.0, 2.0)     # SE is ALWAYS a bonus (≥0)
        self.CLIP_ENERGY = (-2.0, 0.0)    # Energy is ALWAYS a penalty (≤0)
        self.CLIP_LOAD   = (-2.0, 0.0)    # Load imbalance is ALWAYS a penalty (≤0)
        self.CLIP_QUEUE  = (-2.0, 0.0)    # Queue congestion is ALWAYS a penalty (≤0)
        self.CLIP_SMOOTH = (-1.0, 0.0)    # Smoothing is ALWAYS a penalty (≤0)

        # ── Clip bounds (Stage 2 — total) ────────────────────────
        self.CLIP_TOTAL = (-10.0, 2.0)

    def compute(self,
                e2_msg: E2Message,
                action: np.ndarray = None,
                prev_action: np.ndarray = None,
                action_space: spaces.Box = None
                ) -> Tuple[float, Dict[str, float]]:
        """
        Compute scalar reward and per-component breakdown.

        Returns
        -------
        reward    : float, clipped to CLIP_TOTAL
        breakdown : dict with every component + diagnostic metrics
        """
        empty = self._empty_breakdown()
        if not e2_msg.ue_metrics:
            return 0.0, empty

        # ════════════════════════════════════════════════════════
        # 1.  UE UTILITY  (User Satisfaction)
        # ════════════════════════════════════════════════════════
        tputs  = np.array([m['throughput']  for m in e2_msg.ue_metrics.values()])
        delays = np.array([m['delay']       for m in e2_msg.ue_metrics.values()])
        losses = np.array([m['packet_loss'] for m in e2_msg.ue_metrics.values()])
        sinrs  = np.array([m['sinr']        for m in e2_msg.ue_metrics.values()])

        # ── 1a. Throughput  (α-fairness, α=1 → log utility) ────
        #   log(1 + T/T_MAX) :  0 Mbps→0,  50 Mbps→0.41,  100 Mbps→0.69
        #   Averaged over UEs for per-user fairness.
        r_tput = float(np.mean(np.log(1.0 + tputs / self.T_MAX + self.EPSILON))) * self.W_TPUT
        r_tput = float(np.clip(r_tput, *self.CLIP_TPUT))

        # ── 1b. Delay  (Two-tier: linear everywhere + quadratic past SLA) ──
        d_norm    = np.minimum(delays / self.D_MAX, 1.0)                    # Tier 1: 0→1 linear
        d_barrier = np.maximum(delays - self.D_SLA, 0.0) ** 2              # Tier 2: explodes past SLA
        r_delay = float(-np.mean(
            self.W_DELAY_LIN * d_norm
            + (self.W_DELAY_BAR / self.D_MAX ** 2) * d_barrier
        ))
        r_delay = float(np.clip(r_delay, *self.CLIP_DELAY))

        # ── 1c. Packet Loss  (IQX exponential) ─────────────────
        #   exp(5 × 0.01)-1 = 0.05  (1% loss → mild)
        #   exp(5 × 0.10)-1 = 0.65  (10% loss → painful)
        #   exp(5 × 0.50)-1 = 11.2  (50% loss → catastrophic)
        r_loss = float(-np.mean(np.exp(self.BETA_LOSS * losses) - 1.0)) * self.W_LOSS
        r_loss = float(np.clip(r_loss, *self.CLIP_LOSS))

        # ── 1d. Spectral Efficiency  (Shannon keep-alive) ──────
        #   Gives the AI a gradient even when no traffic is flowing,
        #   rewarding good channel conditions.
        sinr_linear = np.power(10.0, sinrs / 10.0)
        se_per_ue   = np.log2(1.0 + sinr_linear)           # bits/s/Hz per UE
        se_avg      = float(np.mean(se_per_ue))
        r_se = se_avg * self.W_SE
        r_se = float(np.clip(r_se, *self.CLIP_SE))

        ue_score = r_tput + r_delay + r_loss + r_se

        # ════════════════════════════════════════════════════════
        # 2.  NETWORK UTILITY  (Efficiency & Stability)
        # ════════════════════════════════════════════════════════
        r_energy = 0.0
        r_load   = 0.0
        r_queue  = 0.0
        r_smooth = 0.0

        if e2_msg.cell_metrics:
            loads  = np.array([m['cell_load']      for m in e2_msg.cell_metrics.values()])
            rbs    = np.array([m['avg_rb_request']  for m in e2_msg.cell_metrics.values()])
            queues = np.array([m['queue_length']    for m in e2_msg.cell_metrics.values()])

            # ── 2a. Energy Efficiency ───────────────────────────
            #   Penalizes high resource block usage across cells.
            #   50 RBs × 100 TTIs = 5000 max RBs per interval.
            total_rbs_max = 5000.0
            r_energy = float(-np.mean(rbs / total_rbs_max)) * self.W_ENERGY
            r_energy = float(np.clip(r_energy, *self.CLIP_ENERGY))

            # ── 2b. Load Balancing ──────────────────────────────
            #   Penalizes uneven load across cells.
            #   std(loads) is high when one cell is overloaded
            #   and another is idle → AI learns to distribute.
            r_load = float(-np.std(loads)) * self.W_LOAD
            r_load = float(np.clip(r_load, *self.CLIP_LOAD))

            # ── 2c. Queue Congestion  (NEW) ─────────────────────
            #   Penalizes cells with growing queues.
            #   This gives early warning BEFORE delay spikes —
            #   a full queue today = high delay tomorrow.
            q_norm = np.minimum(queues / self.Q_MAX, 1.0)
            r_queue = float(-np.mean(q_norm)) * self.W_QUEUE
            r_queue = float(np.clip(r_queue, *self.CLIP_QUEUE))

        network_score = r_energy + r_load + r_queue

        # ── 2d. Action Smoothing (normalized by range) ──────────
        #   Penalizes the AI for jerky, oscillating actions.
        #   Normalizing by range makes power change (10→46 dBm)
        #   comparable to scheduler change (0→2).
        if prev_action is not None and action is not None:
            if action_space is not None:
                a_range = action_space.high - action_space.low
                a_range = np.where(a_range < 1e-6, 1.0, a_range)
                delta = float(np.mean(np.abs(action - prev_action) / a_range))
            else:
                delta = float(np.mean(np.abs(action - prev_action)))
            r_smooth = float(-delta * self.W_SMOOTH)
            r_smooth = float(np.clip(r_smooth, *self.CLIP_SMOOTH))
            network_score += r_smooth

        # ════════════════════════════════════════════════════════
        # 3.  AGGREGATE & BOUND
        # ════════════════════════════════════════════════════════
        total_raw = ue_score + network_score
        total     = float(np.clip(total_raw, *self.CLIP_TOTAL))

        # ════════════════════════════════════════════════════════
        # 4.  DIAGNOSTICS  (for MODULE 5 / debugging)
        # ════════════════════════════════════════════════════════
        jains    = float(self._jains_index(tputs))
        p95_dly  = float(np.percentile(delays, 95)) if len(delays) > 0 else 0.0
        avg_tput = float(np.mean(tputs))
        avg_dly  = float(np.mean(delays))
        avg_loss = float(np.mean(losses))

        breakdown = {
            # Per-component rewards
            'r_total':  total,
            'r_tput':   r_tput,
            'r_delay':  r_delay,
            'r_loss':   r_loss,
            'r_se':     r_se,
            'r_energy': r_energy,
            'r_load':   r_load,
            'r_queue':  r_queue,
            'r_smooth': r_smooth,
            # Diagnostic KPIs (not part of reward, but essential for logging)
            'se_avg':       se_avg,
            'jains':        jains,
            'p95_delay':    p95_dly,
            'avg_throughput': avg_tput,
            'avg_delay':    avg_dly,
            'avg_loss':     avg_loss,
        }

        return total, breakdown

    # ── Helpers ──────────────────────────────────────────────────
    @staticmethod
    def _jains_index(x: np.ndarray) -> float:
        """Jain's Fairness Index: 1.0 = perfectly equal, 1/N = maximally unfair."""
        s = float(np.sum(x))
        if s <= 0:
            return 0.0
        return (s ** 2) / (len(x) * float(np.sum(x ** 2)))

    def _empty_breakdown(self) -> Dict[str, float]:
        """Return zeroed breakdown dict (for edge cases like empty metrics)."""
        return {
            'r_total': 0.0, 'r_tput': 0.0, 'r_delay': 0.0, 'r_loss': 0.0,
            'r_se': 0.0, 'r_energy': 0.0, 'r_load': 0.0, 'r_queue': 0.0,
            'r_smooth': 0.0, 'se_avg': 0.0, 'jains': 0.0, 'p95_delay': 0.0,
            'avg_throughput': 0.0, 'avg_delay': 0.0, 'avg_loss': 0.0,
        }


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
        
        # Phase 2 Metrics
        self.kpm_msg_count = 0
        self.rc_msg_count = 0
        self.e2_loop_latencies = []
        self.start_time = time.time()
        self.last_kpm_rx_time = 0.0
        
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

            '--enableE2=true',
            f'--scenario={self.config.scenario}',
            f'--bandwidthRbs={int(self.config.system_bandwidth_mhz * 5)}' # 10MHz * 5 = 50 RBs
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
                
                self.last_kpm_rx_time = time.time() # Timestamp for E2 Latency
                self.kpm_msg_count += 1
                return self._parse_kpm(kpm_data)
            else:
                return self._get_default_metrics()
            
        except Exception as e:
            print(f"Error receiving KPM: {e}")
            return self._get_default_metrics()
            
    def _parse_kpm(self, kpm_data):
        # Parse KPM metrics
        ue_metrics = {}
        
        # Track how many UEs got non-default values for key metrics
        rsrp_real_count = 0
        cell_real_count = 0
        ho_real_count = 0
        
        for ue_id in range(self.config.num_ues):
            rsrp_val = kpm_data.get(f'ue_{ue_id}_rsrp', -140.0)
            rsrq_val = kpm_data.get(f'ue_{ue_id}_rsrq', -20.0)
            cell_val = kpm_data.get(f'ue_{ue_id}_cell', -1)
            ho_att = kpm_data.get(f'ue_{ue_id}_ho_att', 0)
            
            # Track non-default counts
            if rsrp_val != -140.0:
                rsrp_real_count += 1
            if cell_val != -1:
                cell_real_count += 1
            if ho_att > 0:
                ho_real_count += 1
            
            ue_metrics[ue_id] = {
                'throughput': kpm_data.get(f'ue_{ue_id}_tput', 0.0),  # Mbps
                'delay': kpm_data.get(f'ue_{ue_id}_delay', 0.0),  # ms
                'packet_loss': kpm_data.get(f'ue_{ue_id}_loss', 0.0),  # ratio
                'sinr': kpm_data.get(f'ue_{ue_id}_sinr', 0.0),  # dB
                'rsrp': rsrp_val,  # dBm
                'rsrq': rsrq_val,  # dB
                'ul_rbs': kpm_data.get(f'ue_{ue_id}_ul_rbs', 0.0),  # avg RBs
                'rb_allocated': kpm_data.get(f'ue_{ue_id}_rbs', 0),
                'cqi': kpm_data.get(f'ue_{ue_id}_cqi', 0.0),
                'rsrp_var': kpm_data.get(f'ue_{ue_id}_rsrp_var', 0.0),
                'rsrq_var': kpm_data.get(f'ue_{ue_id}_rsrq_var', 0.0),
                'buffer_occupancy': kpm_data.get(f'ue_{ue_id}_buffer', 0.0),
                'serving_cell': cell_val,
                'handover_attempts': ho_att,
                'handover_successes': kpm_data.get(f'ue_{ue_id}_ho_succ', 0),
            }
        
        # Print data quality summary every 10 KPM reports
        if not hasattr(self, '_kpm_parse_count'):
            self._kpm_parse_count = 0
        self._kpm_parse_count += 1
        if self._kpm_parse_count % 50 == 1:
            num_ues = self.config.num_ues
            print(f"\n  [Data Quality] RSRP: {rsrp_real_count}/{num_ues} real | Cell: {cell_real_count}/{num_ues} real | HO: {ho_real_count}/{num_ues} with events")
        
        cell_metrics = {}
        for cell_id in range(self.config.num_cells):
            cell_metrics[cell_id] = {
                'queue_length': kpm_data.get(f'cell_{cell_id}_queue', 0),
                'rb_utilization': kpm_data.get(f'cell_{cell_id}_rb_util', 0.0),
                'tx_power': kpm_data.get(f'cell_{cell_id}_power', 23.0),  # dBm
                'num_connected_ues': kpm_data.get(f'cell_{cell_id}_ues', 0),
                'cell_load': kpm_data.get(f'cell_{cell_id}_load', 0.0),
                'avg_rb_request': kpm_data.get(f'cell_{cell_id}_avg_rb_req', 0.0),
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
            
            self.rc_msg_count += 1
            if hasattr(self, 'last_kpm_rx_time') and self.last_kpm_rx_time > 0:
                latency = (time.time() - self.last_kpm_rx_time) * 1000.0 # ms
                self.e2_loop_latencies.append(latency)
        except Exception as e:
            # print(f"Error sending RC control: {e}")
            pass 
    
    def _get_default_metrics(self) -> E2Message:
        """Fallback metrics if E2 connection fails"""
        return E2Message(
            timestamp=time.time(),
            ue_metrics={i: {
                'throughput': 0, 'delay': 0, 'packet_loss': 0, 'sinr': -10,
                'rsrp': -140, 'rsrq': -20, 'ul_rbs': 0.0, 'rb_allocated': 0,
                'cqi': 0, 'rsrp_var': 0, 'rsrq_var': 0, 'buffer_occupancy': 0,
                'serving_cell': -1, 'handover_attempts': 0, 'handover_successes': 0
            } for i in range(self.config.num_ues)},
            cell_metrics={i: {'queue_length': 0, 'rb_utilization': 0, 'tx_power': 23, 
                             'num_connected_ues': 0, 'cell_load': 0, 'avg_rb_request': 0}
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
        self.reward_engine = RewardEngine(self.config)
        self.prev_action = None
        
        # State space: flattened network metrics
        # Per-UE features: throughput, delay, loss, sinr, rsrp, rsrq, ul_rbs,
        #                rb_allocated, cqi, rsrp_var, rsrq_var, buffer_occupancy  (12 per UE)
        # Per-cell features: queue_length, rb_utilization, tx_power, cell_load, avg_rb_request (5 per cell)
        state_dim = (
            self.config.num_ues * 12 +  # UE metrics (expanded)
            self.config.num_cells * 5    # Cell metrics (expanded)
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )
        
        # Action space: per-cell + per-UE controls
        # Per-cell: [TxPower, SchedulerType, MaxHarqTx, Hysteresis, MacChDelay, NoiseFigure, SchedulerWeight]
        # Per-UE: [priority_weight] for each UE
        per_cell_low = [
            10.0,  # TxPower min (dBm)
            0.0,   # SchedulerType (discrete index)
            1.0,   # MaxHarqTx min
            0.0,   # Hysteresis min (dB)
            0.0,   # MacChDelay min (TTIs)
            0.0,   # NoiseFigure min (dB)
            0.0,   # SchedulerWeight min
        ]
        per_cell_high = [
            46.0,  # TxPower max
            2.0,   # SchedulerType max
            8.0,   # MaxHarqTx max
            6.0,   # Hysteresis max
            10.0,  # MacChDelay max (TTIs)
            10.0,  # NoiseFigure max (dB)
            5.0,   # SchedulerWeight max
        ]

        # Per-UE priority weight bounds
        per_ue_low = [0.0] * self.config.num_ues
        per_ue_high = [10.0] * self.config.num_ues

        low = np.array(per_cell_low * self.config.num_cells + per_ue_low, dtype=np.float32)
        high = np.array(per_cell_high * self.config.num_cells + per_ue_high, dtype=np.float32)

        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        # Episode tracking
        self.current_step = 0
        self.max_steps = int(self.config.sim_time * 1000 / self.config.kpm_interval_ms)
        self.episode_metrics = []
        
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        """Reset environment and start new ns-3 simulation episode"""
        super().reset(seed=seed)
        
        if seed is not None:
            self.config.seed = seed
        
        # Multi-scenario training: randomly select a scenario each episode
        if self.config.scenarios:
            self.config.scenario = random.choice(self.config.scenarios)
            print(f"\n🎲 [Multi-Scenario] Starting episode with scenario: {self.config.scenario}")
        
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
        self.prev_action = None      # Reset action history for smoothing
        
        info = {
            'episode': 0,
            'seed': self.config.seed,
            'scenario': self.config.scenario
        }
        
        return state, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        rc_actions = self._parse_action(action)
        self.ns3.send_rc_control(rc_actions)
        
        time.sleep(self.config.kpm_interval_ms / 1000.0)
        e2_msg = self.ns3.receive_kpm_report(
            wait_for_new=True,
            max_wait_s=(self.config.kpm_interval_ms / 1000.0) * 1.5
        )
        next_state = self._extract_state(e2_msg)
        
        self.current_action = action
        reward, breakdown = self._compute_reward(e2_msg)
        self.prev_action = action.copy()
        
        self.episode_metrics.append({
            'step': self.current_step,
            'reward': reward,
            **breakdown,
        })
        
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        if self.ns3.ns3_process and self.ns3.ns3_process.poll() is not None:
            terminated = True
        truncated = False
        
        info = {
            'step': self.current_step,
            'e2_metrics': e2_msg,
            'actions_applied': rc_actions,
            'ns3_finished': bool(self.ns3.ns3_process and self.ns3.ns3_process.poll() is not None),
            **breakdown,
        }
        
        return next_state, reward, terminated, truncated, info
    
    def _extract_state(self, e2_msg: E2Message) -> np.ndarray:
        """Convert E2 KPM message to RL state vector"""
        state = []
        
        # UE metrics
        for ue_id in range(self.config.num_ues):
            ue = e2_msg.ue_metrics.get(ue_id, {})


            # Map expanded UE features (11 total):
            # throughput, delay, packet_loss, sinr, rsrp, rsrq,
            # ul_rbs, rb_allocated, cqi, rsrp_var, rsrq_var, buffer_occupancy
            state.extend([
                ue.get('throughput', 0.0) / 100.0,            # Mbps -> ~[0,1]
                ue.get('delay', 0.0) / 1000.0,                # ms -> seconds
                ue.get('packet_loss', 0.0),                   # ratio
                (ue.get('sinr', 0.0) + 10.0) / 40.0,          # SINR normalize [-10,30]
                (ue.get('rsrp', -140.0) + 140.0) / 100.0,     # RSRP [-140,-40]
                (ue.get('rsrq', -20.0) + 20.0) / 20.0,        # RSRQ [-20,0]
                ue.get('ul_rbs', 0.0) / 100.0,               # avg UL RBs
                ue.get('rb_allocated', 0) / 100.0,           # allocated RBs
                ue.get('cqi', 0.0) / 15.0,                   # CQI 0-15
                ue.get('rsrp_var', 0.0) / 50.0,              # variance scaled
                ue.get('rsrq_var', 0.0) / 50.0,              # variance scaled
                ue.get('buffer_occupancy', 0.0) / 10000.0,   # bytes scaled
            ])
        
        # Cell metrics
        # Per-cell features (5 per cell): queue_length, rb_utilization, tx_power, cell_load, avg_rb_request
        for cell_id in range(self.config.num_cells):
            cell = e2_msg.cell_metrics.get(cell_id, {})
            state.extend([
                cell.get('queue_length', 0) / 1000.0,          # queue length
                cell.get('rb_utilization', 0.0),               # ratio
                (cell.get('tx_power', 23.0) - 10.0) / 36.0,    # normalize tx power
                cell.get('cell_load', 0.0) / max(1.0, self.config.num_ues),
                cell.get('avg_rb_request', 0.0) / 100.0,
            ])
        
        return np.array(state, dtype=np.float32)
    
    def _parse_action(self, action: np.ndarray) -> Dict:
        """Convert RL action vector to E2SM-RC control parameters"""
        rc_actions = {}
        
        for cell_id in range(self.config.num_cells):
            # action layout per cell: 7 values
            idx = cell_id * 7
            sched_type = int(np.clip(int(action[idx + 1]), 0, len(SchedulerType) - 1))
            rc_actions[f'cell_{cell_id}'] = {
                'TxPower': float(action[idx]),
                'SchedulerType': SchedulerType(sched_type).name,
                'MaxHarqTx': int(np.clip(int(action[idx + 2]), 1, 8)),
                'Hysteresis': float(action[idx + 3]),
                'MacChDelay': float(action[idx + 4]),
                'NoiseFigure': float(action[idx + 5]),
                'SchedulerWeight': float(action[idx + 6]),
            }
        # Per-UE priority weights (remaining part of the action vector)
        base = self.config.num_cells * 7
        for ue_id in range(self.config.num_ues):
            rc_actions.setdefault('ues', {})
            rc_actions['ues'][f'ue_{ue_id}'] = {'priority_weight': float(action[base + ue_id])}

        return rc_actions
    
    def _compute_reward(self, e2_msg: E2Message) -> Tuple[float, Dict]:
        """Compute reward via RewardEngine. Called by step()."""
        current_action = getattr(self, 'current_action', None) # Note: Action is passed in step(), this is just for signature match or fallback
        # However, step() calls reward_engine.compute directly now.
        # This wrapper calls the engine and also ensures table printing as requested.
        
        self._print_metrics_table(e2_msg)
        
        return self.reward_engine.compute(e2_msg, current_action, self.prev_action, self.action_space)

    def _print_metrics_table(self, e2_msg):
        """Helper to print UE metrics table."""
        num_ues = len(e2_msg.ue_metrics)
        print(f"\n  ╔{'═'*90}╗")
        print(f"  ║  UE Metrics - {num_ues} UEs {'':>60}║")
        print(f"  ╠{'═'*90}╣")
        print(f"  ║ {'UE':>2} │ {'Tput':>6} │ {'Delay':>6} │ {'Loss':>5} │ {'SINR':>6} │ {'RSRP':>7} │ {'Cell':>4} │ {'Buffer':>6} ║")
        print(f"  ╠{'═'*90}╣")
        for ue_id, m in e2_msg.ue_metrics.items():
            rsrp_str = f"{m['rsrp']:.0f}" if m['rsrp'] != -140 else "  --"
            cell_str = f"{m['serving_cell']}" if m['serving_cell'] != -1 else "--"
            buffer_str = f"{m['buffer_occupancy']:.0f}" if m['buffer_occupancy'] > 0 else "  0"
            print(f"  ║ {ue_id:>2} │ {m['throughput']:>5.2f}M │ {m['delay']:>5.0f}ms │ "
                  f"{m['packet_loss']*100:>4.1f}% │ {m['sinr']:>5.1f}dB │ {rsrp_str:>6}dB │ "
                  f"{cell_str:>4} │ {buffer_str:>6} ║")
        print(f"  ╚{'═'*90}╝")
    
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
