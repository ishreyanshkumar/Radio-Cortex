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
    # Topic suffix for parallel environment isolation (e.g., "_0", "_1")
    topic_suffix: str = ""
    # Verbosity control for CLI output
    verbose: bool = True


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
        self.W_DELAY_BAR = 1.0      # Quadratic SLA Barrier
        self.W_LOSS      = 1.0      # Packet Loss (IQX)
        self.W_SE        = 0.05     # Spectral Efficiency (keep-alive signal)
        self.W_ENERGY    = 0.5      # Energy Efficiency
        self.W_LOAD      = 1.0      # Load Balancing
        self.W_QUEUE     = 0.3      # Queue Congestion (NEW)
        self.W_SMOOTH    = 0.05     # Action Smoothing
        self.BIAS        = 1.0      # Survival Bias (Ensures Level 0 is positive)

        # ── Thresholds / Normalizers ─────────────────────────────
        self.T_MAX     = 100.0      # Max Throughput (Mbps)
        self.D_MAX     = 100.0      # Normalizing Delay (ms)
        self.D_SLA     = 60.0       # SLA Threshold (ms) — barrier kicks in after this (Relaxed from 50.0)
        self.BETA_LOSS = 3.0        # IQX Sensitivity parameter (Relaxed from 5.0)
        self.EPSILON   = 1e-6       # Safe log
        self.Q_MAX     = 1000.0     # Queue length normalizer (NEW)

        # ── Clip bounds (Stage 1 — per component) ───────────────
        self.CLIP_TPUT   = (-0.5, 5.0)    # log(1+x) for x≥0 is ≥0, but allow small neg for numerical safety
        self.CLIP_DELAY  = (-50.0, 0.0)    # Delay is ALWAYS a penalty (≤0)
        self.CLIP_LOSS   = (-50.0, 0.0)    # Loss is ALWAYS a penalty (≤0)
        self.CLIP_SE     = (0.0, 2.0)     # SE is ALWAYS a bonus (≥0)
        self.CLIP_ENERGY = (-2.0, 0.0)    # Energy is ALWAYS a penalty (≤0)
        self.CLIP_LOAD   = (-2.0, 0.0)    # Load imbalance is ALWAYS a penalty (≤0)
        self.CLIP_QUEUE  = (-2.0, 0.0)    # Queue congestion is ALWAYS a penalty (≤0)
        self.CLIP_SMOOTH = (-1.0, 0.0)    # Smoothing is ALWAYS a penalty (≤0)

        # ── Clip bounds (Stage 2 — total) ────────────────────────
        self.CLIP_TOTAL = (-100.0, 10.0)

        # ── Curriculum State ─────────────────────────────────────
        self.level = 0              # 0=Bootstrap, 1=Quality, 2=Reliability
        self.success_history = []   # Rolling window of success rate
        self.HISTORY_LEN = 50       # Window size for level promotion
        self.MIN_TPUT_SUCCESS = 2.0 # Mbps required to count as "satisfied"

    def _update_curriculum(self, current_success_rate: float):
        """Update curriculum level based on sustained success rate."""
        self.success_history.append(current_success_rate)
        if len(self.success_history) > self.HISTORY_LEN:
            self.success_history.pop(0)
        
        avg_success = sum(self.success_history) / len(self.success_history)

        # Promotion Logic
        if self.level == 0 and avg_success > 0.5: # >50% UEs satisfied
            self.level = 1
            print(f"\n🎉 PROMOTED TO LEVEL 1 (Quality): Enabling Delay Penalty")
            self.success_history = [] # Reset history for next level
        elif self.level == 1 and avg_success > 0.8: # >80% UEs satisfied
            self.level = 2
            print(f"\n🚀 PROMOTED TO LEVEL 2 (Reliability): Enabling Loss/Jitter Penalty")
            self.success_history = []

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

        # ── Calculate Success Rate (for Curriculum) ──────────────
        # Start with simple check: Fraction of UEs with Tput > Min
        satisfied_ues = np.sum(tputs > self.MIN_TPUT_SUCCESS)
        success_rate = satisfied_ues / max(len(tputs), 1)
        self._update_curriculum(success_rate)

        # ── Apply Curriculum Masking ─────────────────────────────
        # Level 0: Only Tput + Bias. No Penalties.
        # Level 1: Tput + Bias + Delay.
        # Level 2: Full Rewards.
        
        w_delay_eff = self.W_DELAY_LIN if self.level >= 1 else 0.0
        w_queue_eff = self.W_QUEUE     if self.level >= 1 else 0.0  # Early warning (Level 1)

        w_loss_eff   = self.W_LOSS      if self.level >= 2 else 0.0
        w_energy_eff = self.W_ENERGY    if self.level >= 2 else 0.0
        w_load_eff   = self.W_LOAD      if self.level >= 2 else 0.0  # Optimization (Level 2)
        
        # ── 1b. Delay  (Two-tier: linear everywhere + quadratic past SLA) ──
        d_norm    = np.minimum(delays / self.D_MAX, 1.0)                    # Tier 1: 0→1 linear
        d_barrier = np.maximum(delays - self.D_SLA, 0.0) ** 2              # Tier 2: explodes past SLA
        r_delay = float(-np.mean(
            w_delay_eff * d_norm
            + (self.W_DELAY_BAR / self.D_MAX ** 2) * d_barrier
        ))
        r_delay = float(np.clip(r_delay, *self.CLIP_DELAY))

        # ── 1c. Packet Loss  (IQX exponential) ─────────────────
        #   exp(5 × 0.01)-1 = 0.05  (1% loss → mild)
        #   exp(5 × 0.10)-1 = 0.65  (10% loss → painful)
        #   exp(5 × 0.50)-1 = 11.2  (50% loss → catastrophic)
        r_loss = float(-np.mean(np.exp(self.BETA_LOSS * losses) - 1.0)) * w_loss_eff
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
            r_energy = float(-np.mean(rbs / total_rbs_max)) * w_energy_eff
            r_energy = float(np.clip(r_energy, *self.CLIP_ENERGY))

            # ── 2b. Load Balancing ──────────────────────────────
            #   Penalizes uneven load across cells.
            #   std(loads) is high when one cell is overloaded
            #   and another is idle → AI learns to distribute.
            r_load = float(-np.std(loads)) * w_load_eff
            r_load = float(np.clip(r_load, *self.CLIP_LOAD))

            # ── 2c. Queue Congestion  (NEW) ─────────────────────
            #   Penalizes cells with growing queues.
            #   This gives early warning BEFORE delay spikes —
            #   a full queue today = high delay tomorrow.
            q_norm = np.minimum(queues / self.Q_MAX, 1.0)
            r_queue = float(-np.mean(q_norm)) * w_queue_eff
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
        total_raw = ue_score + network_score + self.BIAS
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
            'z_level':      self.level,         # Log Level (z_ prefix sorts to end)
            'z_success':    success_rate,       # Log Success Rate
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
            nested_path = os.path.join('ns-3-allinone', 'ns-3.46.1', 'ns3')
            if os.path.exists(nested_path):
                ns3_path = nested_path
            elif os.path.exists(os.path.join('..', 'ns3')):
                ns3_path = os.path.join('..', 'ns3')
            elif os.path.exists(os.path.join('..', '..', 'ns3')): # Handle scratch/Radio-Cortex case
                ns3_path = os.path.join('..', '..', 'ns3')
        
        # Define working directory for ns-3
        ns3_path = os.path.abspath(ns3_path)
        ns3_dir = os.path.dirname(ns3_path)
        if not ns3_dir: 
            ns3_dir = os.getcwd()

        # Try to find the compiled binary directly to avoid Waf lock contention.
        # PRIORITY: Check for 'optimized' build first (much faster simulation).
        # Fallback to 'default' or 'debug' if optimized is not found.
        binary_name_default = "ns3.46.1-oran-congestion-scenario-default"
        binary_name_opt = "ns3.46.1-oran-congestion-scenario-optimized"
        
        candidate_paths = [
            # 1. Optimized build in standard layout (Most common with './ns3 build')
            os.path.join(ns3_dir, "build/scratch", binary_name_opt),
            # 2. Optimized build in split layout (Legacy/CMake specific)
            os.path.join(ns3_dir, "build/optimized/scratch", binary_name_opt),
            # 3. Default build (Commonly present, often debug-enabled/slow)
            os.path.join(ns3_dir, "build/scratch", binary_name_default),
            # 4. Debug build (Explicit debug)
            os.path.join(ns3_dir, "build/debug/scratch", binary_name_default.replace("default", "debug")),
        ]
        
        binary_path = None
        for path in candidate_paths:
            if os.path.exists(path):
                binary_path = path
                break
        
        if binary_path:
            # Execute binary directly - FORCE ABSOLUTE PATH
            binary_path = os.path.abspath(binary_path)
            if not os.path.exists(binary_path):
                 raise FileNotFoundError(f"Binary not found at {binary_path}")
            
            ns3_cmd = [
                binary_path,
                f'--numUes={self.config.num_ues}',
                f'--numCells={self.config.num_cells}',
                f'--simTime={self.config.sim_time}',
                f'--seed={self.config.seed}',
                f'--kpmInterval={self.config.kpm_interval_ms}',
                f'--enableE2=true',
                f'--scenario={self.config.scenario}',
                f'--bandwidthRbs={int(self.config.system_bandwidth_mhz * 5)}',
            ]
            if self.config.topic_suffix:
                ns3_cmd.append(f'--topicSuffix={self.config.topic_suffix}')
        else:
            # Fallback to wrapper if binary not found (slower startup)
            if self.config.verbose:
                print(f"Warning: Encoded binary not found, falling back to ns3 wrapper (slow startup)")
            ns3_cmd = [
                sys.executable, os.path.basename(ns3_path), 'run',
                f'scratch/oran-congestion-scenario',
                '--',
                f'--numUes={self.config.num_ues}',
                f'--numCells={self.config.num_cells}',
                f'--simTime={self.config.sim_time}',
                f'--seed={self.config.seed}',
                f'--kpmInterval={self.config.kpm_interval_ms}',
                '--enableE2=true',
                f'--scenario={self.config.scenario}',
                f'--bandwidthRbs={int(self.config.system_bandwidth_mhz * 5)}',
                f'--topicSuffix={self.config.topic_suffix}'
            ]
        
        # Start ns-3 in subprocess
        # Redirect output to files for debugging
        telemetry_dir = "logs"
        os.makedirs(telemetry_dir, exist_ok=True)
        self.log_file_out = open(os.path.join(telemetry_dir, f"ns3_out{self.config.topic_suffix}.log"), "w")
        self.log_file_err = open(os.path.join(telemetry_dir, f"ns3_err{self.config.topic_suffix}.log"), "w")
        
        # Prepare environment variables with LD_LIBRARY_PATH
        env = os.environ.copy()
        # Ensure build/lib is in LD_LIBRARY_PATH so direct binary execution works
        # Try both build/lib (standard) and build (sometimes used)
        lib_paths = [
            os.path.join(ns3_dir, "build/lib"),
            os.path.join(ns3_dir, "build"),
        ]
        
        # Also check if we are in 'optimized' or 'debug' build directory structure
        if "optimized" in str(binary_path):
             lib_paths.append(os.path.join(ns3_dir, "build/optimized/lib"))
        if "debug" in str(binary_path):
             lib_paths.append(os.path.join(ns3_dir, "build/debug/lib"))

        valid_lib_paths = [p for p in lib_paths if os.path.exists(p)]
        if valid_lib_paths:
            current_ld_path = env.get('LD_LIBRARY_PATH', '')
            new_ld_path = ':'.join(valid_lib_paths)
            if current_ld_path:
                new_ld_path = f"{new_ld_path}:{current_ld_path}"
            env['LD_LIBRARY_PATH'] = new_ld_path
            # print(f"DEBUG: LD_LIBRARY_PATH set to {new_ld_path}")

        if self.config.verbose:
            print(f"Starting ns-3 simulation with command: {' '.join(ns3_cmd)}")
        self.ns3_process = subprocess.Popen(
            ns3_cmd,
            cwd=ns3_dir,
            stdout=self.log_file_out,
            stderr=self.log_file_err,
            preexec_fn=os.setsid,
            env=env
        )
        if self.config.verbose:
            print(f"ns-3 process started (PID: {self.ns3_process.pid}, Logs: ns3_out{self.config.topic_suffix}.log)")
        time.sleep(4) # Give ns-3 time to initialize (increased for parallel stability)
        
        # Connect to Kafka (if not already connected)
        if not self.kafka_consumer:
            self._connect_kafka()
            
        # Reset timestamp tracking for new episode
        self.last_kpm_ts = None
        
    def _connect_kafka(self):
        """Establish Kafka connections"""
        from kafka import KafkaConsumer, KafkaProducer
        import socket

        try:
            if self.config.verbose:
                print("Connecting to Kafka...")
            kpm_topic = f'e2_kpm_stream{self.config.topic_suffix}'
            rc_topic = f'e2_rc_control{self.config.topic_suffix}'
            self.kafka_consumer = KafkaConsumer(
                kpm_topic,
                bootstrap_servers=['localhost:9092'],
                auto_offset_reset='earliest',  # Read from beginning to catch startup msgs
                enable_auto_commit=False,
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                group_id=f'oran_rl_agent{self.config.topic_suffix}_{int(time.time())}' # Unique group ID
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
            self._rc_topic = rc_topic  # Store for send_rc_control
            if self.config.verbose:
                print(f"✓ Connected to Kafka (topics: {kpm_topic}, {rc_topic})")
            
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
                records = self.kafka_consumer.poll(timeout_ms=100)
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
                # No data yet
                if wait_for_new:
                     # If we were strictly waiting for new data (e.g. during reset) and failed, 
                     # we should check if the process is valid.
                     if self.ns3_process.poll() is not None:
                         raise RuntimeError(f"ns-3 process died unexpectedly with code {self.ns3_process.returncode}. Check ns3_err{self.config.topic_suffix}.log")
                     # If process is alive but no data, raise error to avoid silent failure
                     raise TimeoutError(f"Timed out waiting for initial KPM report from ns-3 (Env {self.config.topic_suffix}). Check Kafka topics and logs.")
            
            if last_record:
                self.last_kpm_ts = last_record.timestamp
                kpm_data = last_record.value
                self.last_kpm_rx_time = time.time() # Timestamp for E2 Latency
                self.kpm_msg_count += 1
                return self._parse_kpm(kpm_data)
            else:
                topic_name = f"e2_kpm_stream{self.config.topic_suffix}"
                raise RuntimeError(f"No KPM data found on Kafka topic {topic_name}. Ensure ns-3 is running and producing data.")
            
        except Exception as e:
            if "TimeoutError" in str(type(e)):
                raise e
            raise RuntimeError(f"E2 Interface Failure (Env {self.config.topic_suffix}): {e}")
            
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
        if self._kpm_parse_count % 50 == 1 and self.config.verbose:
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
            rc_topic = getattr(self, '_rc_topic', 'e2_rc_control')
            self.kafka_producer.send(rc_topic, rc_message)
            self.kafka_producer.flush()
            
            self.rc_msg_count += 1
            if hasattr(self, 'last_kpm_rx_time') and self.last_kpm_rx_time > 0:
                latency = (time.time() - self.last_kpm_rx_time) * 1000.0 # ms
                self.e2_loop_latencies.append(latency)
        except Exception as e:
            # print(f"Error sending RC control: {e}")
            pass 
    
    
    def stop_simulation(self, close_kafka: bool = True):
        """Clean shutdown of ns-3 and Kafka connection"""
        if close_kafka:
            if self.kafka_consumer:
                self.kafka_consumer.close()
                self.kafka_consumer = None
            if self.kafka_producer:
                self.kafka_producer.close()
                self.kafka_producer = None
            
        if hasattr(self, 'adapter_process') and self.adapter_process:
            self.adapter_process.terminate()
            try:
                self.adapter_process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                 self.adapter_process.kill()
            
        if self.ns3_process:
            self.ns3_process.terminate()
            try:
                self.ns3_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"Warning: Force killing ns-3 process {self.ns3_process.pid}...")
                self.ns3_process.kill()
                self.ns3_process.wait()
            self.ns3_process = None


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
        
        # Action space: per-cell + per-UE controls (SIMPLIFIED)
        # Per-cell: [TxPower, SchedulerWeight, Hysteresis]  (3 per cell — high-impact levers)
        # Per-UE: [priority_weight] for each UE
        # Fixed defaults (not RL-controlled): SchedulerType=0(PF), MaxHarq=4, MacDelay=0, NoiseFig=5dB
        per_cell_low = [
            10.0,  # TxPower min (dBm)
            0.0,   # SchedulerWeight min
            0.0,   # Hysteresis min (dB)
        ]
        per_cell_high = [
            46.0,  # TxPower max
            5.0,   # SchedulerWeight max
            6.0,   # Hysteresis max (dB)
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
        
        # Differential Control State Tracking
        self.current_params = {
            'cell': {c: {
                'tx_power': 23.0,       # dBm
                'scheduler_type': 0,    # Discrete
                'max_harq': 4.0, 
                'hysteresis': 2.0,      # dB
                'mac_delay': 0.0,       # ms
                'noise_figure': 5.0,    # dB
                'scheduler_weight': 1.0 
            } for c in range(self.config.num_cells)},
            'ue': {u: {
                'priority_weight': 1.0
            } for u in range(self.config.num_ues)}
        }
        
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        """Reset environment and start new ns-3 simulation episode"""
        start_reset = time.time()
        super().reset(seed=seed)
        
        if seed is not None:
            self.config.seed = seed
            
        # Initialize sorted indices (default identity)
        self.sorted_ue_indices = list(range(self.config.num_ues))
        
        # Reset Differential Control State to Defaults
        self.current_params = {
            'cell': {c: {
                'tx_power': 23.0,       # dBm
                'scheduler_type': 0,    # Discrete (Handled absolutely for now or via small steps)
                'max_harq': 4.0, 
                'hysteresis': 2.0,      # dB
                'mac_delay': 0.0,       # ms
                'noise_figure': 5.0,    # dB
                'scheduler_weight': 1.0 
            } for c in range(self.config.num_cells)},
            'ue': {u: {
                'priority_weight': 1.0
            } for u in range(self.config.num_ues)}
        }
        
        # Multi-scenario training: randomly select a scenario each episode
        # This Domain Randomization ensures the policy is robust to different traffic patterns.
        if self.config.scenarios:
            self.config.scenario = random.choice(self.config.scenarios)
            if self.config.verbose:
                print(f"\n🎲 [Multi-Scenario] Starting episode with scenario: {self.config.scenario}")
        
        # Stop previous simulation if running
        if hasattr(self, 'ns3') and self.ns3:
            # Update scenario in existing interface config
            if self.config.scenarios and self.config.scenario:
                self.ns3.config.scenario = self.config.scenario
                
            # print(f"Calling stop_simulation (keep_kafka=True)...")
            self.ns3.stop_simulation(close_kafka=False)
            # print(f"Calling start_simulation...")
            self.ns3.start_simulation()
        else:
            # First time initialization
            self.ns3 = NS3Interface(self.config)
            self.ns3.start_simulation()
        
        # Wait for first KPM report to establish state
        if self.config.verbose:
            print("Waiting for initial KPM report...")
        try:
            # Longer timeout for initialization (30s)
            e2_msg = self.ns3.receive_kpm_report(max_wait_s=30.0, wait_for_new=True)
        except Exception as e:
            print(f"Failed to initialize environment: {e}")
            self.close()
            raise e
        state = self._extract_state(e2_msg)
        
        reset_duration = time.time() - start_reset
        reset_duration = time.time() - start_reset
        if self.config.verbose:
            print(f"Environment reset complete in {reset_duration:.2f}s")
        
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
        try:
            rc_actions = self._parse_action(action)
            self.ns3.send_rc_control(rc_actions)
            
            time.sleep(self.config.kpm_interval_ms / 1000.0)
            e2_msg = self.ns3.receive_kpm_report(
                wait_for_new=True,
                max_wait_s=(self.config.kpm_interval_ms / 1000.0) * 5.0
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
        except Exception as e:
            # If anything crashes (ns-3 died, Kafka timeout, etc.), gracefully
            # end the episode rather than killing the worker process.
            import traceback, sys
            print(f"[ENV PID {os.getpid()}] step() error (returning done): {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            fallback_state = np.zeros(self.observation_space.shape, dtype=np.float32)
            # Create dummy E2Message to prevent KeyError in evaluation loops
            dummy_msg = E2Message(timestamp=time.time(), ue_metrics={}, cell_metrics={})
            return fallback_state, -10.0, True, False, {'step_error': str(e), 'e2_metrics': dummy_msg}
    
    def _extract_state(self, e2_msg: E2Message) -> np.ndarray:
        """Convert E2 KPM message to RL state vector"""
        state = []
        
        # UE metrics
        # UE metrics with Canonical Sorting
        # Collect all UE metrics first
        ue_data = []
        for ue_id in range(self.config.num_ues):
            ue = e2_msg.ue_metrics.get(ue_id, {})
            # Sorting Key: Buffer Occupancy (Descending), then UE ID (Ascending) for ties
            sort_key = (-ue.get('buffer_occupancy', 0.0), ue_id)
            ue_data.append((ue_id, ue, sort_key))
            
        # Sort UEs
        ue_data.sort(key=lambda x: x[2])
        
        # Update sorted indices for action mapping
        self.sorted_ue_indices = [x[0] for x in ue_data]
        
        # Flatten state based on sorted order
        for _, ue, _ in ue_data:
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
        """
        Convert RL action to E2SM-RC control messages.
        Simplified: 2 dims per cell (TxPower, SchedulerWeight) + 1 per UE.
        Fixed params (HARQ, Hysteresis, etc.) use sensible defaults.
        """
        # Ensure action is a flat 1-D array (VecEnv may pass scalars or 0-d arrays)
        action = np.asarray(action, dtype=np.float64).flatten()
        
        expected_size = self.config.num_cells * 3 + self.config.num_ues
        if action.size != expected_size:
            import sys
            if action.size < expected_size:
                action = np.pad(action, (0, expected_size - action.size), constant_values=0.0)
            else:
                action = action[:expected_size]

        rc_actions = {'cell': [], 'ue': []}
        offset = 0
        
        # 1. Cell Actions (3 dims per cell: TxPower, SchedulerWeight, Hysteresis)
        for c in range(self.config.num_cells):
            cell_act = action[offset : offset + 3]
            offset += 3
            
            # Tx Power: +/- 1.0 dBm step (differential)
            delta_p = cell_act[0] * 1.0 
            self.current_params['cell'][c]['tx_power'] = np.clip(
                self.current_params['cell'][c]['tx_power'] + delta_p, 10.0, 46.0
            )
            
            # Scheduler Weight: +/- 0.1 step (differential)
            delta_w = cell_act[1] * 0.1
            self.current_params['cell'][c]['scheduler_weight'] = np.clip(
                self.current_params['cell'][c]['scheduler_weight'] + delta_w, 0.0, 5.0
            )

            # Hysteresis: +/- 0.5 dB step (differential) — important for mobility scenarios
            delta_hys = cell_act[2] * 0.5
            self.current_params['cell'][c]['hysteresis'] = np.clip(
                self.current_params['cell'][c]['hysteresis'] + delta_hys, 0.0, 6.0
            )

            # Send ALL params to ns-3 (fixed ones use defaults from current_params)
            rc_actions['cell'].append({
                'cell_id': c,
                'tx_power_dbm': float(self.current_params['cell'][c]['tx_power']),
                'scheduler_type': int(self.current_params['cell'][c]['scheduler_type']),
                'max_harq_tx': int(round(self.current_params['cell'][c]['max_harq'])),
                'hysteresis_db': float(self.current_params['cell'][c]['hysteresis']),
                'mac_ch_delay': int(round(self.current_params['cell'][c]['mac_delay'])),
                'noise_figure_db': float(self.current_params['cell'][c]['noise_figure']),
                'scheduler_weight': float(self.current_params['cell'][c]['scheduler_weight']),
            })

        # 2. UE Actions
        # Apply to PHYSICAL UEs using sorted_ue_indices map
        # Action index i corresponds to "i-th most critical UE"
        
        # Ensure we don't go out of bounds if sorted_ue_indices isn't populated (e.g. init)
        if not hasattr(self, 'sorted_ue_indices') or len(self.sorted_ue_indices) != self.config.num_ues:
             self.sorted_ue_indices = list(range(self.config.num_ues))
        
        for i in range(self.config.num_ues):
            if offset >= len(action): break
            
            ue_act = action[offset] # Scalar
            offset += 1
            
            physical_ue_id = self.sorted_ue_indices[i]
            
            # Priority Weight: +/- 0.2
            delta_prio = ue_act * 0.2
            self.current_params['ue'][physical_ue_id]['priority_weight'] = np.clip(
                self.current_params['ue'][physical_ue_id]['priority_weight'] + delta_prio, 0.0, 10.0
            )
            
            rc_actions['ue'].append({
                'ue_id': physical_ue_id,
                'priority_weight': float(self.current_params['ue'][physical_ue_id]['priority_weight'])
            })
            
        return rc_actions
    
    def _compute_reward(self, e2_msg: E2Message) -> Tuple[float, Dict]:
        """Compute reward via RewardEngine. Called by step()."""
        current_action = getattr(self, 'current_action', None) # Note: Action is passed in step(), this is just for signature match or fallback
        # However, step() calls reward_engine.compute directly now.
        # This wrapper calls the engine and also ensures table printing as requested.
        
        if self.config.verbose:
            # self._print_metrics_table(e2_msg) 
            pass
        
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
