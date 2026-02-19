"""
Radio-Cortex O-RAN ns-3 Environment
A Gym-compatible environment for RL-based RAN congestion control
integrating with ns-O-RAN simulation platform.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import subprocess
try:
    import orjson as json
except ImportError:
    import json
import socket
import time
import sys
import os
import random
import csv
from typing import Dict, List, Tuple, Optional, Union
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
    # Scenario Selection (Multi-scenario Training)
    scenarios: Optional[Union[List[str], Dict[str, float]]] = None  # List of names OR {name: weight}
    scenario: str = 'flash_crowd'     # Currently active scenarion
    # Topic suffix for parallel environment isolation (e.g., "_0", "_1")
    topic_suffix: str = ""
    # Shared run timestamp — set once at training start so all envs write to the same CSV.
    # Leave empty to auto-generate per-env (legacy behaviour).
    run_timestamp: str = ""
    # Verbosity control for CLI output
    verbose: bool = True
    # Testing: Randomly shuffle UE indices in the state (Order Scramble)
    shuffle_ues: bool = False

    def __post_init__(self):
        """Automatically adjust configuration defaults based on selected scenarios."""
        # 1. Sync scenario with scenarios list if necessary
        if self.scenarios and self.scenario not in (self.scenarios if isinstance(self.scenarios, list) else self.scenarios.keys()):
            # If current scenario isn't in the list, default to the first one available
            self.scenario = list(self.scenarios)[0] if isinstance(self.scenarios, (list, dict)) else self.scenario

        # 2. Major Bug Fix: Align num_ues/num_cells with C++ overrides for scenario requirements
        # Some ns-3 scenarios forcibly require specific counts to function (e.g., hexagonal layouts).
        target_ue_count = self.num_ues
        target_cell_count = self.num_cells
        
        # Check all possible scenarios in multi-scenario training
        check_scenarios = [self.scenario]
        if self.scenarios:
            check_scenarios.extend(list(self.scenarios) if isinstance(self.scenarios, list) else list(self.scenarios.keys()))
            
        # IoT Tsunami overrides (100 UEs)
        if 'iot_tsunami' in check_scenarios:
            target_ue_count = max(target_ue_count, 100)
        
        # Mobility/Topo scenarios (require 7 cells for the hexagonal cluster)
        topo_scenarios = ['mobility_storm', 'urban_canyon', 'ping_pong', 'commuter_rush']
        if any(s in check_scenarios for s in topo_scenarios):
            target_cell_count = max(target_cell_count, 7)
            
        if target_ue_count != self.num_ues:
            if self.verbose:
                print(f"⚠️  [NS3Config] Scenario Scale detected. Forcing num_ues={target_ue_count} to align with C++.")
            self.num_ues = target_ue_count
            
        if target_cell_count != self.num_cells:
            if self.verbose:
                print(f"⚠️  [NS3Config] Topology Scenario detected. Forcing num_cells={target_cell_count} to prevent crash (hexagonal layout requirement).")
            self.num_cells = target_cell_count


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
        # ── Weights (BALANCED MODE — fix r_se dominance) ─────────
        # r_se was dominating (0.3-0.8) because se_avg ≈ 3-5 bits/s/Hz
        # from typical SINR values (12-25 dB), making even W_SE=0.15
        # produce large rewards. Slashed W_SE to a tiny keep-alive only.
        self.W_TPUT      = 12.0      # Boosted: primary learning signal
        self.W_DELAY_LIN = 2.0      # Boosted: make delay visible in chart (was 1.5)
        self.W_DELAY_BAR = 2.7      # Punish SLA breaches
        self.W_LOSS      = 1.0      # Core stability penalty (was 2.5, then 2.0)
        # self.W_SE removed
        self.W_ENERGY    = 0.05     # Reduced: let it use max power
        self.W_LOAD      = 1.0      # Load Balancing
        self.W_QUEUE     = 0.8      # Backpressure signal
        self.W_SMOOTH    = 0.05     # Action Smoothing
        self.BIAS        = 1.0      # Survival Bias (Ensures Level 0 is positive)

        # ── Thresholds / Normalizers ─────────────────────────────
        self.T_MAX     = 10.0      # Lowered: actual tput ~1-3 Mbps, so 10 Mbps is reachable
        self.D_MAX     = 65.0      # Normalizing Delay (ms)
        self.D_SLA     = 45.0      # Earlier barrier entry
        self.BETA_LOSS = 5.0       # Steep exponential loss penalty (was 10.0)
        self.EPSILON   = 1e-6      # Safe log
        self.Q_MAX     = 500.0     # Queue penalization threshold

        # ── Clip bounds (Stage 1 — per component) ───────────────
        self.CLIP_TPUT   = (-0.5, 50.0)   # Uncapped for high throughput (was 5.0)
        self.CLIP_DELAY  = (-50.0, 0.0)    # Delay is ALWAYS a penalty (≤0)
        self.CLIP_LOSS   = (-50.0, 0.0)     # Loss is ALWAYS a penalty (≤0) (was -5.0)
        # CLIP_SE removed
        self.CLIP_ENERGY = (-2.0, 0.0)    # Energy is ALWAYS a penalty (≤0)
        self.CLIP_LOAD   = (-2.0, 0.0)    # Load imbalance is ALWAYS a penalty (≤0)
        self.CLIP_QUEUE  = (-2.0, 0.0)    # Queue congestion is ALWAYS a penalty (≤0)
        self.CLIP_SMOOTH = (-1.0, 0.0)    # Smoothing is ALWAYS a penalty (≤0)

        # ── Clip bounds (Stage 2 — total) ────────────────────────
        self.CLIP_TOTAL = (-100.0, 50.0) # Uncapped max (was 10.0)

        # ── Curriculum State ─────────────────────────────────────
        self.level = 0              # 0=Survival, 1=Quality, 2=Efficiency
        self.success_history = []   # Rolling window of success rate
        self.HISTORY_LEN = 50       # Window size (was 200 - faster promotion)
        self.MIN_STEPS_PER_LEVEL = 50  # Must spend ≥50 steps before promotion
        self.steps_at_current_level = 0 # Counter for minimum-steps gate
        self.MIN_TPUT_SUCCESS = 1.0 # 1 Mbps required to count as "satisfied" (Standardized)
        self.DEMOTION_THRESHOLD = 0.4 # If <40% users satisfied, consider demotion (was 0.3)
        self.PROMOTION_THRESHOLD_L1 = 0.7 # 70% for Level 1 (was 0.6)
        self.PROMOTION_THRESHOLD_L2 = 0.90 # 90% for Level 2 (was 0.85)
        self.current_scenario = None

    def _update_curriculum(self, current_success_rate: float, jains_index: float, rb_util: float):
        """
        Smart Curriculum with minimum-steps gating.
        Must spend MIN_STEPS_PER_LEVEL at each level AND fill enough history.
        """
        self.success_history.append(current_success_rate)
        if len(self.success_history) > self.HISTORY_LEN:
            self.success_history.pop(0)
        self.steps_at_current_level += 1
        
        avg_success = sum(self.success_history) / len(self.success_history)
        has_enough_history = len(self.success_history) >= min(self.HISTORY_LEN // 2, 25)
        has_min_steps = self.steps_at_current_level >= self.MIN_STEPS_PER_LEVEL

        # Gate: Must have both minimum steps AND enough history before any promotion
        if not (has_enough_history and has_min_steps):
            return

        # === PROMOTION LOGIC ===
        # Level 0 -> 1 (Survival -> Quality)
        # Promote if 70% users satisfied consistently
        if self.level == 0:
            if avg_success > self.PROMOTION_THRESHOLD_L1:
                self.level = 1
                self.steps_at_current_level = 0
                print(f"\n🎉 PROMOTED TO LEVEL 1 (Quality): AvgSuccess={avg_success:.2f}")
                self.success_history = self.success_history[-10:] # Reset history

        # Level 1 -> 2 (Quality -> Efficiency)
        elif self.level == 1:
            if avg_success > self.PROMOTION_THRESHOLD_L2:
                self.level = 2
                self.steps_at_current_level = 0
                print(f"\n🚀 PROMOTED TO LEVEL 2 (Efficiency): AvgSuccess={avg_success:.2f}")
                self.success_history = self.success_history[-10:]
            elif avg_success < self.DEMOTION_THRESHOLD:
                self.level = 0
                self.steps_at_current_level = 0
                print(f"\n📉 DEMOTED TO LEVEL 0: AvgSuccess={avg_success:.2f}")
                self.success_history = self.success_history[-10:]

        # Level 2 Demotion
        elif self.level == 2 and avg_success < self.DEMOTION_THRESHOLD:
            self.level = 1
            self.steps_at_current_level = 0
            print(f"\n📉 DEMOTED TO LEVEL 1: AvgSuccess={avg_success:.2f}")
            self.success_history = self.success_history[-10:]

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
        jains = self._jains_index(tputs)
        
        if e2_msg.cell_metrics:
            avg_rb_util = np.mean([m['rb_utilization'] for m in e2_msg.cell_metrics.values()])
        else:
            avg_rb_util = 0.0

        # Scenario-specific success threshold adjustments
        target_tput = self.MIN_TPUT_SUCCESS
        if self.current_scenario in ['iot_tsunami']: 
            target_tput = 0.5 # Very low bar for massive device scenario
        elif self.current_scenario in ['flash_crowd', 'spectrum_crunch', 'urban_canyon']:
            target_tput = 1.0 # Moderate bar (was 0.5 for flash_crowd — too easy)

        satisfied_ues = np.sum(tputs > target_tput)
        success_rate = satisfied_ues / max(len(tputs), 1)
        self._update_curriculum(success_rate, jains, avg_rb_util)

        # ── Apply Curriculum Masking ("Focus & Promote") ──────────
        # Level 0: Tput, Loss
        # Level 1: + Delay, Fairness (Jain's implied via Tput log)
        # Level 2: + Energy, Load, Handover
        
        # Soft factor removed - strict gating for clear focus
        
        # ── 1b. Delay  ────────────────────────────────────────────
        # Enabled at Level 1+
        r_delay = 0.0
        if self.level >= 1:
             d_norm    = np.minimum(delays / self.D_MAX, 1.0)
             d_barrier = np.maximum(delays - self.D_SLA, 0.0) ** 2
             r_delay = float(-np.mean(
                 self.W_DELAY_LIN * d_norm
                 + (self.W_DELAY_BAR / self.D_MAX ** 2) * d_barrier
             ))
             r_delay = float(np.clip(r_delay, *self.CLIP_DELAY))

        # ── 1c. Packet Loss  ──────────────────────────────────────
        # Always On (Survival Metric)
        mean_loss = float(np.mean(losses))
        loss_penalty_raw = mean_loss * 5.0  # Linear penalty
        if mean_loss > 0.1:                 # Quadratic kicker
            loss_penalty_raw += (mean_loss - 0.1) ** 2 * 20.0

        r_loss = -float(loss_penalty_raw) * self.W_LOSS
        r_loss = float(np.clip(r_loss, -20.0, 0.0))

        # ── 1d. Spectral Efficiency (REMOVED) ─────────────────────
        # User requested removal as it's a proxy for SINR/Tput
        # r_se = 0.0 - REMOVED

        ue_score = r_tput + r_delay + r_loss

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

            # ── 2a. Energy Efficiency (Level 2+) ────────────────
            if self.level >= 2:
                total_rbs_max = 5000.0
                r_energy = float(-np.mean(rbs / total_rbs_max)) * self.W_ENERGY
                r_energy = float(np.clip(r_energy, *self.CLIP_ENERGY))

                # ── 2b. Load Balancing (Level 2+) ──────────────────
                r_load = float(-np.std(loads)) * self.W_LOAD
                r_load = float(np.clip(r_load, *self.CLIP_LOAD))

            # ── 2c. Queue Congestion (Level 1+) ─────────────────
            # Early warning system enabled with Delay
            if self.level >= 1:
                q_norm = np.minimum(queues / self.Q_MAX, 1.0)
                r_queue = float(-np.mean(q_norm)) * self.W_QUEUE
                r_queue = float(np.clip(r_queue, *self.CLIP_QUEUE))

        network_score = r_energy + r_load + r_queue

        # ── 2d. Action Smoothing (Level 1+) ─────────────────────
        if self.level >= 1 and prev_action is not None and action is not None:
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
            # 'r_se':     r_se, # Removed
            'r_energy': r_energy,
            'r_load':   r_load,
            'r_queue':  r_queue,
            'r_smooth': r_smooth,
            # Diagnostic KPIs (not part of reward, but essential for logging)
            # 'se_avg':       se_avg, # Removed
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
            # r_se removed
            'r_energy': 0.0, 'r_load': 0.0, 'r_queue': 0.0,
            'r_smooth': 0.0, 
            # se_avg removed
            'jains': 0.0, 'p95_delay': 0.0,
            'avg_throughput': 0.0, 'avg_delay': 0.0, 'avg_loss': 0.0,
            'z_level': 0, 'z_success': 0.0,
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
            # Check for standard nested structure (version-agnostic)
            import glob as _glob
            candidates = sorted(_glob.glob('ns-3-allinone/ns-3.*/ns3'), reverse=True)
            if candidates:
                ns3_path = candidates[0]
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
        # Uses glob to handle different ns-3 versions (e.g. 3.40, 3.46.1) automatically.
        import glob
        
        # 46.1 is common, but we'll search for any version.
        binary_name_opt_pattern = "ns3.*-oran-congestion-scenario-optimized"
        binary_name_default_pattern = "ns3.*-oran-congestion-scenario-default"
        
        candidate_patterns = [
            # 1. Optimized build in standard layout
            os.path.join(ns3_dir, "build/scratch", binary_name_opt_pattern),
            # 2. Optimized build in split layout
            os.path.join(ns3_dir, "build/optimized/scratch", binary_name_opt_pattern),
            # 3. Default build
            os.path.join(ns3_dir, "build/scratch", binary_name_default_pattern),
            # 4. Debug build
            os.path.join(ns3_dir, "build/debug/scratch", binary_name_default_pattern.replace("default", "debug")),
        ]
        
        binary_path = None
        for pattern in candidate_patterns:
            matches = glob.glob(pattern)
            if matches:
                # Use the first match (most specific)
                binary_path = matches[0]
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

        # Connect to Kafka (if not already connected) BEFORE starting ns-3
        # to avoid missing the very first KPM report.
        if not self.kafka_consumer:
            self._connect_kafka()
        else:
            # If reusing, ensure we are at the end to skip old messages
            partitions = self.kafka_consumer.assignment()
            if partitions:
                self.kafka_consumer.seek_to_end(*partitions)

        if self.config.verbose:
            print(f"Starting ns-3 simulation with command: {' '.join(ns3_cmd)}")
        # Stagger startup for parallel environments to avoid thundering herd
        # Based on topic suffix to ensure different envs start at different times
        try:
            suffix_id = "".join(filter(str.isdigit, self.config.topic_suffix))
            if suffix_id:
                # Optimized staggering: first 4 envs start immediately, others staggered by 0.15s
                idx = int(suffix_id)
                delay = max(0, (idx - 3)) * 0.15 if idx > 3 else 0
                if delay > 0 and self.config.verbose:
                    print(f"Staggering startup for Env {self.config.topic_suffix} by {delay:.2f}s...")
                if delay > 0:
                    time.sleep(delay)
        except:
            pass

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
        # Optimization: Removed hard time.sleep(1). receive_kpm_report handles the wait dynamically.
        
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
                bootstrap_servers=[os.getenv('KAFKA_BOOTSTRAP', 'localhost:9092')],
                auto_offset_reset='earliest',  # Read from beginning to catch startup msgs
                enable_auto_commit=False,
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                group_id=f'oran_rl_agent{self.config.topic_suffix}_{int(time.time())}' # Unique group ID
            )

            # Optimization: Wait for partition assignment with a tight loop rather than a hard 30s poll
            start_poll = time.time()
            found_partitions = False
            # Reduced from 10s to 1s as per user suggestion for "fail fast" behavior
            while time.time() - start_poll < 1.0: 
                self.kafka_consumer.poll(timeout_ms=100) # Faster poll
                partitions = self.kafka_consumer.assignment()
                if partitions:
                    self.kafka_consumer.seek_to_end(*partitions)
                    found_partitions = True
                    break
            
            if not found_partitions and self.config.verbose:
                print(f"Warning: Kafka partitions not assigned within 10s for Env {self.config.topic_suffix}")

            self.last_kpm_ts = None
            
            self.kafka_producer = KafkaProducer(
                bootstrap_servers=[os.getenv('KAFKA_BOOTSTRAP', 'localhost:9092')],
                linger_ms=5,        # 🚀 OPTIMIZATION: Wait 5ms to batch syscalls
                batch_size=32768,   # 🚀 OPTIMIZATION: Allow 32KB batches
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
                # Optimized: Reduced poll timeout from 100ms to 10ms for lower latency detection
                records = self.kafka_consumer.poll(timeout_ms=10)
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
            # self.kafka_producer.flush()
            
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
        
        # KPM Verification Logger — writes every KPM report to logs/kpm_verification.jsonl
        os.makedirs('logs', exist_ok=True)
        suffix = self.config.topic_suffix or '_0'
        self._kpm_log_path = f'logs/kpm_verification{suffix}.jsonl'
        self._kpm_log_file = open(self._kpm_log_path, 'w')
        self._kpm_step_counter = 0
        
        # Reward Component Logger — one timestamped CSV per training run, shared by all envs.
        # Each env writes its own rows with an env_id column.
        # Process-safe concurrent writes use fcntl.flock (works with SubprocVecEnv/spawn).
        import fcntl
        env_id_str = (self.config.topic_suffix or '_0').lstrip('_')
        self._env_id = int(env_id_str) if env_id_str.isdigit() else 0
        try:
            from datetime import datetime
            ts = self.config.run_timestamp or datetime.now().strftime('%Y%m%d_%H%M%S')
            self._reward_log_path = f'logs/reward_metrics_{ts}.csv'
            # Env 0 creates the file with header; others just open for append
            if self._env_id == 0 and not os.path.exists(self._reward_log_path):
                with open(self._reward_log_path, 'w', newline='') as hf:
                    import csv as _csv
                    w = _csv.writer(hf)
                    w.writerow(['env_id', 'step', 'reward',
                                'r_tput', 'r_delay', 'r_loss', 'r_se', 'r_energy',
                                'r_load', 'r_queue', 'r_smooth',
                                'se_avg', 'jains', 'p95_delay',
                                'avg_throughput', 'avg_delay', 'avg_loss',
                                'z_level', 'z_success'])
                print(f"[RewardLogger] Env {self._env_id}: created {self._reward_log_path}")
            self._reward_log_enabled = True
        except Exception as e:
            print(f"Warning: Failed to initialize reward logger: {e}")
            self._reward_log_enabled = False

        
        # ──────────────────────────────────────────────────────────
        # Cell-Centric State Space  (Enriched Cell Tokens)
        # Each cell gets 16 features: 
        #   12 Base Features (5 native + 7 UE agg)
        #   + 3 Delta Features (Queue, RB, Latency)
        #   + 1 Curriculum Level
        # ──────────────────────────────────────────────────────────
        self.features_per_cell = 16
        state_dim = self.config.num_cells * self.features_per_cell
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )
        
        # Cell-Only Action Space: 5 Actions per cell
        # ALL DIFFERENTIAL for stable convergence:
        # 1. TxPower      (diff +/- 1.0 dBm)
        # 2. TimeToTrigger(diff +/- 50 ms)
        # 3. Hysteresis   (diff +/- 1.0 dB)
        # 4. MacDelay     (diff +/- 1 TTI)
        # 5. CqiTimer     (diff +/- 100 ms)
        self.actions_per_cell = 5
        action_dim = self.config.num_cells * self.actions_per_cell

        self.action_space = spaces.Box(
            low=np.array([-1.0] * action_dim, dtype=np.float32), # Continuous differential
            high=np.array([1.0] * action_dim, dtype=np.float32),
            dtype=np.float32
        )
        
        # Episode tracking
        self.current_step = 0
        self.max_steps = int(self.config.sim_time * 1000 / self.config.kpm_interval_ms)
        self.episode_metrics = []
        
        # Tracking for Delta Features
        self._prev_obs_raw = None # Stores raw metrics for delta diffs
        
        # Differential Control State Tracking (Cell-only)
        self.current_params = {
            'cell': {c: {
                'tx_power': 23.0,           # dBm
                'time_to_trigger': 256.0,   # ms (A3 TTT default)
                'cqi_timer': 1000.0,        # ms (CQI report validity)
                'hysteresis': 2.0,          # dB
                'mac_delay': 0.0,           # TTIs
                'noise_figure': 5.0,        # dB
            } for c in range(self.config.num_cells)}
        }
        
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        """Reset environment and start new ns-3 simulation episode"""
        start_reset = time.time()
        super().reset(seed=seed)
        
        self._prev_obs_raw = None # Reset delta tracking
        
        if seed is not None:
            self.config.seed = seed
            
        # Reset Differential Control State to Defaults (Cell-only)
        self.current_params = {
            'cell': {c: {
                'tx_power': 23.0,           # dBm
                'time_to_trigger': 256.0,   # ms (A3 TTT default)
                'cqi_timer': 1000.0,        # ms (CQI report validity)
                'hysteresis': 2.0,          # dB
                'mac_delay': 0.0,           # TTIs
                'noise_figure': 5.0,        # dB
            } for c in range(self.config.num_cells)}
        }
        
        # Multi-scenario training: select scenario for this episode
        if self.config.scenarios:
            if isinstance(self.config.scenarios, dict):
                # Weighted sampling from dictionary
                scenarios = list(self.config.scenarios.keys())
                weights = list(self.config.scenarios.values())
                # Normalize weights if they don't sum to 1
                total_w = sum(weights)
                if total_w > 0:
                    weights = [w / total_w for w in weights]
                    self.config.scenario = np.random.choice(scenarios, p=weights)
            else:
                # Uniform random from list
                self.config.scenario = random.choice(self.config.scenarios)
                
            if self.config.verbose:
                print(f"\n🎲 [Multi-Scenario] Starting episode with scenario: {self.config.scenario}")
            
        # Update RewardEngine with current scenario
        self.reward_engine.current_scenario = self.config.scenario
        
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
            # Increased timeout for robustness
            e2_msg = self.ns3.receive_kpm_report(max_wait_s=60.0, wait_for_new=True)
        except Exception as e:
            print(f"Failed to initialize environment: {e}")
            self.close()
            # Return dummy state to prevent worker crash
            dummy_state = np.zeros(self.observation_space.shape, dtype=np.float32)
            self._log_kpm_verification(None, is_real=False, reason=f'reset_fallback: {e}')
            return dummy_state, {"error": str(e), "is_fallback": True}

        state = self._extract_state(e2_msg)
        
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
            
            e2_msg = self.ns3.receive_kpm_report(
                wait_for_new=True,
                max_wait_s=60.0
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
            
            # Log reward components to CSV (every 5th step to reduce I/O)
            if self._reward_log_enabled and self.current_step % 5 == 0:
                try:
                    import fcntl, csv as _csv
                    row = [
                        self._env_id,
                        self.current_step,
                        round(reward, 4),
                        round(breakdown.get('r_tput', 0), 4),
                        round(breakdown.get('r_delay', 0), 4),
                        round(breakdown.get('r_loss', 0), 4),
                        round(breakdown.get('r_se', 0), 4),
                        round(breakdown.get('r_energy', 0), 4),
                        round(breakdown.get('r_load', 0), 4),
                        round(breakdown.get('r_queue', 0), 4),
                        round(breakdown.get('r_smooth', 0), 4),
                        round(breakdown.get('se_avg', 0), 4),
                        round(breakdown.get('jains', 0), 4),
                        round(breakdown.get('p95_delay', 0), 2),
                        round(breakdown.get('avg_throughput', 0), 2),
                        round(breakdown.get('avg_delay', 0), 2),
                        round(breakdown.get('avg_loss', 0), 4),
                        int(breakdown.get('z_level', 0)),
                        round(breakdown.get('z_success', 0), 4)
                    ]
                    with open(self._reward_log_path, 'a', newline='') as f:
                        fcntl.flock(f, fcntl.LOCK_EX)
                        _csv.writer(f).writerow(row)
                        fcntl.flock(f, fcntl.LOCK_UN)
                except Exception:
                    pass
            
            self.current_step += 1
            terminated = self.current_step >= self.max_steps
            if self.ns3.ns3_process and self.ns3.ns3_process.poll() is not None:
                terminated = True
            truncated = False
            
            # Log KPM verification (real data)
            self._log_kpm_verification(e2_msg, is_real=True)
            
            info = {
                'step': self.current_step,
                'e2_metrics': self._serialize_e2_message(e2_msg),  # Plain dict for SubprocVecEnv pickling
                'actions_applied': rc_actions,
                'ns3_finished': bool(self.ns3.ns3_process and self.ns3.ns3_process.poll() is not None),
                'is_fallback': False,
                **breakdown,
            }
            
            return next_state, reward, terminated, truncated, info
        except Exception as e:
            # If anything crashes (ns-3 died, Kafka timeout, etc.), gracefully
            # end the episode rather than killing the worker process.
            import traceback, sys
            print(f"[ENV PID {os.getpid()}] step() error (returning done): {e}", file=sys.stderr, flush=True)
            
            # Log KPM verification (fallback)
            self._log_kpm_verification(None, is_real=False, reason=f'step_error: {e}')
            
            # Return valid dummy tuple to prevent SubprocVecEnv worker crash
            dummy_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return dummy_obs, 0.0, True, False, {"error": str(e), "is_fallback": True}
    
    # ─── KPM Verification & Serialization Helpers ────────────────
    
    @staticmethod
    def _serialize_e2_message(e2_msg: E2Message) -> dict:
        """Convert E2Message dataclass to a plain dict for SubprocVecEnv pickle compatibility."""
        return {
            'timestamp': e2_msg.timestamp,
            'ue_metrics': e2_msg.ue_metrics,    # Already a plain dict
            'cell_metrics': e2_msg.cell_metrics  # Already a plain dict
        }
    
    def _log_kpm_verification(self, e2_msg, is_real: bool, reason: str = ''):
        """Log every KPM report to logs/kpm_verification_X.jsonl for data authenticity verification."""
        self._kpm_step_counter += 1
        entry = {
            'step': self._kpm_step_counter,
            'timestamp': time.time(),
            'is_real': is_real,
            'env_id': self.config.topic_suffix,
        }
        if is_real and e2_msg is not None:
            # Sample a few UE metrics to prove data is real
            ue_ids = sorted(e2_msg.ue_metrics.keys())[:3]
            entry['num_ues'] = len(e2_msg.ue_metrics)
            entry['num_cells'] = len(e2_msg.cell_metrics)
            entry['sample_ues'] = {}
            for uid in ue_ids:
                m = e2_msg.ue_metrics[uid]
                entry['sample_ues'][uid] = {
                    'tput': round(m.get('throughput', 0), 4),
                    'sinr': round(m.get('sinr', -10), 2),
                    'rsrp': round(m.get('rsrp', -140), 2),
                    'delay': round(m.get('delay', 0), 2),
                    'loss': round(m.get('packet_loss', 0), 4),
                    'cell': m.get('serving_cell', -1),
                }
            # Sample cell metrics
            cell_ids = sorted(e2_msg.cell_metrics.keys())[:2]
            entry['sample_cells'] = {}
            for cid in cell_ids:
                c = e2_msg.cell_metrics[cid]
                entry['sample_cells'][cid] = {
                    'rb_util': round(c.get('rb_utilization', 0), 4),
                    'queue': c.get('queue_length', 0),
                    'power': round(c.get('tx_power', 23), 2),
                    'ues': c.get('num_connected_ues', 0),
                }
        else:
            entry['reason'] = reason
        
        try:
            self._kpm_log_file.write(json.dumps(entry) + '\n')
            # Flush every 10 entries for near-real-time visibility
            if self._kpm_step_counter % 10 == 0:
                self._kpm_log_file.flush()
        except Exception:
            pass  # Don't let logging failures crash training
    
    def _extract_state(self, e2_msg: E2Message) -> np.ndarray:
        """
        Cell-Centric State Extraction.
        Aggregates UE metrics into their serving cells to produce
        Enriched Cell Tokens (12 features per cell).
        """
        # 1. Initialize per-cell UE aggregators
        cell_stats = {c: {'tputs': [], 'delays': [], 'losses': []}
                      for c in range(self.config.num_cells)}
        
        # 2. Map each UE to its serving cell
        for ue_id, m in e2_msg.ue_metrics.items():
            c_id = m.get('serving_cell', -1)
            if 0 <= c_id < self.config.num_cells:
                cell_stats[c_id]['tputs'].append(m.get('throughput', 0.0))
                cell_stats[c_id]['delays'].append(m.get('delay', 0.0))
                cell_stats[c_id]['losses'].append(m.get('packet_loss', 0.0))
        
        # 3. Build flat state vector: 16 features per cell
        state = np.zeros(self.observation_space.shape, dtype=np.float32)
        idx = 0
        
        # Temporary storage for current steps' raw metrics to compute deltas next step
        current_obs_raw = {} 
        
        for c in range(self.config.num_cells):
            # --- A. Native Cell Metrics (5 features) ---
            cm = e2_msg.cell_metrics.get(c, {})
            q_len   = cm.get('queue_length', 0)
            rb_util = cm.get('rb_utilization', 0.0)
            tx_pwr  = cm.get('tx_power', 23.0)
            c_load  = cm.get('cell_load', 0.0)
            avg_rb  = cm.get('avg_rb_request', 0.0)
            
            state[idx]   = q_len / 1000.0
            state[idx+1] = cm.get('rb_utilization', 0.0)
            state[idx+2] = (cm.get('tx_power', 23.0) - 10.0) / 36.0
            state[idx+3] = cm.get('cell_load', 0.0) / max(1.0, self.config.num_ues)
            state[idx+4] = cm.get('avg_rb_request', 0.0) / 100.0
            
            # --- B. Aggregated UE Metrics (7 features) ---
            stats = cell_stats[c]
            n_ues = len(stats['tputs'])
            
            if n_ues > 0:
                # Averages (General Capacity)
                avg_tput  = np.mean(stats['tputs'])
                avg_delay = np.mean(stats['delays'])
                avg_loss  = np.mean(stats['losses'])
                
                # Worst-Case (Special UE / Ambulance Detection)
                max_delay = np.max(stats['delays'])
                max_loss  = np.max(stats['losses'])
                
                # Jain's Fairness Index
                sum_t = sum(stats['tputs'])
                sum_sq_t = sum(x * x for x in stats['tputs'])
                jains = 1.0 if sum_sq_t < 1e-9 else (sum_t ** 2) / (n_ues * sum_sq_t)
            else:
                avg_tput, avg_delay, avg_loss = 0.0, 0.0, 0.0
                max_delay, max_loss = 0.0, 0.0
                jains = 1.0
            
            state[idx+5]  = avg_tput / 20.0          # 5Mbps -> 0.25 (Better gradient)
            state[idx+6]  = avg_delay / 100.0         # Avg delay
            state[idx+7]  = avg_loss                  # Avg packet loss
            state[idx+8]  = max_delay / 100.0         # Worst-case delay ("Is someone lagging?")
            state[idx+9]  = max_loss                  # Worst-case loss  ("Is someone dropping?")
            state[idx+10] = jains                     # Fairness index
            state[idx+11] = n_ues / 50.0              # UE load count
            
            # --- C. Delta Features (3 features) --- NEW
            # Needs previous step's raw values. 
            if self._prev_obs_raw and c in self._prev_obs_raw:
                prev = self._prev_obs_raw[c]
                # Delta Queue: (curr - prev) / 100.0 (Normalize diff)
                d_queue = (q_len - prev['queue']) / 100.0
                # Delta RB: (curr - prev)
                d_rb    = rb_util - prev['rb']
                # Delta Delay: (curr - prev) / 10.0 
                d_delay = (avg_delay - prev['delay']) / 10.0
            else:
                d_queue, d_rb, d_delay = 0.0, 0.0, 0.0
            
            state[idx+12] = np.clip(d_queue, -1.0, 1.0)
            state[idx+13] = np.clip(d_rb, -1.0, 1.0)
            state[idx+14] = np.clip(d_delay, -1.0, 1.0)
            
            # --- D. Curriculum Level (1 feature) --- NEW
            state[idx+15] = float(self.reward_engine.level) / 2.0 # Normalized 0.0, 0.5, 1.0
            
            # Store raw for next step
            current_obs_raw[c] = {'queue': q_len, 'rb': rb_util, 'delay': avg_delay}
            
            idx += self.features_per_cell
            
        self._prev_obs_raw = current_obs_raw
        return state
    
    def _parse_action(self, action: np.ndarray) -> Dict:
        """
        5-Dimensional Cell Control.
        Actions are [0, 1] normalized, mapped to physical ranges here.
        """
        action = np.asarray(action, dtype=np.float64).flatten()
        
        expected_size = self.config.num_cells * self.actions_per_cell
        if action.size != expected_size:
            if action.size < expected_size:
                action = np.pad(action, (0, expected_size - action.size), constant_values=0.5)
            else:
                action = action[:expected_size]

        rc_actions = {'cell': [], 'ue': []}
        offset = 0
        
        for c in range(self.config.num_cells):
            cell_act = action[offset : offset + self.actions_per_cell]
            offset += self.actions_per_cell
            
    def _parse_action(self, action: np.ndarray) -> Dict:
        """
        5-Dimensional Cell Control - UNIFIED DIFFERENTIAL.
        All actions are differentials to nudge the strict baseline.
        """
        action = np.asarray(action, dtype=np.float64).flatten()
        
        expected_size = self.config.num_cells * self.actions_per_cell
        if action.size != expected_size:
            if action.size < expected_size:
                action = np.pad(action, (0, expected_size - action.size), constant_values=0.0)
            else:
                action = action[:expected_size]

        rc_actions = {'cell': [], 'ue': []}
        offset = 0
        self.boundary_penalty = 0.0 # Reset per step
        
        for c in range(self.config.num_cells):
            cell_act = action[offset : offset + self.actions_per_cell]
            offset += self.actions_per_cell
            
            # 1. TxPower: Differential +/- 1.0 dBm
            # LOCKED at Level 0/1 to Focus on Service (Max Power)
            if self.reward_engine.level < 2:
                delta_p = 0.0
                self.current_params['cell'][c]['tx_power'] = 46.0 # Force Max
            else:
                delta_p = float(cell_act[0]) * 1.0 # act is -1..1
                self.current_params['cell'][c]['tx_power'] = np.clip(
                    self.current_params['cell'][c]['tx_power'] + delta_p, 10.0, 46.0
                )

            # 2. TimeToTrigger: Differential +/- 50 ms
            delta_ttt = float(cell_act[1]) * 50.0
            self.current_params['cell'][c]['time_to_trigger'] = np.clip(
                self.current_params['cell'][c]['time_to_trigger'] + delta_ttt, 0.0, 5120.0
            )

            # 3. Hysteresis: Differential +/- 1.0 dB (WAS ABSOLUTE)
            delta_hyst = float(cell_act[2]) * 1.0
            self.current_params['cell'][c]['hysteresis'] = np.clip(
                 self.current_params['cell'][c]['hysteresis'] + delta_hyst, 0.0, 10.0
            )

            # 4. MAC Delay: Differential +/- 1.0 TTI (WAS ABSOLUTE, discrete)
            # Accumulate as float, round for usage
            delta_mac = float(cell_act[3]) * 1.0
            self.current_params['cell'][c]['mac_delay'] = np.clip(
                 self.current_params['cell'][c]['mac_delay'] + delta_mac, 0.0, 30.0 # Cap max delay reasonable
            )
            used_mac_delay = int(round(self.current_params['cell'][c]['mac_delay']))

            # 5. CQI Timer: Differential +/- 100 ms (WAS ABSOLUTE)
            delta_cqi = float(cell_act[4]) * 100.0
            self.current_params['cell'][c]['cqi_timer'] = np.clip(
                 self.current_params['cell'][c]['cqi_timer'] + delta_cqi, 10.0, 2000.0
            )

            # --- Soft Action Masking (Constraint Awareness) ---
            # Calculate penalty for pushing against boundaries (wasted action magnitude)
            # If delta was +1 but value was already max, clip makes no change.
            # Penalty proportional to |desired_change - actual_change|
            # We approximate this by checking if we hit the bounds.
            
            # Simple heuristic: If action magnitude > 0.1 and we are at bounds, penalize.
            # TxPower (10-46)
            tx = self.current_params['cell'][c]['tx_power']
            if (tx >= 46.0 and delta_p > 0) or (tx <= 10.0 and delta_p < 0):
                self.boundary_penalty += 0.05 * abs(float(cell_act[0]))

            # TTT (0-5120)
            ttt = self.current_params['cell'][c]['time_to_trigger']
            if (ttt >= 5120.0 and delta_ttt > 0) or (ttt <= 0.0 and delta_ttt < 0):
                self.boundary_penalty += 0.05 * abs(float(cell_act[1]))
            
            # Hysteresis (0-10)
            hyst = self.current_params['cell'][c]['hysteresis']
            if (hyst >= 10.0 and delta_hyst > 0) or (hyst <= 0.0 and delta_hyst < 0):
                self.boundary_penalty += 0.05 * abs(float(cell_act[2]))

            rc_actions['cell'].append({
                'cell_id': c,
                'tx_power_dbm': float(self.current_params['cell'][c]['tx_power']),
                'time_to_trigger_ms': float(self.current_params['cell'][c]['time_to_trigger']),
                'hysteresis_db': float(self.current_params['cell'][c]['hysteresis']),
                'mac_ch_delay': used_mac_delay,
                'cqi_timer_ms': int(self.current_params['cell'][c]['cqi_timer']),
                'noise_figure_db': 5.0,  # Fixed (environment parameter)
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
        
        reward, terms = self.reward_engine.compute(e2_msg, current_action, self.prev_action, self.action_space)
        
        # Apply Soft Action Masking Penalty
        if hasattr(self, 'boundary_penalty') and self.boundary_penalty > 0:
            reward -= self.boundary_penalty
            terms['boundary_penalty'] = -self.boundary_penalty
            
        return reward, terms

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
        
        if hasattr(self, '_reward_log_file') and self._reward_log_file:
            try:
                self._reward_log_file.close()
            except:
                pass


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
