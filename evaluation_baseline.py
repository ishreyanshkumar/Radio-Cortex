"""
Radio-Cortex Evaluation and Baseline Comparison Suite

Compares congestion control performance:
1. Baseline: Static RAN (no RL control)
2. Radio-Cortex: PPO-based RL agent

Core Metrics:
- Throughput, Delay, Packet Loss, Fairness
- QoS Violations, Recovery Time
- Congestion Intensity, Satisfied User Ratio

Phase 2 Metrics (Network & RIC):
- Spectral Efficiency (bits/sec/Hz)
- Handover Success Rate
- E2 Loop Latency (ms)
- RIC Message Overhead (msgs/sec)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
import json
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
import os
import time
import torch



@dataclass
class EvaluationMetrics:
    """Performance metrics for comparison"""
    avg_throughput: float  # Mbps
    avg_delay: float  # ms
    avg_packet_loss: float  # ratio
    p95_delay: float  # 95th percentile delay
    max_packet_loss: float  # worst-case loss
    jains_fairness: float  # fairness index
    qos_violations: int  # count of SLA breaches
    total_downtime: float  # seconds of service outage
    congestion_intensity: float # % time RB > 90%
    satisfied_user_ratio: float # % UEs > 1 Mbps & < 100ms
    cell_edge_tput: float # 5th percentile throughput
    avg_sinr: float # dB
    avg_rsrp: float # dBm
    avg_handover_count: float # Average handovers per UE per episode
    avg_inference_time: float # Average time to compute action (ms)

    # Advanced Efficiency Metrics
    energy_efficiency: float # Mbps/Watt
    model_params: int # Number of trainable parameters
    
    # Phase 2 Metrics
    handover_success_rate: float # ratio
    control_stability: float # (%) - Score of how stable the AI decisions are (0-100)
    
    # Composite Scores (0-100) for Radar Chart
    qos_score: float
    reliability_score: float
    resource_score: float
    buffer_score: float
    phy_score: float
    architecture_score: float # NEW: Model Efficiency (Size + Speed)


class BaselineController:
    """
    Static RAN configuration (no adaptation)
    Represents current 5G networks without AI (e.g., Max Power, Default Handover)
    """
    
    def __init__(self, num_cells: int):
        self.num_cells = num_cells
        # Fixed parameters (never change) — returned as [-1,1]-normalized
        # 1.0 -> 46 dBm (Max power)
        self.tx_power_norm = 1.0
        # 0.0 -> 0.0 dB CIO (Neutral)
        self.cio_norm = 0.0
        # 0.6 -> 256 ms TTT (Standard agility)
        self.ttt_norm = 0.6
    
    def get_action(self, state):
        """Returns fixed [-1,1]-normalized action array (no adaptation)"""
        action = []
        for _ in range(self.num_cells):
            action.extend([
                self.tx_power_norm,
                self.cio_norm,
                self.ttt_norm
            ])
        return np.array(action, dtype=np.float32)
        
class EvaluationRunner:
    """
    Runs evaluation scenarios and compares controllers
    """
    
    def __init__(self, num_ues: int = 20, num_cells: int = 3):
        self.num_ues = num_ues
        self.num_cells = num_cells
    
    def evaluate_controller(
        self,
        controller,
        env,
        controller_name: str,
        progress_queue = None,
        task_id = None
    ) -> EvaluationMetrics:
        """
        Run online evaluation episode with real ns-3 environment
        """
        if not progress_queue:
            print(f"\nEvaluating {controller_name}...")
        
        # Track metrics
        throughputs = []
        delays = []
        losses = []
        sinrs = []
        rsrps = []
        
        # New: Tracking for advanced metrics
        queue_lengths = [] # List of avg queue length per step
        rb_utils = []      # List of avg RB utilization per step
        per_ue_stats = {}  # {ue_id: {'tput': [], 'delay': []}}
        total_power_w_accum = 0.0 # Accumulate Watts for average calculation
        step_count_power = 0
        
        # Handover tracking
        prev_cells = {}
        total_handovers = 0
        inference_times = []
        actions_history = []
        
        # Get model parameters if available
        model_params = 0
        if hasattr(controller, 'get_model_params'):
            model_params = controller.get_model_params()
        
        # Helper to check if env is VecEnv
        is_vec_env = hasattr(env, 'reset') and hasattr(env, 'step_async')
        
        # Access config (handle wrappers)
        if hasattr(env, 'config'):
            env_config = env.config
        elif is_vec_env:
            # Assumes all envs have same config
            env_config = env.get_attr('config')[0]
        else:
            # Fallback for other wrappers
            env_config = env.unwrapped.config

        # Reset environment
        if is_vec_env:
             state = env.reset()
             # VecEnv reset returns only obs. info is usually empty at start or not returned.
             # We need to simulate initial info? Or just wait for step.
             # State is (n_envs, dim). We assume n_envs=1 for evaluation.
             state = state[0] 
             info = {} 
        else:
             state, info = env.reset()
        
        # CRITICAL ERROR DETECTION: Check if reset failed
        if info.get('is_fallback', False):
            raise RuntimeError(f"Simulation failed to initialize for {controller_name}: {info.get('error', 'Unknown Error')}")
        
        terminated = False
        truncated = False
        
        # Calculate expected steps for progress bar
        total_steps = int(env_config.sim_time * 1000 / env_config.kpm_interval_ms)
        
        # UI Management
        pbar = None
        if progress_queue:
            # Parallel mode: notify main process of new task
            progress_queue.put(('start', task_id, total_steps, f"Eval {controller_name}"))
        else:
            pbar = tqdm(total=total_steps, desc=f"Eval {controller_name}", unit="step")
        
        step_i = 0
        with torch.inference_mode():
            while not (terminated or truncated):
                step_i += 1
                # Controller makes decision
                t0 = time.time()
                if hasattr(controller, 'get_rl_action'):
                    action_arr = controller.get_rl_action(state)
                else:
                    state_dict = self._parse_state(state, env_config)
                    action = controller.get_action(state_dict)
                    action_arr = self._dict_to_action(action, env.action_space)
                t1 = time.time()
                inference_times.append((t1 - t0) * 1000.0) # ms
                
                # Execute step
                if is_vec_env:
                    # VecEnv expects stacked actions
                    # If action_arr is (action_dim,), wrap it to (1, action_dim)
                    # But if controller returns (action_dim), we just pass [action_arr]
                    next_state, reward, done, infos = env.step([action_arr])
                    next_state = next_state[0]
                    reward = reward[0]
                    terminated = done[0]
                    truncated = False # VecEnv handles auto-reset, so 'done' implies term/trunc.
                    info = infos[0]
                else:
                    next_state, reward, terminated, truncated, info = env.step(action_arr)
                
                # CRITICAL ERROR DETECTION: Check if step failed
                if info.get('is_fallback', False):
                    # If it failed extremely early (e.g. step 1), it's a crash
                    if step_i < 5:
                        raise RuntimeError(f"Simulation crashed early for {controller_name}: {info.get('error', 'Process Died')}")
                    # Otherwise treat as early termination
                    terminated = True
                
                actions_history.append(action_arr)
                
                # Periodic logging of RIC decisions (actions) during evaluation
                step_count = step_i # step_i is current loop index
                if not progress_queue and controller_name == "Radio-Cortex" and step_count % 10 == 0:
                    # Format cell-level action summary
                    actions_per_cell = 2
                    if len(action_arr) >= actions_per_cell:
                        c0 = action_arr[:actions_per_cell]
                        print(f"\n  Step {step_count:>3} │ 🤖 RIC Decision (Cell 0): TxΔ={c0[0]:.2f} │ HO_Sens={c0[1]:.2f}")
            
            
                # Track Handovers
                if 'e2_metrics' in info:
                    # Fix: Handle plain dict from VecEnv/SubprocVecEnv
                    e2_data = info['e2_metrics']
                    if isinstance(e2_data, dict):
                        # Wrap in SimpleNamespace for attribute access compatibility
                        from types import SimpleNamespace
                        e2_msg = SimpleNamespace(**e2_data)
                        # Recursive wrap for ue_metrics/cell_metrics if needed, but code uses dict access for those
                        # Actually code uses .ue_metrics and .cell_metrics on e2_msg, but then [key] on those.
                        # The dict has keys 'ue_metrics' and 'cell_metrics'.
                    else:
                        e2_msg = e2_data

                    ue_metrics = e2_msg.ue_metrics
                    for ue_id, m in ue_metrics.items():
                        curr_cell = m.get('serving_cell', -1)
                        if ue_id in prev_cells:
                            if prev_cells[ue_id] != -1 and curr_cell != -1 and curr_cell != prev_cells[ue_id]:
                                total_handovers += 1
                        prev_cells[ue_id] = curr_cell

                # Collect metrics from info (which contains raw KPMs)
                # Ensure e2_msg is available (it was set above)
                if 'e2_metrics' in info:
                    ue_kpms = list(e2_msg.ue_metrics.values())
                    cell_kpms = list(e2_msg.cell_metrics.values())
                    
                    # Aggregate per-step metrics
                    step_tput = np.mean([m['throughput'] for m in ue_kpms]) if ue_kpms else 0.0
                    step_delay = np.mean([m['delay'] for m in ue_kpms]) if ue_kpms else 0.0
                    step_loss = np.mean([m['packet_loss'] for m in ue_kpms]) if ue_kpms else 0.0
                    step_sinr = np.mean([m['sinr'] for m in ue_kpms]) if ue_kpms else -10.0
                    step_rsrp = np.mean([m['rsrp'] for m in ue_kpms]) if ue_kpms else -140.0
                    
                    # New: Collect Congestion Stats
                    avg_queue = np.mean([c['queue_length'] for c in cell_kpms]) if cell_kpms else 0.0
                    avg_rb = np.mean([c['rb_utilization'] for c in cell_kpms]) if cell_kpms else 0.0
                    
                    # Accumulate Power (Watts)
                    # 10^(dBm/10) * 0.001
                    step_power_w = sum([(10**(c['tx_power']/10.0))*0.001 for c in cell_kpms]) if cell_kpms else 0.0
                    total_power_w_accum += step_power_w
                    step_count_power += 1
                    
                    # Send progress and metrics to main process
                    if progress_queue:
                        metrics_payload = {
                            'reward': reward,
                            'tput': step_tput,
                            'delay': step_delay,
                            'loss': step_loss,
                            'power': step_power_w,
                            'sinr': step_sinr,
                            'rsrp': step_rsrp,
                            'queue': avg_queue,
                            'rb': avg_rb
                        }
                        
                        # Per-UE metrics removed for performance
                        pass

                        progress_queue.put(('update', task_id, 1, metrics_payload))
                    elif pbar:
                        pbar.update(1)
                        pbar.set_postfix({'reward': f'{reward:.2f}', 'tput': f'{step_tput:.1f}'})
                    
                    throughputs.append(step_tput)
                    delays.append(step_delay)
                    losses.append(step_loss)
                    sinrs.append(step_sinr)
                    rsrps.append(step_rsrp)
                    queue_lengths.append(avg_queue)
                    rb_utils.append(avg_rb)
                    
                    # New: Collect Per-UE Stats for User Satisfaction
                    for ue_id, m in e2_msg.ue_metrics.items():
                        if ue_id not in per_ue_stats:
                            per_ue_stats[ue_id] = {'tput': [], 'delay': []}
                        per_ue_stats[ue_id]['tput'].append(m['throughput'])
                        per_ue_stats[ue_id]['delay'].append(m['delay'])


                state = next_state
        
        if pbar:
            pbar.update(total_steps - pbar.n) # Ensure full completion
            pbar.close()
        
        if progress_queue:
            progress_queue.put(('complete', task_id))
        
        # Calculate aggregate metrics
        
        # Phase 2: Collect RIC & Network Stats
        ns3_obj = None
        if hasattr(env, 'ns3'):
            ns3_obj = env.ns3
        elif is_vec_env:
            # Try to get ns3 from the first env
            try:
                ns3_obj = env.get_attr('ns3')[0]
            except:
                pass
        else:
            if hasattr(env.unwrapped, 'ns3'):
                ns3_obj = env.unwrapped.ns3
        
        e2_latency = 0.0
        ric_overhead = 0.0
        
        if ns3_obj:
             e2_latency = np.mean(ns3_obj.e2_loop_latencies) if hasattr(ns3_obj, 'e2_loop_latencies') and ns3_obj.e2_loop_latencies else 0.0
             ric_overhead = ((getattr(ns3_obj, 'kpm_msg_count', 0) + getattr(ns3_obj, 'rc_msg_count', 0)) / env_config.sim_time)
        control_stability = self._calculate_control_stability(actions_history)
        
        # Collect Cumulative Handover Stats from last KPM
        ho_attempts = 0
        ho_successes = 0
        if 'e2_metrics' in info:
             e2_data = info['e2_metrics']
             # Handle both dict (serialized) and object (direct) access
             if isinstance(e2_data, dict):
                 ue_metrics = e2_data.get('ue_metrics', {})
             else:
                 ue_metrics = getattr(e2_data, 'ue_metrics', {})

             for m in ue_metrics.values():
                 ho_attempts += m.get('handover_attempts', 0)
                 ho_successes += m.get('handover_successes', 0)

        # Calculate aggregate metrics
        if not throughputs:
            print(f"  [ERROR] No data collected for {controller_name}. Marking as CRASHED.")
            return None

        avg_power_w = total_power_w_accum / step_count_power if step_count_power > 0 else 0.001
        avg_inference = np.mean(inference_times) if inference_times else 0.0
        
        metrics = self._calculate_metrics(
            throughputs, delays, losses, queue_lengths, rb_utils, per_ue_stats, sinrs, rsrps,
            total_handovers,
            ho_attempts=ho_attempts,
            ho_successes=ho_successes,
            control_stability=control_stability,
            total_power_watts=avg_power_w,
            avg_inference_time=avg_inference,
            model_params=model_params
        )
        
        # Print formatted results
        print(f"  {'─'*50}")
        print(f"  │ {'Metric':<25} │ {'Value':>18} │")
        print(f"  {'─'*50}")
        print(f"  │ {'Throughput':<25} │ {metrics.avg_throughput:>15.2f} Mbps │")
        print(f"  │ {'Packet Loss':<25} │ {metrics.avg_packet_loss*100:>16.2f}% │")
        print(f"  │ {'Delay':<25} │ {metrics.avg_delay:>17.1f} ms │")
        print(f"  │ {'SINR':<25} │ {metrics.avg_sinr:>16.1f} dB │")
        print(f"  │ {'Satisfied Users':<25} │ {metrics.satisfied_user_ratio*100:>16.1f}% │")
        print(f"  │ {'Energy Efficiency':<25} │ {metrics.energy_efficiency:>15.2f} M/W │")
        print(f"  │ {'Jain\'s Fairness':<25} │ {metrics.jains_fairness:>18.4f} │")
        print(f"  │ {'Avg HO Count/UE':<25} │ {metrics.avg_handover_count:>18.1f} │")
        print(f"  │ {'Handover Success Rate':<25} │ {metrics.handover_success_rate*100:>16.1f}% │")
        print(f"  │ {'Inference Time':<25} │ {metrics.avg_inference_time:>17.3f} ms │")
        print(f"  │ {'Model Params':<25} │ {metrics.model_params:>18,} │")
        print(f"  │ {'Control Stability':<25} │ {metrics.control_stability:>17.1f}% │")
        print(f"  {'─'*50}")
        
        # 📊 Log to Unified Simulation Database if available
        if hasattr(env, '_sim_db') and env._sim_db:
            try:
                # Convert metrics object to dict
                results_dict = vars(metrics)
                env._sim_db.log_evaluation(results_dict)
                env._sim_db.log_metadata('controller', controller_name)
                print(f"  💾 Evaluation metrics synced to {env._sim_db_path}")
            except Exception as e:
                print(f"  [WARN] Failed to sync to SimDB: {e}")

        return metrics

    def _parse_state(self, state_arr, config) -> Dict:
        """Convert flattened cell-centric state back to dict.
        
        State layout (Cell-Centric, 16 features per cell):
          [0-11] Base Features (Queue, RB, Power, Load, Req, Tput, Delay, Loss, MaxD, MaxL, Jain, NUE)
          [12-14] Delta Features (Queue, RB, Delay)
          [15] Stationary Flag (constant 0.5)
        """
        state_dict = {}
        features_per_cell = 16
        
        for cell_id in range(config.num_cells):
            idx = cell_id * features_per_cell
            # Basic checks to avoid out of bounds if state is smaller
            if idx + 10 < len(state_arr):
                state_dict[f'cell_{cell_id}_queue'] = state_arr[idx] * 1000
                state_dict[f'cell_{cell_id}_rb_util'] = state_arr[idx + 1]
                state_dict[f'cell_{cell_id}_power'] = state_arr[idx + 2] * 36.0 + 10.0
                state_dict[f'cell_{cell_id}_avg_loss'] = state_arr[idx + 7]
                state_dict[f'cell_{cell_id}_max_delay'] = state_arr[idx + 8] * 100.0
                state_dict[f'cell_{cell_id}_jains'] = state_arr[idx + 10]
            
        return state_dict

    def _dict_to_action(self, action_dict, action_space) -> np.ndarray:
        """Convert controller action dict to numpy array.
        For BaselineController, action_dict is already a flat numpy array."""
        if isinstance(action_dict, np.ndarray):
            return action_dict
        # Fallback: fill action space with 0.5 (no-change for differential actions)
        return np.full(action_space.shape, 0.5, dtype=np.float32)

    def _calculate_control_stability(self, actions_history: List[np.ndarray]) -> float:
        """
        Calculates a control stability score based on action changes over time.
        A higher score indicates more stable (less fluctuating) control decisions.
        Score is 0-100.
        """
        if len(actions_history) < 2:
            return 100.0 # Perfectly stable if only one or no actions
        
        total_action_diff = 0.0
        num_comparisons = 0
        
        # Iterate through consecutive actions
        for i in range(1, len(actions_history)):
            prev_action = actions_history[i-1]
            curr_action = actions_history[i]
            
            # Calculate the Euclidean distance between action vectors
            # Normalize by the action space range if possible, or just use raw diff
            diff = np.linalg.norm(curr_action - prev_action)
            total_action_diff += diff
            num_comparisons += 1
            
        if num_comparisons == 0:
            return 100.0
            
        avg_action_diff = total_action_diff / num_comparisons
        
        # Heuristic mapping to a 0-100 score:
        # Assume a typical 'diff' range. For example, if a diff of 10 is very unstable,
        # and 0 is perfectly stable.
        # This mapping might need tuning based on actual action space and observed diffs.
        # Let's assume a max reasonable diff for a single step is around 5-10 for a typical action space.
        # A simple inverse relationship: score = max(0, 100 - (avg_action_diff * scaling_factor))
        
        # Example scaling: if avg_action_diff of 1.0 means 10% instability (90 score)
        # If avg_action_diff of 10.0 means 100% instability (0 score)
        scaling_factor = 100 / 10.0 # Adjust this based on expected action diff magnitude
        
        stability_score = max(0.0, 100.0 - (avg_action_diff * scaling_factor))
        
        return stability_score


    def _calculate_metrics(
        self,
        throughputs: List[float],
        delays: List[float],
        losses: List[float],
        queue_lengths: List[float],
        rb_utils: List[float],
        per_ue_stats: Dict,
        sinrs: List[float],
        rsrps: List[float],
        total_handovers: int,
        ho_attempts: int = 0,
        ho_successes: int = 0,
        system_bandwidth_mhz: float = 10.0,
        control_stability: float = 0.0,
        total_power_watts: float = 0.1, # Avoid div by zero
        avg_inference_time: float = 0.0,
        model_params: int = 0
    ) -> EvaluationMetrics:
        """Compute evaluation metrics including advanced congestion stats"""
        
        # Basic statistics
        avg_throughput = np.mean(throughputs)
        avg_delay = np.mean(delays)
        avg_packet_loss = np.mean(losses)
        
        # New: RSRP (Only report RSRP, not PDR)
        avg_rsrp = np.mean(rsrps) if rsrps else -140.0
        
        # Percentiles
        p95_delay = np.percentile(delays, 95) if delays else 0.0
        max_packet_loss = np.max(losses) if losses else 0.0

        # Efficiency Metrics
        # Energy Eff = Sum Tput (Mbps) / Power (Watts)
        energy_efficiency = sum(throughputs) / total_power_watts if total_power_watts > 0 else 0.0
        
        # Fairness
        ue_avg_tputs = []
        # Need to pre-calculate ue_avg_tputs for fairness
        for ue_id, stats in per_ue_stats.items():
            ue_avg_tputs.append(np.mean(stats['tput']))

        if ue_avg_tputs and sum(ue_avg_tputs) > 0:
             jains_fairness = (sum(ue_avg_tputs) ** 2) / (
                len(ue_avg_tputs) * sum([t**2 for t in ue_avg_tputs])
             )
        else:
            jains_fairness = 0.0
        
        # QoS violations
        qos_violations = sum(
            1 for d, l in zip(delays, losses)
            if d > 100 or l > 0.05
        )
        
        # Total downtime
        total_downtime = sum(1 for l in losses if l > 0.2) * 0.1
        
        # --- Advanced Metrics ---
        
        # 1. Congestion Intensity (% time where Queue > 10 pkts OR RB > 90%)
        # Let's assume RB Utilization is the main indicator.
        congested_steps = sum(1 for rb in rb_utils if rb > 0.9)
        congestion_intensity = congested_steps / len(rb_utils) if rb_utils else 0.0
        
        # 2. Satisfied User Ratio (Capacity Proxy)
        # SLA: Avg Tput > 1 Mbps AND Avg Delay < 100ms
        satisfied_count = 0
        total_ues = len(per_ue_stats)
        
        for ue_id, stats in per_ue_stats.items():
            u_tput = np.mean(stats['tput'])
            u_delay = np.mean(stats['delay'])
            # ue_avg_tputs is already populated
            
            if u_tput >= 1.0 and u_delay <= 100.0:
                satisfied_count += 1
                
        satisfied_user_ratio = satisfied_count / total_ues if total_ues > 0 else 0.0
        
        # 4. Cell Edge Throughput (5th percentile user throughput)
        cell_edge_tput = np.percentile(ue_avg_tputs, 5) if ue_avg_tputs else 0.0
        
        # 6. Avg SINR
        avg_sinr = np.mean(sinrs) if sinrs else -10.0
        
        # 7. Avg Handover Count
        actual_ues = len(per_ue_stats)
        avg_handover_count = total_handovers / actual_ues if actual_ues > 0 else 0.0
        
        # --- Phase 2 Metrics ---
        
        # 2. Handover Success Rate
        handover_success_rate = ho_successes / ho_attempts if ho_attempts > 0 else 0.0
        if ho_attempts == 0 and total_handovers == 0:
            handover_success_rate = 1.0 # No handovers needed = Success? Or N/A. Let's say 1.0 for static.
        
        # --- Health Scores Mapping (0-100) ---
        
        # 1. QoS Score: Weighted mix of Tput, Delay, p95 Delay, Satisfied User Ratio
        norm_tput = min(avg_throughput / 10.0, 1.0)
        norm_delay = max(0.0, 1.0 - (avg_delay / 100.0))
        norm_p95 = max(0.0, 1.0 - (p95_delay / 200.0))
        qos_score = (0.25 * norm_tput + 0.25 * norm_delay + 0.15 * norm_p95 + 0.35 * satisfied_user_ratio) * 100
        
        # 2. Reliability Score: Removed Control Stability. 
        # Redistributed weight to Loss and Max Loss.
        norm_loss = max(0.0, 1.0 - (avg_packet_loss * 10))
        norm_max_loss = max(0.0, 1.0 - (max_packet_loss * 5))
        norm_downtime = max(0.0, 1.0 - (total_downtime / 10.0))
        norm_ho = max(0.0, 1.0 - (avg_handover_count / 3.0))
        norm_ho_success = handover_success_rate if handover_success_rate > 0 else 1.0
        
        # New Weights: Loss (35%), Max Loss (15%), Downtime (20%), HO (10%), HO Success (20%)
        reliability_score = (0.35 * norm_loss + 0.15 * norm_max_loss + 0.20 * norm_downtime + 
                             0.10 * norm_ho + 0.20 * norm_ho_success) * 100
        
        # 3. Resource Score: Utilization, Edge Tput, Fairness, EnergyEff
        avg_util = np.mean(rb_utils) if rb_utils else 0.0
        norm_util = min(avg_util * 100, 100.0) / 100.0
        norm_edge = min(cell_edge_tput / 2.0, 1.0)
        norm_nrg = min(energy_efficiency / 150.0, 1.0)
        
        # New Weights with Energy: Util (10%), Edge (30%), Fairness (30%), Energy (30%)
        resource_score = (0.10 * norm_util + 0.30 * norm_edge + 0.30 * jains_fairness + 0.30 * norm_nrg) * 100
        
        # 4. Buffer Score: Queue Health & Congestion SPIkes
        avg_q = np.mean(queue_lengths) if queue_lengths else 0.0
        norm_q = max(0.0, 1.0 - (avg_q / 100.0)) * 100
        norm_cong = max(0.0, 1.0 - congestion_intensity) * 100
        buffer_score = 0.6 * norm_q + 0.4 * norm_cong
        
        # 5. PHY Score: SINR & Signal Strength
        norm_sinr = min(max((avg_sinr + 10.0) / 40.0, 0.0), 1.0)
        norm_rsrp = min(max((avg_rsrp + 120.0) / 60.0, 0.0), 1.0)
        phy_score = (0.6 * norm_sinr + 0.4 * norm_rsrp) * 100
        
        # 7. Architecture Score: Efficiency of the AI Model itself
        # Params: 0 params (baseline) = 1.0. 1M params = 0.0.
        norm_params = max(0.0, 1.0 - (model_params / 1000000.0))
        # Infer: 0ms = 1.0. 10ms = 0.0.
        norm_infer = max(0.0, 1.0 - (avg_inference_time / 10.0))
        
        architecture_score = (0.5 * norm_params + 0.5 * norm_infer) * 100
        
        return EvaluationMetrics(
            avg_throughput=avg_throughput,
            avg_delay=avg_delay,
            avg_packet_loss=avg_packet_loss,
            p95_delay=p95_delay,
            max_packet_loss=max_packet_loss,
            jains_fairness=jains_fairness,
            qos_violations=qos_violations,
            total_downtime=total_downtime,
            congestion_intensity=congestion_intensity,
            satisfied_user_ratio=satisfied_user_ratio,
            cell_edge_tput=cell_edge_tput,
            avg_sinr=avg_sinr,
            avg_rsrp=avg_rsrp,
            avg_handover_count=avg_handover_count,
            
            energy_efficiency=energy_efficiency,
            model_params=model_params,
            
            handover_success_rate=handover_success_rate,
            control_stability=control_stability,
            avg_inference_time=avg_inference_time,
            
            qos_score=qos_score,
            reliability_score=reliability_score,
            resource_score=resource_score,
            buffer_score=buffer_score,
            phy_score=phy_score,
            architecture_score=architecture_score
        )
    
    def _calculate_recovery_time(self, losses: List[float]) -> float:
        """Calculate time to recover from failure"""
        # Find first occurrence of high loss
        failure_idx = None
        for i, loss in enumerate(losses):
            if loss > 0.1:  # 10% loss = failure
                failure_idx = i
                break
        
        if failure_idx is None:
            return 0.0  # No failure occurred
        
        # Find recovery point (loss < 2% for 10 consecutive samples)
        recovery_window = 10
        for i in range(failure_idx, len(losses) - recovery_window):
            window = losses[i:i+recovery_window]
            if all(l < 0.02 for l in window):
                return (i - failure_idx) * 0.1  # 100ms per sample
        
        return (len(losses) - failure_idx) * 0.1  # Never recovered


class ResultLogger:
    """
    Handles logging of evaluation results to CSV for dashboard visualization.
    Replaces old static plotting functionality.
    """
    
    @staticmethod
    def log_to_csv(
        results: Dict[str, EvaluationMetrics],
        scenario_name: str,
        config: dict = None,
        save_path: str = "results/experiment_results.csv"
    ):
        """
        Append evaluation results to a central CSV file.
        
        Each call appends one row per controller (Baseline / Radio-Cortex)
        with a timestamp, all metrics, and config parameters.
        Creates the file with headers if it doesn't exist.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        config = config or {}
        
        rows = []
        for controller_name, metrics in results.items():
            row = {
                "Timestamp": timestamp,
                "Scenario": scenario_name,
                "Controller": controller_name,
                # Core Metrics
                "Throughput_Mbps": metrics.avg_throughput,
                "Avg_Delay_ms": metrics.avg_delay,
                "PacketLoss_Ratio": metrics.avg_packet_loss,
                "p95_Delay_ms": metrics.p95_delay,
                "Max_PacketLoss": metrics.max_packet_loss,
                "Jains_Fairness": metrics.jains_fairness,
                "QoS_Violations": metrics.qos_violations,
                "Total_Downtime_s": metrics.total_downtime,
                "Congestion_Intensity": metrics.congestion_intensity,
                "Satisfaction_Percent": metrics.satisfied_user_ratio * 100,
                "Cell_Edge_Tput": metrics.cell_edge_tput,
                "Avg_SINR_dB": metrics.avg_sinr,
                "Avg_RSRP_dBm": metrics.avg_rsrp,
                "Avg_HO_per_UE": metrics.avg_handover_count,
                "Avg_Inference_ms": metrics.avg_inference_time,
                # Efficiency
                "EnergyEfficiency_Mbps_W": metrics.energy_efficiency,
                "Model_Params": metrics.model_params,
                # Phase 2
                "HO_Success_Rate": metrics.handover_success_rate,
                "Control_Stability": metrics.control_stability,
                # Composite Scores
                "QoS_Score": metrics.qos_score,
                "Reliability_Score": metrics.reliability_score,
                "Resource_Score": metrics.resource_score,
                "Buffer_Score": metrics.buffer_score,
                "PHY_Score": metrics.phy_score,
                "Architecture_Score": metrics.architecture_score,
            }
            # Add config parameters (prefixed)
            for key, value in config.items():
                row[f"Config_{key}"] = value
            rows.append(row)
        
        df = pd.DataFrame(rows)
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Append if file exists, write header if new
        if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
            try:
                old_df = pd.read_csv(save_path)
                combined_df = pd.concat([old_df, df], ignore_index=True)
                combined_df.to_csv(save_path, index=False)
            except Exception:
                df.to_csv(save_path, mode='a', header=True, index=False)
        else:
            df.to_csv(save_path, index=False)
        print(f"  📊 Results appended to {save_path}")

    @staticmethod
    def get_completed_pairs(save_path: str = "results/experiment_results.csv") -> List[Tuple[str, str]]:
        """
        Returns a list of (scenario, controller) tuples already present in the CSV.
        Used to skip redundant evaluations.
        """
        if not os.path.exists(save_path):
            return []
        try:
            df = pd.read_csv(save_path, usecols=["Scenario", "Controller"])
            return list(zip(df["Scenario"], df["Controller"]))
        except Exception:
            # If CSV is malformed or column names differ, return empty
            return []
