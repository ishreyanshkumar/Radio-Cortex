"""
Radio-Cortex Evaluation and Baseline Comparison Suite

Compares congestion control performance against baselines:
1. Static RAN (no RL control)
2. Heuristic-based control (rule-based)
3. Radio-Cortex (PPO RL)

Metrics:
- Packet loss rate
- Average throughput
- Fairness (Jain's index)
- QoS violations
- Recovery time
- Congestion Intensity
- Satisfied User Ratio
- Peak Burst Loss
- Cell Edge Throughput
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
import json
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
import seaborn as sns
import os


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
    recovery_time: float  # seconds to recover after failure
    total_downtime: float  # seconds of service outage
    congestion_intensity: float # % time RB > 90%
    satisfied_user_ratio: float # % UEs > 1 Mbps & < 100ms
    peak_burst_loss: float # Max loss in 1s window
    cell_edge_tput: float # 5th percentile throughput
    avg_jitter: float # ms (std dev of delay)
    avg_sinr: float # dB
    avg_pdr: float # ratio (0-1)
    avg_rsrp: float # dBm
    avg_handover_count: float # Average handovers per UE per episode
    
    # Composite Scores (0-100) for Radar Chart
    qos_score: float
    reliability_score: float
    resource_score: float
    buffer_score: float
    phy_score: float


class BaselineController:
    """
    Static RAN configuration (no adaptation)
    Represents current 5G networks without AI
    """
    
    def __init__(self, num_cells: int):
        self.num_cells = num_cells
        # Fixed parameters (never change)
        self.tx_power = [23.0] * num_cells  # dBm
        self.scheduler = ['PF'] * num_cells  # Proportional Fair
        self.harq_retx = [4] * num_cells
    
    def get_action(self, state):
        """Returns fixed action (no adaptation)"""
        return {
            'tx_power': self.tx_power,
            'scheduler': self.scheduler,
            'harq_retx': self.harq_retx
        }


class HeuristicController:
    """
    Rule-based RAN control (hand-crafted heuristics)
    Represents "smart" pre-AI approaches
    """
    
    def __init__(self, num_cells: int):
        self.num_cells = num_cells
        self.load_threshold = 0.8  # 80% RB utilization
        self.loss_threshold = 0.05  # 5% packet loss
    
    def get_action(self, state):
        """Apply hand-crafted rules"""
        actions = {
            'tx_power': [],
            'scheduler': [],
            'harq_retx': []
        }
        
        # Extract cell metrics from state
        # Assuming state contains: [ue_metrics..., cell_queue, cell_rb_util, cell_power, ...]
        
        for cell_id in range(self.num_cells):
            # Rule 1: If RB utilization > threshold, reduce power to offload UEs
            rb_util = state.get(f'cell_{cell_id}_rb_util', 0.5)
            if rb_util > self.load_threshold:
                actions['tx_power'].append(20.0)  # Reduce power
            else:
                actions['tx_power'].append(26.0)  # Normal power
            
            # Rule 2: If packet loss > threshold, switch to Round Robin (fairness)
            loss = state.get(f'cell_{cell_id}_avg_loss', 0.0)
            if loss > self.loss_threshold:
                actions['scheduler'].append('RR')
            else:
                actions['scheduler'].append('PF')
            
            # Rule 3: High loss → increase retransmissions
            if loss > self.loss_threshold:
                actions['harq_retx'].append(8)  # Max retries
            else:
                actions['harq_retx'].append(4)  # Default
        
        return actions


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
        controller_name: str
    ) -> EvaluationMetrics:
        """
        Run online evaluation episode with real ns-3 environment
        """
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
        
        # Handover tracking
        prev_cells = {} # {ue_id: cell_id}
        total_handovers = 0
        
        # Reset environment
        state, info = env.reset()
        terminated = False
        truncated = False
        
        # Calculate expected steps for progress bar
        total_steps = int(env.config.sim_time * 1000 / env.config.kpm_interval_ms)
        pbar = tqdm(total=total_steps, desc=f"Eval {controller_name}", unit="step")
        
        while not (terminated or truncated):
            # Controller makes decision
            if hasattr(controller, 'get_rl_action'):
                action_arr = controller.get_rl_action(state)
            else:
                state_dict = self._parse_state(state, env.config)
                action = controller.get_action(state_dict)
                action_arr = self._dict_to_action(action, env.action_space)
            
            # Execute step
            next_state, reward, terminated, truncated, info = env.step(action_arr)
            
            pbar.update(1)
            pbar.set_postfix({'reward': f'{reward:.2f}'})
            
            # Track Handovers
            if 'e2_metrics' in info:
                ue_metrics = info['e2_metrics'].ue_metrics
                for ue_id, m in ue_metrics.items():
                    curr_cell = m.get('serving_cell', -1)
                    if ue_id in prev_cells:
                        if prev_cells[ue_id] != -1 and curr_cell != -1 and curr_cell != prev_cells[ue_id]:
                            total_handovers += 1
                    prev_cells[ue_id] = curr_cell

            # Collect metrics from info (which contains raw KPMs)
            e2_msg = info['e2_metrics']
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
        
        pbar.close()
        
        # Calculate aggregate metrics
        metrics = self._calculate_metrics(throughputs, delays, losses, queue_lengths, rb_utils, per_ue_stats, sinrs, rsrps, total_handovers)
        
        print(f"  Avg Throughput: {metrics.avg_throughput:.2f} Mbps")
        print(f"  Avg Packet Loss: {metrics.avg_packet_loss:.2%}")
        print(f"  Satisfied Users: {metrics.satisfied_user_ratio:.1%}")
        print(f"  Congestion Intensity: {metrics.congestion_intensity:.1%}")
        print(f"  Avg Jitter: {metrics.avg_jitter:.2f} ms")
        print(f"  Avg SINR: {metrics.avg_sinr:.2f} dB")
        print(f"  Avg PDR: {metrics.avg_pdr:.1%}")
        
        return metrics

    def _parse_state(self, state_arr, config) -> Dict:
        """Convert flattened numpy state back to dict for heuristic controllers"""
        state_dict = {}
        
        # We mainly need cell metrics for the Heuristic controller
        offset = config.num_ues * 4
        
        # Approximate packet loss per cell from state_arr
        # State: [UE_DL_TP, UE_DL_LOSS, UE_DL_DELAY, UE_SINR] * num_ues
        # Heuristic needs cell-level loss. We average UE losses.
        ue_losses = state_arr[1:offset:4]
        global_avg_loss = np.mean(ue_losses) if len(ue_losses) > 0 else 0.0
        
        for cell_id in range(config.num_cells):
            idx = offset + cell_id * 3
            state_dict[f'cell_{cell_id}_queue'] = state_arr[idx] * 1000
            state_dict[f'cell_{cell_id}_rb_util'] = state_arr[idx+1] 
            state_dict[f'cell_{cell_id}_power'] = state_arr[idx+2] * 36.0 + 10.0
            
            # Real-ish association: UEs are usually assigned to cells based on index
            # for sim simplicity (num_ues / num_cells).
            ues_per_cell = config.num_ues // config.num_cells
            start_ue = cell_id * ues_per_cell
            end_ue = (cell_id + 1) * ues_per_cell
            cell_ue_losses = ue_losses[start_ue:end_ue]
            
            state_dict[f'cell_{cell_id}_avg_loss'] = np.mean(cell_ue_losses) if len(cell_ue_losses) > 0 else global_avg_loss
            
        return state_dict

    def _dict_to_action(self, action_dict, action_space) -> np.ndarray:
        """Convert controller action dict to numpy array"""
        # Action: [tx_power, scheduler, harq, hyster, mac_delay, noise, weight] per cell
        # defaults
        flat_action = []
        num_cells = len(action_dict['tx_power'])
        
        for i in range(num_cells):
             # 1. Tx Power
             flat_action.append(action_dict['tx_power'][i])
             
             # 2. Scheduler
             sched = action_dict['scheduler'][i]
             sched_val = 0.0 if sched == 'PF' else 1.0 # Simple map
             flat_action.append(sched_val)
             
             # 3. HARQ
             flat_action.append(float(action_dict['harq_retx'][i]))
             
             # 4. Hysteresis (default 3.0)
             flat_action.append(3.0)
             
             # 5. MacChDelay (default 0.0)
             flat_action.append(0.0)
             
             # 6. NoiseFigure (default 5.0)
             flat_action.append(5.0)
             
             # 7. SchedulerWeight (default 1.0)
             flat_action.append(1.0)
             
        # Pad for UE priority weights (oran_ns3_env expects these)
        # We need to know num_ues. accessing env config would be better but pass it via argument?
        # Or just pad with zeros to match action_space size.
        current_len = len(flat_action)
        target_len = action_space.shape[0]
        if current_len < target_len:
            flat_action.extend([1.0] * (target_len - current_len))

        return np.array(flat_action, dtype=np.float32)

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
        total_handovers: int
    ) -> EvaluationMetrics:
        """Compute evaluation metrics including advanced congestion stats"""
        
        # Basic statistics
        avg_throughput = np.mean(throughputs)
        avg_delay = np.mean(delays)
        avg_packet_loss = np.mean(losses)
        
        # New: PDR and RSRP
        avg_pdr = 1.0 - avg_packet_loss
        avg_rsrp = np.mean(rsrps) if rsrps else -140.0
        
        # Percentiles
        p95_delay = np.percentile(delays, 95) if delays else 0.0
        max_packet_loss = np.max(losses) if losses else 0.0
        
        # Fairness
        if sum(throughputs) > 0:
            jains_fairness = (sum(throughputs) ** 2) / (
                len(throughputs) * sum([t**2 for t in throughputs])
            )
        else:
            jains_fairness = 0.0
        
        # QoS violations
        qos_violations = sum(
            1 for d, l in zip(delays, losses)
            if d > 100 or l > 0.05
        )
        
        # Recovery time
        recovery_time = self._calculate_recovery_time(losses)
        
        # Total downtime
        total_downtime = sum(1 for l in losses if l > 0.2) * 0.1
        
        # --- Advanced Metrics ---
        
        # 1. Congestion Intensity (% time where Queue > 10 pkts OR RB > 90%)
        # Note: queue_length from KPM is bytes or packets? Check oran_ns3_env. usually bytes.
        # Assuming normalized [0,1] or raw. In metrics check it was / 1000.
        # Let's assume RB Utilization is the main indicator.
        congested_steps = sum(1 for rb in rb_utils if rb > 0.9)
        congestion_intensity = congested_steps / len(rb_utils) if rb_utils else 0.0
        
        # 2. Satisfied User Ratio (Capacity Proxy)
        # SLA: Avg Tput > 1 Mbps AND Avg Delay < 100ms
        satisfied_count = 0
        total_ues = len(per_ue_stats)
        ue_avg_tputs = []
        
        for ue_id, stats in per_ue_stats.items():
            u_tput = np.mean(stats['tput'])
            u_delay = np.mean(stats['delay'])
            ue_avg_tputs.append(u_tput)
            
            if u_tput >= 1.0 and u_delay <= 100.0:
                satisfied_count += 1
                
        satisfied_user_ratio = satisfied_count / total_ues if total_ues > 0 else 0.0
        
        # 3. Peak Burst Loss (Max loss in 1s sliding window)
        window_size = 10 # 10 * 100ms = 1s
        peak_burst_loss = 0.0
        if len(losses) >= window_size:
            # simple sliding window average or max? User asked for "% packets lost in a time" imply rate.
            # Let's take stats over window.
            # Running average of loss over 1s.
            running_avg_loss = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
            peak_burst_loss = np.max(running_avg_loss)
        else:
            peak_burst_loss = max_packet_loss
            
        # 4. Cell Edge Throughput (5th percentile user throughput)
        cell_edge_tput = np.percentile(ue_avg_tputs, 5) if ue_avg_tputs else 0.0
        
        # 5. Jitter (Standard Deviation of Delay)
        # Using step-averaged delay variation as a proxy for network jitter
        avg_jitter = np.std(delays) if delays else 0.0
        
        # 6. Avg SINR
        avg_sinr = np.mean(sinrs) if sinrs else -10.0
        
        # 7. Avg Handover Count
        avg_handover_count = total_handovers / self.num_ues if self.num_ues > 0 else 0.0
        
        # --- Composite Scores (for Radar Chart) ---
        
        # 1. QoS Score: Weighted mix of Tput, Delay, Jitter, and Satisfied User Ratio
        # Target: Tput=10Mbps, Delay=20ms, Jitter=10ms
        norm_tput = min(avg_throughput / 10.0, 1.0)
        norm_delay = max(0.0, 1.0 - (avg_delay / 100.0)) # 0 score if delay > 100ms
        norm_jitter = max(0.0, 1.0 - (avg_jitter / 50.0))
        qos_score = (0.25 * norm_tput + 0.25 * norm_delay + 0.15 * norm_jitter + 0.35 * satisfied_user_ratio) * 100
        
        # 2. Reliability Score: Mix of Avg Packet Loss, Peak Burst Loss, and Handover Stability
        norm_loss = max(0.0, 1.0 - (avg_packet_loss * 10)) # 0 score if loss > 10%
        norm_burst = max(0.0, 1.0 - (peak_burst_loss * 5)) # 0 score if burst > 20%
        # Handover Stability: Penalize frequent handovers (Ping-Pong)
        # Assuming > 3 handovers per UE in 10s is "unstable"
        norm_ho = max(0.0, 1.0 - (avg_handover_count / 3.0))
        reliability_score = (0.4 * norm_loss + 0.3 * norm_burst + 0.3 * norm_ho) * 100
        
        # 3. Resource Score (Efficiency & Fairness): Usage vs Congestion vs Edge Experience
        # High Score = High Utilization without Congestion AND good Cell Edge performance AND Fairness
        avg_util = np.mean(rb_utils) if rb_utils else 0.0
        norm_edge = min(cell_edge_tput / 2.0, 1.0) # Target 2Mbps edge
        # "Resource Usage" roughly equates to utilization in simulation context, 
        # but "Efficiency" better captures network health.
        # Let's simple combine Utilization (activity), Edge Tput, and Fairness
        resource_score = (0.5 * min(avg_util * 100, 100.0) + 0.2 * (norm_edge * 100) + 0.3 * (jains_fairness * 100))
        
        # 4. Buffer Score: Buffer Health (Low Queues & Low Congestion Spikes)
        avg_q = np.mean(queue_lengths) if queue_lengths else 0.0
        norm_q = max(0.0, 1.0 - (avg_q / 100.0)) * 100 # Assuming 100 pkts is bad
        norm_cong = max(0.0, 1.0 - congestion_intensity) * 100 # 0% congestion is best
        buffer_score = 0.6 * norm_q + 0.4 * norm_cong
        
        # 5. PHY Score: Signal Quality (SINR + RSRP)
        # SINR -10 to 30. Map to 0-100. And include RSRP (-120 to -60)
        norm_sinr = min(max((avg_sinr + 10.0) / 40.0, 0.0), 1.0)
        norm_rsrp = min(max((avg_rsrp + 120.0) / 60.0, 0.0), 1.0) # -120dBm=0, -60dBm=1
        phy_score = (0.6 * norm_sinr + 0.4 * norm_rsrp) * 100
        
        return EvaluationMetrics(
            avg_throughput=avg_throughput,
            avg_delay=avg_delay,
            avg_packet_loss=avg_packet_loss,
            p95_delay=p95_delay,
            max_packet_loss=max_packet_loss,
            jains_fairness=jains_fairness,
            qos_violations=qos_violations,
            recovery_time=recovery_time,
            total_downtime=total_downtime,
            congestion_intensity=congestion_intensity,
            satisfied_user_ratio=satisfied_user_ratio,
            peak_burst_loss=peak_burst_loss,
            cell_edge_tput=cell_edge_tput,
            avg_jitter=avg_jitter,
            avg_sinr=avg_sinr,
            avg_pdr=avg_pdr,
            avg_rsrp=avg_rsrp,
            avg_handover_count=avg_handover_count,
            
            qos_score=qos_score,
            reliability_score=reliability_score,
            resource_score=resource_score,
            buffer_score=buffer_score,
            phy_score=phy_score
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


class VisualizationSuite:
    """Generate plots for paper/presentation"""
    
    @staticmethod
    def plot_comparison(
        results: Dict[str, EvaluationMetrics],
        scenario_name: str,
        save_path: str = None
    ):
        """Create comparison bar charts"""
        # Increased grid size to accommodate new metrics
        fig, axes = plt.subplots(4, 4, figsize=(20, 20))
        fig.suptitle(f'Radio-Cortex Performance: {scenario_name}', fontsize=16)
        
        controllers = list(results.keys())
        colors = ['gray', 'orange', 'green'][:len(controllers)]
        
        # Helper to plot bar chart
        def plot_metric(ax_idx, metric_attr, title, ylabel, scale=1.0):
            row, col = ax_idx
            ax = axes[row, col]
            values = [getattr(results[c], metric_attr) * scale for c in controllers]
            ax.bar(controllers, values, color=colors)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(axis='y', alpha=0.3)

        # Row 1: Core Performance
        plot_metric((0,0), 'avg_throughput', 'Average Throughput', 'Mbps')
        plot_metric((0,1), 'avg_packet_loss', 'Average Packet Loss', '%', scale=100)
        plot_metric((0,2), 'avg_delay', 'Average Delay', 'ms')
        plot_metric((0,3), 'jains_fairness', 'Fairness Index', 'Index')
        
        # Row 2: Stability & Reliability
        plot_metric((1,0), 'recovery_time', 'Recovery Time', 'Seconds')
        plot_metric((1,1), 'qos_violations', 'QoS Violations', 'Count')
        plot_metric((1,2), 'total_downtime', 'Total Downtime', 'Seconds')
        plot_metric((1,3), 'max_packet_loss', 'Max Packet Loss', '%', scale=100)

        # Row 3: Advanced Capacity & User Experience
        plot_metric((2,0), 'satisfied_user_ratio', 'Satisfied User Ratio', '%', scale=100)
        plot_metric((2,1), 'congestion_intensity', 'Congestion Intensity', '%', scale=100)
        plot_metric((2,2), 'peak_burst_loss', 'Peak Burst Loss (1s)', '%', scale=100)
        plot_metric((2,3), 'cell_edge_tput', 'Cell Edge Throughput (5th %)', 'Mbps')
        
        # Row 4: PHY & Jitter (New)
        plot_metric((3,0), 'avg_jitter', 'Avg Jitter', 'ms')
        plot_metric((3,1), 'avg_sinr', 'Avg SINR', 'dB')
        plot_metric((3,2), 'avg_pdr', 'Avg PDR', '%', scale=100)
        plot_metric((3,3), 'avg_rsrp', 'Avg RSRP', 'dBm')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        if save_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        else:
            plt.show()
    
    @staticmethod
    def plot_timeseries(
        kpm_timeline: List[Dict],
        scenario_name: str,
        save_path: str = None
    ):
        """Plot metrics over time"""
        # Extract time series
        df = pd.DataFrame(kpm_timeline)
        
        # Group by timestamp and aggregate
        agg = df.groupby('timestamp').agg({
            'throughput': 'mean',
            'delay': 'mean',
            'packet_loss': 'mean',
            'sinr': 'mean'
        }).reset_index()
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'Time Series: {scenario_name}', fontsize=16)
        
        # Throughput
        axes[0, 0].plot(agg['timestamp'], agg['throughput'], 'b-', linewidth=2)
        axes[0, 0].set_ylabel('Throughput (Mbps)')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].grid(alpha=0.3)
        axes[0, 0].set_title('Average Throughput')
        
        # Delay
        axes[0, 1].plot(agg['timestamp'], agg['delay'], 'r-', linewidth=2)
        axes[0, 1].set_ylabel('Delay (ms)')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].grid(alpha=0.3)
        axes[0, 1].set_title('Average Delay')
        
        # Packet Loss
        axes[1, 0].plot(agg['timestamp'], agg['packet_loss'] * 100, 'orange', linewidth=2)
        axes[1, 0].set_ylabel('Packet Loss (%)')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].grid(alpha=0.3)
        axes[1, 0].set_title('Packet Loss Rate')
        axes[1, 0].axhline(y=5, color='red', linestyle='--', label='QoS Threshold')
        axes[1, 0].legend()
        
        # SINR
        axes[1, 1].plot(agg['timestamp'], agg['sinr'], 'g-', linewidth=2)
        axes[1, 1].set_ylabel('SINR (dB)')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].grid(alpha=0.3)
        axes[1, 1].set_title('Average SINR')
        
        plt.tight_layout()
        
        if save_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    @staticmethod
    def plot_radar(
        results: Dict[str, EvaluationMetrics],
        scenario_name: str,
        save_path: str = None
    ):
        """Generate Radar Chart for Grouped Metrics"""
        labels = ['QoS', 'Reliability', 'Resource', 'Buffer', 'PHY']
        num_vars = len(labels)
        
        # Compute angle for each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1] # Close the circle
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        fig.suptitle(f'Network Health Profile: {scenario_name}', fontsize=16)
        
        controllers = list(results.keys())
        colors = ['gray', 'orange', 'green'][:len(controllers)]
        
        for i, controller in enumerate(controllers):
            metrics = results[controller]
            values = [
                metrics.qos_score,
                metrics.reliability_score,
                metrics.resource_score,
                metrics.buffer_score,
                metrics.phy_score
            ]
            values += values[:1]
            
            ax.plot(angles, values, color=colors[i], linewidth=2, label=controller)
            ax.fill(angles, values, color=colors[i], alpha=0.1)
        
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Draw axis labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        
        # Draw y-labels
        ax.set_rlabel_position(0)
        plt.yticks([20, 40, 60, 80], ["20", "40", "60", "80"], color="grey", size=7)
        plt.ylim(0, 100)
        
        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved radar chart to {save_path}")
        else:
            plt.show()
    
    @staticmethod
    def generate_latex_table(
        results: Dict[str, Dict[str, EvaluationMetrics]],
        save_path: str = None
    ):
        """Generate LaTeX table for paper"""
        latex = "\\begin{table}[h]\n"
        latex += "\\centering\n"
        latex += "\\caption{Radio-Cortex Performance Comparison}\n"
        latex += "\\begin{tabular}{|l|c|c|c|c|c|c|}\n"
        latex += "\\hline\n"
        latex += "\\textbf{Scenario} & \\textbf{Controller} & \\textbf{T-put} & \\textbf{Loss} & \\textbf{Congest} & \\textbf{Satisf} & \\textbf{Edge} & \\textbf{Jitr} & \\textbf{SINR} & \\textbf{PDR} & \\textbf{RSRP} & \\textbf{HO/UE} \\\\\n"
        latex += "& & (Mbps) & (\\%) & (\\%) & (\\%) & (Mbps) & (ms) & (dB) & (\\%) & (dBm) & (Count) \\\\\n"
        latex += "\\hline\n"
        
        for scenario, controllers in results.items():
            first = True
            for controller, metrics in controllers.items():
                if first:
                    latex += f"{scenario} "
                    first = False
                else:
                    latex += " "
                
                latex += f"& {controller} & {metrics.avg_throughput:.1f} & "
                latex += f"{metrics.avg_packet_loss*100:.2f} & {metrics.congestion_intensity*100:.1f} & "
                latex += f"{metrics.satisfied_user_ratio*100:.1f} & {metrics.cell_edge_tput:.2f} & "
                latex += f"{metrics.avg_jitter:.2f} & {metrics.avg_sinr:.1f} & "
                latex += f"{metrics.avg_pdr*100:.1f} & {metrics.avg_rsrp:.1f} & {metrics.avg_handover_count:.1f} \\\\\n"
            latex += "\\hline\n"
        
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"
        
        if save_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(latex)
            print(f"Saved LaTeX table to {save_path}")
        else:
            print(latex)
        
        return latex
