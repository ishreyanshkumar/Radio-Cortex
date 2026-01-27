"""
Radio-Cortex Evaluation and Baseline Comparison Suite

Compares self-healing performance against baselines:
1. Static RAN (no RL control)
2. Heuristic-based control (rule-based)
3. Radio-Cortex (Hebbian + RL)

Metrics:
- Packet loss rate
- Average throughput
- Fairness (Jain's index)
- QoS violations
- Recovery time
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
import json
from pathlib import Path
import seaborn as sns


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
        
        Args:
            controller: Controller instance (Baseline, Heuristic, or RL)
            env: Gym environment (ORANns3Env)
            controller_name: Name for logging
        
        Returns:
            Performance metrics
        """
        print(f"\nEvaluating {controller_name}...")
        
        # Track metrics over time
        throughputs = []
        delays = []
        losses = []
        
        # Reset environment
        state, info = env.reset()
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            # Controller makes decision
            if hasattr(controller, 'get_rl_action'):
                # Real RL Agent: Uses raw numpy state
                action_arr = controller.get_rl_action(state)
            else:
                # Heuristic/Baseline: Uses parsed state dict
                state_dict = self._parse_state(state, env.config)
                action = controller.get_action(state_dict)
                action_arr = self._dict_to_action(action, env.action_space)
            
            # Execute step
            next_state, reward, terminated, truncated, info = env.step(action_arr)
            
            # Collect metrics from info (which contains raw KPMs)
            e2_msg = info['e2_metrics']
            ue_kpms = list(e2_msg.ue_metrics.values())
            
            # Aggregate per-step metrics
            step_tput = np.mean([m['throughput'] for m in ue_kpms]) if ue_kpms else 0.0
            step_delay = np.mean([m['delay'] for m in ue_kpms]) if ue_kpms else 0.0
            step_loss = np.mean([m['packet_loss'] for m in ue_kpms]) if ue_kpms else 0.0
            
            throughputs.append(step_tput)
            delays.append(step_delay)
            losses.append(step_loss)
            
            state = next_state
        
        # Calculate aggregate metrics
        metrics = self._calculate_metrics(throughputs, delays, losses)
        
        print(f"  Avg Throughput: {metrics.avg_throughput:.2f} Mbps")
        print(f"  Avg Packet Loss: {metrics.avg_packet_loss:.2%}")
        print(f"  Fairness Index: {metrics.jains_fairness:.3f}")
        
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
        # Action: [tx_power, scheduler, harq, hyster] per cell
        # defaults
        flat_action = []
        num_cells = len(action_dict['tx_power'])
        
        for i in range(num_cells):
             # Tx Power
             flat_action.append(action_dict['tx_power'][i])
             
             # Scheduler
             sched = action_dict['scheduler'][i]
             sched_val = 0.0 if sched == 'PF' else 1.0 # Simple map
             flat_action.append(sched_val)
             
             # HARQ
             flat_action.append(float(action_dict['harq_retx'][i]))
             
             # Hysteresis (default 3.0)
             flat_action.append(3.0)
             
        return np.array(flat_action, dtype=np.float32)
    

    
    def _calculate_metrics(
        self,
        throughputs: List[float],
        delays: List[float],
        losses: List[float]
    ) -> EvaluationMetrics:
        """Compute evaluation metrics"""
        
        # Basic statistics
        avg_throughput = np.mean(throughputs)
        avg_delay = np.mean(delays)
        avg_packet_loss = np.mean(losses)
        
        # Percentiles
        p95_delay = np.percentile(delays, 95)
        max_packet_loss = np.max(losses)
        
        # Fairness (Jain's index)
        if sum(throughputs) > 0:
            jains_fairness = (sum(throughputs) ** 2) / (
                len(throughputs) * sum([t**2 for t in throughputs])
            )
        else:
            jains_fairness = 0.0
        
        # QoS violations (delay > 100ms or loss > 5%)
        qos_violations = sum(
            1 for d, l in zip(delays, losses)
            if d > 100 or l > 0.05
        )
        
        # Recovery time (time until metrics return to normal)
        recovery_time = self._calculate_recovery_time(losses)
        
        # Total downtime (time with >20% packet loss)
        total_downtime = sum(1 for l in losses if l > 0.2) * 0.1  # 100ms per sample
        
        return EvaluationMetrics(
            avg_throughput=avg_throughput,
            avg_delay=avg_delay,
            avg_packet_loss=avg_packet_loss,
            p95_delay=p95_delay,
            max_packet_loss=max_packet_loss,
            jains_fairness=jains_fairness,
            qos_violations=qos_violations,
            recovery_time=recovery_time,
            total_downtime=total_downtime
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
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Radio-Cortex Performance: {scenario_name}', fontsize=16)
        
        controllers = list(results.keys())
        
        # Plot 1: Throughput
        ax = axes[0, 0]
        throughputs = [results[c].avg_throughput for c in controllers]
        ax.bar(controllers, throughputs, color=['gray', 'orange', 'green'])
        ax.set_ylabel('Throughput (Mbps)')
        ax.set_title('Average Throughput')
        ax.grid(axis='y', alpha=0.3)
        
        # Plot 2: Packet Loss
        ax = axes[0, 1]
        losses = [results[c].avg_packet_loss * 100 for c in controllers]
        ax.bar(controllers, losses, color=['gray', 'orange', 'green'])
        ax.set_ylabel('Packet Loss (%)')
        ax.set_title('Average Packet Loss')
        ax.grid(axis='y', alpha=0.3)
        
        # Plot 3: Delay
        ax = axes[0, 2]
        delays = [results[c].avg_delay for c in controllers]
        ax.bar(controllers, delays, color=['gray', 'orange', 'green'])
        ax.set_ylabel('Delay (ms)')
        ax.set_title('Average Delay')
        ax.grid(axis='y', alpha=0.3)
        
        # Plot 4: Fairness
        ax = axes[1, 0]
        fairness = [results[c].jains_fairness for c in controllers]
        ax.bar(controllers, fairness, color=['gray', 'orange', 'green'])
        ax.set_ylabel("Jain's Fairness Index")
        ax.set_title('Fairness')
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        
        # Plot 5: Recovery Time
        ax = axes[1, 1]
        recovery = [results[c].recovery_time for c in controllers]
        ax.bar(controllers, recovery, color=['gray', 'orange', 'green'])
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Recovery Time')
        ax.grid(axis='y', alpha=0.3)
        
        # Plot 6: QoS Violations
        ax = axes[1, 2]
        violations = [results[c].qos_violations for c in controllers]
        ax.bar(controllers, violations, color=['gray', 'orange', 'green'])
        ax.set_ylabel('Count')
        ax.set_title('QoS Violations')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
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
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
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
        latex += "\\begin{tabular}{|l|c|c|c|c|}\n"
        latex += "\\hline\n"
        latex += "\\textbf{Scenario} & \\textbf{Controller} & \\textbf{Throughput} & \\textbf{Packet Loss} & \\textbf{Recovery Time} \\\\\n"
        latex += "& & (Mbps) & (\\%) & (s) \\\\\n"
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
                latex += f"{metrics.avg_packet_loss*100:.2f} & {metrics.recovery_time:.2f} \\\\\n"
            latex += "\\hline\n"
        
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(latex)
            print(f"Saved LaTeX table to {save_path}")
        else:
            print(latex)
        
        return latex


# ============================================================================
# Main Evaluation Script
# ============================================================================


