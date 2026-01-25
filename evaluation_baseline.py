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
        kpm_timeline: List[Dict],
        controller_name: str
    ) -> EvaluationMetrics:
        """
        Simulate controller behavior on given scenario
        
        Args:
            controller: Controller instance (Baseline, Heuristic, or RL)
            kpm_timeline: Pre-generated KPM reports (scenario)
            controller_name: Name for logging
        
        Returns:
            Performance metrics
        """
        print(f"\nEvaluating {controller_name}...")
        
        # Track metrics over time
        throughputs = []
        delays = []
        losses = []
        
        # Group KPM by timestep
        timesteps = {}
        for kpm in kpm_timeline:
            t = kpm['timestamp']
            if t not in timesteps:
                timesteps[t] = []
            timesteps[t].append(kpm)
        
        # Simulate controller decisions
        for t in sorted(timesteps.keys()):
            kpm_reports = timesteps[t]
            
            # Build state from KPM reports
            state = self._build_state(kpm_reports)
            
            # Controller makes decision
            action = controller.get_action(state)
            
            # Apply action effects (simplified simulation)
            modified_kpm = self._apply_action(kpm_reports, action)
            
            # Collect metrics
            for kpm in modified_kpm:
                throughputs.append(kpm['throughput'])
                delays.append(kpm['delay'])
                losses.append(kpm['packet_loss'])
        
        # Calculate aggregate metrics
        metrics = self._calculate_metrics(throughputs, delays, losses)
        
        print(f"  Avg Throughput: {metrics.avg_throughput:.2f} Mbps")
        print(f"  Avg Packet Loss: {metrics.avg_packet_loss:.2%}")
        print(f"  Fairness Index: {metrics.jains_fairness:.3f}")
        
        return metrics
    
    def _build_state(self, kpm_reports: List[Dict]) -> Dict:
        """Convert KPM reports to state dict"""
        state = {}
        
        # Per-cell aggregates
        for cell_id in range(self.num_cells):
            cell_kpms = [k for k in kpm_reports if k['cell_id'] == cell_id]
            if cell_kpms:
                state[f'cell_{cell_id}_rb_util'] = np.mean([
                    k.get('rb_utilization', 0.5) for k in cell_kpms
                ])
                state[f'cell_{cell_id}_avg_loss'] = np.mean([
                    k['packet_loss'] for k in cell_kpms
                ])
        
        return state
    
    def _apply_action(
        self,
        kpm_reports: List[Dict],
        action: Dict
    ) -> List[Dict]:
        """
        Simulate effect of controller action on network performance
        
        This is a simplified model. In reality, actions would modify
        ns-3 simulation via E2 interface.
        """
        modified = []
        
        for kpm in kpm_reports:
            cell_id = kpm['cell_id']
            kpm_copy = kpm.copy()
            
            # Model action effects
            # 1. Tx Power change affects SINR
            if action['tx_power'][cell_id] < 23.0:
                # Lower power → slightly worse SINR but offloads UEs
                kpm_copy['sinr'] -= 2.0
                kpm_copy['throughput'] *= 1.1  # Offloading benefit
            elif action['tx_power'][cell_id] > 26.0:
                # Higher power → better SINR
                kpm_copy['sinr'] += 2.0
            
            # 2. Scheduler change affects fairness
            if action['scheduler'][cell_id] == 'RR':
                # Round Robin reduces variance (fairness)
                kpm_copy['throughput'] *= 0.9  # Slight throughput penalty
            
            # 3. HARQ retransmissions affect loss vs delay
            if action['harq_retx'][cell_id] > 4:
                # More retries → less loss but higher delay
                kpm_copy['packet_loss'] *= 0.7
                kpm_copy['delay'] *= 1.3
            
            modified.append(kpm_copy)
        
        return modified
    
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

def run_full_evaluation():
    """Run complete evaluation suite"""
    print("=== Radio-Cortex Evaluation Suite ===\n")
    
    # Load scenarios
    from congestion_scenarios import EvaluationSuite as ScenarioSuite
    
    scenario_suite = ScenarioSuite(num_ues=20, num_cells=3)
    scenarios = scenario_suite.generate_all()
    
    # Initialize controllers
    baseline = BaselineController(num_cells=3)
    heuristic = HeuristicController(num_cells=3)
    # RL controller would be loaded here: radio_cortex = load_trained_model()
    
    # For now, simulate RL as improved heuristic
    class SimulatedRLController(HeuristicController):
        def get_action(self, state):
            # Better rules that RL would learn
            actions = super().get_action(state)
            # RL learns to be more aggressive
            actions['tx_power'] = [p * 0.9 for p in actions['tx_power']]
            return actions
    
    radio_cortex = SimulatedRLController(num_cells=3)
    
    # Run evaluations
    evaluator = EvaluationRunner(num_ues=20, num_cells=3)
    all_results = {}
    
    for scenario_name, kpm_timeline in scenarios.items():
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario_name}")
        print('='*60)
        
        results = {
            'Baseline': evaluator.evaluate_controller(baseline, kpm_timeline, 'Baseline'),
            'Heuristic': evaluator.evaluate_controller(heuristic, kpm_timeline, 'Heuristic'),
            'Radio-Cortex': evaluator.evaluate_controller(radio_cortex, kpm_timeline, 'Radio-Cortex')
        }
        
        all_results[scenario_name] = results
        
        # Generate plots
        VisualizationSuite.plot_comparison(
            results,
            scenario_name,
            save_path=f'eval_{scenario_name}.png'
        )
        
        VisualizationSuite.plot_timeseries(
            kpm_timeline,
            scenario_name,
            save_path=f'timeseries_{scenario_name}.png'
        )
    
    # Generate summary table
    VisualizationSuite.generate_latex_table(
        all_results,
        save_path='results_table.tex'
    )
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    for scenario_name, results in all_results.items():
        print(f"\n{scenario_name}:")
        for controller, metrics in results.items():
            improvement = ""
            if controller == "Radio-Cortex":
                baseline_loss = all_results[scenario_name]['Baseline'].avg_packet_loss
                rc_loss = metrics.avg_packet_loss
                reduction = (1 - rc_loss / baseline_loss) * 100
                improvement = f" ({reduction:.1f}% loss reduction)"
            
            print(f"  {controller}: "
                  f"Tput={metrics.avg_throughput:.1f} Mbps, "
                  f"Loss={metrics.avg_packet_loss:.2%}, "
                  f"Recovery={metrics.recovery_time:.1f}s"
                  f"{improvement}")
    
    print("\n✓ Evaluation complete. Results saved to current directory.")


if __name__ == "__main__":
    run_full_evaluation()
