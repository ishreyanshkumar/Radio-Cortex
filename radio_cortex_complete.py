"""
Radio-Cortex: Complete Integration
Self-Healing O-RAN xApp with Hebbian Learning + Reinforcement Learning

This file integrates all components:
1. ns-3 O-RAN environment (E2 interface)
2. Pathway Hebbian trust graph
3. RL training pipeline (PPO)
4. Congestion scenarios
5. Evaluation suite

Usage:
    python radio_cortex_complete.py --mode train
    python radio_cortex_complete.py --mode eval
    python radio_cortex_complete.py --mode demo
"""

import argparse
import numpy as np
import torch
from pathlib import Path
import json
import time
from typing import Dict, List, Optional

# Import all components
from oran_ns3_env import ORANns3Env, NS3Config, create_oran_env
from rl_training_pipeline import PPOTrainer, evaluate_policy
from congestion_scenarios import (
    CongestionScenarioGenerator,
    ScenarioConfig,
    ScenarioType,
    EvaluationSuite
)
from evaluation_baseline import (
    EvaluationRunner,
    BaselineController,
    HeuristicController,
    VisualizationSuite
)


# ============================================================================
# Radio-Cortex Agent (Hebbian + RL Integration)
# ============================================================================

class RadioCortexAgent:
    """
    Radio-Cortex agent:
    - System 2 (RL): Strategic long-term optimization
    """
    
    def __init__(
        self,
        num_ues: int,
        num_cells: int,
        policy_model: Optional[torch.nn.Module] = None
    ):
        self.num_ues = num_ues
        self.num_cells = num_cells
        
        # System 2: RL policy
        self.policy = policy_model
        
        # Metrics tracking
        self.metrics_history = []
    
    def get_rl_action(self, base_state: np.ndarray) -> np.ndarray:
        """
        System 2: Strategic RL decision
        """
        if self.policy is None:
            # Fallback: heuristic
            return self._heuristic_action(base_state)
        
        # RL inference directly on base state
        state_tensor = torch.FloatTensor(base_state).unsqueeze(0)
        with torch.no_grad():
            action = self.policy.get_action(state_tensor, deterministic=False)
            if isinstance(action, tuple):
                action = action[0]
        
        return action.cpu().numpy()[0]
    
    def _heuristic_action(self, state: np.ndarray) -> np.ndarray:
        """Fallback heuristic if RL policy not available"""
        # Simple static rule
        action = np.zeros(self.num_cells * 4)
        for cell_id in range(self.num_cells):
            idx = cell_id * 4
            
            # Default params
            action[idx] = 26.0  # Power
            action[idx + 1] = 0   # Scheduler
            action[idx + 2] = 4   # HARQ
            action[idx + 3] = 3.0 # Hysteresis
        
        return action
    
    def log_metrics(self, kpm_reports: List[Dict]):
        """Track performance over time"""
        metrics = {
            'timestamp': time.time(),
            'avg_throughput': np.mean([k['throughput'] for k in kpm_reports]) if kpm_reports else 0,
            'avg_loss': np.mean([k['packet_loss'] for k in kpm_reports]) if kpm_reports else 0,
        }
        self.metrics_history.append(metrics)


# ============================================================================
# Training Pipeline
# ============================================================================

def train_radio_cortex(
    config: NS3Config,
    total_timesteps: int = 100000,
    save_path: str = 'models/radio_cortex.pt'
):
    """
    Train Radio-Cortex agent
    
    Steps:
    1. Create O-RAN environment
    2. Initialize PPO trainer
    3. Train RL agent
    4. Save trained model
    """
    print("="*60)
    print("TRAINING RADIO-CORTEX")
    print("="*60)
    
    # Create environment
    env = create_oran_env(config)
    
    # Create trainer
    trainer = PPOTrainer(
        env=env,
        hidden_dim=256,
        lr=3e-4,
        gamma=0.99,
        clip_epsilon=0.2
    )
    
    # Train
    print(f"\nTraining for {total_timesteps} timesteps...")
    trainer.train(
        total_timesteps=total_timesteps,
        rollout_steps=2048,
        log_interval=5
    )
    
    # Save
    Path(save_path).parent.mkdir(exist_ok=True)
    trainer.save(save_path)
    
    print(f"\n✓ Training complete. Model saved to {save_path}")
    
    return trainer


# ============================================================================
# Evaluation Pipeline
# ============================================================================

def evaluate_radio_cortex(
    config: NS3Config,
    model_path: str = 'models/radio_cortex.pt'
):
    """
    Evaluate Radio-Cortex against baselines
    
    Scenarios:
    1. Flash crowd
    2. Cell failure
    3. Mobility storm
    4. Black swan
    """
    print("="*60)
    print("EVALUATING RADIO-CORTEX")
    print("="*60)
    
    # Generate scenarios
    scenario_suite = EvaluationSuite(
        num_ues=config.num_ues,
        num_cells=config.num_cells
    )
    scenarios = scenario_suite.generate_all()
    
    # Initialize controllers
    baseline = BaselineController(num_cells=config.num_cells)
    heuristic = HeuristicController(num_cells=config.num_cells)
    
    # Load trained Radio-Cortex
    # (In full implementation, would load actual RL policy)
    radio_cortex = RadioCortexAgent(
        num_ues=config.num_ues,
        num_cells=config.num_cells,
        policy_model=None  # Would load from model_path
    )
    
    # Run evaluations
    evaluator = EvaluationRunner(
        num_ues=config.num_ues,
        num_cells=config.num_cells
    )
    
    all_results = {}
    
    for scenario_name, kpm_timeline in scenarios.items():
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario_name}")
        print('='*60)
        
        results = {
            'Baseline': evaluator.evaluate_controller(
                baseline, kpm_timeline, 'Baseline'
            ),
            'Heuristic': evaluator.evaluate_controller(
                heuristic, kpm_timeline, 'Heuristic'
            ),
            # Would add Radio-Cortex here with actual RL controller
        }
        
        all_results[scenario_name] = results
        
        # Generate visualizations
        VisualizationSuite.plot_comparison(
            results,
            scenario_name,
            save_path=f'results/{scenario_name}_comparison.png'
        )
    
    # Summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    for scenario_name, results in all_results.items():
        print(f"\n{scenario_name}:")
        for controller, metrics in results.items():
            print(f"  {controller}:")
            print(f"    Throughput: {metrics.avg_throughput:.2f} Mbps")
            print(f"    Packet Loss: {metrics.avg_packet_loss:.2%}")
            print(f"    Recovery Time: {metrics.recovery_time:.2f}s")
    
    return all_results


# ============================================================================
# Demo Mode
# ============================================================================

def run_demo():
    """
    Interactive demo of Radio-Cortex capabilities
    """
    print("="*60)
    print("RADIO-CORTEX DEMO")
    print("="*60)
    print("Demo mode is disabled as Hebbian Trust component has been removed.")
    print("Use --mode train or --mode eval to run RL agent.")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Radio-Cortex: Self-Healing O-RAN xApp')
    parser.add_argument(
        '--mode',
        choices=['train', 'eval', 'demo'],
        default='demo',
        help='Operation mode'
    )
    parser.add_argument('--num-ues', type=int, default=20, help='Number of UEs')
    parser.add_argument('--num-cells', type=int, default=3, help='Number of cells')
    parser.add_argument('--timesteps', type=int, default=100000, help='Training timesteps')
    parser.add_argument('--model-path', type=str, default='models/radio_cortex.pt', help='Model path')
    
    args = parser.parse_args()
    
    # Configuration
    config = NS3Config(
        num_ues=args.num_ues,
        num_cells=args.num_cells,
        sim_time=10.0,
        kpm_interval_ms=100,
        seed=42
    )
    
    # Execute mode
    if args.mode == 'train':
        trainer = train_radio_cortex(
            config=config,
            total_timesteps=args.timesteps,
            save_path=args.model_path
        )
    
    elif args.mode == 'eval':
        results = evaluate_radio_cortex(
            config=config,
            model_path=args.model_path
        )
    
    elif args.mode == 'demo':
        run_demo()


if __name__ == "__main__":
    main()
