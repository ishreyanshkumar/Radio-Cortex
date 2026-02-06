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
            raise RuntimeError("RL Policy not loaded! 'Everything Real' mode requires a trained model.")
        
        # RL inference directly on base state
        state_tensor = torch.FloatTensor(base_state).unsqueeze(0)
        with torch.no_grad():
            action = self.policy.get_action(state_tensor, deterministic=True)
            if isinstance(action, tuple):
                action = action[0]
        
        return action.cpu().numpy()[0]
    
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
    total_timesteps: int = 10000,
    save_path: str = 'models/radio_cortex.pt',
    lr: float = 3e-4,
    gamma: float = 0.99,
    batch_size: int = 64,
    hidden_dim: int = 256,
    gae_lambda: float = 0.95,
    clip_epsilon: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    rollout_steps: int = 2048,
    log_interval: int = 5,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
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
    print(f"LR: {lr}, Gamma: {gamma}, Batch: {batch_size}, Device: {device}")
    print("="*60)
    
    # Create environment
    env = create_oran_env(config)
  #  print("[debug] env created, about to build PPO trainer")
    # Create trainer
    trainer = PPOTrainer(
        env=env,
        hidden_dim=hidden_dim,
        lr=lr,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_epsilon=clip_epsilon,
        vf_coef=vf_coef,
        ent_coef=ent_coef,
        max_grad_norm=max_grad_norm,
        device=device
    )
    
    # Train
    print(f"\nTraining for {total_timesteps} timesteps...")
    trainer.train(
        total_timesteps=total_timesteps,
        rollout_steps=rollout_steps,
        log_interval=log_interval,
        batch_size=batch_size
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
    
    # Define REAL scenarios supported by C++ simulation
    scenarios = [
        "flash_crowd",
        "mobility_storm",
        "traffic_burst",
        "handover_ping_pong"
    ]
    
    # Initialize controllers (Baseline only for now to save time, or Heuristic)
    # To properly compare "Before vs After", we should run Baseline vs RadioCortex
    baseline = BaselineController(num_cells=config.num_cells)
    heuristic = HeuristicController(num_cells=config.num_cells)
    
    # Run evaluations
    evaluator = EvaluationRunner(
        num_ues=config.num_ues,
        num_cells=config.num_cells
    )
    
    all_results = {}
    
    for scenario_name in scenarios:
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario_name}")
        print('='*60)
        
        # We need to run the Environment for EACH controller for EACH scenario
        # This is expensive (real time), but necessary for "Realness"
        
        results = {}
        
        # 1. Evaluate Baseline
        print(f"--- Running Baseline on {scenario_name} ---")
        config.scenario = scenario_name
        env = create_oran_env(config)
        try:
            results['Baseline'] = evaluator.evaluate_controller(
                baseline, env, 'Baseline'
            )
        finally:
            env.close()

        # 2. Evaluate Heuristic
        print(f"--- Running Heuristic on {scenario_name} ---")
        config.scenario = scenario_name
        env = create_oran_env(config)
        try:
            results['Heuristic'] = evaluator.evaluate_controller(
                heuristic, env, 'Heuristic'
            )
        finally:
            env.close()

        # 3. Evaluate Radio-Cortex (Real RL Model)
        print(f"--- Running Radio-Cortex on {scenario_name} ---")
        
        # Load real policy from models/
        from neural_networks import ActorCritic
        # state_dim MUST match ORANns3Env (12 per UE + 5 per Cell)
        state_dim = config.num_ues * 12 + config.num_cells * 5
        # action_dim MUST match ORANns3Env (7 per Cell + 1 per UE)
        action_dim = config.num_cells * 7 + config.num_ues
        policy = ActorCritic(state_dim, action_dim).to('cpu')
        
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            policy.load_state_dict(checkpoint['policy_state_dict'])
            print(f"  Successfully loaded model from {model_path}")
        except FileNotFoundError:
            print(f"  [ERROR] Model {model_path} not found! Run with --mode train first.")
            raise
            
        rc_agent = RadioCortexAgent(config.num_ues, config.num_cells, policy_model=policy)
        
        env = create_oran_env(config)
        try:
             # Real Evaluation Runner loop
             results['Radio-Cortex'] = evaluator.evaluate_controller(
                 rc_agent, env, 'Radio-Cortex'
             )
        finally:
            env.close()

        all_results[scenario_name] = results
        
        # Generate visualizations
        VisualizationSuite.plot_comparison(
            results,
            scenario_name,
            save_path=f'results/{scenario_name}_comparison.png'
        )
        VisualizationSuite.plot_radar(
            results,
            scenario_name,
            save_path=f'results/{scenario_name}_radar.png'
        )
        
        # Generate LaTeX Table
        VisualizationSuite.generate_latex_table(
            {scenario_name: results},
            save_path=f'results/{scenario_name}_metrics.tex'
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
        default='train',
        help='Operation mode'
    )
    # Environment configs
    parser.add_argument('--num-ues', type=int, default=20, help='Number of UEs')
    parser.add_argument('--num-cells', type=int, default=3, help='Number of cells')
    parser.add_argument('--scenario', type=str, default='flash_crowd', help='ns-3 Scenario (flash_crowd, mobility_storm)')
    parser.add_argument('--kpm-interval', type=int, default=100, help='KPM Reporting Interval (ms)')
    
    # Training configs
    parser.add_argument('--total-timesteps', type=int, default=10000, help='Training timesteps')
    parser.add_argument('--model-path', type=str, default='models/radio_cortex.pt', help='Model save path')
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for optimization')
    # Advanced PPO configs
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension for actor/critic networks')
    parser.add_argument('--gae-lambda', type=float, default=0.95, help='GAE lambda')
    parser.add_argument('--clip-epsilon', type=float, default=0.2, help='PPO clip epsilon')
    parser.add_argument('--vf-coef', type=float, default=0.5, help='Value function coefficient')
    parser.add_argument('--ent-coef', type=float, default=0.01, help='Entropy coefficient')
    parser.add_argument('--max-grad-norm', type=float, default=0.5, help='Max gradient norm')
    parser.add_argument('--rollout-steps', type=int, default=2048, help='Steps per rollout')
    parser.add_argument('--log-interval', type=int, default=5, help='Logging interval (updates)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device (cpu/cuda)')
    parser.add_argument('--config', type=str, default=None, help='Path to JSON config file to override arguments')

    args = parser.parse_args()

    # Load config file if provided
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config_args = json.load(f)
                for key, value in config_args.items():
                    if hasattr(args, key):
                        setattr(args, key, value)
            print(f"Loaded configuration from {args.config}")
        except FileNotFoundError:
            print(f"Warning: Config file {args.config} not found.")

    
    # Configuration
    config = NS3Config(
        num_ues=args.num_ues,
        num_cells=args.num_cells,
        sim_time=10.0,
        kpm_interval_ms=args.kpm_interval,
        seed=42,
        scenario=args.scenario
    )
    
    # Execute mode
    if args.mode == 'train':
        trainer = train_radio_cortex(
            config=config,
            total_timesteps=args.total_timesteps,
            save_path=args.model_path,
            lr=args.learning_rate,
            gamma=args.gamma,
            batch_size=args.batch_size,
            hidden_dim=args.hidden_dim,
            gae_lambda=args.gae_lambda,
            clip_epsilon=args.clip_epsilon,
            vf_coef=args.vf_coef,
            ent_coef=args.ent_coef,
            max_grad_norm=args.max_grad_norm,
            rollout_steps=args.rollout_steps,
            log_interval=args.log_interval,
            device=args.device
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
