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
    print("="*60)
    print("EVALUATING RADIO-CORTEX")
    print("="*60)
    
    # Scenario to run (12 available: flash_crowd, mobility_storm, traffic_burst, handover_ping_pong, 
    # sleepy_campus, ambulance, adversarial, commuter_rush, mixed_reality, urban_canyon, iot_tsunami, spectrum_crunch)
    all_scenarios = [
        "flash_crowd",        # Sudden surge in usage
        "mobility_storm",     # Rapid handovers
        "traffic_burst",      # Application data spikes
        "handover_ping_pong", # Boundary oscillations
        "sleepy_campus",      # Night/Day load cycles
        "ambulance",          # High-priority stream
        "adversarial",        # Rapid signal fluctuations
        "commuter_rush",      # Mass group mobility
        "mixed_reality",      # Interactive vs Bulk slices
        "urban_canyon",       # Signal blockage
        "iot_tsunami",        # Massive device scale
        "spectrum_crunch"     # Multi-band management
    ]
    
    # If a specific scenario was requested, just run that one.
    # Otherwise, run all if scenario is None or 'all'.
    if config.scenario and config.scenario in all_scenarios:
        scenarios = [config.scenario]
        print(f"Running single scenario evaluation: {config.scenario}")
    else:
        scenarios = all_scenarios
        print(f"Running full benchmark on all {len(all_scenarios)} scenarios...")
    
    # Initialize Baseline controller (static RAN, no AI)
    baseline = BaselineController(num_cells=config.num_cells)
    
    # Run evaluations
    evaluator = EvaluationRunner(
        num_ues=config.num_ues,
        num_cells=config.num_cells
    )
    
    all_results = {}
    
    for scenario_name in scenarios:
        print(f"\n{'='*70}")
        print(f"  SCENARIO: {scenario_name.upper().replace('_', ' ')}")
        print('='*70)
        
        results = {}
        
        # 1. Evaluate Baseline (Static RAN - No AI)
        print(f"\n[1/2] Baseline (Static RAN)...")
        config.scenario = scenario_name
        env = create_oran_env(config)
        try:
            results['Baseline'] = evaluator.evaluate_controller(
                baseline, env, 'Baseline'
            )
        finally:
            env.close()

        # 2. Evaluate Radio-Cortex (PPO RL Agent)
        print(f"\n[2/2] Radio-Cortex (RL Agent)...")
        
        # Load real policy from models/
        from neural_networks import ActorCritic
        
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            state_dict = checkpoint['policy_state_dict']
            
            # Detect dimensions from state_dict
            # feature_net.0.weight shape is [hidden_dim, state_dim]
            # actor_mean.0.bias shape is [action_dim]
            stored_action_dim = state_dict['actor_mean.0.bias'].shape[0]
            stored_state_dim = state_dict['feature_net.0.weight'].shape[1]
            
            # Verify if it matches current config
            expected_state_dim = config.num_ues * 12 + config.num_cells * 5
            expected_action_dim = config.num_cells * 7 + config.num_ues
            
            if stored_action_dim != expected_action_dim or stored_state_dim != expected_state_dim:
                # Calculate what the model expects
                # action_dim = cells * 7 + ues => ues = action_dim - cells * 7
                detected_ues = stored_action_dim - (config.num_cells * 7)
                print(f"      [WARNING] Model dimension mismatch!")
                print(f"      Current Config: {config.num_ues} UEs ({expected_action_dim} actions)")
                print(f"      Model Weights : {detected_ues} UEs ({stored_action_dim} actions)")
                print(f"      -> Re-initializing policy with {detected_ues} UEs to match weights...")
                
                # Force config to match model for evaluation to work
                config.num_ues = detected_ues
                state_dim = stored_state_dim
                action_dim = stored_action_dim
            else:
                state_dim = expected_state_dim
                action_dim = expected_action_dim

            policy = ActorCritic(state_dim, action_dim).to('cpu')
            policy.load_state_dict(state_dict)
            print(f"      Model loaded: {model_path}")
            
        except FileNotFoundError:
            print(f"      [ERROR] Model {model_path} not found! Run with --mode train first.")
            raise
        except KeyError as e:
            print(f"      [ERROR] Invalid model checkpoint format: {e}")
            raise
            
        rc_agent = RadioCortexAgent(config.num_ues, config.num_cells, policy_model=policy)
        
        env = create_oran_env(config)
        try:
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

        # Generate CSV Report
        VisualizationSuite.generate_csv(
            {scenario_name: results},
            save_path=f'results/{scenario_name}_metrics.csv'
        )
    
    # Summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    for scenario_name, results in all_results.items():
        print(f"\n{scenario_name}:")
        print(f"{'Controller':<15} | {'Tput':<6} | {'Loss%':<6} | {'Satisf%':<7} | {'SpecEff':<7} | {'Score':<5}")
        print("-" * 60)
        for controller, metrics in results.items():
            avg_score = (metrics.qos_score + metrics.reliability_score + metrics.resource_score + 
                         metrics.buffer_score + metrics.phy_score + metrics.ric_score) / 6.0
            print(f"{controller:<15} | {metrics.avg_throughput:>6.2f} | {metrics.avg_packet_loss*100:>6.2f} | {metrics.satisfied_user_ratio*100:>7.1f} | {metrics.spectral_efficiency:>7.2f} | {avg_score:>5.1f}")
    
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
    parser.add_argument('--scenario', type=str, default=None, help='ns-3 Scenario (flash_crowd, mobility_storm, etc.). If omitted in eval mode, runs all scenarios.')
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
    parser.add_argument('--system-bandwidth-mhz', type=float, default=10.0, help='System Bandwidth in MHz (5.0, 10.0, 20.0)')
    parser.add_argument('--sim-time', type=float, default=10.0, help='Simulation duration in seconds')

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
        sim_time=args.sim_time,
        kpm_interval_ms=args.kpm_interval,
        seed=42,
        scenario=args.scenario,
        system_bandwidth_mhz=args.system_bandwidth_mhz
    )
    
    # Execute mode
    if args.mode == 'train':
        # Training requires a specific scenario, default to flash_crowd if not specified
        if config.scenario is None:
            config.scenario = "flash_crowd"
            
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
