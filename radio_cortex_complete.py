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
    python radio_cortex_complete.py --mode eval
"""

import argparse
import numpy as np
import torch
from pathlib import Path
import json
import time
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import multiprocessing
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, SpinnerColumn
from rich.live import Live
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.console import Group
from rich.columns import Columns

# Import all components
from oran_ns3_env import ORANns3Env, NS3Config, create_oran_env
from rl_training_pipeline import PPOTrainer, evaluate_policy

# Optional: Vectorized environment support
try:
    from vec_env_wrapper import make_vec_env, save_vec_normalize, load_vec_normalize
    VEC_ENV_AVAILABLE = True
except ImportError:
    VEC_ENV_AVAILABLE = False

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
    
    def get_model_params(self) -> int:
        """Return total trainable parameters (for Architecture Score)"""
        if self.policy is None: return 0
        return sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
    
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
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    checkpoint_interval: int = 5,
    n_envs: int = 1
):
    """
    Train Radio-Cortex agent
    
    Steps:
    1. Create O-RAN environment (single or vectorized)
    2. Initialize PPO trainer
    3. Train RL agent
    4. Save trained model
    """
    print("="*60)
    print("TRAINING RADIO-CORTEX")
    print(f"LR: {lr}, Gamma: {gamma}, Batch: {batch_size}, Device: {device}")
    if n_envs > 1:
        print(f"PARALLEL ENVS: {n_envs}")
    print("="*60)
    
    # Create environment (single or vectorized)
    vec_normalize_path = str(Path(save_path).parent / 'vec_normalize.pkl')
    
    if n_envs > 1:
        if not VEC_ENV_AVAILABLE:
            print("[WARNING] Vectorized env requested but stable-baselines3 not installed.")
            print("          Falling back to single environment. Install: pip install stable-baselines3")
            n_envs = 1
            env = create_oran_env(config)
            is_vec_env = False
        else:
            print(f"\n🚀 Creating {n_envs} parallel environments with VecNormalize...")
            env = make_vec_env(config, n_envs=n_envs, normalize_obs=True, normalize_reward=True)
            is_vec_env = True
    else:
        env = create_oran_env(config)
        is_vec_env = False
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
        device=device,
        checkpoint_interval=checkpoint_interval
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
    
    # Save VecNormalize stats if using vectorized env
    if is_vec_env and VEC_ENV_AVAILABLE:
        save_vec_normalize(env, vec_normalize_path)
        print(f"  ✔ VecNormalize stats saved to {vec_normalize_path}")
    
    # Close environments
    env.close()
    
    print(f"\n✓ Training complete. Model saved to {save_path}")
    
    return trainer


# ============================================================================
# Evaluation Pipeline
# ============================================================================

def evaluate_radio_cortex(
    config: NS3Config,
    model_path: str = 'models/radio_cortex.pt',
    n_envs: int = 1
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
    if config.scenario and config.scenario != 'all':
        if ',' in config.scenario:
            scenarios = [s.strip() for s in config.scenario.split(',')]
            print(f"Running multi-scenario evaluation: {scenarios}")
        elif config.scenario in all_scenarios:
            scenarios = [config.scenario]
            print(f"Running single scenario evaluation: {config.scenario}")
        else:
            print(f"Unknown scenario: {config.scenario}. Defaulting to all.")
            scenarios = all_scenarios
    else:
        scenarios = all_scenarios
        print(f"Running full benchmark on all {len(all_scenarios)} scenarios...")
    
    # Run evaluations
    all_results = {}
    
    if n_envs > 1 and len(scenarios) > 1:
        print(f"🚀 Running parallel evaluation with {n_envs} workers...")
        
        # Use multiprocessing Manager for shared Queue
        manager = multiprocessing.Manager()
        progress_queue = manager.Queue()
        
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            expand=True
        )
        
        panel_progress = Panel(progress, title="[bold blue]Radio-Cortex Parallel Evaluation[/]", border_style="blue", expand=True)
        
        # Live metrics storage: {task_key: {'tput': x, 'loss': y, ...}}
        live_metrics = {} 
        
        def create_metrics_table():
            table = Table(show_header=True, header_style="bold magenta", expand=True)
            table.add_column("Scenario", justify="left")
            table.add_column("Controller", justify="left")
            table.add_column("Tput (Mbps)", justify="right")
            table.add_column("Delay (ms)", justify="right")
            table.add_column("Loss (%)", justify="right")
            table.add_column("SINR (dB)", justify="right")
            table.add_column("RSRP (dBm)", justify="right")
            table.add_column("Queue", justify="right")
            table.add_column("RB Util", justify="right")
            table.add_column("Power (W)", justify="right")
            table.add_column("Reward", justify="right")
            
            # Sort by task key to keep order stable
            for task_key in sorted(live_metrics.keys()):
                m = live_metrics[task_key]
                # Parse task key (e.g. "0_rl", "1_baseline")
                try:
                    s_idx, c_type = task_key.split('_')
                    scenario = scenarios[int(s_idx)]
                    controller = "Radio-Cortex" if c_type == 'rl' else "Baseline"
                except:
                    scenario = task_key
                    controller = "?"
                
                table.add_row(
                    scenario,
                    controller,
                    f"{m.get('tput', 0):.2f}",
                    f"{m.get('delay', 0):.1f}",
                    f"{m.get('loss', 0)*100:.1f}",
                    f"{m.get('sinr', -10):.1f}", 
                    f"{m.get('rsrp', -140):.1f}",
                    f"{m.get('queue', 0):.1f}",
                    f"{m.get('rb', 0)*100:.1f}%",
                    f"{m.get('power', 0):.2f}",
                    f"[green]{m.get('reward', 0):.2f}[/green]" if m.get('reward', 0) > 0 else f"[red]{m.get('reward', 0):.2f}[/red]"
                )
            return Panel(table, title="[bold green]Live Network Metrics[/]", border_style="green", expand=True)

        def create_ue_metrics_grid():
            tables = []
            # Sort by task key to keep stable order
            for task_key in sorted(live_metrics.keys()):
                m = live_metrics[task_key]
                if 'ue_metrics' not in m:
                    continue
                
                # Parse task key
                try:
                    s_idx, c_type = task_key.split('_')
                    scenario = scenarios[int(s_idx)]
                    controller = "RC" if c_type == 'rl' else "Base"
                except:
                    scenario = task_key
                    controller = "?"
                
                # Create table for this scenario
                table = Table(title=f"{scenario} ({controller})", show_header=True, header_style="bold cyan", expand=True, box=None)
                table.add_column("UE", justify="right", style="cyan", width=4)
                table.add_column("Cell", justify="right", style="magenta", width=4)
                table.add_column("Tput", justify="right", style="green")
                table.add_column("Delay", justify="right", style="yellow")
                table.add_column("Loss", justify="right", style="red")
                table.add_column("SINR", justify="right", style="blue")

                ue_data = m['ue_metrics']
                # Limit to first 5 UEs to save space
                ue_ids = sorted(ue_data.keys())
                displayed_ues = ue_ids[:5]
                
                for ue_id in displayed_ues:
                    ud = ue_data[ue_id]
                    table.add_row(
                        str(ue_id),
                        str(ud['cell']),
                        f"{ud['tput']:.1f}",
                        f"{ud['delay']:.0f}",
                        f"{ud['loss']*100:.0f}%",
                        f"{ud['sinr']:.1f}"
                    )
                
                # Add a row indicating more UEs if truncated
                if len(ue_ids) > 5:
                    table.add_row("..", "..", "..", "..", "..", "..")
                
                tables.append(Panel(table, border_style="white", expand=True))
            
            if not tables:
                return Panel("Waiting for UE metrics...", style="dim")
            
            return Columns(tables, expand=True, equal=True)

        with Live(Group(panel_progress, create_metrics_table(), create_ue_metrics_grid()), refresh_per_second=4) as live:
            
            with ProcessPoolExecutor(max_workers=n_envs) as executor:
                futures = {}
                # Create bars for all scenarios (2 per scenario: Baseline and Radio-Cortex)
                task_ids = {}
                for s_idx, scenario_name in enumerate(scenarios):
                    # For simplicity, we create them as they start, or pre-create them?
                    # Let's pre-create placeholders
                    task_ids[s_idx] = {} # {s_idx: {controller: task_id}}
                
                for i, scenario_name in enumerate(scenarios):
                    worker_id = i % n_envs
                    future = executor.submit(
                        evaluate_single_scenario,
                        scenario_name, config, model_path, worker_id, progress_queue, i
                    )
                    futures[future] = scenario_name
                
                # Monitor progress queue
                active_tasks = len(futures)
                scenarios_completed = 0
                rich_tasks = {} # task_id_from_worker -> rich_task_id

                while scenarios_completed < len(scenarios):
                    # Try to get update from queue
                    try:
                        # Non-blocking check for updates
                        while not progress_queue.empty():
                            msg = progress_queue.get_nowait()
                            msg_type = msg[0]
                            
                            if msg_type == 'start':
                                # ('start', scenario_idx_controller, total, description)
                                _, task_key, total, desc = msg
                                rich_tasks[task_key] = progress.add_task(desc, total=total)
                            elif msg_type == 'update':
                                # ('update', task_key, advance, [metrics])
                                if len(msg) >= 4:
                                    _, task_key, advance, metrics = msg
                                    if task_key not in live_metrics:
                                        live_metrics[task_key] = metrics
                                    else:
                                        live_metrics[task_key].update(metrics)
                                else:
                                    _, task_key, advance = msg
                                
                                if task_key in rich_tasks:
                                    progress.update(rich_tasks[task_key], advance=advance)
                                
                                # Force refresh of the whole group (progress + table + grid)
                                live.update(Group(panel_progress, create_metrics_table(), create_ue_metrics_grid()))
                                
                            elif msg_type == 'complete':
                                # ('complete', task_key)
                                _, task_key = msg
                                if task_key in rich_tasks:
                                    progress.update(rich_tasks[task_key], completed=progress.tasks[rich_tasks[task_key]].total)
                                    # Optional: remove completed task to save space? 
                                    # No, keep them for final view.
                        
                        # Check for completed futures
                        for future in list(futures.keys()):
                            if future.done():
                                s_name = futures.pop(future)
                                try:
                                    _, s_results = future.result()
                                    all_results[s_name] = s_results
                                    scenarios_completed += 1
                                except Exception as e:
                                    print(f"\n❌ Evaluation failed for {s_name}: {e}")
                                    scenarios_completed += 1
                        
                        time.sleep(0.1)
                    except Exception as e:
                        # Queue might be empty or other IPC issues
                        pass
    else:
        # Sequential execution
        for scenario_name in scenarios:
            _, results = evaluate_single_scenario(scenario_name, config, model_path, 0)
            all_results[scenario_name] = results
    
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


def evaluate_single_scenario(
    scenario_name: str,
    base_config: NS3Config,
    model_path: str,
    worker_id: int = 0,
    progress_queue = None,
    scenario_idx: int = 0
) -> Tuple[str, Dict]:
    """
    Worker function to evaluate a single scenario.
    Can be run in parallel.
    """
    # Clone config to avoid side effects and set unique topic suffix
    config = copy.deepcopy(base_config)
    config.scenario = scenario_name
    
    # If using parallel workers, append suffix to avoid Kafka collision
    # e.g. _eval_1, _eval_2. 
    # worker_id=0 uses default (empty or base suffix) unless we force isolation
    if worker_id > 0:
        config.topic_suffix = f"{config.topic_suffix or ''}_eval_{worker_id}"
    
    # If running in parallel with progress queue, disable verbose output to keep UI clean
    if progress_queue:
        config.verbose = False
    
    if not progress_queue:
        print(f"\n{'='*70}")
        print(f"  SCENARIO: {scenario_name.upper().replace('_', ' ')} (Worker {worker_id})")
        print('='*70)
    
    results = {}
    
    # Initialize Baseline controller (static RAN, no AI)
    baseline = BaselineController(num_cells=config.num_cells)
    
    # Run evaluations
    evaluator = EvaluationRunner(
        num_ues=config.num_ues,
        num_cells=config.num_cells
    )
    
    # 1. Evaluate Baseline (Static RAN - No AI)
    env = create_oran_env(config)
    # Worker function to notify of start with full name
    controller_label = f"Baseline ({scenario_name})"
    env = create_oran_env(config)
    try:
        results['Baseline'] = evaluator.evaluate_controller(
            baseline, env, controller_label,
            progress_queue=progress_queue,
            task_id=f"{scenario_idx}_baseline"
        )
    finally:
        env.close()

    # 2. Evaluate Radio-Cortex (PPO RL Agent)
    controller_label = f"Radio-Cortex ({scenario_name})"
    from neural_networks import ActorCritic
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint['policy_state_dict']
        
        # Detect dimensions from state_dict
        stored_action_dim = state_dict['actor_mean.0.bias'].shape[0]
        stored_state_dim = state_dict['feature_net.0.weight'].shape[1]
        
        # Verify if it matches current config
        expected_state_dim = config.num_ues * 12 + config.num_cells * 5
        expected_action_dim = config.num_cells * 7 + config.num_ues
        
        if stored_action_dim != expected_action_dim or stored_state_dim != expected_state_dim:
            detected_ues = stored_action_dim - (config.num_cells * 7)
            # Force config to match model
            config.num_ues = detected_ues
            state_dim = stored_state_dim
            action_dim = stored_action_dim
        else:
            state_dim = expected_state_dim
            action_dim = expected_action_dim

        policy = ActorCritic(state_dim, action_dim).to('cpu')
        policy.load_state_dict(state_dict)
        
    except Exception as e:
        print(f"      [ERROR] Could not load model {model_path}: {e}")
        raise
        
    rc_agent = RadioCortexAgent(config.num_ues, config.num_cells, policy_model=policy)
    
    env = create_oran_env(config)
    try:
        results['Radio-Cortex'] = evaluator.evaluate_controller(
            rc_agent, env, controller_label,
            progress_queue=progress_queue,
            task_id=f"{scenario_idx}_rl"
        )
    finally:
        env.close()

    # Generate visualizations (Worker can do this efficiently)
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    VisualizationSuite.plot_comparison(
        results,
        scenario_name,
        save_path=str(results_dir / f"{scenario_name}_comparison.png")
    )
    VisualizationSuite.plot_radar(
        results,
        scenario_name,
        save_path=str(results_dir / f"{scenario_name}_radar.png")
    )
    
    # Generate LaTeX Table
    VisualizationSuite.generate_latex_table(
        {scenario_name: results},
        save_path=str(results_dir / f"{scenario_name}_metrics.tex")
    )

    # Generate CSV Report
    VisualizationSuite.generate_csv(
        {scenario_name: results},
        save_path=str(results_dir / f"{scenario_name}_metrics.csv")
    )
    
    return scenario_name, results



# ============================================================================
# Demo Mode
# ============================================================================



# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Radio-Cortex: Self-Healing O-RAN xApp')
    parser.add_argument(
        '--mode',
        choices=['train', 'eval'],
        default='train',
        help='Operation mode'
    )
    # Environment configs
    parser.add_argument('--num-ues', type=int, default=20, help='Number of UEs')
    parser.add_argument('--num-cells', type=int, default=3, help='Number of cells')
    parser.add_argument('--scenario', type=str, default=None, help='ns-3 Scenario (flash_crowd, mobility_storm, etc.). If omitted in eval mode, runs all scenarios.')
    parser.add_argument('--kpm-interval', type=int, default=100, help='KPM Reporting Interval (ms)')
    
    # Training configs
    parser.add_argument('--total-timesteps', type=int, default=100000, help='Training timesteps')
    parser.add_argument('--model-path', type=str, default='models/radio_cortex.pt', help='Model save path')
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for optimization')
    # Advanced PPO configs
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension for actor/critic networks')
    parser.add_argument('--gae-lambda', type=float, default=0.95, help='GAE lambda')
    parser.add_argument('--clip-epsilon', type=float, default=0.2, help='PPO clip epsilon')
    parser.add_argument('--vf-coef', type=float, default=0.5, help='Value function coefficient')
    parser.add_argument('--ent-coef', type=float, default=0.01, help='Entropy coefficient')
    parser.add_argument('--max-grad-norm', type=float, default=0.5, help='Max gradient norm')
    parser.add_argument('--rollout-steps', type=int, default=128, help='Steps per rollout')
    parser.add_argument('--log-interval', type=int, default=5, help='Logging interval (updates)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device (cpu/cuda)')
    parser.add_argument('--config', type=str, default=None, help='Path to JSON config file to override arguments')
    parser.add_argument('--system-bandwidth-mhz', type=float, default=10.0, help='System Bandwidth in MHz (5.0, 10.0, 20.0)')
    parser.add_argument('--sim-time', type=float, default=300.0, help='Simulation duration in seconds')
    
    # Parallel training configs
    parser.add_argument('--n-envs', type=int, default=4, help='Number of parallel environments (requires stable-baselines3)')
    parser.add_argument('--checkpoint-interval', type=int, default=5, help='Save checkpoint every N updates (0 to disable)')

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
        # Multi-scenario training: if --scenario all, rotate through all scenarios
        ALL_SCENARIOS = [
            "flash_crowd", "mobility_storm", "traffic_burst", "handover_ping_pong",
            "sleepy_campus", "ambulance", "adversarial", "commuter_rush",
            "mixed_reality", "urban_canyon", "iot_tsunami", "spectrum_crunch"
        ]
        
        if config.scenario == "all":
            # ----------------------------------------------------------------
            # MULTI-SCENARIO TRAINING (Domain Randomization)
            # ----------------------------------------------------------------
            # Providing "all" triggers the agent to cycle through ALL 12 available
            # scenarios. This improves generalization by preventing the agent
            # from overfitting to a single traffic pattern.
            #
            # The environment's reset() method will randomly select a new
            # scenario from this list at the start of each episode.
            config.scenarios = ALL_SCENARIOS
            config.scenario = ALL_SCENARIOS[0]  # Initial placeholder (randomized on reset)
            print(f"🎲 Multi-Scenario Training ENABLED: rotating through {len(ALL_SCENARIOS)} scenarios")
        elif config.scenario is None:
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
            device=args.device,
            checkpoint_interval=args.checkpoint_interval,
            n_envs=args.n_envs
        )
    
    elif args.mode == 'eval':
        results = evaluate_radio_cortex(
            config=config,
            model_path=args.model_path,
            n_envs=args.n_envs
        )


if __name__ == "__main__":
    main()
