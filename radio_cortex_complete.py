"""
Radio-Cortex: Complete Integration
RL-based Self-Healing O-RAN xApp

This file integrates all components:
1. ns-3 O-RAN environment (E2 interface)
2. RL training pipeline (PPO / BDH)
3. Congestion scenarios
4. Evaluation suite

Usage:
    python radio_cortex_complete.py --mode train --scenario flash_crowd
    python radio_cortex_complete.py --mode eval
"""

import os
# --- GPU MEMORY OPTIMIZATION ---
# Forces PyTorch to use a more memory-efficient allocation strategy
# to prevent OOM in the wide BDH model. MUST BE SET BEFORE IMPORTING TORCH.
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import numpy as np
import torch
from pathlib import Path
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import multiprocessing

# Set spawn start method for cleaner CUDA memory handling across processes
try:
    if multiprocessing.get_start_method(allow_none=True) is None:
        multiprocessing.set_start_method('spawn', force=True)
except RuntimeWarning:
    pass
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
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    VEC_ENV_AVAILABLE = True
except ImportError:
    VEC_ENV_AVAILABLE = False

from evaluation_baseline import (
    EvaluationRunner,
    BaselineController,
    ResultLogger
)


# ============================================================================
# Radio-Cortex Agent (RL Integration)
# ============================================================================

class RadioCortexAgent:
    """
    Radio-Cortex agent coordinating RL policies for RAN optimization.
    """
    
    def __init__(
        self,
        num_ues: int,
        num_cells: int,
        policy_model: Optional[torch.nn.Module] = None
    ):
        self.num_ues = num_ues
        self.num_cells = num_cells
        
        # RL policy
        self.policy = policy_model
        
        # Metrics tracking
        self.metrics_history = []
    
    def get_model_params(self) -> int:
        """Return total trainable parameters (for Architecture Score)"""
        if self.policy is None: return 0
        return sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
    
    def get_rl_action(self, base_state: np.ndarray) -> np.ndarray:
        """
        Inference section: Strategic RL decision
        """
        if self.policy is None:
            raise RuntimeError("RL Policy not loaded!")
        
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
    ent_coef: float = 0.03,
    max_grad_norm: float = 0.5,
    rollout_steps: int = 256,
    log_interval: int = 5,
    device: Optional[str] = None,
    checkpoint_interval: int = 5,
    n_envs: int = 12,
    model_type: str = 'bdh',
    num_epochs: int = 20,
    lr_scheduler_gamma: float = 0.999,
    target_kl: float = 0.05
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
    print(f"LR: {lr}, Gamma: {gamma}, Batch: {batch_size}")
    if n_envs > 1:
        print(f"PARALLEL ENVS: {n_envs}")
    print("="*60)
    
    # Create environment (single or vectorized)
    vec_normalize_path = str(Path(save_path).parent / f"vec_normalize_{Path(save_path).stem}.pkl")
    
    # Create environments (ALWAYS use VecEnv + VecNormalize for consistency)
    if not VEC_ENV_AVAILABLE:
        print("[WARNING] Stable-baselines3 not installed. Cannot use VecNormalize/VecEnv.")
        print("          Training may be unstable due to unscaled rewards! Install: pip install stable-baselines3")
        env = create_oran_env(config)
        is_vec_env = False
    else:
        # If n_envs=1, use DummyVecEnv for simpler debugging but still get normalization
        vec_env_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
        print(f"\n🚀 Creating {n_envs} environment(s) with {vec_env_cls.__name__} and VecNormalize...")
        
        # IMPORTANT: Create parallel envs BEFORE initializing CUDA to avoid fork issues
        env = make_vec_env(config, n_envs=n_envs, vec_env_cls=vec_env_cls, 
                          normalize_obs=True, normalize_reward=True)
        is_vec_env = True

    # Resolve device AFTER forking
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

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
        checkpoint_interval=checkpoint_interval,
        model_type=model_type,
        lr_scheduler_gamma=lr_scheduler_gamma,
        target_kl=target_kl,
        ns3_config=config  # explicit config passing
    )

    # --- ADDED: Resumption Logic ---
    if os.path.exists(save_path):
        print(f"  📦 Found existing model at {save_path}. Resuming training...")
        try:
            trainer.load(save_path)
        except (RuntimeError, ValueError) as e:
            print(f"  ⚠️ Model checkpoint incompatible (likely stale from old state/action space). Training from scratch: {e}")
            os.remove(save_path)
            print(f"  🗑️ Deleted stale {save_path}")
        # Also load normalization stats if they exist
        if is_vec_env and VEC_ENV_AVAILABLE and os.path.exists(vec_normalize_path):
            try:
                load_vec_normalize(env, vec_normalize_path)
            except (ValueError, RuntimeError) as e:
                print(f"  ⚠️ VecNormalize shape mismatch (likely stale from old action/state space). Skipping: {e}")
                os.remove(vec_normalize_path)
                print(f"  🗑️ Deleted stale {vec_normalize_path}")
    # ------------------------------
    
    # Train
    print(f"\nTraining for {total_timesteps} timesteps...")
    trainer.train(
        total_timesteps=total_timesteps,
        rollout_steps=rollout_steps,
        log_interval=log_interval,
        batch_size=batch_size,
        num_epochs=num_epochs
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

def _get_model_id(model_path: str) -> str:
    """Helper to get the identifier used for results/CSV logging"""
    if model_path == "base":
        return "Baseline"
        
    try:
        repo_root = Path(__file__).parent.absolute()
        abs_model = Path(model_path).absolute()
        if abs_model.is_relative_to(repo_root):
            return str(abs_model.relative_to(repo_root))
        return Path(model_path).name
    except Exception:
        return Path(model_path).name

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
    
    # --- SKIP REDUNDANT EVALUATIONS ---
    model_id = _get_model_id(model_path)
    completed_pairs = ResultLogger.get_completed_pairs() # returns List[Tuple(scenario, controller)]
    
    remaining_scenarios = []
    skipped_scenarios = []
    for s in scenarios:
        if (s, model_id) in completed_pairs:
            skipped_scenarios.append(s)
        else:
            remaining_scenarios.append(s)
            
    if skipped_scenarios:
        print(f"  [INFO] Skipping {len(skipped_scenarios)} scenarios already present in results: {', '.join(skipped_scenarios)}")
        
    if not remaining_scenarios:
        print(f"✅ All requested scenarios for {model_id} are already completed. Nothing to do.")
        return {}
        
    scenarios = remaining_scenarios
    
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
            
            # For heavy models like BDH, parallel evaluation can easily OOM a 16GB GPU.
            # Default to sequential or low-parallelism if requested environment count is high.
            effective_envs = n_envs

            with ProcessPoolExecutor(max_workers=effective_envs) as executor:
                futures = {}
                # Create bars for all scenarios (2 per scenario: Baseline and Radio-Cortex)
                task_ids = {}
                for s_idx, scenario_name in enumerate(scenarios):
                    # For simplicity, we create them as they start, or pre-create them?
                    # Let's pre-create placeholders
                    task_ids[s_idx] = {} # {s_idx: {controller: task_id}}
                
                for i, scenario_name in enumerate(scenarios):
                    # Rotate workers to reuse scenarios/topics
                    worker_id = i % effective_envs
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
                                    # Add placeholder to show in summary
                                    all_results[s_name] = {"CRASHED": None}
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
        if "CRASHED" in results:
            print("  [ERROR] Simulation crashed or failed to initialize.")
            continue
             
        print(f"{'Controller':<15} | {'Tput':<6} | {'Loss%':<6} | {'Satisf%':<7} | {'EnEff':<7} | {'Score':<5}")
        print("-" * 60)
        for controller, metrics in results.items():
            if metrics is None: continue
            avg_score = (metrics.qos_score + metrics.reliability_score + metrics.resource_score + 
                         metrics.buffer_score + metrics.phy_score + metrics.architecture_score) / 6.0
            print(f"{controller:<15} | {metrics.avg_throughput:>6.2f} | {metrics.avg_packet_loss*100:>6.2f} | {metrics.satisfied_user_ratio*100:>7.1f} | {metrics.energy_efficiency:>7.2f} | {avg_score:>5.1f}")
    
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
    
    # Build config dict for CSV logging
    eval_config = {
        'num_ues': config.num_ues,
        'num_cells': config.num_cells,
        'sim_time': config.sim_time,
        'kpm_interval_ms': config.kpm_interval_ms,
        'system_bandwidth_mhz': config.system_bandwidth_mhz,
        'model_type': getattr(config, 'model_type', 'unknown'),
        'model_path': model_path,
    }

    # Decoupled evaluation: run ONLY baseline or ONLY AI based on model_type
    max_retries = 3
    if model_path == "base":
        # ── Baseline ONLY ──
        success = False
        for attempt in range(max_retries):
            env = create_oran_env(config)
            controller_label = f"Baseline ({scenario_name})"
            try:
                results['Baseline'] = evaluator.evaluate_controller(
                    baseline, env, controller_label,
                    progress_queue=progress_queue,
                    task_id=f"{scenario_idx}_baseline"
                )
                success = True
                break
            except Exception as e:
                print(f"      [WARN] Baseline attempt {attempt+1}/{max_retries} failed for {scenario_name}: {e}")
                results['Baseline'] = None
            finally:
                env.close()
        
        if not success:
            print(f"      [ERROR] Baseline failed after {max_retries} attempts for {scenario_name}")
    else:
        # ── RL Model ONLY ──
        model_id = _get_model_id(model_path)
        controller_label = f"{model_id} ({scenario_name})"
        
        success = False
        for attempt in range(max_retries):
            env = None
            try:
                # Setup Policy and Agent for each attempt to ensure clean state
                checkpoint = torch.load(model_path, map_location='cpu')
                state_dict = checkpoint['policy_state_dict']
                model_type = getattr(config, 'model_type', 'bdh')
                
                # Determine dimensions
                stored_action_dim = 0
                stored_state_dim = 0
                try:
                    if 'actor_mean.0.bias' in state_dict:
                        stored_action_dim = state_dict['actor_mean.0.bias'].shape[0]
                    if 'feature_net.0.weight' in state_dict:
                         stored_state_dim = state_dict['feature_net.0.weight'].shape[1]
                except Exception:
                    pass
                
                expected_state_dim = config.num_cells * 16 * 3
                expected_action_dim = config.num_cells * 3
                
                eval_device = 'cpu'
                if getattr(config, 'force_cuda_eval', False):
                    eval_device = 'cuda'
                
                from policies import get_policy
                policy = get_policy(model_type, config.num_ues, config.num_cells, device=eval_device, env_config=config)
                policy.to(eval_device)
                policy.load_state_dict(state_dict, strict=False)
                
                rc_agent = RadioCortexAgent(config.num_ues, config.num_cells, policy_model=policy)

                # Check for VecNormalize stats
                vec_normalize_path = str(Path(model_path).parent / f"vec_normalize_{Path(model_path).stem}.pkl")
                if os.path.exists(vec_normalize_path) and VEC_ENV_AVAILABLE:
                    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
                    def make_env():
                        return create_oran_env(config)
                    env = DummyVecEnv([make_env])
                    env = VecNormalize.load(vec_normalize_path, env)
                    env.training = False
                    env.norm_reward = False
                else:
                    env = create_oran_env(config)

                results[model_id] = evaluator.evaluate_controller(
                    rc_agent, env, controller_label,
                    progress_queue=progress_queue,
                    task_id=f"{scenario_idx}_rl"
                )
                success = True
                break
            except Exception as e:
                print(f"      [WARN] RL attempt {attempt+1}/{max_retries} failed for {scenario_name}: {e}")
                results[model_id] = None
            finally:
                if env:
                    env.close()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        if not success:
            print(f"      [ERROR] RL Model failed after {max_retries} attempts for {scenario_name}")

    # Log results to central CSV (appending) ONLY if we have valid data
    valid_results = {k: v for k, v in results.items() if v is not None}
    if valid_results:
        ResultLogger.log_to_csv(
            valid_results,
            scenario_name,
            config=eval_config
        )
    else:
        results = {"CRASHED": True}
    
    return scenario_name, results

# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Radio-Cortex: RL-based RAN Optimization',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # --- Primary Group ---
    primary = parser.add_argument_group('Core Operation')
    primary.add_argument(
        '--mode',
        choices=['train', 'eval'],
        default='train',
        help='Operation mode'
    )
    primary.add_argument(
        '--scenario', 
        type=str, 
        default='flash_crowd', 
        help='ns-3 Scenario (e.g., flash_crowd, mobility_storm, traffic_burst). Use "all" to rotate through all scenarios.'
    )
    primary.add_argument('--total-timesteps', type=int, default=100000, help='Total training/eval steps')
    primary.add_argument('--model', type=str, default='bdh', 
                         choices=['bdh', 'nn', 'gpt2', 'trxl', 'linear', 'universal', 'reformer', 'base'],
                         help='Policy architecture: bdh (default), nn (MLP), gpt2, trxl, linear, universal, reformer, base')
    primary.add_argument('--n-envs', type=int, default=12, help='Number of parallel environments')
    primary.add_argument('--model-path', type=str, default=None, help='Path to save/load model (default: models/radiocortex_{model}.pt)')
    primary.add_argument('--device', type=str, default=None, help='Compute device (cpu/cuda)')
    primary.add_argument('--config', type=str, default=None, help='JSON config file to override any argument')
    primary.add_argument('--shuffle-ues', action='store_true', help='(Obsolete — ignored in cell-centric mode)')

    # --- Infrastructure Group ---
    infra = parser.add_argument_group('Network & Environment')
    infra.add_argument('--num-ues', type=int, default=20, help='Number of UEs')
    infra.add_argument('--num-cells', type=int, default=3, help='Number of cells')
    infra.add_argument('--sim-time', type=float, default=60.0, help='Simulation duration per episode (seconds)')
    infra.add_argument('--kpm-interval', type=int, default=100, help='KPM Reporting Interval (ms)')
    infra.add_argument('--system-bandwidth-mhz', type=float, default=10.0, help='System Bandwidth (5.0, 10.0, 20.0)')
    infra.add_argument('--topic-suffix', type=str, default="", help='Kafka topic suffix for parallel runs')

    # --- Hyperparameters Group ---
    hyper = parser.add_argument_group('Advanced PPO / RL Tuning')
    hyper.add_argument('--learning-rate', type=float, default=3e-5, help='PPO Learning rate')
    hyper.add_argument('--batch-size', type=int, default=512, help='Batch size for optimization updates')
    hyper.add_argument('--rollout-steps', type=int, default=512, help='Steps per rollout trajectory')
    hyper.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    hyper.add_argument('--hidden-dim', type=int, default=256, help='Network hidden dimension')
    hyper.add_argument('--gae-lambda', type=float, default=0.95, help='GAE normalization lambda')
    hyper.add_argument('--clip-epsilon', type=float, default=0.1, help='PPO clipping bound')
    hyper.add_argument('--vf-coef', type=float, default=0.5, help='Value function loss weight')
    hyper.add_argument('--ent-coef', type=float, default=0.03, help='Entropy regularization weight')
    hyper.add_argument('--max-grad-norm', type=float, default=0.5, help='Gradient clipping threshold')
    hyper.add_argument('--checkpoint-interval', type=int, default=10, help='Checkpoint frequency (updates)')
    hyper.add_argument('--log-interval', type=int, default=10, help='Console log frequency (updates)')
    hyper.add_argument('--ppo-epochs', type=int, default=20, help='PPO update epochs per batch')
    hyper.add_argument('--lr-gamma', type=float, default=0.99, help='Exponential LR decay gamma per update')
    hyper.add_argument('--target-kl', type=float, default=0.03, help='Target KL divergence for early stopping')

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
        system_bandwidth_mhz=args.system_bandwidth_mhz,
        topic_suffix=args.topic_suffix,
        # shuffle_ues is obsolete in cell-centric mode (no per-UE state)
    )
    
    # Attach model_type to config for downstream usage (eval workers, etc.)
    config.model_type = args.model
    
    # Set a shared run timestamp so ALL parallel envs write to the SAME CSV file.
    # Must be set before any env is spawned (datetime already imported at top).
    config.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Execute mode
    if args.mode == 'train':
        # Multi-scenario training support:
        # Handle weighted scenarios (e.g. "sc1:0.8,sc2:0.2") OR comma-separated list
        if ',' in args.scenario or ':' in args.scenario:
            if ':' in args.scenario:
                # Parse weights: "sc1:0.8,sc2:0.2" -> {"sc1": 0.8, "sc2": 0.2}
                pairs = args.scenario.split(',')
                scenarios_dict = {}
                for p in pairs:
                    if ':' in p:
                        name, weight = p.split(':')
                        scenarios_dict[name.strip()] = float(weight)
                    else:
                        scenarios_dict[p.strip()] = 1.0 # Default weight
                config.scenarios = scenarios_dict
                config.scenario = list(scenarios_dict.keys())[0]
            else:
                # Simple list: "sc1,sc2,sc3"
                config.scenarios = [s.strip() for s in args.scenario.split(',')]
                config.scenario = config.scenarios[0]
        elif args.scenario == 'all':
            # Rotate through all standard scenarios if "all" specified
            config.scenarios = [
                'flash_crowd', 'sleepy_campus', 'urban_canyon', 'mobility_storm',
                'traffic_burst', 'mixed_reality', 'adversarial', 'ping_pong',
                'commuter_rush', 'iot_tsunami', 'ambulance', 'spectrum_crunch'
            ]
            config.scenario = config.scenarios[0]
        else:
            config.scenario = args.scenario

        if args.model_path is None:
            # Generate unique filename with timestamp to prevent overwriting
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.model_path = f"models/radiocortex_{args.model}_{timestamp}.pt"
            print(f"Creating new model file: {args.model_path}")

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
            n_envs=args.n_envs,
            model_type=args.model,
            num_epochs=args.ppo_epochs,
            lr_scheduler_gamma=args.lr_gamma,
            target_kl=args.target_kl
        )
    
    elif args.mode == 'eval':
        # Handle 'base' model special case
        if args.model == 'base':
            args.model_path = "base"
        elif args.model_path is None:
            args.model_path = f"models/radiocortex_{args.model}.pt"

        results = evaluate_radio_cortex(
            config=config,
            model_path=args.model_path,
            n_envs=args.n_envs
        )


if __name__ == "__main__":
    main()
