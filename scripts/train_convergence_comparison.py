#!/usr/bin/env python3
"""
Radio-Cortex: Reward Convergence Comparison
============================================
Trains all 7 model architectures on the same scenario and plots
reward convergence curves for direct comparison.

Output:
  train_results/convergence_<scenario>.png   — Reward vs Update plot
  train_results/<model>_training_log.jsonl   — Raw per-update metrics

Usage:
  python3 scripts/train_convergence_comparison.py                          # default: flash_crowd
  python3 scripts/train_convergence_comparison.py --scenario ambulance     # specific scenario
  python3 scripts/train_convergence_comparison.py --timesteps 50000        # longer training
  python3 scripts/train_convergence_comparison.py --models bdh gpt2 trxl   # subset of models
"""

import os
import sys
import json
import shutil
import subprocess
import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# ── Configuration ──────────────────────────────────────────────────────
ALL_MODELS = ['bdh', 'nn', 'gpt2', 'trxl', 'linear', 'universal', 'reformer']

MODEL_DISPLAY = {
    'bdh':       ('BDH (Scale-Free)',       '#FF6B35', '-',  2.5),
    'nn':        ('MLP (Feedforward)',       '#888888', '--', 1.5),
    'gpt2':      ('GPT-2 (Causal)',          '#4ECDC4', '--', 1.5),
    'trxl':      ('TrXL (Recurrent)',        '#45B7D1', '--', 1.5),
    'linear':    ('Linear TF (Kernel)',      '#96CEB4', '--', 1.5),
    'universal':  ('Universal TF (Shared)',  '#DDA0DD', '--', 1.5),
    'reformer':  ('Reformer (LSH)',          '#FFEAA7', '--', 1.5),
}

RESULTS_DIR = 'train_results'

def train_model(model: str, scenario: str, timesteps: int, n_envs: int = 4) -> dict:
    """Train one model, return the path to its training_log.jsonl."""
    model_path = f'{RESULTS_DIR}/models/{model}_{scenario}.pt'
    log_dir = f'logs_{model}'  # Separate log dir per model

    # Clean previous logs for this model
    if os.path.exists(log_dir):
        training_log = os.path.join(log_dir, 'training_log.jsonl')
        if os.path.exists(training_log):
            os.remove(training_log)

    cmd = [
        sys.executable, 'radio_cortex_complete.py',
        '--mode', 'train',
        '--model', model,
        '--scenario', scenario,
        '--total-timesteps', str(timesteps),
        '--model-path', model_path,
        '--n-envs', str(n_envs),
        '--sim-time', '30.0',
    ]

    print(f'\n{"="*70}')
    print(f'  Training {MODEL_DISPLAY[model][0]}')
    print(f'  Scenario: {scenario} | Steps: {timesteps} | Envs: {n_envs}')
    print(f'{"="*70}')

    start = time.time()
    # Set RADIO_CORTEX_LOG_DIR to redirect logs to model-specific dir
    env = os.environ.copy()
    env['RADIO_CORTEX_LOG_DIR'] = log_dir
    result = subprocess.run(cmd, env=env)
    wall_time = time.time() - start

    training_log_path = os.path.join(log_dir, 'training_log.jsonl')

    # Fallback: check default logs/ dir
    if not os.path.exists(training_log_path):
        training_log_path = 'logs/training_log.jsonl'

    # Copy the training log to train_results/ for archival
    dest = os.path.join(RESULTS_DIR, f'{model}_training_log.jsonl')
    if os.path.exists(training_log_path):
        shutil.copy2(training_log_path, dest)
    else:
        print(f'  ⚠️  No training log found for {model} at {training_log_path}')
        dest = None

    return {
        'model': model,
        'wall_time': wall_time,
        'log_path': dest,
        'returncode': result.returncode,
    }


def parse_training_log(log_path: str) -> dict:
    """Parse a training_log.jsonl and return lists of (update, reward, steps)."""
    updates, rewards, steps = [], [], []
    policy_losses, value_losses, entropies = [], [], []

    if not log_path or not os.path.exists(log_path):
        return {'updates': [], 'rewards': [], 'steps': [],
                'policy_losses': [], 'value_losses': [], 'entropies': []}

    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                updates.append(entry.get('update', 0))
                rewards.append(entry.get('reward', 0.0))
                steps.append(entry.get('steps', 0))
                policy_losses.append(entry.get('policy_loss', 0.0))
                value_losses.append(entry.get('value_loss', 0.0))
                entropies.append(entry.get('entropy', 0.0))
            except json.JSONDecodeError:
                continue

    return {
        'updates': updates,
        'rewards': rewards,
        'steps': steps,
        'policy_losses': policy_losses,
        'value_losses': value_losses,
        'entropies': entropies,
    }


def smooth(values, window=5):
    """Simple moving average smoothing."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode='valid').tolist()


def plot_convergence(all_data: dict, scenario: str, save_path: str):
    """
    Generate a 2x2 convergence comparison plot:
      - Reward vs Update (smoothed)
      - Reward vs Update (raw)
      - Policy Loss vs Update
      - Value Loss vs Update
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Radio-Cortex: Model Convergence Comparison\nScenario: {scenario}',
                 fontsize=16, fontweight='bold', y=0.98)

    # ── Reward (Smoothed) ──
    ax = axes[0][0]
    for model, data in all_data.items():
        if not data['rewards']:
            continue
        label, color, ls, lw = MODEL_DISPLAY[model]
        smoothed = smooth(data['rewards'], window=5)
        x = data['updates'][:len(smoothed)]
        ax.plot(x, smoothed, label=label, color=color, linestyle=ls, linewidth=lw)
    ax.set_xlabel('Update')
    ax.set_ylabel('Avg Reward (Smoothed)')
    ax.set_title('Reward Convergence (5-pt Moving Average)')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)

    # ── Reward (Raw) ──
    ax = axes[0][1]
    for model, data in all_data.items():
        if not data['rewards']:
            continue
        label, color, ls, lw = MODEL_DISPLAY[model]
        ax.plot(data['updates'], data['rewards'], label=label, color=color,
                linestyle=ls, linewidth=lw, alpha=0.6)
    ax.set_xlabel('Update')
    ax.set_ylabel('Avg Reward (Raw)')
    ax.set_title('Reward Convergence (Raw)')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)

    # ── Policy Loss ──
    ax = axes[1][0]
    for model, data in all_data.items():
        if not data['policy_losses']:
            continue
        label, color, ls, lw = MODEL_DISPLAY[model]
        smoothed = smooth(data['policy_losses'], window=5)
        x = data['updates'][:len(smoothed)]
        ax.plot(x, smoothed, label=label, color=color, linestyle=ls, linewidth=lw)
    ax.set_xlabel('Update')
    ax.set_ylabel('Policy Loss')
    ax.set_title('Policy Loss (Smoothed)')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

    # ── Value Loss ──
    ax = axes[1][1]
    for model, data in all_data.items():
        if not data['value_losses']:
            continue
        label, color, ls, lw = MODEL_DISPLAY[model]
        smoothed = smooth(data['value_losses'], window=5)
        x = data['updates'][:len(smoothed)]
        ax.plot(x, smoothed, label=label, color=color, linestyle=ls, linewidth=lw)
    ax.set_xlabel('Update')
    ax.set_ylabel('Value Loss')
    ax.set_title('Value Loss (Smoothed)')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\n📊 Convergence plot saved to: {save_path}')


def plot_wall_time(results: list, scenario: str, save_path: str):
    """Bar chart comparing training wall-clock times per model."""
    models = [r['model'] for r in results]
    times = [r['wall_time'] / 60.0 for r in results]  # minutes
    colors = [MODEL_DISPLAY[m][1] for m in models]
    labels = [MODEL_DISPLAY[m][0] for m in models]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, times, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_ylabel('Wall-Clock Time (minutes)')
    ax.set_title(f'Training Time Comparison — {scenario}')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                f'{t:.1f}m', ha='center', va='bottom', fontsize=9)

    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'⏱️  Wall-time plot saved to: {save_path}')


def main():
    parser = argparse.ArgumentParser(description='Radio-Cortex: Model Convergence Comparison')
    parser.add_argument('--scenario', default='flash_crowd', help='Training scenario')
    parser.add_argument('--timesteps', type=int, default=20000, help='Total timesteps per model')
    parser.add_argument('--n-envs', type=int, default=4, help='Number of parallel envs')
    parser.add_argument('--models', nargs='+', default=ALL_MODELS,
                        choices=ALL_MODELS, help='Models to compare')
    parser.add_argument('--plot-only', action='store_true',
                        help='Skip training, just regenerate plots from existing logs')
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(f'{RESULTS_DIR}/models', exist_ok=True)

    results = []

    if not args.plot_only:
        for model in args.models:
            result = train_model(model, args.scenario, args.timesteps, args.n_envs)
            results.append(result)
            print(f'  ✅ {model.upper()} done in {result["wall_time"]/60:.1f} min '
                  f'(exit={result["returncode"]})')

    # ── Parse all training logs ──
    all_data = {}
    for model in args.models:
        log_path = os.path.join(RESULTS_DIR, f'{model}_training_log.jsonl')
        all_data[model] = parse_training_log(log_path)
        n = len(all_data[model]['rewards'])
        print(f'  📄 {model}: {n} update entries parsed')

    # ── Generate Plots ──
    convergence_path = os.path.join(RESULTS_DIR, f'convergence_{args.scenario}.png')
    plot_convergence(all_data, args.scenario, convergence_path)

    if results:
        walltime_path = os.path.join(RESULTS_DIR, f'wall_time_{args.scenario}.png')
        plot_wall_time(results, args.scenario, walltime_path)

    # ── Summary Table ──
    print(f'\n{"="*60}')
    print(f'  CONVERGENCE COMPARISON COMPLETE')
    print(f'  Scenario: {args.scenario} | Timesteps: {args.timesteps}')
    print(f'{"="*60}')
    for model in args.models:
        d = all_data[model]
        if d['rewards']:
            final_reward = d['rewards'][-1]
            best_reward = max(d['rewards'])
            print(f'  {MODEL_DISPLAY[model][0]:30s} | Final: {final_reward:+.4f} | Best: {best_reward:+.4f}')
        else:
            print(f'  {MODEL_DISPLAY[model][0]:30s} | No data')
    print(f'\n  Plots: {RESULTS_DIR}/')


if __name__ == '__main__':
    main()
