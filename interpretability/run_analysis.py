"""
BDH Interpretability — CLI Runner
===================================

Loads a trained BDH checkpoint and runs all four analyses.

Usage (offline — no ns-3/Kafka needed):
    python -m interpretability.run_analysis \
        --checkpoint models/curriculum/stage_8.pt \
        --log action_logs.jsonl \
        --output ./bdh_results

Usage (with saved states):
    python -m interpretability.run_analysis \
        --checkpoint models/curriculum/stage_8.pt \
        --states saved_states.npz \
        --output ./bdh_results

Usage (compare Hebbian across checkpoints):
    python -m interpretability.run_analysis \
        --checkpoint models/curriculum/stage_8.pt \
        --log action_logs.jsonl \
        --hebbian-checkpoints models/curriculum/stage_1.pt models/curriculum/stage_8.pt \
        --output ./bdh_results
"""

import argparse
import json
import sys
import numpy as np
import torch
from pathlib import Path

# ── Add project root to path so we can import policies ──
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from policies.bdh_policy import BDHPolicy
from interpretability.bdh_interpretability_solo import run_full_analysis


# ═══════════════════════════════════════════════════════════════════
# Model Loading
# ═══════════════════════════════════════════════════════════════════

def load_bdh_policy(checkpoint_path: str, num_cells: int = 3) -> BDHPolicy:
    """
    Load a trained BDHPolicy from a .pt checkpoint.

    Matches the factory in policies/__init__.py:
        state_dim  = num_cells * 16 * 3   (Cell-Centric, 3-frame stacked)
        action_dim = num_cells * 3         (TxPower, CIO, TTT)
    """
    state_dim = num_cells * 16 * 3
    action_dim = num_cells * 3

    print(f"Loading BDH policy from: {checkpoint_path}")
    print(f"  state_dim={state_dim}, action_dim={action_dim}, num_cells={num_cells}")

    policy = BDHPolicy(state_dim, action_dim, device='cpu')

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'policy_state_dict' in checkpoint:
            state_dict = checkpoint['policy_state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    policy.load_state_dict(state_dict)
    policy.eval()

    # Print model info
    total_params = sum(p.numel() for p in policy.parameters())
    print(f"  ✓ Loaded ({total_params:,} parameters)")

    return policy


# ═══════════════════════════════════════════════════════════════════
# State Collection — From action_logs.jsonl
# ═══════════════════════════════════════════════════════════════════

def build_state_from_metrics(metrics: dict, num_cells: int = 3) -> np.ndarray:
    """
    Reconstruct a state vector from E2 KPM metrics.

    Mirrors the state extraction in oran_ns3_env.py:
    - 16 features per cell per frame
    - 3 frames stacked → 16 * num_cells * 3 = 144 total

    Since action_logs only give us ONE snapshot of metrics (not 3 frames),
    we replicate the same frame 3 times. This is an approximation —
    temporal dynamics won't be captured, but spatial features will be correct.
    """
    cell_metrics = metrics.get('e2_data', metrics).get('cell_metrics', {})
    ue_metrics = metrics.get('e2_data', metrics).get('ue_metrics', {})

    # Aggregate UE stats per cell
    cell_ue_stats = {}
    for c in range(num_cells):
        cell_ue_stats[c] = {
            'tputs': [], 'delays': [], 'losses': [],
            'sinrs': [], 'n_ues': 0,
        }

    for ue_id, ue in ue_metrics.items():
        if isinstance(ue, dict):
            c_id = ue.get('serving_cell', 0)
            if isinstance(c_id, str) and c_id.isdigit():
                c_id = int(c_id)
            if 0 <= c_id < num_cells:
                cell_ue_stats[c_id]['tputs'].append(ue.get('throughput', 0))
                cell_ue_stats[c_id]['delays'].append(ue.get('delay', 0))
                cell_ue_stats[c_id]['losses'].append(ue.get('packet_loss', 0))
                cell_ue_stats[c_id]['sinrs'].append(ue.get('sinr', 10))
                cell_ue_stats[c_id]['n_ues'] += 1

    # Build one frame (16 features per cell)
    single_frame = []

    for cell_id in range(num_cells):
        cell = cell_metrics.get(str(cell_id), cell_metrics.get(cell_id, {}))
        if not isinstance(cell, dict):
            cell = {}
        stats = cell_ue_stats[cell_id]
        n = stats['n_ues']

        # 5 native cell metrics (normalized)
        queue = cell.get('queue_length', 0) / 1000.0
        rb_util = cell.get('rb_utilization', 0)
        tx_power = (cell.get('tx_power', 23) - 10.0) / 36.0
        load = cell.get('cell_load', 0) / 50.0
        avg_req = cell.get('avg_rb_request', 0) / 100.0

        # 7 aggregated UE metrics
        if n > 0:
            avg_tput = np.mean(stats['tputs']) / 100.0
            avg_delay = np.mean(stats['delays']) / 100.0
            avg_loss = np.mean(stats['losses'])
            max_delay = np.max(stats['delays']) / 100.0
            max_loss = np.max(stats['losses'])
            tput_arr = np.array(stats['tputs'])
            sq_sum = np.sum(tput_arr ** 2)
            jains = float((tput_arr.sum() ** 2) / (n * sq_sum)) if sq_sum > 1e-9 else 1.0
            n_norm = n / 50.0
        else:
            avg_tput = avg_delay = avg_loss = 0.0
            max_delay = max_loss = 0.0
            jains = 1.0
            n_norm = 0.0

        # 4 additional features to reach 16 per cell
        sinr_mean = (np.mean(stats['sinrs']) / 30.0) if stats['sinrs'] else 0.0
        ho_count = 0.0       # Not available in single-step logs
        cqi_mean = 0.5       # Default mid-range
        stationary = 1.0     # Stationary reward flag

        single_frame.extend([
            queue, rb_util, tx_power, load, avg_req,
            avg_tput, avg_delay, avg_loss, max_delay, max_loss,
            jains, n_norm, sinr_mean, ho_count, cqi_mean, stationary,
        ])

    single_frame = np.array(single_frame, dtype=np.float32)

    # Stack 3 copies to simulate 3 temporal frames
    # (approximation — real env stacks t-2, t-1, t-0)
    state = np.tile(single_frame, 3)

    return state


def extract_e2_metrics(metrics: dict) -> dict:
    """Pull out the e2 metrics dict for concept extraction."""
    e2 = metrics.get('e2_data', {})
    if 'cell_metrics' in e2:
        return e2
    # Maybe metrics IS the e2 data
    if 'cell_metrics' in metrics:
        return metrics
    return {'cell_metrics': {}, 'ue_metrics': {}}


def load_from_action_logs(
    log_path: str,
    num_cells: int = 3,
    max_steps: int = 1000,
) -> tuple:
    """
    Load states and E2 metrics from action_logs.jsonl.

    Returns:
        states:          np.ndarray of shape (N, state_dim)
        e2_metrics_list: List of dicts for concept extraction
    """
    log_path = Path(log_path)
    if not log_path.exists():
        print(f"  ✗ Log file not found: {log_path}")
        sys.exit(1)

    print(f"Loading data from: {log_path}")

    states = []
    e2_list = []
    skipped = 0

    with open(log_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= max_steps:
                break

            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            metrics = entry.get('metrics', entry)
            state = build_state_from_metrics(metrics, num_cells)
            e2 = extract_e2_metrics(metrics)

            states.append(state)
            e2_list.append(e2)

    if not states:
        print("  ✗ No valid entries found in log file!")
        sys.exit(1)

    states_arr = np.array(states)
    print(f"  ✓ Loaded {len(states)} timesteps (skipped {skipped})")
    print(f"    State shape: {states_arr.shape}")

    return states_arr, e2_list


def load_from_npz(npz_path: str) -> tuple:
    """Load pre-saved states and metrics from .npz file."""
    data = np.load(npz_path, allow_pickle=True)
    states = data['states']
    e2_list = data['e2_metrics'].tolist() if 'e2_metrics' in data else [{} for _ in range(len(states))]
    print(f"  ✓ Loaded {len(states)} states from {npz_path}")
    return states, e2_list


# ═══════════════════════════════════════════════════════════════════
# Synthetic Data Fallback
# ═══════════════════════════════════════════════════════════════════

def generate_synthetic_data(
    policy: BDHPolicy,
    num_samples: int = 500,
    num_cells: int = 3,
) -> tuple:
    """
    Generate synthetic states + metrics for analysis when no logs exist.

    Creates random but realistic O-RAN states by sampling from
    plausible ranges for each feature. This is enough for structural
    analyses (sparsity, scale-free) — monosemanticity results will
    be less meaningful without real E2 metrics.
    """
    print(f"  Generating {num_samples} synthetic O-RAN states...")

    state_dim = num_cells * 16 * 3
    states = []
    e2_list = []

    for i in range(num_samples):
        # Random scenario weighting
        congested = np.random.random() < 0.3

        frame = []
        cell_metrics = {}
        ue_metrics = {}

        for c in range(num_cells):
            # Plausible normalized feature values
            queue = np.random.uniform(0, 0.9 if congested else 0.3)
            rb_util = np.random.uniform(0.5, 0.95) if congested else np.random.uniform(0.1, 0.6)
            tx_power = np.random.uniform(0.2, 0.8)
            load = np.random.uniform(0.3, 0.9) if congested else np.random.uniform(0.05, 0.4)
            avg_req = np.random.uniform(0.1, 0.7)
            avg_tput = np.random.uniform(0.01, 0.3) if congested else np.random.uniform(0.1, 0.8)
            avg_delay = np.random.uniform(0.3, 0.9) if congested else np.random.uniform(0.01, 0.2)
            avg_loss = np.random.uniform(0.05, 0.3) if congested else np.random.uniform(0, 0.02)
            max_delay = avg_delay + np.random.uniform(0, 0.3)
            max_loss = avg_loss + np.random.uniform(0, 0.1)
            jains = np.random.uniform(0.4, 0.7) if congested else np.random.uniform(0.7, 1.0)
            n_norm = np.random.uniform(0.2, 0.8)
            sinr_mean = np.random.uniform(0.1, 0.4) if congested else np.random.uniform(0.3, 0.8)
            ho_count = np.random.uniform(0, 0.3)
            cqi_mean = np.random.uniform(0.3, 0.7)
            stationary = 1.0

            frame.extend([
                queue, rb_util, tx_power, load, avg_req,
                avg_tput, avg_delay, avg_loss, max_delay, max_loss,
                jains, n_norm, sinr_mean, ho_count, cqi_mean, stationary,
            ])

            # Build fake E2 metrics for concept extraction
            cell_metrics[str(c)] = {
                'queue_length': queue * 1000,
                'rb_utilization': rb_util,
                'tx_power': tx_power * 36 + 10,
                'cell_load': load * 50,
            }

        # Stack 3 frames
        state = np.tile(np.array(frame, dtype=np.float32), 3)
        states.append(state)

        # Build UE metrics
        n_ues = int(np.random.randint(5, 25))
        for u in range(n_ues):
            ue_metrics[str(u)] = {
                'throughput': np.random.uniform(0.5, 20),
                'delay': np.random.uniform(5, 150 if congested else 30),
                'packet_loss': np.random.uniform(0, 0.2 if congested else 0.02),
                'sinr': np.random.uniform(2, 15 if congested else 25),
                'serving_cell': np.random.randint(0, num_cells),
            }

        e2_list.append({
            'cell_metrics': cell_metrics,
            'ue_metrics': ue_metrics,
        })

    states_arr = np.array(states)
    print(f"  ✓ Generated {num_samples} states, shape: {states_arr.shape}")
    return states_arr, e2_list


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='BDH Interpretability Analysis Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From action logs (most common):
  python -m interpretability.run_analysis --checkpoint models/curriculum/stage_8.pt --log action_logs.jsonl

  # With synthetic data (no logs needed):
  python -m interpretability.run_analysis --checkpoint models/curriculum/stage_8.pt --synthetic 500

  # Compare Hebbian across training stages:
  python -m interpretability.run_analysis --checkpoint models/curriculum/stage_8.pt --synthetic 300 \\
      --hebbian-checkpoints models/curriculum/stage_1.pt models/curriculum/stage_4.pt models/curriculum/stage_8.pt
        """
    )

    parser.add_argument('--checkpoint', required=True,
                        help='Path to trained BDH .pt checkpoint')
    parser.add_argument('--num-cells', type=int, default=3,
                        help='Number of cells (default: 3)')
    parser.add_argument('--output', default='./bdh_results',
                        help='Output directory (default: ./bdh_results)')

    # Data source (pick one)
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument('--log', default=None,
                            help='Path to action_logs.jsonl for offline analysis')
    data_group.add_argument('--states', default=None,
                            help='Path to saved .npz file with states')
    data_group.add_argument('--synthetic', type=int, default=None, metavar='N',
                            help='Generate N synthetic states (no logs needed)')

    parser.add_argument('--max-steps', type=int, default=1000,
                        help='Max timesteps to load from logs (default: 1000)')

    # Hebbian comparison
    parser.add_argument('--hebbian-checkpoints', nargs='+', default=None,
                        help='Multiple checkpoint paths for Hebbian cross-stage comparison')

    args = parser.parse_args()

    # ── Validate ──
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        print(f"✗ Checkpoint not found: {ckpt}")
        sys.exit(1)

    # ── Load Model ──
    print("\n" + "=" * 60)
    print("  BDH INTERPRETABILITY ANALYSIS")
    print("=" * 60)

    policy = load_bdh_policy(str(ckpt), num_cells=args.num_cells)

    # ── Load Data ──
    print("\n" + "-" * 60)
    print("  Loading Data")
    print("-" * 60)

    if args.log:
        states, e2_list = load_from_action_logs(
            args.log, num_cells=args.num_cells, max_steps=args.max_steps
        )
    elif args.states:
        states, e2_list = load_from_npz(args.states)
    elif args.synthetic:
        states, e2_list = generate_synthetic_data(
            policy, num_samples=args.synthetic, num_cells=args.num_cells
        )
    else:
        # Default: try action_logs.jsonl in current dir, else synthetic
        default_log = PROJECT_ROOT / 'action_logs.jsonl'
        if default_log.exists():
            print(f"  Found {default_log}, using it")
            states, e2_list = load_from_action_logs(
                str(default_log), num_cells=args.num_cells, max_steps=args.max_steps
            )
        else:
            print("  No data source specified and no action_logs.jsonl found")
            print("  → Generating 500 synthetic states for structural analysis")
            states, e2_list = generate_synthetic_data(
                policy, num_samples=500, num_cells=args.num_cells
            )

    # ── Run Analysis ──
    results = run_full_analysis(
        policy=policy,
        states=states,
        e2_metrics_list=e2_list,
        output_dir=args.output,
        checkpoint_paths=args.hebbian_checkpoints,
    )

    # ── Final Summary ──
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Monosemanticity:  {results['monosemanticity']['score']:.3f}")
    print(f"  Sparsity:         {results['sparsity']['overall_sparsity']*100:.1f}% inactive")
    print(f"  Scale-Free:       {'YES' if results['scale_free']['is_scale_free'] else 'NO'}", end="")
    if results['scale_free'].get('alpha'):
        print(f"  (α={results['scale_free']['alpha']:.2f})")
    else:
        print()
    print(f"  Hub Neurons:      {results['scale_free']['num_hubs']}")
    print(f"  Hebbian Changes:  {results['hebbian']['strengthened_count']} strengthened synapses")
    print()
    print(f"  Output: {args.output}/")
    print()
    print("  Next steps:")
    print(f"    1. python -m interpretability.visualize --results {args.output}")
    print(f"    2. Open ui/index.html → 🔬 Interpretability tab")
    print("=" * 60)


if __name__ == '__main__':
    main()