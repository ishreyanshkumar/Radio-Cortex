"""
Interpretability for Radio-Cortex policy.

Computes saliency (gradient * input) for the policy's action mean
with respect to the input state vector. Uses the latest entry in
action_logs.jsonl by default.

Usage:
  python interpret_policy.py \
      --checkpoint models/radio_cortex.pt \
      --log action_logs.jsonl \
      --action-index 0 \
      --top-k 10
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from policies.neural_networks import ActorCritic
from policies.bdh_policy import BDHPolicy


def _load_last_log_entry(log_path: Path) -> Dict:
    with log_path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        raise RuntimeError(f"No entries found in {log_path}")
    return json.loads(lines[-1])


def _build_state_and_names(metrics: Dict, num_ues: int, num_cells: int) -> Tuple[np.ndarray, List[str]]:
    """Replicates ORANns3Env._extract_state normalization (Cell-Centric)."""
    state: List[float] = []
    names: List[str] = []

    cell_metrics = metrics.get("cell", {})
    ue_metrics = metrics.get("ue", {})
    
    # Aggregate UE metrics per cell (mirrors _extract_state)
    cell_stats = {c: {'tputs': [], 'delays': [], 'losses': []} for c in range(num_cells)}
    for ue_id, ue in ue_metrics.items():
        c_id = ue.get('serving_cell', int(ue_id) % num_cells if str(ue_id).isdigit() else 0)
        if 0 <= c_id < num_cells:
            cell_stats[c_id]['tputs'].append(ue.get('throughput', 0.0))
            cell_stats[c_id]['delays'].append(ue.get('delay', 0.0))
            cell_stats[c_id]['losses'].append(ue.get('packet_loss', 0.0))

    for cell_id in range(num_cells):
        cell = cell_metrics.get(str(cell_id), cell_metrics.get(cell_id, {}))
        stats = cell_stats[cell_id]
        n = len(stats['tputs'])
        
        # Native cell metrics (5)
        state.extend([
            cell.get("queue_length", 0.0) / 1000.0,
            cell.get("rb_utilization", 0.0),
            (cell.get("tx_power", 23.0) - 10.0) / 36.0,
            cell.get("cell_load", 0.0) / max(1.0, num_ues),
            cell.get("avg_rb_request", 0.0) / 100.0,
        ])
        names.extend([
            f"cell_{cell_id}.queue",
            f"cell_{cell_id}.rb_util",
            f"cell_{cell_id}.tx_power",
            f"cell_{cell_id}.load",
            f"cell_{cell_id}.avg_req",
        ])
        
        # Aggregated UE metrics (7)
        if n > 0:
            avg_tput = np.mean(stats['tputs'])
            avg_delay = np.mean(stats['delays'])
            avg_loss = np.mean(stats['losses'])
            max_delay = np.max(stats['delays'])
            max_loss = np.max(stats['losses'])
            sum_t = sum(stats['tputs'])
            sum_sq = sum(x*x for x in stats['tputs'])
            jains = 1.0 if sum_sq < 1e-9 else (sum_t**2) / (n * sum_sq)
        else:
            avg_tput, avg_delay, avg_loss = 0, 0, 0
            max_delay, max_loss = 0, 0
            jains = 1.0
        
        state.extend([
            avg_tput / 100.0,
            avg_delay / 100.0,
            avg_loss,
            max_delay / 100.0,
            max_loss,
            jains,
            n / 50.0,
        ])
        names.extend([
            f"cell_{cell_id}.avg_tput",
            f"cell_{cell_id}.avg_delay",
            f"cell_{cell_id}.avg_loss",
            f"cell_{cell_id}.max_delay",
            f"cell_{cell_id}.max_loss",
            f"cell_{cell_id}.jains",
            f"cell_{cell_id}.n_ues",
        ])

    return np.array(state, dtype=np.float32), names


def _infer_dims_from_metrics(metrics: Dict) -> Tuple[int, int]:
    num_ues = len(metrics.get("ue", {}))
    num_cells = len(metrics.get("cell", {}))
    if num_ues == 0:
        num_ues = 20
    if num_cells == 0:
        num_cells = 3
    return num_ues, num_cells


def _load_policy(checkpoint_path: Path, state_dim: int, action_dim: int, model_type: str = 'nn') -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    if model_type == 'bdh':
        policy = BDHPolicy(state_dim, action_dim, device="cpu")
    else:
        policy = ActorCritic(state_dim, action_dim).to("cpu")
        
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()
    return policy


def _saliency(policy: ActorCritic, state: np.ndarray, action_index: int) -> np.ndarray:
    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    state_t.requires_grad_(True)
    state_t.retain_grad()
    action_mean, _, _ = policy(state_t)
    target = action_mean[0, action_index]
    target.backward()
    grad = state_t.grad[0].detach().numpy()
    return grad * state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/radio_cortex.pt")
    parser.add_argument("--log", default="action_logs.jsonl")
    parser.add_argument("--model", type=str, default=None, help="Model type (bdh, nn). Auto-detected if possible.")
    parser.add_argument("--action-index", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    log_path = Path(args.log)
    ckpt_path = Path(args.checkpoint)

    if not log_path.exists():
        print(f"Error: Log file {log_path} not found. Run training first.")
        return

    try:
        entry = _load_last_log_entry(log_path)
        metrics = entry.get("metrics", {})
    except Exception as e:
        print(f"Error reading log: {e}")
        return

    num_ues, num_cells = _infer_dims_from_metrics(metrics)
    state, names = _build_state_and_names(metrics, num_ues, num_cells)

    state_dim = len(state)
    action_dim = num_cells * 5  # Cell-centric: 5 actions per cell

    if args.action_index < 0 or args.action_index >= action_dim:
        print(f"Error: action-index must be in [0, {action_dim - 1}]")
        return

    if not ckpt_path.exists():
        print(f"Error: Checkpoint {ckpt_path} not found.")
        return

    # Detect model type
    model_type = args.model
    if model_type is None:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        # Heuristic detection from state_dict
        if any("bdh." in k for k in checkpoint["policy_state_dict"].keys()):
            model_type = "bdh"
        else:
            model_type = "nn"
        print(f"[INFO] Auto-detected model type: {model_type}")

    try:
        policy = _load_policy(ckpt_path, state_dim, action_dim, model_type=model_type)
    except Exception as e:
        print(f"Error loading policy: {e}")
        print("Dimension mismatch likely. Ensure num-ues/num-cells match the trained model.")
        return

    sal = _saliency(policy, state, args.action_index)

    top_k = min(args.top_k, len(sal))
    idx = np.argsort(np.abs(sal))[::-1][:top_k]

    print("=== Policy Saliency (gradient * input) ===")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Action index: {args.action_index}")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print("Top features:")
    for i in idx:
        print(f"  {names[i]:>20s}  value={state[i]: .4f}  saliency={sal[i]: .6f}")


if __name__ == "__main__":
    main()
