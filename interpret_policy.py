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


def _load_last_log_entry(log_path: Path) -> Dict:
    with log_path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        raise RuntimeError(f"No entries found in {log_path}")
    return json.loads(lines[-1])


def _build_state_and_names(metrics: Dict, num_ues: int, num_cells: int) -> Tuple[np.ndarray, List[str]]:
    """Replicates ORANns3Env._extract_state normalization."""
    state: List[float] = []
    names: List[str] = []

    ue_metrics = metrics.get("ue", {})
    for ue_id in range(num_ues):
        ue = ue_metrics.get(str(ue_id), ue_metrics.get(ue_id, {}))
        state.extend([
            ue.get("throughput", 0.0) / 100.0,
            ue.get("delay", 0.0) / 1000.0,
            ue.get("packet_loss", 0.0),
            (ue.get("sinr", 0.0) + 10.0) / 40.0,
            (ue.get("rsrp", -140.0) + 140.0) / 100.0,
            (ue.get("rsrq", -20.0) + 20.0) / 20.0,
            ue.get("ul_rbs", 0.0) / 100.0,
            ue.get("rb_allocated", 0) / 100.0,
            ue.get("cqi", 0.0) / 15.0,
            ue.get("rsrp_var", 0.0) / 50.0,
            ue.get("rsrq_var", 0.0) / 50.0,
            ue.get("buffer_occupancy", 0.0) / 10000.0,
        ])
        names.extend([
            f"ue_{ue_id}.tput",
            f"ue_{ue_id}.delay",
            f"ue_{ue_id}.loss",
            f"ue_{ue_id}.sinr",
            f"ue_{ue_id}.rsrp",
            f"ue_{ue_id}.rsrq",
            f"ue_{ue_id}.ul_rbs",
            f"ue_{ue_id}.rb_alloc",
            f"ue_{ue_id}.cqi",
            f"ue_{ue_id}.rsrp_var",
            f"ue_{ue_id}.rsrq_var",
            f"ue_{ue_id}.buffer",
        ])

    cell_metrics = metrics.get("cell", {})
    for cell_id in range(num_cells):
        cell = cell_metrics.get(str(cell_id), cell_metrics.get(cell_id, {}))
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

    return np.array(state, dtype=np.float32), names


def _infer_dims_from_metrics(metrics: Dict) -> Tuple[int, int]:
    num_ues = len(metrics.get("ue", {}))
    num_cells = len(metrics.get("cell", {}))
    if num_ues == 0:
        num_ues = 20
    if num_cells == 0:
        num_cells = 3
    return num_ues, num_cells


def _load_policy(checkpoint_path: Path, state_dim: int, action_dim: int) -> ActorCritic:
    policy = ActorCritic(state_dim, action_dim).to("cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()
    return policy


def _saliency(policy: ActorCritic, state: np.ndarray, action_index: int) -> np.ndarray:
    state_t = torch.tensor(state, dtype=torch.float32, requires_grad=True).unsqueeze(0)
    action_mean, _ = policy(state_t)
    target = action_mean[0, action_index]
    target.backward()
    grad = state_t.grad[0].detach().numpy()
    return grad * state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/radio_cortex.pt")
    parser.add_argument("--log", default="action_logs.jsonl")
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
    action_dim = num_cells * 7 + num_ues  # Matches current ORANns3Env

    if args.action_index < 0 or args.action_index >= action_dim:
        print(f"Error: action-index must be in [0, {action_dim - 1}]")
        return

    if not ckpt_path.exists():
        print(f"Error: Checkpoint {ckpt_path} not found.")
        return

    try:
        policy = _load_policy(ckpt_path, state_dim, action_dim)
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
