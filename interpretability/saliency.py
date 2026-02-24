"""
BDH Saliency Analysis
=======================

Gradient × Input feature attribution for BDH policy decisions.

Breaks down saliency into:
- Per-feature importance  (which of the 16 cell features matter most?)
- Per-cell importance     (which cell drives each action?)
- Temporal importance     (which of the 3 stacked frames matters?)
- Per-action maps         (what drives TxPower vs CIO vs TTT?)

Designed for the Cell-Centric state space:
    State = num_cells × 16 features × 3 temporal frames

Author: Radio-Cortex Team / KRITI 2026
"""

import torch
import numpy as np
from typing import Dict, List, Optional
import json
from pathlib import Path


# ── Feature names matching oran_ns3_env.py state extraction ──
CELL_FEATURE_NAMES = [
    'queue',        # Queue length (normalized)
    'rb_util',      # RB utilization
    'tx_power',     # Transmit power (normalized)
    'load',         # Cell load
    'avg_req',      # Average RB request
    'avg_tput',     # Average throughput (UEs)
    'avg_delay',    # Average delay (UEs)
    'avg_loss',     # Average packet loss (UEs)
    'max_delay',    # Max delay (UEs)
    'max_loss',     # Max packet loss (UEs)
    'jains',        # Jain's fairness index
    'n_ues',        # Number of connected UEs
    'sinr_mean',    # Mean SINR
    'ho_count',     # Handover count
    'cqi_mean',     # Mean CQI
    'stationary',   # Stationary flag
]

ACTION_NAMES = ['TxPower', 'CIO', 'TTT']

FRAME_LABELS = ['t-2 (oldest)', 't-1 (previous)', 't-0 (current)']


class BDHSaliency:
    """
    Comprehensive saliency analysis for BDH policy.

    For each action dimension, computes gradient × input and then
    reshapes into the (frames, cells, features) structure so we can
    answer questions like:
    - "Which features drive the TxPower decision for Cell 0?"
    - "Does the agent use historical frames or only the current one?"
    - "Which cell's metrics matter for Cell 2's CIO action?"
    """

    def __init__(self, policy, num_cells: int = 3):
        """
        Args:
            policy:    Trained BDHPolicy (must support .forward(state) → (mean, logstd, value))
            num_cells: Number of cells in the environment
        """
        self.policy = policy
        self.policy.eval()
        self.num_cells = num_cells
        self.num_features = 16       # Features per cell per frame
        self.num_frames = 3          # Temporal frame stacking
        self.state_dim = num_cells * self.num_features * self.num_frames
        self.action_dim = num_cells * 3  # TxPower, CIO, TTT per cell

    def _compute_raw_saliency(self, state: np.ndarray, action_index: int) -> np.ndarray:
        """
        Compute gradient × input for a single state and action.

        Returns:
            saliency vector of shape (state_dim,)
        """
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        state_t.requires_grad_(True)

        action_mean, _, _ = self.policy(state_t)
        target = action_mean[0, action_index]
        target.backward()

        grad = state_t.grad[0].detach().numpy()
        return grad * state

    def analyze_single_state(self, state: np.ndarray) -> Dict:
        """
        Full saliency analysis for one state vector.

        Returns a dict with per-action breakdowns, feature rankings,
        temporal attribution, and cell attribution.
        """
        assert len(state) == self.state_dim, \
            f"Expected state_dim={self.state_dim}, got {len(state)}"

        per_action = {}

        for a_idx in range(self.action_dim):
            saliency = self._compute_raw_saliency(state, a_idx)

            # ── Reshape to (frames, cells, features) ──
            sal_3d = saliency.reshape(self.num_frames, self.num_cells, self.num_features)
            abs_3d = np.abs(sal_3d)

            # ── Feature importance (average across frames and cells) ──
            feature_imp = abs_3d.mean(axis=(0, 1))  # (16,)
            feature_total = feature_imp.sum()
            if feature_total > 0:
                feature_imp_norm = feature_imp / feature_total
            else:
                feature_imp_norm = feature_imp

            # ── Temporal importance (which frame matters?) ──
            frame_imp = abs_3d.sum(axis=(1, 2))  # (3,)
            frame_total = frame_imp.sum()
            if frame_total > 0:
                frame_imp_norm = frame_imp / frame_total
            else:
                frame_imp_norm = frame_imp

            # ── Cell importance (which cell drives this action?) ──
            cell_imp = abs_3d.sum(axis=(0, 2))  # (num_cells,)
            cell_total = cell_imp.sum()
            if cell_total > 0:
                cell_imp_norm = cell_imp / cell_total
            else:
                cell_imp_norm = cell_imp

            # ── Top-10 individual features ──
            flat_abs = np.abs(saliency)
            top_indices = np.argsort(flat_abs)[::-1][:10]
            top_features = []
            for idx in top_indices:
                frame = idx // (self.num_cells * self.num_features)
                rem = idx % (self.num_cells * self.num_features)
                cell = rem // self.num_features
                feat = rem % self.num_features

                feat_name = CELL_FEATURE_NAMES[feat] if feat < len(CELL_FEATURE_NAMES) else f'feat_{feat}'

                top_features.append({
                    'name': f"{FRAME_LABELS[frame]}_cell{cell}_{feat_name}",
                    'frame': int(frame),
                    'cell': int(cell),
                    'feature': feat_name,
                    'saliency': float(saliency[idx]),
                    'abs_saliency': float(flat_abs[idx]),
                    'state_value': float(state[idx]),
                })

            # ── Label this action ──
            cell_id = a_idx // 3
            action_name = ACTION_NAMES[a_idx % 3]
            label = f"cell{cell_id}_{action_name}"

            per_action[label] = {
                'action_index': int(a_idx),
                'cell_id': int(cell_id),
                'action_name': action_name,
                'feature_importance': {
                    CELL_FEATURE_NAMES[i]: float(feature_imp_norm[i])
                    for i in range(min(len(CELL_FEATURE_NAMES), len(feature_imp_norm)))
                },
                'frame_importance': [float(x) for x in frame_imp_norm],
                'cell_importance': [float(x) for x in cell_imp_norm],
                'top_features': top_features,
                'total_saliency': float(flat_abs.sum()),
            }

        return per_action

    def analyze_batch(
        self,
        states: np.ndarray,
        max_samples: int = 100,
    ) -> Dict:
        """
        Run saliency over multiple states and compute averaged importance.

        Args:
            states:      Array of shape (N, state_dim)
            max_samples: Cap to avoid slow runtimes

        Returns:
            Aggregated results with averaged feature/frame/cell importance
        """
        n = min(len(states), max_samples)
        print(f"    Running saliency on {n} states × {self.action_dim} actions...")

        # Accumulators per action
        accum = {}
        for a_idx in range(self.action_dim):
            cell_id = a_idx // 3
            action_name = ACTION_NAMES[a_idx % 3]
            label = f"cell{cell_id}_{action_name}"
            accum[label] = {
                'feature_imp': np.zeros(self.num_features),
                'frame_imp': np.zeros(self.num_frames),
                'cell_imp': np.zeros(self.num_cells),
                'count': 0,
            }

        # Also accumulate global averages
        global_feature_imp = np.zeros(self.num_features)
        global_frame_imp = np.zeros(self.num_frames)
        global_cell_imp = np.zeros(self.num_cells)
        global_count = 0

        for i in range(n):
            if (i + 1) % 25 == 0:
                print(f"      {i+1}/{n}...")

            per_action = self.analyze_single_state(states[i])

            for label, data in per_action.items():
                a = accum[label]
                feat_vals = [data['feature_importance'].get(fn, 0)
                             for fn in CELL_FEATURE_NAMES]
                a['feature_imp'] += np.array(feat_vals)
                a['frame_imp'] += np.array(data['frame_importance'])
                a['cell_imp'] += np.array(data['cell_importance'])
                a['count'] += 1

                global_feature_imp += np.array(feat_vals)
                global_frame_imp += np.array(data['frame_importance'])
                global_cell_imp += np.array(data['cell_importance'])
                global_count += 1

        # ── Build results ──
        per_action_results = {}
        for label, a in accum.items():
            c = max(a['count'], 1)
            per_action_results[label] = {
                'feature_importance': {
                    CELL_FEATURE_NAMES[j]: float(a['feature_imp'][j] / c)
                    for j in range(self.num_features)
                },
                'frame_importance': (a['frame_imp'] / c).tolist(),
                'cell_importance': (a['cell_imp'] / c).tolist(),
            }

        gc = max(global_count, 1)
        results = {
            'num_samples': n,
            'num_actions': self.action_dim,
            'avg_feature_importance': {
                CELL_FEATURE_NAMES[j]: float(global_feature_imp[j] / gc)
                for j in range(self.num_features)
            },
            'avg_frame_importance': (global_frame_imp / gc).tolist(),
            'avg_cell_importance': (global_cell_imp / gc).tolist(),
            'per_action': per_action_results,
        }

        return results

    def save(self, results: Dict, output_dir: str):
        """Save saliency results to JSON."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        fpath = out / 'saliency.json'
        with open(fpath, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"    ✓ Saved: {fpath}")