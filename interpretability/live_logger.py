"""
BDH Live Interpretability Logger
=================================

Lightweight wrapper that collects live data from BDH training runs
and periodically computes interpretability scores, logging them
alongside existing training output.

Analyses run (configurable interval, default every 5 PPO updates):
  1. Monosemanticity  — neuron-concept correlation
  2. Sparse Activation — activation density
  3. Hebbian Learning  — synapse weight tracking
  4. Scale-Free        — power-law degree distribution
  5. Saliency Maps     — gradient × input feature attribution

Output:
  - bdh_results/interpretability_scores.csv   (summary row per analysis run)
  - bdh_results/update_N.json (full results per run)

Author: Radio-Cortex Team / KRITI 2026
"""

import os
import csv
import json
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


class InterpretabilityLogger:
    """
    Collects live data from BDH training and periodically runs
    all interpretability analyses, logging results to disk.

    Usage in PPOTrainer:
        # __init__:
        self.interp_logger = InterpretabilityLogger(self.policy, log_dir='logs')

        # collect_rollout (per step):
        self.interp_logger.collect_step(state, e2_metrics_dict)

        # train() (every N updates):
        self.interp_logger.run_periodic_analysis(rollout_states, update_num)
    """

    def __init__(
        self,
        policy: nn.Module,
        log_dir: str = 'logs',
        analysis_interval: int = 5,
        num_cells: int = 3,
    ):
        """
        Args:
            policy:             Trained BDHPolicy instance
            log_dir:            Directory for output files
            analysis_interval:  Run full analysis every N PPO updates
            num_cells:          Number of cells in the environment
        """
        self.policy = policy
        self.log_dir = log_dir
        self.analysis_interval = analysis_interval
        self.num_cells = num_cells

        # Output paths
        self.csv_path = 'bdh_results/interpretability_scores.csv'
        self.json_dir = 'bdh_results'
        os.makedirs(self.json_dir, exist_ok=True)

        # Write CSV header if file doesn't exist
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'update', 'total_steps',
                    'mono_score', 'mono_neurons', 'mono_concepts',
                    'sparsity', 'active_pct',
                    'hebbian_max_change', 'hebbian_strengthened',
                    'scale_free', 'sf_alpha', 'sf_r_squared', 'sf_hubs',
                    'saliency_top_feature', 'saliency_top_frame',
                    'analysis_time_s',
                ])

        # Per-step collection buffers (for monosemanticity)
        self._collected_states: List[np.ndarray] = []
        self._collected_e2_metrics: List[Dict] = []
        self._max_collect = 500  # Cap to prevent memory issues

        # Persistent Hebbian tracker — survives across analysis intervals
        # so we compare weights from update_5 vs update_10 vs update_15 etc.
        self._hebbian_tracker = None  # Initialized lazily on first analysis

        self._initialized = True
        print(f"[InterpretabilityLogger] Initialized — "
              f"analysis every {analysis_interval} updates, "
              f"output → {self.csv_path}")

    def collect_step(self, state: np.ndarray, e2_metrics: Optional[Dict]):
        """
        Collect one timestep of live data for later analysis.
        Called inside collect_rollout() on each env step — must be fast.

        Args:
            state:      Raw state vector, shape (state_dim,)
            e2_metrics: Dict with 'cell_metrics' and 'ue_metrics' keys,
                        or None if unavailable
        """
        if len(self._collected_states) >= self._max_collect:
            return  # Buffer full, skip until next analysis cycle

        try:
            self._collected_states.append(np.array(state, dtype=np.float32))
            self._collected_e2_metrics.append(e2_metrics or {})
        except Exception:
            pass  # Never crash training

    def should_analyze(self, update_num: int) -> bool:
        """Check if analysis should run this update."""
        return (
            update_num > 0
            and update_num % self.analysis_interval == 0
            and len(self._collected_states) >= 10
        )

    def run_periodic_analysis(
        self,
        rollout_states: Optional[torch.Tensor],
        update_num: int,
        total_steps: int = 0,
    ) -> Optional[Dict]:
        """
        Run all 5 interpretability analyses and log results.

        Args:
            rollout_states: Tensor of states from current rollout (N, state_dim)
            update_num:     Current PPO update number
            total_steps:    Current total training steps

        Returns:
            Dict of analysis results, or None on failure
        """
        if not self.should_analyze(update_num):
            return None

        t0 = time.time()
        results = {}

        try:
            # Lazy imports to avoid circular deps and keep startup fast
            from .bdh_interpretability_solo import (
                BDHMonosemanticity, BDHSparsity, BDHHebbian, BDHScaleFree, BDHSaliency
            )

            states_np = np.array(self._collected_states)
            e2_list = list(self._collected_e2_metrics)

            # ── 1. Monosemanticity ──
            try:
                mono = BDHMonosemanticity(self.policy)
                for i in range(len(states_np)):
                    metrics = e2_list[i] if i < len(e2_list) else {}
                    mono.collect(states_np[i], metrics)
                results['monosemanticity'] = mono.analyze()
            except Exception as e:
                results['monosemanticity'] = {'score': 0, 'error': str(e)}

            # ── 2. Sparsity ──
            try:
                sparsity = BDHSparsity(threshold=0.01)
                batch = torch.FloatTensor(states_np[:min(200, len(states_np))])
                sparsity.collect(self.policy, batch)
                results['sparsity'] = sparsity.analyze()
            except Exception as e:
                results['sparsity'] = {'overall_sparsity': 0, 'error': str(e)}

            # ── 3. Hebbian (persistent across updates) ──
            try:
                if self._hebbian_tracker is None:
                    self._hebbian_tracker = BDHHebbian(self.policy)
                # Snapshot current encoder weights (changed by optimizer.step() since last analysis)
                self._hebbian_tracker.record()
                results['hebbian'] = self._hebbian_tracker.analyze()
            except Exception as e:
                results['hebbian'] = {'max_change': 0, 'error': str(e)}

            # ── 4. Scale-Free ──
            try:
                sf = BDHScaleFree(self.policy)
                results['scale_free'] = sf.analyze()
            except Exception as e:
                results['scale_free'] = {'is_scale_free': False, 'error': str(e)}

            # ── 5. Saliency (lightweight — small sample) ──
            try:
                sal = BDHSaliency(self.policy, num_cells=self.num_cells)
                sal_states = states_np[:min(20, len(states_np))]
                results['saliency'] = sal.analyze_batch(sal_states, max_samples=20)
            except Exception as e:
                results['saliency'] = {'error': str(e)}

            elapsed = time.time() - t0
            results['analysis_time_s'] = round(elapsed, 2)
            results['update'] = update_num
            results['total_steps'] = total_steps
            results['timestamp'] = datetime.now().isoformat()
            results['num_samples'] = len(states_np)

            # ── Write full JSON ──
            json_path = os.path.join(self.json_dir, f'update_{update_num}.json')
            self._safe_write_json(json_path, results)

            # ── Append CSV summary row ──
            self._append_csv_row(results, update_num, total_steps, elapsed)

            print(f"[Interpretability] Update {update_num}: "
                  f"mono={results.get('monosemanticity', {}).get('score', 0):.3f} | "
                  f"sparsity={results.get('sparsity', {}).get('overall_sparsity', 0):.3f} | "
                  f"scale_free={'YES' if results.get('scale_free', {}).get('is_scale_free') else 'NO'} | "
                  f"({elapsed:.1f}s)")

        except Exception as e:
            print(f"[Interpretability] Analysis failed (non-fatal): {e}")
            results = None

        # Clear buffers for next cycle
        self._collected_states.clear()
        self._collected_e2_metrics.clear()

        return results

    def _append_csv_row(self, results: Dict, update_num: int, total_steps: int, elapsed: float):
        """Append a summary row to the CSV log."""
        try:
            mono = results.get('monosemanticity', {})
            sp = results.get('sparsity', {})
            heb = results.get('hebbian', {})
            sf = results.get('scale_free', {})
            sal = results.get('saliency', {})

            # Find top saliency feature
            top_feat = ''
            top_frame = ''
            avg_fi = sal.get('avg_feature_importance', {})
            if avg_fi:
                top_feat = max(avg_fi, key=avg_fi.get, default='')
            avg_frame = sal.get('avg_frame_importance', [])
            if avg_frame:
                frame_labels = ['t-2', 't-1', 't-0']
                top_idx = int(np.argmax(avg_frame))
                top_frame = frame_labels[top_idx] if top_idx < len(frame_labels) else ''

            row = [
                datetime.now().isoformat(),
                update_num,
                total_steps,
                round(mono.get('score', 0), 4),
                mono.get('num_monosemantic', 0),
                mono.get('num_concepts', 0),
                round(sp.get('overall_sparsity', 0), 4),
                round(sp.get('active_percentage', 0), 4),
                round(heb.get('max_change', 0), 6),
                heb.get('strengthened_count', 0),
                1 if sf.get('is_scale_free') else 0,
                round(sf.get('alpha', 0) or 0, 4),
                round(sf.get('r_squared', 0) or 0, 4),
                sf.get('num_hubs', 0),
                top_feat,
                top_frame,
                round(elapsed, 2),
            ]

            with open(self.csv_path, 'a', newline='') as f:
                csv.writer(f).writerow(row)

        except Exception as e:
            print(f"[Interpretability] CSV write error (non-fatal): {e}")

    def _safe_write_json(self, path: str, data: Dict):
        """Write JSON with numpy/torch serialization support."""
        def _default(o):
            if isinstance(o, (np.floating, np.integer)):
                return o.item()
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, np.bool_):
                return bool(o)
            if isinstance(o, torch.Tensor):
                return o.detach().cpu().numpy().tolist()
            return str(o)

        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2, default=_default)
        except Exception as e:
            print(f"[Interpretability] JSON write error (non-fatal): {e}")
