"""
BDH Interpretability Analysis — Solo Version
=============================================

Analyzes BDH model interpretability (no baseline comparison).

Analyses:
1. Monosemanticity  — which neurons encode specific O-RAN concepts
2. Sparse Activation — activation density measurement
3. Hebbian Learning  — synapse weight drift during inference
4. Scale-Free        — power-law degree distribution in encoder

Designed for the Radio-Cortex Cell-Centric architecture:
  State:  num_cells × 16 features × 3 temporal frames = 144 dims (3 cells)
  Action: num_cells × 3 (TxPower, CIO, TTT) = 9 dims
  BDH:    4-layer, 4-head, 128-dim encoder with sparse ReLU activations

Author: Radio-Cortex Team / KRITI 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════
# 1. MONOSEMANTICITY ANALYZER
# ═══════════════════════════════════════════════════════════════════

class BDHMonosemanticity:
    """
    Analyze which BDH neurons encode specific O-RAN network concepts.
    
    Methodology:
    - Hook into the sparse ReLU activations inside BDH layers
    - Label each timestep with active network concepts from E2 metrics
    - Correlate: does neuron N fire significantly more when concept C is present?
    - A neuron is "monosemantic" if it correlates strongly with ONE concept
    """

    def __init__(self, policy: nn.Module):
        """
        Args:
            policy: The full BDHPolicy (not the inner bdh.BDH module)
        """
        self.policy = policy
        self.policy.eval()

        # Storage
        self.activations = []   # List of 1D numpy arrays (flattened sparse acts)
        self.concepts = []      # List of List[str] per timestep

    def collect(self, state: np.ndarray, e2_metrics: Dict):
        """
        Collect one timestep of activations + concept labels.

        Args:
            state:      Raw state vector, shape (state_dim,)
            e2_metrics: Dict with 'cell_metrics' and 'ue_metrics' from E2 interface
        """
        captured = []

        def hook_fn(module, input, output):
            """Capture output of any module that produces sparse activations."""
            if isinstance(output, torch.Tensor) and output.requires_grad is False:
                captured.append(output.detach().cpu())

        # Register hooks on all leaf modules inside the BDH component
        hooks = []
        bdh_module = self.policy.bdh if hasattr(self.policy, 'bdh') else self.policy
        for name, module in bdh_module.named_modules():
            if len(list(module.children())) == 0:
                hooks.append(module.register_forward_hook(hook_fn))

        # Forward pass through the FULL policy (handles state → tokens → BDH)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            try:
                self.policy(state_t)
            except Exception:
                pass

        # Remove hooks
        for h in hooks:
            h.remove()

        # Flatten all captured activations into one vector per timestep
        if captured:
            flat = torch.cat([c.flatten() for c in captured]).numpy()
            self.activations.append(flat)

        # Extract and store concepts
        concepts = self._extract_concepts(e2_metrics)
        self.concepts.append(concepts)

    def _extract_concepts(self, e2_metrics: Dict) -> List[str]:
        """
        Extract human-readable network concepts from E2 KPM metrics.
        
        These become the labels we correlate with neuron activations.
        Maps directly to O-RAN conditions an operator would care about.
        """
        concepts = []

        if not e2_metrics:
            return ["normal_operation"]

        # ── Cell-level concepts ──
        cell_metrics = e2_metrics.get('cell_metrics', {})
        for cell_id, metrics in cell_metrics.items():
            if isinstance(metrics, dict):
                queue = metrics.get('queue_length', 0)
                rb_util = metrics.get('rb_utilization', 0)
                power = metrics.get('tx_power', 23)
                load = metrics.get('cell_load', 0)

                if queue > 500:
                    concepts.append(f"high_queue_c{cell_id}")
                if queue > 800:
                    concepts.append("severe_congestion")
                if rb_util > 0.8:
                    concepts.append(f"rb_saturated_c{cell_id}")
                if rb_util < 0.3:
                    concepts.append(f"underloaded_c{cell_id}")
                if power > 35:
                    concepts.append(f"high_power_c{cell_id}")
                if power < 15:
                    concepts.append(f"low_power_c{cell_id}")

        # ── UE-level concepts (aggregated) ──
        ue_metrics = e2_metrics.get('ue_metrics', {})
        losses, delays, sinrs, tputs = [], [], [], []

        for ue_id, metrics in ue_metrics.items():
            if isinstance(metrics, dict):
                loss = metrics.get('packet_loss', 0)
                sinr = metrics.get('sinr', 10)
                delay = metrics.get('delay', 0)
                tput = metrics.get('throughput', 0)

                losses.append(loss)
                delays.append(delay)
                sinrs.append(sinr)
                tputs.append(tput)

                if loss > 0.05:
                    concepts.append("packet_loss_event")
                if sinr < 5:
                    concepts.append("poor_sinr")
                if sinr > 20:
                    concepts.append("excellent_sinr")
                if delay > 50:
                    concepts.append("high_delay")
                if tput < 1.0:
                    concepts.append("low_throughput")

        # ── Network-wide concepts ──
        if losses:
            if np.mean(losses) > 0.1:
                concepts.append("network_degraded")
            if np.mean(losses) < 0.01:
                concepts.append("network_healthy")
        if delays:
            if np.mean(delays) > 100:
                concepts.append("network_congested")
            if np.mean(delays) < 10:
                concepts.append("low_latency")
        if tputs:
            if np.mean(tputs) > 10:
                concepts.append("high_throughput")
            tput_arr = np.array(tputs)
            if len(tput_arr) > 1 and tput_arr.sum() > 0:
                sq_sum = np.sum(tput_arr ** 2)
                jains = (tput_arr.sum() ** 2) / (len(tput_arr) * sq_sum) if sq_sum > 0 else 1
                if jains > 0.8:
                    concepts.append("fair_allocation")
                elif jains < 0.5:
                    concepts.append("unfair_allocation")

        # Deduplicate but preserve order
        seen = set()
        unique = []
        for c in concepts:
            if c not in seen:
                seen.add(c)
                unique.append(c)

        return unique if unique else ["normal_operation"]

    def analyze(self, min_samples: int = 5, correlation_threshold: float = 0.5) -> Dict:
        """
        Compute monosemanticity after data collection.

        For each neuron, test if it activates significantly more when a 
        specific concept is present vs absent (Cohen's d-like measure).

        Args:
            min_samples:           Minimum times a concept must appear
            correlation_threshold: Minimum effect size to call "monosemantic"

        Returns:
            Dict with score, neuron mappings, and concept list
        """
        if not self.activations:
            return {'score': 0, 'neurons': {}, 'total_neurons': 0,
                    'num_monosemantic': 0, 'num_concepts': 0, 'concepts': []}

        # Stack into matrix: (num_timesteps, num_neurons)
        # Handle varying sizes by padding to max length
        max_len = max(len(a) for a in self.activations)
        padded = []
        for a in self.activations:
            if len(a) < max_len:
                a = np.pad(a, (0, max_len - len(a)), mode='constant')
            padded.append(a)

        act_matrix = np.array(padded)
        num_samples, num_neurons = act_matrix.shape

        # Collect all unique concepts
        all_concepts = set()
        for concept_list in self.concepts:
            all_concepts.update(concept_list)

        # Test each neuron × concept pair
        monosemantic = {}

        for neuron_idx in range(num_neurons):
            neuron_acts = act_matrix[:, neuron_idx]

            # Skip dead neurons
            if neuron_acts.max() < 1e-6:
                continue

            neuron_std = neuron_acts.std()
            if neuron_std < 1e-8:
                continue

            best_concept = None
            best_corr = 0

            for concept in all_concepts:
                # Binary mask: was this concept present at each timestep?
                n_valid = min(num_samples, len(self.concepts))
                mask = np.array([
                    concept in self.concepts[i] for i in range(n_valid)
                ])

                if len(mask) < num_samples:
                    mask = np.pad(mask, (0, num_samples - len(mask)),
                                  mode='constant', constant_values=False)

                if mask.sum() < min_samples or (~mask).sum() < min_samples:
                    continue

                # Effect size: (mean_present - mean_absent) / pooled_std
                mean_present = neuron_acts[mask].mean()
                mean_absent = neuron_acts[~mask].mean()
                correlation = (mean_present - mean_absent) / (neuron_std + 1e-8)

                if correlation > correlation_threshold and correlation > best_corr:
                    best_corr = correlation
                    best_concept = concept

            if best_concept is not None:
                if neuron_idx not in monosemantic:
                    monosemantic[neuron_idx] = []
                monosemantic[neuron_idx].append({
                    'concept': best_concept,
                    'correlation': float(best_corr),
                })

        # Score = fraction of neurons that are monosemantic
        score = len(monosemantic) / max(num_neurons, 1)

        return {
            'score': float(score),
            'neurons': {str(k): v for k, v in monosemantic.items()},
            'total_neurons': int(num_neurons),
            'num_monosemantic': len(monosemantic),
            'num_concepts': len(all_concepts),
            'concepts': sorted(list(all_concepts)),
        }


# ═══════════════════════════════════════════════════════════════════
# 2. SPARSE ACTIVATION ANALYZER
# ═══════════════════════════════════════════════════════════════════

class BDHSparsity:
    """
    Measure activation sparsity in the BDH network.
    
    BDH uses ReLU after encoder projections, which naturally creates
    sparse activations. We measure what fraction of neurons are
    effectively "off" (below threshold) during inference.
    """

    def __init__(self, threshold: float = 0.01):
        """
        Args:
            threshold: Activation magnitude below which a neuron is "inactive"
        """
        self.threshold = threshold
        self.layer_activations = []  # List of (layer_name, activation_tensor)

    def collect(self, policy: nn.Module, states: torch.Tensor):
        """
        Run a batch of states through the policy and capture all activations.

        Args:
            policy: Full BDHPolicy
            states: Tensor of shape (batch_size, state_dim)
        """
        captured = []

        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    captured.append((name, output.detach().cpu()))
            return hook

        hooks = []
        bdh_module = policy.bdh if hasattr(policy, 'bdh') else policy
        for name, module in bdh_module.named_modules():
            if len(list(module.children())) == 0:
                hooks.append(module.register_forward_hook(hook_fn(name)))

        with torch.no_grad():
            policy.eval()
            # Process in mini-batches to avoid OOM
            batch_size = min(64, states.shape[0])
            for i in range(0, states.shape[0], batch_size):
                batch = states[i:i + batch_size]
                try:
                    policy(batch)
                except Exception:
                    pass

        for h in hooks:
            h.remove()

        self.layer_activations = captured

    def analyze(self) -> Dict:
        """
        Compute sparsity metrics across all captured activations.
        """
        if not self.layer_activations:
            return {
                'overall_sparsity': 0.0, 'active_percentage': 1.0,
                'layer_sparsities': [], 'mean_activation': 0.0,
                'std_activation': 0.0, 'max_activation': 0.0,
            }

        layer_sparsities = []
        all_flat = []

        for name, act in self.layer_activations:
            flat = act.reshape(-1).numpy()
            all_flat.append(flat)

            inactive = (np.abs(flat) < self.threshold).sum()
            sparsity = inactive / max(len(flat), 1)
            layer_sparsities.append({
                'name': name,
                'sparsity': float(sparsity),
                'size': len(flat),
            })

        all_acts = np.concatenate(all_flat)
        overall_sparsity = float((np.abs(all_acts) < self.threshold).mean())
        active_pct = 1.0 - overall_sparsity

        return {
            'overall_sparsity': overall_sparsity,
            'active_percentage': float(active_pct),
            'layer_sparsities': [ls['sparsity'] for ls in layer_sparsities],
            'layer_details': layer_sparsities,
            'mean_activation': float(all_acts.mean()),
            'std_activation': float(all_acts.std()),
            'max_activation': float(all_acts.max()),
        }


# ═══════════════════════════════════════════════════════════════════
# 3. HEBBIAN LEARNING ANALYZER
# ═══════════════════════════════════════════════════════════════════

class BDHHebbian:
    """
    Track synapse weight changes during inference.
    
    BDH's encoder/decoder parameters can exhibit weight drift
    during forward passes due to the Hebbian-like multiplicative
    gating (xy_sparse = x_sparse * y_sparse). We snapshot weights
    at multiple points and measure the delta.
    
    Note: In standard PyTorch inference (torch.no_grad), weights
    don't change. This analysis is meaningful when BDH has
    inference-time plasticity enabled, or we compare weights
    across training checkpoints.
    """

    def __init__(self, policy: nn.Module):
        self.policy = policy
        self.weight_snapshots = []

    def record(self):
        """Snapshot current encoder weights."""
        bdh = self.policy.bdh if hasattr(self.policy, 'bdh') else self.policy

        if hasattr(bdh, 'encoder') and isinstance(bdh.encoder, nn.Parameter):
            snapshot = bdh.encoder.detach().cpu().numpy().copy()
            self.weight_snapshots.append(snapshot)

    def record_from_checkpoint(self, checkpoint_path: str):
        """
        Record weights from a saved checkpoint file.
        Useful for comparing early vs late training.
        """
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        state_dict = ckpt.get('policy_state_dict', ckpt)

        for key, value in state_dict.items():
            if 'bdh.encoder' in key and 'encoder_v' not in key:
                self.weight_snapshots.append(value.numpy().copy())
                break

    def analyze(self) -> Dict:
        """
        Compute weight change statistics between snapshots.
        """
        if len(self.weight_snapshots) < 2:
            return {
                'num_timesteps': len(self.weight_snapshots),
                'strengthened_count': 0,
                'max_change': 0.0,
                'mean_change': 0.0,
                'weight_shape': list(self.weight_snapshots[0].shape)
                    if self.weight_snapshots else [],
            }

        initial = self.weight_snapshots[0]
        final = self.weight_snapshots[-1]
        delta = final - initial

        abs_delta = np.abs(delta)
        threshold = np.percentile(abs_delta, 95)
        strengthened = abs_delta > threshold

        return {
            'num_timesteps': len(self.weight_snapshots),
            'strengthened_count': int(strengthened.sum()),
            'weakened_count': int((delta < -threshold).sum()),
            'max_change': float(abs_delta.max()),
            'mean_change': float(abs_delta.mean()),
            'std_change': float(abs_delta.std()),
            'threshold_95': float(threshold),
            'weight_shape': list(initial.shape),
            'total_synapses': int(initial.size),
        }


# ═══════════════════════════════════════════════════════════════════
# 4. SCALE-FREE TOPOLOGY ANALYZER
# ═══════════════════════════════════════════════════════════════════

class BDHScaleFree:
    """
    Analyze the scale-free network properties of BDH's encoder.
    
    The BDH encoder maps input dimensions to sparse latent codes.
    We treat the weight matrix as an adjacency matrix (thresholded)
    and test if the resulting graph has a power-law degree distribution,
    which is the hallmark of scale-free networks.
    
    Scale-free networks have:
    - Few "hub" neurons with many connections (high degree)
    - Many neurons with few connections
    - Power-law: P(k) ~ k^(-alpha), typically 2 < alpha < 3
    """

    def __init__(self, policy: nn.Module):
        self.policy = policy

    def analyze(self, percentile_threshold: float = 75.0) -> Dict:
        """
        Test encoder weight matrix for scale-free structure.

        Args:
            percentile_threshold: Weight magnitude percentile for 
                                  binary adjacency (default: 75th)
        """
        bdh = self.policy.bdh if hasattr(self.policy, 'bdh') else self.policy

        if not hasattr(bdh, 'encoder'):
            return {'is_scale_free': False, 'error': 'No encoder found'}

        # Get encoder weights: shape [n_head, D, N]
        encoder = bdh.encoder.detach().cpu().numpy()

        # Reshape to 2D: (D, n_head * N)
        if len(encoder.shape) == 3:
            nh, D, N = encoder.shape
            weight_2d = encoder.transpose(1, 0, 2).reshape(D, nh * N)
        elif len(encoder.shape) == 2:
            weight_2d = encoder
        else:
            return {'is_scale_free': False, 'error': f'Unexpected shape {encoder.shape}'}

        # Create binary adjacency matrix (strong connections only)
        magnitudes = np.abs(weight_2d)
        threshold = np.percentile(magnitudes, percentile_threshold)
        adjacency = (magnitudes > threshold).astype(int)

        # Compute degree of each "neuron" (row = input dim)
        degrees = adjacency.sum(axis=1)

        # Degree statistics
        degree_stats = {
            'mean': float(degrees.mean()),
            'std': float(degrees.std()),
            'max': int(degrees.max()),
            'min': int(degrees.min()),
            'median': float(np.median(degrees)),
        }

        # Test for power law: log-log linear regression
        unique_degrees, counts = np.unique(degrees[degrees > 0], return_counts=True)
        probs = counts / counts.sum()

        alpha = None
        r_squared = None
        is_scale_free = False

        if len(unique_degrees) > 3:
            log_k = np.log(unique_degrees.astype(float))
            log_p = np.log(probs.astype(float))

            # Linear regression in log-log space
            n = len(log_k)
            sum_x = log_k.sum()
            sum_y = log_p.sum()
            sum_xy = (log_k * log_p).sum()
            sum_x2 = (log_k ** 2).sum()

            denom = n * sum_x2 - sum_x ** 2
            if abs(denom) > 1e-10:
                slope = (n * sum_xy - sum_x * sum_y) / denom
                intercept = (sum_y - slope * sum_x) / n

                # R² calculation
                y_pred = slope * log_k + intercept
                ss_res = ((log_p - y_pred) ** 2).sum()
                ss_tot = ((log_p - log_p.mean()) ** 2).sum()
                r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0

                alpha = float(-slope)
                is_scale_free = (r_squared > 0.7 and 1.5 < alpha < 3.5)

        # Identify hub neurons (top 10% by degree)
        hub_threshold = np.percentile(degrees, 90)
        hub_neurons = np.where(degrees >= hub_threshold)[0]

        return {
            'is_scale_free': bool(is_scale_free),
            'alpha': alpha,
            'r_squared': r_squared,
            'degree_stats': degree_stats,
            'hub_neurons': hub_neurons.tolist(),
            'num_hubs': len(hub_neurons),
            'degrees': degrees.tolist(),
            'network_size': int(len(degrees)),
            'total_connections': int(adjacency.sum()),
            'encoder_shape': list(encoder.shape),
        }


# ═══════════════════════════════════════════════════════════════════
# ORCHESTRATOR — Run All Four Analyses
# ═══════════════════════════════════════════════════════════════════

def run_full_analysis(
    policy: nn.Module,
    states: np.ndarray,
    e2_metrics_list: List[Dict],
    output_dir: str = './bdh_results',
    checkpoint_paths: Optional[List[str]] = None,
) -> Dict:
    """
    Run all four BDH interpretability analyses.

    Args:
        policy:           Trained BDHPolicy instance (eval mode)
        states:           Array of state vectors, shape (N, state_dim)
        e2_metrics_list:  List of E2 metric dicts, one per state
        output_dir:       Where to save JSON results
        checkpoint_paths: Optional list of checkpoint files for Hebbian
                         cross-checkpoint comparison

    Returns:
        Dict with all analysis results
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results = {}

    # ── 1. Monosemanticity ──
    print("\n" + "=" * 60)
    print("  [1/4] MONOSEMANTICITY ANALYSIS")
    print("=" * 60)

    mono = BDHMonosemanticity(policy)
    for i in range(len(states)):
        metrics = e2_metrics_list[i] if i < len(e2_metrics_list) else {}
        mono.collect(states[i], metrics)
        if (i + 1) % 100 == 0:
            print(f"    Collected {i + 1}/{len(states)} timesteps")

    mono_results = mono.analyze()
    results['monosemanticity'] = mono_results

    with open(out / 'monosemanticity.json', 'w') as f:
        json.dump(mono_results, f, indent=2)
    print(f"  ✓ Score: {mono_results['score']:.3f}")
    print(f"    Monosemantic neurons: {mono_results['num_monosemantic']}/{mono_results['total_neurons']}")
    print(f"    Concepts discovered: {mono_results['num_concepts']}")

    # ── 2. Sparsity ──
    print("\n" + "=" * 60)
    print("  [2/4] SPARSE ACTIVATION ANALYSIS")
    print("=" * 60)

    sparsity = BDHSparsity(threshold=0.01)
    states_tensor = torch.FloatTensor(states[:min(500, len(states))])
    sparsity.collect(policy, states_tensor)
    sparse_results = sparsity.analyze()
    results['sparsity'] = sparse_results

    with open(out / 'sparsity.json', 'w') as f:
        json.dump(sparse_results, f, indent=2)
    print(f"  ✓ Sparsity: {sparse_results['overall_sparsity']:.3f} ({sparse_results['overall_sparsity']*100:.1f}% inactive)")
    print(f"    Active: {sparse_results['active_percentage']*100:.1f}%")

    # ── 3. Hebbian ──
    print("\n" + "=" * 60)
    print("  [3/4] HEBBIAN LEARNING ANALYSIS")
    print("=" * 60)

    hebbian = BDHHebbian(policy)

    if checkpoint_paths and len(checkpoint_paths) >= 2:
        # Compare across training checkpoints
        for cp in checkpoint_paths:
            hebbian.record_from_checkpoint(cp)
        print(f"    Loaded {len(checkpoint_paths)} checkpoints for comparison")
    else:
        # Record current weights (will need at least 2 snapshots)
        hebbian.record()
        # Run some forward passes (weights won't change in eval mode,
        # but this establishes the baseline)
        with torch.no_grad():
            for i in range(min(50, len(states))):
                policy(torch.FloatTensor(states[i]).unsqueeze(0))
        hebbian.record()
        print("    Note: Single checkpoint — comparing pre/post inference (minimal drift expected)")

    hebb_results = hebbian.analyze()
    results['hebbian'] = hebb_results

    with open(out / 'hebbian.json', 'w') as f:
        json.dump(hebb_results, f, indent=2)
    print(f"  ✓ Timesteps: {hebb_results['num_timesteps']}")
    print(f"    Strengthened synapses: {hebb_results['strengthened_count']}")
    print(f"    Max change: {hebb_results['max_change']:.6f}")

    # ── 4. Scale-Free ──
    print("\n" + "=" * 60)
    print("  [4/4] SCALE-FREE TOPOLOGY ANALYSIS")
    print("=" * 60)

    sf = BDHScaleFree(policy)
    sf_results = sf.analyze()
    results['scale_free'] = sf_results

    with open(out / 'scale_free.json', 'w') as f:
        json.dump(sf_results, f, indent=2)
    print(f"  ✓ Scale-Free: {'YES' if sf_results['is_scale_free'] else 'NO'}")
    if sf_results.get('alpha'):
        print(f"    Power law α: {sf_results['alpha']:.3f}")
        print(f"    R² fit: {sf_results['r_squared']:.3f}")
    print(f"    Hub neurons: {sf_results['num_hubs']}")
    print(f"    Network size: {sf_results['network_size']} neurons")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  ALL ANALYSES COMPLETE")
    print("=" * 60)
    print(f"  Results saved to: {out}/")
    print(f"  Files: monosemanticity.json, sparsity.json, hebbian.json, scale_free.json")

    return results