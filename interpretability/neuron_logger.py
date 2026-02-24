"""
BDH Neuron Decision Logger + Network Graph Exporter
=====================================================

NeuronLogger:  Hooks into BDH sparse ReLU activations during inference.
               Records which neurons fire per step, tagged with actions.
               Outputs JSONL log file.

BDHNetworkGraph: Extracts encoder/decoder weight matrices as a JSON
                 node-edge graph (input features → latent neurons → actions).

Author: Radio-Cortex Team / KRITI 2026
"""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional

ACTION_NAMES = ['TxPower', 'CIO', 'TTT']
CELL_FEATURE_NAMES = [
    'queue', 'rb_util', 'tx_power', 'load', 'avg_req',
    'avg_tput', 'avg_delay', 'avg_loss', 'max_delay', 'max_loss',
    'jains', 'n_ues', 'sinr_mean', 'ho_count', 'cqi_mean', 'stationary',
]


# ═══════════════════════════════════════════════════════════════════
# 6. NEURON DECISION LOGGER
# ═══════════════════════════════════════════════════════════════════

class NeuronLogger:
    """
    Records per-step neuron activations from BDH's sparse codes.

    Hooks into the BDH module's leaf modules and captures:
    - x_sparse (post-encoder ReLU)
    - y_sparse (post-value ReLU)
    - gate outputs

    Each logged step contains: action taken, state summary, per-layer
    activation indices/values, hub neuron activity, and decision neurons.
    """

    def __init__(self, policy: nn.Module, num_cells: int = 3,
                 hub_neurons: Optional[List[int]] = None,
                 activation_threshold: float = 0.01, top_k_per_action: int = 10):
        """
        Args:
            policy:                Trained BDHPolicy instance
            num_cells:             Number of cells in the environment
            hub_neurons:           Optional list of known hub neuron indices
                                   (from scale-free analysis)
            activation_threshold:  Minimum magnitude to consider a neuron "active"
            top_k_per_action:      Number of top neurons to record per action
        """
        self.policy = policy
        self.policy.eval()
        self.num_cells = num_cells
        self.hub_neurons = set(hub_neurons or [])
        self.threshold = activation_threshold
        self.top_k = top_k_per_action
        self.log = []
        self.step_count = 0

        # Detect BDH config
        bdh = policy.bdh if hasattr(policy, 'bdh') else policy
        self.n_layers = bdh.config.n_layer if hasattr(bdh, 'config') else 4
        self.n_heads = bdh.config.n_head if hasattr(bdh, 'config') else 4

    def step(self, state: np.ndarray) -> Dict:
        """
        Run one inference step, capture all neuron activations.

        Args:
            state: Raw state vector, shape (state_dim,)

        Returns:
            Dict with 'action' (np.ndarray) and 'log_entry' (Dict)
        """
        all_acts = []
        hooks = []

        def global_hook(module, input, output):
            if isinstance(output, torch.Tensor) and output.dim() >= 2:
                all_acts.append(output.detach().cpu())

        bdh = self.policy.bdh if hasattr(self.policy, 'bdh') else self.policy
        for name, module in bdh.named_modules():
            if len(list(module.children())) == 0:
                hooks.append(module.register_forward_hook(global_hook))

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            action_mean, log_std, value = self.policy(state_t)
            action = action_mean.squeeze(0).numpy()

        for h in hooks:
            h.remove()

        # ── Parse activations into layers ──
        layers_data = {}
        acts_per_layer = max(1, len(all_acts) // max(self.n_layers, 1))

        for layer_idx in range(self.n_layers):
            start = layer_idx * acts_per_layer
            end = start + acts_per_layer
            layer_acts = all_acts[start:end]
            layer_data = {}
            act_names = ['x_sparse', 'y_sparse', 'gate']

            for j, act_tensor in enumerate(layer_acts[:3]):
                flat = act_tensor.flatten().numpy()
                active_mask = np.abs(flat) > self.threshold
                active_indices = np.where(active_mask)[0]
                active_values = flat[active_mask]

                name = act_names[j] if j < len(act_names) else f'act_{j}'
                layer_data[name] = {
                    'active_neuron_indices': active_indices.tolist(),
                    'activations': active_values.tolist(),
                    'total_neurons': int(len(flat)),
                    'num_active': int(active_mask.sum()),
                    'sparsity': float(1.0 - active_mask.mean()),
                    'mean_activation': float(flat[active_mask].mean()) if active_mask.any() else 0.0,
                    'max_activation': float(flat[active_mask].max()) if active_mask.any() else 0.0,
                }

            layers_data[f'layer_{layer_idx}'] = layer_data

        # ── Hub neuron tracking ──
        all_active = set()
        for layer_data in layers_data.values():
            for act_data in layer_data.values():
                all_active.update(act_data['active_neuron_indices'])

        hub_active = sorted(list(self.hub_neurons & all_active))

        # ── Decision neurons (top-k by activation magnitude per action) ──
        decision_neurons = {}
        if all_acts:
            all_flat = torch.cat([a.flatten() for a in all_acts]).numpy()
            abs_acts = np.abs(all_flat)

            for a_idx in range(self.num_cells * 3):
                cell_id = a_idx // 3
                act_name = ACTION_NAMES[a_idx % 3]
                label = f"cell{cell_id}_{act_name}"

                top_idx = np.argsort(abs_acts)[::-1][:self.top_k]
                decision_neurons[label] = [
                    {'neuron': int(idx), 'activation': float(all_flat[idx])}
                    for idx in top_idx if abs_acts[idx] > self.threshold
                ]

        # ── State summary (latest frame only) ──
        state_summary = {}
        frame_start = 2 * self.num_cells * 16  # t-0 frame offset
        for c in range(self.num_cells):
            offset = frame_start + c * 16
            for j in range(min(16, len(CELL_FEATURE_NAMES))):
                if offset + j < len(state):
                    state_summary[f"cell{c}_{CELL_FEATURE_NAMES[j]}"] = float(state[offset + j])

        # ── Action labels ──
        action_labels = {}
        for i in range(len(action)):
            action_labels[f"cell{i // 3}_{ACTION_NAMES[i % 3]}"] = float(action[i])

        entry = {
            'step': self.step_count,
            'action': action.tolist(),
            'action_labels': action_labels,
            'value': float(value.item()) if value.dim() == 0 else float(value.squeeze().item()),
            'state_summary': state_summary,
            'layers': layers_data,
            'hub_neurons_active': hub_active,
            'total_active_neurons': len(all_active),
            'decision_neurons': decision_neurons,
        }

        self.log.append(entry)
        self.step_count += 1
        return {'action': action, 'log_entry': entry}

    def save(self, output_path: str):
        """Save neuron log as JSONL file."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w') as f:
            for entry in self.log:
                f.write(json.dumps(entry) + '\n')
        print(f"  ✓ Neuron log saved: {out} ({len(self.log)} steps)")

    def get_summary(self) -> Dict:
        """Aggregate statistics across all logged steps."""
        if not self.log:
            return {'num_steps': 0}

        neuron_fire_count = {}
        total_active_per_step = []
        sparsities = []

        for entry in self.log:
            total_active_per_step.append(entry.get('total_active_neurons', 0))
            for layer_data in entry.get('layers', {}).values():
                for act_data in layer_data.values():
                    for idx in act_data.get('active_neuron_indices', []):
                        key = str(idx)
                        neuron_fire_count[key] = neuron_fire_count.get(key, 0) + 1
                    sparsities.append(act_data.get('sparsity', 0))

        sorted_neurons = sorted(
            neuron_fire_count.items(), key=lambda x: x[1], reverse=True
        )

        # Hub activation rates
        hub_rates = {}
        if self.hub_neurons:
            hub_counts = {h: 0 for h in self.hub_neurons}
            for entry in self.log:
                for h in entry.get('hub_neurons_active', []):
                    if h in hub_counts:
                        hub_counts[h] += 1
            n = len(self.log)
            hub_rates = {str(h): float(count / n) for h, count in hub_counts.items()}

        return {
            'num_steps': len(self.log),
            'avg_active_neurons': float(np.mean(total_active_per_step)),
            'avg_sparsity': float(np.mean(sparsities)) if sparsities else 0,
            'most_active_neurons': [
                {'neuron': name, 'fire_count': count}
                for name, count in sorted_neurons[:20]
            ],
            'hub_activation_rate': hub_rates,
        }


# ═══════════════════════════════════════════════════════════════════
# 7. NETWORK GRAPH EXPORTER
# ═══════════════════════════════════════════════════════════════════

class BDHNetworkGraph:
    """
    Extract BDH as a node-edge graph for visualization.

    Nodes: input features (48) → latent neurons (4 heads × N) → output actions (9)
    Edges: encoder weights (input→latent), decoder weights (latent→output)

    The graph can be loaded into D3.js, Cytoscape, or NetworkX for
    interactive visualization of the BDH architecture.
    """

    def __init__(self, policy: nn.Module, num_cells: int = 3):
        self.policy = policy
        self.num_cells = num_cells

    def extract(self, weight_threshold_percentile: float = 80.0) -> Dict:
        """
        Extract the BDH architecture as a node-edge graph.

        Args:
            weight_threshold_percentile: Only include edges with weight
                                          magnitude above this percentile

        Returns:
            Dict with 'nodes', 'edges', and 'metadata'
        """
        bdh = self.policy.bdh if hasattr(self.policy, 'bdh') else self.policy
        nodes = []
        edges = []

        # ── Input nodes (one per cell × feature in current frame) ──
        for c in range(self.num_cells):
            for j, fname in enumerate(CELL_FEATURE_NAMES):
                nodes.append({
                    'id': f'input_c{c}_{fname}',
                    'type': 'input',
                    'cell': c,
                    'feature': fname,
                    'label': f'C{c}:{fname}',
                    'group': f'cell_{c}',
                })

        # ── Latent nodes + encoder edges ──
        if hasattr(bdh, 'encoder') and isinstance(bdh.encoder, nn.Parameter):
            encoder = bdh.encoder.detach().cpu().numpy()
            nh, D, N = encoder.shape
            threshold = np.percentile(np.abs(encoder), weight_threshold_percentile)

            # Create latent neuron nodes
            for head in range(nh):
                for n_idx in range(N):
                    col = np.abs(encoder[head, :, n_idx])
                    degree = int((col > threshold).sum())
                    nodes.append({
                        'id': f'latent_h{head}_n{n_idx}',
                        'type': 'latent',
                        'head': head,
                        'neuron_index': n_idx,
                        'label': f'H{head}:N{n_idx}',
                        'group': f'head_{head}',
                        'degree': degree,
                        'is_hub': degree > np.percentile(
                            [np.abs(encoder[head, :, k]).sum() for k in range(N)], 90
                        ),
                    })

            # Create encoder edges (input → latent)
            for head in range(nh):
                for d_idx in range(min(D, self.num_cells * 16)):
                    for n_idx in range(N):
                        w = float(encoder[head, d_idx, n_idx])
                        if abs(w) > threshold:
                            cell = d_idx // 16
                            feat_idx = d_idx % 16
                            if cell < self.num_cells and feat_idx < len(CELL_FEATURE_NAMES):
                                edges.append({
                                    'source': f'input_c{cell}_{CELL_FEATURE_NAMES[feat_idx]}',
                                    'target': f'latent_h{head}_n{n_idx}',
                                    'weight': abs(w),
                                    'raw_weight': w,
                                    'type': 'encoder',
                                    'head': head,
                                })

        # ── Output nodes ──
        for c in range(self.num_cells):
            for a_name in ACTION_NAMES:
                nodes.append({
                    'id': f'output_c{c}_{a_name}',
                    'type': 'output',
                    'cell': c,
                    'action': a_name,
                    'label': f'C{c}:{a_name}',
                    'group': f'action_{a_name}',
                })

        # ── Decoder edges (latent → output) ──
        if hasattr(bdh, 'decoder') and isinstance(bdh.decoder, nn.Parameter):
            decoder = bdh.decoder.detach().cpu().numpy()
            d_thresh = np.percentile(np.abs(decoder), weight_threshold_percentile)
            nh_N, D_dec = decoder.shape
            nh_val = bdh.config.n_head if hasattr(bdh, 'config') else 4
            N_val = nh_N // nh_val if nh_val > 0 else nh_N

            for row in range(nh_N):
                head = row // N_val if N_val > 0 else 0
                n_idx = row % N_val if N_val > 0 else row
                for d_idx in range(min(D_dec, self.num_cells * 16)):
                    w = float(decoder[row, d_idx])
                    if abs(w) > d_thresh:
                        cell = d_idx // 16
                        if cell < self.num_cells:
                            for a_name in ACTION_NAMES:
                                edges.append({
                                    'source': f'latent_h{head}_n{n_idx}',
                                    'target': f'output_c{cell}_{a_name}',
                                    'weight': abs(w),
                                    'raw_weight': w,
                                    'type': 'decoder',
                                    'head': head,
                                })

        return {
            'nodes': nodes,
            'edges': edges,
            'metadata': {
                'num_nodes': len(nodes),
                'num_edges': len(edges),
                'num_input': sum(1 for n in nodes if n['type'] == 'input'),
                'num_latent': sum(1 for n in nodes if n['type'] == 'latent'),
                'num_output': sum(1 for n in nodes if n['type'] == 'output'),
                'num_hubs': sum(1 for n in nodes if n.get('is_hub', False)),
                'encoder_shape': list(bdh.encoder.shape) if hasattr(bdh, 'encoder') else [],
                'decoder_shape': list(bdh.decoder.shape) if hasattr(bdh, 'decoder') else [],
            },
        }

    def save(self, graph: Dict, output_path: str):
        """Save graph as JSON file."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w') as f:
            json.dump(graph, f, indent=2)
        print(f"  ✓ Network graph saved: {out}")
        print(f"    {graph['metadata']['num_nodes']} nodes, "
              f"{graph['metadata']['num_edges']} edges")
