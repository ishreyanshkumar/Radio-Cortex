"""
BDH Interpretability Visualization Tab for Gradio
===================================================

Interactive neural architecture visualization with:
- Plotly network graph (BDH architecture: inputs → latent → outputs)
- Glassmorphism score cards
- Training evolution charts
- Layer-wise sparsity breakdown

Reads from:
  bdh_results/*.json          (offline analysis results)
  bdh_results/      (live training analysis)
  bdh_results/interpretability_scores.csv  (summary over time)

Author: Radio-Cortex Team / KRITI 2026
"""

import os
import json
import glob
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).parent.parent
BDH_RESULTS = ROOT / "bdh_results"
LOGS_DIR = BDH_RESULTS
INTERP_DIR = BDH_RESULTS
SCORES_CSV = BDH_RESULTS / "interpretability_scores.csv"


# ═══════════════════════════════════════════════════════════════════
# Data Loaders
# ═══════════════════════════════════════════════════════════════════

def _load_json(name):
    p = BDH_RESULTS / name
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def _load_scores_csv():
    if SCORES_CSV.exists():
        try:
            return pd.read_csv(SCORES_CSV)
        except Exception:
            return None
    return None


def _list_analysis_updates():
    files = []
    if BDH_RESULTS.exists():
        files.extend(glob.glob(str(BDH_RESULTS / "update_focus_*.json")))
        files.extend(glob.glob(str(BDH_RESULTS / "update_*.json")))
        files.extend(glob.glob(str(BDH_RESULTS / "interp_loop_*.json")))
    return [os.path.basename(f) for f in files]


# ═══════════════════════════════════════════════════════════════════
# Network Graph (Plotly-based — works natively in Gradio)
# ═══════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════
# Saliency Heatmap (Feature Attributions)
# ═══════════════════════════════════════════════════════════════════

def _empty_fig(title="No Data"):
    fig = go.Figure()
    fig.add_annotation(text=title, xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=14, color="#64748b"))
    _style(fig, title)
    fig.update_layout(height=300, xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
    return fig

def _build_saliency_heatmap(saliency):
    if not saliency or 'per_action' not in saliency:
        return _empty_fig("No Saliency Data")

    actions = list(saliency['per_action'].keys())
    # Grab features from the first action to ensure ordering is consistent
    features = list(saliency['per_action'][actions[0]]['feature_importance'].keys())
    
    # Z-matrix: (features on Y, actions on X) -> shape (len(features), len(actions))
    z_data = []
    for feat in features:
        row = []
        for act in actions:
            val = saliency['per_action'][act]['feature_importance'].get(feat, 0.0)
            row.append(val)
        z_data.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=actions,
        y=features,
        colorscale='Inferno',
        hoverongaps=False,
        hovertemplate='Feature: %{y}<br>Action: %{x}<br>Attention: %{z:.4f}<extra></extra>'
    ))

    _style(fig, 'Feature Attributions per RL Action')
    fig.update_layout(
        height=450,
        xaxis_tickangle=-45,
        margin=dict(l=100, b=100)
    )
    return fig


# ═══════════════════════════════════════════════════════════════════

def _build_network_plotly(scale_free, sparsity):
    """Build interactive Plotly network graph of BDH architecture."""

    hub_neurons = set(scale_free.get('hub_neurons', [])) if scale_free else set()
    degrees = scale_free.get('degrees', []) if scale_free else []
    encoder_shape = scale_free.get('encoder_shape', [4, 128, 128]) if scale_free else [4, 128, 128]
    n_heads = encoder_shape[0] if len(encoder_shape) >= 1 else 4
    N = encoder_shape[1] if len(encoder_shape) >= 2 else 128
    overall_sp = sparsity.get('overall_sparsity', 0.85) if sparsity else 0.85

    # Layout: 3 columns (input → latent → output)
    x_in, x_lat, x_out = 0.0, 0.5, 1.0

    # Feature names (input layer)
    feat_names = ['queue', 'rb_util', 'tx_pwr', 'load', 'avg_req',
                  'avg_tput', 'avg_delay', 'avg_loss', 'max_delay', 'max_loss',
                  'jains', 'n_ues', 'sinr', 'ho_cnt', 'cqi', 'stat']
    act_names = ['TxPower', 'CIO', 'TTT']
    n_inputs = len(feat_names)
    n_outputs = len(act_names)

    # Sample latent neurons (show up to 32)
    max_latent = min(N, 32)
    sample_idx = np.linspace(0, max(N - 1, 0), max_latent, dtype=int)

    # Degree normalization
    max_deg = max(degrees) if degrees else 1
    min_deg = min(degrees) if degrees else 0
    deg_range = max(max_deg - min_deg, 1)

    # Colors per head
    head_colors = ['#8b5cf6', '#f472b6', '#34d399', '#fbbf24']

    # ── Build node positions ──
    node_x, node_y, node_color, node_size, node_text, node_label = [], [], [], [], [], []

    # Input nodes (spread vertically)
    for i, fn in enumerate(feat_names):
        y = 1.0 - (i / max(n_inputs - 1, 1))
        node_x.append(x_in)
        node_y.append(y)
        node_color.append('#00d4ff')
        node_size.append(14)
        node_text.append(f'Input: {fn}')
        node_label.append(fn)

    # Latent neurons (spread vertically)
    for j, idx in enumerate(sample_idx):
        y = 1.0 - (j / max(max_latent - 1, 1))
        is_hub = int(idx) in hub_neurons
        deg = degrees[idx] if idx < len(degrees) else 0
        norm_deg = (deg - min_deg) / deg_range
        head = j % n_heads

        node_x.append(x_lat)
        node_y.append(y)
        node_color.append('#facc15' if is_hub else head_colors[head % len(head_colors)])
        node_size.append(38 if is_hub else 8 + norm_deg * 12)
        node_text.append(f'N{idx} | Head {head} | Deg: {deg}{" | ⭐ HUB" if is_hub else ""}')
        node_label.append(f'⭐{idx}' if is_hub else '')

    # Output nodes
    for i, an in enumerate(act_names):
        y = 0.8 - (i / max(n_outputs - 1, 1)) * 0.6
        node_x.append(x_out)
        node_y.append(y)
        node_color.append('#f97316')
        node_size.append(16)
        node_text.append(f'Output: {an}')
        node_label.append(an)

    # ── Build edges ──
    edge_x, edge_y, edge_colors = [], [], []
    np.random.seed(42)

    for j, idx in enumerate(sample_idx):
        lat_node_idx = n_inputs + j
        lat_x = node_x[lat_node_idx]
        lat_y = node_y[lat_node_idx]

        # Input → Latent (2-4 connections per neuron)
        n_conn = np.random.randint(2, 5)
        sources = np.random.choice(n_inputs, size=min(n_conn, n_inputs), replace=False)
        for src in sources:
            w = np.random.uniform(0.15, 0.8)
            edge_x.extend([node_x[src], lat_x, None])
            edge_y.extend([node_y[src], lat_y, None])
            edge_colors.append(f'rgba(100,149,237,{w * 0.4:.2f})')

        # Latent → Output (active neurons only)
        is_active = np.random.random() > overall_sp
        if is_active or int(idx) in hub_neurons:
            n_out = np.random.randint(1, min(4, n_outputs + 1))
            targets = np.random.choice(n_outputs, size=n_out, replace=False)
            for tgt in targets:
                out_idx = n_inputs + max_latent + tgt
                w = np.random.uniform(0.3, 1.0)
                edge_x.extend([lat_x, node_x[out_idx], None])
                edge_y.extend([lat_y, node_y[out_idx], None])
                edge_colors.append(f'rgba(249,115,22,{w * 0.5:.2f})')

    # ── Build figure ──
    fig = go.Figure()

    # Draw edges (one trace per edge for color control is too expensive, batch them)
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=0.6, color='rgba(100,149,237,0.15)'),
        hoverinfo='none',
        showlegend=False,
    ))

    # Draw nodes — separate traces for legend
    groups = {}
    for i, (x, y, c, s, t, l) in enumerate(zip(node_x, node_y, node_color, node_size, node_text, node_label)):
        if i < n_inputs:
            grp = 'Input Features'
        elif i < n_inputs + max_latent:
            j = i - n_inputs
            idx = int(sample_idx[j])
            if idx in hub_neurons:
                grp = '⭐ Hub Neurons'
            else:
                head = j % n_heads
                grp = f'Head {head}'
        else:
            grp = 'Output Actions'
        if grp not in groups:
            groups[grp] = {'x': [], 'y': [], 'c': [], 's': [], 't': [], 'l': []}
        groups[grp]['x'].append(x)
        groups[grp]['y'].append(y)
        groups[grp]['c'].append(c)
        groups[grp]['s'].append(s)
        groups[grp]['t'].append(t)
        groups[grp]['l'].append(l)

    # Define group order and colors
    grp_colors = {
        'Input Features': '#00d4ff',
        'Head 0': '#8b5cf6',
        'Head 1': '#f472b6',
        'Head 2': '#34d399',
        'Head 3': '#fbbf24',
        '⭐ Hub Neurons': '#ef4444',
        'Output Actions': '#f97316',
    }

    for grp, data in groups.items():
        color = grp_colors.get(grp, '#8b5cf6')
        fig.add_trace(go.Scatter(
            x=data['x'], y=data['y'],
            mode='markers+text',
            marker=dict(
                size=data['s'],
                color=color,
                line=dict(width=1.5, color='rgba(255,255,255,0.3)'),
                symbol='circle',
            ),
            text=data['l'],
            textposition='middle right' if grp == 'Input Features' else
                         'middle left' if grp == 'Output Actions' else 'top center',
            textfont=dict(size=9, color='#94a3b8'),
            hovertext=data['t'],
            hoverinfo='text',
            name=grp,
        ))

    # Layer labels
    fig.add_annotation(x=0.0, y=1.08, text='<b>INPUT (16 features)</b>',
                       showarrow=False, font=dict(color='#00d4ff', size=13))
    fig.add_annotation(x=0.5, y=1.08, text=f'<b>LATENT ({N} neurons × {n_heads} heads)</b>',
                       showarrow=False, font=dict(color='#8b5cf6', size=13))
    fig.add_annotation(x=1.0, y=1.08, text='<b>OUTPUT (3 actions)</b>',
                       showarrow=False, font=dict(color='#f97316', size=13))

    # Sparsity annotation
    fig.add_annotation(
        x=0.5, y=-0.08,
        text=f'Sparsity: {overall_sp*100:.1f}% inactive | '
             f'{len(hub_neurons)} hub neurons | '
             f'{N} total neurons × {n_heads} attention heads',
        showarrow=False, font=dict(color='#64748b', size=11),
    )

    fig.update_layout(
        title='🌳 BDH Neural Architecture — Synapse Map',
        paper_bgcolor='#0a0e1a',
        plot_bgcolor='#0a0e1a',
        font=dict(color='#f1f5f9'),
        height=650,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.15, 1.15]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.15, 1.15]),
        legend=dict(
            bgcolor='rgba(17,24,39,0.9)', bordercolor='#1e293b', borderwidth=1,
            font=dict(size=11), x=0.01, y=0.99,
        ),
        margin=dict(l=20, r=20, t=60, b=50),
    )

    return fig


# ═══════════════════════════════════════════════════════════════════
# Score Cards HTML
# ═══════════════════════════════════════════════════════════════════

def _build_score_cards(mono, sparsity, scale_free, hebbian):
    """Build glassmorphism score cards for each analysis."""
    cards = []

    def _card(title, value, subtitle, color, icon):
        return f"""
        <div style="flex:1;min-width:200px;padding:20px;background:linear-gradient(135deg,
            rgba({color},0.15), rgba({color},0.05));
            border:1px solid rgba({color},0.3);border-radius:16px;
            backdrop-filter:blur(10px);text-align:center;">
            <div style="font-size:28px;margin-bottom:6px;">{icon}</div>
            <div style="font-size:32px;font-weight:800;color:rgb({color});
                text-shadow:0 0 20px rgba({color},0.4);">{value}</div>
            <div style="font-size:14px;font-weight:600;color:#e2e8f0;margin-top:4px;">{title}</div>
            <div style="font-size:11px;color:#64748b;margin-top:4px;">{subtitle}</div>
        </div>
        """

    if mono:
        score = mono.get('score', 0)
        n_mono = mono.get('num_monosemantic', 0)
        n_total = mono.get('total_neurons', 0)
        n_concepts = mono.get('num_concepts', 0)
        cards.append(_card(
            'Monosemanticity', f'{score*100:.1f}%',
            f'{n_mono:,} / {n_total:,} neurons • {n_concepts} concepts',
            '139,92,246', '🧠'))

    if sparsity:
        sp = sparsity.get('overall_sparsity', 0)
        active = sparsity.get('active_percentage', 0)
        cards.append(_card(
            'Sparse Activation', f'{sp*100:.1f}%',
            f'{active*100:.1f}% neurons fire per decision',
            '0,212,255', '⚡'))

    if scale_free:
        is_sf = scale_free.get('is_scale_free', False)
        alpha = scale_free.get('alpha', 0)
        r2 = scale_free.get('r_squared', 0)
        n_hubs = scale_free.get('num_hubs', 0)
        cards.append(_card(
            'Scale-Free', 'YES ✓' if is_sf else 'NO ✗',
            f'α={alpha:.2f} R²={r2:.3f} • {n_hubs} hubs',
            '52,211,153' if is_sf else '248,113,113', '🕸️'))

    if hebbian:
        strengthened = hebbian.get('strengthened_count', 0)
        max_change = hebbian.get('max_change', 0)
        cards.append(_card(
            'Hebbian Learning', f'{strengthened:,}',
            f'strengthened synapses • Δmax={max_change:.6f}',
            '251,191,36', '🧬'))

    html = f"""
    <div style="display:flex;gap:16px;flex-wrap:wrap;margin:16px 0;">
        {''.join(cards)}
    </div>
    """
    return html


# ═══════════════════════════════════════════════════════════════════
# Plotly Charts
# ═══════════════════════════════════════════════════════════════════

def _style(fig, title=''):
    fig.update_layout(
        title=title,
        paper_bgcolor='#111827', plot_bgcolor='#1a1f35',
        font=dict(color='#f1f5f9', size=12),
        xaxis=dict(gridcolor='#1e293b'),
        yaxis=dict(gridcolor='#1e293b'),
        legend=dict(bgcolor='#1a1f35'),
        margin=dict(l=50, r=30, t=50, b=40),
    )
    return fig


def _build_evolution_chart(df):
    """Build training evolution chart from interpretability_scores.csv."""
    if df is None or df.empty:
        return None

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Monosemanticity Score', 'Sparsity',
                        'Hebbian Δmax (synapse change)', 'Hub Neurons'],
        vertical_spacing=0.15, horizontal_spacing=0.1,
    )

    if 'mono_score' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['update'], y=df['mono_score'],
            mode='lines+markers', name='Mono Score',
            line=dict(color='#8b5cf6', width=2),
            marker=dict(size=6),
        ), row=1, col=1)

    if 'sparsity' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['update'], y=df['sparsity'],
            mode='lines+markers', name='Sparsity',
            line=dict(color='#00d4ff', width=2),
            marker=dict(size=6),
        ), row=1, col=2)

    if 'hebbian_max_change' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['update'], y=df['hebbian_max_change'],
            mode='lines+markers', name='Hebbian Δmax',
            line=dict(color='#fbbf24', width=2),
            marker=dict(size=6),
        ), row=2, col=1)

    if 'sf_hubs' in df.columns:
        fig.add_trace(go.Bar(
            x=df['update'], y=df['sf_hubs'],
            name='Hubs', marker_color='#34d399',
        ), row=2, col=2)

    fig.update_layout(
        height=500, showlegend=False,
        paper_bgcolor='#111827', plot_bgcolor='#1a1f35',
        font=dict(color='#f1f5f9'),
    )
    for r in range(1, 3):
        for c in range(1, 3):
            fig.update_xaxes(gridcolor='#1e293b', title_text='PPO Update', row=r, col=c)
            fig.update_yaxes(gridcolor='#1e293b', row=r, col=c)

    return fig


def _build_sparsity_chart(sparsity):
    """Build per-layer sparsity breakdown."""
    if not sparsity:
        return None

    details = sparsity.get('layer_details', [])
    if not details:
        return None

    names = [d['name'] for d in details[:30]]
    vals = [d['sparsity'] for d in details[:30]]
    sizes = [d['size'] for d in details[:30]]

    colors = ['#00d4ff' if v > 0.5 else '#f472b6' if v > 0.2 else '#34d399' for v in vals]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(len(names))),
        y=vals,
        text=[f'{v:.1%}' for v in vals],
        textposition='outside',
        marker_color=colors,
        hovertext=[f'{n}: {v:.3f} ({s:,} params)' for n, v, s in zip(names, vals, sizes)],
    ))
    fig.update_layout(
        xaxis=dict(tickvals=list(range(len(names))), ticktext=names, tickangle=-45),
        yaxis=dict(title='Sparsity', range=[0, 1]),
    )
    _style(fig, 'Layer-wise Activation Sparsity')
    fig.update_layout(height=350)
    return fig


def _build_degree_chart(scale_free):
    """Build degree distribution histogram."""
    if not scale_free:
        return None

    degrees = scale_free.get('degrees', [])
    if not degrees:
        return None

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=degrees,
        nbinsx=30,
        marker_color='#8b5cf6',
        opacity=0.8,
    ))

    hub_threshold = np.percentile(degrees, 90)
    fig.add_vline(x=hub_threshold, line_dash='dash', line_color='#ef4444',
                  annotation_text='Hub threshold (90th pctl)',
                  annotation_font_color='#ef4444')

    _style(fig, 'Neuron Degree Distribution')
    fig.update_layout(
        xaxis_title='Degree (connections)',
        yaxis_title='Count',
        height=350,
    )
    return fig


def _build_concepts_chart(mono):
    """Build concepts discovered chart."""
    if not mono:
        return None

    concepts = mono.get('concepts', [])
    if not concepts:
        return None

    neurons = mono.get('neurons', {})
    concept_counts = {}
    for neuron_id, associations in neurons.items():
        for assoc in associations:
            c = assoc.get('concept', '')
            concept_counts[c] = concept_counts.get(c, 0) + 1

    if not concept_counts:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=concepts, y=[1] * len(concepts),
            marker_color='#8b5cf6',
        ))
        _style(fig, f'Concepts Discovered ({len(concepts)} total)')
        fig.update_layout(height=300, xaxis_tickangle=-45)
        return fig

    sorted_concepts = sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)[:15]
    names = [c[0] for c in sorted_concepts]
    counts = [c[1] for c in sorted_concepts]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names, y=counts,
        marker=dict(color=counts, colorscale='Viridis'),
        text=counts, textposition='outside',
    ))
    _style(fig, f'Neurons per O-RAN Concept ({len(concept_counts)} concepts)')
    fig.update_layout(height=350, xaxis_tickangle=-45)
    return fig


def _build_monosemantic_table(mono):
    """Build a DataFrame for top monosemantic neurons."""
    if not mono or 'neurons' not in mono:
        return pd.DataFrame(columns=["Neuron ID", "Concept", "Correlation Score"])

    rows = []
    for nid, concepts in mono.get('neurons', {}).items():
        for c in concepts:
            rows.append({
                "Neuron ID": f"N{nid}",
                "Concept": c.get('concept', ''),
                "Correlation Score": round(c.get('correlation', 0.0), 4)
            })
    
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by="Correlation Score", ascending=False).head(50)
    else:
        df = pd.DataFrame(columns=["Neuron ID", "Concept", "Correlation Score"])
    return df


def _build_temporal_cell_saliency(saliency):
    """Build bar charts for average cell and frame importance."""
    if not saliency or 'avg_cell_importance' not in saliency:
        return _empty_fig("No Cell Data"), _empty_fig("No Frame Data")

    cell_vals = saliency.get('avg_cell_importance', [])
    cells = [f"Cell {i}" for i in range(len(cell_vals))]
    fig_cell = go.Figure(data=go.Bar(x=cells, y=cell_vals, marker_color='#f472b6'))
    _style(fig_cell, 'Average Cell Influence (Which cell drives actions?)')
    fig_cell.update_layout(height=350)

    frame_vals = saliency.get('avg_frame_importance', [])
    frames = ['t-2 (oldest)', 't-1 (prev)', 't-0 (current)'][:len(frame_vals)]
    fig_frame = go.Figure(data=go.Bar(x=frames, y=frame_vals, marker_color='#34d399'))
    _style(fig_frame, 'Average Temporal Influence (Which frames matter?)')
    fig_frame.update_layout(height=350)

    return fig_cell, fig_frame


# ═══════════════════════════════════════════════════════════════════
# Main Tab Builder
# ═══════════════════════════════════════════════════════════════════

def build_interpretability_tab():
    """Build the 🧠 BDH Interpretability tab for Gradio."""

    with gr.Tab("🧠 BDH Interpretability"):
        gr.Markdown("""
        ## 🌳 BDH Neural Architecture & Interpretability
        *Visualize the brain of the O-RAN agent — neurons, synapses, and what drives each decision.*
        """)

        load_btn = gr.Button("🔄 Load / Refresh Analysis Results", variant="primary", size="lg")
        status_box = gr.Textbox(label="Status", interactive=False, visible=True)

        # Score cards
        score_html = gr.HTML(label="Interpretability Scores")

        # Network graph (Plotly — works natively in Gradio)
        network_plot = gr.Plot(label="BDH Neural Architecture")

        # Charts
        with gr.Row():
            with gr.Column():
                sparsity_plot = gr.Plot(label="Layer Sparsity")
            with gr.Column():
                degree_plot = gr.Plot(label="Degree Distribution")

        concepts_plot = gr.Plot(label="O-RAN Concepts")
        
        saliency_plot = gr.Plot(label="Feature Saliency Attention")
        
        with gr.Row():
            with gr.Column():
                cell_saliency_plot = gr.Plot(label="Cell Influence")
            with gr.Column():
                frame_saliency_plot = gr.Plot(label="Temporal Influence")
                
        gr.Markdown("### Top Monosemantic Neurons & Concepts")
        monosemantic_table = gr.Dataframe(label="Monosemanticity Correlations", interactive=False)
        
        evolution_plot = gr.Plot(label="Training Evolution")

        # Detail viewer
        with gr.Accordion("📋 Raw Analysis JSON", open=False):
            detail_dd = gr.Dropdown(
                label="Select analysis file",
                choices=['monosemanticity.json', 'sparsity.json', 'scale_free.json', 'hebbian.json']
                        + _list_analysis_updates(),
                interactive=True,
            )
            detail_json = gr.JSON(label="Raw Data")

        def load_results():
            try:
                # 1. Try to load latest valid live training update
                latest_update = None
                updates = _list_analysis_updates()
                if updates:
                    import re
                    # Sort numerically (e.g. handle 'bdh_results/update_focus_5.json')
                    def _get_num(name):
                        match = re.search(r'_(?:focus_)?(\d+)\.json$', name)
                        if match:
                            return int(match.group(1))
                        return -1
                    updates.sort(key=_get_num)
                    latest = updates[-1]
                    
                    latest_path = INTERP_DIR / latest
                    if not latest_path.exists():
                        latest_path = LOGS_DIR / latest
                        
                    if latest_path.exists():
                        with open(latest_path) as f:
                            latest_update = json.load(f)

                if latest_update:
                    mono = latest_update.get('monosemanticity')
                    sparsity = latest_update.get('sparsity')
                    scale_free = latest_update.get('scale_free')
                    hebbian = latest_update.get('hebbian')
                    saliency = latest_update.get('saliency')
                else:
                    mono = None
                    sparsity = None
                    scale_free = None
                    hebbian = None
                    saliency = None
                    
                # Fallbacks to offline bdh_results/ if any is missing
                if not mono: mono = _load_json('monosemanticity.json')
                if not sparsity: sparsity = _load_json('sparsity.json')
                if not scale_free: scale_free = _load_json('scale_free.json')
                if not hebbian: hebbian = _load_json('hebbian.json')
                if not saliency: saliency = _load_json('saliency.json')

                scores_df = _load_scores_csv()

                found = sum(1 for x in [mono, sparsity, scale_free, hebbian, saliency] if x)
                if found == 0:
                    return ("⚠️ No interpretability telemetry found. Please run an analysis script first:\n\n"
                            "  [Live Engine] python3 -m interpretability.run_analysis --checkpoint YOUR.pt --focus-duration 15.0\n"
                            "  [Static Engine]   python3 -m interpretability.run_analysis --checkpoint YOUR.pt --generate-states 500",
                            "", None, None, None, None, None)

                # Score cards
                cards = _build_score_cards(mono, sparsity, scale_free, hebbian)

                # Network graph (Plotly)
                net_fig = _build_network_plotly(scale_free, sparsity)

                # Charts
                sp_fig = _build_sparsity_chart(sparsity)
                deg_fig = _build_degree_chart(scale_free)
                con_fig = _build_concepts_chart(mono)
                sal_fig = _build_saliency_heatmap(saliency)
                cell_sal_fig, frame_sal_fig = _build_temporal_cell_saliency(saliency)
                mono_df = _build_monosemantic_table(mono)
                evo_fig = _build_evolution_chart(scores_df)

                status = f"✅ Loaded {found}/5 analyses"
                if scores_df is not None:
                    status += f" + {len(scores_df)} training snapshots"

                return status, cards, net_fig, sp_fig, deg_fig, con_fig, sal_fig, cell_sal_fig, frame_sal_fig, mono_df, evo_fig

            except Exception as e:
                return f"❌ Error: {e}\n{traceback.format_exc()}", "", None, None, None, None, None, None, None, None, None

        def load_detail(name):
            if not name:
                return None
            p = BDH_RESULTS / name
            if not p.exists():
                p = INTERP_DIR / name
            if not p.exists():
                p = LOGS_DIR / name
            if p.exists():
                with open(p) as f:
                    return json.load(f)
            return {"error": f"File not found: {name}"}

        load_btn.click(
            load_results,
            outputs=[status_box, score_html, network_plot,
                     sparsity_plot, degree_plot, concepts_plot, saliency_plot,
                     cell_saliency_plot, frame_saliency_plot, monosemantic_table, evolution_plot],
        )
        detail_dd.change(load_detail, inputs=[detail_dd], outputs=[detail_json])

        # Live Auto-Refresh (every 5 seconds)
        timer = gr.Timer(5)
        timer.tick(
            load_results,
            outputs=[status_box, score_html, network_plot,
                     sparsity_plot, degree_plot, concepts_plot, saliency_plot,
                     cell_saliency_plot, frame_saliency_plot, monosemantic_table, evolution_plot],
        )
