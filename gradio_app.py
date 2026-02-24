"""
Radio-Cortex Gradio Dashboard
==============================
Interactive web UI for the BDH RL agent.

Tabs:
  1. Model Inference  — Load a .pt checkpoint, input cell states, see predicted actions
  2. Reward Metrics   — Upload reward_metrics_*.csv, explore interactive Plotly charts
  3. Model Comparison — Compare two BDH checkpoints side-by-side on the same state

Launch:
  python gradio_app.py
  # or
  python gradio_app.py --port 7860 --share
"""

import os
import sys
import glob
import argparse
import warnings
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Interpretability tab (vis.js neural graph + score cards) ──────────────────
from interpretability.gradio_tab import build_interpretability_tab

# ── Add project root to path so we can import policies ───────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Constants ─────────────────────────────────────────────────────────────────
MODELS_DIR = ROOT / "models"
LOGS_DIR   = ROOT / "logs"
RESULTS_DIR= ROOT / "results"

CELL_FEATURES = 12
CELL_ACTIONS  = 2
DEFAULT_CELLS = 3

FEATURE_NAMES = [
    "RSRP (dBm)", "SINR (dB)", "RB Utilization", "Tx Power (dBm)", "CQI",
    "UE Count", "Avg Throughput (Mbps)", "Avg Delay (ms)", "Avg Loss",
    "Avg RSRP (dBm)", "Avg SINR (dB)", "Jain's Fairness",
]
ACTION_NAMES = ["TxPower", "Handover Sensitivity"]

REWARD_COMPONENTS = ["r_tput", "r_delay", "r_loss", "r_load", "r_energy", "r_sla", "r_cio"]
COMPONENT_COLORS  = ["#00d4ff", "#f87171", "#fbbf24", "#f472b6", "#34d399", "#a78bfa", "#6366f1"]

# ── Model loading helpers ─────────────────────────────────────────────────────

def list_models():
    """Return sorted list of .pt model files in models/."""
    pts = sorted(glob.glob(str(MODELS_DIR / "*.pt")))
    return [os.path.basename(p) for p in pts] or ["(no models found)"]


def load_policy(model_name: str, num_cells: int = DEFAULT_CELLS, device: str = "cpu"):
    """Load a BDHPolicy checkpoint from models/<model_name>. Auto-detects dimensions."""
    from policies.bdh_policy import BDHPolicy
    from oran_ns3_env import NS3Config

    model_path = MODELS_DIR / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    ckpt = torch.load(str(model_path), map_location=device, weights_only=False)

    # Support both raw state_dict and wrapped checkpoint formats
    if isinstance(ckpt, dict):
        sd = ckpt.get("policy_state_dict") or ckpt.get("state_dict") or ckpt
    else:
        sd = ckpt

    # ── Auto-detect dimensions from checkpoint weights ──
    # logstd_head shape → action_dim
    logstd_key = next((k for k in sd if 'logstd' in k), None)
    action_key = next((k for k in sd if 'action_head' in k and 'bias' in k), None)

    if logstd_key:
        action_dim = int(sd[logstd_key].shape[-1])
    elif action_key:
        action_dim = int(sd[action_key].shape[0])
    else:
        action_dim = num_cells * 3

    # Derive num_cells from action_dim (3 actions per cell)
    detected_cells = max(action_dim // 3, 1)

    # frame_encoder weight shape → features per cell
    frame_enc_key = next((k for k in sd if 'frame_encoder' in k and 'weight' in k), None)
    if frame_enc_key:
        feat_per_cell = int(sd[frame_enc_key].shape[1])
        state_dim = detected_cells * feat_per_cell * 3  # 3 temporal frames
    else:
        state_dim = detected_cells * 16 * 3

    cfg = NS3Config()
    cfg.num_cells = detected_cells

    policy = BDHPolicy(state_dim=state_dim, action_dim=action_dim, device=device, env_config=cfg)
    policy = policy.to(device)
    policy.load_state_dict(sd, strict=False)
    policy.eval()
    return policy


def run_inference(policy, state_np: np.ndarray, deterministic: bool = True, device: str = "cpu"):
    """Run a single forward pass. Returns (actions, value, action_mean, action_std)."""
    with torch.no_grad():
        state_t = torch.FloatTensor(state_np).unsqueeze(0).to(device)
        action_mean, logstd, value = policy._forward_common(state_t)
        action_std = torch.exp(logstd)

        if deterministic:
            actions = action_mean
        else:
            dist = torch.distributions.Normal(action_mean, action_std)
            actions = dist.sample()

    actions_np = actions.squeeze(0).cpu().numpy()
    mean_np    = action_mean.squeeze(0).cpu().numpy()
    std_np     = action_std.squeeze(0).cpu().numpy()
    value_f    = value.squeeze().item()
    return actions_np, value_f, mean_np, std_np


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Model Inference
# ─────────────────────────────────────────────────────────────────────────────

def build_inference_tab():
    with gr.Tab("🤖 Model Inference"):
        gr.Markdown("""
        ## Model Inference
        Load a BDH checkpoint and feed in cell-level state features to see the agent's predicted actions and value estimate.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                model_dd = gr.Dropdown(
                    label="Select Model Checkpoint",
                    choices=list_models(),
                    value=list_models()[0],
                    interactive=True,
                )
                refresh_btn = gr.Button("🔄 Refresh Model List", size="sm")
                num_cells_sl = gr.Slider(1, 6, value=3, step=1, label="Number of Cells")
                deterministic_cb = gr.Checkbox(value=True, label="Deterministic Action (uncheck to sample)")

            with gr.Column(scale=2):
                gr.Markdown("### Cell State Features")
                gr.Markdown("*Adjust sliders for each cell (12 features × N cells)*")
                state_inputs = []
                for c in range(DEFAULT_CELLS):
                    with gr.Accordion(f"Cell {c}", open=(c == 0)):
                        row_inputs = []
                        for f_idx, fname in enumerate(FEATURE_NAMES):
                            lo, hi, default = _feature_range(f_idx)
                            sl = gr.Slider(lo, hi, value=default, label=fname, step=(hi - lo) / 100)
                            row_inputs.append(sl)
                        state_inputs.extend(row_inputs)

        run_btn = gr.Button("▶ Run Inference", variant="primary")

        with gr.Row():
            value_box = gr.Textbox(label="Value Estimate (V)", interactive=False)
            status_box = gr.Textbox(label="Status", interactive=False)

        actions_plot = gr.Plot(label="Predicted Actions per Cell")
        actions_table = gr.Dataframe(
            label="Action Details",
            headers=["Cell"] + ACTION_NAMES,
            interactive=False,
        )

        def refresh_models():
            return gr.Dropdown(choices=list_models())

        def do_inference(model_name, num_cells, deterministic, *slider_vals):
            try:
                policy = load_policy(model_name, num_cells=int(num_cells))

                # Detect dimensions from the loaded policy
                bdh = policy.bdh if hasattr(policy, 'bdh') else policy
                detected_cells = bdh.config.num_cells if hasattr(bdh, 'config') else int(num_cells)
                feat_per_cell = policy.frame_dim if hasattr(policy, 'frame_dim') else 16
                n_frames = 3
                model_state_dim = detected_cells * feat_per_cell * n_frames

                # Map 12 slider features → 16 model features per cell
                # Slider: RSRP, SINR, RB_Util, TxPower, CQI, UE_Count,
                #         AvgTput, AvgDelay, AvgLoss, AvgRSRP, AvgSINR, Jains
                # Model:  queue, rb_util, tx_power, load, avg_req,
                #         avg_tput, avg_delay, avg_loss, max_delay, max_loss,
                #         jains, n_ues, sinr_mean, ho_count, cqi_mean, stationary
                one_frame = []
                for c in range(detected_cells):
                    sl_offset = c * CELL_FEATURES
                    sl = list(slider_vals[sl_offset:sl_offset + CELL_FEATURES])
                    while len(sl) < CELL_FEATURES:
                        sl.append(0.0)

                    # Build 16-feature vector (normalized)
                    cell_feats = [
                        0.1,                           # queue (normalized)
                        sl[2] if len(sl) > 2 else 0.5, # rb_util
                        (sl[3] - 10) / 36 if len(sl) > 3 else 0.36,  # tx_power (norm)
                        0.2,                           # load (normalized)
                        0.3,                           # avg_req (default)
                        sl[6] / 100 if len(sl) > 6 else 0.05,  # avg_tput (norm)
                        sl[7] / 100 if len(sl) > 7 else 0.3,   # avg_delay (norm)
                        sl[8] if len(sl) > 8 else 0.05,        # avg_loss
                        sl[7] / 80 if len(sl) > 7 else 0.4,    # max_delay (approx)
                        sl[8] if len(sl) > 8 else 0.05,        # max_loss
                        sl[11] if len(sl) > 11 else 0.8,       # jains
                        sl[5] / 50 if len(sl) > 5 else 0.4,    # n_ues (norm)
                        sl[1] / 30 if len(sl) > 1 else 0.5,    # sinr_mean (norm)
                        0.0,                           # ho_count
                        sl[4] / 15 if len(sl) > 4 else 0.53,   # cqi_mean (norm)
                        1.0,                           # stationary flag
                    ]
                    one_frame.extend(cell_feats[:feat_per_cell])

                # Tile across 3 temporal frames (t-2, t-1, t-0)
                state = np.array(one_frame * n_frames, dtype=np.float32)

                # Truncate or pad to exact model_state_dim
                if len(state) > model_state_dim:
                    state = state[:model_state_dim]
                elif len(state) < model_state_dim:
                    state = np.pad(state, (0, model_state_dim - len(state)))

                actions, value, mean, std = run_inference(policy, state, deterministic=deterministic)

                # Reshape by detected dimensions
                act_per_cell = 3  # TxPower, CIO, TTT
                n = detected_cells
                act_names = ["TxPower", "CIO", "TTT"]
                actions_2d = actions.reshape(n, act_per_cell)
                mean_2d    = mean.reshape(n, act_per_cell)
                std_2d     = std.reshape(n, act_per_cell)

                # Build table
                rows = []
                for ci in range(n):
                    rows.append([f"Cell {ci}"] + [f"{v:.4f}" for v in actions_2d[ci]])
                df = pd.DataFrame(rows, columns=["Cell"] + act_names)

                # Build bar chart
                fig = go.Figure()
                for ci in range(n):
                    fig.add_trace(go.Bar(
                        name=f"Cell {ci}",
                        x=act_names,
                        y=actions_2d[ci].tolist(),
                        error_y=dict(type="data", array=std_2d[ci].tolist(), visible=True),
                    ))
                fig.update_layout(
                    barmode="group",
                    title="Predicted Actions (with ±1σ uncertainty)",
                    paper_bgcolor="#111827",
                    plot_bgcolor="#1a1f35",
                    font=dict(color="#f1f5f9"),
                    xaxis=dict(gridcolor="#1e293b"),
                    yaxis=dict(gridcolor="#1e293b", title="Action Value [0–1]"),
                    legend=dict(bgcolor="#1a1f35"),
                )

                return f"{value:.4f}", f"✅ Inference OK ({detected_cells} cell, state_dim={model_state_dim})", fig, df

            except Exception as e:
                return "—", f"❌ Error: {e}\n{traceback.format_exc()}", None, None

        refresh_btn.click(refresh_models, outputs=[model_dd])
        run_btn.click(
            do_inference,
            inputs=[model_dd, num_cells_sl, deterministic_cb] + state_inputs,
            outputs=[value_box, status_box, actions_plot, actions_table],
        )

    return model_dd, num_cells_sl, state_inputs


def _feature_range(f_idx):
    """Return (lo, hi, default) for each of the 12 cell features."""
    ranges = [
        (-140, -44, -90),   # RSRP
        (-10,  40,  15),    # SINR
        (0,    1,   0.5),   # RB Util
        (0,    46,  23),    # Tx Power
        (1,    15,  8),     # CQI
        (0,    200, 20),    # UE Count
        (0,    100, 5),     # Avg Throughput
        (0,    500, 30),    # Avg Delay
        (0,    1,   0.05),  # Avg Loss
        (-140, -44, -90),   # Avg RSRP
        (-10,  40,  15),    # Avg SINR
        (0,    1,   0.8),   # Jain's Fairness
    ]
    return ranges[f_idx]


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Reward Metrics Viewer
# ─────────────────────────────────────────────────────────────────────────────

def build_metrics_tab():
    with gr.Tab("📊 Reward Metrics Viewer"):
        gr.Markdown("""
        ## Reward Metrics Dashboard
        Upload a `reward_metrics_*.csv` file (from `logs/`) or select one from the logs directory.
        """)

        with gr.Row():
            csv_file = gr.File(label="Upload reward_metrics_*.csv", file_types=[".csv"])
            with gr.Column():
                log_dd = gr.Dropdown(
                    label="Or select from logs/",
                    choices=_list_log_csvs(),
                    interactive=True,
                )
                load_log_btn = gr.Button("📂 Load Selected Log", size="sm")
                refresh_log_btn = gr.Button("🔄 Refresh Log List", size="sm")

        load_status = gr.Textbox(label="Status", interactive=False)

        with gr.Row():
            stat_df = gr.Dataframe(label="Summary Statistics", interactive=False)

        reward_plot   = gr.Plot(label="Total Reward Over Time")
        breakdown_plot = gr.Plot(label="Reward Component Breakdown")
        kpi_plot      = gr.Plot(label="KPI Trends (Throughput / Delay / Loss)")
        success_plot  = gr.Plot(label="UE Success Rate")

        def _list_logs_refresh():
            return gr.Dropdown(choices=_list_log_csvs())

        def load_from_file(f):
            if f is None:
                return "No file uploaded.", None, None, None, None, None
            return _process_csv(f.name)

        def load_from_dropdown(log_name):
            if not log_name:
                return "No log selected.", None, None, None, None, None
            path = LOGS_DIR / log_name
            return _process_csv(str(path))

        refresh_log_btn.click(_list_logs_refresh, outputs=[log_dd])
        csv_file.change(load_from_file, inputs=[csv_file],
                        outputs=[load_status, stat_df, reward_plot, breakdown_plot, kpi_plot, success_plot])
        load_log_btn.click(load_from_dropdown, inputs=[log_dd],
                           outputs=[load_status, stat_df, reward_plot, breakdown_plot, kpi_plot, success_plot])


def _list_log_csvs():
    csvs = sorted(glob.glob(str(LOGS_DIR / "reward_metrics_*.csv")))
    return [os.path.basename(p) for p in csvs] or ["(no logs found)"]


def _process_csv(path: str):
    """Load a reward_metrics CSV and return all chart figures + stats."""
    try:
        df = pd.read_csv(path)
        if df.empty:
            return "CSV is empty.", None, None, None, None, None

        status = f"✅ Loaded {len(df)} rows from {os.path.basename(path)}"

        # ── Summary stats ──────────────────────────────────────────────────
        numeric_cols = ["reward", "r_tput", "r_delay", "r_loss", "r_load", "r_energy", "r_sla", "r_cio",
                        "avg_throughput", "avg_delay", "avg_loss", "z_success"]
        available = [c for c in numeric_cols if c in df.columns]
        stats = df[available].describe().round(4).reset_index()
        stats.rename(columns={"index": "Statistic"}, inplace=True)

        # ── Plot 1: Total reward ───────────────────────────────────────────
        fig_reward = go.Figure()
        fig_reward.add_trace(go.Scatter(
            x=df.index, y=df["reward"],
            mode="lines", name="Total Reward",
            line=dict(color="#00d4ff", width=1.5),
        ))
        # Rolling mean
        if len(df) > 20:
            roll = df["reward"].rolling(20, min_periods=1).mean()
            fig_reward.add_trace(go.Scatter(
                x=df.index, y=roll,
                mode="lines", name="Rolling Mean (20)",
                line=dict(color="#f472b6", width=2, dash="dash"),
            ))
        _style_fig(fig_reward, "Total Reward Over Time", "Step", "Reward")

        # ── Plot 2: Component breakdown (stacked area) ─────────────────────
        available_comps = [c for c in REWARD_COMPONENTS if c in df.columns]
        fig_break = go.Figure()
        for i, comp in enumerate(available_comps):
            fig_break.add_trace(go.Scatter(
                x=df.index, y=df[comp],
                mode="lines", name=comp,
                line=dict(color=COMPONENT_COLORS[i % len(COMPONENT_COLORS)], width=1.5),
                stackgroup=None,
            ))
        _style_fig(fig_break, "Reward Components Over Time", "Step", "Value")

        # ── Plot 3: KPI trends ─────────────────────────────────────────────
        fig_kpi = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                subplot_titles=["Throughput (Mbps)", "Delay (ms)", "Packet Loss"])
        kpi_map = [
            ("avg_throughput", "#00d4ff", 1),
            ("avg_delay",      "#fbbf24", 2),
            ("avg_loss",       "#f87171", 3),
        ]
        for col, color, row in kpi_map:
            if col in df.columns:
                fig_kpi.add_trace(go.Scatter(
                    x=df.index, y=df[col],
                    mode="lines", name=col,
                    line=dict(color=color, width=1.5),
                ), row=row, col=1)
        fig_kpi.update_layout(
            paper_bgcolor="#111827", plot_bgcolor="#1a1f35",
            font=dict(color="#f1f5f9"), showlegend=False,
            height=500, title_text="KPI Trends",
        )
        for i in range(1, 4):
            fig_kpi.update_xaxes(gridcolor="#1e293b", row=i, col=1)
            fig_kpi.update_yaxes(gridcolor="#1e293b", row=i, col=1)

        # ── Plot 4: Success Rate ───────────────────────────────────────────
        fig_success = go.Figure()
        if "z_success" in df.columns:
            fig_success.add_trace(go.Scatter(
                x=df.index, y=df["z_success"],
                mode="lines", name="Success Rate",
                line=dict(color="#34d399", width=1.5),
            ))
            if len(df) > 20:
                roll_s = df["z_success"].rolling(20, min_periods=1).mean()
                fig_success.add_trace(go.Scatter(
                    x=df.index, y=roll_s,
                    mode="lines", name="Rolling Mean (20)",
                    line=dict(color="#8b5cf6", width=2, dash="dash"),
                ))
        _style_fig(fig_success, "UE Success Rate (Tput > 1 Mbps)", "Step", "Ratio")

        return status, stats, fig_reward, fig_break, fig_kpi, fig_success

    except Exception as e:
        return f"❌ Error: {e}\n{traceback.format_exc()}", None, None, None, None, None


def _style_fig(fig, title, xlab, ylab):
    fig.update_layout(
        title=title,
        paper_bgcolor="#111827",
        plot_bgcolor="#1a1f35",
        font=dict(color="#f1f5f9"),
        xaxis=dict(gridcolor="#1e293b", title=xlab),
        yaxis=dict(gridcolor="#1e293b", title=ylab),
        legend=dict(bgcolor="#1a1f35"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Model Comparison
# ─────────────────────────────────────────────────────────────────────────────

def build_comparison_tab():
    with gr.Tab("⚖️ Model Comparison"):
        gr.Markdown("""
        ## Side-by-Side Model Comparison
        Compare two BDH checkpoints on the same cell state input.
        """)

        with gr.Row():
            model_a_dd = gr.Dropdown(label="Model A", choices=list_models(), value=list_models()[0], interactive=True)
            model_b_dd = gr.Dropdown(label="Model B", choices=list_models(),
                                     value=list_models()[-1] if len(list_models()) > 1 else list_models()[0],
                                     interactive=True)

        with gr.Row():
            num_cells_cmp = gr.Slider(1, 6, value=3, step=1, label="Number of Cells")
            det_cmp = gr.Checkbox(value=True, label="Deterministic")

        gr.Markdown("### Shared Cell State Input")
        cmp_state_inputs = []
        for c in range(DEFAULT_CELLS):
            with gr.Accordion(f"Cell {c}", open=(c == 0)):
                for f_idx, fname in enumerate(FEATURE_NAMES):
                    lo, hi, default = _feature_range(f_idx)
                    sl = gr.Slider(lo, hi, value=default, label=fname, step=(hi - lo) / 100)
                    cmp_state_inputs.append(sl)

        cmp_btn = gr.Button("▶ Compare Models", variant="primary")
        cmp_status = gr.Textbox(label="Status", interactive=False)

        with gr.Row():
            cmp_plot = gr.Plot(label="Action Comparison")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Model A Actions")
                table_a = gr.Dataframe(headers=["Cell"] + ACTION_NAMES, interactive=False)
            with gr.Column():
                gr.Markdown("### Model B Actions")
                table_b = gr.Dataframe(headers=["Cell"] + ACTION_NAMES, interactive=False)

        with gr.Row():
            val_a_box = gr.Textbox(label="Model A — Value Estimate", interactive=False)
            val_b_box = gr.Textbox(label="Model B — Value Estimate", interactive=False)

        diff_plot = gr.Plot(label="Action Difference (A − B)")

        def do_comparison(model_a, model_b, num_cells, deterministic, *slider_vals):
            try:
                n = int(num_cells)
                state = np.array(slider_vals[:n * CELL_FEATURES], dtype=np.float32)

                pol_a = load_policy(model_a, num_cells=n)
                pol_b = load_policy(model_b, num_cells=n)

                acts_a, val_a, mean_a, std_a = run_inference(pol_a, state, deterministic)
                acts_b, val_b, mean_b, std_b = run_inference(pol_b, state, deterministic)

                acts_a_2d = acts_a.reshape(n, CELL_ACTIONS)
                acts_b_2d = acts_b.reshape(n, CELL_ACTIONS)

                def _make_table(acts_2d):
                    rows = [[f"Cell {ci}"] + [f"{v:.4f}" for v in acts_2d[ci]] for ci in range(n)]
                    return pd.DataFrame(rows, columns=["Cell"] + ACTION_NAMES)

                # Grouped bar chart
                fig = go.Figure()
                colors_a = "#00d4ff"
                colors_b = "#f472b6"
                for ci in range(n):
                    fig.add_trace(go.Bar(
                        name=f"A·Cell{ci}", x=ACTION_NAMES,
                        y=acts_a_2d[ci].tolist(),
                        marker_color=colors_a, opacity=0.85,
                        legendgroup=f"cell{ci}",
                    ))
                    fig.add_trace(go.Bar(
                        name=f"B·Cell{ci}", x=ACTION_NAMES,
                        y=acts_b_2d[ci].tolist(),
                        marker_color=colors_b, opacity=0.85,
                        legendgroup=f"cell{ci}",
                    ))
                fig.update_layout(
                    barmode="group",
                    title="Model A vs Model B — Predicted Actions",
                    paper_bgcolor="#111827", plot_bgcolor="#1a1f35",
                    font=dict(color="#f1f5f9"),
                    xaxis=dict(gridcolor="#1e293b"),
                    yaxis=dict(gridcolor="#1e293b", title="Action Value"),
                    legend=dict(bgcolor="#1a1f35"),
                )

                # Difference plot
                diff = acts_a_2d - acts_b_2d
                fig_diff = go.Figure()
                for ci in range(n):
                    colors = ["#34d399" if v >= 0 else "#f87171" for v in diff[ci]]
                    fig_diff.add_trace(go.Bar(
                        name=f"Cell {ci}", x=ACTION_NAMES,
                        y=diff[ci].tolist(),
                        marker_color=colors,
                    ))
                fig_diff.update_layout(
                    barmode="group",
                    title="Action Difference (A − B)",
                    paper_bgcolor="#111827", plot_bgcolor="#1a1f35",
                    font=dict(color="#f1f5f9"),
                    xaxis=dict(gridcolor="#1e293b"),
                    yaxis=dict(gridcolor="#1e293b", title="Δ Action"),
                    legend=dict(bgcolor="#1a1f35"),
                )

                status = f"✅ Compared {model_a} vs {model_b}"
                return (status, fig,
                        _make_table(acts_a_2d), _make_table(acts_b_2d),
                        f"{val_a:.4f}", f"{val_b:.4f}",
                        fig_diff)

            except Exception as e:
                return f"❌ {e}\n{traceback.format_exc()}", None, None, None, "—", "—", None

        cmp_btn.click(
            do_comparison,
            inputs=[model_a_dd, model_b_dd, num_cells_cmp, det_cmp] + cmp_state_inputs,
            outputs=[cmp_status, cmp_plot, table_a, table_b, val_a_box, val_b_box, diff_plot],
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Evaluation Results
# ─────────────────────────────────────────────────────────────────────────────

def build_evaluation_tab():
    with gr.Tab("🏆 Evaluation Results"):
        gr.Markdown("""
        ## Scenario Evaluation Results
        Upload an `eval_results_*.csv` file (from `results/`) or select one from the directory to view scenario-level performance.
        """)
        
        with gr.Row():
            eval_csv_file = gr.File(label="Upload eval_results_*.csv", file_types=[".csv"])
            with gr.Column():
                eval_log_dd = gr.Dropdown(
                    label="Or select from results/",
                    choices=_list_eval_csvs(),
                    interactive=True,
                )
                load_eval_btn = gr.Button("📂 Load Selected Evaluation", size="sm")
                refresh_eval_btn = gr.Button("🔄 Refresh Eval List", size="sm")
                
        eval_status = gr.Textbox(label="Status", interactive=False)
        
        with gr.Row():
            eval_df = gr.Dataframe(label="Evaluation Summary Table", interactive=False)
            
        score_plot = gr.Plot(label="Overall Score by Scenario")
        
        with gr.Row():
            tput_plot = gr.Plot()
            loss_plot = gr.Plot()
            
        with gr.Row():
            satisf_plot = gr.Plot()
            eneff_plot = gr.Plot()
            
        def _list_evals_refresh():
            return gr.Dropdown(choices=_list_eval_csvs())
            
        def load_eval_from_file(f):
            if f is None:
                return "No file uploaded.", None, None, None, None, None, None
            return _process_eval_csv(f.name)
            
        def load_eval_from_dropdown(log_name):
            if not log_name or "(no eval logs found)" in log_name:
                return "No log selected.", None, None, None, None, None, None
            path = RESULTS_DIR / log_name
            return _process_eval_csv(str(path))
            
        refresh_eval_btn.click(_list_evals_refresh, outputs=[eval_log_dd])
        eval_csv_file.change(load_eval_from_file, inputs=[eval_csv_file],
                             outputs=[eval_status, eval_df, score_plot, tput_plot, loss_plot, satisf_plot, eneff_plot])
        load_eval_btn.click(load_eval_from_dropdown, inputs=[eval_log_dd],
                            outputs=[eval_status, eval_df, score_plot, tput_plot, loss_plot, satisf_plot, eneff_plot])


def _list_eval_csvs():
    if not RESULTS_DIR.exists():
        return ["(no eval logs found)"]
    csvs = sorted(glob.glob(str(RESULTS_DIR / "eval_results_*.csv")))
    return [os.path.basename(p) for p in csvs] or ["(no eval logs found)"]


def _process_eval_csv(path: str):
    try:
        if not os.path.exists(path):
            return f"❌ File not found: {path}", None, None, None, None, None, None
            
        with open(path, 'r') as f:
            lines = f.readlines()
            
        start_idx = -1
        for i, line in enumerate(lines):
            if "EVALUATION SUMMARY" in line:
                start_idx = i
                break
                
        if start_idx == -1:
            return "❌ No EVALUATION SUMMARY section found in the CSV.", None, None, None, None, None, None
            
        data = []
        current_scenario = None
        for line in lines[start_idx+2:]:
            line = line.strip().strip(',')
            if not line or "====" in line or "----" in line:
                continue
            if "Controller" in line:
                continue
            if line.endswith(":"):
                current_scenario = line[:-1]
            elif "|" in line:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 6 and current_scenario:
                    controller = parts[0]
                    try:
                        tput = float(parts[1])
                        loss = float(parts[2])
                        satisf = float(parts[3])
                        eneff = float(parts[4])
                        score = float(parts[5])
                    except ValueError:
                        continue
                        
                    data.append({
                        "Scenario": current_scenario,
                        "Controller": controller,
                        "Tput (Mbps)": tput,
                        "Loss (%)": loss,
                        "Satisf (%)": satisf,
                        "Energy Eff": eneff,
                        "Score": score
                    })
                    
        if not data:
            return "❌ No parsed data from EVALUATION SUMMARY.", None, None, None, None, None, None
            
        df = pd.DataFrame(data)
        status = f"✅ Loaded {len(df)} scenario evaluations from {os.path.basename(path)}"
        
        def _make_bar(df, metric, title, color):
            fig = go.Figure()
            for ctrl in df["Controller"].unique():
                cdf = df[df["Controller"] == ctrl]
                fig.add_trace(go.Bar(
                    name=ctrl,
                    x=cdf["Scenario"],
                    y=cdf[metric],
                    marker_color=color,
                ))
            fig.update_layout(
                barmode="group",
                title=title,
                paper_bgcolor="#111827", plot_bgcolor="#1a1f35",
                font=dict(color="#f1f5f9"),
                xaxis=dict(gridcolor="#1e293b", tickangle=-45),
                yaxis=dict(gridcolor="#1e293b", title=metric),
                legend=dict(bgcolor="#1a1f35"),
                margin=dict(b=100)
            )
            return fig

        fig_score = _make_bar(df, "Score", "Overall Score by Scenario", "#a78bfa")
        fig_tput = _make_bar(df, "Tput (Mbps)", "Throughput by Scenario", "#00d4ff")
        fig_loss = _make_bar(df, "Loss (%)", "Packet Loss by Scenario", "#f87171")
        fig_satisf = _make_bar(df, "Satisf (%)", "Satisfaction (%) by Scenario", "#34d399")
        fig_eneff = _make_bar(df, "Energy Eff", "Energy Efficiency by Scenario", "#fbbf24")
        
        return status, df, fig_score, fig_tput, fig_loss, fig_satisf, fig_eneff
        
    except Exception as e:
        return f"❌ Error: {e}\\n{traceback.format_exc()}", None, None, None, None, None, None


# ─────────────────────────────────────────────────────────────────────────────
# App Assembly
# ─────────────────────────────────────────────────────────────────────────────

def build_app():
    with gr.Blocks(
        title="Radio-Cortex Dashboard",
        theme=gr.themes.Base(
            primary_hue="cyan",
            secondary_hue="violet",
            neutral_hue="slate",
        ).set(
            body_background_fill="#0a0e1a",
            block_background_fill="#111827",
            block_border_color="#1e293b",
            input_background_fill="#1a1f35",
        ),
        css="""
        .gradio-container { max-width: 1400px !important; }
        h1, h2, h3 { color: #00d4ff !important; }
        .tab-nav button { font-weight: 600; }
        """
    ) as demo:
        gr.Markdown("""
        # 📡 Radio-Cortex — BDH Agent Dashboard
        *Interactive inference, reward visualization, and model comparison for the O-RAN RL agent.*
        """)

        build_inference_tab()
        build_metrics_tab()
        build_comparison_tab()
        build_evaluation_tab()
        build_interpretability_tab()

    return demo


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Radio-Cortex Gradio Dashboard")
    parser.add_argument("--port",  type=int,  default=7860,  help="Port to serve on")
    parser.add_argument("--share", action="store_true",       help="Create a public Gradio share link")
    parser.add_argument("--host",  type=str,  default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    demo = build_app()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)
