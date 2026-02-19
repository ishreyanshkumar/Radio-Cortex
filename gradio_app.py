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

# ── Add project root to path so we can import policies ───────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Constants ─────────────────────────────────────────────────────────────────
MODELS_DIR = ROOT / "models"
LOGS_DIR   = ROOT / "logs"

CELL_FEATURES = 12
CELL_ACTIONS  = 2
DEFAULT_CELLS = 3

FEATURE_NAMES = [
    "RSRP (dBm)", "SINR (dB)", "RB Utilization", "Tx Power (dBm)", "CQI",
    "UE Count", "Avg Throughput (Mbps)", "Avg Delay (ms)", "Avg Loss",
    "Avg RSRP (dBm)", "Avg SINR (dB)", "Jain's Fairness",
]
ACTION_NAMES = ["TxPower", "Handover Sensitivity"]

REWARD_COMPONENTS = ["r_tput", "r_delay", "r_loss", "r_se", "r_energy", "r_load", "r_queue", "r_smooth"]
COMPONENT_COLORS  = ["#00d4ff", "#f87171", "#fbbf24", "#34d399", "#8b5cf6", "#f472b6", "#60a5fa", "#a78bfa"]

# ── Model loading helpers ─────────────────────────────────────────────────────

def list_models():
    """Return sorted list of .pt model files in models/."""
    pts = sorted(glob.glob(str(MODELS_DIR / "*.pt")))
    return [os.path.basename(p) for p in pts] or ["(no models found)"]


def load_policy(model_name: str, num_cells: int = DEFAULT_CELLS, device: str = "cpu"):
    """Load a BDHPolicy checkpoint from models/<model_name>."""
    from policies.bdh_policy import BDHPolicy
    from oran_ns3_env import NS3Config

    model_path = MODELS_DIR / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    state_dim  = num_cells * CELL_FEATURES
    action_dim = num_cells * CELL_ACTIONS

    cfg = NS3Config()
    cfg.num_cells = num_cells

    policy = BDHPolicy(state_dim=state_dim, action_dim=action_dim, device=device, env_config=cfg)
    policy = policy.to(device)

    ckpt = torch.load(str(model_path), map_location=device, weights_only=False)

    # Support both raw state_dict and wrapped checkpoint formats
    if isinstance(ckpt, dict):
        sd = ckpt.get("policy_state_dict") or ckpt.get("state_dict") or ckpt
    else:
        sd = ckpt

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
                # Build state from sliders (only use first num_cells worth)
                n = int(num_cells)
                vals_per_cell = CELL_FEATURES
                state = np.array(slider_vals[:n * vals_per_cell], dtype=np.float32)

                policy = load_policy(model_name, num_cells=n)
                actions, value, mean, std = run_inference(policy, state, deterministic=deterministic)

                # Reshape to (num_cells, cell_actions)
                actions_2d = actions.reshape(n, CELL_ACTIONS)
                mean_2d    = mean.reshape(n, CELL_ACTIONS)
                std_2d     = std.reshape(n, CELL_ACTIONS)

                # Build table
                rows = []
                for ci in range(n):
                    rows.append([f"Cell {ci}"] + [f"{v:.4f}" for v in actions_2d[ci]])
                df = pd.DataFrame(rows, columns=["Cell"] + ACTION_NAMES)

                # Build bar chart
                fig = go.Figure()
                for ci in range(n):
                    fig.add_trace(go.Bar(
                        name=f"Cell {ci}",
                        x=ACTION_NAMES,
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

                return f"{value:.4f}", "✅ Inference successful", fig, df

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
        curriculum_plot = gr.Plot(label="Curriculum Level & Success Rate")

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
                        outputs=[load_status, stat_df, reward_plot, breakdown_plot, kpi_plot, curriculum_plot])
        load_log_btn.click(load_from_dropdown, inputs=[log_dd],
                           outputs=[load_status, stat_df, reward_plot, breakdown_plot, kpi_plot, curriculum_plot])


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
        numeric_cols = ["reward", "r_tput", "r_delay", "r_loss", "r_se", "r_energy",
                        "avg_throughput", "avg_delay", "avg_loss", "z_level", "z_success"]
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

        # ── Plot 4: Curriculum ─────────────────────────────────────────────
        fig_curr = make_subplots(specs=[[{"secondary_y": True}]])
        if "z_level" in df.columns:
            fig_curr.add_trace(go.Scatter(
                x=df.index, y=df["z_level"],
                mode="lines+markers", name="Curriculum Level",
                line=dict(color="#8b5cf6", width=2),
                marker=dict(size=4),
            ), secondary_y=False)
        if "z_success" in df.columns:
            fig_curr.add_trace(go.Scatter(
                x=df.index, y=df["z_success"],
                mode="lines", name="Success Rate",
                line=dict(color="#34d399", width=1.5, dash="dot"),
            ), secondary_y=True)
        fig_curr.update_layout(
            paper_bgcolor="#111827", plot_bgcolor="#1a1f35",
            font=dict(color="#f1f5f9"),
            title_text="Curriculum Level & Success Rate",
            legend=dict(bgcolor="#1a1f35"),
        )
        fig_curr.update_xaxes(gridcolor="#1e293b", title_text="Step")
        fig_curr.update_yaxes(gridcolor="#1e293b", title_text="Level", secondary_y=False)
        fig_curr.update_yaxes(gridcolor="#1e293b", title_text="Success Rate", secondary_y=True)

        return status, stats, fig_reward, fig_break, fig_kpi, fig_curr

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
