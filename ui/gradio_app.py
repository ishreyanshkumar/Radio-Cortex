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
# ── Add project root to path so we can import policies ───────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Constants ─────────────────────────────────────────────────────────────────
MODELS_DIR = ROOT / "models"
LOGS_DIR   = ROOT / "logs"

CELL_FEATURES = 16
CELL_ACTIONS  = 3
DEFAULT_CELLS = 3

FEATURE_NAMES = [
    "Queue Length", "RB Util", "Tx Power Norm", "Cell Load", "Avg RB Req",
    "Avg Tput", "Avg Delay", "Avg Loss", "Max Delay", "Max Loss", "Jains",
    "UE Count Norm", "Padding", "Padding", "Padding", "Padding"
]
VISIBLE_FEATURES = 12  # Only show the first 12 features, hide padding
ACTION_NAMES = ["TxPowerDBM", "CIO", "TTT"]

REWARD_COMPONENTS = ["r_tput", "r_delay", "r_loss", "r_load", "r_energy", "r_sla", "r_cio"]
COMPONENT_COLORS  = ["#00d4ff", "#f87171", "#fbbf24", "#f472b6", "#34d399", "#a78bfa", "#6366f1"]

# ── Model loading helpers ─────────────────────────────────────────────────────

def list_models():
    """Return sorted list of .pt model files in models/ with metadata."""
    pts = sorted(glob.glob(str(MODELS_DIR / "*.pt")))
    results = []
    for p in pts:
        name = os.path.basename(p)
        try:
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
            h = ckpt.get("hyperparams", {})
            m_type = h.get("model_type", "unknown")
            steps = ckpt.get("total_steps", ckpt.get("total_timesteps", "unknown"))
            results.append(f"{name} [{m_type}, {steps} steps]")
        except:
             results.append(name)
    return results or ["(no models found)"]


def load_policy(model_display_name: str, num_cells: int = DEFAULT_CELLS, device: str = "cpu"):
    """Load a policy checkpoint from models/."""
    from policies import get_policy
    from oran_ns3_env import NS3Config

    model_name = model_display_name.split(" [")[0]
    model_path = MODELS_DIR / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    ckpt = torch.load(str(model_path), map_location=device, weights_only=False)
    hyperparams = ckpt.get("hyperparams", {})
    model_type = hyperparams.get("model_type", "bdh")
    hidden_dim = hyperparams.get("hidden_dim", 256)

    # ── Auto-Detect num_cells from checkpoint sizes ──────────────────────────
    # frame_encoder.0.weight shape is (emb_dim, num_cells * 16)
    # or action_head.2.weight shape is (num_cells * 3, emb_dim)
    # or logstd_head shape is (1, num_cells * 3)
    
    sd = ckpt.get("policy_state_dict") or ckpt.get("state_dict") or ckpt
    
    detected_num_cells = num_cells
    # ── Detection strategy ────────────────────────────────────────────────────
    # Priority: output-side (arch-agnostic) > input-side (arch-specific)
    # Key name variants by architecture:
    #   Universal/TrXL: actor_logstd [1, n*3], actor_mean.weight [n*3, h]
    #   GPT2/Reformer:  actor_mean.weight [n*3, h] (no actor_logstd param)
    #   NN (ActorCritic): actor_mean.0.weight [n*3, h], actor_logstd_head.weight [n*3, h]
    #   BDH:            frame_encoder.0.weight [emb, n*16] (no actor keys)
    
    def _detect(sd):
        # Try output-side keys first (most reliable across all archs)
        for key in ["actor_logstd", "actor_logstd_head.weight"]:
            if key in sd:
                dim = sd[key].shape[-1]  # [1,dim] or [dim,h]
                if sd[key].dim() == 2 and sd[key].shape[0] != 1:
                    dim = sd[key].shape[0]  # actor_logstd_head is [n*3, h]
                return max(1, dim // CELL_ACTIONS)
        for key in ["actor_mean.weight", "actor_mean.0.weight"]:
            if key in sd:
                return max(1, sd[key].shape[0] // CELL_ACTIONS)
        # Input-side fallbacks
        if "frame_encoder.0.weight" in sd:
            return max(1, sd["frame_encoder.0.weight"].shape[1] // CELL_FEATURES)
        for key in ["state_embed.weight", "feature_net.0.weight"]:
            if key in sd:
                # These receive stacked obs: input = num_cells * CELL_FEATURES * 3
                return max(1, sd[key].shape[1] // (CELL_FEATURES * 3))
        return num_cells  # No detection possible, keep requested

    detected_num_cells = _detect(sd)
    if detected_num_cells != num_cells:
        print(f"⚠️  Model trained on {detected_num_cells} cells (slider={num_cells}) — running on {detected_num_cells}.")
        num_cells = detected_num_cells

    cfg = NS3Config()
    cfg.num_cells = num_cells

    # Instantiate via factory
    policy = get_policy(
        model_type=model_type,
        num_ues=cfg.num_ues,
        num_cells=num_cells,
        hidden_dim=hidden_dim,
        device=device,
        env_config=cfg
    )

    # Support both raw state_dict and wrapped checkpoint formats
    if isinstance(ckpt, dict):
        sd = ckpt.get("policy_state_dict") or ckpt.get("state_dict") or ckpt
    else:
        sd = ckpt

    policy.load_state_dict(sd, strict=False)
    policy.eval()

    # Load VecNormalize if present
    vn_path = MODELS_DIR / f"vec_normalize_{model_name.replace('.pt', '.pkl')}"
    vec_norm = None
    if vn_path.exists():
        try:
            from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
            # We need a dummy env to load VecNormalize
            class DummyEnv:
                def __init__(self):
                    self.observation_space = torch.nn.Identity() # Placeholder
                    self.action_space = torch.nn.Identity()
            
            # This is a bit hacky because SB3 VecNormalize expects a VecEnv
            # But we only need it for the normalization stats.
            # A better way is to use a simplified normalization if SB3 is not fully compatible here.
            # For now, let's try to load it properly if possible.
            from vec_env_wrapper import load_vec_normalize
            import pickle
            with open(vn_path, "rb") as f:
                vec_norm_data = pickle.load(f)
                # vec_norm_data is usually a VecNormalize object or its state
                vec_norm = vec_norm_data
        except Exception as e:
            print(f"Warning: Failed to load VecNormalize from {vn_path}: {e}")

    return policy, vec_norm, num_cells


def run_inference(policy, state_np: np.ndarray, vec_norm=None, deterministic: bool = True, device: str = "cpu"):
    """Run a single forward pass. Returns (actions, value, action_mean, action_std)."""
    # Apply normalization if available
    if vec_norm is not None:
        try:
            # obs_rms.mean is normally (obs_dim,)
            mean = vec_norm.obs_rms.mean
            var = vec_norm.obs_rms.var
            
            # ── Fix Broadcast Error: Slice stats to match input dimension ────────
            # Input might be fewer cells than what was trained
            if len(mean) > len(state_np):
                print(f"Slicing normalization stats from {len(mean)} to {len(state_np)} features.")
                mean = mean[:len(state_np)]
                var = var[:len(state_np)]
            elif len(mean) < len(state_np):
                # This shouldn't happen with auto-detect, but handle just in case
                padding = len(state_np) - len(mean)
                mean = np.pad(mean, (0, padding), 'constant')
                var = np.pad(var, (0, padding), 'constant', constant_values=1.0)
            
            epsilon = 1e-8
            state_np = (state_np - mean) / np.sqrt(var + epsilon)
            state_np = np.clip(state_np, -10, 10) # SB3 default clip
        except Exception as e:
            print(f"Normalization failed: {e}")

    with torch.no_grad():
        state_t = torch.FloatTensor(state_np).unsqueeze(0).to(device)
        
        # Try different possible forward methods
        if hasattr(policy, '_forward_common'):
            action_mean, logstd, value = policy._forward_common(state_t)
        else:
            # Fallback to standard forward
            outputs = policy(state_t)
            if isinstance(outputs, tuple):
                if len(outputs) == 3:
                    action_mean, logstd, value = outputs
                else:
                    # Some might return more or fewer
                    action_mean, logstd = outputs[0], outputs[1]
                    value = outputs[2] if len(outputs) > 2 else torch.zeros(1)
            else:
                action_mean = outputs
                logstd = torch.zeros_like(outputs)
                value = torch.zeros(1)

        # Handle transformer-style sequence outputs (B, T, D) -> (B, D)
        if action_mean.dim() == 3:
            action_mean = action_mean[:, -1, :]
        if torch.is_tensor(logstd) and logstd.dim() == 3:
            logstd = logstd[:, -1, :]
        if torch.is_tensor(value) and value.dim() == 3:
            value = value[:, -1, :]
        
        # Ensure logstd is a tensor and same shape as action_mean
        if not torch.is_tensor(logstd):
            # Might be a Parameter or some other type
            logstd = torch.as_tensor(logstd).to(device)
        
        if logstd.shape != action_mean.shape:
             logstd = logstd.expand_as(action_mean)

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
                deterministic_cb = gr.Checkbox(value=True, label="Deterministic Action (uncheck to sample)")

            with gr.Column(scale=2):
                gr.Markdown("### Cell State Features")
                gr.Markdown(f"*Adjust sliders for each cell (12 features × {DEFAULT_CELLS} cells)*")
                state_inputs = []
                for c in range(DEFAULT_CELLS):
                    with gr.Accordion(f"Cell {c}", open=(c == 0)):
                        row_inputs = []
                        for f_idx in range(VISIBLE_FEATURES):
                            fname = FEATURE_NAMES[f_idx]
                            lo, hi, default = _feature_range(f_idx)
                            sl = gr.Slider(lo, hi, value=default, label=f"C{c}: {fname}", step=(hi - lo) / 100)
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

        def do_inference(model_name, deterministic, *slider_vals):
            try:
                n_int = DEFAULT_CELLS
                # Sliders are VISIBLE_FEATURES per cell (12), but model expects CELL_FEATURES (16)
                # Reconstruct full state with padding zeros
                full_frame = []
                for c in range(n_int):
                    start = c * VISIBLE_FEATURES
                    end   = start + VISIBLE_FEATURES
                    cell_vals = list(slider_vals[start:end])
                    # Pad to CELL_FEATURES
                    cell_vals.extend([0.0] * (CELL_FEATURES - len(cell_vals)))
                    full_frame.extend(cell_vals)
                
                single_frame = np.array(full_frame, dtype=np.float32)
                # Tile the single frame 3 times to satisfy n_stack=3
                state = np.concatenate([single_frame, single_frame, single_frame])

                policy, vec_norm, n_actual = load_policy(model_name, num_cells=n_int)
                
                # ── Fix Broadcast Error: Ensure state matches n_actual ──────────
                if n_actual != n_int:
                    full_frame_actual = []
                    for c in range(n_actual):
                        start = c * VISIBLE_FEATURES
                        end   = start + VISIBLE_FEATURES
                        cell_vals = list(slider_vals[start:end]) if start < len(slider_vals) else []
                        cell_vals.extend([0.0] * (CELL_FEATURES - len(cell_vals)))
                        full_frame_actual.extend(cell_vals)
                    single_frame = np.array(full_frame_actual, dtype=np.float32)
                    state = np.concatenate([single_frame, single_frame, single_frame])

                actions, value, mean, std = run_inference(policy, state, vec_norm=vec_norm, deterministic=deterministic)

                # Run on model's native cell count, then tile to requested count
                actions_2d = actions.reshape(n_actual, CELL_ACTIONS)
                mean_2d    = mean.reshape(n_actual, CELL_ACTIONS)
                std_2d     = std.reshape(n_actual, CELL_ACTIONS)

                # Tile (cycle) outputs to fill n_int cells
                # e.g. model=3 cells, slider=5: [0,1,2,0,1]
                if n_int > n_actual:
                    idxs = [i % n_actual for i in range(n_int)]
                    actions_2d = actions_2d[idxs]
                    mean_2d    = mean_2d[idxs]
                    std_2d     = std_2d[idxs]
                n = n_int  # Always show what the slider requested

                # Build table
                rows = []
                for ci in range(n):
                    tiled_note = f" (↻{ci % n_actual})" if n_int > n_actual and ci >= n_actual else ""
                    rows.append([f"Cell {ci}{tiled_note}"] + [f"{v:.4f}" for v in actions_2d[ci]])
                df = pd.DataFrame(rows, columns=["Cell"] + ACTION_NAMES)

                # Build bar chart
                fig = go.Figure()
                for ci in range(n):
                    is_tiled = n_int > n_actual and ci >= n_actual
                    fig.add_trace(go.Bar(
                        name=f"Cell {ci}" + (f" (↻{ci%n_actual})" if is_tiled else ""),
                        x=ACTION_NAMES,
                        y=actions_2d[ci].tolist(),
                        error_y=dict(type="data", array=std_2d[ci].tolist(), visible=True),
                        opacity=0.65 if is_tiled else 1.0,
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

                cell_note = f" — model has {n_actual} cells, tiled to {n_int}" if n_actual != n_int else ""
                return f"{value:.4f}", f"✅ Inference OK{cell_note}", fig, df

            except Exception as e:
                return "—", f"❌ Error: {e}\n{traceback.format_exc()}", None, None

        refresh_btn.click(refresh_models, outputs=[model_dd])
        run_btn.click(
            do_inference,
            inputs=[model_dd, deterministic_cb] + state_inputs,
            outputs=[value_box, status_box, actions_plot, actions_table],
        )

    return model_dd, state_inputs


def _feature_range(f_idx):
    """Return (lo, hi, default) for each of the 12 cell features."""
    ranges = [
        (0, 1, 0.1),    # Queue Length
        (0, 1, 0.5),    # RB Util
        (0, 1, 0.5),    # Tx Power Norm
        (0, 1, 0.2),    # Cell Load
        (0, 1, 0.1),    # Avg RB Req
        (0, 1, 0.5),    # Avg Tput
        (0, 1, 0.3),    # Avg Delay
        (0, 1, 0.05),   # Avg Loss
        (0, 1, 0.4),    # Max Delay
        (0, 1, 0.1),    # Max Loss
        (0, 1, 0.8),    # Jains
        (0, 1, 0.4),    # UE Count Norm
        (0, 1, 0),      # Feature 12
        (0, 1, 0),      # Feature 13
        (0, 1, 0),      # Feature 14
        (0, 1, 0),      # Feature 15
    ]
    if f_idx < len(ranges):
        return ranges[f_idx]
    return (0, 1, 0)


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
        stat_df = gr.Dataframe(label="Summary Statistics", interactive=False)

        with gr.Row():
            kpi_avg_tput = gr.Number(label="Avg Throughput (Mbps)", value=0, interactive=False)
            kpi_avg_delay = gr.Number(label="Avg Delay (ms)", value=0, interactive=False)
            kpi_avg_loss = gr.Number(label="Avg Loss (%)", value=0, interactive=False)
            kpi_reward = gr.Number(label="Total Reward", value=0, interactive=False)

        with gr.Row():
            reward_plot   = gr.Plot(label="Total Reward Over Time")
            breakdown_plot = gr.Plot(label="Reward Component Breakdown")
        
        with gr.Row():
            kpi_plot      = gr.Plot(label="KPI Trends (Throughput / Delay / Loss)")
            success_plot  = gr.Plot(label="UE Success Rate")

        # ── Live Polling ─────────────────────────────────────────────────────
        live_timer = gr.Timer(value=5, active=False)
        with gr.Row():
            live_toggle = gr.Checkbox(label="Enable Live Polling", value=False)
            poll_interval = gr.Slider(label="Poll Interval (s)", minimum=1, maximum=30, value=5, step=1)

        def _list_logs_refresh():
            return gr.Dropdown(choices=_list_log_csvs())

        def load_from_file(f):
            if f is None:
                return ["No file uploaded."] + [None]*9
            return _process_csv(f.name)

        def load_from_dropdown(log_name):
            if not log_name:
                return ["No log selected."] + [None]*9
            path = LOGS_DIR / log_name
            return _process_csv(str(path))

        def live_update(log_name, current_rows):
            if not log_name or log_name == "(no logs found)":
                return [gr.skip()]*10
            path = LOGS_DIR / log_name
            try:
                # Optimized: could just check file size first
                # But for now, just reload
                return _process_csv(str(path))
            except:
                return [gr.skip()]*10

        refresh_log_btn.click(_list_logs_refresh, outputs=[log_dd])
        
        csv_outputs = [load_status, stat_df, kpi_avg_tput, kpi_avg_delay, kpi_avg_loss, kpi_reward, reward_plot, breakdown_plot, kpi_plot, success_plot]
        
        csv_file.change(load_from_file, inputs=[csv_file], outputs=csv_outputs)
        load_log_btn.click(load_from_dropdown, inputs=[log_dd], outputs=csv_outputs)
        
        live_toggle.change(lambda x: gr.Timer(active=x), inputs=[live_toggle], outputs=[live_timer])
        poll_interval.change(lambda x: gr.Timer(value=x), inputs=[poll_interval], outputs=[live_timer])
        live_timer.tick(live_update, inputs=[log_dd], outputs=csv_outputs)


def _list_log_csvs():
    csvs = sorted(glob.glob(str(LOGS_DIR / "reward_metrics_*.csv")))
    return [os.path.basename(p) for p in csvs] or ["(no logs found)"]


def _process_csv(path: str):
    """Load a reward_metrics CSV and return all chart figures + stats."""
    try:
        df = pd.read_csv(path)
        if df.empty:
            return "CSV is empty.", None, 0, 0, 0, 0, None, None, None, None

        status = f"✅ Loaded {len(df)} rows from {os.path.basename(path)}"

        # ── Summary stats ──────────────────────────────────────────────────
        numeric_cols = ["reward", "r_tput", "r_delay", "r_loss", "r_load", "r_energy", "r_sla", "r_cio",
                        "jains", "p95_delay", "avg_throughput", "avg_delay", "avg_loss", "z_success"]
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

        # ── KPI summary values ─────────────────────────────────────────────
        avg_tput   = round(df["avg_throughput"].mean(), 2) if "avg_throughput" in df.columns else 0
        avg_delay  = round(df["avg_delay"].mean(), 2)      if "avg_delay"      in df.columns else 0
        avg_loss   = round(df["avg_loss"].mean() * 100, 2)  if "avg_loss"       in df.columns else 0
        total_reward = round(df["reward"].sum(), 2)          if "reward"         in df.columns else 0

        return status, stats, avg_tput, avg_delay, avg_loss, total_reward, fig_reward, fig_break, fig_kpi, fig_success

    except Exception as e:
        return f"❌ Error: {e}\n{traceback.format_exc()}", None, 0, 0, 0, 0, None, None, None, None


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
            det_cmp = gr.Checkbox(value=True, label="Deterministic")

        gr.Markdown("### Shared Cell State Input")
        cmp_state_inputs = []
        for c in range(DEFAULT_CELLS):
            with gr.Accordion(f"Cell {c}", open=(c == 0)):
                for f_idx in range(VISIBLE_FEATURES):
                    fname = FEATURE_NAMES[f_idx]
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

        def _build_state_from_sliders(slider_vals, n_cells):
            """Reconstruct a full state vector from visible slider values."""
            full_frame = []
            for c in range(n_cells):
                start = c * VISIBLE_FEATURES
                end   = start + VISIBLE_FEATURES
                cell_vals = list(slider_vals[start:end]) if start < len(slider_vals) else []
                cell_vals.extend([0.0] * (CELL_FEATURES - len(cell_vals)))
                full_frame.extend(cell_vals)
            single_frame = np.array(full_frame, dtype=np.float32)
            return np.concatenate([single_frame, single_frame, single_frame])

        def do_comparison(model_a, model_b, deterministic, *slider_vals):
            try:
                n_int = DEFAULT_CELLS
                state = _build_state_from_sliders(slider_vals, n_int)

                pol_a, vn_a, n_a = load_policy(model_a, num_cells=n_int)
                pol_b, vn_b, n_b = load_policy(model_b, num_cells=n_int)

                # Check if models have different cell counts
                if n_a != n_b:
                     return f"❌ Cannot compare models with different cell counts: {model_a} ({n_a} cells) vs {model_b} ({n_b} cells)", None, None, None, "—", "—", None

                n = n_a
                if n != n_int:
                    state = _build_state_from_sliders(slider_vals, n)

                acts_a, val_a, mean_a, std_a = run_inference(pol_a, state, vec_norm=vn_a, deterministic=deterministic)
                acts_b, val_b, mean_b, std_b = run_inference(pol_b, state, vec_norm=vn_b, deterministic=deterministic)

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
            inputs=[model_a_dd, model_b_dd, det_cmp] + cmp_state_inputs,
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
