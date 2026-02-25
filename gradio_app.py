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

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn

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
# ── Add project root to path so we can import policies ───────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Constants ─────────────────────────────────────────────────────────────────
MODELS_DIR = ROOT / "models"
LOGS_DIR   = ROOT / "logs"
RESULTS_DIR= ROOT / "results"

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

def _hex_alpha(hex_color, alpha):
    """Convert hex color + alpha float to rgba() string Plotly accepts."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

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
    sd = ckpt.get("policy_state_dict") or ckpt.get("state_dict") or ckpt
    
    def _detect(sd):
        for key in ["actor_logstd", "actor_logstd_head.weight"]:
            if key in sd:
                dim = sd[key].shape[-1]
                if sd[key].dim() == 2 and sd[key].shape[0] != 1:
                    dim = sd[key].shape[0]
                return max(1, dim // CELL_ACTIONS)
        for key in ["actor_mean.weight", "actor_mean.0.weight"]:
            if key in sd:
                return max(1, sd[key].shape[0] // CELL_ACTIONS)
        if "frame_encoder.0.weight" in sd:
            return max(1, sd["frame_encoder.0.weight"].shape[1] // CELL_FEATURES)
        for key in ["state_embed.weight", "feature_net.0.weight"]:
            if key in sd:
                return max(1, sd[key].shape[1] // (CELL_FEATURES * 3))
        return num_cells

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

    if isinstance(ckpt, dict):
        sd = ckpt.get("policy_state_dict") or ckpt.get("state_dict") or ckpt
    else:
        sd = ckpt

    policy.load_state_dict(sd, strict=False)
    policy.eval()

    vn_path = MODELS_DIR / f"vec_normalize_{model_name.replace('.pt', '.pkl')}"
    vec_norm = None
    if vn_path.exists():
        try:
            import pickle
            with open(vn_path, "rb") as f:
                vec_norm = pickle.load(f)
        except Exception as e:
            print(f"Warning: Failed to load VecNormalize from {vn_path}: {e}")

    return policy, vec_norm, num_cells

def run_inference(policy, state_np: np.ndarray, vec_norm=None, deterministic: bool = True, device: str = "cpu"):
    if vec_norm is not None:
        try:
            mean = vec_norm.obs_rms.mean
            var = vec_norm.obs_rms.var
            if len(mean) > len(state_np):
                mean = mean[:len(state_np)]
                var = var[:len(state_np)]
            elif len(mean) < len(state_np):
                padding = len(state_np) - len(mean)
                mean = np.pad(mean, (0, padding), 'constant')
                var = np.pad(var, (0, padding), 'constant', constant_values=1.0)
            epsilon = 1e-8
            state_np = (state_np - mean) / np.sqrt(var + epsilon)
            state_np = np.clip(state_np, -10, 10)
        except Exception as e:
            pass

    with torch.no_grad():
        state_t = torch.FloatTensor(state_np).unsqueeze(0).to(device)
        if hasattr(policy, '_forward_common'):
            action_mean, logstd, value = policy._forward_common(state_t)
        else:
            outputs = policy(state_t)
            if isinstance(outputs, tuple):
                if len(outputs) == 3:
                    action_mean, logstd, value = outputs
                else:
                    action_mean, logstd = outputs[0], outputs[1]
                    value = outputs[2] if len(outputs) > 2 else torch.zeros(1)
            else:
                action_mean = outputs
                logstd = torch.zeros_like(outputs)
                value = torch.zeros(1)

        if action_mean.dim() == 3:
            action_mean = action_mean[:, -1, :]
        if torch.is_tensor(logstd) and logstd.dim() == 3:
            logstd = logstd[:, -1, :]
        if torch.is_tensor(value) and value.dim() == 3:
            value = value[:, -1, :]
        
        if not torch.is_tensor(logstd):
            logstd = torch.as_tensor(logstd).to(device)
        
        if logstd.shape != action_mean.shape:
             logstd = logstd.expand_as(action_mean)

        action_std = torch.exp(logstd)
        if deterministic:
            actions = action_mean
        else:
            dist = torch.distributions.Normal(action_mean, action_std)
            actions = dist.sample()

    return actions.squeeze(0).cpu().numpy(), value.squeeze().item(), action_mean.squeeze(0).cpu().numpy(), action_std.squeeze(0).cpu().numpy()

def build_inference_tab():
    with gr.Tab("🤖 Model Inference"):
        gr.Markdown("## Model Inference\nLoad a BDH checkpoint and feed in cell-level state features to see the agent's predicted actions and value estimate.")

        with gr.Row():
            with gr.Column(scale=1):
                model_dd = gr.Dropdown(label="Select Model Checkpoint", choices=list_models(), value=list_models()[0], interactive=True)
                refresh_btn = gr.Button("🔄 Refresh Model List", size="sm")
                deterministic_cb = gr.Checkbox(value=True, label="Deterministic Action (uncheck to sample)")

            with gr.Column(scale=2):
                gr.Markdown("### Cell State Features\n*Adjust sliders for each cell*")
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
        actions_table = gr.Dataframe(label="Action Details", headers=["Cell"] + ACTION_NAMES, interactive=False)

        def refresh_models():
            return gr.Dropdown(choices=list_models())

        def do_inference(model_name, deterministic, *slider_vals):
            try:
                n_int = DEFAULT_CELLS
                full_frame = []
                for c in range(n_int):
                    start = c * VISIBLE_FEATURES
                    end   = start + VISIBLE_FEATURES
                    cell_vals = list(slider_vals[start:end])
                    cell_vals.extend([0.0] * (CELL_FEATURES - len(cell_vals)))
                    full_frame.extend(cell_vals)
                
                single_frame = np.array(full_frame, dtype=np.float32)
                state = np.concatenate([single_frame, single_frame, single_frame])
                policy, vec_norm, n_actual = load_policy(model_name, num_cells=n_int)
                
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

                actions_2d = actions.reshape(n_actual, CELL_ACTIONS)
                mean_2d    = mean.reshape(n_actual, CELL_ACTIONS)
                std_2d     = std.reshape(n_actual, CELL_ACTIONS)

                if n_int > n_actual:
                    idxs = [i % n_actual for i in range(n_int)]
                    actions_2d = actions_2d[idxs]
                    mean_2d    = mean_2d[idxs]
                    std_2d     = std_2d[idxs]
                n = n_int 

                rows = []
                for ci in range(n):
                    tiled_note = f" (↻{ci % n_actual})" if n_int > n_actual and ci >= n_actual else ""
                    rows.append([f"Cell {ci}{tiled_note}"] + [f"{v:.4f}" for v in actions_2d[ci]])
                df = pd.DataFrame(rows, columns=["Cell"] + ACTION_NAMES)

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
                    paper_bgcolor="#111827", plot_bgcolor="#1a1f35", font=dict(color="#f1f5f9"),
                    xaxis=dict(gridcolor="#1e293b"), yaxis=dict(gridcolor="#1e293b", title="Action Value [0–1]"), legend=dict(bgcolor="#1a1f35")
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
# TAB — Model Comparison (side-by-side checkpoint comparison)
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

        def do_comparison(model_a, model_b, deterministic, *slider_vals):
            try:
                n_int = DEFAULT_CELLS
                full_frame = []
                for c in range(n_int):
                    start = c * VISIBLE_FEATURES
                    end   = start + VISIBLE_FEATURES
                    cell_vals = list(slider_vals[start:end])
                    cell_vals.extend([0.0] * (CELL_FEATURES - len(cell_vals)))
                    full_frame.extend(cell_vals)

                single_frame = np.array(full_frame, dtype=np.float32)
                state = np.concatenate([single_frame, single_frame, single_frame])
                
                policy_a, vnorm_a, n_a = load_policy(model_a, num_cells=n_int)
                policy_b, vnorm_b, n_b = load_policy(model_b, num_cells=n_int)
                
                # Check for model A shapes
                if n_a != n_int:
                    full_frame_actual_a = []
                    for c in range(n_a):
                        start = c * VISIBLE_FEATURES
                        end   = start + VISIBLE_FEATURES
                        cell_vals = list(slider_vals[start:end]) if start < len(slider_vals) else []
                        cell_vals.extend([0.0] * (CELL_FEATURES - len(cell_vals)))
                        full_frame_actual_a.extend(cell_vals)
                    single_frame_a = np.array(full_frame_actual_a, dtype=np.float32)
                    state_a = np.concatenate([single_frame_a, single_frame_a, single_frame_a])
                else: 
                    state_a = state
                
                # Check for model B shapes
                if n_b != n_int:
                    full_frame_actual_b = []
                    for c in range(n_b):
                        start = c * VISIBLE_FEATURES
                        end   = start + VISIBLE_FEATURES
                        cell_vals = list(slider_vals[start:end]) if start < len(slider_vals) else []
                        cell_vals.extend([0.0] * (CELL_FEATURES - len(cell_vals)))
                        full_frame_actual_b.extend(cell_vals)
                    single_frame_b = np.array(full_frame_actual_b, dtype=np.float32)
                    state_b = np.concatenate([single_frame_b, single_frame_b, single_frame_b])
                else:
                    state_b = state

                acts_a, val_a, mean_a, std_a = run_inference(policy_a, state_a, vec_norm=vnorm_a, deterministic=deterministic)
                acts_b, val_b, mean_b, std_b = run_inference(policy_b, state_b, vec_norm=vnorm_b, deterministic=deterministic)

                acts_a_2d = acts_a.reshape(n_a, CELL_ACTIONS)
                acts_b_2d = acts_b.reshape(n_b, CELL_ACTIONS)
                
                # Since A and B might have different n, compute overlap n for comparison rendering
                n = min(n_a, n_b)

                def _make_table(acts_2d, num):
                    rows = [[f"Cell {ci}"] + [f"{v:.4f}" for v in acts_2d[ci]] for ci in range(num)]
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
                        _make_table(acts_a_2d, n_a), _make_table(acts_b_2d, n_b),
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
# NEW: Native UI Integrations
# ─────────────────────────────────────────────────────────────────────────────

def build_dashboard_tab():
    with gr.Tab("📊 Evaluation Dashboard"):
        gr.HTML('''
            <iframe src="/ui/index.html?page=dashboard&embed=true&v=10" 
                    width="100%" height="900" 
                    style="border:none; border-radius: 12px; overflow: hidden; background: #0f1118;">
            </iframe>
        ''')

def build_modelbench_tab():
    with gr.Tab("🏆 ModelBench"):
        gr.HTML('''
            <iframe src="/ui/index.html?page=modelbench&embed=true&v=10" 
                    width="100%" height="900" 
                    style="border:none; border-radius: 12px; overflow: hidden; background: #0f1118;">
            </iframe>
        ''')

def build_simulation_tab():
    with gr.Tab("🌐 Live Signal Matrix"):
        gr.HTML('''
            <iframe src="/ui/index.html?page=visualizer&embed=true&v=10" 
                    width="100%" height="900" 
                    style="border:none; border-radius: 12px; overflow: hidden; background: #0f1118;">
            </iframe>
        ''')

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

        build_simulation_tab()
        build_interpretability_tab()
        build_inference_tab()
        build_dashboard_tab()
        build_modelbench_tab()
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
    # Create custom FastAPI backend to reliably serve HTML/CSS/JS without Gradio /file= query sync issues
    app = FastAPI()
    app.mount("/ui", StaticFiles(directory=str(ROOT / "ui")), name="ui")
    app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")

    @app.get("/api/results")
    def list_results_api():
        csvs = [os.path.basename(p) for p in sorted(glob.glob(str(RESULTS_DIR / "*.csv")))]
        dbs = [os.path.basename(p) for p in sorted(glob.glob(str(RESULTS_DIR / "*.db")))]
        return {"csvs": csvs, "dbs": dbs}
    
    demo = build_app()
    app = gr.mount_gradio_app(app, demo, path="/")
    
    print(f"* Hosted Radio-Cortex Interface: http://{args.host}:{args.port}")
    if args.share:
        print("* Note: --share is ignored when using custom FastAPI mounts. Use ngrok or localtunnel.")
        
    uvicorn.run(app, host=args.host, port=args.port)
