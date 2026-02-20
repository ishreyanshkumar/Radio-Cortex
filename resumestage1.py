#!/usr/bin/env python3
"""
Radio-Cortex: Stage 1 Convergence Script
Resumes Stage 1 (Flash Crowd) with extended timesteps to achieve convergence
(Higher Explained Variance, Lower Entropy).
"""

import os
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

# Import core components
from oran_ns3_env import NS3Config
from radio_cortex_complete import train_radio_cortex

def resume_stage_1():
    # ── Configuration ──
    # We target 5,000,000 steps to allow the Transformer (BDH) to specialize
    TOTAL_TIMESTEPS = 5000000
    STAGE_1_PATH = "models/curriculum/stage_1.pt"
    
    # Hyperparameters from train_curriculum.sh (Optimized for BDH stability)
    config = NS3Config(
        num_ues=20,
        num_cells=3,
        sim_time=50.0,
        kpm_interval_ms=100,
        scenario='flash_crowd',
        verbose=True
    )
    
    # Path setup
    save_path = STAGE_1_PATH
    log_dir = "logs"
    Path(log_dir).mkdir(exist_ok=True)
    Path("models/curriculum").mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("🚀 RESUMING STAGE 1: CONVERGENCE PURSUIT")
    print(f"Target steps: {TOTAL_TIMESTEPS:,}")
    print(f"Model path:    {save_path}")
    print("="*60)

    # Check if checkpoint exists
    if not os.path.exists(save_path):
        print(f"⚠️  Warning: {save_path} not found. Starting Stage 1 from scratch.")
    else:
        print(f"✅ Found existing checkpoint. Resuming...")

    # Execute Training
    # We use the EXACT hyperparameters from the curriculum for consistency
    train_radio_cortex(
        config=config,
        total_timesteps=TOTAL_TIMESTEPS,
        save_path=save_path,
        lr=3e-5,               # Conservative learning rate for stability
        gamma=0.99,            # standard discount
        batch_size=512,        # Large batch for stable gradients
        hidden_dim=512,        # BDH embedding dimension
        gae_lambda=0.95,
        clip_epsilon=0.1,      # Tight clipping for BDH
        vf_coef=1.0,           # Boosted value head learning
        ent_coef=0.01,         # Exploration maintenance
        max_grad_norm=0.5,
        rollout_steps=512,     # 48 envs * 512 = 24,576 samples per update
        log_interval=1,
        checkpoint_interval=5,
        n_envs=48,             # Optimized for high-parallelism
        model_type='bdh',
        num_epochs=8,          # Prevent over-optimization on small datasets
        lr_scheduler_gamma=0.995,
        target_kl=0.05
    )

if __name__ == "__main__":
    # Prevent CUDA fragmentation
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    
    try:
        resume_stage_1()
    except KeyboardInterrupt:
        print("\n\nStopping training...")
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
