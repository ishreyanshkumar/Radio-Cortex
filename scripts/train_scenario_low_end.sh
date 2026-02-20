#!/bin/bash
# Low-Resource Training Script for Flash Crowd Scenario
# Optimized for 4GB-8GB VRAM / Dual-Core Systems
# Fixes OOM issues by reducing batch size and environment count.

echo "🚀 Starting Training for Flash Crowd (Low-Resource Mode)..."

# Memory Optimization Flags
export PYTORCH_ALLOC_CONF="expandable_segments:True"

python3 radio_cortex_complete.py \
    --mode train \
    --scenario flash_crowd \
    --model bdh \
    --n-envs 2 \
    --total-timesteps 50000 \
    --kpm-interval 100 \
    --rollout-steps 64 \
    --batch-size 32 \
    --lr-gamma 0.99 \
    --learning-rate 8e-5 \
    --clip-epsilon 0.1 \
    --ent-coef 0.03 \
    --gamma 0.98 \
    --ppo-epochs 10 \
    --device cuda

echo "✅ Training Complete. Check logs/reward_metrics_*.csv for results."
