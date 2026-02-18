#!/bin/bash
# Optimized Training Script for Flash Crowd Scenario
# Uses Tuned Proximal Policy Optimization (PPO) with BDH-Slim
# 


# Run Training with Optimized Hyperparameters
# --scenario flash_crowd: Target specific scenario
# --n-envs 4: Balance between throughput and stability
# --steps 200000: Total training timesteps
# --model bdh: Uses the new Slim BDH architecture
# --rollout-steps 512: Compact rollout for faster feedback (approx 3.4 episodes/update)
# --batch-size 64: Stable batch size
# --ppo-epochs 20: Deep updates per batch
# --lr-gamma 0.98: Slower decay to accomodate more frequent updates

echo "🚀 Starting Optimized Training for Flash Crowd..."
echo "Config: PPO (Rollout=512, Batch=64, Epochs=20) | LR Gamma=0.98"

python3 radio_cortex_complete.py \
    --mode train \
    --scenario flash_crowd \
    --model bdh \
    --n-envs 4 \
    --total-timesteps 200000 \
    --rollout-steps 512 \
    --batch-size 64 \
    --ppo-epochs 20 \
    --lr-gamma 0.98 \

echo "✅ Training Complete. Check logs/reward_metrics_*.csv for results."
