#!/bin/bash
# Optimized Training Script for Flash Crowd Scenario
# Uses Tuned Proximal Policy Optimization (PPO) with BDH-Slim
# 


# Run Training (Stable Low-End Config)
# --scenario flash_crowd: Target specific scenario
# --n-envs 4: Low resource usage (Consumer Laptop/VM)
# --steps 200000: Sufficient for single scenario convergence
# --model bdh: Uses the Balanced BDH architecture (Dim 256)
# --rollout-steps 512: Larger rollout to compensate for few envs (Buffer=2048)
# --batch-size 512: Stable updates with smaller buffer
# --ppo-epochs 10: Standard PPO epochs

echo "🚀 Starting Training for Flash Crowd..."

python3 radio_cortex_complete.py \
    --mode train \
    --scenario flash_crowd \
    --model bdh \
    --n-envs 4 \
    --total-timesteps 100000 \
    --kpm-interval 100 \
    --rollout-steps 512 \
    --batch-size 512 \
    --lr-gamma 0.99 \
    --learning-rate 3e-5 \
    --clip-epsilon 0.1 \
    --ent-coef 0.03 \
    --gamma 0.99 \
    --ppo-epochs 10

echo "✅ Training Complete. Check logs/reward_metrics_*.csv for results."
