#!/bin/bash
# Quick training script for Radio-Cortex
# Minimum steps to verify the RL pipeline works and saves a model

# Run training with minimal steps
./.venv/bin/python3 radio_cortex_complete.py \
    --mode train \
    --total-timesteps 100 \
    --rollout-steps 20 \
    --batch-size 10 \
    --model-path models/radio_cortex_quick.pt \
    --num-ues 5 \
    --num-cells 1 \
    --n-envs 1

echo "Quick training complete. Model saved to models/radio_cortex_quick.pt"
