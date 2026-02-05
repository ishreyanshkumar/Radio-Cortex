#!/bin/bash

# Radio-Cortex: O-RAN RL Congestion Control - Commands Reference
# This file contains verified commands for the current codebase.

# ============================================================================
# 1. SETUP & INSTALLATION
# ============================================================================

# Install Python dependencies
pip install numpy torch gymnasium kafka-python

# Start Kafka (Required for training loop)
# Note: Keep this running in a separate terminal or background
./start_kafka.sh

# ============================================================================
# 2. TRAINING (radio_cortex_complete.py)
# ============================================================================

# --- Standard Training ---
# Trains PPO agent with default settings (3 cells, 20 UEs)
python3 radio_cortex_complete.py --mode train

# --- Custom Topology ---
# Train on a larger network
python3 radio_cortex_complete.py --mode train --num-ues 50 --num-cells 5 --scenario mobility_storm

# --- Hyperparameter Tuning (PPO) ---
# Customize PPO learning parameters
python3 radio_cortex_complete.py --mode train \
    --learning-rate 0.0001 \
    --gamma 0.995 \
    --batch-size 128 \
    --clip-epsilon 0.1 \
    --gae-lambda 0.98 \
    --ent-coef 0.02

# --- Advanced Configuration ---
# Control network size, rollout length, and device
python3 radio_cortex_complete.py --mode train \
    --hidden-dim 512 \
    --rollout-steps 4096 \
    --device cuda \
    --model-path models/advanced_ppo.pt

# --- Using Config File ---
# Load arguments from a JSON file
python3 radio_cortex_complete.py --mode train --config experiments/config_example.json

# ============================================================================
# 3. EVALUATION
# ============================================================================

# Evaluate trained model against baselines
python3 radio_cortex_complete.py --mode eval --model-path models/radio_cortex.pt

# Evaluate specific checkpoint
python3 radio_cortex_complete.py --mode eval --model-path models/advanced_ppo.pt

# ============================================================================
# 4. QUICK VERIFICATION (No ns-3/Kafka required)
# ============================================================================

# Fast training on mock environment to verify RL logic
python3 quick_train.py

# ============================================================================
# 5. UTILITIES
# ============================================================================

# Interpret Policy (Saliency Map)
python3 interpret_policy.py --checkpoint models/radio_cortex.pt --top-k 5

# Plot Training Progress
python3 -c "import json; import matplotlib.pyplot as plt; \
data = [json.loads(l) for l in open('action_logs.jsonl')]; \
plt.plot([d['reward'] for d in data]); plt.savefig('training_curve.png')"

# Monitor Training
tail -f action_logs.jsonl

# ============================================================================
# 6. NS-3 SIMULATION ONLY (Manual Testing)
# ============================================================================

# Run simulation without Python agent (generates traffic, requires Kafka listener)
./ns3 run "oran-congestion-scenario --numUes=20 --numCells=3 --simTime=10"
