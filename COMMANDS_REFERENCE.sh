#!/bin/bash

# Radio-Cortex: O-RAN RL Congestion Control - Commands Reference
# This file contains verified commands for the current codebase.

# ============================================================================
# 1. SETUP & INSTALLATION
# ============================================================================

# Install Python dependencies
pip install numpy torch gymnasium kafka-python matplotlib

# Start Kafka (Required for training loop)
# Note: Keep this running in a separate terminal or background
./start_kafka.sh

# ============================================================================
# 2. TRAINING (radio_cortex_complete.py)
# ============================================================================

# --- Standard Training ---
# Trains PPO agent with default settings (3 cells, 20 UEs, flash_crowd scenario)
python3 radio_cortex_complete.py --mode train

# --- Custom Topology & Scenario ---
# Train on a larger network with a specific scenario and duration
python3 radio_cortex_complete.py --mode train \
    --num-ues 50 \
    --num-cells 5 \
    --scenario mobility_storm \
    --sim-time 20.0

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

# --- Spectrum Management ---
# Train with different system bandwidth (affects capacity)
python3 radio_cortex_complete.py --mode train --system-bandwidth-mhz 20.0

# --- Using Config File ---
# Load arguments from a JSON file
python3 radio_cortex_complete.py --mode train --config experiments/config_example.json

# ============================================================================
# 3. EVALUATION
# ============================================================================

# Evaluate trained model against baselines across all scenarios
python3 radio_cortex_complete.py --mode eval --model-path models/radio_cortex.pt

# Evaluate specific checkpoint on a single scenario
python3 radio_cortex_complete.py --mode eval \
    --model-path models/advanced_ppo.pt \
    --scenario urban_canyon

# ============================================================================
# 4. QUICK VERIFICATION & TESTING
# ============================================================================

# A. Verify E2 Interface (Ensures Kafka + ns-3 are sending real metrics)
python3 verify_kpm_data.py

# B. Fast training on mock environment (No ns-3/Kafka required, verified dimensions)
python3 quick_train.py

# C. Minimal real training (Very few steps, saves model)
./train_quick.sh

# ============================================================================
# 5. UTILITIES
# ============================================================================

# Interpret Policy (Saliency Map)
# Requires a model and a log file (action_logs.jsonl)
python3 interpret_policy.py --checkpoint models/radio_cortex.pt --action-index 0 --top-k 10

# Plot Training Progress (from action_logs.jsonl)
python3 -c "import json; import matplotlib.pyplot as plt; \
data = [json.loads(l) for l in open('action_logs.jsonl')]; \
plt.plot([d['reward'] for d in data]); plt.title('Training Reward'); \
plt.savefig('training_curve.png'); print('Saved training_curve.png')"

# Monitor Training Output
tail -f action_logs.jsonl

# ============================================================================
# 6. NS-3 SIMULATION ONLY (Manual Testing)
# ============================================================================

# Run simulation without Python agent (requires Kafka listener to see data)
./ns3 run "oran-congestion-scenario --numUes=20 --numCells=3 --simTime=10 --scenario=flash_crowd"
