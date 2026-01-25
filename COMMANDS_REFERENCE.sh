#!/bin/bash

# ORAN RL Training - Common Commands Reference

# ============================================================================
# SETUP & INSTALLATION
# ============================================================================

# Install Python dependencies
pip install numpy torch gym stable-baselines3

# Check if everything is installed
python3 << 'EOF'
import numpy as np
import torch
import gym
from stable_baselines3 import DQN, PPO
print("✓ All dependencies installed!")
EOF

# ============================================================================
# RUNNING SIMULATIONS
# ============================================================================

# DEFAULT: 3 cells, 20 UEs, 5 minutes simulation
./ns3 run "oran-congestion-scenario"

# CUSTOM: 5 cells, 30 UEs, 10 minutes
./ns3 run "oran-congestion-scenario --numUes=30 --numCells=5 --simTime=600"

# SMALL: 2 cells, 5 UEs, 2 minutes (for testing)
./ns3 run "oran-congestion-scenario --numUes=5 --numCells=2 --simTime=120"

# LARGE: 10 cells, 50 UEs, 20 minutes (challenging scenario)
./ns3 run "oran-congestion-scenario --numUes=50 --numCells=10 --simTime=1200"

# WITH LOGGING: See what's happening
./ns3 run "oran-congestion-scenario --verbose 1"

# ============================================================================
# TRAINING AGENTS
# ============================================================================

# QUICK START: Default DQN
python3 train.py

# DQN TRAINING: Various configurations
python3 train.py --agent dqn --episodes 100 --steps 500

# DQN WITH CUSTOM HYPERPARAMETERS
python3 train.py \
    --agent dqn \
    --episodes 200 \
    --steps 1000 \
    --learning-rate 0.001 \
    --gamma 0.99 \
    --epsilon-start 1.0 \
    --epsilon-min 0.01 \
    --memory-size 10000 \
    --batch-size 32

# DQN WITH LARGER MEMORY (better for stability)
python3 train.py --agent dqn --memory-size 50000 --batch-size 64

# DQN WITH SLOWER LEARNING (more stable)
python3 train.py --agent dqn --learning-rate 0.0001 --epsilon-decay 0.998

# PPO TRAINING: Generally better convergence
python3 train.py --agent ppo --episodes 200 --steps 1000

# PPO WITH CUSTOM HYPERPARAMETERS
python3 train.py \
    --agent ppo \
    --episodes 200 \
    --steps 1000 \
    --learning-rate 0.0003 \
    --gamma 0.99 \
    --gae-lambda 0.95 \
    --clip-ratio 0.2

# RANDOM BASELINE: For comparison
python3 train.py --agent random --episodes 50

# GPU ACCELERATION (if CUDA available)
python3 train.py --device cuda --agent dqn --episodes 100

# SAVE TO CUSTOM DIRECTORY
python3 train.py --agent dqn --model-dir my_models --log-dir my_logs

# LOAD CONFIG FROM FILE
python3 train.py --config training_config.json

# ============================================================================
# TRAINING ON DIFFERENT NETWORK SIZES
# ============================================================================

# Small network (test setup)
# Terminal 1:
./ns3 run "oran-congestion-scenario --numUes=5 --numCells=2 --simTime=120"
# Terminal 2:
python3 train.py --num-ues 5 --num-cells 2 --episodes 50

# Medium network (default)
# Terminal 1:
./ns3 run "oran-congestion-scenario --numUes=20 --numCells=3 --simTime=300"
# Terminal 2:
python3 train.py --num-ues 20 --num-cells 3 --episodes 100

# Large network (challenging)
# Terminal 1:
./ns3 run "oran-congestion-scenario --numUes=50 --numCells=5 --simTime=600"
# Terminal 2:
python3 train.py --num-ues 50 --num-cells 5 --episodes 200

# ============================================================================
# EVALUATION & TESTING
# ============================================================================

# EVALUATE FINAL MODEL
python3 evaluate.py --model models/agent_final.pth --episodes 10

# EVALUATE SPECIFIC CHECKPOINT
python3 evaluate.py --model models/agent_ep0050.pth --episodes 20

# DETAILED EVALUATION (more episodes)
python3 evaluate.py --model models/agent_final.pth --episodes 50 --steps 1000

# COMPARE TWO MODELS
python3 evaluate.py --model models/agent_ep0050.pth --output eval_50 --episodes 10
python3 evaluate.py --model models/agent_ep0100.pth --output eval_100 --episodes 10
# Then compare eval_50/evaluation_results.json vs eval_100/evaluation_results.json

# EVALUATE WITH CUSTOM NETWORK SIZE
python3 evaluate.py \
    --model models/agent_final.pth \
    --num-cells 3 \
    --num-ues 20 \
    --episodes 20

# ============================================================================
# MONITORING & ANALYSIS
# ============================================================================

# WATCH TRAINING IN REAL-TIME
tail -f logs/training.log

# CONTINUOUS MONITORING (refresh every 2 seconds)
watch -n 2 "tail -5 logs/training.log"

# VIEW METRICS SUMMARY
python3 << 'EOF'
import json
with open('logs/metrics.json') as f:
    data = json.load(f)
    s = data['summary']
    print(f"Episodes: {s['num_episodes']}")
    print(f"Mean Reward: {s['mean_reward']:.4f}")
    print(f"Throughput: {s['mean_throughput']:.2f} Mbps")
    print(f"Delay: {s['mean_delay']:.2f} ms")
    print(f"Fairness: {s['mean_fairness']:.4f}")
EOF

# PLOT TRAINING PROGRESS (requires matplotlib)
python3 << 'EOF'
import json
import matplotlib.pyplot as plt

with open('logs/metrics.json') as f:
    data = json.load(f)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(data['episodes'], data['episode_rewards'])
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training Reward')
plt.grid()

plt.subplot(1, 3, 2)
plt.plot(data['episodes'], data['mean_returns'])
plt.xlabel('Episode')
plt.ylabel('Mean Reward (10-ep window)')
plt.title('Smoothed Reward')
plt.grid()

plt.subplot(1, 3, 3)
plt.plot(data['episodes'], data['episode_lengths'])
plt.xlabel('Episode')
plt.ylabel('Steps')
plt.title('Episode Length')
plt.grid()

plt.tight_layout()
plt.savefig('training_progress.png', dpi=150)
print("Saved to training_progress.png")
EOF

# CHECK LOSS CONVERGENCE
python3 << 'EOF'
import json
with open('logs/metrics.json') as f:
    data = json.load(f)
    losses = data['losses'][-20:]  # Last 20 losses
    print(f"Loss trend (last 20 steps):")
    for i, loss in enumerate(losses):
        print(f"  {i:2d}: {loss:.6f}")
EOF

# ============================================================================
# ADVANCED WORKFLOWS
# ============================================================================

# CURRICULUM LEARNING: Train progressively harder scenarios
echo "Stage 1: Small network"
./ns3 run "oran-congestion-scenario --numUes=5 --numCells=2 --simTime=120" &
sleep 3
python3 train.py --num-ues 5 --num-cells 2 --episodes 50 --model-dir models_s1
pkill -f oran-congestion-scenario
sleep 2

echo "Stage 2: Medium network"
./ns3 run "oran-congestion-scenario --numUes=20 --numCells=3 --simTime=300" &
sleep 3
python3 train.py --num-ues 20 --num-cells 3 --episodes 100 --model-dir models_s2
pkill -f oran-congestion-scenario
sleep 2

echo "Stage 3: Large network"
./ns3 run "oran-congestion-scenario --numUes=30 --numCells=5 --simTime=600" &
sleep 3
python3 train.py --num-ues 30 --num-cells 5 --episodes 150 --model-dir models_s3
pkill -f oran-congestion-scenario

echo "Curriculum learning complete!"

# ============================================================================
# HYPERPARAMETER SWEEP
# ============================================================================

# Test different learning rates
for lr in 0.0001 0.0005 0.001 0.005; do
    echo "Training with learning_rate=$lr"
    python3 train.py \
        --agent dqn \
        --episodes 100 \
        --learning-rate $lr \
        --model-dir models_lr_$lr
done

# Test different epsilon decay rates
for decay in 0.99 0.995 0.998 0.999; do
    echo "Training with epsilon_decay=$decay"
    python3 train.py \
        --agent dqn \
        --episodes 100 \
        --epsilon-decay $decay \
        --model-dir models_decay_$decay
done

# ============================================================================
# TROUBLESHOOTING COMMANDS
# ============================================================================

# CHECK PORT IS LISTENING
netstat -an | grep 36421

# CHECK PYTHON INSTALLATION
python3 --version
python3 -m pip list | grep -E "torch|gym|numpy"

# TEST ENVIRONMENT CONNECTION
python3 << 'EOF'
from rl_oran_env import OranEnvironment
env = OranEnvironment()
if env.connect():
    print("✓ Successfully connected to ns-3!")
    state = env.reset()
    print(f"✓ Initial state shape: {state.shape}")
    next_state, reward, done, info = env.step(0)
    print(f"✓ Step successful! Reward: {reward:.4f}")
    env.disconnect()
else:
    print("✗ Failed to connect. Is ns-3 running?")
EOF

# TEST AGENT CREATION
python3 << 'EOF'
from rl_agent import DQNAgent, PPOAgent
try:
    dqn = DQNAgent(state_size=29, action_size=12)
    print("✓ DQN agent created")
    ppo = PPOAgent(state_size=29, action_size=12)
    print("✓ PPO agent created")
except Exception as e:
    print(f"✗ Error: {e}")
EOF

# ============================================================================
# CLEANUP & MANAGEMENT
# ============================================================================

# REMOVE OLD MODELS
rm models/agent_ep*.pth

# CLEAN ALL OUTPUTS
rm -rf models logs evaluation *.pth

# BACKUP TRAINED MODELS
mkdir -p backups
cp models/agent_final.pth backups/agent_final_$(date +%Y%m%d_%H%M%S).pth

# COMPRESS LOGS
tar -czf logs_backup.tar.gz logs/
rm -rf logs

# ============================================================================
# EXAMPLE: COMPLETE WORKFLOW
# ============================================================================

# FULL EXAMPLE: Start to finish
echo "=== ORAN RL Training Complete Workflow ==="

# Step 1: Build ns-3
echo "Step 1: Building ns-3..."
./ns3 build

# Step 2: Start ns-3
echo "Step 2: Starting ns-3 simulation..."
./ns3 run "oran-congestion-scenario --numUes=20 --numCells=3 --simTime=300" > /tmp/ns3.log 2>&1 &
NS3_PID=$!
sleep 3

# Step 3: Train agent
echo "Step 3: Training RL agent..."
python3 train.py --agent dqn --episodes 100 --steps 500

# Step 4: Stop ns-3
echo "Step 4: Stopping ns-3..."
kill $NS3_PID

# Step 5: Evaluate
echo "Step 5: Evaluating trained model..."
./ns3 run "oran-congestion-scenario --numUes=20 --numCells=3 --simTime=120" > /tmp/ns3.log 2>&1 &
NS3_PID=$!
sleep 3
python3 evaluate.py --model models/agent_final.pth --episodes 10
kill $NS3_PID

# Step 6: Report
echo "Step 6: Training Summary:"
python3 << 'REPORT'
import json
with open('logs/metrics.json') as f:
    data = json.load(f)
    s = data['summary']
    print(f"  Episodes trained: {s['num_episodes']}")
    print(f"  Mean reward: {s['mean_reward']:.4f}")
    print(f"  Max reward: {max(data['episode_rewards']):.4f}")
    print(f"  Throughput: {s['mean_throughput']:.2f} Mbps")
    print(f"  Fairness: {s['mean_fairness']:.4f}")
REPORT

echo "=== Training Complete! ==="
echo "Models saved to: models/"
echo "Logs saved to: logs/"
echo "Evaluation results: evaluation/"

# ============================================================================
# USEFUL ALIASES (add to ~/.bashrc)
# ============================================================================

# alias train-oran="cd /path/to/ns-3.46.1/scratch && python3 train.py"
# alias eval-oran="python3 evaluate.py --model models/agent_final.pth"
# alias logs-oran="tail -f logs/training.log"
# alias clean-oran="rm -rf models logs evaluation *.pth"

# Usage:
# $ train-oran --agent dqn
# $ eval-oran --episodes 20
# $ logs-oran
# $ clean-oran

# ============================================================================
# END
# ============================================================================

# This file contains ready-to-copy commands for all common tasks.
# Pick any command above and run it directly!
