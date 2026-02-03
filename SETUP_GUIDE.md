# 🚀 Complete Setup Guide - ORAN RL Agent Training

## 📊 What You're Building

```
    ┌──────────────────────┐
    │  ns-3 Simulator      │
    │  (Network Sim)       │
    └──────────────────────┘
              ↕ TCP:36421
    ┌──────────────────────┐
    │  Python RL Agent     │
    │  (DQN/PPO)           │
    └──────────────────────┘
              ↕
    ┌──────────────────────┐
    │  Network Control     │
    │  TX Power, Scheduler │
    └──────────────────────┘
```

## ⚡ 5-Minute Setup

### [NEW] Quick Start (Current System)
The new system uses `radio_cortex_complete.py` and manages ns-3 automatically.

1.  **Start Kafka**: `./start_kafka.sh`
2.  **Run Training**: `python3 radio_cortex_complete.py --mode train`

---

### [LEGACY] Step 0: Start Kafka (Required)
The simulation requires a Kafka broker.

**Option A: Using Docker (Recommended)**
```bash
docker compose up -d
```

**Option B: Without Docker (If Docker fails)**
I have created a script to download and run Kafka directly:
```bash
bash run_kafka_native.sh
```
*Keep this terminal open.*

### Step 1: Check Prerequisites
```bash
# Check Python is installed
python3 --version      # Need 3.8+

# Check ns-3 is built
cd /path/to/ns-3.46.1
ls ./ns3               # Should exist
```

### Step 2: Install Python Dependencies
```bash
pip install --upgrade pip
pip install numpy torch gym stable-baselines3

# Verify installation
python3 -c "import torch; print('PyTorch:', torch.__version__)"
python3 -c "import gym; print('Gym installed')"
```

### Step 3: Copy Configuration
```bash
cd /path/to/ns-3.46.1/scratch
# All files should already be here:
ls *.py *.json *.md *.sh
```

## 🎮 Running Your First Training

### Terminal 1: Network Simulator
```bash
cd /path/to/ns-3.46.1

# Start ns-3 with default settings
./ns3 run "oran-congestion-scenario --numUes=20 --numCells=3 --simTime=300"

# You should see:
# INFO: E2 socket listening on port 36421
# INFO: Starting simulation...
# (stays running - that's normal)
```

### Terminal 2: RL Agent Training
```bash
cd /path/to/ns-3.46.1/scratch

# Option A: Interactive (recommended for first time)
bash QUICK_START.sh

# Option B: Direct training
python3 train.py \
    --agent dqn \
    --episodes 100 \
    --steps 500 \
    --learning-rate 0.001

# You should see:
# ======================================================================
# O-RAN Congestion Control - RL Training
# ======================================================================
# ...
# Episode 1: reward=0.245, length=500, mean_return=0.245
# Episode 2: reward=0.312, length=500, mean_return=0.278
# ...
```

### Monitor Progress
```bash
# Terminal 3: Watch training logs
cd /path/to/ns-3.46.1/scratch
tail -f logs/training.log

# Or check metrics
cat logs/metrics.json | python3 -m json.tool | head -50
```

## 📈 Expected Training Progress

```
Episode 1:    reward=0.15  ← Agent is randomly exploring
Episode 10:   reward=0.35  ← Starting to learn
Episode 50:   reward=0.60  ← Good progress
Episode 100:  reward=0.68  ← Converging
```

## 🔍 Key Files Explained

### `rl_oran_env.py` - Environment Interface
```python
env = OranEnvironment(
    num_cells=3,
    num_ues=20,
    ns3_host="127.0.0.1",
    ns3_port=36421
)
state = env.reset()                          # [29-dim vector]
next_state, reward, done, info = env.step(action)
```

### `rl_agent.py` - Learning Algorithms
```python
agent = DQNAgent(state_size=29, action_size=12)
action = agent.select_action(state)          # 0-11
agent.store_experience(state, action, reward, next_state, done)
agent.train_step()                           # Update weights
agent.save("model.pth")
```

### `train.py` - Training Loop
```bash
python3 train.py --agent dqn --episodes 100 --steps 500
# Creates: models/agent_ep0010.pth, models/agent_final.pth
#          logs/metrics.json, logs/training.log
```

### `evaluate.py` - Agent Testing
```bash
python3 evaluate.py --model models/agent_final.pth --episodes 10
# Output: evaluation/evaluation_results.json with KPIs
```

## 🎯 Understanding the Agent

### State Space (29 values, all normalized to [0,1])
```
Cell 0: RB Utilization, Queue Length, TX Power
Cell 1: RB Utilization, Queue Length, TX Power
Cell 2: RB Utilization, Queue Length, TX Power
UE 0-19: SINR (normalized)
UE 0-19: Throughput (normalized)
```

### Actions (Discrete, 0-11)
```
0-2:   TX Power control Cell 0 (decrease/maintain/increase)
3-5:   TX Power control Cell 1
6-8:   TX Power control Cell 2
9-11:  Scheduler selection (3 algorithms)
```

### Reward (Single value, typically 0.3-0.8)
```
Reward = 0.3×Throughput + 0.3×Latency + 0.2×Fairness + 0.2×Congestion
         (maximize)       (minimize)    (maximize)     (minimize)
```

## ✅ Checklist: Is Everything Working?

- [ ] **Terminal 1**: ns-3 shows "E2 socket listening on port 36421"
- [ ] **Terminal 2**: Python shows "Connected to ns-3 E2 interface"
- [ ] **Training**: Episode rewards increasing (not stuck at 0.1)
- [ ] **Logs**: `logs/training.log` contains episode data
- [ ] **Models**: `models/agent_ep*.pth` files being created

If any checkbox fails, see [Troubleshooting](#-troubleshooting) below.

## 📊 Analyze Results

### After Training Completes
```bash
# View final metrics
python3 -c "
import json
with open('logs/metrics.json') as f:
    data = json.load(f)
    s = data['summary']
    print(f\"Mean Reward: {s['mean_reward']:.3f}\")
    print(f\"Throughput: {s['mean_throughput']:.1f} Mbps\")
    print(f\"Latency: {s['mean_delay']:.1f} ms\")
    print(f\"Fairness: {s['mean_fairness']:.3f}\")
"

# Evaluate trained model
python3 evaluate.py \
    --model models/agent_final.pth \
    --episodes 20 \
    --output eval_results

# Compare episodes
python3 evaluate.py --model models/agent_ep0050.pth
python3 evaluate.py --model models/agent_ep0100.pth
# Check which performs better
```

## 🎓 Next: Try Different Approaches

### 1. Switch to PPO Algorithm
```bash
python3 train.py --agent ppo --episodes 200 --learning-rate 0.0003
```

### 2. Train with Different Network Size
```bash
# Larger network (more challenging)
./ns3 run "oran-congestion-scenario --numUes=30 --numCells=5"
python3 train.py --agent dqn --num-cells 5 --num-ues 30
```

### 3. Different Congestion Scenario
```bash
./ns3 run "oran-congestion-scenario --scenario=mobility_storm"
python3 train.py --agent dqn
```

### 4. Hyperparameter Search
```bash
# Lower learning rate
python3 train.py --learning-rate 0.0001 --episodes 200

# Larger experience replay
python3 train.py --memory-size 20000 --batch-size 64

# Faster exploration decay
python3 train.py --epsilon-decay 0.99 --episodes 100
```

## 🐛 Troubleshooting

### "Cannot connect to ns-3"
```
ERROR: Failed to connect to ns-3: [Errno 111] Connection refused
```
**Fix:**
- [ ] Is ns-3 running in Terminal 1?
- [ ] Is it past the "E2 socket listening" message?
- [ ] Try restarting ns-3: Ctrl+C, then re-run

### "Very low rewards (stuck at 0.1)"
```
Episode 10: reward=0.105
Episode 20: reward=0.098
```
**Fix:**
- [ ] Check if agent is actually receiving KPM reports
- [ ] Increase ns-3 simulation time: `--simTime=600`
- [ ] Reduce learning rate: `--learning-rate 0.0001`
- [ ] Check ns-3 logs for errors

### "Training very slow"
**Fix:**
- [ ] Use GPU (if available): `--device cuda`
- [ ] Reduce steps per episode: `--steps 250`
- [ ] Use fewer UEs in ns-3: `--numUes=10`

### "Out of memory"
```
MemoryError: Unable to allocate
```
**Fix:**
- [ ] Reduce replay buffer: `--memory-size 5000`
- [ ] Reduce batch size: `--batch-size 16`
- [ ] Reduce episodes per checkpoint: `--save-freq 50`

## 📈 Typical Results (100 episodes)

```
Training Time:     ~5-10 minutes (CPU)
Training Time:     ~1-2 minutes (GPU)

Final Reward:      0.65-0.75
Throughput:        40-50 Mbps
Latency:           20-30 ms
Fairness:          0.75-0.85
Congestion:        100-150 packets

Loss Trend:        ↓ (decreasing)
Episode Reward:    ↑ (increasing)
```

## 🔗 File Dependencies

```
train.py
  ├─ imports rl_oran_env.py
  │   └─ imports gym, json, socket, numpy
  ├─ imports rl_agent.py
  │   └─ imports torch (PyTorch)
  └─ loads training_config.json

evaluate.py
  ├─ imports rl_oran_env.py
  └─ imports rl_agent.py
```

All circular imports avoided ✓
All dependencies are pure Python or PyTorch ✓

## 🚀 Full Example: Start to Finish

```bash
# 1. Install dependencies (3 min)
pip install numpy torch gym stable-baselines3

# 2. Terminal 1: Start ns-3 (1 min)
cd /path/to/ns-3.46.1
./ns3 run "oran-congestion-scenario"

# 3. Terminal 2: Train agent (5-10 min)
cd scratch
python3 train.py --agent dqn --episodes 100

# 4. Terminal 3: Monitor (real-time)
tail -f logs/training.log

# 5. Evaluate after training (2 min)
python3 evaluate.py --model models/agent_final.pth --episodes 20

# Total: ~20 minutes for complete training + evaluation
```

## 📚 Documentation Map

```
README.md                  ← START HERE (user guide)
  ├─ Quick Start
  ├─ Algorithms
  ├─ Hyperparameter tuning
  └─ Troubleshooting

RL_TRAINING_GUIDE.md       ← Technical details
  ├─ Architecture
  ├─ State/Action/Reward
  └─ Advanced topics

IMPLEMENTATION_SUMMARY.md  ← What was built
  ├─ Files created
  ├─ How it works
  └─ Next steps

rl_oran_env.py            ← Code: Environment
rl_agent.py               ← Code: Algorithms
train.py                  ← Code: Training
evaluate.py               ← Code: Evaluation
```

## 💡 Pro Tips

1. **Start small:** Train on 3 cells, 10 UEs first
2. **Monitor early:** Check `logs/training.log` after episode 5
3. **Save checkpoints:** They're created every 10 episodes
4. **Compare algorithms:** Run DQN vs PPO on same config
5. **Ablate reward:** Modify weights in `rl_oran_env.py` line ~400

## 🎯 Success Criteria

✅ Training runs without errors for 10+ episodes
✅ Episode rewards increase (not flat or decreasing)
✅ Models are saved to `models/` directory
✅ Evaluation completes and shows KPI metrics
✅ Mean reward above 0.5 (out of 1.0)

---

## 🎉 You're Ready!

Follow the **5-Minute Setup** above and you'll have a fully trained agent in less than 20 minutes.

**Questions?** See README.md or RL_TRAINING_GUIDE.md

**Let's train! 🚀**
```bash
python3 train.py --agent dqn --episodes 100
```
