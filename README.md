# RL Agent Training for ORAN Congestion Control

Complete guide to training and evaluating reinforcement learning agents for O-RAN congestion control using ns-3 network simulation and E2 interface.

## 🎯 Project Overview

This project enables training RL agents to optimize radio access network (RAN) congestion control in an O-RAN architecture. The agent observes network metrics (KPM) and issues control commands (RC) to manage:

- **TX Power** - Adjust transmission power per cell
- **Scheduler Type** - Select scheduling algorithm (PF, MaxTput, etc.)
- **Resource Allocation** - Optimize RB distribution

**Optimization Goals:**
- Maximize throughput
- Minimize latency
- Improve fairness
- Reduce congestion

## 📁 Project Structure

```
scratch/
├── README.md                          # This file
├── QUICK_START.sh                     # One-command training startup
├── RL_TRAINING_GUIDE.md               # Detailed technical guide
│
├── oran-congestion-scenario.cc        # ns-3 network simulator
│
├── rl_oran_env.py                     # Gym environment interface
├── rl_agent.py                        # RL algorithm implementations
├── train.py                           # Main training loop
├── evaluate.py                        # Agent evaluation
│
├── training_config.json               # Default hyperparameters
│
├── models/                            # Trained model checkpoints
│   ├── config.json
│   ├── agent_ep0010.pth
│   ├── agent_ep0020.pth
│   └── agent_final.pth
│
├── logs/                              # Training logs
│   ├── training.log
│   └── metrics.json
│
└── evaluation/                        # Evaluation results
    └── evaluation_results.json
```

## 🚀 Quick Start (30 minutes)

### 1. Install Dependencies

```bash
# Python packages
pip install numpy torch gym stable-baselines3

# For CUDA acceleration (optional)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Build ns-3

```bash
cd /path/to/ns-3.46.1
./ns3 configure --enable-examples
./ns3 build
```

### 3. Terminal 1: Start ns-3 Simulation

```bash
cd /path/to/ns-3.46.1

# Run with default parameters (3 cells, 20 UEs, 300s simulation)
./ns3 run "oran-congestion-scenario --numUes=20 --numCells=3 --simTime=300"

# Or customize:
./ns3 run "oran-congestion-scenario --numUes=30 --numCells=5 --simTime=600"
```

### 4. Terminal 2: Train Agent

```bash
cd scratch/

# Quick start script (interactive)
bash QUICK_START.sh

# Or manual training
python3 train.py --agent dqn --episodes 100 --steps 500

# With custom config
python3 train.py --config training_config.json
```

### 5. Evaluate Trained Model

```bash
python3 evaluate.py --model models/agent_final.pth --episodes 10
```

## 📊 Available RL Algorithms

### DQN (Deep Q-Network)
- **Best for:** Discrete action spaces
- **Pros:** Stable, reliable training
- **Cons:** Off-policy, sample inefficient
- **Start:** `python train.py --agent dqn`

```python
# Hyperparameters
--learning-rate 0.001      # Adam LR
--gamma 0.99               # Discount factor
--epsilon-start 1.0        # Initial exploration
--epsilon-decay 0.995      # Exploration decay
--memory-size 10000        # Replay buffer
--batch-size 32            # Training batch
```

### PPO (Proximal Policy Optimization)
- **Best for:** Stable convergence, continuous/discrete actions
- **Pros:** Sample efficient, robust
- **Cons:** Requires more fine-tuning
- **Start:** `python train.py --agent ppo`

```python
# Hyperparameters
--learning-rate 0.0003
--gamma 0.99
--gae-lambda 0.95          # Advantage estimation
--clip-ratio 0.2           # Policy clipping
--epochs-per-step 3        # Update epochs
```

### Random Baseline
- **Purpose:** Compare against random policy
- **Start:** `python train.py --agent random`

## 🔧 State Space

The agent observes 29 normalized values (for 20 UEs, 3 cells):

```python
state = [
    # Cell metrics (3 values)
    rb_utilization[0],      # RB utilization cell 0 (%)
    rb_utilization[1],
    rb_utilization[2],
    
    # Queue metrics (3 values)
    queue_length[0],        # MAC queue cell 0 (packets)
    queue_length[1],
    queue_length[2],
    
    # TX Power (3 values)
    tx_power[0],            # TX power cell 0 (dBm, normalized)
    tx_power[1],
    tx_power[2],
    
    # UE metrics (20 values each)
    sinr[0..19],            # SINR per UE (dB, normalized)
    throughput[0..19],      # Throughput per UE (Mbps, normalized)
]
```

**Normalization ranges:**
```
RB Utilization:  0-100%
Queue Length:    0-1000 packets
TX Power:        0-46 dBm
SINR:           -10 to +30 dB
Throughput:      0-100 Mbps
```

## ⚙️ Action Space

12 discrete actions controlling network parameters:

```python
# TX Power Control per Cell
Action 0: Cell 0 - Decrease TX Power (-2 dBm)
Action 1: Cell 0 - Maintain TX Power
Action 2: Cell 0 - Increase TX Power (+2 dBm)
Action 3: Cell 1 - Decrease
Action 4: Cell 1 - Maintain
Action 5: Cell 1 - Increase
Action 6: Cell 2 - Decrease
Action 7: Cell 2 - Maintain
Action 8: Cell 2 - Increase

# Scheduler Selection
Action 9:  Proportional Fair (PF)
Action 10: Max Throughput
Action 11: Frequency Fair (FFOR)
```

JSON format sent to ns-3:
```json
{
  "timestamp": 123.456,
  "actions": [
    {"cell_id": 0, "action_type": "TxPower", "value": -2.0},
    {"cell_id": 0, "action_type": "Scheduler", "value": "PfFfMacScheduler"}
  ]
}
```

## 💡 Reward Function

Multi-objective reward combining 4 components:

```python
reward = 0.3*r_throughput + 0.3*r_latency + 0.2*r_fairness + 0.2*r_congestion

where:
  r_throughput = avg_throughput / max_throughput    # [0, 1]
  r_latency = exp(-avg_delay_ms / 50)               # [0, 1]
  r_fairness = Jain's Index of throughputs          # [0, 1]
  r_congestion = 1 - (avg_queue / max_queue)        # [0, 1]
```

**Congestion Penalty:**
```python
if queue_length > threshold:
    reward -= 0.5 * (queue_length - threshold) / max_queue
```

## 📈 Training Progress Monitoring

### View Live Logs
```bash
tail -f logs/training.log
```

### Check Metrics
```bash
# View full metrics
cat logs/metrics.json | python3 -m json.tool

# Extract specific metric
python3 -c "import json; m = json.load(open('logs/metrics.json')); print(m['summary']['mean_reward'])"
```

### Common Metrics
- **Episode Reward:** Total reward accumulated per episode (should increase)
- **Mean Return:** Rolling average of last 10 episodes
- **Loss:** DQN/PPO training loss (should decrease)
- **Mean Throughput:** Average throughput achieved (should increase)
- **Mean Delay:** Average latency (should decrease)
- **Mean Fairness:** Jain's fairness index (should increase toward 1.0)
- **Mean Congestion:** Average queue length (should decrease)

## 🔍 Evaluation

### Single Agent Evaluation
```bash
# Evaluate final trained agent
python3 evaluate.py --model models/agent_final.pth --episodes 20

# With custom parameters
python3 evaluate.py \
    --model models/agent_final.pth \
    --episodes 10 \
    --steps 1000 \
    --num-cells 3 \
    --num-ues 20
```

### Compare Multiple Agents
```bash
# Manually evaluate and compare
python3 evaluate.py --model models/agent_ep0050.pth --output eval_ep50
python3 evaluate.py --model models/agent_ep0100.pth --output eval_ep100
python3 evaluate.py --model models/agent_final.pth --output eval_final

# Compare results
cat eval_*/evaluation_results.json | grep mean_reward
```

## 🎓 Advanced Training Scenarios

### Curriculum Learning
Start simple, increase difficulty:

```bash
# Stage 1: Train on 3 cells, 10 UEs
python3 train.py --agent dqn --num-cells 3 --num-ues 10 --episodes 50 \
    --model-dir models_stage1

# Stage 2: Train on 3 cells, 20 UEs (transfer learning)
python3 train.py --agent dqn --num-cells 3 --num-ues 20 --episodes 50 \
    --model-dir models_stage2

# Stage 3: Train on 5 cells, 30 UEs
python3 train.py --agent dqn --num-cells 5 --num-ues 30 --episodes 50 \
    --model-dir models_stage3
```

### Hyperparameter Tuning

```bash
# Lower learning rate for stability
python3 train.py --learning-rate 0.0001 --episodes 200

# Higher gamma for long-term planning
python3 train.py --gamma 0.995 --episodes 100

# Larger memory for better replay
python3 train.py --memory-size 50000 --batch-size 64

# PPO with conservative updates
python3 train.py --agent ppo --clip-ratio 0.1 --epochs-per-step 5
```

### Different Congestion Scenarios

Modify ns-3 command to trigger different scenarios:

```bash
# Flash crowd (sudden traffic spike)
./ns3 run "oran-congestion-scenario --numUes=20 --scenario=flash_crowd"

# Mobility storm (rapid handovers)
./ns3 run "oran-congestion-scenario --numUes=20 --scenario=mobility_storm"

# Traffic burst
./ns3 run "oran-congestion-scenario --numUes=20 --scenario=traffic_burst"
```

## 🐛 Troubleshooting

### Connection Issues
```
ERROR: Cannot connect to ns-3 simulation
```
**Solution:**
- Ensure ns-3 is running: `./ns3 run "oran-congestion-scenario ..."`
- Check port 36421 is listening: `netstat -an | grep 36421`
- Verify firewall allows localhost connections

### No Progress in Training
```
Episode 1: reward=0.001, mean_return=0.001
Episode 2: reward=0.001, mean_return=0.001
```
**Solutions:**
- **Reduce state space:** Check if metrics are normalized correctly
- **Adjust learning rate:** Try 0.0003 or 0.0001
- **Increase batch size:** Use 64 or 128 for better gradient estimates
- **Check reward function:** Print `reward_breakdown` in info dict

### High Memory Usage
```
MemoryError: Unable to allocate
```
**Solutions:**
- Reduce `--memory-size` from 10000 to 5000
- Reduce `--batch-size` from 32 to 16
- Reduce `--episodes` and run multiple smaller trainings
- Use `--device cuda` if available

### Slow Training
**Solutions:**
- Use `--device cuda` for GPU acceleration
- Reduce `--steps` from 500 to 250
- Increase `--batch-size` for better GPU utilization
- Use fewer UEs in ns-3: `--numUes=10`

## 📚 References

### Papers
- **DQN:** Mnih et al., "Human-level control through deep reinforcement learning" (Nature 2015)
- **PPO:** Schulman et al., "Proximal Policy Optimization Algorithms" (ICLR 2017)
- **O-RAN:** O-RAN Alliance specifications (https://www.o-ran.org/)

### Tools
- **ns-3:** https://www.nsnam.org/ - Network simulator
- **Gym:** https://gym.openai.com/ - RL environment API
- **Stable Baselines3:** https://stable-baselines3.readthedocs.io/ - RL algorithms
- **PyTorch:** https://pytorch.org/ - Deep learning framework

### Related Work
- LTE/NR scheduling optimization
- Congestion control algorithms
- Resource allocation in cellular networks
- Multi-agent RL for network control

## 📝 Citation

If you use this code, please cite:

```bibtex
@misc{oran_rl_training,
  title={Reinforcement Learning for O-RAN Congestion Control},
  year={2026}
}
```

## 📧 Support

For issues or questions:
1. Check the [Troubleshooting](#-troubleshooting) section
2. Review training logs: `tail -f logs/training.log`
3. Check ns-3 output for simulation errors
4. Ensure all dependencies are installed: `pip install -r requirements.txt`

## 📄 License

This project uses ns-3 (GPL) and compatible open-source libraries.

---

**Happy Training! 🚀**

For detailed technical information, see [RL_TRAINING_GUIDE.md](RL_TRAINING_GUIDE.md).
