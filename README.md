# Radio-Cortex: O-RAN RL Congestion Control

Radio-Cortex is a closed-loop control system that uses Reinforcement Learning (RL) to optimize network parameters (Tx Power, Schedulers) in an O-RAN compliant ns-3 simulation. It demonstrates a real-time feedback loop where an RL agent (PPO) receives KPM (Key Performance Metrics) from ns-3 via Kafka and sends back RC (RAN Control) actions.


## 🚀 End-to-End Installation Guide

Follow these steps to set up the environment from scratch.

### 1. Clone the Repositories
```bash
# Clone the main project
git clone https://github.com/ishreyanshkumar/Radio-Cortex.git
cd Radio-Cortex

# Clone ns-allinone (Official Gitlab Repository)
git clone https://gitlab.com/nsnam/ns-3-allinone.git
cd ns-3-allinone
./download.py -n ns-3.46.1
cd ..
```

### 2. Set up Python Virtual Environment
```bash
# Create venv in the parent directory (or project root)
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Build ns-3 & Link Scenario
ns-3 requires specific libraries (like `librdkafka`) for the O-RAN interface to work.
```bash
# 1. Install librdkafka (system-level)
sudo apt-get install librdkafka-dev

# 2. Build ns-3
cd ns-3-allinone/ns-3.46.1

./ns3 configure -d optimized --enable-examples --enable-tests
./ns3 build

# 3. Link the Radio-Cortex scenario into ns-3 scratch
cd scratch
ln -sf ../../../oran-congestion-scenario.cc .
cd ../../..
```

### 4. Running the Training

#### A. Bootstrap & Start Kafka
Kafka and Zookeeper must be running for the E2 interface to function. If this is a fresh setup, use the bootstrap script:
```bash
bash scripts/run_kafka_native.sh
```
*Note: This will download Kafka binaries, install `librdkafka-dev`, and start the services.*

For subsequent starts, you can use:
```bash
./scripts/start_kafka.sh
```

#### B. Start Training
```bash
# Ensure venv is active
source .venv/bin/activate

### Quick Start
1. **Train Model**: `python3 radio_cortex_complete.py --mode train --scenario all`
2. **Clean Workspace**: `bash scripts/cleanup.sh`

# Run Training (Fast Parallel Mode)
python3 radio_cortex_complete.py --mode train --scenario all --n-envs 4 --total-timesteps 50000
```

## 🛠️ Utilities

The `scripts/` directory contains multi-language utilities to assist with development:

- **Cleanup**: `bash scripts/cleanup.sh` - Removes logs, residuals, and caches.
- **Log Analyzer (Go)**: `go run scripts/log_analyzer.go`

---

## 🎮 Usage & Workflows

### Advanced Training Configuration
You can customize the training hyperparameters and environment settings via command-line arguments. Radio-Cortex uses organized argument groups to separate core operations from network settings and advanced tuning.

#### 1. Core Operation
| Argument | Default | Description |
|:---|:---|:---|
| `--mode` | `train` | Operation mode: `train` or `eval`. |
| `--scenario` | `flash_crowd` | ns-3 Scenario (12 available). Use `all` for randomized training. |
| `--total-timesteps` | 100000 | Total training or evaluation steps. |
| `--n-envs` | 4 | Number of parallel environments (vectorized). |
| `--model` | `bdh` | Policy architecture: `bdh` (Transformer), `nn` (MLP), `t1`/`t2` (Experimental Transformers), `base` (baseline only, no AI). |
| `--model-path` | `models/radiocortex_{model}.pt` | Path to save/load model checkpoint. |
| `--device` | `None` | Compute device (`cpu` or `cuda`). |
| `--config` | `None` | Path to JSON config file to override any argument. |

#### 2. Network & Environment
| Argument | Default | Description |
|:---|:---|:---|
| `--num-ues` | 20 | Number of User Equipments (UEs). |
| `--num-cells` | 3 | Number of cells (eNodeBs). |
| `--sim-time` | 300.0 | Simulation duration per episode (seconds). **(See Speed Tips below)** |
| `--kpm-interval` | 100 | KPM Reporting Interval in ms. |
| `--system-bandwidth-mhz` | 10.0 | System Bandwidth (5.0, 10.0, 20.0). |

---

### ⚡ Performance & Accuracy Guide

#### What is `sim-time`?
`sim-time` is the **simulated duration** of each training episode.
- **Too High (e.g. 300s)**: Training is slow because each episode takes a long time to complete. If the agent makes a mistake early on, it "suffers" for 5 minutes of sim-time before resetting.
- **Too Low (e.g. 5s)**: Training is fast but inaccurate. The agent doesn't see the full lifecycle of a congestion event (which often takes 10-20s to peak).
- **Sweet Spot**: **20.0 - 40.0 seconds**. This is enough time for all scenarios to manifest while keeping worker throughput high.

#### How to Train Faster (Without Loss of Accuracy)
To reach convergence in minutes rather than hours, use these settings:

1. **Max out Parallelism**: Set `--n-envs` to the number of physical CPU cores you have. 
   - *Example*: `--n-envs 8` on an 8-core machine.
2. **Optimize `sim-time`**: Set `--sim-time 30.0`. This ensures frequent resets and exposure to different scenarios (if using `--scenario all`) without wasting time on "stable" network states.
3. **Use Optimized Build**: (Critical) Ensure you built ns-3 with `-d optimized`. 
4. **Tune Rollout Length**: Match your `--rollout-steps` to your budget. For fast iterations, `1024` or `512` is usually sufficient for convergence if `--n-envs` is high.

**Recommended "Fast & Robust" Command:**
```bash
python3 radio_cortex_complete.py --mode train --scenario all --n-envs 8 --sim-time 30.0 --total-timesteps 50000
```

#### 3. Advanced RL Tuning
| Argument | Default | Description |
|:---|:---|:---|
| `--learning-rate` | 3e-4 | PPO Learning rate. |
| `--batch-size` | 256 | Batch size for optimization updates. |
| `--rollout-steps` | 128 | Steps per rollout trajectory. |
| `--gamma` | 0.99 | Discount factor. |
| `--hidden-dim` | 256 | Network hidden dimension. |
| `--gae-lambda` | 0.95 | GAE normalization lambda. |
| `--clip-epsilon` | 0.2 | PPO clipping bound. |
| `--vf-coef` | 0.5 | Value function loss weight. |
| `--ent-coef` | 0.01 | Entropy regularization weight. |
| `--max-grad-norm` | 0.5 | Gradient clipping threshold. |
| `--checkpoint-interval` | 5 | Checkpoint frequency (updates). |
| `--log-interval` | 5 | Console log frequency (updates). |

### 🌍 Simulation Scenarios

Radio-Cortex supports 12 diverse scenarios that stress-test different aspects of RAN intelligence.

| Scenario | Type | Description | Key Metric |
| :--- | :--- | :--- | :--- |
| `flash_crowd` | Traffic | Sudden influx of users in one cell. | Congestion Intensity |
| `mobility_storm` | Mobility | High-speed users moving across cells. | HO Success Rate |
| `traffic_burst` | Traffic | Periodic surges in application data. | Peak Burst Loss |
| `handover_ping_pong` | Mobility | Users oscillating between cell boundaries. | HO Count per UE |
| `sleepy_campus` | Energy | Low-traffic night-time vs high-traffic day-time. | Energy Efficiency |
| `ambulance` | QoS | High-priority emergency stream in congested RAN. | Priority UE Delay |
| `adversarial` | Reliability | Rapid fluctuation in signal (shadowing). | Stability Score |
| `commuter_rush` | Scaled Mobility | Mass group handover (50+ UEs moving together). | RACH Failure Rate |
| `mixed_reality` | Slicing | Concurrent VR (Latent) and TCP (Bulk) users. | Slice Isolation |
| `urban_canyon` | PHY | Sudden signal blockage behind buildings. | Recovery Time |
| `iot_tsunami` | Scale | Massive device count (100+ UEs, small packets). | Scheduling Delay |
| `spectrum_crunch` | Resources | Multi-band management (Carrier Aggregation). | Spectral Efficiency |


### 🚀 CLI Training Reference

Train the O-RAN Intelligent Controller using PPO (Proximal Policy Optimization).

#### 1. Standard Training (Single Scenario)
```bash
# General usage
python3 radio_cortex_complete.py --mode train --scenario <name>

# Example: High-load Flash Crowd training
python3 radio_cortex_complete.py --mode train --scenario flash_crowd --num-ues 50
```

#### 2. Multi-Scenario Training (Domain Randomization)
Recommended for creating a "Universal Agent" that generalizes across all network conditions.
```bash
python3 radio_cortex_complete.py --mode train --scenario all --total-timesteps 50000
```

#### 3. Parallel & Performance Training
Scale simulations across CPU cores to drastically reduce wall-clock training time.
```bash
# Run 4 parallel simulations (requires optimized build)
python3 radio_cortex_complete.py --mode train --scenario all --n-envs 4

# Custom architecture (e.g., Transformer 1)
python3 radio_cortex_complete.py --mode train --model t1 --scenario all
```

#### 4. Advanced Hyperparameter Tuning
```bash
python3 radio_cortex_complete.py --mode train \
    --learning-rate 0.0001 \
    --gamma 0.995 \
    --batch-size 128 \
    --model-path models/custom_agent.pt
```

---

## 📊 CLI Evaluation & Benchmarking

Benchmarking compares the AI agent against the **Static-RAN** baseline. Results are appended to `results/experiment_results.csv`.

#### 1. Baseline Benchmark (AI disabled)
Run this first to establish a "ground truth" performance floor.
```bash
python3 radio_cortex_complete.py --mode eval --model base --scenario flash_crowd
```

#### 2. AI Agent Evaluation
```bash
# Evaluate the default BDH model on mobility storm
python3 radio_cortex_complete.py --mode eval --model bdh --scenario mobility_storm

# Evaluate in parallel across 4 environments (Faster)
python3 radio_cortex_complete.py --mode eval --model bdh --scenario all --n-envs 4
```

#### 3. Comparing Specific Architectures
```bash
# Compare Transformer 1 vs Transformer 2
python3 radio_cortex_complete.py --mode eval --model t1 --scenario flash_crowd
python3 radio_cortex_complete.py --mode eval --model t2 --scenario flash_crowd
```

#### 4. Interaction & Results Visualization
View metrics, radar charts, and comparison tables.
```bash
# Option A: Standalone HTML (Recommended)
python3 -m http.server 8080 -d results
# Open http://localhost:8080/dashboard.html

# Option B: Streamlit (Python required)
streamlit run results/visualize_results.py
```



**Metrics Tracked:**
(For detailed formulas and definitions, see [`README_EVAL.md`](README_EVAL.md))
(For the full technical breakdown of the KPM JSON Report structure, see [`KPM_REFERENCE.md`](KPM_REFERENCE.md))

| Category | Metric | Description |
|:---|:---|:---|
| **QoS (User Experience)** | Throughput | Average downlink data rate (Mbps). |
| | End-to-End Delay | Average time for packet delivery (ms). |
| | Jitter | Standard deviation of delay (variability). |
| | Satisfied User Ratio | % of users meeting SLA (Tput > 1Mbps, Delay < 100ms). |
| **Reliability** | Packet Loss Ratio | Ratio of lost packets to total sent. |
| | Peak Burst Loss | Max loss in any 1s window (instability indicator). |
| | Recovery Time | Time to return to <2% loss after failure. |
| | Handover Success Rate | Ratio of successful vs attempted handovers. |
| **Resource Efficiency** | Spectrum Utilization | Average usage of Resource Blocks (RBs). |
| | Congestion Intensity | % of time network utilization > 90%. |
| | Cell Edge Throughput | 5th percentile user throughput (fairness proxy). |
| | Jain's Fairness | Measure of resource distribution equality (0-1). |
| | Spectral Efficiency | System Throughput / Bandwidth (bits/sec/Hz). |
| | Energy Efficiency | System Throughput / Total Power (Mbps/Watt). |
| **PHY / Wireless** | Average SINR | Signal-to-Interference-plus-Noise Ratio (dB). |
| | Average RSRP | Reference Signal Received Power (Signal Strength, dBm). |
| **Mobility** | Handover Count | Number of cell switches per UE. |
| **RIC / E2 Interface** | E2 Loop Latency | Control loop response time (ms). |
| | RIC Message Overhead | E2 messages per second. |
| | Control Stability | AI decision consistency score (0-100). |

### 🧠 Hybrid Reward Engine

Radio-Cortex uses a multi-objective **"Hybrid Reward Engine"** that balances individual user experience with global network efficiency. The engine uses a **Two-Stage Safety Clipping** mechanism to prevent any single metric from dominating the gradients:

#### � 1. UE-Level Utility (User Satisfaction)
Computed per-UE and averaged across the network to ensure fairness. Uses E2SM-KPM `ue_metrics`.

*   **Throughput ($\alpha$-fairness):** $r_{tput} = \text{clip}\left( W_{tput} \cdot \log(1 + \frac{T}{T_{max}}), [-0.5, 5.0] \right)$
    Logarithmic utility ensures the agent prioritizes users with low throughput over those already well-served.
*   **Delay (Two-Tier):** $r_{delay} = \text{clip}\left( -\left( W_{d1} \cdot \frac{D}{D_{max}} + W_{d2} \cdot \frac{\max(0, D - D_{sla})^2}{D_{max}^2} \right), [-5.0, 0.0] \right)$
    Combines linear penalty for general delay and a **Quadratic SLA Barrier** that penalizes exponentially if delay exceeds 50ms.
*   **Packet Loss (IQX Model):** $r_{loss} = \text{clip}\left( -W_{loss} \cdot (\exp(\beta \cdot L) - 1), [-5.0, 0.0] \right)$
    Penalizes loss exponentially, capturing the non-linear impact of packet drops on QoE using the Independent Quality X (IQX) model.
*   **Spectral Efficiency:** $r_{se} = \text{clip}\left( W_{se} \cdot \log_2(1 + SINR), [0.0, 2.0] \right)$
    Uses Shannon Capacity to provide a "keep-alive" signal, rewarding good channel quality even during silent periods.

#### 🏗️ 2. Cell-Level Utility (Network Efficiency)
Computed per-cell to optimize infrastructure scaling. Uses E2SM-KPM `cell_metrics`.

*   **Energy Efficiency:** $r_{energy} = \text{clip}\left( -W_{energy} \cdot \frac{RB_{used}}{RB_{max}}, [-2.0, 0.0] \right)$
    Penalizes excessive Resource Block (RB) usage to encourage power-efficient scheduling.
*   **Load Balancing:** $r_{load} = \text{clip}\left( -W_{load} \cdot \sigma(Loads), [-2.0, 0.0] \right)$
    Penalizes high standard deviation in cell loads, driving the agent to distribute users across base stations.
*   **Queue Congestion:** $r_{queue} = \text{clip}\left( -W_{queue} \cdot \frac{Q}{Q_{max}}, [-2.0, 0.0] \right)$
    Penalizes growing buffers as an "early warning" signal to prevent delay spikes before they hit the application layer.

#### ⚖️ 3. Agent Stability (Action Smoothing)
*   **Action Smoothing:** $r_{smooth} = \text{clip}\left( -W_{smooth} \cdot \frac{\|a_t - a_{t-1}\|}{\text{range}(a)}, [-1.0, 0.0] \right)$
    Penalizes "jerky" or oscillatory control decisions to ensure network stability and reduce signaling overhead.

#### 🔴 Stage 2: Total Reward Clipping
Finally, the aggregate reward is clipped once more to ensure overall learning stability:
$$R_{total} = \text{clip}\left( \sum r_{ue} + \sum r_{cell} + r_{smooth}, [-10.0, 2.0] \right)$$

---

### Interpretability
Understand which input features (e.g., Queue Length vs Throughput) drove the agent's decisions.
```bash
python3 interpret_policy.py --checkpoint models/radio_cortex.pt
```

---

## 📂 Codebase Structure & Documentation

### 1. `radio_cortex_complete.py`
**Role:** Main Entry Point & Orchestrator.
This script manages the lifecycle of the training process, initializes the agent, and runs the main loop.

*   **`main()`**: Entry point. Parses arguments (`--mode train/eval/demo`) and routes execution.
*   **`train_radio_cortex(config, ...)`**: The core training loop.
    *   Creates the environment (`create_oran_env`).
    *   Initializes the `PPOTrainer`.
    *   Executes the training steps (rollout collection -> policy update).
    *   Saves the model to `models/radio_cortex.pt`.
*   **`RadioCortexAgent`**: High-level agent class.
    *   `get_rl_action()`: Queries the neural network for an action.
    *   `_heuristic_action()`: Fallback logic rule if RL is disabled.

### 2. `oran_ns3_env.py`
**Role:** Gymnasium Environment Wrapper.
converts ns-3 simulation into a standard OpenAI Gym interface (observation, action, reward).

*   **`ORANns3Env`**: The Gym Environment class.
    *   `step(action)`: Takes an RL action, sends it to ns-3, waits for the next KPM report, and returns (state, reward, done).
    *   `reset()`: Restarts the ns-3 simulation subprocess.
    *   `_compute_reward(e2_msg)`: Delegates to `RewardEngine`.
    *   **`RewardEngine`**: Hybrid reward logic combining 8 components: Throughput (Log Utility), Delay (Linear+SLA), Packet Loss (IQX), Spectral Efficiency, Energy Efficiency, Load Balancing, Queue Congestion, and Action Smoothing.
*   **`NS3Interface`**: Handles low-level communication.
    *   `start_simulation()`: Spawns the `./ns3 run ...` subprocess.
    *   `send_rc_control(actions)`: Serializes actions to JSON and sends via Kafka `e2_rc_control` topic.
    *   `receive_kpm_report()`: Polls Kafka `e2_kpm_stream` topic for metrics.

### 3. `rl_training_pipeline.py`
**Role:** Reinforcement Learning Algorithms (PPO).
Implements the PPO algorithm from scratch using PyTorch.

*   **`PPOTrainer`**: Implementation of PPO logic.
    *   `collect_rollout()`: Interacts with the env to gather a batch of experiences.
    *   `compute_gae()`: Calculates Generalized Advantage Estimation for stable learning.
    *   `update_policy()`: Performs the Gradient Descent update steps on the Actor and Critic networks.

### 4. `neural_networks.py`
**Role:** Neural Network Architectures.
Contains the PyTorch definitions for the RL agents.

*   **`ActorCritic`**: The default PPO network.
    *   `Actor`: Maps state -> action (Gaussian distribution).
    *   `Critic`: Maps state -> value estimate.

### 5. `interpret_policy.py`
**Role:** Model Interpretability.
Computes saliency maps (gradient * input) to understand which state features (e.g., UE throughput, Cell queue) influence the agent's decisions the most.

*   `_saliency()`: Backpropagates from the action mean to the input state.
*   usage: `python interpret_policy.py --checkpoint models/radio_cortex.pt`

### 6. `scripts/train_quick.sh`
**Role:** Fast Verification.
Runs `radio_cortex_complete.py` with minimal steps (100 timesteps) to verify the pipeline implementation quickly.

### 7. `oran-congestion-scenario.cc`
**Role:** ns-3 Simulation Scenario (C++).
The "Digital Twin" of the RAN. Implements the LTE/5G network, traffic generation, and E2 interface.

*   **`E2InterfaceManager`**: Manages the Kafka bridge.
    *   `SetupKafka()`: Configures `librdkafka` producer/consumer.
    *   `SendKpmReport()`: Collects metrics from all UEs, formats as JSON, and produces to Kafka.
    *   `ProcessRcCommand()`: Parses JSON control messages and applies changes (e.g., `SetTxPower`) to eNodeBs.
*   **`MetricCollector`**: Aggregates simulation traces.
    *   `ReportDlScheduling()`: Callback for DL MAC activity (estimates RB usage).
    *   `ReportAppRx()`: Callback for packet reception (calculates Delay).
    *   `GetAndResetUeMetrics()`: Returns accumulated stats and resets counters.

---

## 🔄 System Architecture

```mermaid
graph LR
    %% Data Flow Labels
    subgraph SIM ["ns-3 Simulation (C++)"]
        direction TB
        subgraph RAN ["RAN Infrastructure"]
            L12["PHY / MAC / RLC"]
            STACK["PDCP / RRC"]
        end
        
        subgraph E2 ["E2 Node Interface"]
            COLL["Metric Collector<br/>(KPM Agreggator)"]
            PROC["Action Processor<br/>(RC Handler)"]
        end
    end

    subgraph BUS ["Message Bus (Kafka)"]
        direction TB
        TOPIC_KPM[("e2_kpm_stream<br/>(Reports)")]
        TOPIC_RC[("e2_rc_control<br/>(Commands)")]
    end

    subgraph RIC ["Intelligent Controller (Python)"]
        direction TB
        subgraph ENV ["Gym Environment"]
            GYM["ORANns3Env"]
        end
        
        subgraph AI ["AI Brain (PPO)"]
            AGENT["PPO Agent"]
            NET["Neural Network"]
        end
    end

    %% Connections - Downlink / Metrics
    L12 --> COLL
    STACK --> COLL
    COLL -- "E2SM-KPM<br/>(JSON)" --> TOPIC_KPM
    TOPIC_KPM --> GYM
    GYM -- "State Vector" --> AGENT
    AGENT --> NET

    %% Connections - Uplink / Control
    NET --> AGENT
    AGENT -- "Action Vector" --> GYM
    GYM -- "E2SM-RC<br/>(JSON)" --> TOPIC_RC
    TOPIC_RC --> PROC
    PROC -- "SetTxPower / Sched" --> RAN

    %% Styling
    classDef simNode fill:#f8f9fa,stroke:#343a40,stroke-width:2px,color:#212529;
    classDef kafkaNode fill:#fff9db,stroke:#fcc419,stroke-width:2px,color:#212529;
    classDef pythonNode fill:#e7f5ff,stroke:#228be6,stroke-width:2px,color:#212529;
    classDef stackNode fill:#f1f3f5,stroke:#adb5bd,stroke-style:dashed;

    class SIM,RAN,E2,L12,STACK,COLL,PROC simNode;
    class BUS,TOPIC_KPM,TOPIC_RC kafkaNode;
    class RIC,ENV,AI,GYM,AGENT,NET pythonNode;
    class RAN,E2,ENV,AI stackNode;
```

---

## 🐛 Troubleshooting

### "Failed to connect to Kafka"
-   Ensure you ran `./scripts/start_kafka.sh`.
-   Check logs: `cat kafka.log` or `cat zookeeper.log`.
-   Verify ports: `netstat -tuln | grep 9092`

### "No KPM data received"
-   Wait a few seconds for ns-3 to initialize.
-   Check `ns3.log` (created in project root) to see if the simulation crashed.
-   Ensure `oran-congestion-scenario` compiled successfully.

### 🖥️ Convergence Dashboard (Rich UI)

Radio-Cortex features a high-fidelity convergence dashboard that replaces standard text logs with mission-critical training metrics.

#### Key Metrics to Observe:
- **Reward**: The primary optimization goal. Should show an **Upward Trend (↗)** over the first 50-100 updates.
- **Explained Variance (Expl Var)**: Measures the Accuracy of the RIC's internal reward predictions.
    - **Value range**: `1.0` (Perfect), `0.0` (Guessing Mean), `< 0.0` (Still exploring/Worse than mean).
    - **Coloring**: `Green` (>0.8) indicates a "Converged" critic; `Red` (<0.4 or negative) is normal for the first ~50 updates.
- **Entropy**: Measures the agent's confidence. Should gradually decrease as the agent becomes more specialized at handling specific congestion scenarios.
- **Activity Heartbeat**: A pulsing `●` light showing real-time data influx from ns-3.

---

## 📈 Training Dashboard Guide

| Metric | Target Trend | What it means |
|:---|:---:|:---|
| **Reward** | ↗ Growing | The agent is successfully reducing congestion and improving user QoS. |
| **Trend** | ↗ (Green) | Recent updates have improved performance by 5% or more. |
| **Expl Var** | → 0.9 | High values mean the agent correctly predicts the "cost" of its actions. |
| **Entropy** | ↘ Decreasing | The agent is narrowing down its optimal control strategy (good). |
| **Policy Loss** | ⇄ Oscillating | Normal in PPO; indicates the agent is exploring different tradeoffs. |

> [!TIP]
> **When to Stop?** Stop training when **Explained Variance > 0.8** and the **Reward Trend** stabilizes (→) for 10 consecutive updates. This indicates a "Converged" model.

---

## 📊 Monitoring & Analysis

### 1. Plot Training Metrics
You can plot the training progress (rewards, throughput) using this Python script. It reads from `telemetry/action_logs.jsonl`.
```python
import json
import matplotlib.pyplot as plt

with open('telemetry/action_logs.jsonl') as f:
    # Read line by line
    data = [json.loads(line) for line in f]

rewards = [d['reward'] for d in data]
steps = [d['step'] for d in data]

plt.figure(figsize=(10, 5))
plt.plot(steps, rewards)
plt.xlabel('Steps')
plt.ylabel('Reward')
plt.title('Training Progress')
plt.savefig('training_plot.png')
print("Saved training_plot.png")
```

### 2. View Summary Stats
Quickly check the average metrics from the logs:
```bash
python3 -c "import json; import numpy as np; 
data = [json.loads(l) for l in open('telemetry/action_logs.jsonl')]; 
print(f'Mean Reward: {np.mean([d[\"reward\"] for d in data]):.4f}')"
```

---

## 🔬 Advanced Workflows

### Curriculum Learning Loop
Train on progressively harder scenarios (Small -> Medium -> Large network).

```bash
# Stage 1: Small Network
python3 radio_cortex_complete.py --mode train --num-ues 5 --num-cells 2 --total-timesteps 5000 --model-path models/stage1.pt

# Stage 2: Medium Network (Load Stage 1 model?? - currently training from scratch)
# To implement true curriculum, you'd load the previous model.
python3 radio_cortex_complete.py --mode train --num-ues 20 --num-cells 3 --total-timesteps 10000 --model-path models/stage2.pt

# Stage 3: Large Network
python3 radio_cortex_complete.py --mode train --num-ues 40 --num-cells 5 --total-timesteps 20000 --model-path models/stage3.pt
```

### Batch Experiments (Bash Loop)
Run multiple experiments with different learning rates.
```bash
for lr in 0.0001 0.0003 0.001; do
    echo "Training with LR=$lr"
    python3 radio_cortex_complete.py --mode train --learning-rate $lr --model-path models/lr_${lr}.pt
done
```

