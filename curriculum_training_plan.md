# Progressive Curriculum Training Plan for O-RAN RL Agent
## Sequential Scenario Mastery with Checkpoint Transfer

---

## 📋 Training Philosophy

**Core Principle**: Sequential introduction of complexity — master each scenario fully before moving on. Knowledge is preserved via **checkpoint chaining**: each stage loads the previous stage's weights as initialization.

**Key Strategy**:
- Train on **one primary scenario per stage** (the `--scenario` flag)
- Load previous stage's checkpoint to transfer learned policies
- The built-in **reward curriculum** (Level 0→1→2) handles intra-episode difficulty ramp-up automatically
- Generalization is achieved in Stage 13 via `--scenario all` (uniform random across all 12)

**Anti-Forgetting Mechanism**: Rather than percentage-based mixing, the BDH Transformer's Scale-Free architecture naturally retains knowledge across scenarios because:
1. Cell/UE embeddings are shared — patterns transfer across scenarios
2. Checkpoint chaining preserves learned weights
3. Stage 13 (Multi-Mix) explicitly rehearses all scenarios

---

## 🔧 Shared Hyperparameters (All Stages)

| Parameter | Value | Rationale |
|:---|:---|:---|
| Model | `bdh` | Scale-Free Transformer |
| Environments | `16` | Parallel rollout |
| Sim Time | `30.0s` | Long episodes for delayed-reward learning |
| Learning Rate | `1e-4` | Stable gradient updates |
| Batch Size | `512` | Low variance with 16 envs |
| Rollout Steps | `256` | Rich gradient estimates |
| Gamma | `0.995` | Long horizon for network effects |
| GAE Lambda | `0.95` | Standard advantage estimation |
| Clip Epsilon | `0.15` | Tighter clipping for stability |
| Entropy Coef | `0.005` | BDH logstd adapts exploration |
| VF Coef | `0.5` | Standard value function weight |
| Max Grad Norm | `0.5` | Gradient clipping |
| Hidden Dim | `256` | Network width |

---

## 🎯 Complete Training Curriculum

### **Stage 1: Flash Crowd (Bootstrap)**
**Timesteps**: 10,000
**Scenario**: `flash_crowd`
**Checkpoint**: None (train from scratch) → `models/curriculum/stage_1.pt`

**Goal**: Learn basic throughput control and load balancing under sudden demand spikes.

**What the Agent Learns**:
- Power adjustment to redistribute load
- Scheduler weight tuning for fairness
- Baseline reward curriculum ramp-up (Level 0→1→2)

**Success Criteria**:
- Average reward > -2.0 over last 500 steps
- Jain's Fairness Index > 0.80
- Throughput degradation < 30% during flash crowd peak

---

### **Stage 2: Sleepy Campus (Energy)**
**Timesteps**: 16,000
**Scenario**: `sleepy_campus`
**Checkpoint**: `stage_1.pt` → `stage_2.pt`

**Goal**: Learn energy-efficient power adaptation during low-traffic periods.

**What the Agent Learns**:
- Power down-regulation when traffic is low
- Temporal traffic pattern recognition
- Energy efficiency without sacrificing coverage

**Success Criteria**:
- Energy efficiency improvement > 20% over baseline
- No power oscillation (smooth transitions)
- Coverage maintained (RSRP > -110 dBm)

---

### **Stage 3: Urban Canyon (PHY Recovery)**
**Timesteps**: 20,000
**Scenario**: `urban_canyon`
**Checkpoint**: `stage_2.pt` → `stage_3.pt`

**Goal**: Learn reactive SINR recovery under signal degradation from urban multipath.

**What the Agent Learns**:
- Rapid power boost in response to SINR drops
- Hysteresis tuning for shadow-fading zones
- Recovery without overcorrection

**Success Criteria**:
- Recovery time < 1.0s after SINR drop
- RLF rate < 5%
- SINR stability variance < 6 dB

---

### **Stage 4: Mobility Storm (Handovers)**
**Timesteps**: 30,000
**Scenario**: `mobility_storm`
**Checkpoint**: `stage_3.pt` → `stage_4.pt`

**Goal**: Optimize handover success rate under high-velocity UE movement.

**What the Agent Learns**:
- Hysteresis parameter tuning for HO triggering
- Pre-emptive power adjustment for moving UEs
- Balancing HO frequency vs. signal quality

**Success Criteria**:
- Handover success rate > 92%
- RLF count < 3 per 100 UEs
- Ping-pong rate < 15%

---

### **Stage 5: Traffic Burst (Queue Management)**
**Timesteps**: 40,000
**Scenario**: `traffic_burst`
**Checkpoint**: `stage_4.pt` → `stage_5.pt`

**Goal**: Master scheduler optimization under extreme 5× overload bursts.

**What the Agent Learns**:
- Scheduler weight redistribution under congestion
- Queue management via power and priority
- Preventing starvation of edge UEs

**Success Criteria**:
- Packet loss ratio < 8% during burst
- 5th percentile throughput > 2 Mbps (no starvation)
- Queue length stabilizes within 2s of burst onset

---

### **Stage 6: Mixed Reality (Slicing)**
**Timesteps**: 40,000
**Scenario**: `mixed_reality`
**Checkpoint**: `stage_5.pt` → `stage_6.pt`

**Goal**: Learn multi-objective optimization — VR latency vs. download throughput.

**What the Agent Learns**:
- Network slicing via differentiated priority weights
- QoS differentiation (latency-critical vs. throughput-critical)
- Slice isolation (one slice's load doesn't crash the other)

**Success Criteria**:
- VR slice: 99% packets < 20ms latency
- Download slice: Mean throughput > 8 Mbps
- Cross-slice correlation < 0.2

---

### **Stage 7: Adversarial (Stability)**
**Timesteps**: 36,000
**Scenario**: `adversarial`
**Checkpoint**: `stage_6.pt` → `stage_7.pt`

**Goal**: Develop robust policies under unpredictable channel fluctuations and interference.

**What the Agent Learns**:
- Resilience to random SINR drops
- Conservative power management under uncertainty
- Avoiding overreaction to noise

**Success Criteria**:
- Throughput resilience ratio > 0.70 (degraded/normal)
- Power adaptation within 500ms of SINR change
- No catastrophic failures (reward > -8.0)

---

### **Stage 8: Handover Ping-Pong (Hysteresis)**
**Timesteps**: 50,000
**Scenario**: `handover_ping_pong`
**Checkpoint**: `stage_7.pt` → `stage_8.pt`

**Goal**: Eliminate unnecessary handover oscillations at cell boundaries.

**What the Agent Learns**:
- Fine-grained hysteresis tuning for boundary UEs
- Temporal action smoothness (avoiding rapid parameter flips)
- Balancing HO avoidance vs. signal quality

**Success Criteria**:
- Ping-pong rate < 10% of total handovers
- Handover efficiency > 88%
- SINR stability variance < 4 dB at cell edges

---

### **Stage 9: Commuter Rush (Scale)**
**Timesteps**: 60,000
**Scenario**: `commuter_rush`
**Checkpoint**: `stage_8.pt` → `stage_9.pt`

**Goal**: Handle synchronized mass handover avalanche from commuter movement patterns.

**What the Agent Learns**:
- Multi-cell coordination during mass HO events
- Load balancing across cells under directional migration
- Predictive power adjustment for known movement corridors

**Success Criteria**:
- RLF rate during mass HO < 8%
- Cell load balance ratio (max/min) < 2.0
- Service continuity > 95% during peak HO period

---

### **Stage 10: IoT Tsunami (Device Scale)**
**Timesteps**: 70,000
**Scenario**: `iot_tsunami`
**Checkpoint**: `stage_9.pt` → `stage_10.pt`

**Goal**: Optimize scheduling under massive device count with control-plane pressure.

**What the Agent Learns**:
- Priority scheduling with many low-throughput devices
- RACH congestion mitigation via power management
- Balancing IoT control overhead vs. data throughput

**Success Criteria**:
- RACH collision rate < 25%
- Signaling overhead < 20% of total bytes
- Packet delivery ratio > 95% for IoT packets

---

### **Stage 11: Ambulance (QoS Priority)**
**Timesteps**: 80,000
**Scenario**: `ambulance`
**Checkpoint**: `stage_10.pt` → `stage_11.pt`

**Goal**: Guarantee near-perfect reliability for emergency priority UE.

**What the Agent Learns**:
- Extreme priority weight differentiation
- Preemptive resource reservation for critical UEs
- Acceptable background sacrifice without starvation

**Success Criteria**:
- 99.999%ile latency < 10ms for ambulance UE
- PDR > 99.999% for priority traffic
- Background UEs maintain > 1 Mbps average

---

### **Stage 12: Spectrum Crunch (Efficiency)**
**Timesteps**: 100,000
**Scenario**: `spectrum_crunch`
**Checkpoint**: `stage_11.pt` → `stage_12.pt`

**Goal**: Maximum spectral efficiency under severe bandwidth constraints.

**What the Agent Learns**:
- Aggressive scheduler optimization for spectrum reuse
- Power control for interference minimization
- Carrier aggregation coordination

**Success Criteria**:
- Spectral efficiency > 4 bits/s/Hz
- Aggregated throughput > 35 Mbps
- Inter-cell interference below threshold

---

### **Stage 13: Multi-Mix (Generalization)**
**Timesteps**: 100,000
**Scenario**: `all` (uniform random across all 12 scenarios)
**Checkpoint**: `stage_12.pt` → `stage_13.pt`

**Goal**: Achieve robust performance across all scenarios simultaneously.

**What the Agent Learns**:
- Cross-scenario generalization
- Fast adaptation to scenario switches
- Balanced performance (no single-scenario dominance)

**Success Criteria**:
- All scenarios meet ≥80% of individual success criteria
- No scenario causes catastrophic failure (reward > -6.0)
- Smooth performance across random scenario switches

---

### **Stage 14: Adaptive (Continuous Improvement)**
**Timesteps**: ∞ (ongoing)
**Scenario**: `all` (with reduced LR and entropy)
**Checkpoint**: `stage_13.pt` → continual updates

**Suggested Command** (manual):
```bash
python3 radio_cortex_complete.py --mode train --scenario all \
    --model bdh --n-envs 16 --total-timesteps 200000 \
    --model-path models/curriculum/stage_13.pt \
    --learning-rate 5e-5 --ent-coef 0.002 --sim-time 30.0
```

**Goal**: Continual refinement on weakest scenarios.

---

## 📊 Training Timeline Summary

| Stage | Timesteps | Cumulative | Scenario | Focus Area |
|:---:|---:|---:|:---|:---|
| 1 | 10,000 | 10,000 | `flash_crowd` | Throughput + Load Balancing |
| 2 | 16,000 | 26,000 | `sleepy_campus` | Energy Efficiency |
| 3 | 20,000 | 46,000 | `urban_canyon` | SINR / PHY Recovery |
| 4 | 30,000 | 76,000 | `mobility_storm` | Handover Control |
| 5 | 40,000 | 116,000 | `traffic_burst` | Queue / Congestion |
| 6 | 40,000 | 156,000 | `mixed_reality` | Multi-QoS Slicing |
| 7 | 36,000 | 192,000 | `adversarial` | Robustness |
| 8 | 50,000 | 242,000 | `handover_ping_pong` | Hysteresis Tuning |
| 9 | 60,000 | 302,000 | `commuter_rush` | Mass Mobility |
| 10 | 70,000 | 372,000 | `iot_tsunami` | Device Scale |
| 11 | 80,000 | 452,000 | `ambulance` | URLLC Priority |
| 12 | 100,000 | 552,000 | `spectrum_crunch` | Spectral Efficiency |
| 13 | 100,000 | 652,000 | `all` | Generalization |
| 14 | ∞ | ∞ | `all` | Continual Improvement |

**Total Timesteps to Deployment**: **652,000** (Stages 1–13)
**Estimated Wall-Clock**: 4–8 hours on 16-env / 8-core machine

---

## 🚀 Running the Curriculum

### Full Run (Stage 1–13)
```bash
bash scripts/train_curriculum.sh
```

### Resume from Stage N
```bash
bash scripts/train_curriculum.sh --start 5    # Resume from Stage 5
```

### Dry Run (preview plan)
```bash
bash scripts/train_curriculum.sh --dry-run
```

### Override Environment Count
```bash
bash scripts/train_curriculum.sh --n-envs 8   # Use 8 environments
bash scripts/train_curriculum.sh --device cpu  # Force CPU
```

---

## 🔗 How Checkpoint Chaining Works

```
Stage 1 (from scratch)
    └─► stage_1.pt
         └─ cp ─► stage_2.pt (initialized with Stage 1 weights)
                   └─ Train on sleepy_campus
                        └─► stage_2.pt (updated)
                             └─ cp ─► stage_3.pt
                                       └─ Train on urban_canyon
                                            └─► stage_3.pt (updated)
                                                 └─ ... continues to stage_13.pt
```

At each stage:
1. Copy `stage_{N-1}.pt` → `stage_{N}.pt`
2. Train on new scenario (loads `stage_{N}.pt` as `--model-path`)
3. Save updated weights to `stage_{N}.pt`

This means **Stage 13 contains knowledge from all 12 previous stages** plus generalization training.

---

## 🎓 Built-in Reward Curriculum (Automatic)

Within each stage, the `RewardEngine` in `oran_ns3_env.py` applies a 3-level reward curriculum:

| Level | Trigger | Active Rewards |
|:---:|:---|:---|
| 0 | Start of episode | Throughput + Bias only |
| 1 | Success rate > 40% for 50 steps | + Delay + Queue penalties |
| 2 | Success rate > 60% for 100 steps | + Loss + Energy + Load (full) |

This prevents the agent from being overwhelmed by too many penalty signals early in learning. As performance improves, harder objectives are progressively unlocked.

---

## 📈 Evaluation After Training

### Full Benchmark Suite
```bash
python3 radio_cortex_complete.py --mode eval \
    --model bdh --model-path models/curriculum/stage_13.pt \
    --scenario all
```

### Single Scenario Test
```bash
python3 radio_cortex_complete.py --mode eval \
    --model bdh --model-path models/curriculum/stage_13.pt \
    --scenario flash_crowd
```

---

## ⚠️ Important Notes

1. **Checkpoint Dependency**: Each stage requires the previous stage's checkpoint. If `stage_{N-1}.pt` is missing, that stage trains from scratch (with a warning).

2. **Stage Duration Scaling**: Later stages have more timesteps because:
   - Harder scenarios need more exploration
   - More parameters to fine-tune without forgetting earlier skills
   - The 16-env parallelism makes this feasible

3. **Rollback**: If Stage N degrades performance, you can:
   - Re-run from a specific stage: `bash scripts/train_curriculum.sh --start N`
   - All previous checkpoints are preserved in `models/curriculum/`

4. **Hardware Requirements**:
   - GPU recommended for 16-env training
   - ~8 GB VRAM for BDH with 16 envs
   - CPU-only is supported but ~4× slower
