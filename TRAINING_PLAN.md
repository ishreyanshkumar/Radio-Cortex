# Progressive Curriculum Training Plan for O-RAN RL Agent
## Scenario-by-Scenario Training with Percentage-Based Mixing

---

## 📋 Training Philosophy

**Core Principle**: Gradual introduction of complexity while maintaining exposure to mastered scenarios to prevent **catastrophic forgetting**.

**Key Strategy**: 
- Start with 100% on easiest scenario
- Add new scenarios progressively
- Keep 15-20% exposure to previous scenarios as "maintenance dose"
- Focus 70-80% on current learning target

---

# Progressive Curriculum Training Plan for O-RAN RL Agent
## Scenario-by-Scenario Training: The "Lean Power Suite"

---

## 📋 Training Philosophy

**Core Principle**: Gradual introduction of complexity while maintaining exposure to mastered scenarios to prevent **catastrophic forgetting**.

**Key Strategy**: 
- Start with 100% on easiest scenario (Flash Crowd).
- Add new scenarios progressively via weighted mixing.
- Keep exposure to previous scenarios as a "maintenance dose".
- Focus majority of training on the newest/hardest targets.

---

## 🎯 Complete Training Curriculum (Lean Power Suite)

The Lean Power Suite consists of **8 Stages** designed to guide the BDH agent from basic load balancing to full multi-objective generalization.

### **Stage 1: Flash Crowd (Bootstrap)**
**Duration**: 84,000 timesteps  
**Scenario Mix**: Flash Crowd: **100%**  
**Goal**: Learn basic cell-to-cell load balancing and handover triggering.

### **Stage 2: Sleepy Campus (Green RAN)**
**Duration**: 210,000 timesteps  
**Scenario Mix**: Flash Crowd: 20%, Sleepy Campus: **80%**  
**Goal**: Master energy-efficient power adaptation without sacrificing QoS.

### **Stage 3: Urban Canyon (PHY Robustness)**
**Duration**: 252,000 timesteps  
**Scenario Mix**: 15% each for Flash/Sleepy, Urban Canyon: **70%**  
**Goal**: Learn reactive signal recovery and SINR optimization in blocked environments.

### **Stage 4: Mobility Storm (Handover Mastery)**
**Duration**: 378,000 timesteps  
**Scenario Mix**: 10% each for previous scenarios, Mobility Storm: **70%**  
**Goal**: Optimize handover success rate under high-speed mobility conditions.

### **Stage 5: Traffic Burst (Massive Congestion)**
**Duration**: 504,000 timesteps  
**Scenario Mix**: 8% each for previous scenarios, Traffic Burst: **68%**  
**Goal**: Master scheduler optimization under severe buffer overload (5x-10x traffic).

### **Stage 6: Ambulance (Emergency Slicing)**
**Duration**: 504,000 timesteps  
**Scenario Mix**: 7% each for previous scenarios, Ambulance: **65%**  
**Goal**: Learn network slicing and QoS priority differentiation for emergency streams.

### **Stage 7: Spectrum Crunch (Spectral Efficiency)**
**Duration**: 630,000 timesteps  
**Scenario Mix**: 6% each for previous scenarios, Spectrum Crunch: **64%**  
**Goal**: Optimal resource management and maximizing bits/sec/Hz.

### **Stage 8: Consolidation (Multi-Mix Generalization)**
**Duration**: 1,050,000 timesteps  
**Scenario Mix**: **Uniform Mix (12% each)** across all key scenarios above.  
**Goal**: Achieve robust, production-ready generalization across the complete suite.

---

## 📊 Training Timeline Summary

| Stage | Timesteps | Primary Scenario | Mix Complexity | Goal |
|-------|-----------|------------------|----------------|------|
| 1 | 84,000 | Flash Crowd | Single | Load Balancing |
| 2 | 210,000 | Sleepy Campus | 2-scenario | Energy Efficiency |
| 3 | 252,000 | Urban Canyon | 3-scenario | PHY Robustness |
| 4 | 378,000 | Mobility Storm | 4-scenario | Handover Success |
| 5 | 504,000 | Traffic Burst | 5-scenario | Overload Management |
| 6 | 504,000 | Ambulance | 6-scenario | Emergency Slicing |
| 7 | 630,000 | Spectrum Crunch | 7-scenario | Spectral Efficiency |
| 8 | 1,050,000 | Power Suite Mix | All equal | Full Generalization |

**Total Training Time**: ~3.6M timesteps Focused on mission-critical scenarios.

---

## 🔧 Execution Guide

The curriculum is executed via the `scripts/train_curriculum.sh` wrapper, which manages checkpoint loading, scenario mixing, and GPU memory safety.

```bash
# Start the full 8-stage curriculum
bash scripts/train_curriculum.sh

# Resume from a specific stage (e.g., Stage 5)
bash scripts/train_curriculum.sh --start 5
```

---

## 📈 Performance Tracking

### Best Practices for Success
1. **Checkpointing**: The script saves models to `models/curriculum/stage_N.pt`. Always evaluate the latest stage.
2. **Evaluation**: Use `python3 radio_cortex_complete.py --mode eval --model bdh --scenario all` to verify zero-shot generalization to scenarios NOT included in the curriculum (e.g., `ping_pong`, `iot_tsunami`).
3. **Hardware**: This plan is optimized for high-parallelism (16-24 envs). If using fewer envs, total training time (wall-clock) will increase linearly.

---

## 🎓 Note on Zero-Shot Generalization
The "Lean Power Suite" specifically excludes `ping_pong` and `iot_tsunami` during training to test the agent's ability to generalize to novel challenges without retraining. A successful agent should achieve "Passed" scores on these scenarios despite never seeing them during the 7.2M steps of curriculum training.
