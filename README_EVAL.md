# Radio-Cortex Evaluation Metrics

This document details the evaluation metrics used to benchmark the Radio-Cortex Intelligent RAN Controller against baselines.

## 1. Metric Categories

We categorize metrics based on the network layer they analyze, ensuring a holistic view of performance.

### 🌐 Quality of Service (QoS / Application Layer)
* **Throughput (Mbps):** Average successful data delivery rate to UEs.
* **End-to-End Delay (ms):** Average time for a packet to travel from source to destination.
* **Satisfied User Ratio (%):** Percentage of users meeting the SLA:
    *   *SLA Criteria:* Throughput > 1 Mbps AND Delay < 100 ms.

### 🛡️ Reliability & Stability
*   **Packet Loss Ratio (%):** Ratio of lost packets to total sent.
*   **Handover Success Rate (%):** Successful / Attempted handovers.

### ⚡ Resource Efficiency (Spectrum & Network)
*   **Spectrum Utilization (%):** Average usage of Resource Blocks (RBs) across all cells.
*   **Congestion Intensity (%):** Percentage of time where network utilization > 90%.
*   **Cell Edge Throughput (Mbps):** 5th Percentile throughput. Indicates how well the network serves users with poor coverage (fairness).
*   **Jain's Fairness Index (0-1):** Measures how equally resources are shared. 1.0 = perfect equality.
    *   *Formula:* $(\sum x_i)^2 / (n \cdot \sum x_i^2)$ where $x_i$ is UE throughput.
*   **Energy Efficiency (Mbps/Watt):** System Throughput / Total Power Consumption. Measures the "cost" of transmitting data.


### 📶 PHY / Wireless Layer
*   **Average SINR (dB):** Signal-to-Interference-plus-Noise Ratio.
*   **Average RSRP (dBm):** Reference Signal Received Power (Signal Strength).

### 🚀 Mobility Metrics
*   **Handover Count:** Number of cell switches per UE.

### 🖧 RIC / E2 Interface Metrics
*   **Control Stability (%):** 0-100 score measuring AI "jitter". High score means stable decisions; low score means frequent, large action changes.

### 🧠 Architecture & Compute Metrics
*   **Inference Time (ms):** Average time taken by the agent to compute an action. Critical for comparing model architectures (e.g., Transformer vs MLP) against O-RAN real-time constraints.

---

## 2. Composite Health Scores (Radar Chart)

To provide a quick "Health Check" of the network, we aggregate metrics into **6 composite scores** (0-100).

### 🏆 1. QoS Score (User Experience)
Combines how fast, responsive, and consistent the network felt to users. Includes tail-latency (p95) to capture stuttering.
*   **Formula:** `25% Throughput + 25% Delay + 15% p95 Delay + 35% Satisfied Users`

### 🛡️ 2. Reliability Score (Stability)
Penalizes both constant loss, sudden outages (Peak/Max Loss), service downtime, unstable mobility, and handover failures. Rewards fast recovery and stable control.
*   **Formula:** `30% Avg Loss + 10% Max Loss + 20% Downtime + 10% HO Stability + 20% HO Success + 10% Control Stability`
*   *Note:* Control Stability weight reduced to 10% (Baseline is naturally 100% stable).

### 🏗️ 3. Resource Score (Efficiency & Fairness)
Rewards high spectrum utilization AND efficiency, while ensuring fairness.
*   **Formula:** `10% Utilization + 30% Cell Edge + 30% Jain's Fairness + 30% Energy Efficiency`
*   *Note:* Utilization weight reduced (Baseline often has high utilization due to congestion, not efficiency).

### 📦 4. Buffer Score (Congestion Health)
Measures buffer occupancy and congestion spikes.
*   **Formula:** `60% Norm. Queue Length + 40% Congestion Intensity`

### 📡 5. PHY Score (Signal Quality)
Combined physical layer conditions.
*   **Formula:** `60% SINR + 40% RSRP`



### 🧠 7. Architecture Score (Model Efficiency)
Measures the "cost of intelligence" - how heavy the model is.
*   **Formula:** `50% Normalized Params + 50% Normalized Inference Speed`
*   **Penalties:** 
    *   0 score if Params > 1,000,000 (1M)
    *   0 score if Inference > 10ms (O-RAN limit)
    *   Baseline (Static) gets 100/100 (Efficient).


---

## 3. Data Quality Indicator

Every 50 steps, a data quality summary is printed:
```
[Data Quality] RSRP: 18/20 real | Cell: 18/20 real | HO: 0/20 with events
```

| Indicator | Good | Concerning |
|-----------|------|------------|
| RSRP | >80% real | <50% after 30s |
| Cell | >80% real | Persistently 0 |
| HO | 0 in non-mobility scenarios | 0 in `mobility_storm` |

---

## 4. Excluded Metrics & Limitations

The following metrics were considered but **not implemented** due to simulator constraints:

| Metric | Category | Reason for Exclusion |
| :--- | :--- | :--- |
| **Collision Rate** | Reliability | Requires MAC layer tracing with significant I/O overhead. |
| **Control Overhead** | Resource | Requires deep packet inspection, not feasible in real-time RL. |

---

## 5. How to Run Evaluation

For the full **🚀 CLI Reference Guide** (including all training and scenario flags), see the main [README.md](README.md).

### Quick Eval Commands

Run independently using `--model`:

```bash
# ── Baseline only ──
python3 radio_cortex_complete.py --mode eval --model base --scenario flash_crowd

# ── AI agent only ──
python3 radio_cortex_complete.py --mode eval --model bdh --scenario flash_crowd

# ── All scenarios in parallel ──
python3 radio_cortex_complete.py --mode eval --model bdh --scenario all --n-envs 12
```


> [!IMPORTANT]
> **Performance Note:** Evaluation relies on the ns-3 binary. Ensure you have compiled the **optimized build** (see README.md) to avoid slow evaluation speeds.


### Outputs

All results are **appended** to a single master CSV:

```
results/experiment_results.csv
```

Each row includes a timestamp, all metrics, and config parameters — allowing you to build a dataset incrementally across runs.

## ⚠️ Data Quality & Limitations

While Radio-Cortex provides a comprehensive metric suite, users should be aware of the current data source fidelity for certain fields:

| Metric | Source Fidelity | Note |
|--------|-----------------|------|
| **Throughput/Delay/Loss** | **High** | Measured directly from UE-level NetDevice trace sources in ns-3. |
| **SINR/RSRP/CQI** | **High** | Extracted from the LteAmc and LtePhy layers; highly accurate. |
| **RB Utilization** | **High** | Computed from actual per-cell RB allocations aggregated from UE metrics in `CollectCellMetrics`. |
| **Queue Length** | **Medium** | Estimated from per-cell packet loss counts. Buffer occupancy (`_buffer`) remains a placeholder as ns-3 LTE does not directly expose per-bearer queue depth. |
| **Handover Events** | **High** | Captured via RRC state machine transitions in real-time. |

> [!NOTE]
> `CollectCellMetrics` now derives `rbUtilization`, `numConnectedUes`, and `queueLength` from actual UE metrics via `servingCellId`. The only remaining placeholder is per-UE `bufferOccupancy`.

---

**Explore results interactively:**
```bash
# Standalone HTML dashboard
python3 -m http.server 8080
# Then open http://localhost:8080/dashboard.html
```

