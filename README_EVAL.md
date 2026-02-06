# Radio-Cortex Evaluation Metrics

This document details the evaluation metrics used to benchmark the Radio-Cortex Intelligent RAN Controller against baselines.

## 1. Metric Categories

We categorize metrics based on the network layer they analyze, ensuring a holistic view of performance.

### 🌐 Quality of Service (QoS / Application Layer)
* **Throughput (Mbps):** Average successful data delivery rate to UEs.
* **End-to-End Delay (ms):** Average time for a packet to travel from source to destination.
* **Jitter (ms):** Standard deviation of the delay (variability).
* **Satisfied User Ratio (%):** Percentage of users meeting the SLA:
    *   *SLA Criteria:* Throughput > 1 Mbps AND Delay < 100 ms.

### 🛡️ Reliability & Stability
*   **Packet Loss Ratio (%):** Ratio of lost packets to total sent.
*   **Peak Burst Loss (%):** Maximum packet loss observed in any rolling 1-second window. High burst loss indicates instability.
*   **Recovery Time (s):** Time taken to return to < 2% loss after a failure (> 10% loss).
*   **Handover Success Rate (%):** Successful / Attempted handovers.

### ⚡ Resource Efficiency (Spectrum & Network)
*   **Spectrum Utilization (%):** Average usage of Resource Blocks (RBs) across all cells.
*   **Congestion Intensity (%):** Percentage of time where network utilization > 90%.
*   **Cell Edge Throughput (Mbps):** 5th Percentile throughput. Indicates how well the network serves users with poor coverage (fairness).
*   **Jain's Fairness Index (0-1):** Measures how equally resources are shared. 1.0 = perfect equality.
    *   *Formula:* $(\sum x_i)^2 / (n \cdot \sum x_i^2)$ where $x_i$ is UE throughput.
*   **Spectral Efficiency (bits/sec/Hz):** System Throughput / Bandwidth. Measures how efficiently the available spectrum is used.
*   **Energy Efficiency (Mbps/Watt):** System Throughput / Total Power Consumption. Measures the "cost" of transmitting data.


### 📶 PHY / Wireless Layer
*   **Average SINR (dB):** Signal-to-Interference-plus-Noise Ratio.
*   **Average RSRP (dBm):** Reference Signal Received Power (Signal Strength).

### 🚀 Mobility Metrics
*   **Handover Count:** Number of cell switches per UE.

### 🖧 RIC / E2 Interface Metrics
*   **E2 Loop Latency (ms):** Control loop response time.
*   **RIC Message Overhead (msg/s):** E2 messages per second.
*   **Control Stability (%):** 0-100 score measuring AI "jitter". High score means stable decisions; low score means frequent, large action changes.

---

## 2. Composite Health Scores (Radar Chart)

To provide a quick "Health Check" of the network, we aggregate metrics into **6 composite scores** (0-100).

### 🏆 1. QoS Score (User Experience)
Combines how fast, responsive, and consistent the network felt to users. Includes tail-latency (p95) to capture stuttering.
*   **Formula:** `20% Throughput + 20% Delay + 10% p95 Delay + 15% Jitter + 35% Satisfied Users`

### 🛡️ 2. Reliability Score (Stability)
Penalizes both constant loss, sudden outages (Peak/Max Loss), service downtime, unstable mobility, and handover failures. Rewards fast recovery.
*   **Formula:** `20% Avg Loss + 10% Max Loss + 15% Peak Burst Loss + 10% Total Downtime + 10% Handover Stability + 20% Handover Success Rate + 15% Recovery Time`

### 🏗️ 3. Resource Score (Efficiency & Fairness)
Rewards high spectrum utilization AND efficiency, while ensuring fairness.
*   **Formula:** `20% Utilization + 20% Cell Edge + 20% Jain's Fairness + 20% Spectral Efficiency + 20% Energy Efficiency`

### 📦 4. Buffer Score (Congestion Health)
Measures buffer occupancy and congestion spikes.
*   **Formula:** `60% Norm. Queue Length + 40% Congestion Intensity`

### 📡 5. PHY Score (Signal Quality)
Combined physical layer conditions.
*   **Formula:** `60% SINR + 40% RSRP`

### 🖧 6. RIC Score (E2 Interface & AI Stability)
Measures control loop latency, message overhead, and AI "jitteriness".
*   **Formula:** `40% E2 Latency + 30% Message Overhead + 30% Control Stability`
*   High stability, low latency, and reasonable overhead yield high scores.

---

## 3. Per-UE Metrics Display

During evaluation, a formatted table shows **all key metrics per UE**:

```
  ╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
  ║  UE Metrics - 20 UEs                                                                                                    ║
  ╠════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
  ║ UE │   Tput │  Delay │  Loss │   SINR │    RSRP │   Cell │ Buffer ║
  ╠════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
  ║  0 │  1.41M │    53ms │  0.0% │  29.1dB │   -100dB │    2 │      0 ║
  ...
  ╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
```

**Legend:**
- `--` indicates a default/unavailable value (e.g., RSRP=-140, Cell=-1)
- Variance summary printed below table if non-zero

---

## 4. Data Quality Indicator

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

## 5. Excluded Metrics & Limitations

The following metrics were considered but **not implemented** due to simulator constraints:

| Metric | Category | Reason for Exclusion |
| :--- | :--- | :--- |
| **Collision Rate** | Reliability | Requires MAC layer tracing with significant I/O overhead. |
| **Control Overhead** | Resource | Requires deep packet inspection, not feasible in real-time RL. |

---

## 6. How to Run Evaluation

Run the evaluation mode comparing `Radio-Cortex` agent against `Static-RAN` baseline:

```bash
# Run all scenarios sequentially (Benchmark Mode)
python3 radio_cortex_complete.py --mode eval

# Run only one specific scenario (Fast Test)
python3 radio_cortex_complete.py --mode eval --scenario mobility_storm
```

### Outputs
Results are saved in the `results/` directory:
1.  **`*_comparison.png`:** Bar charts comparing all 20 metrics across 5 rows.
2.  **`*_radar.png`:** Radar chart comparing the **6 composite health scores**.
3.  **`*_metrics.tex`:** LaTeX table for paper inclusion.
4.  **`*_metrics.csv`:** Spreadsheet-ready CSV report for detailed data analysis.
