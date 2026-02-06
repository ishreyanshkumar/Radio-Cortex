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
*   **PDR (Packet Delivery Ratio):** `100% - Packet Loss %`.

### ⚡ Resource Efficiency (Spectrum & Network)
*   **Spectrum Utilization (%):** Average usage of Resource Blocks (RBs) across all cells.
*   **Congestion Intensity (%):** Percentage of time where network utilization > 90%.
*   **Cell Edge Throughput (Mbps):** 5th Percentile throughput. Indicates how well the network serves users with poor coverage (fairness).

### 📶 PHY / Wireless Layer
*   **Average SINR (dB):** Signal-to-Interference-plus-Noise Ratio.
*   **Average RSRP (dBm):** Reference Signal Received Power (Signal Strength).

### 🚀 Mobility Metrics
*   **Handover Count:** Number of cell switches per UE.

---

## 2. Composite Health Scores (Radar Chart)

To provide a quick "Health Check" of the network, we aggregate metrics into 5 composite scores (0-100).

### 🏆 1. QoS Score (User Experience)
Combines how fast, responsive, and consistent the network felt to users.
*   **Formula:** `25% Throughput + 25% Delay + 15% Jitter + 35% Satisfied Users`

### 🛡️ 2. Reliability Score (Stability)
Penalizes both constant loss, sudden outages, and unstable mobility.
*   **Formula:** `40% Avg Loss + 30% Peak Burst Loss + 30% Handover Stability`

### 🏗️ 3. Resource Score (Efficiency & Fairness)
Rewards high spectrum utilization *only* if it is distributed fairly and doesn't starve edge users.
*   **Formula:** `50% Utilization + 20% Cell Edge Performance + 30% Jain's Fairness`

### 📦 4. Buffer Score (Congestion Health)
Measures buffer occupancy and congestion spikes.
*   **Formula:** `60% Norm. Queue Length + 40% Congestion Intensity`

### 📡 5. PHY Score (Signal Quality)
Combined physical layer conditions.
*   **Formula:** `60% SINR + 40% RSRP`

---

## 3. Excluded Metrics & Limitations

The following metrics were considered but **not implemented** in the current scope due to simulator constraints:

| Metric | Category | Reason for Exclusion |
| :--- | :--- | :--- |
| **Energy Consumption** | Resource | Requires `EnergyModel` and `WifiRadioEnergyModel` helpers enabled in the C++ ns-3 script (`oran-congestion-scenario.cc`). |
| **Collision Rate** | Reliability | Requires explicit MAC layer tracing (`EnableAscii` or `MacTxDrop` callbacks) which adds significant I/O overhead and is not currently exposed via E2 KPMs. |
| **Control Overhead** | Resource | Requires deep packet inspection (DPI) of control vs. data planes in PCAP traces, which is not feasible in real-time RL loops. |

---

## 3. How to Run Evaluation

Run the evaluation mode of the main script. This will compare the `Radio-Cortex` agent against `Static-RAN` and `Heuristic` baselines.

```bash
python3 radio_cortex_complete.py --mode eval
```

### Outputs
Results are saved in the `results/` directory:
1.  **`*_comparison.png`:** Bar charts comparing raw metrics.
2.  **`*_radar.png`:** Radar chart comparing the 5 composite health scores.
3.  **`evaluation_results.tex`:** LaTeX table for paper inclusion.
