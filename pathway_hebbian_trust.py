"""
Radio-Cortex Hebbian Trust Graph with Pathway
Real-time streaming trust calculation for O-RAN network

System 1 (Spinal Cord): Hebbian learning for instant reflexes
- Trust between UE-Cell pairs based on performance
- Exponential decay on packet loss/poor SINR
- Triggers <1ms routing decisions
"""

import pathway as pw
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import json


# ============================================================================
# Trust Calculation Core (Hebbian Learning)
# ============================================================================

@dataclass
class HebbianParams:
    """Hebbian learning hyperparameters"""
    learning_rate: float = 0.1
    decay_rate: float = 0.95
    loss_threshold: float = 0.01  # 1% packet loss triggers decay
    sinr_threshold: float = -5.0  # dB
    delay_threshold: float = 100.0  # ms
    min_trust: float = 0.0
    max_trust: float = 1.0


class HebbianTrustCalculator:
    """
    Implements Hebbian learning rule for UE-Cell trust
    
    Trust update rule:
    Δtrust = η * (reward - trust) where:
    - η = learning rate
    - reward = f(throughput, delay, loss, sinr)
    
    Trust decays exponentially on poor performance:
    trust *= exp(-α * penalty)
    """
    
    def __init__(self, params: HebbianParams = None):
        self.params = params or HebbianParams()
    
    def compute_trust(
        self,
        current_trust: float,
        throughput: float,
        delay: float,
        packet_loss: float,
        sinr: float
    ) -> Tuple[float, Dict]:
        """
        Update trust score based on network performance
        
        Returns:
            (new_trust, diagnostics)
        """
        # Calculate reward signal (performance metric)
        reward = self._calculate_reward(throughput, delay, packet_loss, sinr)
        
        # Hebbian update: trust moves toward reward
        delta_trust = self.params.learning_rate * (reward - current_trust)
        new_trust = current_trust + delta_trust
        
        # Apply exponential decay on critical failures
        decay_factor = self._calculate_decay(packet_loss, sinr, delay)
        new_trust *= decay_factor
        
        # Clamp to valid range
        new_trust = np.clip(new_trust, self.params.min_trust, self.params.max_trust)
        
        diagnostics = {
            'reward': reward,
            'delta_trust': delta_trust,
            'decay_factor': decay_factor,
            'final_trust': new_trust
        }
        
        return new_trust, diagnostics
    
    def _calculate_reward(
        self,
        throughput: float,
        delay: float,
        packet_loss: float,
        sinr: float
    ) -> float:
        """
        Reward function mapping performance to [0, 1]
        
        Good performance → reward ≈ 1
        Bad performance → reward ≈ 0
        """
        # Normalize throughput (assume max 100 Mbps)
        tput_score = min(throughput / 100.0, 1.0)
        
        # Delay score (lower is better)
        delay_score = np.exp(-delay / 100.0)  # Exponential decay
        
        # Loss score (catastrophic if high)
        loss_score = 1.0 - min(packet_loss * 10, 1.0)
        
        # SINR score (map [-10, 30] dB to [0, 1])
        sinr_score = (sinr + 10) / 40.0
        sinr_score = np.clip(sinr_score, 0.0, 1.0)
        
        # Weighted combination
        reward = (
            0.4 * tput_score +
            0.2 * delay_score +
            0.3 * loss_score +
            0.1 * sinr_score
        )
        
        return reward
    
    def _calculate_decay(
        self,
        packet_loss: float,
        sinr: float,
        delay: float
    ) -> float:
        """
        Exponential decay factor for trust
        Applied when performance crosses critical thresholds
        """
        decay = 1.0
        
        # Severe decay on packet loss
        if packet_loss > self.params.loss_threshold:
            decay *= np.exp(-10 * (packet_loss - self.params.loss_threshold))
        
        # Decay on poor SINR
        if sinr < self.params.sinr_threshold:
            decay *= np.exp(-(self.params.sinr_threshold - sinr) / 10)
        
        # Decay on high delay
        if delay > self.params.delay_threshold:
            decay *= np.exp(-(delay - self.params.delay_threshold) / 100)
        
        return max(decay, self.params.decay_rate)  # Minimum decay rate


# ============================================================================
# Pathway Streaming Pipeline
# ============================================================================

class PathwayTrustGraph:
    """
    Streaming trust graph computation using Pathway
    Processes E2 KPM reports in real-time
    """
    
    def __init__(self, kafka_servers: List[str] = None):
        self.kafka_servers = kafka_servers or ['localhost:9092']
        self.trust_calculator = HebbianTrustCalculator()
        self.trust_state = {}  # {(ue_id, cell_id): trust_score}
    
    def build_pipeline(self):
        """
        Construct Pathway streaming pipeline
        
        Input: E2 KPM reports from Kafka
        Output: Real-time trust graph updates
        """
        # Define schema for E2 KPM messages
        class KPMSchema(pw.Schema):
            timestamp: float
            ue_id: int
            cell_id: int
            throughput: float
            delay: float
            packet_loss: float
            sinr: float
        
        # Read from Kafka (or file in dev mode)
        kpm_stream = pw.io.kafka.read(
            rdkafka_settings={
                "bootstrap.servers": ','.join(self.kafka_servers),
                "group.id": "radio-cortex-trust",
                "auto.offset.reset": "latest",
            },
            topics=["e2_kpm_stream"],
            format="json",
            schema=KPMSchema
        )
        
        # Add previous trust state (windowed join)
        trust_table = self._initialize_trust_table()
        
        # Compute Hebbian trust updates
        @pw.udf
        def update_trust(
            ue_id: int,
            cell_id: int,
            throughput: float,
            delay: float,
            packet_loss: float,
            sinr: float,
            prev_trust: float
        ) -> dict:
            """Apply Hebbian update to trust score"""
            calculator = HebbianTrustCalculator()
            new_trust, diagnostics = calculator.compute_trust(
                current_trust=prev_trust,
                throughput=throughput,
                delay=delay,
                packet_loss=packet_loss,
                sinr=sinr
            )
            
            return {
                'ue_id': ue_id,
                'cell_id': cell_id,
                'trust': new_trust,
                'reward': diagnostics['reward'],
                'decay': diagnostics['decay_factor'],
                'timestamp': datetime.now().isoformat()
            }
        
        # Join with previous trust and compute updates
        trust_updates = kpm_stream.join(
            trust_table,
            kpm_stream.ue_id == trust_table.ue_id,
            kpm_stream.cell_id == trust_table.cell_id
        ).select(
            updated=update_trust(
                pw.this.ue_id,
                pw.this.cell_id,
                pw.this.throughput,
                pw.this.delay,
                pw.this.packet_loss,
                pw.this.sinr,
                pw.this.prev_trust
            )
        )
        
        # Detect trust collapse (critical for reflexes)
        @pw.udf
        def detect_collapse(trust: float, prev_trust: float) -> bool:
            """Trigger alert if trust drops below critical threshold"""
            return trust < 0.3 and (prev_trust - trust) > 0.2
        
        alerts = trust_updates.filter(
            detect_collapse(pw.this.updated['trust'], trust_table.prev_trust)
        )
        
        # Output streams
        # 1. Trust graph updates → to RL agent
        pw.io.kafka.write(
            trust_updates,
            rdkafka_settings={
                "bootstrap.servers": ','.join(self.kafka_servers),
            },
            topic="trust_graph_updates",
            format="json"
        )
        
        # 2. Critical alerts → immediate action
        pw.io.kafka.write(
            alerts,
            rdkafka_settings={
                "bootstrap.servers": ','.join(self.kafka_servers),
            },
            topic="trust_collapse_alerts",
            format="json"
        )
        
        return trust_updates, alerts
    
    def _initialize_trust_table(self):
        """Initialize trust state table (all UE-Cell pairs start at 0.5)"""
        # In production, this would be a persistent state store
        # For now, create from static config
        class TrustStateSchema(pw.Schema):
            ue_id: int
            cell_id: int
            prev_trust: float
        
        # This would be loaded from Redis/database in production
        initial_trust = pw.debug.table_from_markdown(
            """
            | ue_id | cell_id | prev_trust
            0 | 0     | 0       | 0.5
            1 | 0     | 1       | 0.5
            2 | 0     | 2       | 0.5
            """
        )
        
        return initial_trust
    
    def run(self):
        """Start Pathway streaming computation"""
        trust_stream, alert_stream = self.build_pipeline()
        pw.run()


# ============================================================================
# Integration with RL Agent
# ============================================================================

class TrustFeatureExtractor:
    """
    Extracts trust graph features for RL state
    Converts trust graph to fixed-size vector
    """
    
    def __init__(self, num_ues: int, num_cells: int):
        self.num_ues = num_ues
        self.num_cells = num_cells
    
    def extract_features(self, trust_graph: Dict[Tuple[int, int], float]) -> np.ndarray:
        """
        Convert trust graph to feature vector for RL
        
        Features:
        - Trust matrix (UE x Cell)
        - Per-UE max trust
        - Per-UE trust variance
        - Per-Cell average trust
        - Graph entropy (measure of uncertainty)
        """
        # Build trust matrix
        trust_matrix = np.zeros((self.num_ues, self.num_cells))
        for (ue_id, cell_id), trust in trust_graph.items():
            trust_matrix[ue_id, cell_id] = trust
        
        # Aggregate features
        features = []
        
        # 1. Flattened trust matrix
        features.extend(trust_matrix.flatten())
        
        # 2. Per-UE max trust (best cell for each UE)
        ue_max_trust = trust_matrix.max(axis=1)
        features.extend(ue_max_trust)
        
        # 3. Per-UE trust variance (measure of choice quality)
        ue_trust_var = trust_matrix.var(axis=1)
        features.extend(ue_trust_var)
        
        # 4. Per-Cell average trust (cell quality)
        cell_avg_trust = trust_matrix.mean(axis=0)
        features.extend(cell_avg_trust)
        
        # 5. Graph entropy (overall uncertainty)
        trust_flat = trust_matrix.flatten()
        trust_flat = trust_flat[trust_flat > 0]  # Remove zeros
        if len(trust_flat) > 0:
            entropy = -np.sum(trust_flat * np.log(trust_flat + 1e-10))
        else:
            entropy = 0.0
        features.append(entropy)
        
        return np.array(features, dtype=np.float32)


# ============================================================================
# Mock E2 KPM Generator (for testing without ns-3)
# ============================================================================

class MockKPMGenerator:
    """
    Generates synthetic E2 KPM reports for testing
    Simulates congestion scenarios
    """
    
    def __init__(self, num_ues: int = 20, num_cells: int = 3):
        self.num_ues = num_ues
        self.num_cells = num_cells
        self.time = 0.0
    
    def generate_normal_traffic(self) -> List[Dict]:
        """Generate normal network conditions"""
        kpm_reports = []
        
        for ue_id in range(self.num_ues):
            # Each UE connected to random cell
            cell_id = ue_id % self.num_cells
            
            kpm_reports.append({
                'timestamp': self.time,
                'ue_id': ue_id,
                'cell_id': cell_id,
                'throughput': np.random.uniform(10, 50),  # Mbps
                'delay': np.random.uniform(20, 80),  # ms
                'packet_loss': np.random.uniform(0, 0.005),  # <0.5%
                'sinr': np.random.uniform(5, 25),  # dB
            })
        
        self.time += 0.1  # 100ms interval
        return kpm_reports
    
    def generate_congestion_scenario(self) -> List[Dict]:
        """Generate flash crowd congestion"""
        kpm_reports = []
        
        # Cell 0 is congested (high load)
        congested_cell = 0
        
        for ue_id in range(self.num_ues):
            cell_id = ue_id % self.num_cells
            
            if cell_id == congested_cell:
                # Degraded performance
                throughput = np.random.uniform(1, 10)  # Low throughput
                delay = np.random.uniform(200, 500)  # High delay
                packet_loss = np.random.uniform(0.05, 0.2)  # 5-20% loss
                sinr = np.random.uniform(-5, 5)  # Poor signal
            else:
                # Normal performance
                throughput = np.random.uniform(20, 50)
                delay = np.random.uniform(20, 80)
                packet_loss = np.random.uniform(0, 0.005)
                sinr = np.random.uniform(10, 25)
            
            kpm_reports.append({
                'timestamp': self.time,
                'ue_id': ue_id,
                'cell_id': cell_id,
                'throughput': throughput,
                'delay': delay,
                'packet_loss': packet_loss,
                'sinr': sinr,
            })
        
        self.time += 0.1
        return kpm_reports


# ============================================================================
# Testing and Demo
# ============================================================================

def demo_hebbian_trust():
    """Demonstrate Hebbian trust calculation"""
    calculator = HebbianTrustCalculator()
    
    print("=== Hebbian Trust Demo ===\n")
    
    # Scenario 1: Good performance → trust increases
    print("Scenario 1: Good Performance")
    trust = 0.5
    for step in range(5):
        trust, diag = calculator.compute_trust(
            current_trust=trust,
            throughput=30.0,
            delay=50.0,
            packet_loss=0.001,
            sinr=15.0
        )
        print(f"  Step {step}: Trust = {trust:.3f}, Reward = {diag['reward']:.3f}")
    
    print("\nScenario 2: Sudden Packet Loss → trust collapses")
    trust = 0.8
    for step in range(5):
        # Inject packet loss at step 2
        loss = 0.001 if step < 2 else 0.15
        
        trust, diag = calculator.compute_trust(
            current_trust=trust,
            throughput=30.0,
            delay=50.0,
            packet_loss=loss,
            sinr=15.0
        )
        print(f"  Step {step}: Trust = {trust:.3f}, Loss = {loss:.3f}, Decay = {diag['decay_factor']:.3f}")
    
    print("\nScenario 3: Recovery → trust gradually rebuilds")
    trust = 0.2  # After collapse
    for step in range(10):
        trust, diag = calculator.compute_trust(
            current_trust=trust,
            throughput=40.0,
            delay=40.0,
            packet_loss=0.0,
            sinr=20.0
        )
        if step % 2 == 0:
            print(f"  Step {step}: Trust = {trust:.3f}")


def test_trust_feature_extraction():
    """Test feature extraction for RL"""
    print("\n=== Trust Feature Extraction ===\n")
    
    # Mock trust graph
    trust_graph = {
        (0, 0): 0.8, (0, 1): 0.3, (0, 2): 0.5,
        (1, 0): 0.6, (1, 1): 0.9, (1, 2): 0.4,
        (2, 0): 0.2, (2, 1): 0.7, (2, 2): 0.8,
    }
    
    extractor = TrustFeatureExtractor(num_ues=3, num_cells=3)
    features = extractor.extract_features(trust_graph)
    
    print(f"Feature vector shape: {features.shape}")
    print(f"Features: {features[:15]}...")  # First 15 elements
    print(f"Graph entropy: {features[-1]:.3f}")


if __name__ == "__main__":
    # Run demos
    demo_hebbian_trust()
    test_trust_feature_extraction()
    
    print("\n=== Mock KPM Generation ===\n")
    generator = MockKPMGenerator(num_ues=5, num_cells=2)
    
    print("Normal traffic:")
    reports = generator.generate_normal_traffic()
    for r in reports[:3]:
        print(f"  UE {r['ue_id']} → Cell {r['cell_id']}: "
              f"Tput={r['throughput']:.1f} Mbps, Loss={r['packet_loss']:.4f}")
    
    print("\nCongestion scenario:")
    reports = generator.generate_congestion_scenario()
    for r in reports[:3]:
        print(f"  UE {r['ue_id']} → Cell {r['cell_id']}: "
              f"Tput={r['throughput']:.1f} Mbps, Loss={r['packet_loss']:.4f}")
    
    print("\n✓ Pathway trust pipeline ready for integration")
