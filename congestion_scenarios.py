"""
Radio-Cortex Congestion Scenario Generator
Creates realistic RAN failure scenarios for testing self-healing capabilities

Scenarios:
1. Flash Crowd - Sudden surge of users
2. Mobility Storm - Rapid UE movements causing handover cascade
3. Traffic Tsunami - Burst of high-bandwidth applications
4. Cell Failure - eNB/gNB outage forcing load redistribution
5. Interference Storm - External RF interference
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import json


class ScenarioType(Enum):
    """Types of congestion scenarios"""
    FLASH_CROWD = "flash_crowd"
    MOBILITY_STORM = "mobility_storm"
    TRAFFIC_TSUNAMI = "traffic_tsunami"
    CELL_FAILURE = "cell_failure"
    INTERFERENCE_STORM = "interference_storm"
    HANDOVER_PINGPONG = "handover_pingpong"
    BLACK_SWAN = "black_swan"  # Combination of multiple failures


@dataclass
class ScenarioConfig:
    """Configuration for congestion scenario"""
    scenario_type: ScenarioType
    start_time: float  # seconds
    duration: float  # seconds
    severity: float  # 0.0 (mild) to 1.0 (catastrophic)
    affected_cells: List[int]
    affected_ues: Optional[List[int]] = None
    parameters: Optional[Dict] = None


class CongestionScenarioGenerator:
    """
    Generates synthetic congestion scenarios for O-RAN testing
    Used to stress-test Radio-Cortex self-healing capabilities
    """
    
    def __init__(self, num_ues: int, num_cells: int, baseline_throughput: float = 30.0):
        self.num_ues = num_ues
        self.num_cells = num_cells
        self.baseline_throughput = baseline_throughput
        self.current_time = 0.0
        
        # Baseline KPM values (normal operation)
        self.baseline_kpm = {
            'throughput': baseline_throughput,  # Mbps
            'delay': 50.0,  # ms
            'packet_loss': 0.001,  # 0.1%
            'sinr': 15.0,  # dB
            'queue_length': 50,  # packets
            'rb_utilization': 0.5,  # 50%
        }
    
    def generate_scenario(
        self,
        config: ScenarioConfig,
        time_step: float = 0.1
    ) -> List[Dict]:
        """
        Generate KPM reports for specified scenario
        
        Returns:
            List of KPM reports over scenario duration
        """
        if config.scenario_type == ScenarioType.FLASH_CROWD:
            return self._generate_flash_crowd(config, time_step)
        elif config.scenario_type == ScenarioType.MOBILITY_STORM:
            return self._generate_mobility_storm(config, time_step)
        elif config.scenario_type == ScenarioType.TRAFFIC_TSUNAMI:
            return self._generate_traffic_tsunami(config, time_step)
        elif config.scenario_type == ScenarioType.CELL_FAILURE:
            return self._generate_cell_failure(config, time_step)
        elif config.scenario_type == ScenarioType.INTERFERENCE_STORM:
            return self._generate_interference_storm(config, time_step)
        elif config.scenario_type == ScenarioType.HANDOVER_PINGPONG:
            return self._generate_handover_pingpong(config, time_step)
        elif config.scenario_type == ScenarioType.BLACK_SWAN:
            return self._generate_black_swan(config, time_step)
        else:
            raise ValueError(f"Unknown scenario type: {config.scenario_type}")
    
    def _generate_flash_crowd(
        self,
        config: ScenarioConfig,
        time_step: float
    ) -> List[Dict]:
        """
        Flash Crowd: Sudden surge of UEs connecting to same cell
        
        Characteristics:
        - Queue overflow
        - Resource block exhaustion
        - Increasing packet loss
        - Degraded throughput for all UEs in cell
        """
        kpm_timeline = []
        num_steps = int(config.duration / time_step)
        
        # Ramp-up phase: users gradually join
        ramp_up_steps = num_steps // 3
        
        for step in range(num_steps):
            t = config.start_time + step * time_step
            
            # Calculate load increase (sigmoid curve)
            progress = step / ramp_up_steps if step < ramp_up_steps else 1.0
            load_multiplier = 1.0 + config.severity * (1.0 / (1.0 + np.exp(-10 * (progress - 0.5))))
            
            kpm_reports = []
            
            for ue_id in range(self.num_ues):
                cell_id = ue_id % self.num_cells
                
                if cell_id in config.affected_cells:
                    # Congested cell
                    # Throughput degrades as more users share capacity
                    throughput = self.baseline_kpm['throughput'] / load_multiplier
                    
                    # Delay increases due to queue buildup
                    delay = self.baseline_kpm['delay'] * load_multiplier
                    
                    # Packet loss increases (buffer overflow)
                    packet_loss = min(0.5, self.baseline_kpm['packet_loss'] * (load_multiplier ** 2))
                    
                    # SINR relatively unaffected (interference, not physical)
                    sinr = self.baseline_kpm['sinr'] + np.random.normal(0, 2)
                    
                    # Queue length grows
                    queue_length = int(self.baseline_kpm['queue_length'] * load_multiplier)
                    
                    # RB utilization saturates
                    rb_util = min(0.99, self.baseline_kpm['rb_utilization'] * load_multiplier)
                else:
                    # Normal cell
                    throughput = self.baseline_kpm['throughput'] + np.random.normal(0, 5)
                    delay = self.baseline_kpm['delay'] + np.random.normal(0, 10)
                    packet_loss = self.baseline_kpm['packet_loss'] + np.random.uniform(0, 0.001)
                    sinr = self.baseline_kpm['sinr'] + np.random.normal(0, 3)
                    queue_length = self.baseline_kpm['queue_length']
                    rb_util = self.baseline_kpm['rb_utilization']
                
                kpm_reports.append({
                    'timestamp': t,
                    'ue_id': ue_id,
                    'cell_id': cell_id,
                    'throughput': max(0.1, throughput),
                    'delay': delay,
                    'packet_loss': packet_loss,
                    'sinr': sinr,
                    'queue_length': queue_length,
                    'rb_utilization': rb_util,
                })
            
            kpm_timeline.extend(kpm_reports)
        
        return kpm_timeline
    
    def _generate_mobility_storm(
        self,
        config: ScenarioConfig,
        time_step: float
    ) -> List[Dict]:
        """
        Mobility Storm: Rapid UE movements causing handover cascade
        
        Characteristics:
        - Frequent handovers
        - Handover failures
        - Ping-pong handovers (back and forth)
        - Temporary throughput drops during HO
        """
        kpm_timeline = []
        num_steps = int(config.duration / time_step)
        
        # Track UE cell assignments (changes during handovers)
        ue_cells = [ue_id % self.num_cells for ue_id in range(self.num_ues)]
        handover_cooldown = [0] * self.num_ues  # Prevent immediate ping-pong
        
        for step in range(num_steps):
            t = config.start_time + step * time_step
            kpm_reports = []
            
            # Trigger handovers with high probability during storm
            ho_probability = 0.1 * config.severity  # 10% per step at max severity
            
            for ue_id in range(self.num_ues):
                cell_id = ue_cells[ue_id]
                
                # Handover decision
                if handover_cooldown[ue_id] == 0 and np.random.rand() < ho_probability:
                    # Trigger handover to random neighbor cell
                    available_cells = [c for c in range(self.num_cells) if c != cell_id]
                    new_cell = np.random.choice(available_cells)
                    ue_cells[ue_id] = new_cell
                    handover_cooldown[ue_id] = 5  # Cooldown for 5 steps
                    
                    # Handover causes temporary performance drop
                    throughput = self.baseline_kpm['throughput'] * 0.3  # 70% drop
                    delay = self.baseline_kpm['delay'] * 3.0
                    packet_loss = 0.05  # 5% loss during HO
                    sinr = self.baseline_kpm['sinr'] - 10  # Poor signal during HO
                else:
                    # Normal operation (with some variance)
                    throughput = self.baseline_kpm['throughput'] + np.random.normal(0, 5)
                    delay = self.baseline_kpm['delay'] + np.random.normal(0, 10)
                    packet_loss = self.baseline_kpm['packet_loss'] + np.random.uniform(0, 0.002)
                    sinr = self.baseline_kpm['sinr'] + np.random.normal(0, 3)
                
                # Decrement cooldown
                if handover_cooldown[ue_id] > 0:
                    handover_cooldown[ue_id] -= 1
                
                kpm_reports.append({
                    'timestamp': t,
                    'ue_id': ue_id,
                    'cell_id': ue_cells[ue_id],
                    'throughput': max(0.1, throughput),
                    'delay': delay,
                    'packet_loss': packet_loss,
                    'sinr': sinr,
                    'handover_in_progress': handover_cooldown[ue_id] > 0,
                })
            
            kpm_timeline.extend(kpm_reports)
        
        return kpm_timeline
    
    def _generate_traffic_tsunami(
        self,
        config: ScenarioConfig,
        time_step: float
    ) -> List[Dict]:
        """
        Traffic Tsunami: Sudden spike in bandwidth demand
        
        Example: Major sporting event, emergency broadcast
        
        Characteristics:
        - All UEs demand max throughput simultaneously
        - Network capacity exceeded
        - Fairness issues (some UEs starved)
        """
        kpm_timeline = []
        num_steps = int(config.duration / time_step)
        
        for step in range(num_steps):
            t = config.start_time + step * time_step
            kpm_reports = []
            
            # All UEs in affected cells request maximum bandwidth
            demand_multiplier = 1.0 + 5.0 * config.severity  # Up to 6x normal demand
            
            for ue_id in range(self.num_ues):
                cell_id = ue_id % self.num_cells
                
                if cell_id in config.affected_cells:
                    # Network cannot satisfy all demands
                    # Some UEs get throughput, others are starved
                    success_rate = 1.0 / demand_multiplier
                    
                    if np.random.rand() < success_rate:
                        # Lucky UE gets throughput
                        throughput = self.baseline_kpm['throughput'] * 1.5
                        delay = self.baseline_kpm['delay']
                        packet_loss = 0.01
                    else:
                        # Unlucky UE is starved
                        throughput = self.baseline_kpm['throughput'] * 0.1
                        delay = self.baseline_kpm['delay'] * 10
                        packet_loss = 0.3
                    
                    sinr = self.baseline_kpm['sinr'] + np.random.normal(0, 2)
                else:
                    # Normal cell
                    throughput = self.baseline_kpm['throughput'] + np.random.normal(0, 5)
                    delay = self.baseline_kpm['delay'] + np.random.normal(0, 10)
                    packet_loss = self.baseline_kpm['packet_loss']
                    sinr = self.baseline_kpm['sinr'] + np.random.normal(0, 3)
                
                kpm_reports.append({
                    'timestamp': t,
                    'ue_id': ue_id,
                    'cell_id': cell_id,
                    'throughput': throughput,
                    'delay': delay,
                    'packet_loss': packet_loss,
                    'sinr': sinr,
                })
            
            kpm_timeline.extend(kpm_reports)
        
        return kpm_timeline
    
    def _generate_cell_failure(
        self,
        config: ScenarioConfig,
        time_step: float
    ) -> List[Dict]:
        """
        Cell Failure: eNB/gNB goes offline
        
        Characteristics:
        - Complete loss of service for connected UEs
        - Load redistribution to neighbor cells
        - Neighbor cells become congested
        """
        kpm_timeline = []
        num_steps = int(config.duration / time_step)
        
        # Determine which UEs need to relocate
        affected_ue_ids = [ue for ue in range(self.num_ues) 
                          if (ue % self.num_cells) in config.affected_cells]
        
        # Redistribute to neighbor cells
        available_cells = [c for c in range(self.num_cells) if c not in config.affected_cells]
        ue_cells = {}
        for ue_id in range(self.num_ues):
            original_cell = ue_id % self.num_cells
            if original_cell in config.affected_cells:
                ue_cells[ue_id] = np.random.choice(available_cells)
            else:
                ue_cells[ue_id] = original_cell
        
        for step in range(num_steps):
            t = config.start_time + step * time_step
            kpm_reports = []
            
            # Count load on each cell
            cell_loads = {c: 0 for c in range(self.num_cells)}
            for ue_id, cell_id in ue_cells.items():
                cell_loads[cell_id] += 1
            
            for ue_id in range(self.num_ues):
                cell_id = ue_cells[ue_id]
                
                if cell_id in config.affected_cells:
                    # Failed cell - complete outage
                    throughput = 0.0
                    delay = 99999.0
                    packet_loss = 1.0
                    sinr = -50.0
                else:
                    # Neighbor cell (potentially overloaded)
                    load_factor = cell_loads[cell_id] / (self.num_ues / self.num_cells)
                    
                    throughput = self.baseline_kpm['throughput'] / load_factor
                    delay = self.baseline_kpm['delay'] * load_factor
                    packet_loss = min(0.5, self.baseline_kpm['packet_loss'] * (load_factor ** 2))
                    sinr = self.baseline_kpm['sinr'] - 5 * (load_factor - 1)
                
                kpm_reports.append({
                    'timestamp': t,
                    'ue_id': ue_id,
                    'cell_id': cell_id,
                    'throughput': throughput,
                    'delay': delay,
                    'packet_loss': packet_loss,
                    'sinr': sinr,
                    'cell_failed': cell_id in config.affected_cells,
                })
            
            kpm_timeline.extend(kpm_reports)
        
        return kpm_timeline
    
    def _generate_interference_storm(
        self,
        config: ScenarioConfig,
        time_step: float
    ) -> List[Dict]:
        """
        Interference Storm: External RF interference
        
        Example: Jammer, competing network, weather
        
        Characteristics:
        - Degraded SINR
        - Increased retransmissions
        - Throughput collapse despite low load
        """
        kpm_timeline = []
        num_steps = int(config.duration / time_step)
        
        for step in range(num_steps):
            t = config.start_time + step * time_step
            kpm_reports = []
            
            # Interference level varies over time (simulate weather patterns)
            interference_level = config.severity * (1 + 0.3 * np.sin(step / 10))
            
            for ue_id in range(self.num_ues):
                cell_id = ue_id % self.num_cells
                
                if cell_id in config.affected_cells:
                    # Severe SINR degradation
                    sinr = self.baseline_kpm['sinr'] - 20 * interference_level
                    
                    # Throughput drops due to poor modulation
                    throughput = self.baseline_kpm['throughput'] * (1 - 0.8 * interference_level)
                    
                    # More retransmissions needed
                    delay = self.baseline_kpm['delay'] * (1 + 3 * interference_level)
                    packet_loss = self.baseline_kpm['packet_loss'] + 0.1 * interference_level
                else:
                    # Normal cell
                    throughput = self.baseline_kpm['throughput'] + np.random.normal(0, 5)
                    delay = self.baseline_kpm['delay'] + np.random.normal(0, 10)
                    packet_loss = self.baseline_kpm['packet_loss']
                    sinr = self.baseline_kpm['sinr'] + np.random.normal(0, 3)
                
                kpm_reports.append({
                    'timestamp': t,
                    'ue_id': ue_id,
                    'cell_id': cell_id,
                    'throughput': max(0.1, throughput),
                    'delay': delay,
                    'packet_loss': packet_loss,
                    'sinr': sinr,
                })
            
            kpm_timeline.extend(kpm_reports)
        
        return kpm_timeline
    
    def _generate_handover_pingpong(
        self,
        config: ScenarioConfig,
        time_step: float
    ) -> List[Dict]:
        """
        Handover Ping-Pong: UEs rapidly switch between cells
        
        Characteristics:
        - Oscillating cell assignments
        - Wasted signaling overhead
        - Poor UE experience
        """
        kpm_timeline = []
        num_steps = int(config.duration / time_step)
        
        # UEs at cell edge oscillate
        ue_cells = [ue_id % self.num_cells for ue_id in range(self.num_ues)]
        
        for step in range(num_steps):
            t = config.start_time + step * time_step
            kpm_reports = []
            
            for ue_id in range(self.num_ues):
                # Oscillate every few steps
                if step % 3 == 0:  # Ping-pong every 0.3s
                    current_cell = ue_cells[ue_id]
                    neighbor = (current_cell + 1) % self.num_cells
                    ue_cells[ue_id] = neighbor
                
                cell_id = ue_cells[ue_id]
                
                # Degraded performance due to constant handovers
                throughput = self.baseline_kpm['throughput'] * 0.5
                delay = self.baseline_kpm['delay'] * 2.0
                packet_loss = 0.02
                sinr = self.baseline_kpm['sinr'] - 5
                
                kpm_reports.append({
                    'timestamp': t,
                    'ue_id': ue_id,
                    'cell_id': cell_id,
                    'throughput': throughput,
                    'delay': delay,
                    'packet_loss': packet_loss,
                    'sinr': sinr,
                })
            
            kpm_timeline.extend(kpm_reports)
        
        return kpm_timeline
    
    def _generate_black_swan(
        self,
        config: ScenarioConfig,
        time_step: float
    ) -> List[Dict]:
        """
        Black Swan: Multiple simultaneous failures
        
        Example: Natural disaster, cyberattack
        
        Combines:
        - Cell failure
        - Flash crowd
        - Interference
        """
        # Combine multiple scenarios
        cell_failure_config = ScenarioConfig(
            scenario_type=ScenarioType.CELL_FAILURE,
            start_time=config.start_time,
            duration=config.duration,
            severity=config.severity,
            affected_cells=[config.affected_cells[0]] if config.affected_cells else [0]
        )
        
        flash_crowd_config = ScenarioConfig(
            scenario_type=ScenarioType.FLASH_CROWD,
            start_time=config.start_time,
            duration=config.duration,
            severity=config.severity * 0.8,
            affected_cells=[1] if self.num_cells > 1 else [0]
        )
        
        # Merge scenarios (worst-case metrics for each UE)
        kpm_failure = self._generate_cell_failure(cell_failure_config, time_step)
        kpm_crowd = self._generate_flash_crowd(flash_crowd_config, time_step)
        
        # Combine by taking worst metrics
        kpm_combined = {}
        for kpm in kpm_failure + kpm_crowd:
            key = (kpm['timestamp'], kpm['ue_id'])
            if key not in kpm_combined:
                kpm_combined[key] = kpm
            else:
                # Take worst metrics
                kpm_combined[key]['throughput'] = min(
                    kpm_combined[key]['throughput'], kpm['throughput']
                )
                kpm_combined[key]['delay'] = max(
                    kpm_combined[key]['delay'], kpm['delay']
                )
                kpm_combined[key]['packet_loss'] = max(
                    kpm_combined[key]['packet_loss'], kpm['packet_loss']
                )
                kpm_combined[key]['sinr'] = min(
                    kpm_combined[key]['sinr'], kpm['sinr']
                )
        
        return list(kpm_combined.values())
    
    def export_scenario(self, kpm_timeline: List[Dict], filename: str):
        """Export scenario to JSON file for replay"""
        with open(filename, 'w') as f:
            json.dump(kpm_timeline, f, indent=2)
        print(f"Exported {len(kpm_timeline)} KPM reports to {filename}")


# ============================================================================
# Scenario Suite for Evaluation
# ============================================================================

class EvaluationSuite:
    """
    Standard suite of scenarios for benchmarking Radio-Cortex
    """
    
    def __init__(self, num_ues: int = 20, num_cells: int = 3):
        self.generator = CongestionScenarioGenerator(num_ues, num_cells)
        self.scenarios = self._create_scenarios()
    
    def _create_scenarios(self) -> Dict[str, ScenarioConfig]:
        """Define standard evaluation scenarios"""
        return {
            'mild_congestion': ScenarioConfig(
                scenario_type=ScenarioType.FLASH_CROWD,
                start_time=2.0,
                duration=5.0,
                severity=0.3,
                affected_cells=[0]
            ),
            'severe_congestion': ScenarioConfig(
                scenario_type=ScenarioType.FLASH_CROWD,
                start_time=2.0,
                duration=5.0,
                severity=0.8,
                affected_cells=[0]
            ),
            'mobility_chaos': ScenarioConfig(
                scenario_type=ScenarioType.MOBILITY_STORM,
                start_time=2.0,
                duration=5.0,
                severity=0.7,
                affected_cells=list(range(3))
            ),
            'cell_outage': ScenarioConfig(
                scenario_type=ScenarioType.CELL_FAILURE,
                start_time=3.0,
                duration=4.0,
                severity=1.0,
                affected_cells=[1]
            ),
            'black_swan_event': ScenarioConfig(
                scenario_type=ScenarioType.BLACK_SWAN,
                start_time=2.0,
                duration=6.0,
                severity=0.9,
                affected_cells=[0, 1]
            ),
        }
    
    def generate_all(self) -> Dict[str, List[Dict]]:
        """Generate all scenarios in the suite"""
        results = {}
        for name, config in self.scenarios.items():
            print(f"Generating scenario: {name}")
            kpm_timeline = self.generator.generate_scenario(config)
            results[name] = kpm_timeline
        return results


# ============================================================================
# Demo and Testing
# ============================================================================

if __name__ == "__main__":
    print("=== Radio-Cortex Congestion Scenario Generator ===\n")
    
    generator = CongestionScenarioGenerator(num_ues=20, num_cells=3)
    
    # Demo 1: Flash Crowd
    print("1. Generating Flash Crowd scenario...")
    config = ScenarioConfig(
        scenario_type=ScenarioType.FLASH_CROWD,
        start_time=2.0,
        duration=5.0,
        severity=0.8,
        affected_cells=[0]
    )
    
    kpm_data = generator.generate_scenario(config, time_step=0.5)
    
    print(f"   Generated {len(kpm_data)} KPM reports")
    print(f"   Sample report: {kpm_data[0]}")
    
    # Analyze congestion impact
    cell_0_reports = [r for r in kpm_data if r['cell_id'] == 0]
    avg_loss = np.mean([r['packet_loss'] for r in cell_0_reports])
    avg_tput = np.mean([r['throughput'] for r in cell_0_reports])
    
    print(f"   Cell 0 (congested): Avg Loss = {avg_loss:.2%}, Avg Throughput = {avg_tput:.1f} Mbps")
    
    # Demo 2: Generate full evaluation suite
    print("\n2. Generating evaluation suite...")
    suite = EvaluationSuite(num_ues=20, num_cells=3)
    all_scenarios = suite.generate_all()
    
    print(f"\n✓ Generated {len(all_scenarios)} scenarios")
    for name, data in all_scenarios.items():
        print(f"   {name}: {len(data)} KPM reports")
    
    # Export for later use
    print("\n3. Exporting scenarios...")
    for name, data in all_scenarios.items():
        generator.export_scenario(data, f"scenario_{name}.json")
    
    print("\n✓ Scenario generation complete")
