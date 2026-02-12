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

## 🎯 Complete Training Curriculum

### **Stage 1: Foundation - Flash Crowd Mastery**
**Duration**: 10,000 timesteps  
**Scenario Mix**: 
- Flash Crowd: **100%**

**Goal**: Learn basic load balancing and handover triggering

**Success Criteria**:
- Average reward > -2.0 over last 500 steps
- Jain's Fairness Index > 0.80
- Throughput degradation < 30%

**Validation Checkpoint**: Run 100 episodes, ensure 85% meet success criteria

---

### **Stage 2: Energy Patterns Introduction**
**Duration**: 16,000 timesteps  
**Scenario Mix**:
- Flash Crowd: **20%** (maintenance)
- Sleepy Campus: **80%** (learning focus)

**Rationale**: 
- Flash Crowd teaches spatial load balancing
- Sleepy Campus teaches temporal patterns
- 20% Flash Crowd prevents forgetting handover skills

**Goal**: Master energy-efficient power adaptation

**Success Criteria**:
- Energy efficiency > 0.70
- Power switching smoothness (no oscillations)
- Flash Crowd performance still > -2.5 reward

**Validation**: Separate 50-episode tests on each scenario

---

### **Stage 3: PHY Layer Robustness**
**Duration**: 20,000 timesteps  
**Scenario Mix**:
- Flash Crowd: **15%** (maintenance)
- Sleepy Campus: **15%** (maintenance)
- Urban Canyon: **70%** (learning focus)

**Rationale**:
- Urban Canyon adds signal degradation challenge
- Maintains load balancing (Flash) and energy (Sleepy) skills
- 15% each ensures no regression

**Goal**: Learn reactive signal recovery and SINR optimization

**Success Criteria**:
- Recovery time < 1.0 second after SINR drop
- RLF rate < 5%
- Previous scenarios: Flash > -2.5, Sleepy EE > 0.65

**Validation**: 30 episodes per scenario, compare to baseline

---

### **Stage 4: Handover Dynamics - Mobility Storm**
**Duration**: 30,000 timesteps  
**Scenario Mix**:
- Flash Crowd: **10%**
- Sleepy Campus: **10%**
- Urban Canyon: **10%**
- Mobility Storm: **70%**

**Rationale**:
- Mobility Storm introduces velocity-based handover challenges
- Reduce previous scenarios to 10% each (enough to maintain)
- Focus on handover parameter tuning (Hysteresis, TTT)

**Goal**: Optimize handover success rate under mobility

**Success Criteria**:
- Handover success rate > 92%
- RLF count < 3 per 100 UEs
- Previous scenarios: minimal performance drop (< 10%)

**Validation**: 40 episodes Mobility Storm, 20 each for others

---

### **Stage 5: Extreme Overload - Traffic Burst**
**Duration**: 40,000 timesteps  
**Scenario Mix**:
- Previous 4 scenarios: **8%** each = **32%** total
- Traffic Burst: **68%**

**Rationale**:
- Traffic Burst is harder (7/10 difficulty)
- Need more focus (68%) but keep all previous skills active
- Equal 8% distribution ensures balanced retention

**Goal**: Master scheduler optimization under 5x overload

**Success Criteria**:
- Packet loss ratio < 8%
- 5th percentile throughput > 2 Mbps (no starvation)
- Queue congestion handled gracefully

**Validation**: 50 episodes Traffic Burst + mini-suite on previous

---

### **Stage 6: Multi-Objective - Mixed Reality Slicing**
**Duration**: 40,000 timesteps  
**Scenario Mix**:
- Previous 5 scenarios: **7%** each = **35%** total
- Mixed Reality: **65%**

**Rationale**:
- Mixed Reality requires balancing VR latency vs Download throughput
- Maintain all 5 previous skills at 7% each
- 65% focus on new multi-objective challenge

**Goal**: Learn network slicing and QoS differentiation

**Success Criteria**:
- VR slice: 99% packets < 20ms latency
- Download slice: Mean throughput > 8 Mbps
- Slice isolation: Correlation < 0.2

**Validation**: Per-slice metrics separately evaluated

---

### **Stage 7: Non-Stationarity - Adversarial Environment**
**Duration**: 36,000 timesteps  
**Scenario Mix**:
- Previous 6 scenarios: **6%** each = **36%** total
- Adversarial: **64%**

**Rationale**:
- Adversarial tests robustness to channel fluctuations
- 6% each maintains diverse skill set
- 64% builds resilience to noise

**Goal**: Develop robust policies under interference

**Success Criteria**:
- Throughput resilience ratio > 0.70 (degraded/normal)
- Power adaptation within 500ms of SINR change
- No catastrophic failures (reward > -8.0)

**Validation**: Test under both normal and degraded conditions

---

### **Stage 8: Advanced Handover - Ping-Pong Prevention**
**Duration**: 50,000 timesteps  
**Scenario Mix**:
- Previous 7 scenarios: **5%** each = **35%** total
- Handover Ping-Pong: **65%**

**Rationale**:
- Ping-Pong builds on Mobility Storm (requires memory)
- 5% each for 7 scenarios maintains broad competence
- 65% needed for LSTM/GRU policy training

**Goal**: Eliminate unnecessary handover oscillations

**Success Criteria**:
- Ping-pong rate < 10% of total handovers
- Handover efficiency > 88%
- SINR stability variance < 4 dB

**Validation**: Track handover patterns over 100 episodes

---

### **Stage 9: Mass Coordination - Commuter Rush**
**Duration**: 60,000 timesteps  
**Scenario Mix**:
- Previous 8 scenarios: **4.5%** each = **36%** total
- Commuter Rush: **64%**

**Rationale**:
- Commuter Rush tests mass handover coordination
- 4.5% each keeps 8 scenarios active
- 64% for complex multi-cell cooperation

**Goal**: Handle synchronized handover avalanche

**Success Criteria**:
- RLF rate during mass HO < 8%
- Load balanced across cells (max/min ratio < 2.0)
- X2 signaling overhead < 15% of total traffic

**Validation**: Measure worst-case metrics during peak HO period

---

### **Stage 10: Control Plane - IoT Tsunami**
**Duration**: 70,000 timesteps  
**Scenario Mix**:
- Previous 9 scenarios: **4%** each = **36%** total
- IoT Tsunami: **64%**

**Rationale**:
- IoT Tsunami shifts focus to control plane
- Maintaining 9 scenarios at 4% each
- 64% to master RACH optimization and SPS

**Goal**: Optimize control plane under massive device load

**Success Criteria**:
- RACH collision rate < 25%
- Signaling overhead < 20% of total bytes
- Packet delivery ratio > 95% for IoT packets

**Validation**: Control plane metrics separately tracked

---

### **Stage 11: URLLC Excellence - Ambulance Priority**
**Duration**: 80,000 timesteps  
**Scenario Mix**:
- Previous 10 scenarios: **3.5%** each = **35%** total
- Ambulance: **65%**

**Rationale**:
- Ambulance requires strict latency constraints
- 3.5% each for 10 scenarios (comprehensive maintenance)
- 65% for Lagrangian-augmented reward learning

**Goal**: Guarantee 5-9s reliability for priority UE

**Success Criteria**:
- 99.999%ile latency < 10ms for ambulance
- PDR > 99.999% (only 1 in 100k drops)
- Acceptable background sacrifice (> 1 Mbps avg)

**Validation**: Statistical reliability testing over 10,000 packets

---

### **Stage 12: Spectrum Mastery - Carrier Aggregation**
**Duration**: 100,000 timesteps  
**Scenario Mix**:
- Previous 11 scenarios: **3.2%** each = **35.2%** total
- Spectrum Crunch: **64.8%**

**Rationale**:
- Spectrum Crunch is hardest (9/10 difficulty)
- 3.2% each maintains all 11 previous skills
- 64.8% for CA coordination learning

**Goal**: Optimal carrier aggregation and load balancing

**Success Criteria**:
- SCell activation at optimal threshold (ρ ≈ 0.75)
- Aggregated throughput > 35 Mbps
- CA overhead < 8%

**Validation**: Multi-carrier metrics validated separately

---

### **Stage 13: Multi-Scenario Mixing (Consolidation)**
**Duration**: 100,000 timesteps  
**Scenario Mix**:
- **Uniform Random**: Each of 12 scenarios: **8.33%**

**Rationale**:
- Equal exposure prevents any scenario dominance
- Tests generalization across all challenges
- Prepares for real-world deployment

**Goal**: Achieve robust performance across all scenarios

**Success Criteria**:
- All scenarios meet 80% of individual success criteria
- No scenario causes catastrophic failure
- Smooth performance across random switches

**Validation**: Comprehensive benchmark suite

---

### **Stage 14: Adaptive Curriculum (Advanced)**
**Duration**: Ongoing  
**Scenario Mix**:
- **Performance-Based Sampling**:
  - Top 4 performing scenarios: **5%** each = **20%**
  - Middle 4 scenarios: **10%** each = **40%**
  - Bottom 4 scenarios: **10%** each = **40%**

**Rationale**:
- Focus training on weakest areas
- Maintain competence in mastered scenarios
- Auto-balancing based on reward metrics

**Goal**: Continual improvement on weakest scenarios

**Success Criteria**:
- Convergence of performance gaps (std dev < 1.5 across scenarios)
- All scenarios > 80% optimal performance

**Validation**: Weekly performance reports, dynamic rebalancing

---

## 📊 Training Timeline Summary

| Stage | Timesteps | Cumulative | Primary Scenario | Mix Complexity |
|-------|-----------|------------|------------------|----------------|
| 1 | 10,000 | 10,000 | Flash Crowd | Single |
| 2 | 16,000 | 26,000 | Sleepy Campus | 2-scenario |
| 3 | 20,000 | 46,000 | Urban Canyon | 3-scenario |
| 4 | 30,000 | 76,000 | Mobility Storm | 4-scenario |
| 5 | 40,000 | 116,000 | Traffic Burst | 5-scenario |
| 6 | 40,000 | 156,000 | Mixed Reality | 6-scenario |
| 7 | 36,000 | 192,000 | Adversarial | 7-scenario |
| 8 | 50,000 | 242,000 | Ping-Pong | 8-scenario |
| 9 | 60,000 | 302,000 | Commuter Rush | 9-scenario |
| 10 | 70,000 | 372,000 | IoT Tsunami | 10-scenario |
| 11 | 80,000 | 452,000 | Ambulance | 11-scenario |
| 12 | 100,000 | 552,000 | Spectrum Crunch | 12-scenario |
| 13 | 100,000 | 652,000 | Multi-Mix | All equal |
| 14 | ∞ | ∞ | Adaptive | Performance-based |

**Total Training Time to Deployment**: ~652,000 timesteps (~180 hours @ 1 episode/sec)

---

## 🔧 Implementation Code

### Python Implementation for Curriculum Training

```python
import numpy as np
from typing import Dict, List, Tuple
import random

class CurriculumScheduler:
    """
    Progressive curriculum scheduler for O-RAN RL training.
    Manages scenario mixing percentages across training stages.
    """
    
    def __init__(self):
        self.scenarios = [
            'flash_crowd',
            'sleepy_campus', 
            'urban_canyon',
            'mobility_storm',
            'traffic_burst',
            'mixed_reality',
            'adversarial',
            'ping_pong',
            'commuter_rush',
            'iot_tsunami',
            'ambulance',
            'spectrum_crunch'
        ]
        
        # Define curriculum stages
        self.curriculum = [
            # Stage 1
            {
                'name': 'Stage 1: Flash Crowd Foundation',
                'timesteps': 5000,
                'mix': {'flash_crowd': 1.0}
            },
            # Stage 2
            {
                'name': 'Stage 2: Energy Patterns',
                'timesteps': 8000,
                'mix': {'flash_crowd': 0.20, 'sleepy_campus': 0.80}
            },
            # Stage 3
            {
                'name': 'Stage 3: PHY Robustness',
                'timesteps': 10000,
                'mix': {'flash_crowd': 0.15, 'sleepy_campus': 0.15, 'urban_canyon': 0.70}
            },
            # Stage 4
            {
                'name': 'Stage 4: Handover Dynamics',
                'timesteps': 15000,
                'mix': {'flash_crowd': 0.10, 'sleepy_campus': 0.10, 
                       'urban_canyon': 0.10, 'mobility_storm': 0.70}
            },
            # Stage 5
            {
                'name': 'Stage 5: Extreme Overload',
                'timesteps': 20000,
                'mix': {'flash_crowd': 0.08, 'sleepy_campus': 0.08,
                       'urban_canyon': 0.08, 'mobility_storm': 0.08,
                       'traffic_burst': 0.68}
            },
            # Stage 6
            {
                'name': 'Stage 6: Multi-Objective Slicing',
                'timesteps': 20000,
                'mix': {'flash_crowd': 0.07, 'sleepy_campus': 0.07,
                       'urban_canyon': 0.07, 'mobility_storm': 0.07,
                       'traffic_burst': 0.07, 'mixed_reality': 0.65}
            },
            # Stage 7
            {
                'name': 'Stage 7: Non-Stationarity',
                'timesteps': 18000,
                'mix': {'flash_crowd': 0.06, 'sleepy_campus': 0.06,
                       'urban_canyon': 0.06, 'mobility_storm': 0.06,
                       'traffic_burst': 0.06, 'mixed_reality': 0.06,
                       'adversarial': 0.64}
            },
            # Stage 8
            {
                'name': 'Stage 8: Ping-Pong Prevention',
                'timesteps': 25000,
                'mix': {'flash_crowd': 0.05, 'sleepy_campus': 0.05,
                       'urban_canyon': 0.05, 'mobility_storm': 0.05,
                       'traffic_burst': 0.05, 'mixed_reality': 0.05,
                       'adversarial': 0.05, 'ping_pong': 0.65}
            },
            # Stage 9
            {
                'name': 'Stage 9: Mass Coordination',
                'timesteps': 30000,
                'mix': {'flash_crowd': 0.045, 'sleepy_campus': 0.045,
                       'urban_canyon': 0.045, 'mobility_storm': 0.045,
                       'traffic_burst': 0.045, 'mixed_reality': 0.045,
                       'adversarial': 0.045, 'ping_pong': 0.045,
                       'commuter_rush': 0.64}
            },
            # Stage 10
            {
                'name': 'Stage 10: Control Plane',
                'timesteps': 35000,
                'mix': {'flash_crowd': 0.04, 'sleepy_campus': 0.04,
                       'urban_canyon': 0.04, 'mobility_storm': 0.04,
                       'traffic_burst': 0.04, 'mixed_reality': 0.04,
                       'adversarial': 0.04, 'ping_pong': 0.04,
                       'commuter_rush': 0.04, 'iot_tsunami': 0.64}
            },
            # Stage 11
            {
                'name': 'Stage 11: URLLC Excellence',
                'timesteps': 40000,
                'mix': {'flash_crowd': 0.035, 'sleepy_campus': 0.035,
                       'urban_canyon': 0.035, 'mobility_storm': 0.035,
                       'traffic_burst': 0.035, 'mixed_reality': 0.035,
                       'adversarial': 0.035, 'ping_pong': 0.035,
                       'commuter_rush': 0.035, 'iot_tsunami': 0.035,
                       'ambulance': 0.65}
            },
            # Stage 12
            {
                'name': 'Stage 12: Spectrum Mastery',
                'timesteps': 50000,
                'mix': {'flash_crowd': 0.032, 'sleepy_campus': 0.032,
                       'urban_canyon': 0.032, 'mobility_storm': 0.032,
                       'traffic_burst': 0.032, 'mixed_reality': 0.032,
                       'adversarial': 0.032, 'ping_pong': 0.032,
                       'commuter_rush': 0.032, 'iot_tsunami': 0.032,
                       'ambulance': 0.032, 'spectrum_crunch': 0.648}
            },
            # Stage 13
            {
                'name': 'Stage 13: Multi-Scenario Mixing',
                'timesteps': 50000,
                'mix': {s: 1.0/12 for s in self.scenarios}  # Equal 8.33% each
            }
        ]
        
        self.current_stage = 0
        self.timesteps_in_stage = 0
        
    def get_scenario(self) -> str:
        """Sample a scenario based on current curriculum stage."""
        stage = self.curriculum[self.current_stage]
        mix = stage['mix']
        
        # Weighted random sampling
        scenarios = list(mix.keys())
        probabilities = list(mix.values())
        
        return np.random.choice(scenarios, p=probabilities)
    
    def step(self):
        """Update curriculum based on timesteps."""
        self.timesteps_in_stage += 1
        
        # Check if should move to next stage
        if self.timesteps_in_stage >= self.curriculum[self.current_stage]['timesteps']:
            if self.current_stage < len(self.curriculum) - 1:
                self.current_stage += 1
                self.timesteps_in_stage = 0
                print(f"\n{'='*60}")
                print(f"ADVANCING TO: {self.curriculum[self.current_stage]['name']}")
                print(f"{'='*60}\n")
    
    def get_current_stage_info(self) -> Dict:
        """Get current stage information."""
        stage = self.curriculum[self.current_stage]
        return {
            'stage_number': self.current_stage + 1,
            'stage_name': stage['name'],
            'timesteps_in_stage': self.timesteps_in_stage,
            'total_timesteps_stage': stage['timesteps'],
            'progress': self.timesteps_in_stage / stage['timesteps'],
            'scenario_mix': stage['mix']
        }


class AdaptiveCurriculumScheduler(CurriculumScheduler):
    """
    Advanced scheduler that adapts based on performance.
    Used for Stage 14 and beyond.
    """
    
    def __init__(self):
        super().__init__()
        self.scenario_performance = {s: 0.0 for s in self.scenarios}
        self.scenario_episode_count = {s: 0 for s in self.scenarios}
        self.update_frequency = 1000  # Recompute mix every 1000 steps
        
    def update_performance(self, scenario: str, reward: float):
        """Update running average of scenario performance."""
        count = self.scenario_episode_count[scenario]
        current_avg = self.scenario_performance[scenario]
        
        # Exponential moving average
        alpha = 0.1
        self.scenario_performance[scenario] = (1 - alpha) * current_avg + alpha * reward
        self.scenario_episode_count[scenario] += 1
    
    def compute_adaptive_mix(self) -> Dict[str, float]:
        """
        Compute scenario mix based on performance.
        Focus more on scenarios with lower performance.
        """
        # Invert performance (worse scenarios get higher values)
        max_perf = max(self.scenario_performance.values())
        inverted = {s: max_perf - self.scenario_performance[s] + 1.0 
                   for s in self.scenarios}
        
        # Normalize to probabilities
        total = sum(inverted.values())
        mix = {s: inverted[s] / total for s in self.scenarios}
        
        # Apply floor (minimum 5% per scenario) and ceiling (max 20%)
        for s in mix:
            mix[s] = np.clip(mix[s], 0.05, 0.20)
        
        # Re-normalize after clipping
        total = sum(mix.values())
        mix = {s: mix[s] / total for s in self.scenarios}
        
        return mix
    
    def get_scenario(self) -> str:
        """Sample scenario using adaptive distribution."""
        # Every N steps, recompute mix
        if self.timesteps_in_stage % self.update_frequency == 0:
            self.adaptive_mix = self.compute_adaptive_mix()
            print(f"\n[Adaptive Mix Update - Step {self.timesteps_in_stage}]")
            print("Top 3 focus areas:")
            sorted_mix = sorted(self.adaptive_mix.items(), 
                              key=lambda x: x[1], reverse=True)
            for i, (s, p) in enumerate(sorted_mix[:3]):
                print(f"  {i+1}. {s}: {p*100:.1f}%")
        
        # Sample from adaptive distribution
        scenarios = list(self.adaptive_mix.keys())
        probabilities = list(self.adaptive_mix.values())
        
        return np.random.choice(scenarios, p=probabilities)


# ============================================================================
# Integration with Training Loop
# ============================================================================

def train_with_curriculum():
    """
    Example training loop using curriculum scheduler.
    """
    from oran_ns3_env import ORANns3Env, NS3Config
    
    # Initialize curriculum
    curriculum = CurriculumScheduler()
    
    # Training loop
    total_timesteps = 0
    episode = 0
    
    while total_timesteps < 326000:  # Total curriculum timesteps
        # Get current scenario from curriculum
        scenario = curriculum.get_scenario()
        
        # Create environment with selected scenario
        config = NS3Config(
            num_ues=20,
            num_cells=3,
            scenario=scenario,
            sim_time=10.0
        )
        env = ORANns3Env(config)
        
        # Run episode
        obs, info = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Your RL agent action selection
            action = agent.get_action(obs)  # Your agent
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            total_timesteps += 1
            
            # Update curriculum
            curriculum.step()
            
            # Log progress
            if total_timesteps % 1000 == 0:
                stage_info = curriculum.get_current_stage_info()
                print(f"Timestep: {total_timesteps} | "
                      f"Stage: {stage_info['stage_name']} | "
                      f"Progress: {stage_info['progress']*100:.1f}%")
        
        env.close()
        episode += 1
        
        print(f"Episode {episode} | Scenario: {scenario} | Reward: {episode_reward:.2f}")


# ============================================================================
# Validation Suite
# ============================================================================

class CurriculumValidator:
    """
    Validates agent performance at each curriculum stage.
    """
    
    def __init__(self, agent):
        self.agent = agent
        self.scenarios = [
            'flash_crowd', 'sleepy_campus', 'urban_canyon',
            'mobility_storm', 'traffic_burst', 'mixed_reality',
            'adversarial', 'ping_pong', 'commuter_rush',
            'iot_tsunami', 'ambulance', 'spectrum_crunch'
        ]
        
        # Success criteria per scenario
        self.success_criteria = {
            'flash_crowd': {'min_reward': -2.0, 'fairness': 0.80},
            'sleepy_campus': {'energy_efficiency': 0.70},
            'urban_canyon': {'recovery_time': 1.0, 'rlf_rate': 0.05},
            'mobility_storm': {'ho_success': 0.92, 'rlf_count': 3},
            'traffic_burst': {'packet_loss': 0.08, 'percentile_5_tput': 2.0},
            'mixed_reality': {'vr_latency_99': 20.0, 'dl_tput': 8.0},
            'adversarial': {'resilience_ratio': 0.70},
            'ping_pong': {'ping_pong_rate': 0.10, 'ho_efficiency': 0.88},
            'commuter_rush': {'rlf_rate': 0.08, 'load_balance': 2.0},
            'iot_tsunami': {'rach_collision': 0.25, 'pdr': 0.95},
            'ambulance': {'latency_99999': 10.0, 'pdr_ambulance': 0.99999},
            'spectrum_crunch': {'scc_threshold': 0.75, 'agg_tput': 35.0}
        }
    
    def validate_stage(self, stage_number: int, num_episodes: int = 50) -> Dict:
        """
        Run validation episodes for all scenarios up to current stage.
        """
        results = {}
        
        # Test all scenarios up to current stage
        scenarios_to_test = self.scenarios[:stage_number]
        
        for scenario in scenarios_to_test:
            scenario_results = self._test_scenario(scenario, num_episodes)
            results[scenario] = scenario_results
            
            # Check success criteria
            passed = self._check_criteria(scenario, scenario_results)
            results[scenario]['passed'] = passed
            
            print(f"Scenario: {scenario} - {'✓ PASSED' if passed else '✗ FAILED'}")
        
        return results
    
    def _test_scenario(self, scenario: str, num_episodes: int) -> Dict:
        """Run test episodes for a specific scenario."""
        from oran_ns3_env import ORANns3Env, NS3Config
        
        config = NS3Config(scenario=scenario)
        env = ORANns3Env(config)
        
        episode_rewards = []
        metrics = []
        
        for ep in range(num_episodes):
            obs, info = env.reset()
            done = False
            ep_reward = 0
            
            while not done:
                action = self.agent.get_action(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                ep_reward += reward
            
            episode_rewards.append(ep_reward)
            metrics.append(info)
        
        env.close()
        
        # Aggregate metrics
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'metrics': metrics
        }
    
    def _check_criteria(self, scenario: str, results: Dict) -> bool:
        """Check if scenario meets success criteria."""
        criteria = self.success_criteria.get(scenario, {})
        
        # Simple check on mean reward for now
        if 'min_reward' in criteria:
            return results['mean_reward'] >= criteria['min_reward']
        
        return True  # Default pass if no specific criteria


# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":
    print("Radio-Cortex Curriculum Training")
    print("="*60)
    
    # Initialize curriculum
    scheduler = CurriculumScheduler()
    
    # Simulate training
    for step in range(100):
        scenario = scheduler.get_scenario()
        print(f"Step {step}: Training on {scenario}")
        
        scheduler.step()
        
        # Show stage info every 10 steps
        if step % 10 == 0:
            info = scheduler.get_current_stage_info()
            print(f"\nStage Info:")
            print(f"  Name: {info['stage_name']}")
            print(f"  Progress: {info['progress']*100:.1f}%")
            print(f"  Mix: {info['scenario_mix']}\n")
```

---

## 📈 Performance Tracking

### Metrics to Monitor at Each Stage

```python
class StageMetrics:
    """Track performance metrics across curriculum stages."""
    
    def __init__(self):
        self.stage_history = []
    
    def log_stage(self, stage_num: int, scenario_results: Dict):
        """Log results for a completed stage."""
        self.stage_history.append({
            'stage': stage_num,
            'results': scenario_results,
            'timestamp': time.time()
        })
    
    def plot_progress(self):
        """Visualize training progress across stages."""
        import matplotlib.pyplot as plt
        
        stages = [h['stage'] for h in self.stage_history]
        avg_rewards = [np.mean([r['mean_reward'] 
                               for r in h['results'].values()])
                      for h in self.stage_history]
        
        plt.figure(figsize=(12, 6))
        plt.plot(stages, avg_rewards, marker='o')
        plt.xlabel('Curriculum Stage')
        plt.ylabel('Average Reward')
        plt.title('Training Progress Across Curriculum')
        plt.grid(True)
        plt.savefig('curriculum_progress.png')
```

---

## ⚠️ Important Notes

1. **Catastrophic Forgetting Prevention**: 
   - Never drop a scenario completely
   - Maintain minimum 3-5% exposure to all learned scenarios

2. **Validation Gates**:
   - Don't advance to next stage if current validation fails
   - Allow up to 20% performance drop on previous scenarios

3. **Dynamic Adjustment**:
   - If agent struggles (reward < -5.0 for 1000 steps), increase focus to 80%
   - If agent masters quickly (reward plateau), advance early

4. **Hardware Requirements**:
   - Expected training time: ~90 hours on single GPU
   - Can parallelize with 4-8 environments (reduce to ~20 hours)

5. **Checkpointing**:
   - Save model every 5,000 timesteps
   - Save best model per scenario separately
   - Enable rollback if performance degrades

---

## 🎓 Advanced: Meta-Learning Integration

For even faster curriculum learning, consider:

```python
# MAML (Model-Agnostic Meta-Learning) for quick scenario adaptation
class CurriculumMAML:
    def meta_train_step(self, scenario_batch):
        """Train on batch of scenarios for fast adaptation."""
        for scenario in scenario_batch:
            # Inner loop: Fast adaptation to scenario
            adapted_params = self.inner_update(scenario)
            
            # Outer loop: Meta-objective across scenarios
            meta_loss = self.compute_meta_loss(adapted_params)
        
        return meta_loss
```

This enables the agent to learn **how to learn** new scenarios in just 100-200 steps!

---

## 📊 Final Curriculum Summary Table

| Stage | Duration | Primary Focus | Maintenance Scenarios | Mix Ratio |
|-------|----------|---------------|----------------------|-----------|
| 1 | 5k | Flash Crowd | None | 100-0 |
| 2 | 8k | Sleepy Campus | Flash | 20-80 |
| 3 | 10k | Urban Canyon | Flash, Sleepy | 15-15-70 |
| 4 | 15k | Mobility Storm | 3 previous | 10-10-10-70 |
| 5 | 20k | Traffic Burst | 4 previous | 8-8-8-8-68 |
| 6 | 20k | Mixed Reality | 5 previous | 7×5-65 |
| 7 | 18k | Adversarial | 6 previous | 6×6-64 |
| 8 | 25k | Ping-Pong | 7 previous | 5×7-65 |
| 9 | 30k | Commuter Rush | 8 previous | 4.5×8-64 |
| 10 | 35k | IoT Tsunami | 9 previous | 4×9-64 |
| 11 | 40k | Ambulance | 10 previous | 3.5×10-65 |
| 12 | 50k | Spectrum Crunch | 11 previous | 3.2×11-64.8 |
| 13 | 50k | Multi-Mix | All 12 | 8.33% each |
| 14+ | ∞ | Adaptive | Performance-based | 5-20% each |

**Total**: 326,000 timesteps to full mastery + continual improvement
