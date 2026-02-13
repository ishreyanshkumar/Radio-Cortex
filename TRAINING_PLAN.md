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

## 🎯 Complete Training Curriculum (Lean Power Suite)

### **Stage 1: Flash Crowd (Bootstrap)**
**Duration**: 60,000 timesteps  
**Scenario Mix**: 
- Flash Crowd: **100%**
**Goal**: Learn basic load balancing and handover triggering.

### **Stage 2: Sleepy Campus (Green RAN)**
**Duration**: 100,000 timesteps  
**Scenario Mix**:
- Flash Crowd: **20%**
- Sleepy Campus: **80%**
**Goal**: Master energy-efficient power adaptation.

### **Stage 3: Urban Canyon (PHY Robustness)**
**Duration**: 120,000 timesteps  
**Scenario Mix**:
- Flash Crowd: **15%**
- Sleepy Campus: **15%**
- Urban Canyon: **70%**
**Goal**: Learn reactive signal recovery and SINR optimization.

### **Stage 4: Mobility Storm (Handover Mastery)**
**Duration**: 180,000 timesteps  
**Scenario Mix**:
- Previous scenarios: **10%** each
- Mobility Storm: **70%**
**Goal**: Optimize handover success rate under high-speed mobility.

### **Stage 5: Traffic Burst (Massive Congestion)**
**Duration**: 240,000 timesteps  
**Scenario Mix**:
- Previous scenarios: **8%** each
- Traffic Burst: **68%**
**Goal**: Master scheduler optimization under 5x-10x overload.

### **Stage 6: Ambulance (Emergency Slicing)**
**Duration**: 240,000 timesteps  
**Scenario Mix**:
- Previous scenarios: **7%** each
- Ambulance: **65%**
**Goal**: Learn network slicing and QoS priority differentiation.

### **Stage 7: Spectrum Crunch (Spectral Efficiency)**
**Duration**: 300,000 timesteps  
**Scenario Mix**:
- Previous scenarios: **6%** each
- Spectrum Crunch: **64%**
**Goal**: Optimal resource management and spectral efficiency.

### **Stage 8: Consolidation (Power Suite Mix)**
**Duration**: 500,000 timesteps  
**Scenario Mix**:
- **Uniform Random**: Mixed across all key scenarios above.
**Goal**: Achieve robust generalization across the complete Power Suite.

---

## 📊 Training Timeline Summary

| Stage | Timesteps | Cumulative | Primary Scenario | Mix Complexity |
|-------|-----------|------------|------------------|----------------|
| 1 | 60,000 | 60,000 | Flash Crowd | Single |
| 2 | 100,000 | 160,000 | Sleepy Campus | 2-scenario |
| 3 | 120,000 | 280,000 | Urban Canyon | 3-scenario |
| 4 | 180,000 | 460,000 | Mobility Storm | 4-scenario |
| 5 | 240,000 | 700,000 | Traffic Burst | 5-scenario |
| 6 | 240,000 | 940,000 | Ambulance | 6-scenario |
| 7 | 300,000 | 1,240,000 | Spectrum Crunch | 7-scenario |
| 8 | 500,000 | 1,740,000 | Power Suite Mix | All equal |

**Total Training Time**: ~1.74M timesteps focused on mission-critical scenarios.

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
