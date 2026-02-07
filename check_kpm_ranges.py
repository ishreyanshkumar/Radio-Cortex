import time
import numpy as np
import sys
from collections import defaultdict
from oran_ns3_env import create_oran_env, NS3Config

def main():
    print("="*60)
    print("KPM RANGE CHECKER")
    print("="*60)
    
    # 1. Setup environment (Using a high-traffic scenario to ensure good range coverage)
    print("Initializing environment (10 UEs, 3 Cells, Flash Crowd)...")
    config = NS3Config(
        num_ues=10, 
        num_cells=3, 
        sim_time=20.0, # Run for 20 seconds to catch variations
        scenario="flash_crowd"
    )
    env = create_oran_env(config)
    
    # Trackers for min/max values
    # Structure: {'metric_name': {'min': float('inf'), 'max': float('-inf'), 'count': 0}}
    ue_stats = defaultdict(lambda: {'min': float('inf'), 'max': float('-inf'), 'count': 0})
    cell_stats = defaultdict(lambda: {'min': float('inf'), 'max': float('-inf'), 'count': 0})
    
    try:
        # 2. Start Simulation
        print("Starting ns-3 simulation...")
        state, info = env.reset()
        print("✓ Connected. Collecting samples...")
        
        # 3. Collection Loop
        # 1 step ~ 1-2 seconds
        steps_to_collect = 20 
        
        for i in range(steps_to_collect):
            # Random action to see if control affects metrics (it should)
            action = env.action_space.sample()
            next_state, reward, done, truncated, info = env.step(action)
            
            e2 = info.get('e2_metrics')
            if not e2:
                continue
                
            # --- Aggregation Logic ---
            
            # UE Metrics
            if e2.ue_metrics:
                # Get keys from first UE to know what to look for (assuming uniform keys)
                first_ue = list(e2.ue_metrics.values())[0]
                keys = first_ue.keys()
                
                for ue_data in e2.ue_metrics.values():
                    for k in keys:
                        val = ue_data.get(k)
                        if val is not None and isinstance(val, (int, float)):
                            if val < ue_stats[k]['min']: ue_stats[k]['min'] = val
                            if val > ue_stats[k]['max']: ue_stats[k]['max'] = val
                            ue_stats[k]['count'] += 1

            # Cell Metrics
            if e2.cell_metrics:
                first_cell = list(e2.cell_metrics.values())[0]
                keys = first_cell.keys()
                
                for cell_data in e2.cell_metrics.values():
                    for k in keys:
                        val = cell_data.get(k)
                        if val is not None and isinstance(val, (int, float)):
                            if val < cell_stats[k]['min']: cell_stats[k]['min'] = val
                            if val > cell_stats[k]['max']: cell_stats[k]['max'] = val
                            cell_stats[k]['count'] += 1

            # Progress bar
            print(f"\rProgress: {i+1}/{steps_to_collect} steps", end="")
            
            if done:
                break
        
        print("\n\n" + "="*80)
        print(f"{'METRIC (UE)':<25} | {'MIN':<15} | {'MAX':<15} | {'SAMPLES':<10}")
        print("-" * 80)
        for k in sorted(ue_stats.keys()):
            s = ue_stats[k]
            if s['count'] > 0:
                print(f"{k:<25} | {s['min']:<15.4f} | {s['max']:<15.4f} | {s['count']:<10}")
            else:
                print(f"{k:<25} | {'--':<15} | {'--':<15} | 0")

        print("\n" + "="*80)
        print(f"{'METRIC (CELL)':<25} | {'MIN':<15} | {'MAX':<15} | {'SAMPLES':<10}")
        print("-" * 80)
        for k in sorted(cell_stats.keys()):
            s = cell_stats[k]
            if s['count'] > 0:
                print(f"{k:<25} | {s['min']:<15.4f} | {s['max']:<15.4f} | {s['count']:<10}")
            else:
                print(f"{k:<25} | {'--':<15} | {'--':<15} | 0")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nClosing environment...")
        env.close()

if __name__ == "__main__":
    main()
