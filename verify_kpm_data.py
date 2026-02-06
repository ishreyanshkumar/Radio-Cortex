"""
Script to verify that real KPM data is being received from ns-3 via Kafka.
Prints raw metrics to console to confirm they are not defaults/zeroes.
"""
import time
import numpy as np
import sys
from oran_ns3_env import create_oran_env, NS3Config

def main():
    print("="*60)
    print("KPM DATA VERIFICATION")
    print("="*60)
    
    # 1. Setup minimal environment
    print("Initializing minimal environment (5 UEs, 1 Cell)...")
    config = NS3Config(
        num_ues=5, 
        num_cells=1, 
        sim_time=5.0, # 5 seconds
        scenario="flash_crowd"
    )
    env = create_oran_env(config)
    
    try:
        # 2. Start Simulation
        print("Starting ns-3 simulation (may take a few seconds)...")
        state, info = env.reset()
        print("✓ Connected to Environment")
        
        # 3. Step Loop
        print("\nCollecting KPM Reports...")
        # Create a dummy action (default parameters)
        # Action space depends on num_cells. 
        # But we can just sample from action space for verification
        action = env.action_space.sample()
        
        non_zero_detected = False
        
        for i in range(10): # Collect 10 steps
            next_state, reward, done, truncated, info = env.step(action)
            
            e2 = info.get('e2_metrics')
            if not e2:
                print(f"Step {i}: [No Data] Waiting for KPM...")
                continue
                
            # Extract key metrics
            ue_tputs = [m.get('throughput', 0.0) for m in e2.ue_metrics.values()]
            ue_delays = [m.get('delay', 0.0) for m in e2.ue_metrics.values()]
            cell_queues = [m.get('queue_length', 0) for m in e2.cell_metrics.values()]
            
            total_tput = sum(ue_tputs)
            
            print(f"Step {i}:")
            print(f"  UE Throughputs (Mbps): {[f'{x:.2f}' for x in ue_tputs]}")
            print(f"  UE Delays (ms):        {[f'{x:.2f}' for x in ue_delays]}")
            print(f"  Cell Queue Len:        {cell_queues}")
            
            if total_tput > 1e-6 or any(d > 0 for d in ue_delays):
                print("  STATUS: ✅ LIVE DATA (Non-zero values received)")
                non_zero_detected = True
            else:
                print("  STATUS: ⚠️  ZEROS (Simulation starting up?)")
            
            if done:
                break
            
            time.sleep(0.1)
            
        print("\n" + "="*60)
        if non_zero_detected:
            print("VERIFICATION SUCCESSFUL: Real KPM data detected.")
        else:
            print("VERIFICATION WARNING: Only zeros received. Check if traffic generation is active in scenario.")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Closing environment...")
        env.close()

if __name__ == "__main__":
    main()
