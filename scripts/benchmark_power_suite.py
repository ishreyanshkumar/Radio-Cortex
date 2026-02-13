import os
import subprocess
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# POWER SUITE SCENARIOS (Technical Scoping)
SCENARIOS = ['flash_crowd', 'sleepy_campus', 'ambulance', 'spectrum_crunch', 'iot_tsunami']
MODELS = ['bdh', 'nn']
# Adjust timesteps: enough to see a trend but fast for a demo.
TIMESTEPS = 20000 

RESULTS_FILE = "results/benchmark_comparison.csv"

def run_experiment(model, scenario, timesteps, ues=20):
    model_path = f"models/bench_{model}_{scenario}.pt"
    cmd = [
        "python3", "radio_cortex_complete.py",
        "--mode", "train",
        "--model", model,
        "--scenario", scenario,
        "--total-timesteps", str(timesteps),
        "--num-ues", str(ues),
        "--model-path", model_path,
        "--n-envs", "4",
        "--sim-time", "30.0"
    ]
    print(f"\n🚀 Running: {' '.join(cmd)}")
    start = time.time()
    subprocess.run(cmd)
    duration = time.time() - start
    return duration, model_path

def evaluate_zero_shot(model_path, model_type, scenario, test_ues=100):
    cmd = [
        "python3", "radio_cortex_complete.py",
        "--mode", "eval",
        "--model", model_type,
        "--scenario", scenario,
        "--num-ues", str(test_ues),
        "--model-path", model_path,
        "--total-timesteps", "5000" # Evaluation steps
    ]
    print(f"\n🧪 Evaluating Zero-Shot: {' '.join(cmd)}")
def evaluate_scramble(model_path, model_type, scenario):
    cmd = [
        "python3", "radio_cortex_complete.py",
        "--mode", "eval",
        "--model", model_type,
        "--scenario", scenario,
        "--model-path", model_path,
        "--shuffle-ues", # THE SCRAMBLE FLAG
        "--total-timesteps", "5000"
    ]
    print(f"\n🌪️  Evaluating Order Scramble: {' '.join(cmd)}")
    subprocess.run(cmd)

def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    performance_data = []

    for scenario in SCENARIOS:
        for model in MODELS:
            print(f"\n--- Benchmarking {model.upper()} on {scenario.upper()} ---")
            
            # 1. Train
            # Standard training on 20 UEs
            duration, model_path = run_experiment(model, scenario, TIMESTEPS)
            
            # 2. Evaluate (Zero-Shot Scaling for IoT)
            if scenario == 'iot_tsunami':
                # Test scalability on 100 UEs (The Scalability Trap)
                evaluate_zero_shot(model_path, model, scenario, test_ues=100)
            else:
                # Standard Evaluation
                evaluate_zero_shot(model_path, model, scenario, test_ues=20)
                
                # 3. Order Scramble Test (Permutation Invariance)
                # Shuffling UE indices. BDH should pass, NN should fail.
                print(f"🔄 [Order Scramble] Testing {model.upper()} with shuffled observations...")
                evaluate_scramble(model_path, model, scenario)
                
            # Note: In a real run, we'd parse results/experiment_results.csv here 
            # and append to our summary.
            
    print("\n✅ Benchmark Suite Completed.")
    print("Check 'results/experiment_results.csv' for detailed metrics.")

if __name__ == "__main__":
    main()
