import os
import subprocess
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# POWER SUITE: Full Multi-Architecture Benchmark
# ============================================================================
# Compares BDH against all transformer and non-transformer baselines
# across multiple O-RAN scenarios. Produces:
#   - Per-model training results
#   - Zero-shot scaling evaluation (20 UE → 100 UE)
#   - Results CSV in results/
# ============================================================================

SCENARIOS = ['flash_crowd', 'sleepy_campus', 'ambulance', 'spectrum_crunch', 'iot_tsunami']

# ALL architectures for comprehensive comparison
MODELS = ['bdh', 'nn', 'gpt2', 'trxl', 'linear', 'universal', 'reformer']

# Adjust timesteps: enough to see a trend but fast for a demo.
TIMESTEPS = 20000

RESULTS_FILE = "results/benchmark_comparison.csv"

def run_experiment(model, scenario, timesteps, ues=20):
    """Train a model on a scenario and return wall-clock time."""
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

def evaluate_model(model_path, model_type, scenario, test_ues=20, tag="standard"):
    """Run evaluation episode and return wall-clock time."""
    cmd = [
        "python3", "radio_cortex_complete.py",
        "--mode", "eval",
        "--model", model_type,
        "--scenario", scenario,
        "--num-ues", str(test_ues),
        "--model-path", model_path,
        "--total-timesteps", "5000"
    ]
    print(f"\n🧪 [{tag}] Evaluating {model_type.upper()} on {scenario} (UEs={test_ues}): {' '.join(cmd)}")
    start = time.time()
    subprocess.run(cmd)
    duration = time.time() - start
    return duration

def evaluate_zero_shot(model_path, model_type, scenario, test_ues=100):
    """Zero-shot scaling: train on 20 UEs, evaluate on test_ues.
    This tests BDH's scale-invariance vs per-UE models."""
    return evaluate_model(model_path, model_type, scenario, test_ues=test_ues, tag="Zero-Shot Scale")


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    performance_data = []

    for scenario in SCENARIOS:
        for model in MODELS:
            print(f"\n{'='*60}")
            print(f"  Benchmarking {model.upper()} on {scenario.upper()}")
            print(f"{'='*60}")

            # 1. Train (20 UEs standard)
            train_time, model_path = run_experiment(model, scenario, TIMESTEPS)

            # 2. Standard Evaluation (same UE count as training)
            eval_time = evaluate_model(model_path, model, scenario, test_ues=20, tag="Standard")

            # 3. Zero-Shot Scale Test (train on 20, eval on 100)
            # Only cell-centric models (BDH) should survive this cleanly.
            # Per-UE models will have dimension mismatches or degraded performance.
            if scenario == 'iot_tsunami':
                zs_time = evaluate_zero_shot(model_path, model, scenario, test_ues=100)
            else:
                zs_time = 0.0

            row = {
                'model': model,
                'scenario': scenario,
                'train_time_s': round(train_time, 1),
                'eval_time_s': round(eval_time, 1),
                'zero_shot_time_s': round(zs_time, 1),
                'timesteps': TIMESTEPS,
            }
            performance_data.append(row)

    # Save summary
    df = pd.DataFrame(performance_data)
    df.to_csv(RESULTS_FILE, index=False)
    print(f"\n✅ Benchmark Suite Completed. Summary saved to {RESULTS_FILE}")
    print(df.to_string(index=False))

    # Also check results/experiment_results.csv for detailed per-step metrics
    print("\nCheck 'results/experiment_results.csv' for detailed per-step metrics.")


if __name__ == "__main__":
    main()
