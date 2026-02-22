#!/usr/bin/env python3
"""
evaluate_all.py: Evaluates all Radio-Cortex models found in the models/ directory.
"""

import os
import subprocess
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Evaluate all models in the models directory.")
    parser.add_argument("--scenario", type=str, default="flash_crowd", 
                        help="Scenario to evaluate (e.g., flash_crowd, all)")
    parser.add_argument("--n-envs", type=int, default=4, 
                        help="Number of parallel environments/workers")
    args = parser.parse_args()

    models_dir = Path("models")
    if not models_dir.exists() or not models_dir.is_dir():
        print(f"Error: {models_dir} directory not found.")
        return

    pt_files = sorted(list(models_dir.glob("**/*.pt")))
    if not pt_files:
        print(f"No .pt files found in {models_dir} (or its subdirectories).")
        return

    print("="*60)
    print(f"FOUND {len(pt_files)} MODELS TO EVALUATE")
    print(f"PARALLEL WORKERS: {args.n_envs}")
    print("="*60)

    for pt_file in pt_files:
        print(f"\n>>> Starting evaluation for: {pt_file} <<<")
        
        # Try to infer model architecture from the path instead of filename.
        # Check if the folder name is one of the valid architectures.
        model_type = "bdh"  # Default fallback as requested
        valid_models = ['bdh', 'nn', 'gpt2', 'trxl', 'linear', 'universal', 'reformer']
        found_arch = False
        
        # Look through all parent parts of the path
        for part in pt_file.parts:
            # Also handle if the folder has a suffix or prefix (e.g., 'bdh_evals')
            for v_model in valid_models:
                if v_model in part.lower():
                    model_type = v_model
                    found_arch = True
                    # Break out of inner loop, but we want the most specific (deepest) match
                    # so we let the outer loop continue.
        
        if not found_arch:
             print(f"  [INFO] Could not infer architecture from path '{pt_file}'. Defaulting to 'bdh'.")
        else:
             print(f"  [INFO] Inferred architecture '{model_type}' for path '{pt_file}'")
        
        # Build the CLI command exactly as if you were running it manually
        cmd = [
            "python3", "radio_cortex_complete.py",
            "--mode", "eval",
            "--model", model_type,
            "--model-path", str(pt_file),
            "--scenario", args.scenario,
            "--n-envs", str(args.n_envs)
        ]
        
        print(f"  Command: {' '.join(cmd)}")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Running as subprocess handles memory bounds safely and guarantees 
                # ns-3 process destruction/cleanup between evaluate runs.
                result = subprocess.run(cmd)
                
                if result.returncode == 0:
                    print(f"✅ Finished evaluating {pt_file.name}\n")
                    break
                else:
                    print(f"⚠️  Evaluation attempt {attempt+1}/{max_retries} for {pt_file.name} failed (code {result.returncode})")
                    if attempt < max_retries - 1:
                        print(f"   Retrying...")
                    else:
                        print(f"❌ Evaluation of {pt_file.name} failed after {max_retries} attempts.\n")
            except Exception as e:
                print(f"❌ Exception during attempt {attempt+1}: {e}")
                if attempt == max_retries - 1:
                    print(f"❌ Failed to reach completion for {pt_file.name}\n")

if __name__ == "__main__":
    main()
