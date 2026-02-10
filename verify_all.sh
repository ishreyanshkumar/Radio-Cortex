#!/bin/bash
set -e

# Transformer 1 Training
echo "Testing Transformer 1 Training..."
/home/hp/.venv/bin/python radio_cortex_complete.py --mode train --model t1 --total-timesteps 1000 --n-envs 1

# Check if model exists
if [ -f "models/radiocortex_t1.pt" ]; then
    echo "SUCCESS: models/radiocortex_t1.pt created."
else
    echo "FAILURE: models/radiocortex_t1.pt not found."
    exit 1
fi

# Transformer 2 Training
echo "Testing Transformer 2 Training..."
/home/hp/.venv/bin/python radio_cortex_complete.py --mode train --model t2 --total-timesteps 1000 --n-envs 1

# Check if model exists
if [ -f "models/radiocortex_t2.pt" ]; then
    echo "SUCCESS: models/radiocortex_t2.pt created."
else
    echo "FAILURE: models/radiocortex_t2.pt not found."
    exit 1
fi

# Verify Evaluation for BDH (should load vec_normalize)
echo "Testing Evaluation for BDH..."
/home/hp/.venv/bin/python radio_cortex_complete.py --mode eval --model bdh --scenario flash_crowd

# Verify Evaluation for T1 (no vec_normalize expected for n_envs=1)
echo "Testing Evaluation for T1..."
/home/hp/.venv/bin/python radio_cortex_complete.py --mode eval --model t1 --scenario flash_crowd

echo "ALL TESTS PASSED"
