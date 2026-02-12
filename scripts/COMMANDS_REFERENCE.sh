#!/bin/bash
# Radio-Cortex Commands Reference

# --- Setup ---
# Build ns-3
# cd ns-3-allinone/ns-3.46.1
# ./ns3 configure -d optimized --enable-examples --enable-tests
# ./ns3 build

# Start Kafka
# ./scripts/start_kafka.sh

# --- Training ---
# Quick start
# python3 radio_cortex_complete.py --mode train --scenario all

# Fast training (Recommended)
# python3 radio_cortex_complete.py --mode train --scenario all --n-envs 4 --total-timesteps 50000

# Training specific model
# python3 radio_cortex_complete.py --mode train --model bdh --scenario flash_crowd

# --- Evaluation ---
# Baseline (No AI)
# python3 radio_cortex_complete.py --mode eval --model base --scenario flash_crowd

# AI Agent (Default BDH)
# python3 radio_cortex_complete.py --mode eval --model bdh --scenario flash_crowd

# Parallel Eval
# python3 radio_cortex_complete.py --mode eval --model bdh --scenario all --n-envs 4

# --- Dashboard ---
# python3 -m http.server 8080
# Open http://localhost:8080/dashboard.html
