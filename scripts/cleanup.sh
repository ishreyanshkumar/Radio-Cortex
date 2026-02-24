#!/bin/bash

# Radio-Cortex Cleanup Utility
# This script removes logs, residuals, and temporary files.

# Ensure we are in the project root
cd "$(dirname "$0")/.."

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting Radio-Cortex cleanup...${NC}"

# 1. Remove Python cache
echo -e "Cleaning __pycache__..."
find . -type d -name "__pycache__" -exec rm -rf {} +

# 2. Remove log files
echo -e "Cleaning root and results log files (.log, .jsonl)..."
rm -f *.log *.jsonl
rm -f results/*.log results/*.jsonl

# 3. Remove logs directory contents (logs, metrics, etc.)
echo -e "Cleaning logs directory..."
rm -f logs/*.log logs/*.jsonl logs/*.csv

# 4. Remove Kafka logs
echo -e "Cleaning Kafka logs..."
rm -f kafka_*/logs/*.log

# 5. Optional: Remove results and models (commented out by default)
# echo -e "Cleaning results and models..."
# rm -rf results/*
# rm -rf models/*.pth

echo -e "${GREEN}Cleanup complete!${NC}"
