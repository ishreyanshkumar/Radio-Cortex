#!/bin/bash

# Radio-Cortex Cleanup Utility
# This script removes logs, residuals, and temporary files.

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting Radio-Cortex cleanup...${NC}"

# 1. Remove Python cache
echo -e "Cleaning __pycache__..."
find . -type d -name "__pycache__" -exec rm -rf {} +

# 2. Remove log files
echo -e "Cleaning log files (.log, .jsonl)..."
rm -f *.log *.jsonl
rm -f results/*.log results/*.jsonl

# 3. Optional: Remove results and models (commented out by default)
# echo -e "Cleaning results and models..."
# rm -rf results/*
# rm -rf models/*.pth

echo -e "${GREEN}Cleanup complete!${NC}"
