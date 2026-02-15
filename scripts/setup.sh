#!/bin/bash
set -e  # Exit on error

echo "=================================================="
echo "      Radio-Cortex End-to-End Setup"
echo "=================================================="

# 1. Clone Repositories (if needed)
if [ ! -d "ns-3-allinone" ]; then
    echo "[1/4] Cloning ns-3-allinone..."
    git clone https://gitlab.com/nsnam/ns-3-allinone.git
    cd ns-3-allinone
    ./download.py ns-3.46.1
    cd ..
else
    echo "[1/4] ns-3-allinone already exists."
fi

# 2. Set up Python Virtual Environment
if [ ! -d ".venv" ]; then
    echo "[2/4] Creating virtual environment (.venv)..."
    python3 -m venv .venv
fi
echo "Activating .venv..."
source .venv/bin/activate
echo "Installing python dependencies..."
pip install -r requirements.txt

# 3. Build ns-3 & Link Scenario
echo "[3/4] Building ns-3..."
# Install system dependency if possible (requires sudo, might fail in some envs)
if command -v apt-get &> /dev/null; then
    echo "Attempting to install librdkafka-dev (sudo required)..."
    sudo apt-get update && sudo apt-get install -y librdkafka-dev || echo "Skipping apt install (permission denied or unnecessary)"
fi

cd ns-3-allinone/ns-3.46.1

echo "Linking scenario..."
cd scratch
rm -rf * # CLEANUP as per README
ln -sf ../../../oran-congestion-scenario.cc .
ln -sf ../../../CMakeLists.txt .
cd ..

./ns3 configure -d optimized --enable-examples --enable-tests
./ns3 build

cd ../..

# 4. Bootstrap Kafka
echo "[4/4] Starting Kafka..."
if [ -f "scripts/run_kafka_native.sh" ]; then
    bash scripts/run_kafka_native.sh
else
    bash scripts/start_kafka.sh
fi

echo "=================================================="
echo "      Setup Complete! Starting Training..."
echo "=================================================="

# 5. Run Training
python3 radio_cortex_complete.py --mode train --scenario all --n-envs 4 --total-timesteps 50000
