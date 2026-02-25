#!/bin/bash
set -euo pipefail 

# Export JAVA_HOME if not already set, commonly needed for Kafka/Zookeeper
export JAVA_HOME=${JAVA_HOME:-/usr/lib/jvm/default-java}
export PATH=$JAVA_HOME/bin:$PATH

# ==============================================================================
# Colors & Formatting Setup
# ==============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

function print_header() {
    echo -e "\n${BOLD}${CYAN}==================================================${NC}"
    echo -e "${BOLD}${CYAN}      $1${NC}"
    echo -e "${BOLD}${CYAN}==================================================${NC}\n"
}

function print_step() {
    echo -e "${BOLD}${BLUE}🚀 [$1] ${NC}${2}"
}

function print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

function print_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

function print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# ==============================================================================
# Error Handling Trap
# ==============================================================================
function catch_err() {
    local exit_code=$1
    local line_no=$2
    echo ""
    print_error "Command failed with exit code ${exit_code} on line ${line_no}!"
    echo -e "${RED}Aborting setup to prevent unintended consequences.${NC}"
    exit "$exit_code"
}
trap 'catch_err $? $LINENO' ERR

# ==============================================================================
# Spinner for long-running / silent commands
# ==============================================================================
function run_with_spin() {
    local msg=$1
    shift
    local logfile="/tmp/rc_setup_$$.log"
    
    # Run the command in background, redirecting output to logfile
    "$@" > "$logfile" 2>&1 &
    local pid=$!
    
    local spin='-\|/'
    local i=0
    # Save cursor position (fallback if tput fails)
    tput sc 2>/dev/null || true 
    
    while kill -0 $pid 2>/dev/null; do
        i=$(( (i+1) % 4 ))
        # Restore cursor
        tput rc 2>/dev/null || echo -ne "\r"
        echo -ne "${YELLOW}[${spin:$i:1}]${NC} ${msg} ... "
        sleep 0.1
    done
    
    # Clear line
    tput rc 2>/dev/null || echo -ne "\r"
    echo -ne "\033[K" 
    
    wait $pid
    local exit_code=$?
    
    if [ $exit_code -ne 0 ]; then
        print_error "'${msg}' failed! (Exit code $exit_code)"
        echo -e "${RED}--- Error Logs ---${NC}"
        cat "$logfile" || true
        echo -e "${RED}------------------${NC}"
        rm -f "$logfile"
        return $exit_code
    fi
    rm -f "$logfile"
}

# ==============================================================================
# MAIN SETUP EXECUTION
# ==============================================================================

print_header "Radio-Cortex End-to-End Setup"

# ---------------------------------------------------------
# 1. Clone Repositories
# ---------------------------------------------------------
print_step "1/4" "Setting up ns-3-allinone repository..."
if [ ! -d "ns-3-allinone" ]; then
    print_info "Cloning https://gitlab.com/nsnam/ns-3-allinone.git"
    # Run git clone interactively to keep its native progress bar
    git clone --progress https://gitlab.com/nsnam/ns-3-allinone.git
    cd ns-3-allinone
    
    run_with_spin "Downloading ns-3 dependencies" ./download.py
    cd ..
    print_success "ns-3-allinone cloned and dependencies downloaded."
else
    print_success "ns-3-allinone already exists. Skipping clone."
fi

# ---------------------------------------------------------
# 2. Set up Python Virtual Environment
# ---------------------------------------------------------
print_step "2/4" "Setting up Python Virtual Environment (.venv)..."
if [ ! -d ".venv" ]; then
    run_with_spin "Creating Python virtual environment" python3 -m venv .venv
    print_success ".venv created."
else
    print_success ".venv already exists."
fi

print_info "Activating .venv..."
# Using 'set +u' temporarily because activate script might reference unbound variables
set +u
source .venv/bin/activate
set -u

print_info "Installing Python dependencies (interactive progress)..."
pip install -r requirements.txt
print_success "Python dependencies installed."

# ---------------------------------------------------------
# 3. Build ns-3 & Link Scenario
# ---------------------------------------------------------
print_step "3/4" "Setting up & Building ns-3..."

# Target system dependencies conditionally
if command -v apt-get >/dev/null 2>&1; then
    if sudo -n true 2>/dev/null; then
       run_with_spin "Updating apt packages" sudo apt-get update -qq
       run_with_spin "Installing (librdkafka-dev, default-jre)" sudo apt-get install -y librdkafka-dev default-jre
       print_success "System dependencies installed."
    else
       print_info "Sudo password required for dependencies (librdkafka-dev, default-jre)."
       print_info "Please enter it if prompted, or safely skip if already installed."
       sudo apt-get update || true
       sudo apt-get install -y librdkafka-dev default-jre || print_info "Skipping apt install (permission denied or unnecessary)"
    fi
else
    print_info "apt-get not found. Skipping auto-install of system dependencies."
fi

# Find ns-3 directory safely
NS3_DIR=$(ls -d ns-3-allinone/ns-3.* 2>/dev/null | head -1 || true)
if [ -z "$NS3_DIR" ]; then
    print_error "No ns-3 directory found in ns-3-allinone/!"
    exit 1
fi
print_info "Detected ns-3 directory: $NS3_DIR"
cd "$NS3_DIR"

print_info "Linking scenario files to scratch/ ..."
cd scratch
rm -rf * # CLEANUP as per README
ln -sf ../../../oran-congestion-scenario.cc .
ln -sf ../../../CMakeLists.txt .
cd ..
print_success "Scenario linked."

run_with_spin "Configuring ns-3" ./ns3 configure -d optimized --enable-examples --enable-tests --disable-modules=lorawan,nr

print_info "Building ns-3... (showing native progress)"
./ns3 build
cd ../..
print_success "ns-3 built successfully."

# ---------------------------------------------------------
# 4. Bootstrap Kafka
# ---------------------------------------------------------
print_step "4/4" "Starting Kafka..."
if [ -f "scripts/run_kafka_native.sh" ]; then
    run_with_spin "Starting Kafka (Native)" bash scripts/run_kafka_native.sh
elif [ -f "scripts/start_kafka.sh" ]; then
    run_with_spin "Starting Kafka (Docker)" bash scripts/start_kafka.sh
else
    print_error "Cannot find a Kafka startup script (scripts/run_kafka_native.sh or scripts/start_kafka.sh)!"
    exit 1
fi
print_success "Kafka started."

# ---------------------------------------------------------
# 5. Run Training
# ---------------------------------------------------------
print_header "Setup Complete! Starting Training..."
print_info "Launching Radio-Cortex training mode..."

# Using set +e here optionally if training crashes are expected, but keeping it strict is safer.
bash scripts/train_quick.sh
