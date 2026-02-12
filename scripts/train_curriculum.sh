#!/usr/bin/env bash
# =============================================================================
# Radio-Cortex: 14-Stage Curriculum Training Script
# =============================================================================
# This script implements progressive scenario mastery for the BDH policy.
# Each stage focuses on a new scenario while maintaining proficiency on
# previously learned ones via a weighted mix ratio.
#
# Usage:
#   bash scripts/train_curriculum.sh              # Full curriculum (Stage 1-14)
#   bash scripts/train_curriculum.sh --start 5    # Resume from Stage 5
#   bash scripts/train_curriculum.sh --dry-run     # Print plan without executing
#
# Total timesteps: ~652,000 (Stages 1-13) + infinite adaptive (Stage 14)
# Estimated wall-clock: 4-8 hours on 16-env / 8-core machine
# =============================================================================

set -euo pipefail

# ── Configuration ──
MODEL="bdh"
N_ENVS=16
SIM_TIME=30.0
DEVICE=""  # auto-detect (set to "cuda" or "cpu" to override)
MODEL_DIR="models/curriculum"
LOG_INTERVAL=3
CHECKPOINT_INTERVAL=3
START_STAGE=1
DRY_RUN=false

# ── Optimal Hyperparameters for 16 envs + BDH ──
LR="1e-4"
BATCH_SIZE=512
ROLLOUT_STEPS=256
GAMMA=0.995
GAE_LAMBDA=0.95
CLIP_EPSILON=0.15
VF_COEF=0.5
ENT_COEF=0.005
MAX_GRAD_NORM=0.5
HIDDEN_DIM=256

# ── Parse CLI args ──
while [[ $# -gt 0 ]]; do
    case "$1" in
        --start) START_STAGE="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --device) DEVICE="$2"; shift 2 ;;
        --n-envs) N_ENVS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Helpers ──
DEVICE_FLAG=""
if [[ -n "$DEVICE" ]]; then
    DEVICE_FLAG="--device $DEVICE"
fi

mkdir -p "$MODEL_DIR"

run_stage() {
    local stage=$1
    local desc=$2
    
    # Get configuration from STAGES array
    local config_str=${STAGES[$stage]}
    
    # Extract scenario and timesteps from config_str
    # Format: --scenario <scENario> --total-timesteps <steps>
    local scenario=$(echo "$config_str" | grep -oP '(?<=--scenario )\S+')
    local timesteps=$(echo "$config_str" | grep -oP '(?<=--total-timesteps )\d+')

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  STAGE $stage: $desc"
    echo "  Scenario: $scenario | Steps: $timesteps"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    local model_path="$MODEL_DIR/stage_${stage}.pt"
    local load_flag=""

    # Load previous stage checkpoint (if stage > 1)
    if [[ $stage -gt 1 ]]; then
        local prev=$((stage - 1))
        local prev_path="$MODEL_DIR/stage_${prev}.pt"
        if [[ -f "$prev_path" ]]; then
            echo "  📦 Loading checkpoint from Stage $prev: $prev_path"
            # Copy previous model as starting point
            cp "$prev_path" "$model_path"
        else
            echo "  ⚠️  No checkpoint found for Stage $prev ($prev_path), training from scratch"
        fi
    fi

    if $DRY_RUN; then
        echo "  [DRY RUN] Would run: python3 radio_cortex_complete.py --mode train \\"
        echo "      --scenario $scenario --model $MODEL --n-envs $N_ENVS \\"
        echo "      --total-timesteps $timesteps --model-path $model_path \\"
        echo "      --sim-time $SIM_TIME --learning-rate $LR --batch-size $BATCH_SIZE"
        return 0
    fi

    python3 radio_cortex_complete.py \
        --mode train \
        --scenario "$scenario" \
        --model "$MODEL" \
        --n-envs "$N_ENVS" \
        --total-timesteps "$timesteps" \
        --model-path "$model_path" \
        --sim-time "$SIM_TIME" \
        --learning-rate "$LR" \
        --batch-size "$BATCH_SIZE" \
        --rollout-steps "$ROLLOUT_STEPS" \
        --gamma "$GAMMA" \
        --gae-lambda "$GAE_LAMBDA" \
        --clip-epsilon "$CLIP_EPSILON" \
        --vf-coef "$VF_COEF" \
        --ent-coef "$ENT_COEF" \
        --max-grad-norm "$MAX_GRAD_NORM" \
        --hidden-dim "$HIDDEN_DIM" \
        --log-interval "$LOG_INTERVAL" \
        --checkpoint-interval "$CHECKPOINT_INTERVAL" \
        $DEVICE_FLAG
}

# =============================================================================
# CURRICULUM STAGES
# =============================================================================
# Each stage introduces one new scenario as the PRIMARY focus while cycling
# through previously mastered scenarios for maintenance.
#
# The "--scenario" flag selects the primary focus. Maintenance scenarios
# are handled by the built-in reward curriculum (Level 0→1→2) which
# automatically adjusts difficulty based on agent performance.
# =============================================================================

# ==============================================================================
# CURRICULUM STAGES DEFINITION
# ==============================================================================

declare -A STAGES

# Stage 1: Foundation - Flash Crowd Mastery
# Goal: Learn basic load balancing.
# Duration: 10,000 steps (Plan: 5k * 2)
STAGES[1]="--scenario flash_crowd --total-timesteps 10000"

# Stage 2: Energy Patterns Introduction
# Mix: Flash Crowd: 20%, Sleepy Campus: 80%
# Duration: 16,000 steps (Plan: 8k * 2)
STAGES[2]="--scenario flash_crowd:0.2,sleepy_campus:0.8 --total-timesteps 16000"

# Stage 3: PHY Layer Robustness
# Mix: Flash: 15%, Sleepy: 15%, Urban Canyon: 70%
# Duration: 20,000 steps (Plan: 10k * 2)
STAGES[3]="--scenario flash_crowd:0.15,sleepy_campus:0.15,urban_canyon:0.7 --total-timesteps 20000"

# Stage 4: Handover Dynamics - Mobility Storm
# Mix: Flash: 10%, Sleepy: 10%, Urban: 10%, Mobility: 70%
# Duration: 30,000 steps (Plan: 15k * 2)
STAGES[4]="--scenario flash_crowd:0.1,sleepy_campus:0.1,urban_canyon:0.1,mobility_storm:0.7 --total-timesteps 30000"

# Stage 5: Extreme Overload - Traffic Burst
# Mix: Prev 4: 8% each (32%), Traffic Burst: 68%
# Duration: 40,000 steps (Plan: 20k * 2)
STAGES[5]="--scenario flash_crowd:0.08,sleepy_campus:0.08,urban_canyon:0.08,mobility_storm:0.08,traffic_burst:0.68 --total-timesteps 40000"

# Stage 6: Multi-Objective - Mixed Reality Slicing
# Mix: Prev 5: 7% each (35%), Mixed Reality: 65%
# Duration: 40,000 steps (Plan: 20k * 2)
STAGES[6]="--scenario flash_crowd:0.07,sleepy_campus:0.07,urban_canyon:0.07,mobility_storm:0.07,traffic_burst:0.07,mixed_reality:0.65 --total-timesteps 40000"

# Stage 7: Non-Stationarity - Adversarial Environment
# Mix: Prev 6: 6% each (36%), Adversarial: 64%
# Duration: 36,000 steps (Plan: 18k * 2)
STAGES[7]="--scenario flash_crowd:0.06,sleepy_campus:0.06,urban_canyon:0.06,mobility_storm:0.06,traffic_burst:0.06,mixed_reality:0.06,adversarial:0.64 --total-timesteps 36000"

# Stage 8: Advanced Handover - Ping-Pong Prevention
# Mix: Prev 7: 5% each (35%), Ping-Pong: 65%
# Duration: 50,000 steps (Plan: 25k * 2)
STAGES[8]="--scenario flash_crowd:0.05,sleepy_campus:0.05,urban_canyon:0.05,mobility_storm:0.05,traffic_burst:0.05,mixed_reality:0.05,adversarial:0.05,ping_pong:0.65 --total-timesteps 50000"

# Stage 9: Mass Coordination - Commuter Rush
# Mix: Prev 8: 4.5% each (36%), Commuter Rush: 64%
# Duration: 60,000 steps (Plan: 30k * 2)
STAGES[9]="--scenario flash_crowd:0.045,sleepy_campus:0.045,urban_canyon:0.045,mobility_storm:0.045,traffic_burst:0.045,mixed_reality:0.045,adversarial:0.045,ping_pong:0.045,commuter_rush:0.64 --total-timesteps 60000"

# Stage 10: Control Plane - IoT Tsunami
# Mix: Prev 9: 4% each (36%), IoT Tsunami: 64%
# Duration: 70,000 steps (Plan: 35k * 2)
STAGES[10]="--scenario flash_crowd:0.04,sleepy_campus:0.04,urban_canyon:0.04,mobility_storm:0.04,traffic_burst:0.04,mixed_reality:0.04,adversarial:0.04,ping_pong:0.04,commuter_rush:0.04,iot_tsunami:0.64 --total-timesteps 70000"

# Stage 11: URLLC Excellence - Ambulance Priority
# Mix: Prev 10: 3.5% each (35%), Ambulance: 65%
# Duration: 80,000 steps (Plan: 40k * 2)
STAGES[11]="--scenario flash_crowd:0.035,sleepy_campus:0.035,urban_canyon:0.035,mobility_storm:0.035,traffic_burst:0.035,mixed_reality:0.035,adversarial:0.035,ping_pong:0.035,commuter_rush:0.035,iot_tsunami:0.035,ambulance:0.65 --total-timesteps 80000"

# Stage 12: Spectrum Mastery - Carrier Aggregation
# Mix: Prev 11: 3.2% each (35.2%), Spectrum Crunch: 64.8%
# Duration: 100,000 steps (Plan: 50k * 2)
STAGES[12]="--scenario flash_crowd:0.032,sleepy_campus:0.032,urban_canyon:0.032,mobility_storm:0.032,traffic_burst:0.032,mixed_reality:0.032,adversarial:0.032,ping_pong:0.032,commuter_rush:0.032,iot_tsunami:0.032,ambulance:0.032,spectrum_crunch:0.648 --total-timesteps 100000"

# Stage 13: Multi-Scenario Mixing (Consolidation)
# Mix: Uniform random (8.33% each)
# Duration: 100,000 steps (Plan: 50k * 2)
STAGES[13]="--scenario all --total-timesteps 100000"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     RADIO-CORTEX: 14-STAGE CURRICULUM TRAINING             ║"
echo "║     Model: $MODEL | Envs: $N_ENVS | Start: Stage $START_STAGE            ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# Stage 1: Flash Crowd (easiest – learn basic throughput control)
[[ $START_STAGE -le 1 ]]  && run_stage 1  "Flash Crowd (Bootstrap)"

# Stage 2: Sleepy Campus (easy – learn energy awareness)
[[ $START_STAGE -le 2 ]]  && run_stage 2  "Sleepy Campus (Energy)"

# Stage 3: Urban Canyon (medium – learn SINR recovery)
[[ $START_STAGE -le 3 ]]  && run_stage 3  "Urban Canyon (PHY Recovery)"

# Stage 4: Mobility Storm (medium – learn handover control)
[[ $START_STAGE -le 4 ]]  && run_stage 4  "Mobility Storm (Handovers)"

# Stage 5: Traffic Burst (medium – learn burst absorption)
[[ $START_STAGE -le 5 ]]  && run_stage 5  "Traffic Burst (Queue Mgmt)"

# Stage 6: Mixed Reality (hard – learn slice isolation)
[[ $START_STAGE -le 6 ]]  && run_stage 6  "Mixed Reality (Slicing)"

# Stage 7: Adversarial (hard – learn stability under chaos)
[[ $START_STAGE -le 7 ]]  && run_stage 7  "Adversarial (Stability)"

# Stage 8: Handover Ping-Pong (hard – learn hysteresis tuning)
[[ $START_STAGE -le 8 ]]  && run_stage 8  "Ping-Pong (Hysteresis)"

# Stage 9: Commuter Rush (hard – learn mass mobility)
[[ $START_STAGE -le 9 ]]  && run_stage 9  "Commuter Rush (Scale)"

# Stage 10: IoT Tsunami (hard – learn massive device scheduling)
[[ $START_STAGE -le 10 ]] && run_stage 10 "IoT Tsunami (Device Scale)"

# Stage 11: Ambulance (critical – learn QoS priority)
[[ $START_STAGE -le 11 ]] && run_stage 11 "Ambulance (QoS Priority)"

# Stage 12: Spectrum Crunch (critical – learn spectrum efficiency)
[[ $START_STAGE -le 12 ]] && run_stage 12 "Spectrum Crunch (Efficiency)"

# Stage 13: Multi-Mix (all 12 scenarios, generalization)
[[ $START_STAGE -le 13 ]] && run_stage 13 "Multi-Mix (Generalization)"

# Stage 14: Adaptive (infinite – continuous improvement)
if [[ $START_STAGE -le 14 ]]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  STAGE 14: ADAPTIVE (Continuous Improvement)"
    echo "  Run manually with scenario rotation until convergence."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "  Suggested command:"
    echo "  python3 radio_cortex_complete.py --mode train --scenario all \\"
    echo "      --model $MODEL --n-envs $N_ENVS --total-timesteps 200000 \\"
    echo "      --model-path $MODEL_DIR/stage_13.pt \\"
    echo "      --learning-rate 5e-5 --ent-coef 0.002 --sim-time 30.0"
fi

# ── Final Summary ──
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  CURRICULUM COMPLETE                                        ║"
echo "║  Final model: $MODEL_DIR/stage_13.pt                        ║"
echo "║                                                             ║"
echo "║  To evaluate:                                               ║"
echo "║  python3 radio_cortex_complete.py --mode eval --model bdh \\ ║"
echo "║      --model-path $MODEL_DIR/stage_13.pt --scenario all     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
