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
    local timesteps=$2
    local scenario=$3
    local desc=$4

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
        echo "      --sim-time $SIM_TIME --learning-rate $LR --batch-size $BATCH_SIZE \\"
        echo "      --rollout-steps $ROLLOUT_STEPS --gamma $GAMMA"
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

    echo "  ✅ Stage $stage complete → $model_path"
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

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     RADIO-CORTEX: 14-STAGE CURRICULUM TRAINING             ║"
echo "║     Model: $MODEL | Envs: $N_ENVS | Start: Stage $START_STAGE            ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# Stage 1: Flash Crowd (easiest – learn basic throughput control)
[[ $START_STAGE -le 1 ]]  && run_stage 1  10000 "flash_crowd"         "Flash Crowd (Bootstrap)"

# Stage 2: Sleepy Campus (easy – learn energy awareness)
[[ $START_STAGE -le 2 ]]  && run_stage 2  16000 "sleepy_campus"       "Sleepy Campus (Energy)"

# Stage 3: Urban Canyon (medium – learn SINR recovery)
[[ $START_STAGE -le 3 ]]  && run_stage 3  20000 "urban_canyon"        "Urban Canyon (PHY Recovery)"

# Stage 4: Mobility Storm (medium – learn handover control)
[[ $START_STAGE -le 4 ]]  && run_stage 4  30000 "mobility_storm"      "Mobility Storm (Handovers)"

# Stage 5: Traffic Burst (medium – learn burst absorption)
[[ $START_STAGE -le 5 ]]  && run_stage 5  40000 "traffic_burst"       "Traffic Burst (Queue Mgmt)"

# Stage 6: Mixed Reality (hard – learn slice isolation)
[[ $START_STAGE -le 6 ]]  && run_stage 6  40000 "mixed_reality"       "Mixed Reality (Slicing)"

# Stage 7: Adversarial (hard – learn stability under chaos)
[[ $START_STAGE -le 7 ]]  && run_stage 7  36000 "adversarial"         "Adversarial (Stability)"

# Stage 8: Handover Ping-Pong (hard – learn hysteresis tuning)
[[ $START_STAGE -le 8 ]]  && run_stage 8  50000 "handover_ping_pong"  "Ping-Pong (Hysteresis)"

# Stage 9: Commuter Rush (hard – learn mass mobility)
[[ $START_STAGE -le 9 ]]  && run_stage 9  60000 "commuter_rush"       "Commuter Rush (Scale)"

# Stage 10: IoT Tsunami (hard – learn massive device scheduling)
[[ $START_STAGE -le 10 ]] && run_stage 10 70000 "iot_tsunami"         "IoT Tsunami (Device Scale)"

# Stage 11: Ambulance (critical – learn QoS priority)
[[ $START_STAGE -le 11 ]] && run_stage 11 80000 "ambulance"           "Ambulance (QoS Priority)"

# Stage 12: Spectrum Crunch (critical – learn spectrum efficiency)
[[ $START_STAGE -le 12 ]] && run_stage 12 100000 "spectrum_crunch"    "Spectrum Crunch (Efficiency)"

# Stage 13: Multi-Mix (all 12 scenarios, generalization)
[[ $START_STAGE -le 13 ]] && run_stage 13 100000 "all"                "Multi-Mix (Generalization)"

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
