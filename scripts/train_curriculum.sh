#!/usr/bin/env bash
# =============================================================================
# Radio-Cortex: 8-Stage Curriculum Training Script
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

# Optimization: Prevent CUDA fragmentation
export PYTORCH_ALLOC_CONF="expandable_segments:True"

# ── Configuration (Optimized for 64-Core / 32GB VRAM Hardware) ──
MODEL="bdh"
N_ENVS=48                     # 64 cores - 16 (OS/Kafka/Python) = 48 workers
SIM_TIME=50.0                 # Longer episodes for meaningful congestion dynamics
DEVICE=""                     # auto-detect
MODEL_DIR="models/curriculum"
LOG_INTERVAL=15
CHECKPOINT_INTERVAL=5
START_STAGE=1
DRY_RUN=false

# ── Optimized Hyperparameters (Speed + Accuracy on 64-core) ──
# Buffer: 48 envs × 512 steps = 24,576 samples/update
# Mini-batches: 24,576 / 512 = 48 per epoch
# Grad steps/update: 8 epochs × 48 = 384
LR="3e-5"                        # Lower LR for stability with smaller batch size
BATCH_SIZE=512                   # Fast, frequent updates
ROLLOUT_STEPS=512                # 48×256 = 12,288 buffer (Fast batch)
GAMMA=0.99                       # Standard discount for 50s episodes
GAE_LAMBDA=0.95
CLIP_EPSILON=0.1                 # Tighter clipping for stability with small batches
VF_COEF=1.0                      # Increased to help value head keep up with sparse BDH policy
ENT_COEF=0.03                    # Decayed initial entropy to prevent randomness loops
MAX_GRAD_NORM=0.5
HIDDEN_DIM=512                   # Standard network for optimal GPU VRAM utilization
PPO_EPOCHS=10                    # More epochs over cleaner data
TARGET_KL=0.03                   # Stronger early-stopping threshold (Safe-Fast mode)
LR_GAMMA=0.995                   # Gentle decay across curriculum

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
    local norm_path="$MODEL_DIR/vec_normalize_stage_${stage}.pkl"
    local load_flag=""

    # Load previous stage checkpoint (if stage > 1)
    if [[ $stage -gt 1 ]]; then
        local prev=$((stage - 1))
        local prev_model="$MODEL_DIR/stage_${prev}.pt"
        local prev_norm="$MODEL_DIR/vec_normalize_stage_${prev}.pkl"
        
        if [[ -f "$prev_model" ]]; then
            echo "  📦 Loading model from Stage $prev: $prev_model"
            cp "$prev_model" "$model_path"
        fi
        
        if [[ -f "$prev_norm" ]]; then
            echo "  📦 Loading normalization stats from Stage $prev: $prev_norm"
            cp "$prev_norm" "$norm_path"
        fi
    fi

    if $DRY_RUN; then
        echo "  [DRY RUN] Would run: .venv/bin/python radio_cortex_complete.py --mode train \\"
        echo "      --scenario $scenario --model $MODEL --n-envs $N_ENVS \\"
        echo "      --total-timesteps $timesteps --model-path $model_path \\"
        echo "      --sim-time $SIM_TIME --learning-rate $LR --batch-size $BATCH_SIZE"
        return 0
    fi

    .venv/bin/python radio_cortex_complete.py \
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
        --ppo-epochs "$PPO_EPOCHS" \
        --target-kl "$TARGET_KL" \
        --lr-gamma "$LR_GAMMA" \
        --log-interval "$LOG_INTERVAL" \
        --checkpoint-interval "$CHECKPOINT_INTERVAL" \
        $DEVICE_FLAG

    # --- Log Cleanup: Reclaim space from ns-3 and metrics (GBs potentially) ---
    echo "  [Log Cleanup] Removing stage logs and CSV metrics to save room..."
    rm -f logs/ns3_out*.log logs/ns3_err*.log
    rm -f logs/reward_metrics_*.csv logs/kpm_verification*.jsonl
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
# CURRICULUM STAGES DEFINITION: The "Lean Power Suite" (8 Stages)
# ==============================================================================

# ==============================================================================
# CURRICULUM STAGES DEFINITION: The "Lean Power Suite" (8 Stages)
# ==============================================================================

declare -A STAGES

# Stage 1: Foundation - Flash Crowd Mastery (400k)
STAGES[1]="--scenario flash_crowd --total-timesteps 400000"

# Stage 2: Green RAN - Sleepy Campus (+50k = 450k)
STAGES[2]="--scenario flash_crowd:0.2,sleepy_campus:0.8 --total-timesteps 450000"

# Stage 3: PHY Robustness - Urban Canyon (+50k = 500k)
STAGES[3]="--scenario flash_crowd:0.15,sleepy_campus:0.15,urban_canyon:0.7 --total-timesteps 500000"

# Stage 4: Mobility - Mobility Storm (+50k = 550k)
STAGES[4]="--scenario flash_crowd:0.1,sleepy_campus:0.1,urban_canyon:0.1,mobility_storm:0.7 --total-timesteps 550000"

# Stage 5: Congestion - Traffic Burst (+50k = 600k)
STAGES[5]="--scenario flash_crowd:0.08,sleepy_campus:0.08,urban_canyon:0.08,mobility_storm:0.08,traffic_burst:0.68 --total-timesteps 600000"

# Stage 6: URLLC - Ambulance Priority (+50k = 650k)
STAGES[6]="--scenario flash_crowd:0.07,sleepy_campus:0.07,urban_canyon:0.07,mobility_storm:0.07,traffic_burst:0.07,ambulance:0.65 --total-timesteps 650000"

# Stage 7: Capacity - Spectrum Crunch (+50k = 700k)
STAGES[7]="--scenario flash_crowd:0.06,sleepy_campus:0.06,urban_canyon:0.06,mobility_storm:0.06,traffic_burst:0.06,ambulance:0.06,spectrum_crunch:0.64 --total-timesteps 700000"

# Stage 8: Consolidation - Multi-Mix Generalization (+300k = 1000k)
STAGES[8]="--scenario flash_crowd:0.12,sleepy_campus:0.12,urban_canyon:0.12,mobility_storm:0.12,traffic_burst:0.12,ambulance:0.12,spectrum_crunch:0.12 --total-timesteps 1000000"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     RADIO-CORTEX: LEAN POWER SUITE (8-STAGE)               ║"
echo "║     Model: $MODEL | Envs: $N_ENVS | Start: Stage $START_STAGE            ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# Execute Stages
[[ $START_STAGE -le 1 ]]  && run_stage 1  "Flash Crowd (Bootstrap)"
[[ $START_STAGE -le 2 ]]  && run_stage 2  "Sleepy Campus (Green RAN)"
[[ $START_STAGE -le 3 ]]  && run_stage 3  "Urban Canyon (PHY Robustness)"
[[ $START_STAGE -le 4 ]]  && run_stage 4  "Mobility Storm (Handover Mastery)"
[[ $START_STAGE -le 5 ]]  && run_stage 5  "Traffic Burst (Massive Congestion)"
[[ $START_STAGE -le 6 ]]  && run_stage 6  "Ambulance (Emergency Slicing)"
[[ $START_STAGE -le 7 ]]  && run_stage 7  "Spectrum Crunch (Spectral Efficiency)"
[[ $START_STAGE -le 8 ]]  && run_stage 8  "Consolidation (Power Suite Mix)"

# Final Summary
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  CURRICULUM COMPLETE: LEAN POWER SUITE"
echo "  Final model: $MODEL_DIR/stage_8.pt"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  Zero-Shot Generalization Test Candidates (Evaluation Only):"
echo "  1. ping_pong (Zero-Shot Hysteresis Adaptation)"
echo "  2. iot_tsunami (Zero-Shot Massive Device Scale)"
echo ""
echo "  To evaluate all (including Zero-Shot):"
echo "  python3 radio_cortex_complete.py --mode eval --model $MODEL \\"
echo "      --model-path $MODEL_DIR/stage_8.pt --scenario all"