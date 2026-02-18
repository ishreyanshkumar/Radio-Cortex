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

# Optimization: Prevent CUDA fragmentation
export PYTORCH_ALLOC_CONF="expandable_segments:True"

# ── Configuration (Optimized for 40-Core / 32GB VRAM Hardware) ──
MODEL="bdh"
N_ENVS=32                     # Optimized for 40+ core CPU (Leave 8 cores for OS/Kafka)
SIM_TIME=50.0                 # Longer episodes for meaningful congestion dynamics
DEVICE=""                     # auto-detect
MODEL_DIR="models/curriculum"
LOG_INTERVAL=10                
CHECKPOINT_INTERVAL=10        
START_STAGE=1
DRY_RUN=false

# ── Turbo-Charged Hyperparameters (Deep & Wide Training) ──
LR="3e-4"                        # Standard reliable LR
BATCH_SIZE=4096                  # Massive batch size for stable gradients (Requires >24GB VRAM)
ROLLOUT_STEPS=512                # 32 envs * 512 steps = 16,384 step buffer per update
GAMMA=0.995                      # Longer horizon for complex congestion
GAE_LAMBDA=0.95
CLIP_EPSILON=0.2
VF_COEF=0.5
ENT_COEF=0.01                    # Restored to 0.01 for better exploration in large batch
MAX_GRAD_NORM=0.5
HIDDEN_DIM=512                   # Wider network for higher capacity
PPO_EPOCHS=20                    # Deep updates per batch (extract max value from expensive rollouts)
LR_GAMMA=0.99                    # Slow decay for long curriculum

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
        --ppo-epochs "$PPO_EPOCHS" \
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

# Stage 1: Foundation - Flash Crowd Mastery (Bootstrap: ~48 updates)
STAGES[1]="--scenario flash_crowd --total-timesteps 800000"

# Stage 2: Green RAN - Sleepy Campus (Energy: ~90 updates)
STAGES[2]="--scenario flash_crowd:0.2,sleepy_campus:0.8 --total-timesteps 1500000"

# Stage 3: PHY Robustness - Urban Canyon (~120 updates)
STAGES[3]="--scenario flash_crowd:0.15,sleepy_campus:0.15,urban_canyon:0.7 --total-timesteps 2000000"

# Stage 4: Mobility - Mobility Storm (~150 updates)
STAGES[4]="--scenario flash_crowd:0.1,sleepy_campus:0.1,urban_canyon:0.1,mobility_storm:0.7 --total-timesteps 2500000"

# Stage 5: Congestion - Traffic Burst (~180 updates)
STAGES[5]="--scenario flash_crowd:0.08,sleepy_campus:0.08,urban_canyon:0.08,mobility_storm:0.08,traffic_burst:0.68 --total-timesteps 3000000"

# Stage 6: URLLC - Ambulance Priority (~180 updates)
STAGES[6]="--scenario flash_crowd:0.07,sleepy_campus:0.07,urban_canyon:0.07,mobility_storm:0.07,traffic_burst:0.07,ambulance:0.65 --total-timesteps 3000000"

# Stage 7: Capacity - Spectrum Crunch (~180 updates)
STAGES[7]="--scenario flash_crowd:0.06,sleepy_campus:0.06,urban_canyon:0.06,mobility_storm:0.06,traffic_burst:0.06,ambulance:0.06,spectrum_crunch:0.64 --total-timesteps 3000000"

# Stage 8: Consolidation - Multi-Mix Generalization (~240 updates)
STAGES[8]="--scenario flash_crowd:0.12,sleepy_campus:0.12,urban_canyon:0.12,mobility_storm:0.12,traffic_burst:0.12,ambulance:0.12,spectrum_crunch:0.12 --total-timesteps 4000000"

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