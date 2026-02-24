#!/bin/bash

# Configuration
MODEL="bdh"
TIMESTEPS=2000
SIM_TIME=200.0
# Define the 12 scenarios
SCENARIOS=(
    "flash_crowd"
    "mobility_storm"
    "traffic_burst"
    "handover_ping_pong"
    "sleepy_campus"
    "ambulance"
    "adversarial"
    "commuter_rush"
    "mixed_reality"
    "urban_canyon"
    "iot_tsunami"
    "spectrum_crunch"
)

# Output directory for databases
RESULTS_DIR="results"

echo "================================================================="
echo "Starting evaluation of model: $MODEL across ${#SCENARIOS[@]} scenarios"
echo "Timesteps: $TIMESTEPS | Simulation Time: $SIM_TIME seconds"
echo "================================================================="

for SCENARIO in "${SCENARIOS[@]}"; do
    echo "--------------------------------------------------------"
    echo "Running Scenario: $SCENARIO..."
    
    # Run the evaluation command
    python3 radio_cortex_complete.py \
        --mode eval \
        --model "$MODEL" \
        --scenario "$SCENARIO" \
        --total-timesteps "$TIMESTEPS" \
        --sim-time "$SIM_TIME"
        
    # Check if the command succeeded
    if [ $? -eq 0 ]; then
        # Find the latest simulation database file generated in 'results'
        LATEST_DB=$(ls -t "$RESULTS_DIR"/simulation_data_*.db 2>/dev/null | head -n 1)
        
        if [ -n "$LATEST_DB" ]; then
            # Construct the new filename
            NEW_DB="$RESULTS_DIR/visualise_${SCENARIO}.db"
            
            # Rename the file
            mv "$LATEST_DB" "$NEW_DB"
            echo "Successfully generated and saved simulation database to: $NEW_DB"
        else
            echo "Warning: Evaluation completed, but no simulation_data_*.db file was found in $RESULTS_DIR/."
        fi
    else
        echo "Error: Evaluation failed for scenario '$SCENARIO'."
    fi
done

echo "================================================================="
echo "All evaluations completed!"
echo "================================================================="
