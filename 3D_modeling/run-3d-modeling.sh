#!/usr/bin/env bash

# Set shell options for better error handling and debugging
set -euo pipefail

# Define variables
LOGS_DIR="logs"
LOG_FILE="${LOGS_DIR}/3D_modeling_$(date +%Y%m%d_%H%M%S).log"
PYTHON_SCRIPT="src/main.py"
PDB_DIR="Your input structure directory (without metal ions)"
OUTPUT_DIR="Your output directory"
PREDICTION_RESULT="Residue level prediction result directory"

# Create logs directory if it doesn't exist
mkdir -p "$LOGS_DIR"

# Start time
start_time=$(date +%s)

# Command to run
cmd="python -u $PYTHON_SCRIPT --ion FE \
 --gpu_index 0 \
 --restraint_force_constant 41840 \
 --prediction_result $PREDICTION_RESULT \
 --pdb-dir $PDB_DIR \
 --output-dir $OUTPUT_DIR"

# Write the command to the terminal and log file
printf "Running command: %s\n" "$cmd" | tee -a "$LOG_FILE"

# Run the python script and redirect both stdout and stderr to log file
if ! eval "$cmd" 2>&1 | tee -a "$LOG_FILE"; then
    printf "Error: Command failed. Check the log file for details: %s\n" "$LOG_FILE" >&2
    exit 1
fi

# Capture the end time
end_time=$(date +%s)

# Calculate and display the duration
duration=$((end_time - start_time))
printf "Job finished in %d seconds\n" "$duration" | tee -a "$LOG_FILE"

# Log the current date and time
date | tee -a "$LOG_FILE"