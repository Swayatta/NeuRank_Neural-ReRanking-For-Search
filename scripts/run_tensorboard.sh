#!/bin/bash

# Activate the virtual environment
source ~/my_venv/bin/activate

# Set the base log directory
BASE_LOG_DIR="./outputs/logs"

# Find the most recent log directory
LATEST_LOG_DIR=$(find "$BASE_LOG_DIR" -type d | sort -r | head -n 1)

if [ -z "$LATEST_LOG_DIR" ]; then
    echo "No log directory found. Exiting."
    deactivate
    exit 1
fi

# Start TensorBoard with the latest log directory
echo "Starting TensorBoard with log directory: $LATEST_LOG_DIR"
tensorboard --logdir="$LATEST_LOG_DIR" --port=6006

# Deactivate the virtual environment
deactivate