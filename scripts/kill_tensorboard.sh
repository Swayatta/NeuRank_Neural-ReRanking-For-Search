#!/bin/bash

echo "Finding and killing TensorBoard processes..."

# Find TensorBoard processes
TB_PROCESSES=$(ps aux | grep "[t]ensorboard")

if [ -z "$TB_PROCESSES" ]; then
    echo "No TensorBoard processes found."
else
    echo "Found TensorBoard processes:"
    echo "$TB_PROCESSES"
    
    # Extract the PIDs
    TB_PIDS=$(echo "$TB_PROCESSES" | awk '{print $2}')
    
    echo -e "\nKilling TensorBoard processes with PIDs:"
    echo "$TB_PIDS"
    
    # Kill the processes
    echo "$TB_PIDS" | xargs kill
    
    echo "TensorBoard processes have been terminated."
    
    # Double-check if processes were killed
    REMAINING_PROCESSES=$(ps aux | grep "[t]ensorboard")
    if [ -z "$REMAINING_PROCESSES" ]; then
        echo "All TensorBoard processes successfully terminated."
    else
        echo "Warning: Some TensorBoard processes may still be running. You might need to manually kill them or use 'kill -9'."
    fi
fi