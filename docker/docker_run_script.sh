#!/bin/bash

# Script to run bandersnatch in Docker with proper volume mounts
# Usage: ./run_bandersnatch.sh config_file.yaml

if [ $# -eq 0 ]; then
    echo "Usage: $0 <config_file.yaml>"
    echo "Example: $0 20250429.yaml"
    exit 1
fi

CONFIG_FILE=$1

# Set your local paths here
DATA_DIR="/Users/matthewhooton/data"
CONFIG_DIR="/Users/matthewhooton/bandersnatch_runs/configs"
OUTPUT_DIR="/Users/matthewhooton/bandersnatch_runs"

# Run the container with volume mounts
docker run --rm \
    -v "$DATA_DIR":/app/data:ro \
    -v "$CONFIG_DIR":/app/configs:ro \
    -v "$OUTPUT_DIR":/app/output \
    bandersnatch:latest \
    /app/configs/$CONFIG_FILE

echo "Processing complete. Check $OUTPUT_DIR for results."