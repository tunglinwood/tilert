#!/bin/bash
# TileRT Interactive Mode Startup Script for GLM-4.5-Air Model
# Usage: ./start_glm45_air_interactive.sh [MODEL_WEIGHTS_DIR]

set -e

# Default configuration
MODEL_WEIGHTS_DIR="${1:-/data1/ZhipuAI/glm_4_5_air_converted_full/}"
MODEL_TYPE="glm_4_5_air"

# Ensure we're in the right directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================"
echo "TileRT Interactive Mode"
echo "======================================"
echo "Model: GLM-4.5-Air"
echo "Weights: $MODEL_WEIGHTS_DIR"
echo "======================================"

# Check if model weights exist
if [ ! -d "$MODEL_WEIGHTS_DIR" ]; then
    echo "ERROR: Model weights directory not found: $MODEL_WEIGHTS_DIR"
    exit 1
fi

# Set up Python path
export PYTHONPATH="$SCRIPT_DIR/python:$PYTHONPATH"

# Clear Python cache to ensure fresh imports
find "$SCRIPT_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$SCRIPT_DIR" -name "*.pyc" -delete 2>/dev/null || true

echo ""
echo "Starting TileRT in interactive mode..."
echo "Type '/exit' to quit"
echo ""

# Start the service in interactive mode
exec python python/generate.py \
    --model-weights-dir "$MODEL_WEIGHTS_DIR" \
    --model "$MODEL_TYPE" \
    --interactive \
    --max-new-tokens 4000
