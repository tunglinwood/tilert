#!/bin/bash
# TileRT Service Startup Script for GLM-4.5-Air Model with MTP (Multi-Token Prediction)
# Usage: ./start_glm45_air_with_mtp.sh [MODEL_WEIGHTS_DIR] [MAX_NEW_TOKENS]

set -e

# Default configuration
MODEL_WEIGHTS_DIR="${1:-/data1/ZhipuAI/glm_4_5_air_converted_full/}"
MAX_NEW_TOKENS="${2:-20}"
MODEL_TYPE="glm_4_5_air"

# Ensure we're in the right directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================"
echo "TileRT Service Startup (with MTP)"
echo "======================================"
echo "Model: GLM-4.5-Air"
echo "Weights: $MODEL_WEIGHTS_DIR"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "MTP: Enabled"
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
echo "Starting TileRT service with MTP..."
echo ""

# Start the service with MTP enabled
exec python python/generate.py \
    --model-weights-dir "$MODEL_WEIGHTS_DIR" \
    --model "$MODEL_TYPE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --with-mtp
