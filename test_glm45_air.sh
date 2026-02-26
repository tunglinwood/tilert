#!/bin/bash
# Quick test script for GLM-4.5-Air model
# Usage: ./test_glm45_air.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_WEIGHTS_DIR="/data1/ZhipuAI/glm_4_5_air_converted_full/"

echo "Testing TileRT with GLM-4.5-Air..."
echo ""

# Set up Python path
export PYTHONPATH="$SCRIPT_DIR/python:$PYTHONPATH"

# Clear cache
find "$SCRIPT_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Run a quick test with just 2 tokens
timeout 120 python python/generate.py \
    --model-weights-dir "$MODEL_WEIGHTS_DIR" \
    --model glm_4_5_air \
    --max-new-tokens 2 2>&1 | tee /tmp/tilert_test.log

if grep -q "RuntimeError" /tmp/tilert_test.log; then
    echo ""
    echo "TEST FAILED - See errors above"
    exit 1
fi

echo ""
echo "TEST PASSED - Service started successfully!"
