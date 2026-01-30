#!/bin/bash
# Local testing script for knowledge distillation
# Use this for quick testing on M2 or local machines

set -e  # Exit on error

# Configuration
CONFIG=${1:-configs/mnli/bert_distilbert/default.yaml}
OUTPUT_DIR=${2:-./outputs/local_test}
MAX_STEPS=${3:-10}  # Default to 10 steps for quick testing

echo "========================================="
echo "Local Testing Script"
echo "========================================="
echo "Config: $CONFIG"
echo "Output: $OUTPUT_DIR"
echo "Max steps: $MAX_STEPS"
echo "========================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run training
python train.py \
    --config "$CONFIG" \
    --output_dir "$OUTPUT_DIR" \
    --max_steps "$MAX_STEPS"

echo ""
echo "========================================="
echo "Local test complete!"
echo "Check outputs in: $OUTPUT_DIR"
echo "========================================="
