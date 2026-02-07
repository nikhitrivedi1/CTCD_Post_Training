#!/bin/bash
# Local testing script for knowledge distillation
# Use this for quick testing on M2 or local machines

set -e  # Exit on error

# Configuration
CONFIG=${1:-configs/mnli/bert_distilbert/default.yaml}
MAX_STEPS=${2:-10}  # Default to 10 steps for quick testing

echo "========================================="
echo "Local Testing Script"
echo "========================================="
echo "Config: $CONFIG"
echo "Max steps: $MAX_STEPS"
echo "========================================="

# Run distillation (distill.py creates output directory from config)
python distill.py \
    --config "$CONFIG" \
    --max_steps "$MAX_STEPS"

echo ""
echo "========================================="
echo "Local test complete!"
echo "========================================="
