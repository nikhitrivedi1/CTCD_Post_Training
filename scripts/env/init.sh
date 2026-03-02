#!/bin/bash
# One-time setup: install conda hooks so `conda activate distillation`
# automatically configures PATH, LD_PRELOAD, and HF_HOME.
#
# Run this once after creating the conda environment:
#   conda env create -f environment.yml
#   conda activate distillation
#   source scripts/env/init.sh
#
# What it does:
#   Creates activate.d/deactivate.d scripts inside $CONDA_PREFIX so that
#   conda activate/deactivate automatically sets and unsets env vars.
#   These hooks are local to your conda env and not shared via git.

if [ -z "$CONDA_PREFIX" ]; then
    echo "ERROR: No conda environment active. Run 'conda activate distillation' first." >&2
    return 1
fi

ENV_NAME=$(basename "$CONDA_PREFIX")
if [ "$ENV_NAME" != "distillation" ]; then
    echo "ERROR: Expected 'distillation' env, got '$ENV_NAME'." >&2
    return 1
fi

ACTIVATE_DIR="$CONDA_PREFIX/etc/conda/activate.d"
DEACTIVATE_DIR="$CONDA_PREFIX/etc/conda/deactivate.d"
mkdir -p "$ACTIVATE_DIR" "$DEACTIVATE_DIR"

ACTIVATE_SH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/activate.sh"

cat > "$ACTIVATE_DIR/distillation.sh" << HOOK
# Installed by scripts/env/init.sh — delegates to activate.sh
source "$ACTIVATE_SH"
HOOK

cat > "$DEACTIVATE_DIR/distillation.sh" << 'HOOK'
unset HF_HOME
unset LD_PRELOAD
HOOK

echo "Conda hooks installed for '$ENV_NAME' environment."
echo "  $ACTIVATE_DIR/distillation.sh"
echo "  $DEACTIVATE_DIR/distillation.sh"

# Re-activate to apply hooks immediately
conda deactivate
conda activate distillation
echo "Environment ready."
