#!/bin/bash
# One-time setup: install conda hooks so `conda activate forge-ctcd`
# automatically configures PATH, LD_PRELOAD, and HF_HOME.
#
# Run this once after creating the conda environment:
#   conda env create -f environment.yml
#   conda activate forge-ctcd
#   source scripts/env/init.sh
#
# What it does:
#   Creates activate.d/deactivate.d scripts inside $CONDA_PREFIX so that
#   conda activate/deactivate automatically sets and unsets env vars.
#   These hooks are local to your conda env and not shared via git.

if [ -z "$CONDA_PREFIX" ]; then
    echo "ERROR: No conda environment active. Run 'conda activate forge-ctcd' first." >&2
    return 1
fi

ENV_NAME=$(basename "$CONDA_PREFIX")
if [ "$ENV_NAME" != "forge-ctcd" ]; then
    echo "ERROR: Expected 'forge-ctcd' env, got '$ENV_NAME'." >&2
    return 1
fi

# Resolve script directory (works in both bash and zsh)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Install pip deps from lockfile (--no-deps since versions are pre-resolved by uv)
REQUIREMENTS="$REPO_DIR/requirements.txt"
if [ -f "$REQUIREMENTS" ]; then
    echo "Installing pip dependencies from lockfile..."
    pip install --no-deps -r "$REQUIREMENTS"
fi

# Install Playwright Chromium browser (needed for KaTeX math scoring in olmOCR-Bench)
if python -c "import playwright" 2>/dev/null; then
    echo "Installing Playwright Chromium (timeout 5min)..."
    if ! timeout 300 python -m playwright install chromium; then
        echo "WARNING: Playwright install failed or timed out. olmocr_bench.py will install on-demand." >&2
    fi
fi

ACTIVATE_DIR="$CONDA_PREFIX/etc/conda/activate.d"
DEACTIVATE_DIR="$CONDA_PREFIX/etc/conda/deactivate.d"
mkdir -p "$ACTIVATE_DIR" "$DEACTIVATE_DIR"

ACTIVATE_SH="$SCRIPT_DIR/activate.sh"

cat > "$ACTIVATE_DIR/forge.sh" << HOOK
# Installed by scripts/env/init.sh — delegates to activate.sh
source "$ACTIVATE_SH"
HOOK

cat > "$DEACTIVATE_DIR/forge.sh" << 'HOOK'
unset HF_HOME
unset LD_PRELOAD

# Clean up aliases (from scripts/env/aliases.sh)
unset _FORGE_ROOT
unset -f sqw stail slog serr sc sjob sswitch sdefault snode 2>/dev/null
unalias sq sqall sgpu sh200 scall scpend saccts sfair 2>/dev/null
HOOK

echo "Conda hooks installed for '$ENV_NAME' environment."
echo "  $ACTIVATE_DIR/forge.sh"
echo "  $DEACTIVATE_DIR/forge.sh"

# Re-activate to apply hooks immediately
conda deactivate
conda activate forge-ctcd
echo "Environment ready."
