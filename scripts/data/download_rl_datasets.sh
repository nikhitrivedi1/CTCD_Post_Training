#!/bin/bash
# Download and prepare RL training datasets (synthmix + OmniDocBench)
#
# Run on cluster (interactive or SLURM):
#   bash scripts/data/download_rl_datasets.sh              # both datasets
#   bash scripts/data/download_rl_datasets.sh synthmix     # synthmix only
#   bash scripts/data/download_rl_datasets.sh omnidocbench # OmniDocBench only
#
# SLURM:
#   sbatch scripts/slurm/run_data_prep.sh rl

set -euo pipefail

DATASET="${1:-all}"
SYNTHMIX_DIR="data/ocr/synthmix"
OMNIDOC_DIR="data/ocr/omnidocbench"

echo "============================================================"
echo "RL Dataset Prep"
echo "Dataset: $DATASET"
echo "============================================================"

# ── synthmix ──────────────────────────────────────────────────────
if [[ "$DATASET" == "all" || "$DATASET" == "synthmix" ]]; then
    echo ""
    echo "=== Step 1: Download olmOCR-synthmix-1025 ==="
    if [[ -d "$SYNTHMIX_DIR/bench_data" ]]; then
        echo "Already downloaded: $SYNTHMIX_DIR/bench_data exists"
    else
        mkdir -p "$(dirname "$SYNTHMIX_DIR")"
        GIT_LFS_SKIP_SMUDGE=0 git clone \
            https://huggingface.co/datasets/allenai/olmOCR-synthmix-1025 \
            "$SYNTHMIX_DIR"
        echo "Download complete"
    fi

    echo ""
    echo "=== Step 2: Convert synthmix PDFs to images ==="
    if [[ -d "$SYNTHMIX_DIR/images" ]] && [[ $(ls "$SYNTHMIX_DIR/images/"*.jpg 2>/dev/null | wc -l) -gt 0 ]]; then
        echo "Images already exist: $(ls "$SYNTHMIX_DIR/images/"*.jpg | wc -l) files"
        echo "Re-running to fill any gaps..."
    fi
    python scripts/data/convert_synthmix.py --data_dir "$SYNTHMIX_DIR"

    echo ""
    echo "=== Step 3: Validate synthmix ==="
    python -c "
from pathlib import Path
import json, glob

data = Path('$SYNTHMIX_DIR')
jsonl_files = list((data / 'bench_data').glob('*.jsonl'))
images = list((data / 'images').glob('*.jpg'))

# Count tests
tests = 0
pages = set()
for f in sorted(jsonl_files):
    if f.name == 'pdfs' or f.is_dir():
        continue
    with open(f) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            tests += 1
            pages.add((t['pdf'], t['page']))

print(f'JSONL files: {len(jsonl_files)}')
print(f'Unique pages: {len(pages)}')
print(f'Total tests: {tests}')
print(f'Images: {len(images)}')
if len(images) < len(pages):
    print(f'WARNING: {len(pages) - len(images)} pages missing images!')
else:
    print('OK: All pages have images')
"
fi

# ── OmniDocBench ──────────────────────────────────────────────────
if [[ "$DATASET" == "all" || "$DATASET" == "omnidocbench" ]]; then
    echo ""
    echo "=== Step 4: Download OmniDocBench ==="
    if [[ -f "$OMNIDOC_DIR/OmniDocBench.json" ]]; then
        echo "Already downloaded: $OMNIDOC_DIR/OmniDocBench.json exists"
    else
        mkdir -p "$(dirname "$OMNIDOC_DIR")"
        GIT_LFS_SKIP_SMUDGE=0 git clone \
            https://huggingface.co/datasets/opendatalab/OmniDocBench \
            "$OMNIDOC_DIR"
        echo "Download complete"
    fi

    echo ""
    echo "=== Step 5: Convert OmniDocBench JSON to markdown ==="
    if [[ -d "$OMNIDOC_DIR/markdown" ]] && [[ $(ls "$OMNIDOC_DIR/markdown/"*.md 2>/dev/null | wc -l) -gt 0 ]]; then
        echo "Markdown already exists: $(ls "$OMNIDOC_DIR/markdown/"*.md | wc -l) files"
        echo "Skipping (delete markdown/ to reconvert)"
    else
        python scripts/data/convert_omnidocbench.py --data_dir "$OMNIDOC_DIR"
    fi

    echo ""
    echo "=== Step 6: Validate OmniDocBench ==="
    python -c "
from pathlib import Path

data = Path('$OMNIDOC_DIR')
images = list((data / 'images').glob('*.jpg')) + list((data / 'images').glob('*.png'))
md_files = list((data / 'markdown').glob('*.md'))

print(f'Images: {len(images)}')
print(f'Markdown GT files: {len(md_files)}')

# Check alignment
img_stems = {p.stem for p in images}
md_stems = {p.stem for p in md_files}
matched = img_stems & md_stems
unmatched_md = md_stems - img_stems

print(f'Matched (image + markdown): {len(matched)}')
if unmatched_md:
    print(f'WARNING: {len(unmatched_md)} markdown files without images')
else:
    print('OK: All markdown files have matching images')
"

    echo ""
    echo "=== Step 7: Spot-check OmniDocBench conversions ==="
    python scripts/data/convert_omnidocbench.py --data_dir "$OMNIDOC_DIR" --verify
fi

echo ""
echo "============================================================"
echo "Done!"
echo "============================================================"
