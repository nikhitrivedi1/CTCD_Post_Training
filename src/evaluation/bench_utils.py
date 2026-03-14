"""Shared utilities for bench scripts.

Provides consistent output directory handling, metadata, sharding, job
registration, and result saving so all bench scripts behave the same way.

Used by:
    scripts/benchmarks/omnidoc_bench.py      - OmniDocBench (text, tables, formulas, reading order)
    scripts/benchmarks/olmocr_bench.py       - olmOCR-Bench (pass/fail test suite)
    scripts/benchmarks/formatting_bench.py   - Formatting bench (bold/italic/strikethrough F1)

Output directory layout (consistent across all benches):
    outputs/bench/{bench_name}/{model_name}/{timestamp}/
        metadata.json       - model info, generation params, SLURM job history
        results.json        - aggregate scores
        markdowns/          - generated markdown files (one per page)
        page_scores.jsonl   - per-page detail scores (omnidoc, formatting)
        test_details.jsonl  - per-test detail results (olmocr)

Examples (patterns work the same across all three benches):

    # --- New run (auto-derives output dir) ---
    python scripts/benchmarks/omnidoc_bench.py --model Qwen/Qwen3-VL-2B-Instruct
    python scripts/benchmarks/olmocr_bench.py --checkpoint outputs/fmt_sft
    python scripts/benchmarks/formatting_bench.py --checkpoint outputs/fmt_sft

    # --- Explicit output dir ---
    python scripts/benchmarks/omnidoc_bench.py --model X --output outputs/bench/omnidoc/my-run
    python scripts/benchmarks/formatting_bench.py --checkpoint outputs/fmt_sft --output outputs/bench/formatting/my-run

    # --- Resume (model restored from metadata.json, skips already-generated pages) ---
    python scripts/benchmarks/omnidoc_bench.py --output outputs/bench/omnidoc/my-run
    python scripts/benchmarks/olmocr_bench.py --output outputs/bench/olmocr/my-run
    python scripts/benchmarks/formatting_bench.py --output outputs/bench/formatting/my-run

    # --- Sharded run (4 parallel jobs, each writes to same output dir) ---
    python scripts/benchmarks/olmocr_bench.py --model X --output D --shard 1/4
    python scripts/benchmarks/olmocr_bench.py --model X --output D --shard 2/4
    python scripts/benchmarks/olmocr_bench.py --model X --output D --shard 3/4
    python scripts/benchmarks/olmocr_bench.py --model X --output D --shard 4/4

    # --- Score only (after all shards finish, or to re-score existing markdowns) ---
    python scripts/benchmarks/olmocr_bench.py --score_only --output D
    python scripts/benchmarks/omnidoc_bench.py --score_only --output outputs/bench/omnidoc/my-run
    python scripts/benchmarks/formatting_bench.py --score_only --output outputs/bench/formatting/my-run

    # --- Sharded resume (shard 2 interrupted, resubmit same shard) ---
    python scripts/benchmarks/omnidoc_bench.py --output D --shard 2/4

    # --- Ad-hoc formatting score (unique to formatting bench) ---
    python scripts/benchmarks/formatting_bench.py --pred_dir outputs/bench/omnidoc/run/markdowns --gt_dir data/ocr/omnidocbench/markdown
"""

import glob
import json
import logging
import os
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.utils.jobs import get_job_key, guard_duplicate_job, locked_incomplete_jobs

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(level: int = logging.INFO):
    """Configure logging for bench scripts. Logs to stdout so SLURM captures
    output in the .out file (not .err)."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


# ---------------------------------------------------------------------------
# Graceful shutdown (SLURM SIGUSR1 / SIGTERM)
# ---------------------------------------------------------------------------
_shutdown_requested = False


def is_shutdown_requested() -> bool:
    """Check if a graceful shutdown was requested."""
    return _shutdown_requested


def _handle_shutdown_signal(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    sig_name = signal.Signals(signum).name
    logger.warning(f"{sig_name} received - finishing current page then shutting down")


signal.signal(signal.SIGTERM, _handle_shutdown_signal)
signal.signal(signal.SIGUSR1, _handle_shutdown_signal)


# ---------------------------------------------------------------------------
# Model name / output directory
# ---------------------------------------------------------------------------
def model_name_from_args(model_id: str = None, checkpoint_path: str = None) -> str:
    """Extract a short model name from CLI args.

    Args:
        model_id: HuggingFace model ID (e.g. "Qwen/Qwen3-VL-2B-Instruct").
        checkpoint_path: Local checkpoint path (e.g. "outputs/fmt_sft").

    Returns:
        Short name string (e.g. "Qwen3-VL-2B-Instruct" or "fmt_sft").
    """
    if checkpoint_path:
        cp = Path(checkpoint_path)
        # outputs/fmt_sft/checkpoint-400 -> "fmt_sft"
        if cp.name.startswith("checkpoint-"):
            return cp.parent.name
        return cp.name
    return model_id.split("/")[-1]


def derive_output_dir(
    base_dir: str, model_id: str = None, checkpoint_path: str = None,
) -> str:
    """Build default output directory: {base_dir}/{model_name}/{timestamp}.

    Args:
        base_dir: Bench-specific base (e.g. "outputs/bench/omnidoc").
        model_id: HuggingFace model ID.
        checkpoint_path: Local checkpoint path.

    Returns:
        Output directory path.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%MZ")
    name = model_name_from_args(model_id, checkpoint_path)
    return os.path.join(base_dir, name, timestamp)


# ---------------------------------------------------------------------------
# Resume logic
# ---------------------------------------------------------------------------
def resolve_resume(args, output_dir: str, parser) -> str:
    """Handle output directory resume from existing metadata.json.

    If output_dir exists with metadata.json, restores --model/--checkpoint
    from it. If --model/--checkpoint were provided, validates they match.

    Args:
        args: Parsed argparse namespace (mutated in place).
        output_dir: Output directory to check.
        parser: ArgumentParser for error reporting.

    Returns:
        The (possibly unchanged) output_dir.
    """
    meta_path = os.path.join(output_dir, "metadata.json")
    if not (os.path.isdir(output_dir) and os.path.exists(meta_path)):
        return output_dir

    with open(meta_path) as f:
        prev_meta = json.load(f)

    if not args.model and not args.checkpoint:
        stored_id = prev_meta.get("model_id", "")
        if not stored_id:
            parser.error("metadata.json has no model_id - specify --model or --checkpoint")
        if stored_id.startswith("local:"):
            args.checkpoint = stored_id.replace("local:", "")
        else:
            args.model = stored_id
    else:
        curr = model_name_from_args(args.model, args.checkpoint)
        prev = prev_meta.get("model_name", "")
        if prev and curr != prev:
            parser.error(f"Model mismatch: '{curr}' vs stored '{prev}'")

    logger.info(f"Resuming into existing dir: {output_dir}")
    return output_dir


# ---------------------------------------------------------------------------
# Sharding
# ---------------------------------------------------------------------------
def apply_shard(items: list, shard_spec: str) -> list:
    """Slice a list according to a shard spec like "1/4".

    Args:
        items: Full list to shard.
        shard_spec: "N/M" where N is 1-indexed shard index, M is total shards.

    Returns:
        Sliced sublist for the requested shard.
    """
    shard_idx, num_shards = [int(x) for x in shard_spec.split("/")]
    if shard_idx < 1 or shard_idx > num_shards:
        raise ValueError(f"Invalid shard spec: {shard_spec} (need 1 <= N <= M)")
    total = len(items)
    shard_size = total // num_shards
    start = (shard_idx - 1) * shard_size
    end = total if shard_idx == num_shards else shard_idx * shard_size
    logger.info(f"Shard {shard_idx}/{num_shards}: items {start+1}-{end} of {total}")
    return items[start:end]


# ---------------------------------------------------------------------------
# Job registration
# ---------------------------------------------------------------------------
def register_bench_job(output_dir: str, bench_type: str, shard: str = None):
    """Register current bench job in the incomplete jobs registry.

    Args:
        output_dir: Output directory path.
        bench_type: Bench name (e.g. "omnidoc", "olmocr", "formatting").
        shard: Shard spec or None.
    """
    key = get_job_key()
    prev = os.environ.get("PREV_JOB_ID")

    with locked_incomplete_jobs() as jobs:
        # Guard on a composite key so different shards of the same output_dir
        # don't trigger duplicate detection, but re-submitting the same shard does.
        guard_key = f"{output_dir}:{shard}" if shard else output_dir
        guard_duplicate_job(jobs, "guard_key", guard_key, key)
        if prev and prev in jobs:
            del jobs[prev]
        jobs[key] = {
            "type": bench_type,
            "output_dir": output_dir,
            "shard": shard or "",
            "guard_key": guard_key,
        }
    logger.info(f"Registered bench job in incomplete registry: {key}")


def remove_bench_job():
    """Remove current job from incomplete registry (marks it complete)."""
    key = get_job_key()
    with locked_incomplete_jobs() as jobs:
        if key in jobs:
            del jobs[key]
            logger.info(f"Removed job from incomplete registry: {key}")


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
def build_metadata(
    args, engine, bench_name: str, generation_params: dict = None,
) -> dict:
    """Build or load metadata for a bench run.

    If metadata.json already exists in the output dir, loads it.
    Otherwise builds fresh metadata from the engine and args.

    Args:
        args: Parsed argparse namespace (needs .model, .checkpoint, .output).
        engine: InferenceEngine instance.
        bench_name: Bench identifier (e.g. "omnidoc").
        generation_params: Dict of generation settings to record.

    Returns:
        Metadata dict.
    """
    output_dir = args.output
    meta_path = os.path.join(output_dir, "metadata.json")

    if os.path.exists(meta_path):
        with open(meta_path) as f:
            metadata = json.load(f)
    else:
        metadata = {
            "model_id": args.model or f"local:{args.checkpoint}",
            "model_name": engine.model_name,
            "commit_hash": engine.commit_hash,
            "benchmark": bench_name,
            "generation_params": generation_params or {},
            "slurm_jobs": [],
        }

    # Append current SLURM job info
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id:
        jobs = metadata.setdefault("slurm_jobs", [])
        jobs.append({
            "job_id": slurm_job_id,
            "node": os.environ.get("SLURM_NODELIST", "unknown"),
            "started": datetime.now(timezone.utc).isoformat(),
        })

    return metadata


def save_metadata(output_dir: str, metadata: dict):
    """Write metadata.json to the output directory.

    Args:
        output_dir: Output directory path.
        metadata: Metadata dict to save.
    """
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


def save_bench_output(output_dir: str, results: dict, metadata: dict = None):
    """Write results.json (and optionally metadata.json) to output directory.

    Args:
        output_dir: Output directory path.
        results: Results dict to save.
        metadata: If provided, also writes metadata.json.
    """
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    if metadata:
        save_metadata(output_dir, metadata)
    logger.info(f"Results saved to {output_dir}")


# ---------------------------------------------------------------------------
# Common CLI args
# ---------------------------------------------------------------------------
def add_common_bench_args(parser, max_new_tokens: int = 2048):
    """Add CLI args shared across all bench scripts.

    Adds --model/--checkpoint (mutually exclusive), --output, --shard,
    --max_new_tokens, and --score_only.

    Args:
        parser: ArgumentParser to add args to.
        max_new_tokens: Default for --max_new_tokens.
    """
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--model", type=str, help="HuggingFace model ID")
    group.add_argument("--checkpoint", type=str, help="Path to LoRA checkpoint")

    parser.add_argument("--output", type=str, default=None,
                        help="Output directory. If it exists with metadata.json, resumes automatically.")
    parser.add_argument("--shard", type=str, default=None, help="Shard spec N/M (e.g. 1/4)")
    parser.add_argument("--max_new_tokens", type=int, default=max_new_tokens)
    parser.add_argument("--score_only", action="store_true",
                        help="Skip generation, only score existing markdowns")


def validate_bench_args(args, parser, default_output_base: str) -> tuple[str, str]:
    """Validate common bench args and set up output directory.

    Handles score_only validation, resume detection, output dir derivation,
    and markdowns dir creation.

    Args:
        args: Parsed argparse namespace (mutated: sets args.output if not provided).
        parser: ArgumentParser for error reporting.
        default_output_base: Base dir for auto-derived output (e.g. "outputs/bench/omnidoc").

    Returns:
        Tuple of (output_dir, markdowns_dir).
    """
    if args.score_only and not args.output:
        parser.error("--score_only requires --output")

    if args.output:
        resolve_resume(args, args.output, parser)
        if not args.score_only and not args.model and not args.checkpoint:
            parser.error("--model or --checkpoint required for new runs")
    else:
        if not args.model and not args.checkpoint:
            parser.error("--model or --checkpoint required for new runs")
        args.output = derive_output_dir(default_output_base, args.model, args.checkpoint)

    output_dir = args.output
    markdowns_dir = os.path.join(output_dir, "markdowns")
    os.makedirs(markdowns_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    return output_dir, markdowns_dir


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------
def cleanup_temp_files(markdowns_dir: str):
    """Remove stale .tmp files from a markdowns directory.

    Args:
        markdowns_dir: Path to the markdowns directory.
    """
    for tmp_file in glob.glob(os.path.join(markdowns_dir, "*.tmp")):
        os.remove(tmp_file)
