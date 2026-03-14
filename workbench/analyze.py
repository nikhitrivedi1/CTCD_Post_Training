#!/usr/bin/env python
"""
Workbench entry point.

Usage:
    # Run specific analyses
    python workbench/analyze.py --model M --reference R --task mnli --test probe cka

    # Run all analyses
    python workbench/analyze.py --model M --reference R --task mnli --test all

    # Probe a single model (no comparison)
    python workbench/analyze.py --model M --task mnli --test probe

    # Named run (reusable across days)
    python workbench/analyze.py --model M --reference R --task mnli --run_name v1 --test probe cka

    # Timestamped filenames (never collide)
    python workbench/analyze.py --model M --reference R --task mnli --run_name v1 --timestamp --test all

    # Overwrite existing results
    python workbench/analyze.py --model M --reference R --task mnli --run_name v1 --force --test probe
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from utils.logging import setup_file_logging


def get_default_output(
    command: str, task: str, model_name: str = None,
    run_name: str = None, timestamp: bool = False, force: bool = False,
) -> str:
    """
    Generate output path organized by model and run.

    Naming semantics:
    - --run_name X                    -> <model>/X/<command>.json
    - --run_name X --timestamp        -> <model>/X_20260206Z/<command>_20260206-1712Z.json
    - (no --run_name)                 -> <model>/20260206Z/<command>.json
    - (no --run_name) --timestamp     -> <model>/20260206Z/<command>_20260206-1712Z.json
    - --force overwrites existing output file on collision.
    - Without --timestamp or --force, collision = error.

    Args:
        command: Analysis command (probe, cka, attention-entropy, disagreement)
        task: Task name (mnli, qqp, etc.)
        model_name: Optional model path for naming
        run_name: Optional run name for grouping results
        timestamp: Whether to append time-level timestamp to output filename
        force: Whether to overwrite existing output file

    Returns:
        Output path like: workbench/results/<model>/<run>/<command>.json
    """
    base_dir = Path("workbench/results")

    if model_name:
        short_name = Path(model_name).name
        model_dir = base_dir / short_name
    else:
        model_dir = base_dir / task

    if run_name:
        if timestamp:
            day_ts = datetime.now(timezone.utc).strftime("%Y%m%dZ")
            run_dir = f"{run_name}_{day_ts}"
        else:
            run_dir = run_name
    else:
        run_dir = datetime.now(timezone.utc).strftime("%Y%m%dZ")

    output_dir = model_dir / run_dir

    if timestamp:
        time_ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%MZ")
        output_path = output_dir / f"{command}_{time_ts}.json"
    else:
        output_path = output_dir / f"{command}.json"
        if output_path.exists() and not force:
            logger.error(f"Output already exists: {output_path}")
            logger.error("Use --force to overwrite or --timestamp for unique filenames.")
            sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_path)


def setup_logging(output_path: str):
    """
    Configure logging to both console and file.

    Uses TeeWriter to capture stdout/stderr (tqdm progress bars) while
    handling \\r so progress bars don't spam the log file.

    Args:
        output_path: Path to JSON output file (log uses same base name)
    """
    log_path = Path(output_path).with_suffix(".log")

    # Use shared logging utility
    setup_file_logging(log_path)

    # Clear any existing handlers and set up console handler
    root_logger = logging.getLogger()
    root_logger.handlers = []

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)

    return log_path


# Basic logging for module import (overridden in commands)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def cmd_probe(args):
    """Run linear probing analysis."""
    from workbench.probing import run_probing, compare_probes

    output_path = args.output or get_default_output("probe", args.task, args.model, args.run_name, args.timestamp, args.force)
    log_path = setup_logging(output_path)
    max_samples = args.max_samples  # default: full dataset

    results = {}

    logger.info("=" * 60)
    logger.info(f"PROBING: {args.model}")
    logger.info("=" * 60)
    results["model"] = {
        "model": args.model,
        "probes": run_probing(args.model, args.task, args.device, max_samples, args.seed)
    }

    if args.reference:
        logger.info("=" * 60)
        logger.info(f"PROBING: {args.reference}")
        logger.info("=" * 60)
        results["reference"] = {
            "model": args.reference,
            "probes": run_probing(args.reference, args.task, args.device, max_samples, args.seed)
        }
        results["comparison"] = compare_probes(
            reference_probes=results["reference"]["probes"],
            target_probes=results["model"]["probes"]
        )

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")
    logger.info(f"Log saved to {log_path}")

    print_probe_summary(results)


def cmd_cka(args):
    """Run CKA analysis."""
    from workbench.cka import run_cka

    if not args.reference:
        logger.error("CKA requires --reference model")
        sys.exit(1)

    output_path = args.output or get_default_output("cka", args.task, args.model, args.run_name, args.timestamp, args.force)
    log_path = setup_logging(output_path)
    max_samples = args.max_samples or 1000  # CKA is O(N^2)

    logger.info("=" * 60)
    logger.info("CKA ANALYSIS")
    logger.info("=" * 60)

    results = run_cka(
        args.model, args.reference, args.task,
        args.device, max_samples
    )

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")
    logger.info(f"Log saved to {log_path}")

    print_cka_summary(results)


def print_probe_summary(results: dict):
    """Print probing results summary."""
    print("\n" + "=" * 60)
    print("PROBING RESULTS")
    print("=" * 60)

    model_probes = results["model"]["probes"]
    print(f"\nModel ({results['model']['model']}):")
    for layer, acc in model_probes.items():
        print(f"  {layer}: {acc:.2%}")

    if "reference" in results:
        ref_probes = results["reference"]["probes"]
        print(f"\nReference ({results['reference']['model']}):")
        for layer, acc in ref_probes.items():
            print(f"  {layer}: {acc:.2%}")

        comp = results["comparison"]
        print(f"\nComparison:")
        print(f"  Reference peak: {comp['reference_peak']['layer']} @ {comp['reference_peak']['accuracy']:.2%}")
        print(f"  Model peak: {comp['target_peak']['layer']} @ {comp['target_peak']['accuracy']:.2%}")
        print(f"  Retention: {comp['retention']:.1%}")
        if comp['target_saturation_layer']:
            print(f"  Model saturates at: {comp['target_saturation_layer']}")


def print_cka_summary(results: dict):
    """Print CKA results summary."""
    print("\n" + "=" * 60)
    print("CKA ALIGNMENT")
    print("=" * 60)

    print("\nBest reference layer match for each model layer:")
    for m_layer, alignment in results["best_alignment"].items():
        print(f"  {m_layer} -> {alignment['reference_layer']} (CKA: {alignment['cka_score']:.3f})")


def cmd_attention_entropy(args):
    """Run attention entropy analysis."""
    from workbench.attention import run_attention_analysis

    if not args.reference:
        logger.error("Attention entropy analysis requires --reference model")
        sys.exit(1)

    output_path = args.output or get_default_output("attention-entropy", args.task, args.model, args.run_name, args.timestamp, args.force)
    log_path = setup_logging(output_path)
    max_samples = args.max_samples or 1000

    logger.info("=" * 60)
    logger.info("ATTENTION ENTROPY ANALYSIS")
    logger.info("=" * 60)

    results = run_attention_analysis(
        args.model, args.reference, args.task,
        args.device, max_samples
    )

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")
    logger.info(f"Log saved to {log_path}")

    print_attention_summary(results)


def cmd_disagreement(args):
    """Run prediction disagreement analysis."""
    from workbench.disagreement import run_disagreement_analysis

    if not args.reference:
        logger.error("Disagreement analysis requires --reference model")
        sys.exit(1)

    output_path = args.output or get_default_output("disagreement", args.task, args.model, args.run_name, args.timestamp, args.force)
    log_path = setup_logging(output_path)
    max_samples = args.max_samples  # default: full dataset

    logger.info("=" * 60)
    logger.info("PREDICTION DISAGREEMENT ANALYSIS")
    logger.info("=" * 60)

    results = run_disagreement_analysis(
        args.model, args.reference, args.task,
        args.device, max_samples
    )

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")
    logger.info(f"Log saved to {log_path}")

    print_disagreement_summary(results)


def print_attention_summary(results: dict):
    """Print attention analysis summary."""
    print("\n" + "=" * 60)
    print("ATTENTION ENTROPY")
    print("=" * 60)

    print(f"\nModel ({results['model']['model']}):")
    print(f"  Mean entropy: {results['model']['mean_entropy']:.3f}")
    for layer, stats in results['model']['layer_stats'].items():
        print(f"  {layer}: {stats['mean_entropy']:.3f}")

    print(f"\nReference ({results['reference']['model']}):")
    print(f"  Mean entropy: {results['reference']['mean_entropy']:.3f}")
    for layer, stats in results['reference']['layer_stats'].items():
        print(f"  {layer}: {stats['mean_entropy']:.3f}")

    print(f"\nComparison:")
    print(f"  Entropy ratio (M/R): {results['comparison']['entropy_ratio']:.3f}")
    print(f"\n  Layer mapping:")
    for m_layer, comp in results['comparison']['layer_mapping'].items():
        ratio = comp['ratio']
        indicator = "\u2191" if ratio > 1.05 else "\u2193" if ratio < 0.95 else "\u2248"
        print(f"    {m_layer} -> {comp['reference_layer']}: {ratio:.2f} {indicator}")


def print_disagreement_summary(results: dict):
    """Print disagreement analysis summary."""
    print("\n" + "=" * 60)
    print("PREDICTION DISAGREEMENT ANALYSIS")
    print("=" * 60)

    dis = results["disagreement"]
    print(f"\nSamples analyzed: {dis['total_samples']}")

    print(f"\nAccuracy:")
    print(f"  Model: {dis['accuracy']['model']:.2%}")
    print(f"  Reference: {dis['accuracy']['reference']:.2%}")

    print(f"\nAgreement/Disagreement:")
    print(f"  Agree: {dis['agreement']['agree']} ({dis['agreement']['agree_pct']:.1%})")
    print(f"    Both correct: {dis['agreement']['agree_and_correct']} ({dis['agreement']['agree_and_correct_pct']:.1%})")
    print(f"    Both wrong: {dis['agreement']['agree_and_wrong']} ({dis['agreement']['agree_and_wrong_pct']:.1%})")
    print(f"  Disagree: {dis['disagreement']['disagree']} ({dis['disagreement']['disagree_pct']:.1%})")
    print(f"    Model only correct: {dis['disagreement']['model_only_correct']} ({dis['disagreement']['model_only_correct_pct']:.1%})")
    print(f"    Reference only correct: {dis['disagreement']['reference_only_correct']} ({dis['disagreement']['reference_only_correct_pct']:.1%})")

    print(f"\nConfidence (mean):")
    print(f"  Model: {dis['confidence']['model_mean']:.3f}")
    print(f"  Reference: {dis['confidence']['reference_mean']:.3f}")
    print(f"  Model when wrong: {dis['confidence']['model_when_wrong']:.3f}")
    print(f"  Reference when wrong: {dis['confidence']['reference_when_wrong']:.3f}")

    print(f"\nPer-class disagreement rate:")
    for class_name, stats in dis.get("per_class_disagreement", {}).items():
        print(f"  {class_name}: {stats['disagree_pct']:.1%} (M:{stats['model_acc']:.1%} R:{stats['reference_acc']:.1%})")

    # Per-class accuracy breakdown
    print(f"\nPer-class accuracy breakdown:")
    for model_key in ["model", "reference"]:
        model_data = results[model_key]["per_class"]
        print(f"\n  {model_key.capitalize()} ({model_data['model']}):")
        for class_name, stats in model_data["per_class"].items():
            print(f"    {class_name}: {stats['accuracy']:.1%} ({stats['correct']}/{stats['count']}) F1:{stats['f1']:.3f}")


ANALYSES = ["probe", "cka", "attention-entropy", "disagreement"]


def main():
    parser = argparse.ArgumentParser(
        description="Workbench for model analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Shared model/task args
    parser.add_argument("--model", type=str, required=True,
                        help="Model to analyze (HuggingFace name or checkpoint path)")
    parser.add_argument("--reference", type=str, default=None,
                        help="Reference model for comparison (optional for probe, required for others)")
    parser.add_argument("--task", type=str, default="mnli",
                        choices=["mnli", "qqp", "sst2", "cola"])
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples (default: command-specific)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output JSON file (only valid with single analysis)")

    # Run management
    parser.add_argument("--run_name", type=str, default=None,
                        help="Run name for grouping results (default: auto-timestamp)")
    parser.add_argument("--timestamp", action="store_true",
                        help="Add day-stamp to run dir, time-stamp to filenames")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing output files")

    # Analyses to run (required, one or more)
    parser.add_argument("--test", nargs="+", required=True,
                        choices=ANALYSES + ["all"],
                        metavar="ANALYSIS",
                        help="Analyses to run: probe, cka, attention-entropy, disagreement, or 'all'")

    args = parser.parse_args()

    # Expand "all" to full list
    if "all" in args.test:
        args.test = ANALYSES

    # --output only valid with single analysis
    if args.output and len(args.test) > 1:
        parser.error("--output can only be used with a single analysis")

    # Dispatch
    dispatch = {
        "probe": cmd_probe,
        "cka": cmd_cka,
        "attention-entropy": cmd_attention_entropy,
        "disagreement": cmd_disagreement,
    }

    for analysis in args.test:
        dispatch[analysis](args)


if __name__ == "__main__":
    main()
