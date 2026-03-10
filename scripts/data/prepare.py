"""Unified data preparation CLI.

Single entry point for preparing any dataset in the Forge pipeline.
Dispatches to the appropriate handler based on --dataset.

Usage:
    python scripts/data/prepare.py --dataset omnidocbench
    python scripts/data/prepare.py --dataset formatting --num_pages 200 --augment
    python scripts/data/prepare.py --dataset synthmix --action validate
    python scripts/data/prepare.py --dataset all
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root for src imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default data directories per dataset
DEFAULT_DIRS = {
    "omnidocbench": "data/ocr/omnidocbench",
    "synthmix": "data/ocr/synthmix",
    "formatting": "data/ocr/formatting",
    "olmocr-mix": "data/ocr/olmocr-mix-full",
    "europarl": "",  # Downloaded from HF Hub
    "mnli": "",
    "qqp": "",
    "sst2": "",
    "cola": "",
}


def run_dataset(name: str, action: str, data_dir: str | None, extra_kwargs: dict) -> bool:
    """Run prepare/validate for a single dataset.

    Args:
        name: Dataset name (key in DATASET_REGISTRY).
        action: One of "prepare", "validate", "all".
        data_dir: Override data directory.
        extra_kwargs: Additional kwargs for prepare().

    Returns:
        True if all actions succeeded.
    """
    from src.data.handlers import get_handler

    handler = get_handler(name)
    target_dir = data_dir or DEFAULT_DIRS.get(name, f"data/{name}")

    ok = True

    if action in ("prepare", "all"):
        logger.info(f"--- Preparing: {name} -> {target_dir}")
        try:
            handler.prepare(target_dir, **extra_kwargs)
        except Exception as e:
            logger.error(f"Prepare failed for {name}: {e}")
            ok = False

    if action in ("validate", "all"):
        logger.info(f"--- Validating: {name}")
        result = handler.validate(target_dir)
        status = "OK" if result["ok"] else "FAILED"
        logger.info(f"  {status}: {result['details']}")
        if not result["ok"]:
            ok = False

    return ok


def main():
    parser = argparse.ArgumentParser(
        description="Prepare datasets for Forge training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --dataset omnidocbench                    # download + convert + validate
  %(prog)s --dataset formatting --num_pages 200      # generate 200 pages/category
  %(prog)s --dataset synthmix --action validate      # just check integrity
  %(prog)s --dataset all                             # prepare everything
  %(prog)s --dataset formatting --data_dir /tmp/fmt  # custom output directory
        """,
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Dataset name or 'all'. Options: " + ", ".join(sorted(DEFAULT_DIRS.keys())),
    )
    parser.add_argument(
        "--action", type=str, default="all", choices=["prepare", "validate", "all"],
        help="Action to perform (default: all)",
    )
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Override default data directory",
    )

    # Formatting-specific args
    parser.add_argument("--num_pages", type=int, default=200, help="Pages per category (formatting)")
    parser.add_argument("--categories", type=str, default="all", help="Categories to generate (formatting)")
    parser.add_argument("--augment", action="store_true", help="Apply augmentations (formatting)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # olmOCR-mix specific args
    parser.add_argument("--subset", type=str, default=None, help="Subset name (olmocr-mix)")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples (olmocr-mix)")

    # Synthmix specific args
    parser.add_argument("--dpi", type=int, default=150, help="Render DPI (synthmix)")
    parser.add_argument("--workers", type=int, default=None, help="Parallel workers (synthmix)")

    args = parser.parse_args()

    # Build extra kwargs for prepare() based on dataset type
    extra_kwargs = {"seed": args.seed}
    if args.dataset == "formatting":
        extra_kwargs.update(
            num_pages=args.num_pages,
            categories=args.categories,
            augment=args.augment,
        )
    elif args.dataset == "olmocr-mix":
        extra_kwargs.update(subset=args.subset, max_samples=args.max_samples)
    elif args.dataset == "synthmix":
        extra_kwargs.update(dpi=args.dpi, workers=args.workers)

    if args.dataset == "all":
        # Prepare all datasets that need local prep
        datasets_to_prep = ["omnidocbench", "synthmix", "formatting", "olmocr-mix"]
        all_ok = True
        for name in datasets_to_prep:
            logger.info(f"\n{'='*60}")
            logger.info(f"Dataset: {name}")
            logger.info(f"{'='*60}")
            if not run_dataset(name, args.action, None, {}):
                all_ok = False
        if not all_ok:
            logger.error("Some datasets had issues")
            sys.exit(1)
    else:
        if not run_dataset(args.dataset, args.action, args.data_dir, extra_kwargs):
            sys.exit(1)

    logger.info("Done.")


if __name__ == "__main__":
    main()
