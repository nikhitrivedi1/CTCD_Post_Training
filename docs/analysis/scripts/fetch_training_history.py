#!/usr/bin/env python3
"""Fetch training history (loss curves) from W&B for analysis.

Usage:
    python fetch_training_history.py --runs mnli_ablations --output ../data/
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import wandb


def fetch_run_history(
    entity: str,
    project: str,
    run_name_pattern: str | None = None,
    metrics: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Fetch training history for runs matching pattern.

    Args:
        entity: W&B entity
        project: W&B project name
        run_name_pattern: Filter runs by name pattern (substring match)
        metrics: List of metrics to fetch (default: all logged metrics)

    Returns:
        Dictionary mapping run names to DataFrames of history
    """
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}", filters={"state": "finished"})

    histories = {}
    for run in runs:
        if run_name_pattern and run_name_pattern not in run.name:
            continue

        print(f"Fetching history for: {run.name}")

        # Get history as DataFrame
        history_df = run.history(samples=10000)  # Get all logged steps

        if metrics:
            # Filter to requested metrics (plus _step)
            available = [m for m in metrics if m in history_df.columns]
            history_df = history_df[["_step"] + available]

        histories[run.name] = history_df

    return histories


def save_histories(
    histories: dict[str, pd.DataFrame],
    output_dir: Path,
    prefix: str = "history",
) -> None:
    """Save histories to JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to serializable format
    data = {}
    for run_name, df in histories.items():
        data[run_name] = df.to_dict(orient="records")

    json_path = output_dir / f"{prefix}.json"
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Saved histories to {json_path}")


def analyze_loss_dynamics(histories: dict[str, pd.DataFrame]) -> None:
    """Print analysis of loss dynamics."""
    print("\n" + "=" * 70)
    print("TRAINING DYNAMICS ANALYSIS")
    print("=" * 70)

    for run_name, df in sorted(histories.items()):
        print(f"\n--- {run_name} ---")

        # Check available columns
        loss_cols = [c for c in df.columns if "loss" in c.lower()]
        print(f"  Available loss metrics: {loss_cols}")

        for col in loss_cols:
            if col in df.columns and df[col].notna().any():
                vals = df[col].dropna()
                if len(vals) > 0:
                    print(f"  {col}:")
                    print(f"    Start: {vals.iloc[0]:.4f}")
                    print(f"    End:   {vals.iloc[-1]:.4f}")
                    print(f"    Min:   {vals.min():.4f}")
                    print(f"    Max:   {vals.max():.4f}")


def main():
    parser = argparse.ArgumentParser(description="Fetch W&B training history")
    parser.add_argument(
        "--entity",
        default="karthik-ravi-1537-personal",
        help="W&B entity",
    )
    parser.add_argument(
        "--project",
        default="Knowledge Distillation",
        help="W&B project name",
    )
    parser.add_argument(
        "--runs",
        default="ablations",
        help="Run name pattern to match",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Specific metrics to fetch",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "data",
        help="Output directory",
    )
    parser.add_argument(
        "--prefix",
        default="training_history",
        help="Output filename prefix",
    )

    args = parser.parse_args()

    print(f"Fetching history from {args.entity}/{args.project}...")
    print(f"  Pattern: {args.runs}")

    histories = fetch_run_history(
        entity=args.entity,
        project=args.project,
        run_name_pattern=args.runs,
        metrics=args.metrics,
    )

    if not histories:
        print("No runs found!")
        return

    print(f"\nFetched history for {len(histories)} runs")

    analyze_loss_dynamics(histories)
    save_histories(histories, args.output, args.prefix)


if __name__ == "__main__":
    main()
