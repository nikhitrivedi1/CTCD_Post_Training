#!/usr/bin/env python3
"""Fetch experiment data from Weights & Biases.

Usage:
    python fetch_wandb.py --output ../data/
    python fetch_wandb.py --tags ablation --output ../data/
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import wandb


def fetch_runs(
    entity: str,
    project: str,
    tags: list[str] | None = None,
    state: str | None = "finished",
) -> list[dict]:
    """
    Fetch runs from W&B project.

    Args:
        entity: W&B entity (username or team)
        project: W&B project name
        tags: Filter by tags (optional)
        state: Filter by state (finished, running, crashed, etc.)

    Returns:
        List of run dictionaries with config, summary, and metadata
    """
    api = wandb.Api()

    filters = {}
    if state:
        filters["state"] = state
    if tags:
        filters["tags"] = {"$in": tags}

    runs = api.runs(f"{entity}/{project}", filters=filters)

    results = []
    for run in runs:
        run_data = {
            # Metadata
            "id": run.id,
            "name": run.name,
            "state": run.state,
            "tags": run.tags,
            "created_at": run.created_at,
            "duration_seconds": run.summary.get("_runtime"),
            "url": run.url,
            # Config
            "task": run.config.get("data", {}).get("task"),
            "teacher": run.config.get("model", {}).get("teacher"),
            "student": run.config.get("model", {}).get("student"),
            "alpha": run.config.get("distillation", {}).get("alpha"),
            "temperature": run.config.get("distillation", {}).get("temperature"),
            "freeze_teacher": run.config.get("distillation", {}).get("freeze_teacher"),
            "loss_type": run.config.get("distillation", {}).get("loss_type"),
            "batch_size": run.config.get("training", {}).get("batch_size"),
            "learning_rate": run.config.get("training", {}).get("learning_rate"),
            "num_epochs": run.config.get("training", {}).get("num_epochs"),
            # Summary metrics
            "eval_loss": run.summary.get("eval_loss"),
            "eval_accuracy": run.summary.get("eval_accuracy"),
            "eval_f1": run.summary.get("eval_f1"),
            "eval_mcc": run.summary.get("eval_mcc"),
            "bleu": run.summary.get("bleu"),
            "comet": run.summary.get("comet"),
            "train_loss": run.summary.get("train_loss"),
        }
        results.append(run_data)

    return results


def save_runs(runs: list[dict], output_dir: Path, prefix: str = "runs") -> Path:
    """
    Save runs to CSV and JSON.

    Args:
        runs: List of run dictionaries
        output_dir: Output directory
        prefix: Filename prefix

    Returns:
        Path to CSV file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d")

    # Save CSV
    df = pd.DataFrame(runs)
    csv_path = output_dir / f"{prefix}_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(runs)} runs to {csv_path}")

    # Save JSON (full data)
    json_path = output_dir / f"{prefix}_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(runs, f, indent=2, default=str)
    print(f"Saved JSON to {json_path}")

    return csv_path


def print_summary(runs: list[dict]) -> None:
    """Print summary of fetched runs."""
    df = pd.DataFrame(runs)

    print("\n" + "=" * 60)
    print("RUNS SUMMARY")
    print("=" * 60)

    print(f"\nTotal runs: {len(df)}")

    if "task" in df.columns:
        print("\nBy task:")
        print(df["task"].value_counts().to_string())

    if "state" in df.columns:
        print("\nBy state:")
        print(df["state"].value_counts().to_string())

    if "tags" in df.columns:
        all_tags = [tag for tags in df["tags"] if tags for tag in tags]
        if all_tags:
            print("\nTop tags:")
            tag_counts = pd.Series(all_tags).value_counts().head(10)
            print(tag_counts.to_string())

    # Best runs by task
    print("\n" + "-" * 60)
    print("BEST RUNS BY TASK")
    print("-" * 60)

    for task in df["task"].dropna().unique():
        task_df = df[df["task"] == task]

        # Determine primary metric
        if task in ["mnli", "sst2"]:
            metric = "eval_accuracy"
        elif task == "qqp":
            metric = "eval_f1"
        elif task == "cola":
            metric = "eval_mcc"
        elif task in ["europarl", "wmt14"]:
            metric = "bleu"
        else:
            metric = "eval_accuracy"

        if metric in task_df.columns and task_df[metric].notna().any():
            best = task_df.loc[task_df[metric].idxmax()]
            print(f"\n{task.upper()} (best {metric}):")
            print(f"  Run: {best['name']}")
            print(f"  {metric}: {best[metric]:.4f}")
            print(f"  alpha: {best.get('alpha')}, freeze: {best.get('freeze_teacher')}")


def main():
    parser = argparse.ArgumentParser(description="Fetch W&B experiment data")
    parser.add_argument(
        "--entity",
        default="karthik-ravi-1537-personal",
        help="W&B entity",
    )
    parser.add_argument(
        "--project",
        default=os.environ.get("WANDB_PROJECT", "Forge-Default"),
        help="W&B project name (default: WANDB_PROJECT env var or Forge-Default)",
    )
    parser.add_argument(
        "--tags",
        nargs="+",
        help="Filter by tags",
    )
    parser.add_argument(
        "--state",
        default="finished",
        help="Filter by state (finished, running, crashed, all)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "data",
        help="Output directory",
    )
    parser.add_argument(
        "--prefix",
        default="runs",
        help="Output filename prefix",
    )

    args = parser.parse_args()

    state = None if args.state == "all" else args.state

    print(f"Fetching runs from {args.entity}/{args.project}...")
    if args.tags:
        print(f"  Tags: {args.tags}")
    if state:
        print(f"  State: {state}")

    runs = fetch_runs(
        entity=args.entity,
        project=args.project,
        tags=args.tags,
        state=state,
    )

    if not runs:
        print("No runs found!")
        return

    print_summary(runs)
    save_runs(runs, args.output, args.prefix)


if __name__ == "__main__":
    main()
