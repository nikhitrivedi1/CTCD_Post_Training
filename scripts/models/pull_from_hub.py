#!/usr/bin/env python3
"""Download a model from HuggingFace Hub to the outputs directory.

Usage:
    # Download a model (repo name becomes directory name)
    python scripts/models/pull_from_hub.py org/mnli_01_kd_frozen-base_20260203-2235Z

    # Download to a custom directory name
    python scripts/models/pull_from_hub.py org/repo-name --name my-model

    # Delete existing and re-download
    python scripts/models/pull_from_hub.py org/repo-name --force

    # Dry run (show what would be downloaded)
    python scripts/models/pull_from_hub.py org/repo-name --dry-run

    # Download to a custom base directory
    python scripts/models/pull_from_hub.py org/repo-name --output-dir checkpoints
"""

import argparse
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, list_repo_files


def pull_model(repo_id, output_dir="outputs", name=None, force=False, dry_run=False):
    """Download model files from HuggingFace Hub.

    Args:
        repo_id: Full repo ID (org/name).
        output_dir: Base directory for downloads.
        name: Custom local directory name (default: repo name).
        force: Whether to delete existing directory and re-download.
        dry_run: If True, only print what would be downloaded.
    """
    api = HfApi()

    # Verify repo exists
    try:
        repo_info = api.repo_info(repo_id)
    except Exception as e:
        print(f"Error: Could not find repo '{repo_id}': {e}")
        sys.exit(1)

    # Determine local directory
    local_name = name or repo_id.split("/")[-1]
    local_dir = Path(output_dir) / local_name

    # Collision check
    if local_dir.exists():
        if not force:
            print(f"Error: Directory already exists: {local_dir}")
            print("Use --force to delete and re-download, or --name to choose a different name.")
            sys.exit(1)
        print(f"Removing existing directory: {local_dir}")
        shutil.rmtree(local_dir)

    # List remote files
    try:
        remote_files = list_repo_files(repo_id)
    except Exception as e:
        print(f"Error listing repo files: {e}")
        sys.exit(1)

    print(f"Repo:      {repo_id}")
    print(f"Local dir: {local_dir}")
    print(f"Files:     {len(remote_files)}")
    for fname in sorted(remote_files):
        print(f"  {fname}")

    if dry_run:
        print("\nDry run — nothing downloaded.")
        return

    # Download each file
    local_dir.mkdir(parents=True, exist_ok=True)

    for fname in sorted(remote_files):
        print(f"Downloading {fname}...", end=" ", flush=True)
        hf_hub_download(
            repo_id=repo_id,
            filename=fname,
            local_dir=str(local_dir),
        )
        print("done")

    print(f"\nDownloaded to {local_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download models from HuggingFace Hub")
    parser.add_argument(
        "repo_id",
        help="HuggingFace repo ID (org/name)",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Local directory name (default: repo name from repo_id)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Base directory for downloads (default: outputs)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing directory and re-download",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without downloading",
    )

    args = parser.parse_args()

    pull_model(
        repo_id=args.repo_id,
        output_dir=args.output_dir,
        name=args.name,
        force=args.force,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
