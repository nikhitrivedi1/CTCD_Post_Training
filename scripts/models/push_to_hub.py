#!/usr/bin/env python3
"""Push a trained model or checkpoint to HuggingFace Hub.

Usage:
    # Push a final model
    python scripts/models/push_to_hub.py outputs/mnli_01_kd_frozen-base_20260203-2235Z

    # Push a specific checkpoint
    python scripts/models/push_to_hub.py outputs/.../checkpoint-4000

    # Push to a specific org/user
    python scripts/models/push_to_hub.py outputs/... --org my-org

    # Push as public
    python scripts/models/push_to_hub.py outputs/... --public

    # Dry run (show what would be uploaded)
    python scripts/models/push_to_hub.py outputs/... --dry-run

    # Repair missing base/model_ref.json (backfill from config.yaml)
    python scripts/models/push_to_hub.py outputs/... --repair

    # List models on the Hub
    python scripts/models/push_to_hub.py --list

    # Delete a model from the Hub
    python scripts/models/push_to_hub.py --delete org/repo-name
"""

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi, list_models

# Files that make up a usable HF model (full weights)
MODEL_FILES = {
    "model.safetensors",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
}

# LoRA adapter files (from use_lora training)
ADAPTER_FILES = {
    "adapter_model.safetensors",
    "adapter_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
}

# Extra provenance files from our training pipeline
PROVENANCE_FILES = {
    "config.yaml",
    "run_metadata.json",
    "generation_config.json",
    "processor_config.json",
}

# Checkpoint-specific files needed for resume
RESUME_FILES = {
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
    "training_args.bin",
    "rng_state.pth",
}


def get_default_org():
    """Get the HF username of the logged-in user."""
    api = HfApi()
    user_info = api.whoami()
    return user_info["name"]


def derive_repo_name(model_dir):
    """Derive a Hub repo name from the output directory path.

    Args:
        model_dir: Path to the model directory.

    Returns:
        Repo name string. For checkpoints, appends the checkpoint suffix.
    """
    model_dir = Path(model_dir).resolve()
    name = model_dir.name

    # If this is a sub-directory (checkpoint or teacher), include parent run name
    if name.startswith("checkpoint-") or name == "teacher":
        run_name = model_dir.parent.name
        return f"{run_name}_{name}"

    return name


def validate_model_dir(model_dir):
    """Check that the directory contains the minimum required model files.

    Accepts either full model files (model.safetensors + config.json)
    or LoRA adapter files (adapter_model.safetensors + adapter_config.json).

    Args:
        model_dir: Path to the model directory.

    Returns:
        Tuple of (existing files set, bool is_adapter).
    """
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        print(f"Error: {model_dir} is not a directory")
        sys.exit(1)

    existing = {f.name for f in model_dir.iterdir() if f.is_file()}

    # Check for LoRA adapter files first
    adapter_required = {"adapter_model.safetensors", "adapter_config.json"}
    if adapter_required.issubset(existing):
        return existing, True

    # Fall back to full model files
    model_required = {"model.safetensors", "config.json"}
    missing = model_required - existing
    if missing:
        print(f"Error: Missing required files: {missing}")
        print("Expected either model.safetensors + config.json (full model)")
        print("or adapter_model.safetensors + adapter_config.json (LoRA adapter)")
        sys.exit(1)

    return existing, False


def _ensure_base_ref(model_dir):
    """Create base/model_ref.json if missing, reconstructing from config.yaml.

    For completed runs that predate model_ref.json tracking, this backfills
    the reference from available metadata. No-op if the ref already exists
    or if base/ contains full weights (from --adapter).

    Args:
        model_dir: Path to the model/checkpoint directory.

    Returns:
        True if a ref was created, False otherwise.
    """
    model_dir = Path(model_dir)
    base_dir = model_dir / "base"
    ref_path = base_dir / "model_ref.json"

    # Already has ref or full weights in base/
    if ref_path.exists() or (base_dir / "config.json").exists():
        return False

    # Find config.yaml (in run root, or parent for checkpoint dirs)
    config_path = model_dir / "config.yaml"
    if not config_path.exists():
        config_path = model_dir.parent / "config.yaml"
    if not config_path.exists():
        return False

    try:
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except Exception:
        return False

    model_name = config.get("model", {}).get("name")
    if not model_name:
        return False

    # Try to get commit hash from HF config.json
    revision = None
    hf_config = model_dir / "config.json"
    if hf_config.exists():
        try:
            revision = json.loads(hf_config.read_text()).get("_commit_hash")
        except Exception:
            pass

    base_dir.mkdir(exist_ok=True)
    ref = {"model_name_or_path": model_name}
    if revision:
        ref["revision"] = revision
    with open(ref_path, 'w') as f:
        json.dump(ref, f, indent=2)
    print(f"Backfilled {ref_path.relative_to(model_dir)} -> {model_name}")
    return True


def push_model(model_dir, org, private=True, include_resume=False, dry_run=False,
               repo_name=None, prefix=None):
    """Push model files to HuggingFace Hub.

    Args:
        model_dir: Path to the model/checkpoint directory.
        org: HuggingFace username or organization.
        private: Whether to create a private repo.
        include_resume: Whether to include optimizer/scheduler state for full resume.
        dry_run: If True, only print what would be uploaded.
        repo_name: Override auto-derived repo name.
        prefix: Upload files under this subdirectory in the repo (e.g., "model").
    """
    model_dir = Path(model_dir).resolve()
    _ensure_base_ref(model_dir)
    existing, is_adapter = validate_model_dir(model_dir)

    if repo_name:
        pass
    elif prefix:
        # With --prefix, use parent dir as repo name (e.g., teacher/ -> parent run name)
        repo_name = model_dir.parent.name
    else:
        repo_name = derive_repo_name(model_dir)
    repo_id = f"{org}/{repo_name}"

    # Determine which files to upload
    if is_adapter:
        upload_files = existing & ADAPTER_FILES
    else:
        upload_files = existing & MODEL_FILES
    upload_files |= existing & PROVENANCE_FILES
    if include_resume:
        upload_files |= existing & RESUME_FILES

    print(f"Repo:    {repo_id}")
    print(f"Type:    {'LoRA adapter' if is_adapter else 'full model'}")
    if prefix:
        print(f"Prefix:  {prefix}/")
    print(f"Private: {private}")
    print(f"Files:   {len(upload_files)}")
    for f in sorted(upload_files):
        size_mb = (model_dir / f).stat().st_size / (1024 * 1024)
        dest = f"{prefix}/{f}" if prefix else f
        print(f"  {dest} ({size_mb:.1f} MB)")

    if dry_run:
        print("\nDry run — nothing uploaded.")
        return

    api = HfApi()
    api.create_repo(repo_id, private=private, exist_ok=True)

    print("Uploading...", end=" ", flush=True)
    api.upload_folder(
        folder_path=str(model_dir),
        path_in_repo=prefix or "",
        repo_id=repo_id,
        allow_patterns=list(upload_files),
        commit_message=f"Upload {prefix or 'model'} files",
    )
    print("done")

    # Upload model_ref.json with Hub-resolved base reference
    if not prefix:
        hub_ref = _build_hub_model_ref(model_dir, org)
        if hub_ref:
            api.upload_file(
                path_or_fileobj=json.dumps(hub_ref, indent=2).encode(),
                path_in_repo="model_ref.json",
                repo_id=repo_id,
                commit_message="Add model reference",
            )
            print(f"Base:    {hub_ref['model_name_or_path']}")

    # Upload model card at repo root (skip when uploading to a subdirectory)
    if not prefix:
        metadata = _load_metadata(model_dir)
        if metadata:
            card = _build_model_card(metadata, repo_id, is_adapter=is_adapter)
            api.upload_file(
                path_or_fileobj=card.encode(),
                path_in_repo="README.md",
                repo_id=repo_id,
                commit_message="Add model card",
            )

    print(f"\nPushed to https://huggingface.co/{repo_id}")


def _build_hub_model_ref(model_dir, org):
    """Read base/model_ref.json and translate local paths to Hub repo IDs.

    Args:
        model_dir: Path to the model directory.
        org: HuggingFace username or organization.

    Returns:
        Dict with Hub-resolved model_name_or_path, or None if no ref exists.
    """
    ref_path = model_dir / "base" / "model_ref.json"
    if not ref_path.exists():
        return None

    ref = json.loads(ref_path.read_text())
    model_path = ref.get("model_name_or_path", "")

    # Local path (outputs/ocr_sft) -> Hub repo (org/ocr_sft)
    if model_path.startswith("outputs/") or model_path.startswith("/"):
        local_name = Path(model_path).name
        ref["model_name_or_path"] = f"{org}/{local_name}"

    return ref


def _load_metadata(model_dir):
    """Load run_metadata.json if available.

    Args:
        model_dir: Path to the model directory.

    Returns:
        Dict of metadata, or None.
    """
    meta_path = model_dir / "run_metadata.json"
    if not meta_path.exists():
        # Check parent for checkpoint dirs
        meta_path = model_dir.parent / "run_metadata.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    return None


def _build_model_card(metadata, repo_id, is_adapter=False):
    """Build a minimal model card from run metadata.

    Args:
        metadata: Dict from run_metadata.json.
        repo_id: HuggingFace repo ID.
        is_adapter: Whether this is a LoRA adapter.

    Returns:
        Model card string.
    """
    tags = ["- knowledge-distillation", "- forge"]
    if is_adapter:
        tags.append("- peft")
        tags.append("- lora")

    lines = [
        "---",
        "library_name: peft" if is_adapter else "library_name: transformers",
        "tags:",
        *tags,
        "---",
        "",
        f"# {repo_id.split('/')[-1]}",
        "",
        f"{'LoRA adapter' if is_adapter else 'Model'} produced by [Forge](https://github.com/NU-Deep-Learning/Forge).",
        "",
    ]

    if metadata:
        lines.append("## Training Info")
        lines.append("")
        lines.append(f"- **Config**: `{metadata.get('config_id', 'N/A')}`")
        lines.append(f"- **Created**: {metadata.get('created', 'N/A')}")
        lines.append(f"- **Seed**: {metadata.get('seed', 'N/A')}")
        if metadata.get("wandb_run_id"):
            lines.append(f"- **W&B Run**: `{metadata['wandb_run_id']}`")
        lines.append("")

    return "\n".join(lines)


def list_hub_models(org):
    """List models on the Hub for the given org/user.

    Args:
        org: HuggingFace username or organization.
    """
    models = list(list_models(author=org, sort="lastModified", direction=-1))
    if not models:
        print(f"No models found for {org}")
        return

    print(f"Models for {org} ({len(models)}):\n")
    for m in models:
        private_tag = " [private]" if m.private else ""
        print(f"  {m.id}{private_tag}")
        if m.last_modified:
            print(f"    Modified: {m.last_modified}")
    print()


def delete_model(repo_id):
    """Delete a model repo from the Hub.

    Args:
        repo_id: Full repo ID (org/name).
    """
    api = HfApi()
    confirm = input(f"Delete {repo_id} from HuggingFace Hub? [y/N] ")
    if confirm.lower() != "y":
        print("Cancelled.")
        return
    api.delete_repo(repo_id)
    print(f"Deleted {repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Push models to HuggingFace Hub")
    parser.add_argument(
        "model_dir",
        nargs="?",
        help="Path to model or checkpoint directory",
    )
    parser.add_argument(
        "--org",
        default=None,
        help="HuggingFace username or org (default: logged-in user)",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Create a public repo (default: private)",
    )
    parser.add_argument(
        "--include-resume",
        action="store_true",
        help="Include optimizer/scheduler state for full training resume",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without uploading",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List your models on the Hub",
    )
    parser.add_argument(
        "--repo-name",
        default=None,
        help="Override auto-derived repo name",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Upload files under this subdirectory (e.g., 'model')",
    )
    parser.add_argument(
        "--repair",
        action="store_true",
        help="Backfill missing base/model_ref.json from config.yaml (no upload)",
    )
    parser.add_argument(
        "--delete",
        metavar="REPO_ID",
        help="Delete a model repo from the Hub (org/name)",
    )

    args = parser.parse_args()

    # Resolve org
    org = args.org or get_default_org()

    if args.list:
        list_hub_models(org)
        return

    if args.delete:
        delete_model(args.delete)
        return

    if not args.model_dir:
        parser.error("model_dir is required (unless using --list or --delete)")

    if args.repair:
        created = _ensure_base_ref(Path(args.model_dir).resolve())
        if not created:
            print("Nothing to repair (ref already exists or no config.yaml found)")
        return

    push_model(
        model_dir=args.model_dir,
        org=org,
        private=not args.public,
        include_resume=args.include_resume,
        dry_run=args.dry_run,
        repo_name=args.repo_name,
        prefix=args.prefix,
    )


if __name__ == "__main__":
    main()
