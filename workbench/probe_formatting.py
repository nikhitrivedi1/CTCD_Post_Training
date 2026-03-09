"""Quick probe: does the current model produce inline formatting?

Runs a small set of formatting-heavy document images through the model and
checks whether the output contains bold, italic, or strikethrough markers.
This answers whether the model has latent formatting capability that just
needs activation vs. being completely absent.

Usage:
    # With OmniDocBench images (default)
    python workbench/probe_formatting.py --model Qwen/Qwen3-VL-2B-Instruct

    # With a LoRA adapter
    python workbench/probe_formatting.py --model Qwen/Qwen3-VL-2B-Instruct --adapter outputs/ablations_00/checkpoint-XXX

    # With custom images directory
    python workbench/probe_formatting.py --model Qwen/Qwen3-VL-2B-Instruct --images_dir path/to/images

    # Test updated system prompt vs original
    python workbench/probe_formatting.py --model Qwen/Qwen3-VL-2B-Instruct --compare_prompts
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def extract_formatting_spans(text: str) -> dict[str, list[str]]:
    """Extract inline formatting spans from markdown text.

    Args:
        text: Markdown text.

    Returns:
        Dict with keys 'bold', 'italic', 'strikethrough'.
    """
    bold = re.findall(r"\*\*(.+?)\*\*", text)
    italic = re.findall(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", text)
    strikethrough = re.findall(r"~~(.+?)~~", text)
    return {"bold": bold, "italic": italic, "strikethrough": strikethrough}


def count_formatting(text: str) -> dict[str, int]:
    """Count formatting spans.

    Args:
        text: Markdown text.

    Returns:
        Dict with counts per formatting type plus total.
    """
    spans = extract_formatting_spans(text)
    counts = {k: len(v) for k, v in spans.items()}
    counts["total"] = sum(counts.values())
    return counts


ORIGINAL_PROMPT = (
    "Below is an image of a document page. Extract all text content "
    "and output it in clean markdown format, preserving the document's "
    "structure (headings, lists, tables, etc.)."
)

UPDATED_PROMPT = (
    "Below is an image of a document page. Extract all text content "
    "and output it in clean markdown format, preserving the document's "
    "structure (headings, lists, tables) and inline formatting "
    "(bold, italic, strikethrough) exactly as shown."
)


def find_probe_images(images_dir: str, max_images: int = 20) -> list[Path]:
    """Find document images to probe.

    Args:
        images_dir: Directory containing document images.
        max_images: Maximum number of images to use.

    Returns:
        List of image file paths.
    """
    exts = {".jpg", ".jpeg", ".png"}
    images = [
        p for p in Path(images_dir).iterdir()
        if p.suffix.lower() in exts
    ]
    images.sort(key=lambda p: p.name)
    # Sample evenly across the set
    if len(images) > max_images:
        step = len(images) // max_images
        images = images[::step][:max_images]
    return images


def run_probe(engine, images: list[Path], prompt: str, max_new_tokens: int) -> list[dict]:
    """Run model on images and collect formatting stats.

    Args:
        engine: InferenceEngine instance.
        images: List of image paths.
        prompt: OCR prompt to use.
        max_new_tokens: Max tokens to generate.

    Returns:
        List of result dicts per image.
    """
    from PIL import Image
    from tqdm import tqdm

    results = []
    for img_path in tqdm(images, desc="Probing"):
        image = Image.open(img_path).convert("RGB")
        outputs = engine.generate([(image, prompt)], max_new_tokens=max_new_tokens)
        text = outputs[0] if outputs else ""

        counts = count_formatting(text)
        spans = extract_formatting_spans(text)
        results.append({
            "image": img_path.name,
            "counts": counts,
            "spans": {k: v[:5] for k, v in spans.items()},  # First 5 examples
            "output_length": len(text),
            "output_preview": text[:200],
        })
    return results


def print_summary(results: list[dict], label: str):
    """Print formatting probe summary.

    Args:
        results: List of per-image result dicts.
        label: Label for this run (e.g., prompt name).
    """
    total_images = len(results)
    images_with_formatting = sum(1 for r in results if r["counts"]["total"] > 0)
    total_bold = sum(r["counts"]["bold"] for r in results)
    total_italic = sum(r["counts"]["italic"] for r in results)
    total_strike = sum(r["counts"]["strikethrough"] for r in results)
    total_all = total_bold + total_italic + total_strike

    print(f"\n{'=' * 60}")
    print(f"Formatting Probe: {label}")
    print(f"{'=' * 60}")
    print(f"Images probed:         {total_images}")
    print(f"With any formatting:   {images_with_formatting}/{total_images} ({100*images_with_formatting/total_images:.0f}%)")
    print(f"Total bold spans:      {total_bold}")
    print(f"Total italic spans:    {total_italic}")
    print(f"Total strikethrough:   {total_strike}")
    print(f"Total all formatting:  {total_all}")
    print(f"Avg per image:         {total_all/total_images:.1f}")

    if total_all > 0:
        print(f"\nSample formatting spans:")
        for r in results:
            if r["counts"]["total"] > 0:
                print(f"  {r['image']}:")
                for fmt_type in ["bold", "italic", "strikethrough"]:
                    if r["spans"][fmt_type]:
                        examples = ", ".join(f'"{s}"' for s in r["spans"][fmt_type][:3])
                        print(f"    {fmt_type}: {examples}")
                break  # Just show first example
    print()


def main():
    parser = argparse.ArgumentParser(description="Probe model formatting capability")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--adapter", type=str, default=None, help="LoRA adapter path")
    parser.add_argument("--images_dir", type=str, default="data/ocr/omnidocbench/images",
                        help="Directory with document images")
    parser.add_argument("--max_images", type=int, default=20, help="Max images to probe")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--compare_prompts", action="store_true",
                        help="Run with both original and updated prompts")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results JSON to this path")
    args = parser.parse_args()

    from src.inference import InferenceEngine

    images = find_probe_images(args.images_dir, args.max_images)
    if not images:
        print(f"No images found in {args.images_dir}")
        sys.exit(1)
    print(f"Found {len(images)} images to probe")

    engine = InferenceEngine(args.model, adapter_path=args.adapter)
    all_results = {}

    if args.compare_prompts:
        # Run with both prompts
        results_orig = run_probe(engine, images, ORIGINAL_PROMPT, args.max_new_tokens)
        print_summary(results_orig, "Original prompt")
        all_results["original_prompt"] = results_orig

        results_new = run_probe(engine, images, UPDATED_PROMPT, args.max_new_tokens)
        print_summary(results_new, "Updated prompt (with formatting)")
        all_results["updated_prompt"] = results_new

        # Comparison
        orig_total = sum(r["counts"]["total"] for r in results_orig)
        new_total = sum(r["counts"]["total"] for r in results_new)
        delta = new_total - orig_total
        print(f"Delta: {delta:+d} formatting spans ({orig_total} -> {new_total})")
    else:
        results = run_probe(engine, images, UPDATED_PROMPT, args.max_new_tokens)
        print_summary(results, f"{args.model}")
        all_results["results"] = results

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
