"""Convert OmniDocBench annotations to markdown ground truth for GRPO training.

Reads OmniDocBench.json structured annotations and converts each page to
clean markdown, preserving headings, tables, formulas, and reading order.

Run this on the cluster after downloading the dataset.

Usage:
    python scripts/data/convert_omnidocbench.py --data_dir data/ocr/omnidocbench
    python scripts/data/convert_omnidocbench.py --data_dir data/ocr/omnidocbench --verify
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Block types to skip (not textual content)
SKIP_TYPES = {
    "figure", "abandon", "text_mask", "table_mask", "need_mask",
    "page_number", "header", "footer",
}


def html_table_to_markdown(html: str) -> str:
    """Convert an HTML table to markdown table format.

    Args:
        html: HTML table string from OmniDocBench annotations.

    Returns:
        Markdown table string.
    """
    # Clean up custom tags: <li>...<i> used as list markers in some cells
    html = html.replace("<li>", "").replace("<i>", "")

    rows = re.findall(r"<tr>(.*?)</tr>", html, re.DOTALL)
    if not rows:
        return html

    table_rows = []
    for row_html in rows:
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row_html, re.DOTALL)
        cells = [c.strip().replace("\n", " ") for c in cells]
        table_rows.append(cells)

    if not table_rows:
        return html

    # Build markdown table
    lines = []
    for i, row in enumerate(table_rows):
        lines.append("| " + " | ".join(row) + " |")
        if i == 0:
            lines.append("| " + " | ".join(["---"] * len(row)) + " |")

    return "\n".join(lines)


def block_to_markdown(block: dict) -> str | None:
    """Convert a single annotation block to markdown text.

    Args:
        block: Block dict from layout_dets with category_type, text/html/latex, etc.

    Returns:
        Markdown string, or None if block should be skipped.
    """
    btype = block.get("category_type", "")

    if btype in SKIP_TYPES or block.get("ignore", False):
        return None

    if btype == "title":
        text = block.get("text", "").strip()
        return f"# {text}" if text else None

    if btype == "text_block":
        text = block.get("text", "").strip()
        return text if text else None

    if btype == "table":
        html = block.get("html", "")
        if html:
            return html_table_to_markdown(html)
        return None

    if btype in ("equation_isolated", "equation_inline"):
        latex = block.get("latex", "").strip()
        return latex if latex else None

    if btype == "table_caption":
        text = block.get("text", "").strip()
        return f"*{text}*" if text else None

    if btype == "figure_caption":
        text = block.get("text", "").strip()
        return f"*{text}*" if text else None

    if btype == "equation_caption":
        text = block.get("text", "").strip()
        return f"*{text}*" if text else None

    if btype == "reference":
        text = block.get("text", "").strip()
        return text if text else None

    if btype in ("figure_footnote", "table_footnote", "page_footnote"):
        text = block.get("text", "").strip()
        return text if text else None

    if btype == "code_txt":
        text = block.get("text", "").strip()
        return f"```\n{text}\n```" if text else None

    if btype == "code_txt_caption":
        text = block.get("text", "").strip()
        return f"*{text}*" if text else None

    if btype == "equation_explanation":
        text = block.get("text", "").strip()
        return text if text else None

    # Unknown type - include text if available
    text = block.get("text", "").strip()
    if text:
        logger.debug(f"Unknown block type '{btype}', including text")
        return text
    return None


def page_to_markdown(page: dict) -> str:
    """Convert a full page annotation to markdown.

    Blocks are sorted by their 'order' field to preserve reading order.

    Args:
        page: Page dict with layout_dets, page_info, extra.

    Returns:
        Full markdown string for the page.
    """
    blocks = page.get("layout_dets", [])
    # Sort by order field
    blocks = sorted(blocks, key=lambda b: b.get("order") if b.get("order") is not None else 9999)

    parts = []
    for block in blocks:
        md = block_to_markdown(block)
        if md:
            parts.append(md)

    return "\n\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Convert OmniDocBench to markdown GT")
    parser.add_argument("--data_dir", type=str, default="data/ocr/omnidocbench",
                        help="Path to OmniDocBench dataset root")
    parser.add_argument("--verify", action="store_true",
                        help="Print stats and sample outputs instead of writing files")
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    annotations_path = data_path / "OmniDocBench.json"
    images_dir = data_path / "images"
    markdown_dir = data_path / "markdown"

    if not annotations_path.exists():
        raise FileNotFoundError(
            f"OmniDocBench.json not found in {args.data_dir}. Download with:\n"
            f"  python -c \"from huggingface_hub import snapshot_download; "
            f"snapshot_download('opendatalab/OmniDocBench', local_dir='{args.data_dir}', repo_type='dataset')\""
        )

    with open(annotations_path) as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data)} page annotations")

    if args.verify:
        # Print stats and samples
        from collections import Counter
        block_types = Counter()
        for page in data:
            for det in page["layout_dets"]:
                block_types[det["category_type"]] += 1
        print("Block type counts:")
        for bt, count in block_types.most_common():
            print(f"  {bt}: {count}")

        # Show 3 sample conversions
        for i in [0, len(data) // 2, len(data) - 1]:
            page = data[i]
            img = page["page_info"]["image_path"]
            md = page_to_markdown(page)
            print(f"\n{'='*60}")
            print(f"Page {i}: {img}")
            print(f"{'='*60}")
            print(md[:500])
            if len(md) > 500:
                print(f"... ({len(md)} chars total)")
        return

    # Write markdown files
    os.makedirs(markdown_dir, exist_ok=True)
    missing_images = 0
    written = 0

    for page in data:
        image_path = page["page_info"]["image_path"]

        # Verify image exists
        if not (images_dir / image_path).exists():
            missing_images += 1
            continue

        md = page_to_markdown(page)
        if not md.strip():
            continue

        # Use image filename as markdown filename
        md_name = Path(image_path).stem + ".md"
        md_path = markdown_dir / md_name

        with open(md_path, "w") as f:
            f.write(md)
        written += 1

    if missing_images > 0:
        logger.warning(f"Skipped {missing_images} pages with missing images")
    logger.info(f"Written {written} markdown files to {markdown_dir}")


if __name__ == "__main__":
    main()
