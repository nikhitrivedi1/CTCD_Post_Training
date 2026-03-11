"""OmniDocBench dataset handler.

Public OCR dataset from opendatalab. Contains document page images with
structured annotations (headings, tables, formulas, reading order).

Pipeline: download from HF -> convert JSON annotations to markdown -> load.
"""

import json
import logging
import os
import re
from pathlib import Path

from datasets import Dataset
from PIL import Image

from .base import DatasetHandler

logger = logging.getLogger(__name__)

# Block types to skip (not textual content)
_SKIP_TYPES = {
    "figure", "abandon", "text_mask", "table_mask", "need_mask",
    "page_number", "header", "footer",
}


def _html_table_to_markdown(html: str) -> str:
    """Convert an HTML table to markdown table format.

    Args:
        html: HTML table string from OmniDocBench annotations.

    Returns:
        Markdown table string.
    """
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

    lines = []
    for i, row in enumerate(table_rows):
        lines.append("| " + " | ".join(row) + " |")
        if i == 0:
            lines.append("| " + " | ".join(["---"] * len(row)) + " |")
    return "\n".join(lines)


def _block_to_markdown(block: dict) -> str | None:
    """Convert a single annotation block to markdown text.

    Args:
        block: Block dict from layout_dets.

    Returns:
        Markdown string, or None if block should be skipped.
    """
    btype = block.get("category_type", "")

    if btype in _SKIP_TYPES or block.get("ignore", False):
        return None

    if btype == "title":
        text = block.get("text", "").strip()
        return f"# {text}" if text else None

    if btype == "text_block":
        text = block.get("text", "").strip()
        return text if text else None

    if btype == "table":
        html = block.get("html", "")
        return _html_table_to_markdown(html) if html else None

    if btype in ("equation_isolated", "equation_inline"):
        latex = block.get("latex", "").strip()
        return latex if latex else None

    if btype in ("table_caption", "figure_caption", "equation_caption", "code_txt_caption"):
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

    if btype == "equation_explanation":
        text = block.get("text", "").strip()
        return text if text else None

    # Unknown type - include text if available
    text = block.get("text", "").strip()
    if text:
        logger.debug(f"Unknown block type '{btype}', including text")
        return text
    return None


def _page_to_markdown(page: dict) -> str:
    """Convert a full page annotation to markdown.

    Args:
        page: Page dict with layout_dets, page_info, extra.

    Returns:
        Full markdown string for the page.
    """
    blocks = page.get("layout_dets", [])
    blocks = sorted(blocks, key=lambda b: b.get("order") if b.get("order") is not None else 9999)

    parts = []
    for block in blocks:
        md = _block_to_markdown(block)
        if md:
            parts.append(md)
    return "\n\n".join(parts)


class OmniDocBenchHandler(DatasetHandler):
    """Handler for OmniDocBench (opendatalab/OmniDocBench)."""

    def prepare(self, data_dir: str, **kwargs) -> None:
        """Download from HuggingFace and convert JSON annotations to markdown.

        Args:
            data_dir: Root directory (e.g. data/ocr/omnidocbench).
        """
        data_path = Path(data_dir)
        annotations_path = data_path / "OmniDocBench.json"
        images_dir = data_path / "images"
        markdown_dir = data_path / "markdown"

        # Step 1: Download if needed
        if not annotations_path.exists():
            logger.info("Downloading OmniDocBench from HuggingFace...")
            from huggingface_hub import snapshot_download
            snapshot_download(
                "opendatalab/OmniDocBench",
                local_dir=str(data_path),
                repo_type="dataset",
            )
            logger.info("Download complete")

        # Step 2: Convert JSON to markdown (idempotent - skip if exists)
        if markdown_dir.exists() and any(markdown_dir.glob("*.md")):
            logger.info(f"Markdown already exists ({len(list(markdown_dir.glob('*.md')))} files), skipping conversion")
            return

        logger.info("Converting OmniDocBench annotations to markdown...")
        with open(annotations_path) as f:
            data = json.load(f)

        os.makedirs(markdown_dir, exist_ok=True)
        missing_images = 0
        written = 0

        for page in data:
            image_path = page["page_info"]["image_path"]
            if not (images_dir / image_path).exists():
                missing_images += 1
                continue

            md = _page_to_markdown(page)
            if not md.strip():
                continue

            md_name = Path(image_path).stem + ".md"
            with open(markdown_dir / md_name, "w") as f:
                f.write(md)
            written += 1

        if missing_images > 0:
            logger.warning(f"Skipped {missing_images} pages with missing images")
        logger.info(f"Written {written} markdown files to {markdown_dir}")

    def validate(self, data_dir: str) -> dict:
        """Check that images and markdown files exist and are aligned.

        Args:
            data_dir: Root directory.

        Returns:
            Validation results dict.
        """
        data_path = Path(data_dir)
        images_dir = data_path / "images"
        markdown_dir = data_path / "markdown"

        if not images_dir.exists():
            return {"ok": False, "details": f"images/ not found in {data_dir}"}
        if not markdown_dir.exists():
            return {"ok": False, "details": f"markdown/ not found in {data_dir}"}

        images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        md_files = list(markdown_dir.glob("*.md"))
        img_stems = {p.stem for p in images}
        md_stems = {p.stem for p in md_files}
        matched = img_stems & md_stems
        unmatched = md_stems - img_stems

        result = {
            "ok": len(unmatched) == 0 and len(matched) > 0,
            "images": len(images),
            "markdown": len(md_files),
            "matched": len(matched),
            "unmatched_markdown": len(unmatched),
        }
        details = f"{len(matched)} matched pairs, {len(images)} images, {len(md_files)} markdown"
        if unmatched:
            details += f", {len(unmatched)} markdown without images"
        result["details"] = details
        return result

    def load_paths(self, data_dir: str, split: str | None = None) -> list[dict]:
        """List image/GT file paths.

        Args:
            data_dir: Root directory with images/ and markdown/.
            split: Ignored (OmniDocBench has no pre-defined splits).

        Returns:
            List of dicts with "image_path" and "gt_path" as Path objects.
        """
        data_path = Path(data_dir)
        images_dir = data_path / "images"
        markdown_dir = data_path / "markdown"

        if not images_dir.exists():
            raise FileNotFoundError(f"images/ not found in {data_dir}. Run: prepare.py --dataset omnidocbench")
        if not markdown_dir.exists():
            raise FileNotFoundError(f"markdown/ not found in {data_dir}. Run: prepare.py --dataset omnidocbench")

        pairs = []
        for md_path in sorted(markdown_dir.glob("*.md")):
            image_path = None
            for ext in (".jpg", ".png", ".jpeg"):
                candidate = images_dir / (md_path.stem + ext)
                if candidate.exists():
                    image_path = candidate
                    break
            if image_path:
                pairs.append({"image_path": image_path, "gt_path": md_path})

        return pairs

    def load(
        self,
        data_dir: str,
        max_train_samples: int | None = None,
        eval_fraction: float = 0.1,
        **kwargs,
    ) -> tuple[Dataset, Dataset | None]:
        """Load image + markdown pairs as raw samples.

        Args:
            data_dir: Root directory with images/ and markdown/.
            max_train_samples: Cap on training samples.
            eval_fraction: Fraction held out for eval.

        Returns:
            Tuple of (train_dataset, eval_dataset).
        """
        data_path = Path(data_dir)
        images_dir = data_path / "images"
        markdown_dir = data_path / "markdown"

        if not images_dir.exists():
            raise FileNotFoundError(f"images/ not found in {data_dir}. Run: prepare.py --dataset omnidocbench")
        if not markdown_dir.exists():
            raise FileNotFoundError(f"markdown/ not found in {data_dir}. Run: prepare.py --dataset omnidocbench")

        samples = []
        for md_path in sorted(markdown_dir.glob("*.md")):
            # Find matching image
            image_path = None
            for ext in (".jpg", ".png", ".jpeg"):
                candidate = images_dir / (md_path.stem + ext)
                if candidate.exists():
                    image_path = candidate
                    break
            if image_path is None:
                continue

            with open(md_path) as f:
                ground_truth = f.read().strip()
            if not ground_truth:
                continue

            with Image.open(image_path) as im:
                image = im.convert("RGB")
                image.load()

            samples.append({"image": image, "ground_truth": ground_truth})

        if not samples:
            raise ValueError(f"No valid samples found in {data_dir}")

        logger.info(f"Loaded {len(samples)} OmniDocBench pages")

        dataset = Dataset.from_list(samples)
        splits = dataset.train_test_split(test_size=eval_fraction, seed=42)
        train_dataset = splits["train"]
        eval_dataset = splits["test"]

        if max_train_samples and len(train_dataset) > max_train_samples:
            train_dataset = train_dataset.select(range(max_train_samples))

        logger.info(f"OmniDocBench: {len(train_dataset)} train, {len(eval_dataset)} eval")
        return train_dataset, eval_dataset
