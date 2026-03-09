"""Audit training data for inline formatting content.

Scans ground truth markdown files to quantify how much bold, italic, and
strikethrough formatting exists in the training data. Helps decide whether
we need external formatting-rich data or can filter from existing data.

Usage:
    # Audit olmocr-mix (on cluster)
    python workbench/audit_formatting.py --data_dir data/ocr/olmocr-mix-full

    # Audit OmniDocBench ground truth
    python workbench/audit_formatting.py --data_dir data/ocr/omnidocbench/markdown

    # Audit with detailed per-file output
    python workbench/audit_formatting.py --data_dir data/ocr/olmocr-mix-full --verbose

    # Export formatting-rich file list for filtered training
    python workbench/audit_formatting.py --data_dir data/ocr/olmocr-mix-full --export_rich formatting_rich_files.txt --min_spans 3
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def extract_formatting_counts(text: str) -> dict[str, int]:
    """Count inline formatting spans in text.

    Args:
        text: Markdown text.

    Returns:
        Dict with counts per formatting type.
    """
    bold = len(re.findall(r"\*\*(.+?)\*\*", text))
    italic = len(re.findall(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", text))
    strikethrough = len(re.findall(r"~~(.+?)~~", text))
    underline = len(re.findall(r"<u>(.+?)</u>", text))
    code_inline = len(re.findall(r"`[^`]+`", text))
    return {
        "bold": bold,
        "italic": italic,
        "strikethrough": strikethrough,
        "underline": underline,
        "code_inline": code_inline,
        "total_inline": bold + italic + strikethrough,
    }


def extract_structure_counts(text: str) -> dict[str, int]:
    """Count structural markdown elements.

    Args:
        text: Markdown text.

    Returns:
        Dict with counts of headings, lists, tables, code blocks.
    """
    headings = len(re.findall(r"^#{1,6}\s", text, re.MULTILINE))
    bullets = len(re.findall(r"^[\-\*]\s", text, re.MULTILINE))
    numbered = len(re.findall(r"^\d+\.\s", text, re.MULTILINE))
    tables = len(re.findall(r"^\|.+\|$", text, re.MULTILINE))
    code_blocks = len(re.findall(r"^```", text, re.MULTILINE)) // 2
    return {
        "headings": headings,
        "bullet_lists": bullets,
        "numbered_lists": numbered,
        "table_rows": tables,
        "code_blocks": code_blocks,
    }


def find_markdown_files(data_dir: str) -> list[Path]:
    """Find all markdown/text ground truth files.

    Supports multiple directory structures:
    - Flat directory of .md files (OmniDocBench)
    - Nested directories with ground_truth.md or .txt files (olmocr-mix)
    - HuggingFace dataset directories with text columns

    Args:
        data_dir: Root directory to search.

    Returns:
        List of file paths.
    """
    data_path = Path(data_dir)
    files = []

    # Try flat .md files first
    md_files = list(data_path.glob("*.md"))
    if md_files:
        return md_files

    # Try nested structure (olmocr-mix style)
    nested = list(data_path.glob("**/*.md"))
    if nested:
        return nested

    # Try .txt files
    txt_files = list(data_path.glob("**/*.txt"))
    if txt_files:
        return txt_files

    return files


def audit_files(files: list[Path], verbose: bool = False) -> dict:
    """Audit a list of files for formatting content.

    Args:
        files: List of file paths.
        verbose: Print per-file details.

    Returns:
        Audit results dict.
    """
    from tqdm import tqdm

    total_files = len(files)
    formatting_counts = Counter()
    structure_counts = Counter()
    files_with_formatting = 0
    files_with_bold = 0
    files_with_italic = 0
    files_with_strike = 0
    rich_files = []  # Files with substantial formatting

    for fpath in tqdm(files, desc="Auditing", unit="file"):
        text = fpath.read_text(errors="replace")
        fmt = extract_formatting_counts(text)
        struct = extract_structure_counts(text)

        for k, v in fmt.items():
            formatting_counts[k] += v
        for k, v in struct.items():
            structure_counts[k] += v

        has_formatting = fmt["total_inline"] > 0
        if has_formatting:
            files_with_formatting += 1
        if fmt["bold"] > 0:
            files_with_bold += 1
        if fmt["italic"] > 0:
            files_with_italic += 1
        if fmt["strikethrough"] > 0:
            files_with_strike += 1
        if fmt["total_inline"] >= 3:
            rich_files.append((str(fpath), fmt))

        if verbose and has_formatting:
            print(f"  {fpath.name}: bold={fmt['bold']}, italic={fmt['italic']}, "
                  f"strike={fmt['strikethrough']}")

    return {
        "total_files": total_files,
        "files_with_any_formatting": files_with_formatting,
        "files_with_bold": files_with_bold,
        "files_with_italic": files_with_italic,
        "files_with_strikethrough": files_with_strike,
        "formatting_counts": dict(formatting_counts),
        "structure_counts": dict(structure_counts),
        "rich_files": rich_files,
        "pct_with_formatting": 100 * files_with_formatting / total_files if total_files else 0,
        "pct_with_bold": 100 * files_with_bold / total_files if total_files else 0,
    }


def print_audit(results: dict, data_dir: str):
    """Print formatted audit results.

    Args:
        results: Audit results dict.
        data_dir: Data directory that was audited.
    """
    total = results["total_files"]

    print(f"\n{'=' * 60}")
    print(f"Formatting Audit: {data_dir}")
    print(f"{'=' * 60}")
    print(f"Total files:              {total}")
    print()

    print("Inline Formatting:")
    print(f"  Files with ANY:         {results['files_with_any_formatting']:>6} ({results['pct_with_formatting']:.1f}%)")
    print(f"  Files with bold:        {results['files_with_bold']:>6} ({results['pct_with_bold']:.1f}%)")
    print(f"  Files with italic:      {results['files_with_italic']:>6} ({100*results['files_with_italic']/total:.1f}%)")
    print(f"  Files with strikethrough:{results['files_with_strikethrough']:>5} ({100*results['files_with_strikethrough']/total:.1f}%)")
    print()

    fmt = results["formatting_counts"]
    print("Total Spans:")
    print(f"  Bold (**...**):         {fmt.get('bold', 0):>6}")
    print(f"  Italic (*...*):         {fmt.get('italic', 0):>6}")
    print(f"  Strikethrough (~~...~~):{fmt.get('strikethrough', 0):>6}")
    print(f"  Underline (<u>...</u>): {fmt.get('underline', 0):>6}")
    print(f"  Inline code (`...`):    {fmt.get('code_inline', 0):>6}")
    print()

    struct = results["structure_counts"]
    print("Structure Elements:")
    print(f"  Headings:               {struct.get('headings', 0):>6}")
    print(f"  Bullet lists:           {struct.get('bullet_lists', 0):>6}")
    print(f"  Numbered lists:         {struct.get('numbered_lists', 0):>6}")
    print(f"  Table rows:             {struct.get('table_rows', 0):>6}")
    print(f"  Code blocks:            {struct.get('code_blocks', 0):>6}")
    print()

    rich = results["rich_files"]
    print(f"Rich files (>=3 inline spans): {len(rich)}")
    if rich:
        print(f"  Top examples:")
        top = sorted(rich, key=lambda x: x[1]["total_inline"], reverse=True)[:5]
        for fpath, counts in top:
            name = Path(fpath).name
            print(f"    {name}: bold={counts['bold']}, italic={counts['italic']}, "
                  f"strike={counts['strikethrough']}")
    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(description="Audit training data for formatting content")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory with ground truth markdown files")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-file details for files with formatting")
    parser.add_argument("--export_rich", type=str, default=None,
                        help="Export list of formatting-rich files to this path")
    parser.add_argument("--min_spans", type=int, default=3,
                        help="Minimum inline formatting spans to count as 'rich'")
    parser.add_argument("--output", type=str, default=None,
                        help="Save full audit results as JSON")
    args = parser.parse_args()

    files = find_markdown_files(args.data_dir)
    if not files:
        print(f"No markdown/text files found in {args.data_dir}")
        sys.exit(1)

    print(f"Found {len(files)} files to audit")
    results = audit_files(files, verbose=args.verbose)
    print_audit(results, args.data_dir)

    if args.export_rich:
        rich = [fpath for fpath, counts in results["rich_files"]
                if counts["total_inline"] >= args.min_spans]
        with open(args.export_rich, "w") as f:
            f.write("\n".join(rich) + "\n")
        print(f"Exported {len(rich)} formatting-rich file paths to {args.export_rich}")

    if args.output:
        # Convert rich_files for JSON serialization
        save_results = {k: v for k, v in results.items() if k != "rich_files"}
        save_results["rich_file_count"] = len(results["rich_files"])
        save_results["rich_files_top20"] = [
            {"file": fpath, "counts": counts}
            for fpath, counts in sorted(results["rich_files"],
                                        key=lambda x: x[1]["total_inline"], reverse=True)[:20]
        ]
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(save_results, f, indent=2)
        print(f"Full results saved to {args.output}")


if __name__ == "__main__":
    main()
