"""Generate short, formatting-dense training data for GRPO/RL fine-tuning.

Produces compact (image, markdown_gt) pairs optimized for RL: high formatting
density, balanced type distribution (bold/italic/strikethrough), and short
documents (target 128-256 tokens). Shares the render pipeline with the SFT
generator but uses different content generation tuned for last-mile alignment.

Output structure (same as SFT generator, compatible with FormattingHandler):
    output_dir/
      00_text/       train/{images,markdown}/  eval/{images,markdown}/
      01_tables/     ...
      02_academic/   ...
      03_financial/  ...
      04_legal/      ...
      05_technical/  ...
      06_books/      ...
      07_slides/     ...
      08_news/       ...
      09_forms/      ...
      metadata.json

Usage:
    # Default: 150 pages/category, 1500 total
    python scripts/data/formatting_rl_generator.py \
        --output_dir data/ocr/formatting-rl --augment

    # Fewer pages for quick iteration
    python scripts/data/formatting_rl_generator.py \
        --output_dir data/ocr/formatting-rl --num_pages 50

    # Specific categories only
    python scripts/data/formatting_rl_generator.py \
        --output_dir data/ocr/formatting-rl --categories 00_text 03_financial
"""

import argparse
import hashlib
import json
import logging
import os
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.formatting_generator import (
    CSS_TEMPLATES,
    augment_image,
    count_formatting_stats,
    generate_table_markdown,
    markdown_to_html,
    render_batch,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Short paragraph pools (1-2 sentences each, ~30-60 tokens)
# ---------------------------------------------------------------------------

SHORT_PARAGRAPHS = {
    "00_text": [
        "The water cycle describes the continuous movement of water through the Earth's systems.",
        "Photosynthesis converts sunlight into glucose and oxygen in plant cells.",
        "The Renaissance marked a cultural rebirth across Europe in art and philosophy.",
        "Plate tectonics explains how crustal plates glide over the mantle.",
        "Democracy originated in ancient Athens with direct citizen participation.",
        "The human brain contains roughly 86 billion neurons connected through synapses.",
        "Climate change refers to long-term shifts in global temperatures and weather.",
        "The periodic table organizes elements by atomic number and electron configuration.",
        "Ocean currents distribute heat across the globe and influence regional climates.",
        "Mitosis divides a cell into two genetically identical daughter cells.",
    ],
    "01_tables": [
        "The table below summarizes quarterly performance across all divisions.",
        "Results are presented with bold entries indicating values above target.",
        "Comparative benchmarks demonstrate improvement across model sizes.",
        "Survey responses are aggregated by demographic category below.",
        "Performance metrics are averaged over three independent runs.",
        "Hardware configurations were tested under identical conditions.",
    ],
    "02_academic": [
        "Supervised learning requires labeled data pairing inputs with desired outputs.",
        "The transformer architecture uses self-attention instead of recurrence.",
        "Gradient descent iteratively adjusts parameters to reduce the loss function.",
        "Bayesian inference updates beliefs by combining priors with observed data.",
        "Convolutional filters detect spatial features at different scales.",
        "The central limit theorem applies regardless of population distribution shape.",
        "Clinical trials showed significant symptom reduction versus placebo.",
        "Protein folding predicts 3D structure from amino acid sequence.",
    ],
    "03_financial": [
        "Revenue increased 12.3% year-over-year driven by cloud services growth.",
        "The board approved a quarterly dividend of $0.85 per share.",
        "Risk factors include market volatility and competitive pressures.",
        "Total assets grew to $4.2 billion, up 8.7% from the prior year.",
        "Earnings per share reached $3.42, surpassing estimates of $3.28.",
        "Capital expenditures totaled $890 million for infrastructure upgrades.",
        "Free cash flow generation remained strong at $1.1 billion.",
        "Provision for credit losses rose to $45 million from $32 million.",
    ],
    "04_legal": [
        "The licensee shall maintain insurance coverage for the contract duration.",
        "Users may request data deletion by submitting a written request.",
        "Neither party shall be liable for delays caused by force majeure events.",
        "Confidential information includes any data marked as confidential.",
        "Arbitration follows American Arbitration Association rules.",
        "IP rights vest in the client upon full payment of deliverables.",
        "This agreement is governed by the laws of the State of Delaware.",
        "Modifications to the terms must be submitted in writing.",
    ],
    "05_technical": [
        "The microservices architecture uses event-driven communication between components.",
        "The API accepts JSON payloads with rate limiting at 1000 requests per minute.",
        "Kubernetes manages container orchestration with automatic scaling.",
        "Database migrations use versioned schemas with rollback support.",
        "OAuth 2.0 tokens expire after 15 minutes with 30-day refresh tokens.",
        "Logs follow structured JSON format with correlation IDs.",
        "The build pipeline runs compilation, testing, and deployment stages.",
        "Memory allocation follows a buddy system to minimize fragmentation.",
    ],
    "06_books": [
        "Natural selection drives adaptation through differential survival and reproduction.",
        "General relativity showed that massive objects curve the fabric of spacetime.",
        "Language acquisition follows consistent developmental stages across cultures.",
        "The printing press transformed knowledge dissemination after 1440.",
        "Watson and Crick revealed the double helix structure of DNA in 1953.",
        "The steam engine powered the Industrial Revolution in the late 1700s.",
        "Cognitive dissonance describes discomfort from holding contradictory beliefs.",
        "Feudalism organized medieval society through mutual obligations.",
    ],
    "07_slides": [
        "Revenue grew 23% year-over-year with expansion into three new markets.",
        "Team grew from 50 to 120 employees in the past twelve months.",
        "Customer satisfaction improved from 4.2 to 4.7 out of 5.0.",
        "The platform processes over 2 million requests daily at 99.97% uptime.",
        "Next quarter: mobile app launch and SOC 2 certification.",
        "Total addressable market is $8.5 billion, growing 15% annually.",
    ],
    "08_news": [
        "City officials announced a $2.4 billion transit infrastructure investment.",
        "The biotech firm reported promising Phase III trial results.",
        "The central bank held rates steady at 4.75% citing inflation pressures.",
        "Researchers unveiled a quantum processor prototype at the national lab.",
        "Voters in three states approved renewable energy ballot measures.",
        "The tech giant agreed to a $4.7 billion antitrust settlement.",
    ],
    "09_forms": [
        "Complete all sections in black or blue ink. Allow 4-6 weeks for review.",
        "List all current medications including dosage and frequency.",
        "Describe relevant work experience in reverse chronological order.",
        "Show complete work in the space provided for partial credit.",
        "Sign and date to certify all information is accurate and complete.",
        "Rate your satisfaction on a scale of 1 to 10.",
    ],
}

# Formatting density ranges per category (higher than SFT generator)
RL_CATEGORIES = {
    "00_text": {
        "css_templates": ["academic", "textbook", "modern"],
        "bold_density": (0.40, 0.60),
        "italic_density": (0.30, 0.50),
        "strike_density": (0.10, 0.25),
    },
    "01_tables": {
        "css_templates": ["technical", "financial", "compact"],
        "bold_density": (0.45, 0.65),
        "italic_density": (0.25, 0.45),
        "strike_density": (0.15, 0.30),
        "table_probability": 0.85,
    },
    "02_academic": {
        "css_templates": ["academic", "textbook"],
        "bold_density": (0.35, 0.55),
        "italic_density": (0.35, 0.55),
        "strike_density": (0.08, 0.20),
    },
    "03_financial": {
        "css_templates": ["financial", "compact"],
        "bold_density": (0.45, 0.65),
        "italic_density": (0.20, 0.40),
        "strike_density": (0.15, 0.30),
    },
    "04_legal": {
        "css_templates": ["academic", "compact", "financial"],
        "bold_density": (0.40, 0.60),
        "italic_density": (0.35, 0.55),
        "strike_density": (0.12, 0.25),
    },
    "05_technical": {
        "css_templates": ["technical", "modern", "compact"],
        "bold_density": (0.45, 0.65),
        "italic_density": (0.20, 0.35),
        "strike_density": (0.20, 0.35),
    },
    "06_books": {
        "css_templates": ["textbook", "academic", "book"],
        "bold_density": (0.35, 0.55),
        "italic_density": (0.40, 0.60),
        "strike_density": (0.05, 0.15),
    },
    "07_slides": {
        "css_templates": ["slides", "modern"],
        "bold_density": (0.50, 0.70),
        "italic_density": (0.20, 0.35),
        "strike_density": (0.10, 0.25),
        "list_probability": 0.7,
    },
    "08_news": {
        "css_templates": ["news", "compact", "modern"],
        "bold_density": (0.40, 0.60),
        "italic_density": (0.25, 0.45),
        "strike_density": (0.10, 0.25),
    },
    "09_forms": {
        "css_templates": ["compact", "technical", "forms"],
        "bold_density": (0.45, 0.65),
        "italic_density": (0.30, 0.50),
        "strike_density": (0.10, 0.25),
        "list_probability": 0.5,
    },
}

# Headings pool (shared across categories, short)
HEADINGS = [
    "Overview",
    "Summary",
    "Results",
    "Analysis",
    "Key Points",
    "Discussion",
    "Notes",
    "Details",
    "Findings",
    "Report",
    "Section A",
    "Section B",
    "Part 1",
    "Part 2",
    "Highlights",
]

# Formatted phrase pools for parenthetical insertions
BOLD_PHRASES = [
    "the key concept",
    "this result",
    "total revenue",
    "the main finding",
    "neural networks",
    "the algorithm",
    "our approach",
    "the model",
    "critical threshold",
    "the primary factor",
    "observed behavior",
    "important note",
    "key insight",
    "final answer",
]

ITALIC_PHRASES = [
    "see note below",
    "as described above",
    "in this context",
    "for reference",
    "emphasis added",
    "per the specification",
    "in vitro",
    "et al.",
    "de novo",
    "ad hoc",
]

STRIKE_PHRASES = [
    "previously reported",
    "initial estimate",
    "old calculation",
    "former value",
    "draft version",
    "original assumption",
    "deprecated",
    "removed in v2",
    "superseded",
]


# ---------------------------------------------------------------------------
# Formatting injection (higher density than SFT)
# ---------------------------------------------------------------------------


def inject_dense_formatting(
    text: str,
    rng: random.Random,
    bold_density: float = 0.5,
    italic_density: float = 0.4,
    strike_density: float = 0.2,
) -> str:
    """Inject formatting with higher density and more type combinations.

    Compared to the SFT generator, this uses:
    - Higher base probabilities for all formatting types
    - More frequent combination formatting (bold+italic, etc.)
    - Shorter gaps between formatted spans (1-2 words vs 1-4)

    Args:
        text: Plain text to format.
        rng: Random number generator.
        bold_density: Probability of bolding each span.
        italic_density: Probability of italicizing each span.
        strike_density: Probability of strikethrough on each span.

    Returns:
        Text with dense inline formatting markers.
    """
    words = text.split()
    if len(words) < 3:
        return text

    # Build word-index to char-position mapping
    word_positions = []
    search_start = 0
    for w in words:
        idx = text.index(w, search_start)
        word_positions.append((idx, idx + len(w)))
        search_start = idx + len(w)

    # Collect spans with higher density and shorter gaps
    candidates = []
    i = 0
    while i < len(words):
        span_len = rng.randint(1, 3)
        span_end = min(i + span_len, len(words))
        char_start = word_positions[i][0]
        char_end = word_positions[span_end - 1][1]

        is_bold = rng.random() < bold_density
        is_italic = rng.random() < italic_density
        is_strike = rng.random() < strike_density

        # Higher combo probability (15% chance of forcing a combo)
        combo_roll = rng.random()
        if combo_roll < 0.06:
            is_bold = is_italic = is_strike = True
        elif combo_roll < 0.15:
            pair = rng.choice(["bi", "bs", "is"])
            if pair == "bi":
                is_bold = is_italic = True
            elif pair == "bs":
                is_bold = is_strike = True
            else:
                is_italic = is_strike = True

        if is_bold or is_italic or is_strike:
            candidates.append((char_start, char_end, is_bold, is_italic, is_strike))

        # Shorter gaps between formatted spans (1-2 words)
        i = span_end + rng.randint(1, 2)

    # Apply from right to left
    for start, end, bold, italic, strike in reversed(candidates):
        span_text = text[start:end]
        result = span_text
        if bold and italic:
            result = f"***{result}***"
        elif bold:
            result = f"**{result}**"
        elif italic:
            result = f"*{result}*"
        if strike:
            result = f"~~{result}~~"
        text = text[:start] + result + text[end:]

    return text


# ---------------------------------------------------------------------------
# Short list generation
# ---------------------------------------------------------------------------


def _generate_short_list(rng: random.Random, paragraphs: list[str], numbered: bool = False) -> str:
    """Generate a short formatted list (2-4 items).

    Args:
        rng: Random number generator.
        paragraphs: Content pool.
        numbered: Use numbered list.

    Returns:
        Markdown list string.
    """
    num_items = rng.randint(2, 4)
    items = []
    label_pool = BOLD_PHRASES + ITALIC_PHRASES + STRIKE_PHRASES
    for i in range(num_items):
        label = rng.choice(label_pool)
        # Format the label
        fmt_roll = rng.random()
        if fmt_roll < 0.4:
            label = f"**{label}**"
        elif fmt_roll < 0.7:
            label = f"*{label}*"
        elif fmt_roll < 0.85:
            label = f"~~{label}~~"
        else:
            label = f"~~**{label}**~~"

        # Short description (truncate paragraph)
        desc = rng.choice(paragraphs)
        words = desc.split()
        desc = " ".join(words[: rng.randint(6, 12)])
        if not desc.endswith("."):
            desc += "."

        prefix = f"{i + 1}." if numbered else "-"
        items.append(f"{prefix} {label}: {desc}")
    return "\n".join(items)


# ---------------------------------------------------------------------------
# Page generation (short, formatting-dense)
# ---------------------------------------------------------------------------


def generate_rl_page(rng: random.Random, cat_name: str, cat_config: dict) -> str:
    """Generate a single short, formatting-dense page.

    Targets 128-256 tokens output length with high formatting marker density
    and balanced type distribution.

    Args:
        rng: Random number generator.
        cat_name: Category name (e.g. "00_text").
        cat_config: Category config dict from RL_CATEGORIES.

    Returns:
        Markdown string with dense inline formatting.
    """
    paragraphs = SHORT_PARAGRAPHS[cat_name]
    bold_range = cat_config["bold_density"]
    italic_range = cat_config["italic_density"]
    strike_range = cat_config["strike_density"]
    table_prob = cat_config.get("table_probability", 0.15)
    list_prob = cat_config.get("list_probability", 0.3)

    parts = []

    # Heading (always one, sometimes two)
    heading = rng.choice(HEADINGS)
    level = rng.choice(["#", "##"])
    parts.append(f"{level} {heading}")

    # 1-2 short paragraphs with dense formatting
    num_paras = rng.randint(1, 2)
    for _ in range(num_paras):
        para = rng.choice(paragraphs)

        para = inject_dense_formatting(
            para,
            rng,
            bold_density=rng.uniform(*bold_range),
            italic_density=rng.uniform(*italic_range),
            strike_density=rng.uniform(*strike_range),
        )

        # Parenthetical formatted phrase (higher probability than SFT)
        if rng.random() < 0.5:
            phrase_type = rng.choice(["bold", "italic", "strike"])
            if phrase_type == "bold":
                para += f" (**{rng.choice(BOLD_PHRASES)}**)"
            elif phrase_type == "italic":
                para += f" (*{rng.choice(ITALIC_PHRASES)}*)"
            else:
                para += f" (~~{rng.choice(STRIKE_PHRASES)}~~)"

        parts.append(para)

    # Optional second heading + content block
    if rng.random() < 0.4:
        heading2 = rng.choice(HEADINGS)
        parts.append(f"## {heading2}")

        # Table, list, or another paragraph
        roll = rng.random()
        if roll < table_prob:
            table = generate_table_markdown(
                rng,
                bold_density=rng.uniform(*bold_range),
                italic_density=rng.uniform(*italic_range),
                strike_density=rng.uniform(*strike_range),
            )
            parts.append(table)
        elif roll < table_prob + list_prob:
            numbered = rng.random() < 0.4
            parts.append(_generate_short_list(rng, paragraphs, numbered=numbered))
        else:
            para = rng.choice(paragraphs)
            para = inject_dense_formatting(
                para,
                rng,
                bold_density=rng.uniform(*bold_range),
                italic_density=rng.uniform(*italic_range),
                strike_density=rng.uniform(*strike_range),
            )
            parts.append(para)

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Generation pipeline
# ---------------------------------------------------------------------------


def generate_category(
    output_dir: str,
    cat_name: str,
    cat_config: dict,
    num_pages: int,
    eval_frac: float,
    augment: bool,
    seed: int,
    width: int = 800,
    height: int = 1100,
) -> dict:
    """Generate RL training data for a single category.

    Args:
        output_dir: Root output directory.
        cat_name: Category directory name.
        cat_config: Category config from RL_CATEGORIES.
        num_pages: Total pages for this category.
        eval_frac: Fraction for eval split.
        augment: Apply scan-like augmentations.
        seed: Random seed.
        width: Rendered page width.
        height: Rendered page height.

    Returns:
        Stats dict for this category.
    """
    from PIL import Image

    rng = random.Random(seed)
    num_eval = max(1, int(num_pages * eval_frac))
    num_train = num_pages - num_eval
    cat_stats = {"train": 0, "eval": 0, "formatting": {}}

    for split, count in [("train", num_train), ("eval", num_eval)]:
        images_dir = os.path.join(output_dir, cat_name, split, "images")
        markdown_dir = os.path.join(output_dir, cat_name, split, "markdown")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(markdown_dir, exist_ok=True)

        render_queue = []
        css_names = cat_config["css_templates"]

        for i in range(count):
            page_id = f"{cat_name}_{split}_{i:05d}"
            md_text = generate_rl_page(rng, cat_name, cat_config)

            md_path = os.path.join(markdown_dir, f"{page_id}.md")
            with open(md_path, "w") as f:
                f.write(md_text)

            css_name = rng.choice(css_names)
            css = CSS_TEMPLATES[css_name]
            html = markdown_to_html(md_text, css)
            img_path = os.path.join(images_dir, f"{page_id}.png")
            render_queue.append((html, img_path, page_id))

        if render_queue:
            render_batch(render_queue, width=width, height=height)

        if augment and render_queue:
            from tqdm import tqdm

            aug_rng = random.Random(
                seed + int(hashlib.md5(split.encode()).hexdigest(), 16) % (2**31)
            )
            for _, img_path, _ in tqdm(
                render_queue, desc=f"Augmenting {cat_name}/{split}", unit="pg"
            ):
                if os.path.exists(img_path):
                    img = Image.open(img_path).convert("RGB")
                    img = augment_image(img, aug_rng)
                    img.save(img_path)

        cat_stats[split] = count
        stats = count_formatting_stats(markdown_dir)
        cat_stats["formatting"][split] = stats
        logger.info(
            f"  {cat_name}/{split}: {count} pages, "
            f"bold={stats['bold']}, italic={stats['italic']}, "
            f"strike={stats['strikethrough']}"
        )

    return cat_stats


def generate_dataset(
    output_dir: str,
    num_pages: int,
    categories: list[str] | None = None,
    eval_frac: float = 0.05,
    augment: bool = False,
    seed: int = 42,
    width: int = 800,
    height: int = 1100,
):
    """Generate a full RL formatting dataset.

    Args:
        output_dir: Root output directory.
        num_pages: Pages per category.
        categories: Category names to generate (None = all).
        eval_frac: Fraction for eval split.
        augment: Apply scan-like augmentations.
        seed: Random seed.
        width: Rendered page width.
        height: Rendered page height.
    """
    cats_to_gen = categories or list(RL_CATEGORIES.keys())
    all_stats = {}
    total_pages = 0

    logger.info(
        f"Generating RL dataset: {num_pages} pages x {len(cats_to_gen)} categories "
        f"= {num_pages * len(cats_to_gen)} total"
    )
    logger.info(f"Categories: {', '.join(cats_to_gen)}")

    for cat_name in cats_to_gen:
        if cat_name not in RL_CATEGORIES:
            logger.warning(f"Unknown category '{cat_name}', skipping")
            continue

        cat_config = RL_CATEGORIES[cat_name]
        logger.info(f"Generating {cat_name}")

        cat_seed = seed + int(hashlib.md5(cat_name.encode()).hexdigest(), 16) % (2**31)
        stats = generate_category(
            output_dir,
            cat_name,
            cat_config,
            num_pages=num_pages,
            eval_frac=eval_frac,
            augment=augment,
            seed=cat_seed,
            width=width,
            height=height,
        )
        all_stats[cat_name] = stats
        total_pages += stats["train"] + stats["eval"]

    metadata = {
        "dataset_type": "rl",
        "total_pages": total_pages,
        "pages_per_category": num_pages,
        "categories": list(cats_to_gen),
        "eval_frac": eval_frac,
        "augmented": augment,
        "seed": seed,
        "render_size": f"{width}x{height}",
        "stats": all_stats,
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Done. {total_pages} total pages in {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate short, formatting-dense data for GRPO/RL training"
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Root output directory")
    parser.add_argument(
        "--num_pages", type=int, default=150, help="Pages per category (default: 150, total = 1500)"
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="*",
        default=None,
        help=f"Categories to generate (default: all). Options: {', '.join(RL_CATEGORIES.keys())}",
    )
    parser.add_argument(
        "--eval_frac", type=float, default=0.05, help="Fraction for eval split (default: 0.05)"
    )
    parser.add_argument("--augment", action="store_true", help="Apply scan-like augmentations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--width", type=int, default=800, help="Rendered page width in pixels")
    parser.add_argument("--height", type=int, default=1100, help="Rendered page height in pixels")
    args = parser.parse_args()

    generate_dataset(
        output_dir=args.output_dir,
        num_pages=args.num_pages,
        categories=args.categories,
        eval_frac=args.eval_frac,
        augment=args.augment,
        seed=args.seed,
        width=args.width,
        height=args.height,
    )


if __name__ == "__main__":
    main()
