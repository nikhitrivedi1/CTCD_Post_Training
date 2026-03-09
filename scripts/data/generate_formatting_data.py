"""Generate synthetic formatting-rich training data for OCR models.

Creates (image, markdown_gt) pairs where images contain visually formatted
text (bold, italic, strikethrough) rendered via headless Chromium, and the
ground truth is the exact markdown used to generate the HTML.

Pipeline:
    Markdown GT -> HTML+CSS -> Playwright render -> PNG -> Augmentation -> Final image

Usage:
    # Generate 1000 pages with default settings
    python scripts/data/generate_formatting_data.py --output_dir data/ocr/formatting-synth --num_pages 1000

    # Generate with augmentation
    python scripts/data/generate_formatting_data.py --output_dir data/ocr/formatting-synth --num_pages 1000 --augment

    # Quick test (10 pages, no augmentation)
    python scripts/data/generate_formatting_data.py --output_dir data/ocr/formatting-synth --num_pages 10

    # Custom content source
    python scripts/data/generate_formatting_data.py --output_dir data/ocr/formatting-synth --num_pages 500 --content_source wikitext
"""

import argparse
import json
import logging
import math
import os
import random
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

# ---------------------------------------------------------------------------
# Content pools - paragraphs to draw from
# ---------------------------------------------------------------------------

# Diverse paragraph pool covering different document domains.
# These serve as base content; formatting is injected on top.
PARAGRAPH_POOL = [
    # Academic / textbook
    "Machine learning algorithms can be broadly categorized into supervised, unsupervised, and reinforcement learning paradigms. Supervised learning requires labeled training data, where each input is paired with the desired output. The model learns a mapping function from inputs to outputs by minimizing a loss function over the training set.",
    "The transformer architecture introduced self-attention as a replacement for recurrence, enabling parallel processing of sequence elements. Each attention head computes query, key, and value projections, then uses scaled dot-product attention to aggregate information across positions.",
    "Gradient descent is the foundational optimization algorithm used in training neural networks. The basic idea is to iteratively adjust model parameters in the direction that reduces the loss function. Variants include stochastic gradient descent, Adam, and AdaGrad, each with different properties.",
    "The central limit theorem states that the sampling distribution of the sample mean approaches a normal distribution as the sample size increases, regardless of the population distribution shape. This theorem is fundamental to inferential statistics and hypothesis testing.",
    "In organic chemistry, functional groups determine the chemical properties of molecules. Common functional groups include hydroxyl, carbonyl, carboxyl, and amino groups. Each functional group exhibits characteristic reactivity patterns in chemical reactions.",
    # Financial / business
    "Revenue for the fiscal year increased by 12.3% compared to the prior year, driven primarily by strong performance in the cloud services segment. Operating margins improved to 28.4%, reflecting cost optimization initiatives implemented in the third quarter.",
    "The board of directors approved a quarterly dividend of $0.85 per share, payable to shareholders of record as of March 15. This represents a 5% increase from the previous quarter and reflects the company's strong cash flow generation.",
    "Risk factors include market volatility, regulatory changes, and competitive pressures in key segments. The company maintains a hedging strategy for foreign currency exposure across its international operations.",
    # Legal / policy
    "Under Section 4.2 of the agreement, the licensee shall maintain comprehensive insurance coverage for the duration of the contract. Any modifications to the terms must be submitted in writing and require mutual consent of both parties.",
    "The privacy policy outlines how personal information is collected, used, and protected. Users may request deletion of their data at any time by submitting a written request to the data protection officer.",
    # Technical / engineering
    "The system architecture employs a microservices pattern with event-driven communication between components. Each service maintains its own database to ensure loose coupling and independent deployability.",
    "Memory allocation in the kernel follows a buddy system algorithm, splitting and coalescing blocks of power-of-two sizes. This approach minimizes external fragmentation while maintaining reasonable allocation speed.",
    "The API endpoint accepts JSON payloads with a maximum size of 10MB. Rate limiting is enforced at 1000 requests per minute per API key. All responses include standard pagination headers for list endpoints.",
    # Medical / scientific
    "Clinical trials demonstrated a statistically significant reduction in symptom severity compared to the placebo group. The primary endpoint was achieved with a p-value of less than 0.001 across all treatment arms.",
    "The protein folding problem involves predicting the three-dimensional structure of a protein from its amino acid sequence. Recent advances in deep learning have dramatically improved prediction accuracy.",
    # General / educational
    "The Renaissance period marked a cultural rebirth in Europe, characterized by renewed interest in classical art, literature, and philosophy. Key figures include Leonardo da Vinci, Michelangelo, and Raphael.",
    "Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen. This process occurs primarily in the chloroplasts of leaf cells.",
    "The water cycle describes the continuous movement of water through the Earth's systems. Evaporation from oceans and lakes produces water vapor, which condenses into clouds and eventually returns as precipitation.",
]

# Heading pool
HEADING_POOL = [
    "Introduction", "Background", "Methods", "Results", "Discussion",
    "Conclusion", "Overview", "Analysis", "Summary", "Key Findings",
    "Technical Approach", "Implementation Details", "Experimental Setup",
    "Data Processing", "Performance Evaluation", "Related Work",
    "System Architecture", "Design Principles", "Risk Assessment",
    "Financial Summary", "Quarterly Results", "Market Analysis",
    "Literature Review", "Theoretical Framework", "Case Study",
    "Recommendations", "Future Directions", "Appendix",
]

# Words/phrases to apply formatting to
BOLD_CANDIDATES = [
    "important", "critical", "key", "significant", "primary", "essential",
    "Note", "Warning", "Required", "Mandatory", "Definition", "Theorem",
    "Result", "Conclusion", "Summary", "Total", "Net Income", "Revenue",
    "machine learning", "neural network", "gradient descent", "attention",
    "Section", "Chapter", "Figure", "Table", "Algorithm", "Equation",
]

ITALIC_CANDIDATES = [
    "et al.", "in vivo", "in vitro", "de novo", "ad hoc", "per se",
    "emphasis", "see also", "note", "i.e.", "e.g.", "cf.", "ibid.",
    "The Art of War", "Origin of Species", "Principia Mathematica",
    "approximately", "typically", "generally", "specifically",
]

STRIKETHROUGH_CANDIDATES = [
    "deprecated", "removed", "obsolete", "outdated", "replaced",
    "old value", "previous version", "former approach", "draft text",
    "incorrect assumption", "initial estimate", "preliminary data",
]


# ---------------------------------------------------------------------------
# Markdown generation
# ---------------------------------------------------------------------------

def _wrap_formatted(span_text: str, bold: bool, italic: bool, strike: bool) -> str:
    """Wrap text with the appropriate markdown formatting markers.

    Nesting order (outer to inner): strikethrough > bold > italic.
    Combined examples: ***bold italic***, ~~**bold strike**~~,
    ~~***all three***~~.

    Args:
        span_text: Plain text to format.
        bold: Apply bold.
        italic: Apply italic.
        strike: Apply strikethrough.

    Returns:
        Formatted text string.
    """
    result = span_text
    if bold and italic:
        result = f"***{result}***"
    elif bold:
        result = f"**{result}**"
    elif italic:
        result = f"*{result}*"
    if strike:
        result = f"~~{result}~~"
    return result


def inject_formatting(text: str, rng: random.Random,
                      bold_density: float = 0.4,
                      italic_density: float = 0.3,
                      strike_density: float = 0.15) -> str:
    """Inject bold, italic, and strikethrough formatting in a single pass.

    Uses a span-based approach: identifies word-boundary positions, assigns
    formatting types (which can be combined on the same span), then applies
    all at once from right-to-left to avoid index shifting. This prevents
    overlapping markers by design.

    Args:
        text: Plain text paragraph.
        rng: Random number generator.
        bold_density: Probability of bolding each eligible span.
        italic_density: Probability of italicizing each eligible span.
        strike_density: Probability of strikethrough on each eligible span.

    Returns:
        Text with **bold**, *italic*, ~~strike~~, and combined markers.
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

    # Collect non-overlapping candidate spans with independent formatting flags
    candidates = []
    i = 0
    while i < len(words):
        span_len = rng.randint(1, 3)
        span_end = min(i + span_len, len(words))
        char_start = word_positions[i][0]
        char_end = word_positions[span_end - 1][1]
        span_text = text[char_start:char_end].lower()

        # Each formatting type is decided independently
        is_bold = False
        is_italic = False
        is_strike = False

        # Candidate-list matching (boosted probability)
        for phrase in BOLD_CANDIDATES:
            if phrase.lower() in span_text:
                is_bold = rng.random() < bold_density
                break
        for phrase in ITALIC_CANDIDATES:
            if phrase.lower() in span_text:
                is_italic = rng.random() < italic_density
                break
        for phrase in STRIKETHROUGH_CANDIDATES:
            if phrase.lower() in span_text:
                is_strike = rng.random() < strike_density
                break

        # Random assignment (lower probability, types are independent)
        if not is_bold:
            is_bold = rng.random() < bold_density * 0.25
        if not is_italic:
            is_italic = rng.random() < italic_density * 0.25
        if not is_strike:
            is_strike = rng.random() < strike_density * 0.25

        # Boost: small chance of forcing combined formatting
        combo_roll = rng.random()
        if combo_roll < 0.03:  # ~3% chance of all three
            is_bold = is_italic = is_strike = True
        elif combo_roll < 0.08:  # ~5% chance of double
            pair = rng.choice(["bi", "bs", "is"])
            if pair == "bi":
                is_bold = is_italic = True
            elif pair == "bs":
                is_bold = is_strike = True
            else:
                is_italic = is_strike = True

        if is_bold or is_italic or is_strike:
            candidates.append((char_start, char_end, is_bold, is_italic, is_strike))

        # Skip ahead (gap between formatted spans)
        i = span_end + rng.randint(1, 4)

    # Apply formatting from right to left (preserves positions)
    for start, end, bold, italic, strike in reversed(candidates):
        span_text = text[start:end]
        text = text[:start] + _wrap_formatted(span_text, bold, italic, strike) + text[end:]

    return text


def _random_formatted_phrase(rng: random.Random, fmt: str = "bold") -> str:
    """Generate a random short phrase with formatting.

    Args:
        rng: Random number generator.
        fmt: "bold", "italic", or "strikethrough".

    Returns:
        A formatted phrase.
    """
    bold_phrases = ["the key concept", "this result", "total revenue", "the main finding",
                    "neural networks", "the algorithm", "our approach", "the model",
                    "critical threshold", "the primary factor", "observed behavior"]
    italic_phrases = ["see note below", "as described above", "in this context",
                      "for reference", "emphasis added", "per the specification",
                      "the underlying assumption", "this distinction"]
    strike_phrases = ["previously reported", "initial estimate", "old calculation",
                      "former value", "draft version", "original assumption",
                      "earlier finding", "prior methodology"]

    if fmt == "bold":
        return f"**{rng.choice(bold_phrases)}**"
    elif fmt == "italic":
        return f"*{rng.choice(italic_phrases)}*"
    else:
        return f"~~{rng.choice(strike_phrases)}~~"


def generate_page_markdown(rng: random.Random, page_id: int) -> str:
    """Generate a single page of formatting-rich markdown.

    Creates a document page with headings, paragraphs, and injected
    formatting (bold, italic, strikethrough).

    Args:
        rng: Random number generator for reproducibility.
        page_id: Unique page identifier (for variety).

    Returns:
        Markdown string with inline formatting.
    """
    parts = []
    num_sections = rng.randint(2, 4)

    for s in range(num_sections):
        # Heading (sometimes bold within heading)
        heading = rng.choice(HEADING_POOL)
        if rng.random() < 0.3:
            level = rng.choice(["##", "###"])
        else:
            level = "#"
        parts.append(f"{level} {heading}")

        # 2-3 paragraphs per section
        num_paras = rng.randint(2, 3)
        for _ in range(num_paras):
            para = rng.choice(PARAGRAPH_POOL)

            # Single-pass formatting injection with varying densities
            para = inject_formatting(
                para, rng,
                bold_density=rng.uniform(0.3, 0.5),
                italic_density=rng.uniform(0.2, 0.4),
                strike_density=rng.uniform(0.05, 0.2),
            )

            # Insert extra formatted parenthetical phrases
            sentences = para.split(". ")
            if len(sentences) > 1:
                if rng.random() < 0.4:
                    idx = rng.randint(0, len(sentences) - 1)
                    sentences[idx] += f" ({_random_formatted_phrase(rng, 'bold')})"
                if rng.random() < 0.4:
                    idx = rng.randint(0, len(sentences) - 1)
                    sentences[idx] += f" ({_random_formatted_phrase(rng, 'italic')})"
                if rng.random() < 0.25:
                    idx = rng.randint(0, len(sentences) - 1)
                    sentences[idx] += f" ({_random_formatted_phrase(rng, 'strikethrough')})"
                para = ". ".join(sentences)

            parts.append(para)

        # Sometimes add a list with formatted items
        if rng.random() < 0.35:
            num_items = rng.randint(2, 5)
            items = []
            for i in range(num_items):
                item_text = rng.choice(BOLD_CANDIDATES + ITALIC_CANDIDATES)
                fmt_roll = rng.random()
                if fmt_roll < 0.4:
                    item_text = f"**{item_text}**"
                elif fmt_roll < 0.65:
                    item_text = f"*{item_text}*"
                elif fmt_roll < 0.75:
                    item_text = f"~~{item_text}~~"
                items.append(f"- {item_text}: {rng.choice(PARAGRAPH_POOL)[:80]}")
            parts.append("\n".join(items))

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# CSS templates for visual diversity
# ---------------------------------------------------------------------------

CSS_TEMPLATES = [
    # Academic paper style
    {
        "name": "academic",
        "css": """
            body { font-family: 'Times New Roman', serif; font-size: 12pt;
                   line-height: 1.6; max-width: 700px; margin: 40px auto;
                   color: #1a1a1a; background: #fff; padding: 40px; }
            h1 { font-size: 18pt; margin-top: 24pt; }
            h2 { font-size: 15pt; margin-top: 20pt; }
            h3 { font-size: 13pt; margin-top: 16pt; }
        """,
    },
    # Textbook style
    {
        "name": "textbook",
        "css": """
            body { font-family: 'Georgia', serif; font-size: 11pt;
                   line-height: 1.5; max-width: 650px; margin: 30px auto;
                   color: #2c2c2c; background: #fafaf5; padding: 35px; }
            h1 { font-size: 20pt; color: #1a3c5e; border-bottom: 2px solid #1a3c5e; padding-bottom: 6px; }
            h2 { font-size: 16pt; color: #2a5c8e; }
            h3 { font-size: 13pt; color: #3a7cae; }
        """,
    },
    # Technical report style
    {
        "name": "technical",
        "css": """
            body { font-family: 'Arial', 'Helvetica', sans-serif; font-size: 10pt;
                   line-height: 1.45; max-width: 680px; margin: 25px auto;
                   color: #333; background: #fff; padding: 30px; }
            h1 { font-size: 16pt; font-weight: 700; }
            h2 { font-size: 13pt; font-weight: 700; }
            h3 { font-size: 11pt; font-weight: 600; }
        """,
    },
    # Financial report style
    {
        "name": "financial",
        "css": """
            body { font-family: 'Calibri', 'Segoe UI', sans-serif; font-size: 10.5pt;
                   line-height: 1.4; max-width: 720px; margin: 30px auto;
                   color: #1a1a1a; background: #fff; padding: 35px; }
            h1 { font-size: 18pt; color: #003366; }
            h2 { font-size: 14pt; color: #004488; }
            h3 { font-size: 12pt; color: #005599; }
        """,
    },
    # Modern / clean style
    {
        "name": "modern",
        "css": """
            body { font-family: 'Helvetica Neue', 'Arial', sans-serif; font-size: 11pt;
                   line-height: 1.55; max-width: 660px; margin: 35px auto;
                   color: #222; background: #fefefe; padding: 40px; }
            h1 { font-size: 22pt; font-weight: 300; letter-spacing: -0.5px; }
            h2 { font-size: 16pt; font-weight: 400; }
            h3 { font-size: 13pt; font-weight: 500; }
        """,
    },
    # Dense / compact style (like a reference manual)
    {
        "name": "compact",
        "css": """
            body { font-family: 'Verdana', sans-serif; font-size: 9pt;
                   line-height: 1.35; max-width: 750px; margin: 20px auto;
                   color: #1c1c1c; background: #fff; padding: 25px; }
            h1 { font-size: 14pt; margin-top: 18pt; }
            h2 { font-size: 12pt; margin-top: 14pt; }
            h3 { font-size: 10pt; margin-top: 10pt; }
        """,
    },
]


def markdown_to_html(md_text: str, css: str) -> str:
    """Convert markdown text to styled HTML for rendering.

    Handles bold, italic, strikethrough, headings, lists, and paragraphs.
    Does NOT use a full markdown parser to keep things simple and predictable.

    Args:
        md_text: Markdown text with formatting.
        css: CSS stylesheet string.

    Returns:
        Complete HTML document string.
    """
    lines = md_text.split("\n")
    html_parts = []
    in_list = False

    for line in lines:
        stripped = line.strip()

        if not stripped:
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            continue

        # Headings
        if stripped.startswith("### "):
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            html_parts.append(f"<h3>{_inline_format(stripped[4:])}</h3>")
        elif stripped.startswith("## "):
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            html_parts.append(f"<h2>{_inline_format(stripped[3:])}</h2>")
        elif stripped.startswith("# "):
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            html_parts.append(f"<h1>{_inline_format(stripped[2:])}</h1>")
        # List items
        elif stripped.startswith("- "):
            if not in_list:
                html_parts.append("<ul>")
                in_list = True
            html_parts.append(f"<li>{_inline_format(stripped[2:])}</li>")
        # Regular paragraph
        else:
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            html_parts.append(f"<p>{_inline_format(stripped)}</p>")

    if in_list:
        html_parts.append("</ul>")

    body = "\n".join(html_parts)
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>{css}</style></head>
<body>{body}</body></html>"""


def _inline_format(text: str) -> str:
    """Convert markdown inline formatting to HTML.

    Handles combined markers like ***bold+italic*** and ~~**bold+strike**~~.

    Args:
        text: Text with **bold**, *italic*, ***bold+italic***, ~~strike~~ markers.

    Returns:
        Text with <b>, <i>, <s> HTML tags.
    """
    # Strikethrough first (outermost): ~~text~~ -> <s>text</s>
    text = re.sub(r"~~(.+?)~~", r"<s>\1</s>", text)
    # Bold+Italic: ***text*** -> <b><i>text</i></b>
    text = re.sub(r"\*\*\*(.+?)\*\*\*", r"<b><i>\1</i></b>", text)
    # Bold: **text** -> <b>text</b>
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    # Italic: *text* -> <i>text</i>
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", text)
    return text


# ---------------------------------------------------------------------------
# Rendering via Playwright
# ---------------------------------------------------------------------------

def render_html_to_image(html: str, output_path: str, width: int = 800, height: int = 1100):
    """Render HTML to a PNG image using headless Chromium.

    Args:
        html: Complete HTML document string.
        output_path: Path to save the PNG.
        width: Viewport width in pixels.
        height: Viewport height in pixels.
    """
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": width, "height": height})
        page.set_content(html, wait_until="networkidle")
        page.screenshot(path=output_path, full_page=False)
        browser.close()


def render_batch(pages: list[tuple[str, str, str]], width: int = 800, height: int = 1100):
    """Render multiple HTML pages to images using a single browser instance.

    Args:
        pages: List of (html, output_path, page_id) tuples.
        width: Viewport width.
        height: Viewport height.
    """
    from playwright.sync_api import sync_playwright
    from tqdm import tqdm

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": width, "height": height})

        for html, output_path, page_id in tqdm(pages, desc="Rendering", unit="pg"):
            try:
                page.set_content(html, wait_until="networkidle")
                page.screenshot(path=output_path, full_page=False)
            except Exception as e:
                logger.error(f"Failed to render page {page_id}: {e}")

        browser.close()


# ---------------------------------------------------------------------------
# Image augmentation (scan simulation)
# ---------------------------------------------------------------------------

def augment_image(image, rng: random.Random):
    """Apply scan-like augmentations to a rendered image.

    Simulates real-world document scanning artifacts: noise, rotation,
    JPEG compression, brightness variation.

    Args:
        image: PIL Image.
        rng: Random number generator.

    Returns:
        Augmented PIL Image.
    """
    from PIL import Image, ImageEnhance, ImageFilter
    import numpy as np

    img = image.copy()

    # Slight rotation (0-2 degrees)
    if rng.random() < 0.5:
        angle = rng.uniform(-2, 2)
        img = img.rotate(angle, fillcolor=(255, 255, 255), expand=False)

    # Brightness variation
    if rng.random() < 0.4:
        factor = rng.uniform(0.85, 1.05)
        img = ImageEnhance.Brightness(img).enhance(factor)

    # Contrast variation
    if rng.random() < 0.3:
        factor = rng.uniform(0.9, 1.1)
        img = ImageEnhance.Contrast(img).enhance(factor)

    # Gaussian blur (simulates slightly out-of-focus scan)
    if rng.random() < 0.2:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 0.8)))

    # Add noise
    if rng.random() < 0.3:
        arr = np.array(img, dtype=np.float32)
        noise = np.random.normal(0, rng.uniform(2, 6), arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    # JPEG compression artifact simulation
    if rng.random() < 0.3:
        import io
        quality = rng.randint(60, 85)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

    # Grayscale conversion (some scanned docs are grayscale)
    if rng.random() < 0.15:
        img = img.convert("L").convert("RGB")

    return img


# ---------------------------------------------------------------------------
# Content sources
# ---------------------------------------------------------------------------

def load_wikitext_paragraphs(max_paragraphs: int = 500) -> list[str]:
    """Load paragraphs from wikitext dataset.

    Args:
        max_paragraphs: Maximum paragraphs to load.

    Returns:
        List of paragraph strings.
    """
    try:
        import datasets
        ds = datasets.load_dataset("wikitext", "wikitext-103-raw-v1",
                                   split="train", streaming=True)
        paragraphs = []
        for sample in ds:
            text = sample["text"].strip()
            # Filter for substantial paragraphs
            if len(text) > 100 and not text.startswith("="):
                paragraphs.append(text)
                if len(paragraphs) >= max_paragraphs:
                    break
        logger.info(f"Loaded {len(paragraphs)} paragraphs from wikitext")
        return paragraphs
    except Exception as e:
        logger.warning(f"Could not load wikitext: {e}. Using built-in pool.")
        return []


# ---------------------------------------------------------------------------
# Main generation pipeline
# ---------------------------------------------------------------------------

def generate_dataset(output_dir: str, num_pages: int, augment: bool = False,
                     content_source: str = "builtin", seed: int = 42,
                     width: int = 800, height: int = 1100):
    """Generate a full synthetic formatting dataset.

    Args:
        output_dir: Output directory (will contain images/ and markdown/ subdirs).
        num_pages: Number of pages to generate.
        augment: Whether to apply scan-like augmentations.
        content_source: "builtin" or "wikitext".
        seed: Random seed for reproducibility.
        width: Rendered page width in pixels.
        height: Rendered page height in pixels.
    """
    from PIL import Image

    rng = random.Random(seed)
    images_dir = os.path.join(output_dir, "images")
    markdown_dir = os.path.join(output_dir, "markdown")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(markdown_dir, exist_ok=True)

    # Optionally extend paragraph pool
    global PARAGRAPH_POOL
    if content_source == "wikitext":
        extra = load_wikitext_paragraphs(500)
        if extra:
            PARAGRAPH_POOL = PARAGRAPH_POOL + extra

    # Generate markdown content
    logger.info(f"Generating {num_pages} pages (content_source={content_source}, augment={augment})")
    render_queue = []

    for i in range(num_pages):
        page_id = f"fmt_{i:05d}"
        md_text = generate_page_markdown(rng, i)

        # Save markdown GT
        md_path = os.path.join(markdown_dir, f"{page_id}.md")
        with open(md_path, "w") as f:
            f.write(md_text)

        # Pick a random CSS template
        template = rng.choice(CSS_TEMPLATES)

        # Small random variations to width/height
        w = width + rng.randint(-30, 30)
        h = height + rng.randint(-50, 50)

        html = markdown_to_html(md_text, template["css"])
        img_path = os.path.join(images_dir, f"{page_id}.png")
        render_queue.append((html, img_path, page_id))

    # Render all pages
    render_batch(render_queue, width=width, height=height)

    # Apply augmentation if requested
    if augment:
        logger.info("Applying augmentations...")
        from tqdm import tqdm
        aug_rng = random.Random(seed + 1)
        for _, img_path, _ in tqdm(render_queue, desc="Augmenting", unit="pg"):
            if os.path.exists(img_path):
                img = Image.open(img_path).convert("RGB")
                img = augment_image(img, aug_rng)
                img.save(img_path)

    # Save metadata
    metadata = {
        "num_pages": num_pages,
        "content_source": content_source,
        "augmented": augment,
        "seed": seed,
        "css_templates": [t["name"] for t in CSS_TEMPLATES],
        "paragraph_pool_size": len(PARAGRAPH_POOL),
        "render_size": f"{width}x{height}",
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Quick stats on generated formatting
    total_bold = 0
    total_italic = 0
    total_strike = 0
    for md_file in Path(markdown_dir).glob("*.md"):
        text = md_file.read_text()
        total_bold += len(re.findall(r"\*\*(.+?)\*\*", text))
        total_italic += len(re.findall(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", text))
        total_strike += len(re.findall(r"~~(.+?)~~", text))

    logger.info(f"Generated {num_pages} pages to {output_dir}")
    logger.info(f"Formatting stats: bold={total_bold}, italic={total_italic}, strikethrough={total_strike}")
    logger.info(f"Avg per page: bold={total_bold/num_pages:.1f}, italic={total_italic/num_pages:.1f}, strike={total_strike/num_pages:.1f}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic formatting training data")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for images/ and markdown/")
    parser.add_argument("--num_pages", type=int, default=1000,
                        help="Number of pages to generate")
    parser.add_argument("--augment", action="store_true",
                        help="Apply scan-like augmentations")
    parser.add_argument("--content_source", type=str, default="builtin",
                        choices=["builtin", "wikitext"],
                        help="Content source for paragraphs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--width", type=int, default=800,
                        help="Rendered page width in pixels")
    parser.add_argument("--height", type=int, default=1100,
                        help="Rendered page height in pixels")
    args = parser.parse_args()

    generate_dataset(
        output_dir=args.output_dir,
        num_pages=args.num_pages,
        augment=args.augment,
        content_source=args.content_source,
        seed=args.seed,
        width=args.width,
        height=args.height,
    )


if __name__ == "__main__":
    main()
