"""Shared utilities for OCR data loading and processing.

Common constants and functions used across OCR dataset loaders,
conversion scripts, and evaluation.
"""

from pathlib import Path

# Shared system prompt for all OCR tasks (training and eval)
OCR_SYSTEM_PROMPT = (
    "Below is an image of a document page. Extract all text content "
    "and output it in clean markdown format, preserving the document's "
    "structure (headings, lists, tables) and inline formatting "
    "(bold, italic, strikethrough) exactly as shown."
)


def render_pdf_page(pdf_path: str, page_num: int, dpi: int = 150):
    """Render a single PDF page to a PIL Image.

    Args:
        pdf_path: Path to PDF file.
        page_num: 1-indexed page number.
        dpi: Render resolution.

    Returns:
        PIL Image of the rendered page.
    """
    import pypdfium2

    pdf_doc = pypdfium2.PdfDocument(pdf_path)
    if page_num < 1 or page_num > len(pdf_doc):
        pdf_doc.close()
        raise ValueError(f"Page {page_num} out of range for {pdf_path} ({len(pdf_doc)} pages)")
    page = pdf_doc[page_num - 1]
    bitmap = page.render(scale=dpi / 72)
    image = bitmap.to_pil().convert("RGB")
    pdf_doc.close()
    return image


def pdf_image_name(pdf_rel: str, page_num: int) -> str:
    """Generate a flat image filename from a relative PDF path and page number.

    Args:
        pdf_rel: Relative PDF path, e.g. "arxiv_cs/pdf_00001_page3.pdf"
        page_num: 1-indexed page number.

    Returns:
        Flat filename like "arxiv_cs_pdf_00001_page3_p1.jpg"
    """
    category = Path(pdf_rel).parts[0] if "/" in pdf_rel else ""
    stem = Path(pdf_rel).stem
    return f"{category}_{stem}_p{page_num}.jpg" if category else f"{stem}_p{page_num}.jpg"
