"""Generate synthetic formatting-rich training data for OCR models.

Creates (image, markdown_gt) pairs organized into numbered category subsets,
following the olmOCR-mix-1025 convention. Each category has domain-specific
content, formatting patterns, and visual styles.

Pipeline:
    Markdown GT -> HTML+CSS -> Playwright render -> PNG -> Augmentation

Output structure (matches load_omnidocbench_rl expectations):
    output_dir/
      00_text/       train/{images,markdown}/  eval/{images,markdown}/
      01_tables/     train/{images,markdown}/  eval/{images,markdown}/
      02_academic/   train/{images,markdown}/  eval/{images,markdown}/
      03_financial/  train/{images,markdown}/  eval/{images,markdown}/
      04_legal/      train/{images,markdown}/  eval/{images,markdown}/
      05_technical/  train/{images,markdown}/  eval/{images,markdown}/
      06_books/      train/{images,markdown}/  eval/{images,markdown}/
      07_slides/     train/{images,markdown}/  eval/{images,markdown}/
      08_news/       train/{images,markdown}/  eval/{images,markdown}/
      09_forms/      train/{images,markdown}/  eval/{images,markdown}/
      metadata.json

Usage:
    # Generate all categories (200 pages each, 2000 total)
    python scripts/data/generate_formatting_data.py \\
        --output_dir data/ocr/formatting-synth --num_pages 200 --augment

    # Single category
    python scripts/data/generate_formatting_data.py \\
        --output_dir data/ocr/formatting-synth --num_pages 500 --categories 02_academic

    # Quick test
    python scripts/data/generate_formatting_data.py \\
        --output_dir /tmp/fmt_test --num_pages 10

    # Custom eval fraction
    python scripts/data/generate_formatting_data.py \\
        --output_dir data/ocr/formatting-synth --num_pages 1000 --eval_frac 0.1
"""

import argparse
import json
import logging
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
# Category definitions
# ---------------------------------------------------------------------------

CATEGORIES = {
    "00_text": {
        "description": "General prose with mixed inline formatting",
        "paragraphs": [
            "The water cycle describes the continuous movement of water through the Earth's systems. Evaporation from oceans and lakes produces water vapor, which condenses into clouds and eventually returns as precipitation.",
            "Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen. This process occurs primarily in the chloroplasts of leaf cells.",
            "The Renaissance period marked a cultural rebirth in Europe, characterized by renewed interest in classical art, literature, and philosophy. Key figures include Leonardo da Vinci, Michelangelo, and Raphael.",
            "Plate tectonics describes how the outer shell of the Earth is divided into several plates that glide over the mantle. The movement of these plates causes earthquakes, volcanic activity, and the formation of mountain ranges.",
            "Democracy originated in ancient Athens, where citizens participated directly in decision-making. Modern democracies use representative systems where elected officials govern on behalf of the people.",
            "The human brain contains approximately 86 billion neurons, each connected to thousands of others through synapses. This vast network enables consciousness, memory, language, and all cognitive functions.",
            "Climate change refers to long-term shifts in global temperatures and weather patterns. While natural causes contribute, human activities have been the primary driver of climate change since the industrial era.",
            "The periodic table organizes chemical elements by atomic number and electron configuration. Elements in the same column share similar chemical properties due to their outer electron arrangements.",
        ],
        "headings": [
            "Introduction", "Overview", "Background", "Discussion",
            "Summary", "Analysis", "Key Points", "Conclusion",
        ],
        "css_templates": ["academic", "textbook", "modern"],
        "bold_density": (0.25, 0.45),
        "italic_density": (0.15, 0.35),
        "strike_density": (0.05, 0.15),
    },
    "01_tables": {
        "description": "Documents with formatted tables and surrounding text",
        "paragraphs": [
            "The following table summarizes the quarterly performance metrics across all divisions. Figures marked in bold indicate values exceeding the target threshold.",
            "Results are presented in the table below, with italic entries representing estimated values subject to revision in the final audit.",
            "Comparative analysis of the three approaches is shown below. Strikethrough values indicate deprecated measurements replaced by updated methodology.",
            "The benchmark results demonstrate consistent improvement across all model sizes. Bold entries highlight the best-performing configuration for each metric.",
            "Survey responses are aggregated by demographic category in the table below. Sample sizes vary across groups as noted in the footnotes.",
            "Performance comparison across different hardware configurations. All measurements averaged over three independent runs with standard deviation reported.",
        ],
        "headings": [
            "Results", "Performance Summary", "Comparison",
            "Benchmark Results", "Data Summary", "Metrics Overview",
        ],
        "css_templates": ["technical", "financial", "compact"],
        "bold_density": (0.3, 0.5),
        "italic_density": (0.15, 0.3),
        "strike_density": (0.08, 0.2),
        "table_probability": 0.8,
    },
    "02_academic": {
        "description": "Academic papers with citations, key terms, and scientific content",
        "paragraphs": [
            "Machine learning algorithms can be broadly categorized into supervised, unsupervised, and reinforcement learning paradigms. Supervised learning requires labeled training data, where each input is paired with the desired output.",
            "The transformer architecture introduced self-attention as a replacement for recurrence, enabling parallel processing of sequence elements. Each attention head computes query, key, and value projections.",
            "Gradient descent is the foundational optimization algorithm used in training neural networks. The basic idea is to iteratively adjust model parameters in the direction that reduces the loss function.",
            "The central limit theorem states that the sampling distribution of the sample mean approaches a normal distribution as the sample size increases, regardless of the population distribution shape.",
            "In organic chemistry, functional groups determine the chemical properties of molecules. Common functional groups include hydroxyl, carbonyl, carboxyl, and amino groups.",
            "The protein folding problem involves predicting the three-dimensional structure of a protein from its amino acid sequence. Recent advances in deep learning have dramatically improved prediction accuracy.",
            "Clinical trials demonstrated a statistically significant reduction in symptom severity compared to the placebo group. The primary endpoint was achieved with a p-value of less than 0.001.",
            "Bayesian inference provides a principled framework for updating beliefs in light of new evidence. The posterior distribution combines prior knowledge with observed data through the likelihood function.",
            "Convolutional neural networks exploit spatial locality through learned filters that detect features at different scales. Pooling layers reduce dimensionality while preserving the most salient activations.",
            "The Navier-Stokes equations describe the motion of viscous fluid substances. These partial differential equations arise from applying Newton's second law to fluid motion.",
        ],
        "headings": [
            "Abstract", "Introduction", "Related Work", "Methods",
            "Experimental Setup", "Results", "Discussion", "Conclusion",
            "Literature Review", "Theoretical Framework", "Appendix",
        ],
        "css_templates": ["academic", "textbook"],
        "bold_density": (0.2, 0.4),
        "italic_density": (0.25, 0.45),
        "strike_density": (0.03, 0.1),
    },
    "03_financial": {
        "description": "Financial reports with bold totals, italic footnotes, strikethrough revisions",
        "paragraphs": [
            "Revenue for the fiscal year increased by 12.3% compared to the prior year, driven primarily by strong performance in the cloud services segment. Operating margins improved to 28.4%, reflecting cost optimization initiatives.",
            "The board of directors approved a quarterly dividend of $0.85 per share, payable to shareholders of record as of March 15. This represents a 5% increase from the previous quarter.",
            "Risk factors include market volatility, regulatory changes, and competitive pressures in key segments. The company maintains a hedging strategy for foreign currency exposure.",
            "Total assets grew to $4.2 billion, an increase of 8.7% year-over-year. Current liabilities decreased by 3.2%, improving the current ratio to 1.85 from 1.72 in the prior period.",
            "Earnings per share reached $3.42, surpassing analyst consensus estimates of $3.28. Adjusted EBITDA margin expanded 120 basis points to 34.6% for the full fiscal year.",
            "Capital expenditures totaled $890 million, primarily directed toward infrastructure modernization and capacity expansion in high-growth markets. Free cash flow generation remained robust at $1.1 billion.",
            "The company completed three strategic acquisitions during the year for an aggregate consideration of $2.3 billion. Integration synergies are expected to realize $150 million in annual cost savings.",
            "Provision for credit losses increased to $45 million from $32 million in the prior year, reflecting a more conservative approach to risk management amid evolving market conditions.",
        ],
        "headings": [
            "Financial Summary", "Quarterly Results", "Market Analysis",
            "Risk Assessment", "Balance Sheet", "Income Statement",
            "Cash Flow Analysis", "Shareholder Information", "Outlook",
        ],
        "css_templates": ["financial", "compact"],
        "bold_density": (0.35, 0.55),
        "italic_density": (0.1, 0.25),
        "strike_density": (0.1, 0.25),
    },
    "04_legal": {
        "description": "Legal documents with bold section references, italic definitions",
        "paragraphs": [
            "Under Section 4.2 of the agreement, the licensee shall maintain comprehensive insurance coverage for the duration of the contract. Any modifications to the terms must be submitted in writing.",
            "The privacy policy outlines how personal information is collected, used, and protected. Users may request deletion of their data at any time by submitting a written request.",
            "The indemnifying party shall defend, indemnify, and hold harmless the indemnified party from any claims, damages, or expenses arising from breach of this agreement.",
            "Force majeure events include natural disasters, acts of war, government restrictions, and pandemic-related disruptions. Neither party shall be liable for delays caused by such events.",
            "Confidential information means any data or materials disclosed by either party that is marked as confidential or would reasonably be understood to be confidential given the nature of the information.",
            "The arbitration shall be conducted in accordance with the rules of the American Arbitration Association. The decision of the arbitrator shall be final and binding upon all parties.",
            "Intellectual property rights in all deliverables shall vest in the client upon full payment. The contractor retains a non-exclusive license to use general knowledge and techniques.",
            "This agreement shall be governed by and construed in accordance with the laws of the State of Delaware, without regard to its conflict of laws provisions.",
        ],
        "headings": [
            "Terms and Conditions", "Definitions", "Obligations",
            "Liability", "Confidentiality", "Dispute Resolution",
            "Intellectual Property", "General Provisions", "Amendments",
        ],
        "css_templates": ["academic", "compact", "financial"],
        "bold_density": (0.3, 0.5),
        "italic_density": (0.2, 0.4),
        "strike_density": (0.08, 0.2),
    },
    "05_technical": {
        "description": "Technical documentation with bold warnings, strikethrough deprecated items",
        "paragraphs": [
            "The system architecture employs a microservices pattern with event-driven communication between components. Each service maintains its own database to ensure loose coupling.",
            "Memory allocation in the kernel follows a buddy system algorithm, splitting and coalescing blocks of power-of-two sizes. This approach minimizes external fragmentation.",
            "The API endpoint accepts JSON payloads with a maximum size of 10MB. Rate limiting is enforced at 1000 requests per minute per API key.",
            "Container orchestration is managed through Kubernetes, with automatic scaling based on CPU and memory utilization thresholds. Pod disruption budgets ensure minimum availability.",
            "The build pipeline consists of compilation, static analysis, unit testing, integration testing, and deployment stages. Each stage must pass before proceeding to the next.",
            "Database migrations are applied using a versioned schema approach. Each migration file contains both up and down operations to support rollback scenarios.",
            "Authentication uses OAuth 2.0 with JWT tokens. Access tokens expire after 15 minutes, while refresh tokens have a 30-day lifetime. Token rotation is enforced on each refresh.",
            "Logging follows a structured JSON format with severity levels, timestamps, correlation IDs, and service identifiers. Logs are shipped to a centralized aggregation service.",
        ],
        "headings": [
            "System Architecture", "API Reference", "Configuration",
            "Deployment Guide", "Troubleshooting", "Migration Notes",
            "Security", "Performance Tuning", "Deprecated Features",
        ],
        "css_templates": ["technical", "modern", "compact"],
        "bold_density": (0.3, 0.5),
        "italic_density": (0.1, 0.25),
        "strike_density": (0.15, 0.3),
    },
    "06_books": {
        "description": "Book pages with chapters, epigraphs, bold key terms, italic emphasis",
        "paragraphs": [
            "The concept of natural selection, first articulated by Charles Darwin, describes how organisms with favorable traits are more likely to survive and reproduce. Over many generations, this process drives the adaptation of species to their environments.",
            "In the years following the Great War, the political landscape of Europe was irrevocably altered. Empires that had stood for centuries crumbled, replaced by a patchwork of nation-states grappling with the burden of reconstruction.",
            "The theory of general relativity fundamentally changed our understanding of gravity. Rather than a force acting at a distance, Einstein showed that massive objects curve the fabric of spacetime, and objects follow the resulting geodesics.",
            "Language acquisition in children follows a remarkably consistent pattern across cultures. Babbling gives way to single words around twelve months, followed by two-word combinations and eventually complex grammatical structures.",
            "The invention of the printing press by Johannes Gutenberg around 1440 transformed the dissemination of knowledge. Books, previously copied by hand at great expense, could now be produced rapidly and affordably.",
            "Feudalism organized medieval European society into a hierarchical system of mutual obligations. Lords granted land to vassals in exchange for military service, while serfs worked the land in return for protection.",
            "The double helix structure of DNA, discovered by Watson and Crick in 1953, revealed how genetic information is stored and replicated. The complementary base pairing of adenine with thymine and guanine with cytosine ensures faithful copying.",
            "Philosophical empiricism holds that all knowledge ultimately derives from sensory experience. John Locke argued that the mind begins as a blank slate, shaped entirely by perception and reflection.",
            "The steam engine, perfected by James Watt in the late eighteenth century, powered the Industrial Revolution. Factories could now operate independently of water power, concentrating production in urban centers.",
            "Cognitive dissonance theory, proposed by Leon Festinger, describes the discomfort people feel when holding contradictory beliefs. Individuals are motivated to reduce this tension by changing their attitudes or behaviors.",
        ],
        "headings": [
            "Chapter One", "Chapter Two", "Chapter Three",
            "Prologue", "Epilogue", "Part I", "Part II",
            "The Beginning", "A New Era", "Foundations",
            "Origins", "Consequences", "Reflections",
        ],
        "css_templates": ["textbook", "academic", "book"],
        "bold_density": (0.2, 0.4),
        "italic_density": (0.25, 0.5),
        "strike_density": (0.02, 0.08),
        "blockquote_probability": 0.4,
        "numbered_list_probability": 0.15,
    },
    "07_slides": {
        "description": "Presentation slides with large bold titles, bullet-heavy, minimal paragraphs",
        "paragraphs": [
            "Revenue grew 23% year-over-year, driven by expansion into three new markets and increased adoption of the enterprise tier across existing customers.",
            "Our team has grown from 50 to 120 employees in the past twelve months, with key hires in engineering, product management, and customer success.",
            "Customer satisfaction scores improved from 4.2 to 4.7 out of 5.0, reflecting investments in onboarding experience and dedicated support channels.",
            "The platform now processes over 2 million requests per day, with 99.97% uptime achieved consistently across all production regions.",
            "Competitive analysis reveals three primary differentiators: speed of deployment, depth of integrations, and total cost of ownership over a three-year period.",
            "Next quarter priorities include launching the mobile application, expanding the partner ecosystem, and achieving SOC 2 Type II certification.",
            "Market research indicates a total addressable market of $8.5 billion, growing at 15% annually through 2030 according to independent analyst estimates.",
            "Technical debt reduction program delivered a 40% improvement in build times and reduced critical bug count from 47 to 12 over the past two sprints.",
        ],
        "headings": [
            "Key Highlights", "Q4 Results", "Strategic Priorities",
            "Product Roadmap", "Market Opportunity", "Team Update",
            "Financial Overview", "Next Steps", "Action Items",
            "Agenda", "Executive Summary", "Takeaways",
        ],
        "css_templates": ["slides", "modern"],
        "bold_density": (0.4, 0.6),
        "italic_density": (0.1, 0.2),
        "strike_density": (0.05, 0.15),
        "numbered_list_probability": 0.5,
        "list_probability": 0.6,
        "short_paragraphs": True,
    },
    "08_news": {
        "description": "News articles with bold headlines, italic bylines, dense prose",
        "paragraphs": [
            "City officials announced plans to invest $2.4 billion in public transit infrastructure over the next decade, marking the largest transportation commitment in the region's history.",
            "The biotechnology firm reported promising results from its Phase III clinical trial, with the experimental treatment showing a 34% reduction in disease progression compared to the standard of care.",
            "International trade negotiations entered their fifth round as delegates from twelve nations gathered to discuss tariff reductions and intellectual property protections in the digital economy.",
            "A magnitude 6.2 earthquake struck the coastal province early Tuesday morning, damaging hundreds of structures and displacing an estimated 15,000 residents. Emergency services were deployed within the hour.",
            "The central bank held interest rates steady at 4.75%, citing persistent inflationary pressures despite recent signs of slowing economic growth in the manufacturing sector.",
            "Researchers at the national laboratory unveiled a prototype quantum processor capable of performing specific calculations one million times faster than current supercomputers.",
            "The education ministry released new guidelines requiring all secondary schools to integrate digital literacy and computational thinking into their core curricula by the 2027 academic year.",
            "Voters in three states approved ballot measures to expand renewable energy mandates, setting targets of 80% clean electricity generation by 2035.",
            "The technology giant agreed to a landmark settlement of $4.7 billion to resolve antitrust claims, the largest such penalty in the industry's history.",
            "Police arrested twelve suspects in a coordinated operation spanning four countries, dismantling a network responsible for an estimated $500 million in financial fraud.",
        ],
        "headings": [
            "Breaking News", "World", "Business", "Technology",
            "Science", "Health", "Politics", "Economy",
            "Local News", "Analysis", "Opinion", "Editorial",
        ],
        "css_templates": ["news", "compact", "modern"],
        "bold_density": (0.3, 0.5),
        "italic_density": (0.15, 0.35),
        "strike_density": (0.05, 0.15),
        "byline_probability": 0.6,
    },
    "09_forms": {
        "description": "Forms and exams with numbered items, bold labels, italic instructions",
        "paragraphs": [
            "Please complete all sections of this form in black or blue ink. Incomplete applications will not be processed. Allow 4-6 weeks for review.",
            "Provide your full legal name as it appears on your government-issued identification. If your name has changed, attach supporting documentation.",
            "List all medications currently prescribed, including dosage and frequency. Include over-the-counter supplements and vitamins taken regularly.",
            "Describe your relevant work experience in reverse chronological order, starting with your most recent position. Include dates of employment and key responsibilities.",
            "Read each statement carefully and select the response that best describes your level of agreement. There are no right or wrong answers.",
            "Calculate the total amount due by summing all line items and applying the applicable tax rate. Round to the nearest cent.",
            "For each question, show your complete work in the space provided. Partial credit may be awarded for correct methodology even if the final answer is incorrect.",
            "Sign and date this form to certify that all information provided is accurate and complete to the best of your knowledge.",
            "Enter the reference number from your confirmation email in the space below. If you did not receive a confirmation, contact the support desk.",
            "Rate your overall satisfaction with the service on a scale of 1 to 10, where 1 indicates very dissatisfied and 10 indicates very satisfied.",
        ],
        "headings": [
            "Application Form", "Section A: Personal Information",
            "Section B: Employment History", "Section C: Education",
            "Part 1: Instructions", "Part 2: Questions",
            "Patient Intake Form", "Registration",
            "Exam: Final Assessment", "Survey Questions",
            "Order Form", "Evaluation Criteria",
        ],
        "css_templates": ["compact", "technical", "forms"],
        "bold_density": (0.35, 0.55),
        "italic_density": (0.2, 0.4),
        "strike_density": (0.05, 0.15),
        "numbered_list_probability": 0.7,
        "list_probability": 0.4,
    },
}


# Shared formatting candidates (used across all categories)
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
# CSS templates
# ---------------------------------------------------------------------------

CSS_TEMPLATES = {
    "academic": """
        body { font-family: 'Times New Roman', serif; font-size: 12pt;
               line-height: 1.6; max-width: 700px; margin: 40px auto;
               color: #1a1a1a; background: #fff; padding: 40px; }
        h1 { font-size: 18pt; margin-top: 24pt; }
        h2 { font-size: 15pt; margin-top: 20pt; }
        h3 { font-size: 13pt; margin-top: 16pt; }
    """,
    "textbook": """
        body { font-family: 'Georgia', serif; font-size: 11pt;
               line-height: 1.5; max-width: 650px; margin: 30px auto;
               color: #2c2c2c; background: #fafaf5; padding: 35px; }
        h1 { font-size: 20pt; color: #1a3c5e; border-bottom: 2px solid #1a3c5e; padding-bottom: 6px; }
        h2 { font-size: 16pt; color: #2a5c8e; }
        h3 { font-size: 13pt; color: #3a7cae; }
    """,
    "technical": """
        body { font-family: 'Arial', 'Helvetica', sans-serif; font-size: 10pt;
               line-height: 1.45; max-width: 680px; margin: 25px auto;
               color: #333; background: #fff; padding: 30px; }
        h1 { font-size: 16pt; font-weight: 700; }
        h2 { font-size: 13pt; font-weight: 700; }
        h3 { font-size: 11pt; font-weight: 600; }
    """,
    "financial": """
        body { font-family: 'Calibri', 'Segoe UI', sans-serif; font-size: 10.5pt;
               line-height: 1.4; max-width: 720px; margin: 30px auto;
               color: #1a1a1a; background: #fff; padding: 35px; }
        h1 { font-size: 18pt; color: #003366; }
        h2 { font-size: 14pt; color: #004488; }
        h3 { font-size: 12pt; color: #005599; }
    """,
    "modern": """
        body { font-family: 'Helvetica Neue', 'Arial', sans-serif; font-size: 11pt;
               line-height: 1.55; max-width: 660px; margin: 35px auto;
               color: #222; background: #fefefe; padding: 40px; }
        h1 { font-size: 22pt; font-weight: 300; letter-spacing: -0.5px; }
        h2 { font-size: 16pt; font-weight: 400; }
        h3 { font-size: 13pt; font-weight: 500; }
    """,
    "compact": """
        body { font-family: 'Verdana', sans-serif; font-size: 9pt;
               line-height: 1.35; max-width: 750px; margin: 20px auto;
               color: #1c1c1c; background: #fff; padding: 25px; }
        h1 { font-size: 14pt; margin-top: 18pt; }
        h2 { font-size: 12pt; margin-top: 14pt; }
        h3 { font-size: 10pt; margin-top: 10pt; }
    """,
    "book": """
        body { font-family: 'Palatino', 'Book Antiqua', serif; font-size: 11.5pt;
               line-height: 1.65; max-width: 620px; margin: 50px auto;
               color: #2a2a2a; background: #fffff8; padding: 45px; }
        h1 { font-size: 24pt; font-weight: 400; margin-top: 36pt;
             border-bottom: 1px solid #999; padding-bottom: 8px; }
        h2 { font-size: 16pt; font-weight: 400; font-style: italic; margin-top: 24pt; }
        h3 { font-size: 13pt; margin-top: 18pt; }
        blockquote { border-left: 3px solid #999; margin-left: 20px;
                     padding-left: 15px; font-style: italic; color: #555; }
    """,
    "slides": """
        body { font-family: 'Helvetica Neue', 'Arial', sans-serif; font-size: 13pt;
               line-height: 1.5; max-width: 760px; margin: 30px auto;
               color: #1a1a1a; background: #fff; padding: 35px; }
        h1 { font-size: 28pt; font-weight: 700; color: #003366;
             margin-bottom: 16pt; margin-top: 30pt; }
        h2 { font-size: 20pt; font-weight: 600; color: #0055aa; }
        h3 { font-size: 16pt; font-weight: 500; color: #2277cc; }
        ul, ol { font-size: 12pt; line-height: 1.7; }
        li { margin-bottom: 6px; }
    """,
    "news": """
        body { font-family: 'Georgia', 'Times New Roman', serif; font-size: 10.5pt;
               line-height: 1.5; max-width: 700px; margin: 25px auto;
               color: #1a1a1a; background: #fff; padding: 30px; }
        h1 { font-size: 22pt; font-weight: 700; line-height: 1.2;
             margin-bottom: 4pt; }
        h2 { font-size: 16pt; font-weight: 700; }
        h3 { font-size: 13pt; font-weight: 600; }
        .byline { font-style: italic; color: #666; font-size: 9.5pt;
                   margin-bottom: 12pt; }
    """,
    "forms": """
        body { font-family: 'Arial', 'Helvetica', sans-serif; font-size: 10pt;
               line-height: 1.4; max-width: 720px; margin: 20px auto;
               color: #1a1a1a; background: #fff; padding: 30px;
               border: 1px solid #ccc; }
        h1 { font-size: 16pt; font-weight: 700; text-align: center;
             border-bottom: 2px solid #333; padding-bottom: 8px; }
        h2 { font-size: 13pt; font-weight: 700; background: #f0f0f0;
             padding: 4px 8px; }
        h3 { font-size: 11pt; font-weight: 600; }
        ol { counter-reset: item; }
        li { margin-bottom: 8px; }
    """,
}


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
    all at once from right-to-left to avoid index shifting.

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
        if combo_roll < 0.03:
            is_bold = is_italic = is_strike = True
        elif combo_roll < 0.08:
            pair = rng.choice(["bi", "bs", "is"])
            if pair == "bi":
                is_bold = is_italic = True
            elif pair == "bs":
                is_bold = is_strike = True
            else:
                is_italic = is_strike = True

        if is_bold or is_italic or is_strike:
            candidates.append((char_start, char_end, is_bold, is_italic, is_strike))

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


# ---------------------------------------------------------------------------
# Table generation (for 01_tables category)
# ---------------------------------------------------------------------------

TABLE_HEADERS = [
    ["Metric", "Q1", "Q2", "Q3", "Q4"],
    ["Model", "Accuracy", "F1 Score", "Latency (ms)", "Parameters"],
    ["Category", "Count", "Percentage", "Change", "Status"],
    ["Region", "Revenue", "Costs", "Margin", "Growth"],
    ["Feature", "Baseline", "Proposed", "Delta", "p-value"],
    ["Component", "Version", "License", "Size", "Notes"],
    ["Test Case", "Expected", "Actual", "Pass/Fail", "Duration"],
]

TABLE_DATA_TEMPLATES = [
    # Numeric data
    lambda rng: f"{rng.uniform(0.5, 99.9):.1f}",
    lambda rng: f"${rng.randint(10, 9999):,}",
    lambda rng: f"{rng.uniform(-15, 45):.1f}%",
    lambda rng: f"{rng.randint(1, 500)}ms",
    lambda rng: f"{rng.uniform(0.001, 0.999):.3f}",
]


def generate_table_markdown(rng: random.Random, bold_density: float = 0.3,
                            italic_density: float = 0.2,
                            strike_density: float = 0.1) -> str:
    """Generate a markdown table with formatted cells.

    Args:
        rng: Random number generator.
        bold_density: Probability of bolding a cell.
        italic_density: Probability of italicizing a cell.
        strike_density: Probability of strikethrough on a cell.

    Returns:
        Markdown table string.
    """
    headers = rng.choice(TABLE_HEADERS)
    num_rows = rng.randint(3, 7)
    gen = rng.choice(TABLE_DATA_TEMPLATES)

    # Header row (often bold)
    header_cells = []
    for h in headers:
        if rng.random() < 0.6:
            header_cells.append(f"**{h}**")
        else:
            header_cells.append(h)
    lines = ["| " + " | ".join(header_cells) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    # Data rows
    for _ in range(num_rows):
        row_cells = []
        for j in range(len(headers)):
            if j == 0:
                # First column: label
                cell = rng.choice(["Alpha", "Beta", "Gamma", "Delta", "Model A",
                                   "Model B", "Config 1", "Config 2", "Baseline",
                                   "Proposed", "v2.0", "v3.1", "Test 1", "Test 2"])
            else:
                cell = gen(rng)

            # Apply formatting to cell
            fmt_roll = rng.random()
            if fmt_roll < bold_density * 0.5:
                cell = f"**{cell}**"
            elif fmt_roll < (bold_density + italic_density) * 0.4:
                cell = f"*{cell}*"
            elif fmt_roll < (bold_density + italic_density + strike_density) * 0.4:
                cell = f"~~{cell}~~"

            row_cells.append(cell)
        lines.append("| " + " | ".join(row_cells) + " |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Page generation (category-aware)
# ---------------------------------------------------------------------------

BLOCKQUOTE_POOL = [
    "The only true wisdom is in knowing you know nothing.",
    "In the middle of difficulty lies opportunity.",
    "The measure of intelligence is the ability to change.",
    "Education is not the filling of a pail, but the lighting of a fire.",
    "Those who cannot remember the past are condemned to repeat it.",
    "The unexamined life is not worth living.",
    "Science is organized knowledge. Wisdom is organized life.",
    "The important thing is not to stop questioning.",
]

BYLINE_POOL = [
    "By **Sarah Chen**, Staff Reporter",
    "By **James O'Brien** and **Maria Gonzalez**",
    "By **Dr. Rachel Kim**, Science Correspondent",
    "*Associated Press* | March 9, 2026",
    "By **Thomas Wright**, Senior Analyst | *Financial Times*",
    "*Reuters* | Updated 2:45 PM EST",
    "By **Emily Park**, Technology Editor",
    "Special Report by **David Nakamura**",
]


def _generate_list(rng: random.Random, paragraphs: list[str],
                   numbered: bool = False, num_items: int = 0) -> str:
    """Generate a formatted bullet or numbered list.

    Args:
        rng: Random number generator.
        paragraphs: Content pool to draw from.
        numbered: Use numbered list instead of bullets.
        num_items: Number of items (0 = random 2-6).

    Returns:
        Markdown list string.
    """
    if num_items == 0:
        num_items = rng.randint(2, 6)
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
        prefix = f"{i + 1}." if numbered else "-"
        items.append(f"{prefix} {item_text}: {rng.choice(paragraphs)[:80]}")
    return "\n".join(items)


def generate_page_markdown(rng: random.Random, page_id: int, category: dict) -> str:
    """Generate a single page of formatting-rich markdown for a given category.

    Args:
        rng: Random number generator.
        page_id: Unique page identifier.
        category: Category config dict from CATEGORIES.

    Returns:
        Markdown string with inline formatting.
    """
    paragraphs = category["paragraphs"]
    headings = category["headings"]
    bold_range = category["bold_density"]
    italic_range = category["italic_density"]
    strike_range = category["strike_density"]
    table_prob = category.get("table_probability", 0.15)
    list_prob = category.get("list_probability", 0.3)
    numbered_prob = category.get("numbered_list_probability", 0.15)
    blockquote_prob = category.get("blockquote_probability", 0.1)
    byline_prob = category.get("byline_probability", 0.0)
    short_paras = category.get("short_paragraphs", False)

    parts = []
    num_sections = rng.randint(2, 4)

    # Byline after first heading (news style)
    if byline_prob > 0 and rng.random() < byline_prob:
        heading = rng.choice(headings)
        parts.append(f"# {heading}")
        parts.append(rng.choice(BYLINE_POOL))
    else:
        heading = rng.choice(headings)
        level = rng.choice(["##", "###"]) if rng.random() < 0.3 else "#"
        parts.append(f"{level} {heading}")

    for s in range(num_sections):
        if s > 0:
            heading = rng.choice(headings)
            level = rng.choice(["##", "###"]) if rng.random() < 0.4 else "#"
            parts.append(f"{level} {heading}")

        # Blockquote (books, occasionally others)
        if rng.random() < blockquote_prob:
            quote = rng.choice(BLOCKQUOTE_POOL)
            parts.append(f"> *{quote}*")

        # 1-3 paragraphs per section (1-2 for slides)
        num_paras = rng.randint(1, 2) if short_paras else rng.randint(2, 3)
        for _ in range(num_paras):
            para = rng.choice(paragraphs)

            # Slides: sometimes truncate to a single sentence
            if short_paras and rng.random() < 0.4:
                sentences = para.split(". ")
                para = sentences[0] + "."

            para = inject_formatting(
                para, rng,
                bold_density=rng.uniform(*bold_range),
                italic_density=rng.uniform(*italic_range),
                strike_density=rng.uniform(*strike_range),
            )

            # Parenthetical formatted phrases
            sentences = para.split(". ")
            if len(sentences) > 1:
                if rng.random() < 0.35:
                    idx = rng.randint(0, len(sentences) - 1)
                    sentences[idx] += f" ({_random_formatted_phrase(rng, 'bold')})"
                if rng.random() < 0.35:
                    idx = rng.randint(0, len(sentences) - 1)
                    sentences[idx] += f" ({_random_formatted_phrase(rng, 'italic')})"
                if rng.random() < 0.2:
                    idx = rng.randint(0, len(sentences) - 1)
                    sentences[idx] += f" ({_random_formatted_phrase(rng, 'strikethrough')})"
                para = ". ".join(sentences)

            parts.append(para)

        # Table insertion
        if rng.random() < table_prob:
            table = generate_table_markdown(
                rng,
                bold_density=rng.uniform(*bold_range),
                italic_density=rng.uniform(*italic_range),
                strike_density=rng.uniform(*strike_range),
            )
            parts.append(table)

        # Numbered list
        if rng.random() < numbered_prob:
            parts.append(_generate_list(rng, paragraphs, numbered=True))

        # Bullet list
        if rng.random() < list_prob:
            parts.append(_generate_list(rng, paragraphs, numbered=False))

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

def markdown_to_html(md_text: str, css: str) -> str:
    """Convert markdown text to styled HTML for rendering.

    Handles bold, italic, strikethrough, headings, lists, tables, and paragraphs.

    Args:
        md_text: Markdown text with formatting.
        css: CSS stylesheet string.

    Returns:
        Complete HTML document string.
    """
    lines = md_text.split("\n")
    html_parts = []
    in_list = False
    list_tag = "ul"  # "ul" or "ol"
    in_table = False

    def _close_list():
        nonlocal in_list, list_tag
        if in_list:
            html_parts.append(f"</{list_tag}>")
            in_list = False

    for line in lines:
        stripped = line.strip()

        if not stripped:
            _close_list()
            if in_table:
                html_parts.append("</table>")
                in_table = False
            continue

        # Table rows
        if stripped.startswith("|") and stripped.endswith("|"):
            _close_list()
            cells = [c.strip() for c in stripped.strip("|").split("|")]

            if all(re.match(r"^-+$", c) for c in cells):
                continue

            if not in_table:
                html_parts.append('<table style="border-collapse: collapse; width: 100%; '
                                  'margin: 12px 0;">')
                in_table = True
                html_parts.append("<tr>" + "".join(
                    f'<th style="border: 1px solid #ccc; padding: 6px 10px; '
                    f'background: #f5f5f5; text-align: left;">'
                    f'{_inline_format(c)}</th>' for c in cells
                ) + "</tr>")
            else:
                html_parts.append("<tr>" + "".join(
                    f'<td style="border: 1px solid #ccc; padding: 6px 10px;">'
                    f'{_inline_format(c)}</td>' for c in cells
                ) + "</tr>")
            continue

        if in_table:
            html_parts.append("</table>")
            in_table = False

        # Blockquotes
        if stripped.startswith("> "):
            _close_list()
            html_parts.append(f"<blockquote>{_inline_format(stripped[2:])}</blockquote>")
        # Headings
        elif stripped.startswith("### "):
            _close_list()
            html_parts.append(f"<h3>{_inline_format(stripped[4:])}</h3>")
        elif stripped.startswith("## "):
            _close_list()
            html_parts.append(f"<h2>{_inline_format(stripped[3:])}</h2>")
        elif stripped.startswith("# "):
            _close_list()
            html_parts.append(f"<h1>{_inline_format(stripped[2:])}</h1>")
        # Numbered list items
        elif re.match(r"^\d+\.\s", stripped):
            if in_list and list_tag != "ol":
                _close_list()
            if not in_list:
                html_parts.append("<ol>")
                in_list = True
                list_tag = "ol"
            content = re.sub(r"^\d+\.\s", "", stripped)
            html_parts.append(f"<li>{_inline_format(content)}</li>")
        # Bullet list items
        elif stripped.startswith("- "):
            if in_list and list_tag != "ul":
                _close_list()
            if not in_list:
                html_parts.append("<ul>")
                in_list = True
                list_tag = "ul"
            html_parts.append(f"<li>{_inline_format(stripped[2:])}</li>")
        # Regular paragraph
        else:
            _close_list()
            html_parts.append(f"<p>{_inline_format(stripped)}</p>")

    _close_list()
    if in_table:
        html_parts.append("</table>")

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

    Args:
        image: PIL Image.
        rng: Random number generator.

    Returns:
        Augmented PIL Image.
    """
    from PIL import Image, ImageEnhance, ImageFilter
    import numpy as np

    img = image.copy()

    if rng.random() < 0.5:
        angle = rng.uniform(-2, 2)
        img = img.rotate(angle, fillcolor=(255, 255, 255), expand=False)

    if rng.random() < 0.4:
        factor = rng.uniform(0.85, 1.05)
        img = ImageEnhance.Brightness(img).enhance(factor)

    if rng.random() < 0.3:
        factor = rng.uniform(0.9, 1.1)
        img = ImageEnhance.Contrast(img).enhance(factor)

    if rng.random() < 0.2:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 0.8)))

    if rng.random() < 0.3:
        arr = np.array(img, dtype=np.float32)
        noise = np.random.normal(0, rng.uniform(2, 6), arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    if rng.random() < 0.3:
        import io
        quality = rng.randint(60, 85)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

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
# Formatting stats
# ---------------------------------------------------------------------------

def count_formatting_stats(markdown_dir: str) -> dict:
    """Count formatting spans across all markdown files in a directory.

    Args:
        markdown_dir: Path to directory of .md files.

    Returns:
        Dict with counts and averages.
    """
    total_bold = total_italic = total_strike = 0
    num_files = 0
    for md_file in Path(markdown_dir).glob("*.md"):
        text = md_file.read_text()
        total_bold += len(re.findall(r"\*\*(.+?)\*\*", text))
        total_italic += len(re.findall(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", text))
        total_strike += len(re.findall(r"~~(.+?)~~", text))
        num_files += 1
    return {
        "files": num_files,
        "bold": total_bold,
        "italic": total_italic,
        "strikethrough": total_strike,
        "total": total_bold + total_italic + total_strike,
    }


# ---------------------------------------------------------------------------
# Main generation pipeline
# ---------------------------------------------------------------------------

def generate_category(output_dir: str, cat_name: str, cat_config: dict,
                      num_pages: int, eval_frac: float, augment: bool,
                      content_source: str, seed: int,
                      width: int = 800, height: int = 1100) -> dict:
    """Generate data for a single category with train/eval split.

    Args:
        output_dir: Root output directory.
        cat_name: Category directory name (e.g. "00_text").
        cat_config: Category config dict.
        num_pages: Total pages for this category.
        eval_frac: Fraction of pages for eval split.
        augment: Apply scan-like augmentations.
        content_source: "builtin" or "wikitext".
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

    # Optionally extend content pool
    if content_source == "wikitext":
        extra = load_wikitext_paragraphs(200)
        if extra:
            cat_config = dict(cat_config)
            cat_config["paragraphs"] = cat_config["paragraphs"] + extra

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
            md_text = generate_page_markdown(rng, i, cat_config)

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
            aug_rng = random.Random(seed + hash(split))
            for _, img_path, _ in tqdm(render_queue, desc=f"Augmenting {cat_name}/{split}",
                                       unit="pg"):
                if os.path.exists(img_path):
                    img = Image.open(img_path).convert("RGB")
                    img = augment_image(img, aug_rng)
                    img.save(img_path)

        cat_stats[split] = count
        stats = count_formatting_stats(markdown_dir)
        cat_stats["formatting"][split] = stats
        logger.info(f"  {cat_name}/{split}: {count} pages, "
                    f"bold={stats['bold']}, italic={stats['italic']}, "
                    f"strike={stats['strikethrough']}")

    return cat_stats


def generate_dataset(output_dir: str, num_pages: int, categories: list[str] | None = None,
                     eval_frac: float = 0.05, augment: bool = False,
                     content_source: str = "builtin", seed: int = 42,
                     width: int = 800, height: int = 1100):
    """Generate a full synthetic formatting dataset with category subsets.

    Args:
        output_dir: Root output directory.
        num_pages: Pages per category.
        categories: List of category names to generate (None = all).
        eval_frac: Fraction reserved for eval split.
        augment: Apply scan-like augmentations.
        content_source: "builtin" or "wikitext".
        seed: Random seed.
        width: Rendered page width.
        height: Rendered page height.
    """
    cats_to_gen = categories or list(CATEGORIES.keys())
    all_stats = {}
    total_pages = 0

    logger.info(f"Generating {num_pages} pages x {len(cats_to_gen)} categories "
                f"= {num_pages * len(cats_to_gen)} total")
    logger.info(f"Categories: {', '.join(cats_to_gen)}")
    logger.info(f"Eval fraction: {eval_frac}, Augment: {augment}")

    for cat_name in cats_to_gen:
        if cat_name not in CATEGORIES:
            logger.warning(f"Unknown category '{cat_name}', skipping")
            continue

        cat_config = CATEGORIES[cat_name]
        logger.info(f"Generating {cat_name}: {cat_config['description']}")

        cat_seed = seed + hash(cat_name) % (2**31)
        stats = generate_category(
            output_dir, cat_name, cat_config,
            num_pages=num_pages, eval_frac=eval_frac,
            augment=augment, content_source=content_source,
            seed=cat_seed, width=width, height=height,
        )
        all_stats[cat_name] = stats
        total_pages += stats["train"] + stats["eval"]

    # Save metadata
    metadata = {
        "total_pages": total_pages,
        "pages_per_category": num_pages,
        "categories": {name: CATEGORIES[name]["description"] for name in cats_to_gen
                       if name in CATEGORIES},
        "eval_frac": eval_frac,
        "augmented": augment,
        "content_source": content_source,
        "seed": seed,
        "render_size": f"{width}x{height}",
        "stats": all_stats,
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Done. {total_pages} total pages across {len(cats_to_gen)} categories in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic formatting training data")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Root output directory")
    parser.add_argument("--num_pages", type=int, default=200,
                        help="Pages per category")
    parser.add_argument("--categories", type=str, nargs="*", default=None,
                        help="Categories to generate (default: all). "
                        f"Options: {', '.join(CATEGORIES.keys())}")
    parser.add_argument("--eval_frac", type=float, default=0.05,
                        help="Fraction of pages for eval split (default: 0.05)")
    parser.add_argument("--augment", action="store_true",
                        help="Apply scan-like augmentations")
    parser.add_argument("--content_source", type=str, default="builtin",
                        choices=["builtin", "wikitext"],
                        help="Content source for paragraphs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--width", type=int, default=800,
                        help="Rendered page width in pixels")
    parser.add_argument("--height", type=int, default=1100,
                        help="Rendered page height in pixels")
    args = parser.parse_args()

    generate_dataset(
        output_dir=args.output_dir,
        num_pages=args.num_pages,
        categories=args.categories,
        eval_frac=args.eval_frac,
        augment=args.augment,
        content_source=args.content_source,
        seed=args.seed,
        width=args.width,
        height=args.height,
    )


if __name__ == "__main__":
    main()
