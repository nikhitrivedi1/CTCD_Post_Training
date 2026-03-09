"""Reward functions for GRPO training.

Each function follows trl's reward function signature:
    def reward_fn(completions, **kwargs) -> list[float]

The trainer passes all dataset columns as kwargs, plus `prompts`,
`completion_ids`, and `trainer_state`.

When multiple rewards are used, they are wrapped in a ParallelRewardGroup
that precomputes all rewards concurrently on the first call per batch,
then returns cached results for subsequent calls. This eliminates
sequential GPU idle time during reward scoring.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
logger = logging.getLogger(__name__)

# Lazy-initialized pool for within-function parallelism
_reward_pool = None


def _get_reward_pool():
    """Get or create the shared thread pool for parallel scoring."""
    global _reward_pool
    if _reward_pool is None:
        _reward_pool = ThreadPoolExecutor(max_workers=8)
    return _reward_pool


def _unwrap_completion(completion):
    """Extract text from conversational format if needed."""
    if isinstance(completion, list):
        return completion[-1]["content"]
    return completion


def _score_parallel(score_fn, completions, references=None):
    """Score completions in parallel using threads.

    Args:
        score_fn: Function that scores a single (completion,) or (completion, reference) pair.
        completions: List of generated strings.
        references: Optional list of reference strings (zipped with completions).

    Returns:
        List of float rewards.
    """
    completions = [_unwrap_completion(c) for c in completions]
    if references is not None:
        items = list(zip(completions, references))
        return list(_get_reward_pool().map(lambda args: score_fn(*args), items))
    return list(_get_reward_pool().map(score_fn, completions))


def _score_translation(completion, reference):
    """Score a single completion-reference pair for BLEU."""
    import sacrebleu

    try:
        bleu = sacrebleu.sentence_bleu(completion, [reference])
        return bleu.score / 100.0
    except Exception:
        return 0.0


def translation_reward(completions, ground_truth, **kwargs):
    """Reward based on sentence-level BLEU between generated translation and reference.

    Args:
        completions: List of generated translation strings.
        ground_truth: List of reference translation strings (from dataset column).

    Returns:
        List of float rewards in [0, 1].
    """
    return _score_parallel(_score_translation, completions, ground_truth)


def _score_ocr(completion, reference):
    """Score a single completion-reference pair for edit similarity."""
    from rapidfuzz.distance import Levenshtein

    if not reference:
        return 1.0 if not completion else 0.0
    distance = Levenshtein.distance(completion, reference)
    max_len = max(len(completion), len(reference))
    return max(0.0, 1.0 - (distance / max_len)) if max_len > 0 else 1.0


def ocr_reward(completions, ground_truth, **kwargs):
    """Reward based on edit similarity between generated markdown and reference.

    Uses 1 - CER (character error rate) so higher is better.

    Args:
        completions: List of generated markdown strings.
        ground_truth: List of reference markdown strings (from dataset column).

    Returns:
        List of float rewards in [0, 1].
    """
    return _score_parallel(_score_ocr, completions, ground_truth)


def unit_test_reward(completions, tests, **kwargs):
    """Reward based on olmOCR-bench unit test pass rate.

    Each sample has a list of test dicts (absent, order, math, table).
    Score = fraction of tests passed per completion.

    Args:
        completions: List of generated markdown strings.
        tests: List of lists of test dicts (one list per completion).

    Returns:
        List of float rewards in [0, 1].
    """
    # Try olmocr's test runner for exact scoring (handles math, tables, etc.)
    try:
        from olmocr.bench.tests import load_single_test
        use_olmocr = True
    except ImportError:
        use_olmocr = False
        logger.warning("olmocr not installed, using fallback unit test scoring")

    rewards = []
    for completion, test_list in zip(completions, tests):
        if isinstance(completion, list):
            completion = completion[-1]["content"]

        if not test_list:
            rewards.append(0.0)
            continue

        passed = 0
        for test in test_list:
            try:
                if use_olmocr:
                    # Strip private keys and None values (HF Dataset adds None
                    # for missing keys when schema-unifying heterogeneous dicts)
                    test_data = {k: v for k, v in test.items() if not k.startswith("_") and v is not None}
                    test_obj = load_single_test(test_data)
                    result, _ = test_obj.run(completion)
                    if result:
                        passed += 1
                else:
                    if _fallback_unit_test(test, completion):
                        passed += 1
            except Exception as e:
                logger.debug(f"Test {test.get('id', '?')} failed with error: {e}")

        rewards.append(passed / len(test_list))
    return rewards


def _fallback_unit_test(test: dict, text: str) -> bool:
    """Simple fallback for unit tests when olmocr is not installed.

    Args:
        test: Test dict with 'type' and test-specific fields.
        text: Generated markdown text.

    Returns:
        True if test passes.
    """
    test_type = test.get("type", "")
    if test_type == "present":
        return test.get("text", "") in text
    elif test_type == "absent":
        return test.get("text", "") not in text
    elif test_type == "order":
        before = test.get("before", "")
        after = test.get("after", "")
        b_idx = text.find(before)
        a_idx = text.find(after)
        return b_idx >= 0 and a_idx >= 0 and b_idx < a_idx
    elif test_type == "math":
        return test.get("math", "") in text
    elif test_type == "table":
        return test.get("cell", "") in text
    return False


def _score_format(completion):
    """Score a single completion for markdown structure and formatting."""
    import re

    score = 0.0
    if completion.strip():
        score += 0.2
    if re.search(r"^#{1,6}\s", completion, re.MULTILINE):
        score += 0.2
    if re.search(r"^[\-\*]\s|^\d+\.\s", completion, re.MULTILINE):
        score += 0.2
    if "|" in completion and "-|-" in completion.replace(" ", ""):
        score += 0.2
    if len(completion.strip()) > 50:
        score += 0.2
    # Inline formatting: bold or italic markers present
    if re.search(r"\*\*[^*]+\*\*", completion) or re.search(r"(?<!\*)\*(?!\*)[^*]+(?<!\*)\*(?!\*)", completion):
        score += 0.2
    return min(1.0, score)


def format_reward(completions, **kwargs):
    """Reward for well-structured markdown output (OCR tasks).

    Gives partial credit for outputs that contain markdown structure
    elements like headings, lists, or tables.

    Args:
        completions: List of generated strings.

    Returns:
        List of float rewards in [0, 1].
    """
    return _score_parallel(_score_format, completions)


def _score_table_cell_f1(completion, reference):
    """Score a single completion-reference pair for table cell F1."""
    if not reference:
        return 0.0
    gt_tables = _extract_markdown_tables(reference)
    if not gt_tables:
        return 0.5
    pred_tables = _extract_markdown_tables(completion)
    if not pred_tables:
        return 0.0
    scores = []
    for i, gt_table in enumerate(gt_tables):
        if i < len(pred_tables):
            scores.append(_compute_table_similarity(pred_tables[i], gt_table))
        else:
            scores.append(0.0)
    return sum(scores) / len(scores)


def table_cell_f1_reward(completions, ground_truth, **kwargs):
    """Reward based on cell-level F1 similarity for markdown tables.

    Extracts markdown tables from completion and GT, computes cell-level
    precision/recall/F1 as a lightweight alternative to full TEDS.
    Score is averaged across all GT tables in the document.

    Args:
        completions: List of generated markdown strings.
        ground_truth: List of reference markdown strings.

    Returns:
        List of float rewards in [0, 1].
    """
    return _score_parallel(_score_table_cell_f1, completions, ground_truth)


def _extract_markdown_tables(text: str) -> list[list[list[str]]]:
    """Extract tables from markdown as list of row-cell grids.

    Args:
        text: Markdown text.

    Returns:
        List of tables, each a list of rows, each a list of cell strings.
    """
    import re

    tables = []
    current_table = []
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("|") and line.endswith("|"):
            # Skip separator rows (| --- | --- |)
            inner = line.strip("|")
            parts = [p.strip() for p in inner.split("|")]
            if all(re.match(r"^:?-+:?$", p) for p in parts if p):
                continue
            cells = [c.strip() for c in line.split("|")[1:-1]]
            current_table.append(cells)
        else:
            if current_table:
                tables.append(current_table)
                current_table = []
    if current_table:
        tables.append(current_table)
    return tables


def _compute_table_similarity(pred: list[list[str]], gt: list[list[str]]) -> float:
    """Compute cell-level F1 similarity between two table grids.

    A lightweight alternative to full TEDS that still captures table
    structure and content quality without needing a tree edit distance library.

    Args:
        pred: Predicted table as list of rows of cells.
        gt: Ground truth table as list of rows of cells.

    Returns:
        Float in [0, 1] representing table similarity.
    """
    from rapidfuzz.distance import Levenshtein

    # Flatten to cell lists for comparison
    gt_cells = [cell for row in gt for cell in row if cell.strip()]
    pred_cells = [cell for row in pred for cell in row if cell.strip()]

    if not gt_cells:
        return 1.0 if not pred_cells else 0.0

    # For each GT cell, find best matching pred cell
    matched = 0
    used = set()
    for gt_cell in gt_cells:
        best_sim = 0.0
        best_idx = -1
        for j, pred_cell in enumerate(pred_cells):
            if j in used:
                continue
            dist = Levenshtein.distance(gt_cell, pred_cell)
            max_len = max(len(gt_cell), len(pred_cell))
            sim = 1.0 - (dist / max_len) if max_len > 0 else 1.0
            if sim > best_sim:
                best_sim = sim
                best_idx = j
        if best_idx >= 0 and best_sim > 0.3:
            matched += best_sim
            used.add(best_idx)

    precision = matched / len(pred_cells) if pred_cells else 0.0
    recall = matched / len(gt_cells) if gt_cells else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _score_formula_cdm(completion, reference):
    """Score a single completion-reference pair for formula similarity."""
    if not reference:
        return 0.0
    gt_formulas = _extract_latex(reference)
    if not gt_formulas:
        return 0.5
    pred_formulas = _extract_latex(completion)
    if not pred_formulas:
        return 0.0
    scores = []
    for i, gt_f in enumerate(gt_formulas):
        if i < len(pred_formulas):
            scores.append(_latex_similarity(pred_formulas[i], gt_f))
        else:
            scores.append(0.0)
    return sum(scores) / len(scores)


def formula_cdm_reward(completions, ground_truth, **kwargs):
    """Reward based on formula similarity between completion and GT.

    Extracts LaTeX formulas ($$...$$ and $...$) from both texts and
    computes normalized edit distance on cleaned LaTeX. This is a
    practical proxy for full CDM (computational distance metric).

    Args:
        completions: List of generated markdown strings.
        ground_truth: List of reference markdown strings.

    Returns:
        List of float rewards in [0, 1].
    """
    return _score_parallel(_score_formula_cdm, completions, ground_truth)


def _extract_latex(text: str) -> list[str]:
    """Extract LaTeX formulas from markdown text.

    Args:
        text: Markdown text with $$...$$ or $...$ formulas.

    Returns:
        List of cleaned LaTeX strings.
    """
    import re

    formulas = []
    # Display math: $$...$$
    for match in re.finditer(r"\$\$(.*?)\$\$", text, re.DOTALL):
        formulas.append(_clean_latex(match.group(1)))
    # Inline math: $...$ (but not $$)
    for match in re.finditer(r"(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)", text):
        formulas.append(_clean_latex(match.group(1)))
    return formulas


def _clean_latex(latex: str) -> str:
    """Normalize LaTeX for comparison.

    Args:
        latex: Raw LaTeX string.

    Returns:
        Cleaned LaTeX with normalized whitespace.
    """
    import re

    latex = latex.strip()
    latex = re.sub(r"\s+", " ", latex)
    # Remove common no-op commands
    latex = latex.replace(r"\left", "").replace(r"\right", "")
    latex = latex.replace(r"\displaystyle", "").replace(r"\textstyle", "")
    return latex.strip()


def _latex_similarity(pred: str, gt: str) -> float:
    """Compute normalized edit distance similarity between two LaTeX strings.

    Args:
        pred: Predicted LaTeX.
        gt: Ground truth LaTeX.

    Returns:
        Float in [0, 1].
    """
    from rapidfuzz.distance import Levenshtein

    if not gt:
        return 1.0 if not pred else 0.0
    dist = Levenshtein.distance(pred, gt)
    max_len = max(len(pred), len(gt))
    return 1.0 - (dist / max_len) if max_len > 0 else 1.0


def _score_reading_order(completion, reference):
    """Score a single completion-reference pair for reading order."""
    if not reference:
        return 0.0
    gt_blocks = _extract_text_blocks(reference)
    if not gt_blocks:
        return 0.5
    pred_blocks = _extract_text_blocks(completion)
    if not pred_blocks:
        return 0.0
    gt_order = list(range(len(gt_blocks)))
    pred_order = _match_block_order(pred_blocks, gt_blocks)
    if not pred_order:
        return 0.0
    lcs_len = _lcs_length(pred_order, gt_order)
    return lcs_len / len(gt_blocks)


def reading_order_reward(completions, ground_truth, **kwargs):
    """Reward based on reading order correctness.

    Extracts text blocks (paragraphs) from completion and GT, then
    computes the longest common subsequence ratio to measure whether
    blocks appear in the correct order.

    Args:
        completions: List of generated markdown strings.
        ground_truth: List of reference markdown strings.

    Returns:
        List of float rewards in [0, 1].
    """
    return _score_parallel(_score_reading_order, completions, ground_truth)


def _extract_text_blocks(text: str) -> list[str]:
    """Split text into blocks by double newline.

    Args:
        text: Markdown text.

    Returns:
        List of non-empty text blocks.
    """
    blocks = []
    for block in text.split("\n\n"):
        block = block.strip()
        if block and len(block) > 10:
            blocks.append(block)
    return blocks


def _match_block_order(pred_blocks: list[str], gt_blocks: list[str]) -> list[int]:
    """Match predicted blocks to GT blocks and return GT indices in pred order.

    Args:
        pred_blocks: List of predicted text blocks.
        gt_blocks: List of GT text blocks.

    Returns:
        List of GT indices that matched, in the order they appear in pred.
    """
    from rapidfuzz.distance import Levenshtein

    # Use prefixes for matching (full blocks are too expensive for Levenshtein)
    pred_prefixes = [b[:50] for b in pred_blocks]
    gt_prefixes = [b[:50] for b in gt_blocks]

    order = []
    used = set()
    for pred_b in pred_prefixes:
        best_sim = 0.0
        best_idx = -1
        for j, gt_b in enumerate(gt_prefixes):
            if j in used:
                continue
            dist = Levenshtein.distance(pred_b, gt_b)
            max_len = max(len(pred_b), len(gt_b))
            sim = 1.0 - (dist / max_len) if max_len > 0 else 1.0
            if sim > best_sim:
                best_sim = sim
                best_idx = j
        if best_idx >= 0 and best_sim > 0.3:
            order.append(best_idx)
            used.add(best_idx)
    return order


def _lcs_length(a: list[int], b: list[int]) -> int:
    """Compute length of longest common subsequence.

    Args:
        a: First sequence of integers.
        b: Second sequence of integers.

    Returns:
        Length of LCS.
    """
    m, n = len(a), len(b)
    dp = [0] * (n + 1)
    for i in range(m):
        prev = 0
        for j in range(n):
            temp = dp[j + 1]
            if a[i] == b[j]:
                dp[j + 1] = prev + 1
            else:
                dp[j + 1] = max(dp[j + 1], dp[j])
            prev = temp
    return dp[n]


def _extract_formatting_spans(text: str) -> dict[str, set[str]]:
    """Extract inline formatting spans from markdown text.

    Args:
        text: Markdown text.

    Returns:
        Dict with keys 'bold', 'italic', 'strikethrough', each a set of
        the text content within those markers.
    """
    import re

    bold = set(re.findall(r"\*\*(.+?)\*\*", text))
    italic = set(re.findall(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", text))
    strikethrough = set(re.findall(r"~~(.+?)~~", text))
    return {"bold": bold, "italic": italic, "strikethrough": strikethrough}


def _fuzzy_set_f1(pred_spans: set[str], gt_spans: set[str], threshold: float = 0.5) -> float:
    """Compute F1 between two sets of text spans using fuzzy matching.

    Args:
        pred_spans: Predicted formatting spans.
        gt_spans: Ground truth formatting spans.
        threshold: Minimum similarity to count as a match.

    Returns:
        F1 score in [0, 1].
    """
    from rapidfuzz.distance import Levenshtein

    if not gt_spans:
        return 1.0 if not pred_spans else 0.0
    if not pred_spans:
        return 0.0

    matched = 0.0
    used = set()
    for gt_s in gt_spans:
        best_sim = 0.0
        best_idx = -1
        for j, pred_s in enumerate(pred_spans):
            if j in used:
                continue
            dist = Levenshtein.distance(gt_s.lower(), pred_s.lower())
            max_len = max(len(gt_s), len(pred_s))
            sim = 1.0 - (dist / max_len) if max_len > 0 else 1.0
            if sim > best_sim:
                best_sim = sim
                best_idx = j
        if best_idx >= 0 and best_sim >= threshold:
            matched += best_sim
            used.add(best_idx)

    precision = matched / len(pred_spans)
    recall = matched / len(gt_spans)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _score_inline_formatting(completion, reference):
    """Score inline formatting preservation between completion and reference.

    Extracts bold, italic, and strikethrough spans from both texts and
    computes F1 using fuzzy matching. Returns 0.5 (neutral) if ground
    truth has no formatting.

    Args:
        completion: Generated markdown string.
        reference: Ground truth markdown string.

    Returns:
        Float in [0, 1].
    """
    if not reference:
        return 0.0

    gt = _extract_formatting_spans(reference)
    all_gt = gt["bold"] | gt["italic"] | gt["strikethrough"]

    if not all_gt:
        return 0.5  # No formatting in GT, neutral

    pred = _extract_formatting_spans(completion)
    all_pred = pred["bold"] | pred["italic"] | pred["strikethrough"]

    if not all_pred:
        return 0.0

    return _fuzzy_set_f1(all_pred, all_gt)


def inline_formatting_reward(completions, ground_truth, **kwargs):
    """Reward based on inline formatting preservation (bold, italic, strikethrough).

    Compares formatting markers between generated and reference markdown.
    Returns 0.5 (neutral) for pages with no formatting in ground truth,
    so it doesn't penalize or reward when formatting isn't expected.

    Args:
        completions: List of generated markdown strings.
        ground_truth: List of reference markdown strings.

    Returns:
        List of float rewards in [0, 1].
    """
    return _score_parallel(_score_inline_formatting, completions, ground_truth)


class ParallelRewardGroup:
    """Runs multiple reward functions concurrently to minimize GPU idle time.

    trl calls each reward function sequentially. This wrapper detects when
    a new batch arrives, runs all reward functions in parallel via threads,
    caches results, and returns the cached result for each function.

    Thread-based parallelism works here because the heavy operations
    (rapidfuzz Levenshtein) are C extensions that release the GIL.
    """

    def __init__(self, funcs):
        self._funcs = funcs
        self._cache = {}
        self._last_completions = None

    def _run_all(self, **kwargs):
        completions = kwargs.get("completions")
        if completions is self._last_completions:
            return
        with ThreadPoolExecutor(max_workers=len(self._funcs)) as pool:
            futures = {f.__name__: pool.submit(f, **kwargs) for f in self._funcs}
            self._cache = {name: fut.result() for name, fut in futures.items()}
        self._last_completions = completions

    def wrap(self):
        """Return list of wrapper functions that trl calls individually.

        Returns:
            List of callables with the same names as the original functions.
        """
        wrappers = []
        for func in self._funcs:
            def _make(f):
                def wrapper(**kwargs):
                    self._run_all(**kwargs)
                    return self._cache[f.__name__]
                wrapper.__name__ = f.__name__
                return wrapper
            wrappers.append(_make(func))
        return wrappers


# Registry mapping reward names to functions
REWARD_REGISTRY = {
    "bleu": translation_reward,
    "edit_similarity": ocr_reward,
    "format": format_reward,
    "unit_test": unit_test_reward,
    "table_cell_f1": table_cell_f1_reward,
    "table_teds": table_cell_f1_reward,  # backward compat alias
    "formula_cdm": formula_cdm_reward,
    "reading_order": reading_order_reward,
    "inline_formatting": inline_formatting_reward,
}

# Default reward function per task (used by reward: verifiable)
TASK_REWARDS = {
    "europarl": "bleu",
    "ocr": "edit_similarity",
}


def get_reward_funcs(config):
    """Build reward function list from config.

    Args:
        config: Full config dict with 'rl' and 'data' sections.

    Returns:
        List of reward callables for GRPOTrainer.
    """
    rl_config = config["rl"]
    task = config["data"]["task"]
    reward_spec = rl_config.get("reward", "verifiable")

    if reward_spec == "verifiable":
        reward_name = TASK_REWARDS.get(task)
        if reward_name is None:
            raise ValueError(
                f"No default verifiable reward for task '{task}'. "
                f"Supported: {list(TASK_REWARDS.keys())}. "
                f"Use 'reward: <name>' to specify explicitly."
            )
        logger.info(f"Using verifiable reward '{reward_name}' for task '{task}'")
        return [REWARD_REGISTRY[reward_name]]

    # List of reward names (check before dict lookup to avoid unhashable type error)
    if isinstance(reward_spec, list):
        funcs = []
        for name in reward_spec:
            if name not in REWARD_REGISTRY:
                raise ValueError(f"Unknown reward: {name}. Available: {list(REWARD_REGISTRY.keys())}")
            funcs.append(REWARD_REGISTRY[name])
        logger.info(f"Using reward functions: {reward_spec}")
        if len(funcs) > 1:
            logger.info("Wrapping in ParallelRewardGroup for concurrent scoring")
            funcs = ParallelRewardGroup(funcs).wrap()
        return funcs

    # Single reward name
    if isinstance(reward_spec, str) and reward_spec in REWARD_REGISTRY:
        logger.info(f"Using reward function: {reward_spec}")
        return [REWARD_REGISTRY[reward_spec]]

    raise ValueError(
        f"Unknown reward spec: {reward_spec}. "
        f"Use 'verifiable', a reward name ({list(REWARD_REGISTRY.keys())}), "
        f"or a list of names."
    )
