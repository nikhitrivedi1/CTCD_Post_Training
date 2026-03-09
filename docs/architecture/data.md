# Data Module

Dataset loading, preprocessing, collation, and OCR utilities.

**Source:** `src/data/` (4 files: `datasets.py`, `collators.py`, `ocr_utils.py`, `rl_datasets.py`)

---

## Overview

The data module handles loading and formatting datasets for all three training paradigms:
- **Supervised (SFT/KD):** Tokenized datasets with input_ids and labels
- **Reinforcement Learning (GRPO):** Raw prompts + ground truth for reward computation

Every supervised loader returns `(train_dataset, eval_dataset, tokenizer_or_processor)`. RL loaders return `(train_dataset, eval_dataset)`.

---

## datasets.py - Supervised Loaders

### Classification Tasks (GLUE)

All four GLUE loaders follow the same pattern:

1. Load dataset via `datasets.load_dataset("glue", task_name)`
2. Tokenize with `tokenizer(text_a, text_b, truncation=True, max_length=128, padding="max_length")`
3. Rename `label` to `labels` (HuggingFace Trainer convention)
4. Return train/eval splits

| Function | Task | Input Format | Labels | Notes |
|----------|------|-------------|--------|-------|
| `load_mnli` | 3-class entailment | premise + hypothesis | entailment, neutral, contradiction | Uses `validation_matched` split |
| `load_qqp` | Binary duplicate detection | question1 + question2 | duplicate, not duplicate | Filters out -1 labels (unlabeled) |
| `load_sst2` | Binary sentiment | single sentence | positive, negative | |
| `load_cola` | Binary acceptability | single sentence | acceptable, unacceptable | Primary metric: MCC |

### load_europarl (Translation)

Loads Europarl EN-FR (or any language pair) for causal LM training.

**Key details:**
- Uses 45,715 examples (configurable via `max_samples`)
- Split: 70% train, 20% val, 10% test (via two-stage `train_test_split`)
- **Chat template formatting:** if tokenizer has a chat template, formats as system/user/assistant messages. Otherwise uses a simple prompt template.
- **Prompt masking:** labels are set to `-100` for all tokens before the assistant's response. Only the target translation tokens contribute to CE loss. The prompt boundary is found by tokenizing the prompt-only portion and computing the longest common prefix with the full sequence (handles BPE tokenization inconsistencies at the boundary).

```python
# Prompt masking logic
prompt_ids = tokenizer(prompt_text)["input_ids"]
input_ids = tokenizer(full_text)["input_ids"]
# Find longest common prefix
prompt_len = len(common_prefix(prompt_ids, input_ids))
labels[:prompt_len] = [-100] * prompt_len
```

### load_ocr (OCR/VLM)

Loads pre-processed OCR datasets (created by `scripts/data/prepare_ocr_data.py`).

**Key details:**
- Expects local datasets with `image` (PIL) and `text` (markdown) columns
- Supports single directory (`data_dir/train/`) or multi-subset directory (`data_dir/{subset_name}/train/`)
- Multi-subset directories are concatenated via `datasets.concatenate_datasets()`
- **Deferred processing:** only applies chat template here (`formatted_text` column). Tokenization and image processing are deferred to collation time via `VLMDataCollator` to avoid Arrow serialization issues with variable-size pixel tensors.
- Supports `include_reasoning=True` to use a `reasoning_and_output` column instead of `text`
- **Visual token cap:** if `data.max_visual_tokens` is set in config, adjusts processor's `image_processor.max_pixels` to limit visual tokens per image (prevents OOM on high-res images)

### load_dataset_for_task (Router)

Routes to the appropriate loader based on `config.data.task`:

```python
if is_ocr_task(task):        # -> load_ocr(processor, ...)
elif task == "mnli":          # -> load_mnli(tokenizer)
elif task == "europarl":      # -> load_europarl(tokenizer, ...)
# etc.
```

For OCR tasks, loads an `AutoProcessor` (handles image + text). For all others, loads an `AutoTokenizer` from the teacher model ID (or student if no teacher).

For causal LMs (Llama, translation), sets `padding_side="left"` and uses `eos_token` as pad token.

---

## collators.py - VLMDataCollator

Custom data collator for Vision-Language Models (Qwen2.5-VL / Qwen3-VL).

### Why a Custom Collator?

Standard HuggingFace data collators expect pre-tokenized datasets. For VLMs, image processing produces variable-size pixel tensors (`pixel_values`) that cannot be stored in Arrow format (which requires fixed schemas). The solution: defer tokenization and image processing to collation time.

### How It Works

```python
class VLMDataCollator:
    def __call__(self, features: list[dict]) -> dict:
        # 1. Extract images and formatted texts
        images = [f["image"] for f in features]
        texts = [f["formatted_text"] for f in features]

        # 2. Process batch: tokenize text + process images
        inputs = self.processor(text=texts, images=images, padding=True, return_tensors="pt")

        # 3. Create labels with prompt masking
        # Find last <|im_start|> position (assistant turn)
        # Mask everything before it with -100
        # Also mask padding tokens
```

### Prompt Masking Logic

In Qwen VL chat format, turns are delimited by `<|im_start|>`. The assistant turn is the last one. The collator:

1. Finds all `<|im_start|>` positions in the input
2. The last one starts the assistant turn
3. Masks everything up to and including `<|im_start|>assistant\n` with `-100`
4. Masks all padding tokens with `-100`

This ensures only the assistant's response tokens contribute to the loss.

---

## ocr_utils.py - Shared OCR Utilities

Small utility module with constants and functions shared across OCR loaders, conversion scripts, and evaluation.

### OCR_SYSTEM_PROMPT

```python
OCR_SYSTEM_PROMPT = (
    "Below is an image of a document page. Extract all text content "
    "and output it in clean markdown format, preserving the document's "
    "structure (headings, lists, tables, etc.)."
)
```

Used consistently across training (SFT, GRPO) and evaluation (benchmarks).

### render_pdf_page(pdf_path, page_num, dpi=150)

Renders a single PDF page to a PIL Image using pypdfium2.
- `page_num` is 1-indexed
- Default DPI of 150 balances quality vs. processing time
- Closes the PDF document after rendering

### pdf_image_name(pdf_rel, page_num)

Generates a flat image filename from a relative PDF path. Handles both with-category (`arxiv_cs/pdf_00001.pdf` -> `arxiv_cs_pdf_00001_p1.jpg`) and without-category paths.

---

## rl_datasets.py - GRPO Dataset Loaders

Format datasets for GRPO training. Unlike supervised loaders, these return raw prompts (not tokenized) because GRPOTrainer handles tokenization internally during generation.

Each example has:
- `prompt`: list of chat messages (system + user)
- `ground_truth` or `tests`: reference for reward computation
- `image` (optional): PIL Image for VLM tasks

### load_europarl_rl

Same data source and splits as supervised `load_europarl`, but formats as:
```python
{
    "prompt": [
        {"role": "system", "content": "You are a professional translator..."},
        {"role": "user", "content": source_text},
    ],
    "ground_truth": target_text
}
```

### load_ocr_rl

Loads pre-processed OCR dataset for GRPO. Each example includes the image and ground truth markdown. Prompt uses the shared `OCR_SYSTEM_PROMPT`.

```python
{
    "prompt": [{"role": "user", "content": [
        {"type": "text", "text": OCR_SYSTEM_PROMPT},
        {"type": "image"},
    ]}],
    "image": pil_image,
    "ground_truth": markdown_text
}
```

### load_synthmix_rl

Loads olmOCR-synthmix dataset for unit test reward training.

**Key details:**
- Groups test assertions from `bench_data/*.jsonl` by (pdf, page)
- Pairs with pre-rendered images from `images/` directory
- Each sample has a list of test dicts (not ground_truth text)
- Requires `scripts/data/convert_synthmix.py` to pre-render images

```python
{
    "prompt": [...],
    "image": pil_image,
    "tests": [{"type": "present", "text": "..."}, ...]
}
```

### load_omnidocbench_rl

Loads OmniDocBench for edit similarity reward training.

**Key details:**
- Reads pre-converted markdown from `markdown/` directory (created by `scripts/data/convert_omnidocbench.py`)
- Pairs with images from `images/` directory (tries .jpg, .png, .jpeg extensions)
- Each sample has ground_truth text (not tests)

### load_rl_dataset (Router)

Routes to appropriate RL loader based on task and dataset name:
```python
if task == "europarl":                    # -> load_europarl_rl
elif task == "ocr" and dataset == "synthmix":   # -> load_synthmix_rl
elif task == "ocr" and dataset == "omnidocbench": # -> load_omnidocbench_rl
elif task == "ocr":                       # -> load_ocr_rl (default)
```

Classification tasks raise `ValueError` because encoder models cannot generate.

---

*Source: `src/data/`*
