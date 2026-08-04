"""Data collators for different task types."""

import torch


class VLMDataCollator:
    """Data collator for Vision-Language Models (Qwen2.5-VL / Qwen3-VL).

    Tokenizes text and processes images at collation time, avoiding Arrow
    serialization issues with variable-size pixel tensors in datasets.map().

    Each feature must have:
        - 'image': PIL Image
        - 'formatted_text': chat-templated text with image placeholders

    Args:
        processor: AutoProcessor (handles both image and text)
    """

    def __init__(self, processor):
        self.processor = processor
        self.tokenizer = processor.tokenizer

    def __call__(self, features: list[dict]) -> dict:
        """Tokenize text, process images, and create labels with prompt masking.

        Args:
            features: List of dicts with 'image' and 'formatted_text' keys

        Returns:
            Batched dict ready for model forward pass
        """
        images = [f["image"] for f in features]
        texts = [f["formatted_text"] for f in features]

        # Process batch: tokenize text + process images → pixel_values, input_ids, etc.
        inputs = self.processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )

        # Create labels: mask prompt tokens and padding with -100
        # In Qwen VL chat format, the assistant turn starts at the last <|im_start|>
        im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        # Tokens for "assistant\n" (right after <|im_start|> for assistant turn)
        assistant_prefix_ids = self.tokenizer.encode("assistant\n", add_special_tokens=False)
        prefix_len = len(assistant_prefix_ids)
        pad_id = self.tokenizer.pad_token_id

        batch_labels = []
        for i in range(len(features)):
            input_ids = inputs["input_ids"][i]
            labels = input_ids.clone()

            # Find all <|im_start|> positions — last one starts the assistant turn
            positions = (input_ids == im_start_id).nonzero(as_tuple=True)[0].tolist()
            if len(positions) >= 2:
                # Mask everything up to and including <|im_start|>assistant\n
                content_start = positions[-1] + 1 + prefix_len
                labels[:content_start] = -100

            # Mask padding tokens
            if pad_id is not None:
                labels[input_ids == pad_id] = -100

            batch_labels.append(labels)

        inputs["labels"] = torch.stack(batch_labels)
        return dict(inputs)
