"""Unified inference engine with SDPA attention.

Centralizes model loading and text generation across tasks (OCR, translation).
Uses PyTorch SDPA on CUDA for optimal attention performance.

Usage:
    # Load for inference (auto SDPA on CUDA)
    engine = InferenceEngine.load(model_id="allenai/olmOCR-2-7B-1025-FP8", model_type="vlm")

    # Wrap existing model (e.g., from training callback)
    engine = InferenceEngine.from_model(model, tokenizer=tokenizer)

    # Generate
    results = engine.generate([(image, "Extract text...")], max_new_tokens=2048)
    results = engine.generate(["Translate this..."], max_new_tokens=128, batch_size=8)
"""

import json
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

_VALID_MODEL_TYPES = {"vlm", "causal_lm"}


def _is_cuda(device):
    """Check if device is any CUDA device (e.g. 'cuda', 'cuda:0')."""
    return str(device).startswith("cuda")


def _detect_attn_impl():
    """Select best attention implementation for inference.

    Returns:
        "sdpa" on CUDA, None otherwise (lets transformers choose).
    """
    if torch.cuda.is_available():
        return "sdpa"
    return None


class InferenceEngine:
    """Unified inference engine with auto SDPA attention.

    Supports two model types:
    - "vlm": Vision-Language Models (Qwen2-VL, olmOCR) — image + text → text
    - "causal_lm": Causal Language Models (Llama) — text → text
    """

    def __init__(self, model=None, tokenizer=None, processor=None,
                 model_type="vlm", device="cuda", model_name="unknown",
                 commit_hash="unknown"):
        self._model = model
        self._tokenizer = tokenizer
        self._processor = processor
        self._model_type = model_type
        self._device = device
        self._model_name = model_name
        self._commit_hash = commit_hash

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, model_id=None, checkpoint_path=None, model_type="vlm",
             device="cuda"):
        """Load a model for inference with optimal attention settings.

        Uses SDPA on CUDA for best performance. Supports LoRA checkpoint
        loading (merges adapter into base model).

        Args:
            model_id: HuggingFace model ID (e.g. "allenai/olmOCR-2-7B-1025-FP8")
            checkpoint_path: Path to LoRA checkpoint directory
            model_type: "vlm" or "causal_lm"
            device: Device string ("cuda", "cuda:0", "mps", "cpu")

        Returns:
            InferenceEngine instance
        """
        if model_type not in _VALID_MODEL_TYPES:
            raise ValueError(
                f"Unsupported model_type '{model_type}'. "
                f"Expected one of: {_VALID_MODEL_TYPES}"
            )
        if model_id is None and checkpoint_path is None:
            raise ValueError("Must provide either 'model_id' or 'checkpoint_path'.")

        if model_type == "vlm":
            return cls._load_hf_vlm(model_id, checkpoint_path, device)
        else:
            return cls._load_hf_causal(model_id, checkpoint_path, device)

    @classmethod
    def from_model(cls, model, tokenizer=None, processor=None):
        """Wrap an already-loaded HF model for generation.

        Used by training callbacks where the model is loaded by the trainer.
        Uses whatever attention implementation the model was loaded with
        (may be eager for attention distillation).

        Args:
            model: HuggingFace model instance
            tokenizer: Tokenizer (for causal LMs)
            processor: AutoProcessor (for VLMs)

        Returns:
            InferenceEngine instance
        """
        if processor is not None:
            model_type = "vlm"
        elif tokenizer is not None:
            model_type = "causal_lm"
        else:
            raise ValueError("Must provide tokenizer (causal LM) or processor (VLM)")

        device = str(next(model.parameters()).device)

        return cls(
            model=model,
            tokenizer=tokenizer,
            processor=processor,
            model_type=model_type,
            device=device,
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, inputs, max_new_tokens=128, batch_size=1):
        """Generate text from inputs.

        Args:
            inputs: For VLM: list of (PIL.Image, str) tuples
                    For causal LM: list of prompt strings
            max_new_tokens: Maximum new tokens to generate
            batch_size: Batch size for causal LM batched generation (ignored for VLM)

        Returns:
            List of generated text strings
        """
        # Normalize single inputs to lists
        if isinstance(inputs, tuple):
            inputs = [inputs]
        elif isinstance(inputs, str):
            inputs = [inputs]

        if self._model_type == "vlm":
            return self._hf_generate_vlm(inputs, max_new_tokens)
        else:
            return self._hf_generate_causal(inputs, max_new_tokens, batch_size)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model_name(self):
        return self._model_name

    @property
    def commit_hash(self):
        return self._commit_hash

    @property
    def model_type(self):
        return self._model_type

    # ------------------------------------------------------------------
    # HF model loading (internal)
    # ------------------------------------------------------------------

    @classmethod
    def _load_hf_vlm(cls, model_id, checkpoint_path, device):
        """Load HF VLM with auto SDPA and optional LoRA merge."""
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from peft import PeftModel

        is_cuda = _is_cuda(device)
        dtype = torch.bfloat16 if is_cuda else (torch.float16 if device == "mps" else torch.float32)
        attn_impl = _detect_attn_impl() if is_cuda else None

        load_kwargs = {"torch_dtype": dtype}
        if is_cuda:
            load_kwargs["device_map"] = "auto"
        if attn_impl:
            load_kwargs["attn_implementation"] = attn_impl
            logger.info(f"Using {attn_impl}")

        if checkpoint_path:
            checkpoint_path = Path(checkpoint_path)
            adapter_config_path = checkpoint_path / "adapter_config.json"
            if not adapter_config_path.exists():
                raise FileNotFoundError(f"No adapter_config.json in {checkpoint_path}")

            with open(adapter_config_path) as f:
                base_model_id = json.load(f)["base_model_name_or_path"]

            logger.info(f"Loading base model: {base_model_id}")
            model = AutoModelForImageTextToText.from_pretrained(
                base_model_id, **load_kwargs
            )
            logger.info(f"Loading LoRA adapter: {checkpoint_path}")
            model = PeftModel.from_pretrained(model, str(checkpoint_path))
            model = model.merge_and_unload()

            processor = AutoProcessor.from_pretrained(base_model_id)
            if checkpoint_path.name.startswith("checkpoint-"):
                model_name = checkpoint_path.parent.name
            else:
                model_name = checkpoint_path.name
            commit_hash = f"local:{checkpoint_path}"
        else:
            logger.info(f"Loading model: {model_id}")
            model = AutoModelForImageTextToText.from_pretrained(
                model_id, **load_kwargs
            )
            processor = AutoProcessor.from_pretrained(model_id)
            model_name = model_id.split("/")[-1]
            commit_hash = getattr(model.config, "_commit_hash", "unknown")

        model.eval()
        if not is_cuda:
            model = model.to(device)
        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        logger.info(f"Model loaded: {model_name} ({param_count:.0f}M params)")

        return cls(
            model=model, processor=processor, model_type="vlm",
            device=device, model_name=model_name, commit_hash=commit_hash,
        )

    @classmethod
    def _load_hf_causal(cls, model_id, checkpoint_path, device):
        """Load HF causal LM with auto SDPA."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        is_cuda = _is_cuda(device)
        dtype = torch.float16 if (is_cuda or device == "mps") else torch.float32
        attn_impl = _detect_attn_impl() if is_cuda else None

        load_kwargs = {"torch_dtype": dtype}
        if is_cuda:
            load_kwargs["device_map"] = "auto"
        if attn_impl:
            load_kwargs["attn_implementation"] = attn_impl
            logger.info(f"Using {attn_impl}")

        if checkpoint_path:
            checkpoint_path = Path(checkpoint_path)
            adapter_config_path = checkpoint_path / "adapter_config.json"
            if not adapter_config_path.exists():
                raise FileNotFoundError(f"No adapter_config.json in {checkpoint_path}")

            with open(adapter_config_path) as f:
                base_model_id = json.load(f)["base_model_name_or_path"]

            from peft import PeftModel
            logger.info(f"Loading base model: {base_model_id}")
            model = AutoModelForCausalLM.from_pretrained(
                base_model_id, **load_kwargs
            )
            logger.info(f"Loading LoRA adapter: {checkpoint_path}")
            model = PeftModel.from_pretrained(model, str(checkpoint_path))
            model = model.merge_and_unload()

            tokenizer = AutoTokenizer.from_pretrained(base_model_id)
            if checkpoint_path.name.startswith("checkpoint-"):
                model_name = checkpoint_path.parent.name
            else:
                model_name = checkpoint_path.name
            commit_hash = f"local:{checkpoint_path}"
        else:
            logger.info(f"Loading model: {model_id}")
            model = AutoModelForCausalLM.from_pretrained(
                model_id, **load_kwargs
            )
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model_name = model_id.split("/")[-1]
            commit_hash = getattr(model.config, "_commit_hash", "unknown")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model.eval()
        if not is_cuda:
            model = model.to(device)
        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        logger.info(f"Model loaded: {model_name} ({param_count:.0f}M params)")

        return cls(
            model=model, tokenizer=tokenizer, model_type="causal_lm",
            device=device, model_name=model_name, commit_hash=commit_hash,
        )

    # ------------------------------------------------------------------
    # Generation (internal)
    # ------------------------------------------------------------------

    def _hf_generate_vlm(self, inputs, max_new_tokens):
        """HF VLM generation — one image at a time.

        Args:
            inputs: List of (PIL.Image, str) tuples
            max_new_tokens: Max tokens to generate per image

        Returns:
            List of generated strings
        """
        results = []
        for image, prompt in inputs:
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]}]
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            model_inputs = self._processor(
                text=[text], images=[image], return_tensors="pt"
            ).to(self._device)

            with torch.no_grad():
                output_ids = self._model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self._processor.tokenizer.pad_token_id,
                )

            generated_ids = output_ids[0, model_inputs["input_ids"].shape[1]:]
            text = self._processor.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            ).strip()
            results.append(text)

        return results

    def _hf_generate_causal(self, prompts, max_new_tokens, batch_size):
        """HF causal LM generation — batched.

        Args:
            prompts: List of prompt strings (already formatted by caller)
            max_new_tokens: Max tokens to generate
            batch_size: Number of prompts per batch

        Returns:
            List of generated strings
        """
        results = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]

            inputs = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self._tokenizer.pad_token_id,
                )

            for j, output in enumerate(outputs):
                input_len = inputs["input_ids"][j].shape[0]
                generated = self._tokenizer.decode(
                    output[input_len:], skip_special_tokens=True
                ).strip()
                results.append(generated)

        return results
