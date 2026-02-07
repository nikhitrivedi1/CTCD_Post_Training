"""Workbench for post-hoc distillation analysis and experiments."""

from .probing import run_probing, compare_probes
from .cka import run_cka
from .attention import run_attention_analysis
from .disagreement import run_disagreement_analysis
from .utils import load_model, load_task_data, extract_hidden_states, extract_attention_weights

__all__ = [
    "run_probing",
    "compare_probes",
    "run_cka",
    "run_attention_analysis",
    "run_disagreement_analysis",
    "load_model",
    "load_task_data",
    "extract_hidden_states",
    "extract_attention_weights",
]
