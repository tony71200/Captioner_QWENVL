"""
Captioner backends.

The two backends are exposed lazily (PEP 562): importing GGUFCaptioner must not
drag in HFCaptioner's torch, because torch's Intel OpenMP (libiomp5md) aborts
llama.cpp's own libomp140 with "OMP: Error #15" on Windows.
"""
from .base import BaseCaptioner
from .prompts import PROMPT_TEMPLATES, get_prompt_names, get_prompt_by_name

__all__ = [
    "BaseCaptioner",
    "HFCaptioner",
    "GGUFCaptioner",
    "PROMPT_TEMPLATES",
    "get_prompt_names",
    "get_prompt_by_name",
]

_LAZY = {
    "HFCaptioner": ".hf_captioner",
    "GGUFCaptioner": ".gguf_captioner",
}


def __getattr__(name):
    if name in _LAZY:
        from importlib import import_module
        return getattr(import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
