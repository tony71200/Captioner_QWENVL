"""
Model catalog for QwenVL Image Captioner.
Source: https://github.com/1038lab/ComfyUI-QwenVL
Includes HuggingFace and GGUF models with VRAM requirements.
"""

# ── HuggingFace VL Models ────────────────────────────────────────────────────
HF_VL_MODELS = {
    # ── Qwen3-VL Series ──────────────────────────────────────────────────────
    "Qwen3-VL-2B-Instruct": {
        "repo_id": "Qwen/Qwen3-VL-2B-Instruct",
        "series": "Qwen3-VL",
        "size": "2B",
        "quantized": False,
        "vram": {"full": 4.0, "8bit": 2.5, "4bit": 1.5},
        "min_vram_4gb": True,   # works on 4GB with 4bit
        "description": "Qwen3-VL 2B — compact, runs on 4GB VRAM (4-bit)",
        "hf_url": "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct",
    },
    "Qwen3-VL-2B-Thinking": {
        "repo_id": "Qwen/Qwen3-VL-2B-Thinking",
        "series": "Qwen3-VL",
        "size": "2B",
        "quantized": False,
        "vram": {"full": 4.0, "8bit": 2.5, "4bit": 1.5},
        "min_vram_4gb": True,
        "description": "Qwen3-VL 2B Thinking — chain-of-thought reasoning, 4GB friendly",
        "hf_url": "https://huggingface.co/Qwen/Qwen3-VL-2B-Thinking",
    },
    "Qwen3-VL-2B-Instruct-FP8": {
        "repo_id": "Qwen/Qwen3-VL-2B-Instruct-FP8",
        "series": "Qwen3-VL",
        "size": "2B",
        "quantized": True,
        "vram": {"full": 2.5},
        "min_vram_4gb": True,
        "description": "Qwen3-VL 2B FP8 — pre-quantized, very low VRAM (~2.5GB)",
        "hf_url": "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-FP8",
    },
    "Qwen3-VL-2B-Thinking-FP8": {
        "repo_id": "Qwen/Qwen3-VL-2B-Thinking-FP8",
        "series": "Qwen3-VL",
        "size": "2B",
        "quantized": True,
        "vram": {"full": 2.5},
        "min_vram_4gb": True,
        "description": "Qwen3-VL 2B Thinking FP8 — reasoning + ultra-low VRAM",
        "hf_url": "https://huggingface.co/Qwen/Qwen3-VL-2B-Thinking-FP8",
    },
    "Qwen3-VL-4B-Instruct": {
        "repo_id": "Qwen/Qwen3-VL-4B-Instruct",
        "series": "Qwen3-VL",
        "size": "4B",
        "quantized": False,
        "vram": {"full": 6.0, "8bit": 3.5, "4bit": 2.0},
        "min_vram_4gb": True,   # 4bit only
        "description": "Qwen3-VL 4B — good balance of quality/speed, 4GB with 4-bit",
        "hf_url": "https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct",
    },
    "Qwen3-VL-4B-Thinking": {
        "repo_id": "Qwen/Qwen3-VL-4B-Thinking",
        "series": "Qwen3-VL",
        "size": "4B",
        "quantized": False,
        "vram": {"full": 6.0, "8bit": 3.5, "4bit": 2.0},
        "min_vram_4gb": True,
        "description": "Qwen3-VL 4B Thinking — deep reasoning, 4GB with 4-bit",
        "hf_url": "https://huggingface.co/Qwen/Qwen3-VL-4B-Thinking",
    },
    "Qwen3-VL-4B-Instruct-FP8": {
        "repo_id": "Qwen/Qwen3-VL-4B-Instruct-FP8",
        "series": "Qwen3-VL",
        "size": "4B",
        "quantized": True,
        "vram": {"full": 2.5},
        "min_vram_4gb": True,
        "description": "Qwen3-VL 4B FP8 — pre-quantized, ~2.5GB VRAM",
        "hf_url": "https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-FP8",
    },
    "Qwen3-VL-4B-Thinking-FP8": {
        "repo_id": "Qwen/Qwen3-VL-4B-Thinking-FP8",
        "series": "Qwen3-VL",
        "size": "4B",
        "quantized": True,
        "vram": {"full": 2.5},
        "min_vram_4gb": True,
        "description": "Qwen3-VL 4B Thinking FP8 — reasoning + FP8 quantized",
        "hf_url": "https://huggingface.co/Qwen/Qwen3-VL-4B-Thinking-FP8",
    },
    "Qwen3-VL-8B-Instruct": {
        "repo_id": "Qwen/Qwen3-VL-8B-Instruct",
        "series": "Qwen3-VL",
        "size": "8B",
        "quantized": False,
        "vram": {"full": 12.0, "8bit": 7.0, "4bit": 4.5},
        "min_vram_4gb": False,
        "description": "Qwen3-VL 8B — high quality, needs 8GB+ VRAM",
        "hf_url": "https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct",
    },
    "Qwen3-VL-8B-Thinking": {
        "repo_id": "Qwen/Qwen3-VL-8B-Thinking",
        "series": "Qwen3-VL",
        "size": "8B",
        "quantized": False,
        "vram": {"full": 12.0, "8bit": 7.0, "4bit": 4.5},
        "min_vram_4gb": False,
        "description": "Qwen3-VL 8B Thinking — best reasoning, 8GB+ VRAM",
        "hf_url": "https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking",
    },
    "Qwen3-VL-8B-Instruct-FP8": {
        "repo_id": "Qwen/Qwen3-VL-8B-Instruct-FP8",
        "series": "Qwen3-VL",
        "size": "8B",
        "quantized": True,
        "vram": {"full": 7.5},
        "min_vram_4gb": False,
        "description": "Qwen3-VL 8B FP8 — pre-quantized, needs 8GB VRAM",
        "hf_url": "https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-FP8",
    },
    "Qwen3-VL-8B-Thinking-FP8": {
        "repo_id": "Qwen/Qwen3-VL-8B-Thinking-FP8",
        "series": "Qwen3-VL",
        "size": "8B",
        "quantized": True,
        "vram": {"full": 7.5},
        "min_vram_4gb": False,
        "description": "Qwen3-VL 8B Thinking FP8 — reasoning + pre-quantized",
        "hf_url": "https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking-FP8",
    },
    "Qwen3-VL-32B-Instruct": {
        "repo_id": "Qwen/Qwen3-VL-32B-Instruct",
        "series": "Qwen3-VL",
        "size": "32B",
        "quantized": False,
        "vram": {"full": 28.0, "8bit": 14.0, "4bit": 8.5},
        "min_vram_4gb": False,
        "description": "Qwen3-VL 32B — SOTA quality, multi-GPU recommended",
        "hf_url": "https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct",
    },
    "Qwen3-VL-32B-Thinking": {
        "repo_id": "Qwen/Qwen3-VL-32B-Thinking",
        "series": "Qwen3-VL",
        "size": "32B",
        "quantized": False,
        "vram": {"full": 28.0, "8bit": 14.0, "4bit": 8.5},
        "min_vram_4gb": False,
        "description": "Qwen3-VL 32B Thinking — deepest reasoning, multi-GPU",
        "hf_url": "https://huggingface.co/Qwen/Qwen3-VL-32B-Thinking",
    },
    # ── Qwen2.5-VL Series ────────────────────────────────────────────────────
    "Qwen2.5-VL-3B-Instruct": {
        "repo_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "series": "Qwen2.5-VL",
        "size": "3B",
        "quantized": False,
        "vram": {"full": 6.0, "8bit": 3.5, "4bit": 2.0},
        "min_vram_4gb": True,
        "description": "Qwen2.5-VL 3B — lightweight, 4GB VRAM with 4-bit",
        "hf_url": "https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct",
    },
    "Qwen2.5-VL-7B-Instruct": {
        "repo_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "series": "Qwen2.5-VL",
        "size": "7B",
        "quantized": False,
        "vram": {"full": 15.0, "8bit": 8.5, "4bit": 5.0},
        "min_vram_4gb": False,
        "description": "Qwen2.5-VL 7B — proven quality, needs 6GB+ VRAM",
        "hf_url": "https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct",
    },
}

# ── GGUF VL Models ───────────────────────────────────────────────────────────
GGUF_VL_MODELS = {
    "Qwen3-VL-4B-Instruct-GGUF": {
        "repo_id": "Qwen/Qwen3-VL-4B-Instruct-GGUF",
        "series": "Qwen3-VL",
        "size": "4B",
        "mmproj_file": "mmproj-Qwen3VL-4B-Instruct-F16.gguf",
        "gguf_defaults": {
            "context_length": 8192,
            "image_max_tokens": 4096,
            "n_batch": 512,
            "gpu_layers": -1,
            "top_k": 0,
            "pool_size": 4194304,
        },
        "model_files": {
            "Q4_K_M (recommended, ~2.5GB)": "Qwen3VL-4B-Instruct-Q4_K_M.gguf",
            "Q8_0 (high quality, ~4.5GB)":  "Qwen3VL-4B-Instruct-Q8_0.gguf",
            "F16 (full precision, ~8GB)":   "Qwen3VL-4B-Instruct-F16.gguf",
        },
        "min_vram_4gb": True,
        "description": "Qwen3-VL 4B GGUF — ideal for 4GB VRAM with Q4_K_M",
        "hf_url": "https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF",
    },
    "Qwen3-VL-8B-Instruct-GGUF": {
        "repo_id": "Qwen/Qwen3-VL-8B-Instruct-GGUF",
        "series": "Qwen3-VL",
        "size": "8B",
        "mmproj_file": "mmproj-Qwen3VL-8B-Instruct-F16.gguf",
        "gguf_defaults": {
            "context_length": 8192,
            "image_max_tokens": 4096,
            "n_batch": 512,
            "gpu_layers": -1,
            "top_k": 0,
            "pool_size": 4194304,
        },
        "model_files": {
            "Q4_K_M (recommended, ~5GB)":   "Qwen3VL-8B-Instruct-Q4_K_M.gguf",
            "Q8_0 (high quality, ~9GB)":    "Qwen3VL-8B-Instruct-Q8_0.gguf",
            "F16 (full precision, ~16GB)":  "Qwen3VL-8B-Instruct-F16.gguf",
        },
        "min_vram_4gb": False,
        "description": "Qwen3-VL 8B GGUF — high quality, needs 6GB+ VRAM",
        "hf_url": "https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF",
    },
    "Qwen3-VL-4B-Thinking-GGUF": {
        "repo_id": "Qwen/Qwen3-VL-4B-Thinking-GGUF",
        "series": "Qwen3-VL",
        "size": "4B",
        "mmproj_file": "mmproj-Qwen3VL-4B-Thinking-F16.gguf",
        "gguf_defaults": {
            "context_length": 8192,
            "image_max_tokens": 4096,
            "n_batch": 512,
            "gpu_layers": -1,
            "top_k": 0,
            "pool_size": 4194304,
        },
        "model_files": {
            "Q4_K_M (recommended, ~2.5GB)": "Qwen3VL-4B-Thinking-Q4_K_M.gguf",
            "Q8_0 (high quality, ~4.5GB)":  "Qwen3VL-4B-Thinking-Q8_0.gguf",
            "F16 (full precision, ~8GB)":   "Qwen3VL-4B-Thinking-F16.gguf",
        },
        "min_vram_4gb": True,
        "description": "Qwen3-VL 4B Thinking GGUF — reasoning model, 4GB friendly",
        "hf_url": "https://huggingface.co/Qwen/Qwen3-VL-4B-Thinking-GGUF",
    },
    "Qwen3-VL-8B-Thinking-GGUF": {
        "repo_id": "Qwen/Qwen3-VL-8B-Thinking-GGUF",
        "series": "Qwen3-VL",
        "size": "8B",
        "mmproj_file": "mmproj-Qwen3VL-8B-Thinking-F16.gguf",
        "gguf_defaults": {
            "context_length": 8192,
            "image_max_tokens": 4096,
            "n_batch": 512,
            "gpu_layers": -1,
            "top_k": 0,
            "pool_size": 4194304,
        },
        "model_files": {
            "Q4_K_M (recommended, ~5GB)":   "Qwen3VL-8B-Thinking-Q4_K_M.gguf",
            "Q8_0 (high quality, ~9GB)":    "Qwen3VL-8B-Thinking-Q8_0.gguf",
            "F16 (full precision, ~16GB)":  "Qwen3VL-8B-Thinking-F16.gguf",
        },
        "min_vram_4gb": False,
        "description": "Qwen3-VL 8B Thinking GGUF — best quality reasoning GGUF",
        "hf_url": "https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking-GGUF",
    },
}

# ── VRAM Profile Definitions ─────────────────────────────────────────────────
VRAM_PROFILES = {
    "UltraLow (4GB)": {
        "label": "UltraLow (4GB)",
        "hf_quant":     "4bit",
        "pixel_config": {"min_pixels": 128 * 28 * 28, "max_pixels": 512 * 28 * 28},
        "gguf_layers":  5,
        "gguf_ctx":     1024,
        "description":  "4-bit NF4, minimal resolution — for 4GB VRAM GPUs",
        "recommended_models": ["Qwen3-VL-2B-Instruct", "Qwen3-VL-2B-Instruct-FP8",
                               "Qwen3-VL-4B-Instruct-FP8", "Qwen2.5-VL-3B-Instruct"],
    },
    "LowVRAM (6–8GB)": {
        "label": "LowVRAM (6–8GB)",
        "hf_quant":     "4bit",
        "pixel_config": {"min_pixels": 256 * 28 * 28, "max_pixels": 768 * 28 * 28},
        "gguf_layers":  10,
        "gguf_ctx":     2048,
        "description":  "4-bit NF4 quantization — for 6–8GB VRAM GPUs",
        "recommended_models": ["Qwen3-VL-4B-Instruct", "Qwen3-VL-4B-Thinking",
                               "Qwen2.5-VL-3B-Instruct"],
    },
    "NormalVRAM (12–16GB)": {
        "label": "NormalVRAM (12–16GB)",
        "hf_quant":     "8bit",
        "pixel_config": {"min_pixels": 256 * 28 * 28, "max_pixels": 1280 * 28 * 28},
        "gguf_layers":  25,
        "gguf_ctx":     4096,
        "description":  "8-bit quantization — for 12–16GB VRAM GPUs",
        "recommended_models": ["Qwen3-VL-8B-Instruct", "Qwen2.5-VL-7B-Instruct"],
    },
    "HighVRAM (20GB+)": {
        "label": "HighVRAM (20GB+)",
        "hf_quant":     "none",
        "pixel_config": {"min_pixels": 256 * 28 * 28, "max_pixels": 2560 * 28 * 28},
        "gguf_layers":  -1,
        "gguf_ctx":     8192,
        "description":  "BF16 full precision, max resolution — for 20GB+ VRAM",
        "recommended_models": ["Qwen3-VL-32B-Instruct", "Qwen3-VL-8B-Instruct"],
    },
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_hf_model_names(filter_4gb: bool = False) -> list:
    if filter_4gb:
        return [k for k, v in HF_VL_MODELS.items() if v["min_vram_4gb"]]
    return list(HF_VL_MODELS.keys())


def get_gguf_model_names(filter_4gb: bool = False) -> list:
    if filter_4gb:
        return [k for k, v in GGUF_VL_MODELS.items() if v["min_vram_4gb"]]
    return list(GGUF_VL_MODELS.keys())


def get_vram_profile_names() -> list:
    return list(VRAM_PROFILES.keys())


def get_model_info_html(model_name: str, backend: str) -> str:
    """Return an HTML badge string describing the model."""
    catalog = HF_VL_MODELS if backend == "HuggingFace" else GGUF_VL_MODELS
    info = catalog.get(model_name)
    if not info:
        return ""
    vram_str = ""
    if "vram" in info:
        parts = [f"{k}: {v}GB" for k, v in info["vram"].items()]
        vram_str = " | ".join(parts)
    badge = "🟢 4GB OK" if info.get("min_vram_4gb") else "🔴 6GB+"
    return (
        f'<div style="font-size:13px;color:#94a3b8;margin-top:4px;">'
        f'{badge} &nbsp;·&nbsp; {info["description"]}'
        + (f' &nbsp;·&nbsp; <span style="color:#64748b">VRAM: {vram_str}</span>' if vram_str else "")
        + f'</div>'
    )
