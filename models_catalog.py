"""
Catalog model QwenVL — DỮ LIỆU THUẦN, không chứa logic.

Mọi dung lượng tính bằng GiB (bytes / 2**30), đã đối chiếu với HuggingFace API
ngày 2026-09-20. Phép tính VRAM nằm ở utils/vram_plan.py.

Chỉ nhận model từ org Qwen/ trên HuggingFace.

Model bị loại và lý do:
  - Qwen3-VL-30B-A3B-*-GGUF, Qwen3-VL-32B-*-GGUF: file split 2 phần
    (-split-00001-of-00002.gguf), hf_hub_download một filename không tải được;
    Q4_K_M đã 17.3-18.4 GiB.
  - Qwen3-VL-32B-Instruct/-Thinking (HF): 62.1 GiB trọng số, 4-bit vẫn ~19.9 GiB.
  - Qwen3-VL-235B-A22B-*: ngoài phạm vi.
  - Qwen2-VL, Qwen-VL: bị Qwen2.5-VL thay thế.
Muốn thêm 30B/32B thì phải hỗ trợ tải & load file GGUF split trước.
"""

CATALOG_VERIFIED = {
    "checked": "2026-09-20",
    "source": "https://huggingface.co/api/models/{repo_id}?blobs=true",
    "org": "Qwen",
}

# Tham số KV cache, lấy từ config.json của từng model (text_config).
_KV_QWEN3_2B = {"layers": 28, "kv_heads": 8, "head_dim": 128}
_KV_QWEN3_4B = {"layers": 36, "kv_heads": 8, "head_dim": 128}
_KV_QWEN3_8B = {"layers": 36, "kv_heads": 8, "head_dim": 128}
_KV_QWEN25_3B = {"layers": 36, "kv_heads": 2, "head_dim": 128}
_KV_QWEN25_7B = {"layers": 28, "kv_heads": 4, "head_dim": 128}

_QUANTS_FULL = ["bf16", "8bit", "4bit"]


def _hf(repo, arch, series, size, weights_gib, kv, quality, native=None):
    """Dựng một entry HF. quants là [native] khi checkpoint đã quantize sẵn."""
    return {
        "repo_id": repo,
        "arch": arch,
        "series": series,
        "size": size,
        "weights_gib": weights_gib,   # verified 2026-09-20
        "native_quant": native,
        "quants": [native] if native else list(_QUANTS_FULL),
        "kv": dict(kv),
        "quality": quality,
        "hf_url": f"https://huggingface.co/{repo}",
    }


_A3 = "Qwen3VLForConditionalGeneration"
_A25 = "Qwen2_5_VLForConditionalGeneration"

# ── HuggingFace ──────────────────────────────────────────────────────────────
HF_VL_MODELS = {
    "Qwen3-VL-2B-Instruct":
        _hf("Qwen/Qwen3-VL-2B-Instruct", _A3, "Qwen3-VL", "2B", 3.97, _KV_QWEN3_2B, 2.0),
    "Qwen3-VL-2B-Thinking":
        _hf("Qwen/Qwen3-VL-2B-Thinking", _A3, "Qwen3-VL", "2B", 3.97, _KV_QWEN3_2B, 2.0),
    "Qwen3-VL-2B-Instruct-FP8":
        _hf("Qwen/Qwen3-VL-2B-Instruct-FP8", _A3, "Qwen3-VL", "2B", 3.23, _KV_QWEN3_2B, 2.0, "fp8"),
    "Qwen3-VL-2B-Thinking-FP8":
        _hf("Qwen/Qwen3-VL-2B-Thinking-FP8", _A3, "Qwen3-VL", "2B", 3.23, _KV_QWEN3_2B, 2.0, "fp8"),
    "Qwen3-VL-4B-Instruct":
        _hf("Qwen/Qwen3-VL-4B-Instruct", _A3, "Qwen3-VL", "4B", 8.27, _KV_QWEN3_4B, 4.0),
    "Qwen3-VL-4B-Thinking":
        _hf("Qwen/Qwen3-VL-4B-Thinking", _A3, "Qwen3-VL", "4B", 8.27, _KV_QWEN3_4B, 4.0),
    "Qwen3-VL-4B-Instruct-FP8":
        _hf("Qwen/Qwen3-VL-4B-Instruct-FP8", _A3, "Qwen3-VL", "4B", 5.61, _KV_QWEN3_4B, 4.0, "fp8"),
    "Qwen3-VL-4B-Thinking-FP8":
        _hf("Qwen/Qwen3-VL-4B-Thinking-FP8", _A3, "Qwen3-VL", "4B", 5.61, _KV_QWEN3_4B, 4.0, "fp8"),
    "Qwen3-VL-8B-Instruct":
        _hf("Qwen/Qwen3-VL-8B-Instruct", _A3, "Qwen3-VL", "8B", 16.33, _KV_QWEN3_8B, 8.0),
    "Qwen3-VL-8B-Thinking":
        _hf("Qwen/Qwen3-VL-8B-Thinking", _A3, "Qwen3-VL", "8B", 16.33, _KV_QWEN3_8B, 8.0),
    "Qwen3-VL-8B-Instruct-FP8":
        _hf("Qwen/Qwen3-VL-8B-Instruct-FP8", _A3, "Qwen3-VL", "8B", 9.86, _KV_QWEN3_8B, 8.0, "fp8"),
    "Qwen3-VL-8B-Thinking-FP8":
        _hf("Qwen/Qwen3-VL-8B-Thinking-FP8", _A3, "Qwen3-VL", "8B", 9.86, _KV_QWEN3_8B, 8.0, "fp8"),
    "Qwen2.5-VL-3B-Instruct":
        _hf("Qwen/Qwen2.5-VL-3B-Instruct", _A25, "Qwen2.5-VL", "3B", 6.99, _KV_QWEN25_3B, 3.0),
    "Qwen2.5-VL-3B-Instruct-AWQ":
        _hf("Qwen/Qwen2.5-VL-3B-Instruct-AWQ", _A25, "Qwen2.5-VL", "3B", 3.17, _KV_QWEN25_3B, 3.0, "awq"),
    "Qwen2.5-VL-7B-Instruct":
        _hf("Qwen/Qwen2.5-VL-7B-Instruct", _A25, "Qwen2.5-VL", "7B", 15.44, _KV_QWEN25_7B, 7.0),
    "Qwen2.5-VL-7B-Instruct-AWQ":
        _hf("Qwen/Qwen2.5-VL-7B-Instruct-AWQ", _A25, "Qwen2.5-VL", "7B", 6.44, _KV_QWEN25_7B, 7.0, "awq"),
}


def _gguf(repo, series, size, kv, quality, stem, sizes, mmproj_sizes):
    """
    stem: phần giữa tên file, ví dụ 'Qwen3VL-8B-Instruct'
    sizes: {quant: gib} cho model, mmproj_sizes: {quant: gib} cho mmproj
    """
    return {
        "repo_id": repo,
        "series": series,
        "size": size,
        "kv": dict(kv),
        "quality": quality,
        "model_files": {q: (f"{stem}-{q}.gguf", gib) for q, gib in sizes.items()},
        "mmproj_files": {q: (f"mmproj-{stem}-{q}.gguf", gib) for q, gib in mmproj_sizes.items()},
        "hf_url": f"https://huggingface.co/{repo}",
    }


# ── GGUF ─────────────────────────────────────────────────────────────────────
GGUF_VL_MODELS = {
    "Qwen3-VL-2B-Instruct-GGUF": _gguf(
        "Qwen/Qwen3-VL-2B-Instruct-GGUF", "Qwen3-VL", "2B", _KV_QWEN3_2B, 2.0,
        "Qwen3VL-2B-Instruct",
        {"Q4_K_M": 1.03, "Q8_0": 1.70, "F16": 3.21},
        {"Q8_0": 0.42, "F16": 0.76}),
    "Qwen3-VL-2B-Thinking-GGUF": _gguf(
        "Qwen/Qwen3-VL-2B-Thinking-GGUF", "Qwen3-VL", "2B", _KV_QWEN3_2B, 2.0,
        "Qwen3VL-2B-Thinking",
        {"Q4_K_M": 1.03, "Q8_0": 1.70, "F16": 3.21},
        {"Q8_0": 0.42, "F16": 0.76}),
    "Qwen3-VL-4B-Instruct-GGUF": _gguf(
        "Qwen/Qwen3-VL-4B-Instruct-GGUF", "Qwen3-VL", "4B", _KV_QWEN3_4B, 4.0,
        "Qwen3VL-4B-Instruct",
        {"Q4_K_M": 2.33, "Q8_0": 3.99, "F16": 7.50},
        {"Q8_0": 0.42, "F16": 0.78}),
    "Qwen3-VL-4B-Thinking-GGUF": _gguf(
        "Qwen/Qwen3-VL-4B-Thinking-GGUF", "Qwen3-VL", "4B", _KV_QWEN3_4B, 4.0,
        "Qwen3VL-4B-Thinking",
        {"Q4_K_M": 2.33, "Q8_0": 3.99, "F16": 7.50},
        {"Q8_0": 0.42, "F16": 0.78}),
    "Qwen3-VL-8B-Instruct-GGUF": _gguf(
        "Qwen/Qwen3-VL-8B-Instruct-GGUF", "Qwen3-VL", "8B", _KV_QWEN3_8B, 8.0,
        "Qwen3VL-8B-Instruct",
        {"Q4_K_M": 4.68, "Q8_0": 8.11, "F16": 15.26},
        {"Q8_0": 0.70, "F16": 1.08}),
    "Qwen3-VL-8B-Thinking-GGUF": _gguf(
        "Qwen/Qwen3-VL-8B-Thinking-GGUF", "Qwen3-VL", "8B", _KV_QWEN3_8B, 8.0,
        "Qwen3VL-8B-Thinking",
        {"Q4_K_M": 4.68, "Q8_0": 8.11, "F16": 15.26},
        {"Q8_0": 0.70, "F16": 1.08}),
}
