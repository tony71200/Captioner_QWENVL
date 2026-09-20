"""
Ước lượng VRAM và xếp hạng cấu hình model.

Hàm thuần: nhận số + dict catalog, trả số + dataclass. KHÔNG import torch,
KHÔNG gọi mạng, KHÔNG đọc đĩa. Nhờ vậy toàn bộ logic chọn model test được
mà không cần GPU hay tải model.

Mọi dung lượng tính bằng GiB.
"""
import math
from dataclasses import dataclass
from typing import Optional

# ── Núm hiệu chỉnh ───────────────────────────────────────────────────────────
# Hệ số VRAM thực tế / trọng số gốc bf16. Số đo được, không phải lý thuyết —
# chỉnh ở đây khi app báo vừa ngân sách nhưng nvidia-smi cho thấy lệch.
QUANT_FACTOR = {
    "bf16": 1.00,
    "8bit": 0.55,
    "4bit": 0.32,
    "fp8": 1.00,    # checkpoint đã quantize sẵn, nạp nguyên trạng
    "awq": 1.00,    # như trên
}

OVERHEAD_GIB = 0.60        # CUDA context + buffer của runtime
ACT_PER_MPIXEL_GIB = 0.35  # activation của vision tower, theo max_pixels

# ── Bậc thang hạ cấp ─────────────────────────────────────────────────────────
QUANT_ORDER_GGUF = ["F16", "Q8_0", "Q4_K_M"]   # chất lượng cao → thấp
QUANT_ORDER_HF = ["bf16", "8bit", "4bit"]
CTX_LADDER = [8192, 4096, 2048]

# Dưới tỉ lệ này thì CPU gánh quá nhiều layer, chậm hơn là chọn quant thấp hơn.
# Dùng tỉ lệ chứ không dùng số tuyệt đối: 8 trên 28 layer khác hẳn 8 trên 36.
MIN_OFFLOAD_FRAC = 0.75

COMFORTABLE_RATIO = 0.80   # dưới mức này của ngân sách thì coi là thoải mái


def kv_cache_gib(kv: dict, n_ctx: int, bytes_per_elem: int = 2) -> float:
    """KV cache cho n_ctx token. Phần mà ước lượng cũ bỏ sót hoàn toàn."""
    return (2 * kv["layers"] * kv["kv_heads"] * kv["head_dim"]
            * n_ctx * bytes_per_elem) / 2 ** 30


def estimate_hf(entry: dict, quant: str, n_ctx: int, max_pixels: int) -> float:
    return (entry["weights_gib"] * QUANT_FACTOR[quant]
            + kv_cache_gib(entry["kv"], n_ctx)
            + ACT_PER_MPIXEL_GIB * max_pixels / 1e6
            + OVERHEAD_GIB)


def estimate_gguf(entry: dict, quant: str, mmproj_quant: str,
                  n_ctx: int, gpu_layers: int) -> float:
    """gpu_layers < 0 nghĩa là offload toàn bộ. mmproj luôn nằm trên GPU."""
    model_gib = entry["model_files"][quant][1]
    total = entry["kv"]["layers"]
    frac = 1.0 if gpu_layers < 0 else min(gpu_layers, total) / total
    return (model_gib * frac
            + entry["mmproj_files"][mmproj_quant][1]
            + kv_cache_gib(entry["kv"], n_ctx) * frac
            + OVERHEAD_GIB)


def derive_runtime(budget: float) -> dict:
    """
    Ngân sách → thông số runtime. Thay thế trực tiếp cho pixel_config /
    gguf_ctx / gguf_layers của VRAM_PROFILES cũ, khác ở chỗ đầu vào là số
    GiB thật chứ không phải tên profile.
    """
    if budget < 3.0:
        return {"n_ctx": 2048, "max_pixels": 512 * 28 * 28, "mmproj_quant": "Q8_0"}
    if budget < 6.0:
        return {"n_ctx": 4096, "max_pixels": 768 * 28 * 28, "mmproj_quant": "Q8_0"}
    if budget < 10.0:
        return {"n_ctx": 4096, "max_pixels": 1280 * 28 * 28, "mmproj_quant": "Q8_0"}
    if budget < 16.0:
        return {"n_ctx": 8192, "max_pixels": 1280 * 28 * 28, "mmproj_quant": "F16"}
    return {"n_ctx": 8192, "max_pixels": 2560 * 28 * 28, "mmproj_quant": "F16"}
