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


FIT_ORDER = {"comfortable": 0, "tight": 1, "over": 2}
FIT_ICON = {"comfortable": "\U0001F7E2", "tight": "\U0001F7E1", "over": "\U0001F534"}


@dataclass
class PlanOption:
    """Một cấu hình chạy được cụ thể: model + quant + ctx + offload."""
    backend: str            # "hf" | "gguf"
    model_name: str
    quant: str
    n_ctx: int
    gpu_layers: int         # -1 = toàn bộ; chỉ có nghĩa với GGUF
    max_pixels: int
    mmproj_quant: Optional[str]
    est_gib: float
    fit: str                # "comfortable" | "tight" | "over"
    quality: float

    @property
    def label(self) -> str:
        """
        Nhãn hiển thị — SINH RA từ dữ liệu, không bao giờ là nguồn của dữ liệu.
        Không parse ngược chuỗi này; tra PlanOption qua dict {label: option}.
        """
        parts = [FIT_ICON[self.fit], self.model_name, "\u00b7", self.quant]
        if self.gpu_layers >= 0:
            parts += ["\u00b7", f"{self.gpu_layers} layer GPU"]
        parts += ["\u00b7", f"{self.est_gib:.2f} GiB"]
        return " ".join(parts)


def classify_fit(est_gib: float, budget: float) -> str:
    if budget <= 0:
        return "over"
    if est_gib <= COMFORTABLE_RATIO * budget:
        return "comfortable"
    if est_gib <= budget:
        return "tight"
    return "over"


def _pick_mmproj(entry: dict, wanted: str) -> str:
    """mmproj mong muốn, rơi về bản có sẵn đầu tiên nếu repo không có bản đó."""
    if wanted in entry["mmproj_files"]:
        return wanted
    return next(iter(entry["mmproj_files"]))


def build_option(backend: str, name: str, entry: dict, quant: str,
                 rt: dict, budget: float) -> Optional[PlanOption]:
    """
    Dựng một PlanOption. Trả None nếu quant không có trong repo.

    GGUF: thử offload toàn bộ trước; không vừa thì tính offload một phần.
    Nếu offload một phần không đạt sàn MIN_OFFLOAD_FRAC, vẫn trả cấu hình
    full-offload nhưng đánh 'over' — để người dùng thấy nó tồn tại (đỏ) chứ
    không phải biến mất khỏi danh sách.
    """
    if backend == "hf":
        if quant not in QUANT_FACTOR:
            return None
        est = estimate_hf(entry, quant, rt["n_ctx"], rt["max_pixels"])
        return PlanOption("hf", name, quant, rt["n_ctx"], -1, rt["max_pixels"],
                          None, round(est, 2), classify_fit(est, budget),
                          entry["quality"])

    if quant not in entry["model_files"]:
        return None
    mmproj_quant = _pick_mmproj(entry, rt["mmproj_quant"])
    n_ctx = rt["n_ctx"]

    est_full = estimate_gguf(entry, quant, mmproj_quant, n_ctx, -1)
    full = PlanOption("gguf", name, quant, n_ctx, -1, rt["max_pixels"],
                      mmproj_quant, round(est_full, 2),
                      classify_fit(est_full, budget), entry["quality"])
    if full.fit != "over":
        return full

    # Offload một phần
    total = entry["kv"]["layers"]
    model_gib = entry["model_files"][quant][1]
    mmproj_gib = entry["mmproj_files"][mmproj_quant][1]
    per_layer = (model_gib + kv_cache_gib(entry["kv"], n_ctx)) / total
    usable = budget - mmproj_gib - OVERHEAD_GIB
    if per_layer <= 0 or usable <= 0:
        return full

    gpu_layers = min(int(math.floor(usable / per_layer)), total)
    if gpu_layers < MIN_OFFLOAD_FRAC * total:
        return full

    est = estimate_gguf(entry, quant, mmproj_quant, n_ctx, gpu_layers)
    return PlanOption("gguf", name, quant, n_ctx, gpu_layers, rt["max_pixels"],
                      mmproj_quant, round(est, 2), classify_fit(est, budget),
                      entry["quality"])


def _quants_for(backend: str, entry: dict) -> list:
    if backend == "gguf":
        return [q for q in QUANT_ORDER_GGUF if q in entry["model_files"]]
    native = entry.get("native_quant")
    if native:
        return [native]
    return [q for q in QUANT_ORDER_HF if q in entry["quants"]]


def plan_options(budget: float, backend: str, catalog: dict,
                 supports_fp8: bool = True) -> list:
    """
    Mọi cấu hình chạy được, xếp tốt nhất lên đầu.

    Thứ tự: comfortable -> tight -> over; trong mỗi nhóm quality giảm dần,
    rồi est_gib tăng dần, rồi tên model để kết quả ổn định.
    """
    rt = derive_runtime(budget)
    options = []
    for name, entry in catalog.items():
        if backend == "hf" and entry.get("native_quant") == "fp8" and not supports_fp8:
            continue
        for quant in _quants_for(backend, entry):
            opt = build_option(backend, name, entry, quant, rt, budget)
            if opt is not None:
                options.append(opt)
    options.sort(key=lambda o: (FIT_ORDER[o.fit], -o.quality, o.est_gib,
                                o.model_name, o.quant))
    return options
