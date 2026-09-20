"""
Đọc phần cứng thật và quy ra ngân sách VRAM/RAM dùng cho việc chọn model.

Đây là chỗ DUY NHẤT trong codebase chạm tới torch/psutil để hỏi phần cứng.
Mọi module khác nhận một con số GiB và tính toán thuần trên đó.
"""
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

GIB = 1024 ** 3

# ── Núm hiệu chỉnh ───────────────────────────────────────────────────────────
# Số đo được trên máy thật, không phải lý thuyết. Chỉnh ở đây nếu thấy app
# báo vừa ngân sách nhưng nvidia-smi vẫn OOM.
HEADROOM_RATIO = 0.10    # KV cache phình khi ảnh độ phân giải cao
HEADROOM_FIXED = 0.8     # GiB chừa cho desktop compositor / driver
CPU_RAM_RATIO = 0.6      # phần RAM trống dám giao cho model
CPU_RAM_FALLBACK = 4.0   # GiB, dùng khi không có psutil

FP8_MIN_COMPUTE = (8, 9)  # Ada Lovelace trở lên mới có kernel FP8


def budget_from_free(free_gib: float) -> float:
    """
    VRAM trống (GiB) → ngân sách cho model (GiB).

    Hàm thuần, tách riêng để test được mà không cần GPU.
    """
    return max(0.0, free_gib * (1.0 - HEADROOM_RATIO) - HEADROOM_FIXED)


def get_devices() -> List[Dict[str, Any]]:
    """
    Trả danh sách GPU CUDA. Rỗng nếu không có torch hoặc không có CUDA.

    VRAM trống lấy từ torch.cuda.mem_get_info() — số của driver, nên phản ánh
    cả tiến trình khác đang chiếm VRAM (ComfyUI, game). Khác hẳn
    torch.cuda.memory_reserved() vốn chỉ thấy allocator của chính tiến trình này.
    """
    devices: List[Dict[str, Any]] = []
    try:
        import torch
        if not torch.cuda.is_available():
            return devices
        for i in range(torch.cuda.device_count()):
            free_b, total_b = torch.cuda.mem_get_info(i)
            props = torch.cuda.get_device_properties(i)
            devices.append({
                "index": i,
                "name": props.name,
                "vram_total": total_b / GIB,
                "vram_free": free_b / GIB,
                "vram_used": (total_b - free_b) / GIB,
                "compute": (props.major, props.minor),
            })
    except ImportError:
        logger.debug("Không có torch — bỏ qua thông tin GPU.")
    except Exception as e:
        logger.debug("Lỗi đọc GPU: %s", e)
    return devices


def vram_budget(device_index: int = 0) -> float:
    devices = get_devices()
    if device_index >= len(devices):
        return 0.0
    return budget_from_free(devices[device_index]["vram_free"])


def ram_budget() -> float:
    try:
        import psutil
        return psutil.virtual_memory().available / GIB * CPU_RAM_RATIO
    except ImportError:
        logger.debug("Không có psutil — dùng ngân sách RAM mặc định.")
        return CPU_RAM_FALLBACK


def budget(device_kind: str, device_index: int = 0) -> float:
    """
    Ngân sách cho lựa chọn device của người dùng.

    device_kind: 'cpu' | 'cuda' | 'auto'
    'auto' ưu tiên GPU, không có GPU thì rơi về RAM.
    """
    kind = (device_kind or "auto").strip().lower()
    if kind == "cpu":
        return ram_budget()
    gpu = vram_budget(device_index)
    if gpu > 0.0:
        return gpu
    return ram_budget() if kind == "auto" else 0.0


def supports_fp8(device_index: int = 0) -> bool:
    """FP8 cần compute capability >= 8.9 (Ada/Hopper/Blackwell)."""
    devices = get_devices()
    if device_index >= len(devices):
        return False
    return devices[device_index]["compute"] >= FP8_MIN_COMPUTE
