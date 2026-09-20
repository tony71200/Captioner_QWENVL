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


def _devices_via_nvml() -> List[Dict[str, Any]]:
    """Đọc qua NVML (pynvml / nvidia-ml-py) — nhanh, không cần torch."""
    import warnings

    # Gói `pynvml` cũ tự cảnh báo deprecated mỗi lần import. `nvidia-ml-py`
    # cung cấp cùng tên module và không cảnh báo, nhưng người dùng không cần
    # thấy chuyện này ở mỗi lần khởi động — cả hai đều chạy được.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        import pynvml

    pynvml.nvmlInit()
    try:
        devices = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(h)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode()
            devices.append({
                "index": i,
                "name": name,
                "vram_total": mem.total / GIB,
                "vram_free": mem.free / GIB,
                "vram_used": (mem.total - mem.free) / GIB,
                "compute": (int(major), int(minor)),
            })
        return devices
    finally:
        pynvml.nvmlShutdown()


def _devices_via_smi() -> List[Dict[str, Any]]:
    """Dự phòng khi không có NVML: hỏi nvidia-smi. Chậm hơn (~50ms/lần)."""
    import subprocess

    out = subprocess.run(
        ["nvidia-smi",
         "--query-gpu=index,name,memory.total,memory.free,compute_cap",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10,
    )
    if out.returncode != 0:
        return []
    devices = []
    for line in out.stdout.strip().splitlines():
        idx, name, total_mib, free_mib, cc = [p.strip() for p in line.split(",")]
        total = float(total_mib) * 1024 ** 2 / GIB
        free = float(free_mib) * 1024 ** 2 / GIB
        major, _, minor = cc.partition(".")
        devices.append({
            "index": int(idx),
            "name": name,
            "vram_total": total,
            "vram_free": free,
            "vram_used": total - free,
            "compute": (int(major), int(minor or 0)),
        })
    return devices


def get_devices() -> List[Dict[str, Any]]:
    """
    Trả danh sách GPU NVIDIA. Rỗng nếu không có GPU hoặc không hỏi được driver.

    **Cố ý KHÔNG dùng torch.** torch kéo theo libiomp5md.dll (Intel OpenMP của
    MKL), xung đột với libomp140 của llama-cpp-python và giết tiến trình khi
    caption GGUF (OMP Error #15). Đường GGUF phải sạch torch — xem
    test_duong_gguf_khong_nap_torch.

    VRAM trống lấy từ driver nên phản ánh cả tiến trình khác đang chiếm
    (ComfyUI, game), không chỉ tiến trình này.
    """
    for source in (_devices_via_nvml, _devices_via_smi):
        try:
            devices = source()
            if devices:
                return devices
        except Exception as e:
            logger.debug("%s không dùng được: %s", source.__name__, e)
    return []


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
