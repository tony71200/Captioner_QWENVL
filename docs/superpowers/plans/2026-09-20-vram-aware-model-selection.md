# VRAM-Aware Model Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Thay giả định 4GB VRAM cứng bằng ngân sách tính từ VRAM trống thật, để app đề xuất và load được model đúng tầm mọi cấu hình từ 4GB tới 24GB+.

**Architecture:** Ba module tách bạch — `utils/hardware.py` là chỗ duy nhất chạm phần cứng, `models_catalog.py` là dữ liệu thuần đã verify với HuggingFace API, `utils/vram_plan.py` là phép tính thuần (không torch, không mạng, không đĩa) sinh ra danh sách cấu hình xếp hạng. `app.py` và `captioner/` nhận thông số cụ thể thay vì tên profile.

**Tech Stack:** Python 3, Gradio 4, PyTorch 2.14+cu130, transformers 5.17, bitsandbytes 0.50.2, llama-cpp-python 0.3.39-preview, huggingface_hub

**Spec:** [`docs/superpowers/specs/2026-09-20-vram-aware-model-selection-design.md`](../specs/2026-09-20-vram-aware-model-selection-design.md)

## Global Constraints

- **Đơn vị**: mọi dung lượng là **GiB** (`bytes / 2**30`). Không có chỗ nào dùng GB thập phân. Vi phạm hằng số này là lớp bug mà spec §3.1 được viết ra để chặn.
- **Python interpreter**: `D:\001_Personal_Proj\Comfy\.venv\Scripts\python.exe`. Trong plan viết tắt là `$PY`. Repo không có `.venv` riêng.
- **Test framework**: KHÔNG dùng pytest. Test là script `assert` chạy bằng `$PY test_vram_plan.py`, khớp phong cách `test_caption.py` sẵn có.
- **`utils/vram_plan.py` không được import torch, huggingface_hub, gradio, hay đọc file.** Chỉ `math`, `dataclasses`, `typing`. Đây là điều kiện để toàn bộ logic chọn model test được không cần GPU.
- **Núm hiệu chỉnh** (`QUANT_FACTOR`, `OVERHEAD_GIB`, `ACT_PER_MPIXEL_GIB`, `HEADROOM_RATIO`, `HEADROOM_FIXED`) phải ở module level kèm comment nói rõ đây là số đo được, chỉnh khi lệch `nvidia-smi`.
- **Ngôn ngữ**: comment và thông báo cho người dùng viết tiếng Việt (khớp badge hiện có trong `app.py`); tên hàm/biến tiếng Anh.
- **Không tự ý thêm dependency.** `psutil` là tùy chọn (đã có nhánh fallback).
- **Commit sau mỗi task.** Không gộp nhiều task vào một commit.

---

## Sai lệch so với spec — đã quyết, ghi lại để không bị đọc là lỗi

Ba điểm phát hiện khi lập plan. Spec sẽ được sửa trong Task 9.

1. **`MIN_OFFLOAD_FRAC = 0.75` thay cho sàn tuyệt đối `gpu_layers >= 8`.**
   Spec §6.5 đặt sàn 8 layer. Với sàn đó, máy 4GB (budget 2.53) sẽ xếp
   Qwen3-VL-4B Q4_K_M ở 20/36 layer là "vừa", mâu thuẫn với chính spec §6.6 (ghi 🔴) và
   với test §10.4. Sàn theo tỉ lệ 0.75 làm mọi khẳng định của spec đúng, và đúng tinh thần
   "dưới mức đó CPU gánh quá nhiều, chậm hơn là chọn quant thấp hơn" — 8 trên 28 layer
   khác hẳn 8 trên 36 layer.

2. **Bậc thang `n_ctx` chỉ dùng trong `preflight`, không dùng trong `plan_options`.**
   Spec §6.5 gạch đầu dòng 3 cho `plan_options` giảm `n_ctx` khi vượt ngân sách.
   Nhưng `derive_runtime(budget)` đã chọn `n_ctx` theo ngân sách rồi; giảm tiếp trong
   `plan_options` sẽ sinh hai lựa chọn cùng model khác ctx trong cùng dropdown, gây rối.
   Bậc thang ctx giữ nguyên ở `preflight` (spec §7 bước 3), nơi nó thực sự cần.

3. **Ví dụ lỗi ở spec §7 và test §10.7 đổi `2.10 GiB` → `1.00 GiB`.**
   Với `budget_now = 2.10`, chuỗi hạ cấp *thành công* (Qwen3-VL-2B Q4_K_M offload 24/28
   layer = 2.09 GiB), nên `preflight` không raise và test §10.7 sai như đã viết. `1.00`
   là giá trị thực sự cạn đường.

**Bổ sung ngoài spec** — hai chỗ trong `app.py` spec §8 bỏ sót, đều đọc trường sắp bị xóa:
- Tab 4 "📦 Model Library" (`app.py:975-999`) đọc `minfo["vram"]` và `minfo["min_vram_4gb"]` → `AttributeError` ngay khi khởi động nếu không sửa. Xử lý ở Task 7.
- `captioner/base.py` có `load_model(self, vram_profile: str, **kwargs)` trong abstract signature và state `_vram_profile`. Xử lý ở Task 6.

---

## File Structure

| File | Trách nhiệm | Task |
|---|---|---|
| `utils/hardware.py` | **Tạo mới.** Chỗ DUY NHẤT đọc phần cứng. Trả device + ngân sách GiB + gate FP8 | 1 |
| `test_vram_plan.py` | **Tạo mới.** Toàn bộ assert, chạy không cần GPU | 1–5 |
| `models_catalog.py` | **Viết lại.** Dữ liệu thuần đã verify. Không còn hàm logic nào | 2 |
| `utils/vram_plan.py` | **Tạo mới.** Ước lượng, `derive_runtime`, `plan_options`, `preflight` | 3–5 |
| `captioner/base.py` | Bỏ `vram_profile` khỏi interface | 6 |
| `captioner/hf_captioner.py` | `AutoModelForImageTextToText`, bỏ chặn Qwen3-VL, nhận thông số cụ thể | 6 |
| `captioner/gguf_captioner.py` | Nhận thông số cụ thể thay `vram_profile` | 6 |
| `app.py` | Bỏ 11 hàm/biến theo profile, thẻ phần cứng + Advanced, preflight, Tab Model Library | 7 |
| `utils/system_info.py` | `get_gpu_info()` gọi `hardware.get_devices()` | 7 |
| `test_caption.py` | Bỏ `--vram-profile`, thêm `--quant`/`--n-ctx` | 8 |
| `README.md`, Tab Help | Bỏ mô tả "4GB VRAM", tả cơ chế ngân sách động | 8 |

**Thứ tự phụ thuộc:** 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9. Task 6 và 7 đều cần 1–5 xong.

---

### Task 1: `utils/hardware.py` — đọc phần cứng thật

**Files:**
- Create: `utils/hardware.py`
- Create: `test_vram_plan.py`

**Interfaces:**
- Consumes: không có (task đầu tiên)
- Produces:
  - `hardware.budget_from_free(free_gib: float) -> float` — hàm thuần
  - `hardware.get_devices() -> list[dict]` với khóa `index, name, vram_total, vram_free, vram_used, compute`; `compute` là tuple `(major, minor)`
  - `hardware.vram_budget(device_index: int = 0) -> float`
  - `hardware.ram_budget() -> float`
  - `hardware.budget(device_kind: str, device_index: int = 0) -> float` — `device_kind` ∈ `"cpu" | "cuda" | "auto"`
  - `hardware.supports_fp8(device_index: int = 0) -> bool`
  - Hằng số: `HEADROOM_RATIO = 0.10`, `HEADROOM_FIXED = 0.8`, `FP8_MIN_COMPUTE = (8, 9)`

- [ ] **Step 1: Viết test thất bại**

Tạo `test_vram_plan.py`:

```python
"""
Test cho logic chọn model theo VRAM. Hàm thuần — KHÔNG cần GPU, không tải model.

Chạy:  D:\\001_Personal_Proj\\Comfy\\.venv\\Scripts\\python.exe test_vram_plan.py
"""
import sys

from utils.hardware import budget_from_free, HEADROOM_RATIO, HEADROOM_FIXED


# ── Task 1: ngân sách phần cứng ──────────────────────────────────────────────

def test_budget_may_tham_chieu():
    """RTX 5070 Ti Laptop: 10.78 GiB trống → 8.90 GiB ngân sách."""
    assert abs(budget_from_free(10.78) - 8.90) < 0.01, budget_from_free(10.78)


def test_budget_khong_bao_gio_am():
    """Card gần đầy phải trả 0.0, không phải số âm."""
    assert budget_from_free(0.5) == 0.0
    assert budget_from_free(0.0) == 0.0


def test_budget_headroom_dung_cong_thuc():
    assert HEADROOM_RATIO == 0.10
    assert HEADROOM_FIXED == 0.8
    assert abs(budget_from_free(24.0) - (24.0 * 0.9 - 0.8)) < 1e-9


# ── Test runner ──────────────────────────────────────────────────────────────

def _run_all():
    failed = []
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            failed.append(name)
            print(f"FAIL  {name}: {e}")
        except Exception as e:
            failed.append(name)
            print(f"ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{'FAILED' if failed else 'OK'} — {len(failed)} lỗi")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
```

- [ ] **Step 2: Chạy test để xác nhận nó fail**

```bash
"D:/001_Personal_Proj/Comfy/.venv/Scripts/python.exe" test_vram_plan.py
```

Expected: FAIL với `ModuleNotFoundError: No module named 'utils.hardware'`

- [ ] **Step 3: Viết `utils/hardware.py`**

```python
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
```

- [ ] **Step 4: Chạy test để xác nhận pass**

```bash
"D:/001_Personal_Proj/Comfy/.venv/Scripts/python.exe" test_vram_plan.py
```

Expected: `PASS test_budget_khong_bao_gio_am`, `PASS test_budget_headroom_dung_cong_thuc`, `PASS test_budget_may_tham_chieu`, `OK — 0 lỗi`

- [ ] **Step 5: Kiểm tra thủ công trên GPU thật**

```bash
"D:/001_Personal_Proj/Comfy/.venv/Scripts/python.exe" -c "from utils import hardware; print(hardware.get_devices()); print('budget', hardware.budget('cuda')); print('fp8', hardware.supports_fp8())"
```

Expected: một dict với `vram_free` khoảng 10.7, `compute` là `(12, 0)`, `budget` khoảng 8.9, `fp8` là `True`. Nếu máy khác thì con số khác — chỉ cần `vram_free` nhỏ hơn `vram_total` và `budget` dương.

- [ ] **Step 6: Commit**

```bash
git add utils/hardware.py test_vram_plan.py
git commit -m "feat: add hardware module reading real free VRAM"
```

---

### Task 2: `models_catalog.py` — viết lại thành dữ liệu đã verify

**Files:**
- Modify: `models_catalog.py` (viết lại toàn bộ, 330 dòng hiện tại)
- Modify: `test_vram_plan.py` (thêm test schema)

**Interfaces:**
- Consumes: không có
- Produces:
  - `HF_VL_MODELS: dict[str, dict]` — mỗi entry có `repo_id, arch, series, size, weights_gib, native_quant, quants, kv, quality, hf_url`
  - `GGUF_VL_MODELS: dict[str, dict]` — mỗi entry có `repo_id, series, size, kv, quality, model_files, mmproj_files, hf_url`
  - `model_files` / `mmproj_files` là `dict[str, tuple[str, float]]` = `{quant: (filename, gib)}`
  - `kv` là `dict` với khóa `layers, kv_heads, head_dim`
  - `CATALOG_VERIFIED: dict` với khóa `checked, source, org`
  - **Không export hàm nào.** `get_hf_model_names`, `get_gguf_model_names`, `get_vram_profile_names`, `get_model_info_html`, `VRAM_PROFILES` bị xóa.

- [ ] **Step 1: Viết test thất bại**

Thêm vào `test_vram_plan.py`, ngay trước `# ── Test runner ──`:

```python
# ── Task 2: schema catalog ───────────────────────────────────────────────────

from models_catalog import HF_VL_MODELS, GGUF_VL_MODELS, CATALOG_VERIFIED


def test_catalog_khong_con_truong_cu():
    """min_vram_4gb / vram / VRAM_PROFILES là gốc của giả định 4GB cứng."""
    import models_catalog
    assert not hasattr(models_catalog, "VRAM_PROFILES")
    for name, info in {**HF_VL_MODELS, **GGUF_VL_MODELS}.items():
        assert "min_vram_4gb" not in info, name
        assert "vram" not in info, name


def test_catalog_hf_du_truong():
    need = {"repo_id", "arch", "series", "size", "weights_gib",
            "native_quant", "quants", "kv", "quality", "hf_url"}
    for name, info in HF_VL_MODELS.items():
        assert need <= set(info), f"{name} thiếu {need - set(info)}"
        assert info["repo_id"].startswith("Qwen/"), name
        assert {"layers", "kv_heads", "head_dim"} <= set(info["kv"]), name


def test_catalog_hf_native_quant_la_don_le():
    """Không được chồng bitsandbytes lên checkpoint đã quantize sẵn."""
    for name, info in HF_VL_MODELS.items():
        if info["native_quant"] is not None:
            assert info["quants"] == [info["native_quant"]], name


def test_catalog_gguf_du_truong():
    need = {"repo_id", "series", "size", "kv", "quality",
            "model_files", "mmproj_files", "hf_url"}
    for name, info in GGUF_VL_MODELS.items():
        assert need <= set(info), f"{name} thiếu {need - set(info)}"
        assert info["mmproj_files"], f"{name} không có mmproj"
        for quant, value in info["model_files"].items():
            assert isinstance(value, tuple) and len(value) == 2, f"{name}/{quant}"
            filename, gib = value
            assert filename.endswith(".gguf"), f"{name}/{quant}"
            assert isinstance(gib, float), f"{name}/{quant}"


def test_catalog_khong_co_file_split():
    """File -split-NNNNN-of-NNNNN.gguf không tải được bằng hf_hub_download."""
    for name, info in GGUF_VL_MODELS.items():
        for quant, (filename, _gib) in info["model_files"].items():
            assert "-split-" not in filename, f"{name}/{quant}"


def test_catalog_dung_don_vi_gib():
    """Bắt lỗi trộn bytes / GB thập phân / GiB — mọi số phải trong (0, 70)."""
    for name, info in HF_VL_MODELS.items():
        assert 0.0 < info["weights_gib"] < 70.0, f"{name}={info['weights_gib']}"
    for name, info in GGUF_VL_MODELS.items():
        for quant, (_f, gib) in info["model_files"].items():
            assert 0.0 < gib < 70.0, f"{name}/{quant}={gib}"
        for quant, (_f, gib) in info["mmproj_files"].items():
            assert 0.0 < gib < 5.0, f"{name}/mmproj/{quant}={gib}"


def test_catalog_co_ngay_verify():
    assert CATALOG_VERIFIED["checked"] == "2026-09-20"
    assert CATALOG_VERIFIED["org"] == "Qwen"
```

- [ ] **Step 2: Chạy test để xác nhận fail**

```bash
"D:/001_Personal_Proj/Comfy/.venv/Scripts/python.exe" test_vram_plan.py
```

Expected: FAIL — `ImportError: cannot import name 'CATALOG_VERIFIED'`

- [ ] **Step 3: Viết lại `models_catalog.py`**

Thay toàn bộ nội dung file bằng:

```python
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
```

- [ ] **Step 4: Chạy test để xác nhận pass**

```bash
"D:/001_Personal_Proj/Comfy/.venv/Scripts/python.exe" test_vram_plan.py
```

Expected: 10 dòng `PASS`, `OK — 0 lỗi`

- [ ] **Step 5: Đối chiếu tên file GGUF với HuggingFace thật**

```bash
"D:/001_Personal_Proj/Comfy/.venv/Scripts/python.exe" -c "
import json, urllib.request
from models_catalog import GGUF_VL_MODELS
bad = 0
for name, info in GGUF_VL_MODELS.items():
    d = json.load(urllib.request.urlopen(f\"https://huggingface.co/api/models/{info['repo_id']}?blobs=true\"))
    have = {s['rfilename']: s.get('size') for s in d['siblings']}
    for q, (fn, gib) in list(info['model_files'].items()) + list(info['mmproj_files'].items()):
        if fn not in have:
            print('MISSING', info['repo_id'], fn); bad += 1
        elif abs(have[fn]/2**30 - gib) > 0.02:
            print('SIZE', fn, round(have[fn]/2**30,2), 'vs', gib); bad += 1
print('bad =', bad)
"
```

Expected: `bad = 0`. Nếu có dòng `MISSING` hoặc `SIZE` thì sửa catalog theo số thật rồi chạy lại. Bước này cần mạng; nếu không có mạng, bỏ qua và ghi chú lại trong commit.

- [ ] **Step 6: Commit**

```bash
git add models_catalog.py test_vram_plan.py
git commit -m "feat: rewrite model catalog with verified HuggingFace sizes"
```

---

### Task 3: `utils/vram_plan.py` — ước lượng VRAM

**Files:**
- Create: `utils/vram_plan.py`
- Modify: `test_vram_plan.py`

**Interfaces:**
- Consumes: schema catalog từ Task 2 (`entry["kv"]`, `entry["weights_gib"]`, `entry["model_files"][q] == (filename, gib)`, `entry["mmproj_files"][q] == (filename, gib)`)
- Produces:
  - `kv_cache_gib(kv: dict, n_ctx: int, bytes_per_elem: int = 2) -> float`
  - `estimate_hf(entry: dict, quant: str, n_ctx: int, max_pixels: int) -> float`
  - `estimate_gguf(entry: dict, quant: str, mmproj_quant: str, n_ctx: int, gpu_layers: int) -> float`
  - `derive_runtime(budget: float) -> dict` với khóa `n_ctx, max_pixels, mmproj_quant`
  - Hằng số: `QUANT_FACTOR`, `OVERHEAD_GIB = 0.60`, `ACT_PER_MPIXEL_GIB = 0.35`, `QUANT_ORDER_GGUF = ["F16", "Q8_0", "Q4_K_M"]`, `QUANT_ORDER_HF = ["bf16", "8bit", "4bit"]`, `CTX_LADDER = [8192, 4096, 2048]`, `MIN_OFFLOAD_FRAC = 0.75`, `COMFORTABLE_RATIO = 0.80`

- [ ] **Step 1: Viết test thất bại**

Thêm vào `test_vram_plan.py` trước `# ── Test runner ──`:

```python
# ── Task 3: ước lượng VRAM ───────────────────────────────────────────────────

from utils.vram_plan import (
    kv_cache_gib, estimate_hf, estimate_gguf, derive_runtime,
    OVERHEAD_GIB, MIN_OFFLOAD_FRAC,
)


def test_kv_cache_dung_so_that():
    """Qwen3-VL-8B (36 layer, 8 kv-head, head_dim 128) ở ctx 8192 tốn đúng 1.125 GiB."""
    kv8b = {"layers": 36, "kv_heads": 8, "head_dim": 128}
    assert abs(kv_cache_gib(kv8b, 8192) - 1.125) < 1e-9
    assert abs(kv_cache_gib(kv8b, 4096) - 0.5625) < 1e-9
    kv2b = {"layers": 28, "kv_heads": 8, "head_dim": 128}
    assert abs(kv_cache_gib(kv2b, 2048) - 0.21875) < 1e-9


def test_estimate_gguf_8b_q4_full_offload():
    """8B Q4_K_M + mmproj Q8_0 + ctx 4096 = 6.54 GiB. Catalog cũ bỏ sót mmproj và KV."""
    entry = GGUF_VL_MODELS["Qwen3-VL-8B-Instruct-GGUF"]
    est = estimate_gguf(entry, "Q4_K_M", "Q8_0", 4096, -1)
    assert abs(est - 6.54) < 0.01, est


def test_estimate_gguf_offload_mot_phan_re_hon():
    """gpu_layers < tổng số layer phải cho ước lượng nhỏ hơn full offload."""
    entry = GGUF_VL_MODELS["Qwen3-VL-8B-Instruct-GGUF"]
    full = estimate_gguf(entry, "Q8_0", "Q8_0", 4096, -1)
    part = estimate_gguf(entry, "Q8_0", "Q8_0", 4096, 31)
    assert abs(full - 9.97) < 0.01, full
    assert part < full, (part, full)
    assert abs(part - 8.77) < 0.02, part


def test_estimate_hf_8b_4bit():
    """8B 4-bit, ctx 4096, max_pixels 1280*28*28 = 6.74 GiB."""
    entry = HF_VL_MODELS["Qwen3-VL-8B-Instruct"]
    est = estimate_hf(entry, "4bit", 4096, 1280 * 28 * 28)
    assert abs(est - 6.74) < 0.01, est


def test_estimate_hf_fp8_that_su_khong_vua_may_12gb():
    """Lỗi gốc: catalog cũ ghi 8B-FP8 là 7.5GB nên app mời load rồi OOM."""
    entry = HF_VL_MODELS["Qwen3-VL-8B-Instruct-FP8"]
    est = estimate_hf(entry, "fp8", 4096, 1280 * 28 * 28)
    assert abs(est - 11.37) < 0.01, est
    assert est > 8.90


def test_derive_runtime_theo_ngan_sach():
    assert derive_runtime(8.90) == {
        "n_ctx": 4096, "max_pixels": 1280 * 28 * 28, "mmproj_quant": "Q8_0"}
    assert derive_runtime(2.53) == {
        "n_ctx": 2048, "max_pixels": 512 * 28 * 28, "mmproj_quant": "Q8_0"}
    assert derive_runtime(24.0) == {
        "n_ctx": 8192, "max_pixels": 2560 * 28 * 28, "mmproj_quant": "F16"}


def test_vram_plan_khong_import_torch():
    """vram_plan phải thuần — nếu nó kéo torch vào thì test chạy không cần GPU sẽ vỡ."""
    import inspect
    import utils.vram_plan as vp
    src = inspect.getsource(vp)
    for banned in ("import torch", "import gradio", "huggingface_hub", "open("):
        assert banned not in src, banned
```

- [ ] **Step 2: Chạy test để xác nhận fail**

```bash
"D:/001_Personal_Proj/Comfy/.venv/Scripts/python.exe" test_vram_plan.py
```

Expected: FAIL — `ModuleNotFoundError: No module named 'utils.vram_plan'`

- [ ] **Step 3: Viết `utils/vram_plan.py`**

```python
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
```

(`math`, `dataclass`, `Optional` sẽ dùng ở Task 4 — cứ import sẵn, Task 4 nối tiếp vào cùng file.)

- [ ] **Step 4: Chạy test để xác nhận pass**

```bash
"D:/001_Personal_Proj/Comfy/.venv/Scripts/python.exe" test_vram_plan.py
```

Expected: 17 dòng `PASS`, `OK — 0 lỗi`

- [ ] **Step 5: Commit**

```bash
git add utils/vram_plan.py test_vram_plan.py
git commit -m "feat: add VRAM estimation with KV cache and mmproj accounted"
```

---

### Task 4: `PlanOption` và `plan_options` — sinh và xếp hạng cấu hình

**Files:**
- Modify: `utils/vram_plan.py` (nối vào cuối)
- Modify: `test_vram_plan.py`

**Interfaces:**
- Consumes: `kv_cache_gib`, `estimate_hf`, `estimate_gguf`, `derive_runtime`, các hằng số từ Task 3
- Produces:
  - `@dataclass PlanOption` với field `backend, model_name, quant, n_ctx, gpu_layers, max_pixels, mmproj_quant, est_gib, fit, quality` và **property** `label -> str`
  - `classify_fit(est_gib: float, budget: float) -> str` → `"comfortable" | "tight" | "over"`
  - `build_option(backend: str, name: str, entry: dict, quant: str, rt: dict, budget: float) -> Optional[PlanOption]`
  - `plan_options(budget: float, backend: str, catalog: dict, supports_fp8: bool = True) -> list[PlanOption]`
  - `FIT_ICON: dict`, `FIT_ORDER: dict`

**Lưu ý cho người triển khai:** `label` là **property** chứ không phải field lưu sẵn — nhãn phải luôn sinh ra từ dữ liệu, không bao giờ là nguồn của dữ liệu. Đây chính là lỗi `_extract_vram_from_label()` cũ (regex bóc số VRAM ra từ chuỗi hiển thị). `app.py` **không được parse ngược label**; nó giữ một dict `{label: PlanOption}` (xem Task 7).

- [ ] **Step 1: Viết test thất bại**

Thêm vào `test_vram_plan.py`:

```python
# ── Task 4: sinh và xếp hạng cấu hình ────────────────────────────────────────

from utils.vram_plan import PlanOption, plan_options, classify_fit

BUDGET_12GB = 8.90   # RTX 5070 Ti Laptop: 10.78 trống
BUDGET_4GB = 2.53    # card 4GB: ~3.7 trống


def test_classify_fit():
    assert classify_fit(7.0, 8.90) == "comfortable"   # <= 0.80 * 8.90
    assert classify_fit(8.50, 8.90) == "tight"
    assert classify_fit(9.50, 8.90) == "over"
    assert classify_fit(1.0, 0.0) == "over"           # không có GPU


def test_plan_may_12gb_chon_8b_q4():
    """Kỳ vọng cốt lõi: máy 12GB được mời 8B Q4_K_M, không phải 8B FP8."""
    opts = plan_options(BUDGET_12GB, "gguf", GGUF_VL_MODELS)
    top = opts[0]
    assert top.model_name == "Qwen3-VL-8B-Instruct-GGUF", top.model_name
    assert top.quant == "Q4_K_M", top.quant
    assert top.fit == "comfortable", top.fit
    assert abs(top.est_gib - 6.54) < 0.01, top.est_gib


def test_plan_may_12gb_hf_chon_8b_4bit():
    opts = plan_options(BUDGET_12GB, "hf", HF_VL_MODELS, supports_fp8=True)
    top = opts[0]
    assert top.model_name == "Qwen3-VL-8B-Instruct", top.model_name
    assert top.quant == "4bit", top.quant
    assert top.fit == "comfortable", top.fit


def test_plan_may_4gb_chon_2b_q4():
    """Phân khúc 4GB vẫn được phục vụ đúng — chỉ là không còn hard-code."""
    opts = plan_options(BUDGET_4GB, "gguf", GGUF_VL_MODELS)
    top = opts[0]
    assert top.model_name == "Qwen3-VL-2B-Instruct-GGUF", top.model_name
    assert top.quant == "Q4_K_M", top.quant
    for o in opts:
        if "8B" in o.model_name:
            assert o.fit != "comfortable", (o.model_name, o.quant, o.est_gib)


def test_plan_4gb_khong_nhan_offload_qua_thap():
    """4B Q4_K_M trên máy 4GB chỉ chạy được 20/36 layer → phải bị đánh 'over'."""
    opts = plan_options(BUDGET_4GB, "gguf", GGUF_VL_MODELS)
    four_b = [o for o in opts
              if o.model_name == "Qwen3-VL-4B-Instruct-GGUF" and o.quant == "Q4_K_M"]
    assert four_b, "4B Q4_K_M phải xuất hiện trong danh sách, dù là 🔴"
    assert four_b[0].fit == "over", four_b[0].est_gib
    assert four_b[0].gpu_layers == -1, "không đạt sàn offload thì báo full-offload"


def test_plan_12gb_nhan_offload_mot_phan():
    """8B Q8_0 chạy 31/36 layer = 86% > sàn 0.75 → giữ, xếp 'tight'."""
    opts = plan_options(BUDGET_12GB, "gguf", GGUF_VL_MODELS)
    q8 = [o for o in opts
          if o.model_name == "Qwen3-VL-8B-Instruct-GGUF" and o.quant == "Q8_0"][0]
    assert q8.gpu_layers == 31, q8.gpu_layers
    assert q8.fit == "tight", (q8.fit, q8.est_gib)
    assert q8.gpu_layers >= MIN_OFFLOAD_FRAC * 36


def test_plan_gate_fp8_theo_compute_capability():
    """Card không có kernel FP8 thì không được mời model FP8."""
    opts = plan_options(BUDGET_12GB, "hf", HF_VL_MODELS, supports_fp8=False)
    assert not [o for o in opts if o.quant == "fp8"]
    opts_on = plan_options(BUDGET_12GB, "hf", HF_VL_MODELS, supports_fp8=True)
    assert [o for o in opts_on if o.quant == "fp8"]


def test_plan_thu_tu_comfortable_truoc_tight_truoc_over():
    rank = {"comfortable": 0, "tight": 1, "over": 2}
    for backend, catalog in (("gguf", GGUF_VL_MODELS), ("hf", HF_VL_MODELS)):
        opts = plan_options(BUDGET_12GB, backend, catalog)
        seq = [rank[o.fit] for o in opts]
        assert seq == sorted(seq), (backend, seq)


def test_plan_label_sinh_ra_tu_du_lieu():
    opts = plan_options(BUDGET_12GB, "gguf", GGUF_VL_MODELS)
    top = opts[0]
    assert top.label.startswith("🟢")
    assert top.model_name in top.label
    assert top.quant in top.label
    assert f"{top.est_gib:.2f}" in top.label
    assert len({o.label for o in opts}) == len(opts), "label phải là khóa duy nhất"
```

- [ ] **Step 2: Chạy test để xác nhận fail**

```bash
"D:/001_Personal_Proj/Comfy/.venv/Scripts/python.exe" test_vram_plan.py
```

Expected: FAIL — `ImportError: cannot import name 'PlanOption'`

- [ ] **Step 3: Nối vào cuối `utils/vram_plan.py`**

```python
FIT_ORDER = {"comfortable": 0, "tight": 1, "over": 2}
FIT_ICON = {"comfortable": "🟢", "tight": "🟡", "over": "🔴"}


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
        parts = [FIT_ICON[self.fit], self.model_name, "·", self.quant]
        if self.gpu_layers >= 0:
            parts += ["·", f"{self.gpu_layers} layer GPU"]
        parts += ["·", f"{self.est_gib:.2f} GiB"]
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
    full-offload nhưng đánh 'over' — để người dùng thấy nó tồn tại (🔴) chứ
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

    Thứ tự: comfortable → tight → over; trong mỗi nhóm quality giảm dần,
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
```

- [ ] **Step 4: Chạy test để xác nhận pass**

```bash
"D:/001_Personal_Proj/Comfy/.venv/Scripts/python.exe" test_vram_plan.py
```

Expected: 26 dòng `PASS`, `OK — 0 lỗi`

- [ ] **Step 5: In ra bảng lựa chọn thật để mắt người đọc được**

```bash
"D:/001_Personal_Proj/Comfy/.venv/Scripts/python.exe" -c "
from utils.vram_plan import plan_options
from models_catalog import GGUF_VL_MODELS, HF_VL_MODELS
for b, cat in (('gguf', GGUF_VL_MODELS), ('hf', HF_VL_MODELS)):
    print('===', b, 'budget 8.90')
    for o in plan_options(8.90, b, cat)[:6]:
        print('  ', o.label)
"
```

Expected: nhóm GGUF mở đầu bằng `🟢 Qwen3-VL-8B-Instruct-GGUF · Q4_K_M · 6.54 GiB`; nhóm HF mở đầu bằng `🟢 Qwen3-VL-8B-Instruct · 4bit · 6.74 GiB`. Không có dòng nào là 8B-FP8 ở nhóm 🟢.

- [ ] **Step 6: Commit**

```bash
git add utils/vram_plan.py test_vram_plan.py
git commit -m "feat: rank concrete model configs against real VRAM budget"
```

---

### Task 5: `preflight` — kiểm tra trước khi load và tự hạ cấp

**Files:**
- Modify: `utils/vram_plan.py` (nối vào cuối)
- Modify: `test_vram_plan.py`

**Interfaces:**
- Consumes: `PlanOption`, `build_option`, `estimate_*`, `QUANT_ORDER_*`, `CTX_LADDER` từ Task 3–4
- Produces:
  - `class VramPlanError(RuntimeError)`
  - `preflight(option: PlanOption, budget_now: float, catalog: dict, auto_downgrade: bool = True) -> tuple[PlanOption, Optional[str]]`
  - Trả `(option, None)` khi vừa; `(option_mới, thông_báo_str)` khi đã hạ cấp; raise `VramPlanError` khi không hạ được hoặc `auto_downgrade=False`

- [ ] **Step 1: Viết test thất bại**

Thêm vào `test_vram_plan.py`:

```python
# ── Task 5: preflight và hạ cấp ──────────────────────────────────────────────

from utils.vram_plan import preflight, VramPlanError, build_option, derive_runtime


def _option_8b_q8_full():
    """8B Q8_0 offload toàn bộ — 9.97 GiB, vượt ngân sách 8.90 của máy tham chiếu."""
    entry = GGUF_VL_MODELS["Qwen3-VL-8B-Instruct-GGUF"]
    rt = derive_runtime(BUDGET_12GB)
    return PlanOption("gguf", "Qwen3-VL-8B-Instruct-GGUF", "Q8_0", rt["n_ctx"],
                      -1, rt["max_pixels"], "Q8_0", 9.97, "over", 8.0)


def test_preflight_vua_thi_tra_nguyen():
    opts = plan_options(BUDGET_12GB, "gguf", GGUF_VL_MODELS)
    top = opts[0]
    result, note = preflight(top, BUDGET_12GB, GGUF_VL_MODELS)
    assert result is top
    assert note is None


def test_preflight_tu_ha_cap_quant():
    """8B Q8_0 (9.97) trên ngân sách 8.90 → hạ xuống Q4_K_M (6.54)."""
    result, note = preflight(_option_8b_q8_full(), BUDGET_12GB, GGUF_VL_MODELS)
    assert result.quant == "Q4_K_M", result.quant
    assert result.model_name == "Qwen3-VL-8B-Instruct-GGUF"
    assert note is not None
    assert "9.97" in note and "8.90" in note and "6.54" in note, note


def test_preflight_tat_ha_cap_thi_chan():
    try:
        preflight(_option_8b_q8_full(), BUDGET_12GB, GGUF_VL_MODELS,
                  auto_downgrade=False)
    except VramPlanError as e:
        assert "9.97" in str(e) and "8.90" in str(e), str(e)
        return
    raise AssertionError("phải raise VramPlanError khi tắt tự hạ cấp")


def test_preflight_het_duong_thi_raise():
    """Ngân sách 1.00 GiB: mọi quant, mọi ctx, mọi model nhỏ hơn đều không vừa."""
    opts = plan_options(BUDGET_12GB, "gguf", GGUF_VL_MODELS)
    top = opts[0]   # 8B Q4_K_M, 6.54 GiB
    try:
        preflight(top, 1.00, GGUF_VL_MODELS)
    except VramPlanError as e:
        assert "6.54" in str(e) and "1.00" in str(e), str(e)
        return
    raise AssertionError("phải raise khi không còn cấu hình nào vừa")


def test_preflight_ha_sang_model_nho_hon():
    """Ngân sách 3.50: 8B không cách nào vừa → phải rơi sang 4B hoặc 2B."""
    opts = plan_options(BUDGET_12GB, "gguf", GGUF_VL_MODELS)
    top = opts[0]
    result, note = preflight(top, 3.50, GGUF_VL_MODELS)
    assert result.model_name != "Qwen3-VL-8B-Instruct-GGUF", result.model_name
    assert result.est_gib <= 3.50, result.est_gib
    assert note is not None
```

- [ ] **Step 2: Chạy test để xác nhận fail**

```bash
"D:/001_Personal_Proj/Comfy/.venv/Scripts/python.exe" test_vram_plan.py
```

Expected: FAIL — `ImportError: cannot import name 'preflight'`

- [ ] **Step 3: Nối vào cuối `utils/vram_plan.py`**

```python
class VramPlanError(RuntimeError):
    """Không còn cấu hình nào vừa ngân sách hiện tại."""


def _quants_below(backend: str, entry: dict, current_quant: Optional[str]) -> list:
    """Các quant thấp hơn current_quant. current_quant=None → toàn bộ."""
    ranked = _quants_for(backend, entry)
    if current_quant is None or current_quant not in ranked:
        return ranked
    return ranked[ranked.index(current_quant) + 1:]


def _downgrade_candidates(option: PlanOption, catalog: dict):
    """
    Sinh (model_name, quant, n_ctx) theo đúng thứ tự hạ cấp của spec §7:
      1. quant thấp hơn của cùng model (offload một phần tự được thử trong
         build_option)
      2. giảm n_ctx một bậc, thử lại các quant
      3. model nhỏ hơn (quality giảm dần), bắt đầu lại từ quant cao nhất
    """
    ctx_below = [c for c in CTX_LADDER if c <= option.n_ctx] or [CTX_LADDER[-1]]

    same = catalog[option.model_name]
    for quant in _quants_below(option.backend, same, option.quant):
        yield option.model_name, quant, option.n_ctx

    for n_ctx in ctx_below[1:]:
        for quant in [option.quant] + _quants_below(option.backend, same, option.quant):
            yield option.model_name, quant, n_ctx

    smaller = sorted(
        [(n, e) for n, e in catalog.items() if e["quality"] < option.quality],
        key=lambda item: -item[1]["quality"],
    )
    for name, entry in smaller:
        for n_ctx in ctx_below:
            for quant in _quants_below(option.backend, entry, None):
                yield name, quant, n_ctx


def preflight(option: PlanOption, budget_now: float, catalog: dict,
              auto_downgrade: bool = True):
    """
    Kiểm tra NGAY TRƯỚC KHI LOAD, không phải lúc dựng UI — VRAM trống thay đổi
    khi người dùng mở ComfyUI hoặc game giữa chừng.

    Trả (option, None) nếu vừa, (option_mới, thông_báo) nếu đã hạ cấp.
    Raise VramPlanError nếu tắt tự hạ cấp, hoặc hạ hết cách vẫn không vừa.
    """
    entry = catalog[option.model_name]
    if option.backend == "gguf":
        est = estimate_gguf(entry, option.quant, option.mmproj_quant,
                            option.n_ctx, option.gpu_layers)
    else:
        est = estimate_hf(entry, option.quant, option.n_ctx, option.max_pixels)

    if est <= budget_now:
        return option, None

    need = f"{est:.2f} GiB"
    have = f"{budget_now:.2f} GiB"

    if not auto_downgrade:
        raise VramPlanError(
            f"{option.model_name} {option.quant} cần {need}, chỉ còn {have}. "
            f"Bật 'Tự hạ cấp khi thiếu VRAM', hoặc tự chọn cấu hình nhỏ hơn."
        )

    for name, quant, n_ctx in _downgrade_candidates(option, catalog):
        rt = {"n_ctx": n_ctx, "max_pixels": option.max_pixels,
              "mmproj_quant": option.mmproj_quant or "Q8_0"}
        cand = build_option(option.backend, name, catalog[name], quant, rt, budget_now)
        if cand is not None and cand.fit != "over":
            return cand, (
                f"{option.model_name} {option.quant} cần {need} nhưng chỉ còn {have} "
                f"→ đã chuyển sang {cand.model_name} {cand.quant} "
                f"({cand.est_gib:.2f} GiB)"
            )

    raise VramPlanError(
        f"{option.model_name} {option.quant} cần {need}, chỉ còn {have}. "
        f"Đã thử mọi mức quant và ctx thấp hơn. Hãy đóng bớt ứng dụng đang dùng "
        f"GPU, hoặc chuyển Device sang CPU."
    )
```

- [ ] **Step 4: Chạy test để xác nhận pass**

```bash
"D:/001_Personal_Proj/Comfy/.venv/Scripts/python.exe" test_vram_plan.py
```

Expected: 32 dòng `PASS`, `OK — 0 lỗi`

- [ ] **Step 5: Commit**

```bash
git add utils/vram_plan.py test_vram_plan.py
git commit -m "feat: add preflight VRAM check with ordered auto-downgrade"
```

---

### Task 6: `captioner/` — nhận thông số cụ thể, bỏ chặn Qwen3-VL

**Files:**
- Modify: `captioner/base.py:17-22` (abstract `load_model`), `captioner/base.py:13-15` (`__init__`), `captioner/base.py:44-51` (property + `__repr__`)
- Modify: `captioner/hf_captioner.py:56-175` (toàn bộ `load_model`)
- Modify: `captioner/gguf_captioner.py:88-250` (chữ ký + phần đọc profile)

**Interfaces:**
- Consumes: không phụ thuộc `vram_plan` (captioner nhận số rời, không nhận `PlanOption` — giữ `captioner/` độc lập với module chọn model)
- Produces:
  - `HFCaptioner.load_model(model_id: str, quant: str, n_ctx: int, max_pixels: int, device: str = "auto", use_flash_attn: bool = False) -> None`
  - `GGUFCaptioner.load_model(model_path: str, mmproj_path: str, n_ctx: int, n_gpu_layers: int, device: str = "auto", n_batch: int = 512, image_max_tokens: int = 4096) -> None`
  - `BaseCaptioner.load_model(self, **kwargs) -> None` (abstract, không còn `vram_profile`)
  - `HFCaptioner.runtime_device` và `.fallback_reason` như `GGUFCaptioner` đã có (để `app.py` xử lý đồng nhất; HF luôn đặt `runtime_device` = device đã resolve, **không** thêm fallback OOM→CPU — nằm ngoài phạm vi theo spec §7)

- [ ] **Step 1: Sửa `captioner/base.py`**

Thay ba chỗ:

```python
    def __init__(self):
        self._loaded = False
        self._runtime_desc: Optional[str] = None

    @abstractmethod
    def load_model(self, **kwargs) -> None:
        """
        Nạp model. Backend nhận thông số cụ thể (quant, n_ctx, n_gpu_layers…),
        do utils/vram_plan tính ra — không nhận tên profile.
        """
        ...
```

và:

```python
    @property
    def runtime_desc(self) -> Optional[str]:
        return self._runtime_desc

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(loaded={self._loaded}, {self._runtime_desc})"
```

Xóa hẳn property `vram_profile` và field `_vram_profile`.

- [ ] **Step 2: Viết lại `HFCaptioner.load_model`**

Thay toàn bộ thân hàm `load_model` trong `captioner/hf_captioner.py` bằng:

```python
    def load_model(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        quant: str = "4bit",
        n_ctx: int = 4096,
        max_pixels: int = 1280 * 28 * 28,
        device: str = "auto",
        use_flash_attn: bool = False,
        **kwargs,
    ) -> None:
        """
        Nạp model Qwen-VL qua HuggingFace Transformers.

        quant: 'bf16' | '8bit' | '4bit' | 'fp8' | 'awq'
               'fp8'/'awq' nghĩa là checkpoint đã quantize sẵn — nạp nguyên
               trạng, KHÔNG chồng bitsandbytes lên trên.
        """
        from transformers import AutoProcessor, AutoModelForImageTextToText

        device_kind = _pick_hf_device(device)
        self._device_kind = device_kind
        self.runtime_device = device_kind
        self.fallback_reason = ""
        logger.info("HF device: yêu cầu='%s' → thực tế='%s'", device, device_kind)

        if self._loaded:
            self.unload_model()

        self.model_id = model_id
        self._max_pixels = max_pixels
        self._n_ctx = n_ctx

        # bitsandbytes không chạy trên CPU
        if device_kind == "cpu" and quant in ("4bit", "8bit"):
            logger.warning(
                "bitsandbytes (%s) không hỗ trợ CPU — rơi về float32. "
                "RAM cao và chậm hơn nhiều.", quant,
            )
            quant = "bf16"

        model_kwargs = {
            "dtype": torch.float32 if device_kind == "cpu" else torch.bfloat16,
            "device_map": "cpu" if device_kind == "cpu" else "auto",
            "low_cpu_mem_usage": True,
        }

        # Chỉ áp bitsandbytes lên checkpoint CHƯA quantize.
        bnb_config = self._build_bnb_config(quant)
        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config

        self._configure_torch_runtime()

        if device_kind == "cuda":
            if use_flash_attn and quant == "bf16":
                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Bật Flash Attention 2.")
            else:
                model_kwargs["attn_implementation"] = "sdpa"

        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id, **model_kwargs
            )
        except Exception as e:
            raise RuntimeError(
                f"Không nạp được model HuggingFace '{model_id}'. "
                f"Lỗi gốc: {e}"
            ) from e

        self.model.eval()
        if hasattr(self.model, "config"):
            self.model.config.use_cache = True
        if getattr(self.model, "generation_config", None) is not None:
            self.model.generation_config.use_cache = True

        try:
            self.processor = AutoProcessor.from_pretrained(
                model_id,
                min_pixels=256 * 28 * 28,
                max_pixels=max_pixels,
            )
        except Exception as e:
            self.unload_model()
            raise RuntimeError(
                f"Không nạp được processor cho '{model_id}'. Lỗi gốc: {e}"
            ) from e

        self._loaded = True
        self._runtime_desc = f"quant={quant} ctx={n_ctx} device={device_kind}"
        logger.info("Đã nạp model HF. %s", self._runtime_desc)
```

Sửa kèm:
- `__init__`: thêm `self.runtime_device = "unknown"`, `self.fallback_reason = ""`, `self._max_pixels = 0`, `self._n_ctx = 0`; xóa `self._pixel_config`.
- `_build_bnb_config`: đổi điều kiện — `"4bit"` → `BitsAndBytesConfig(load_in_4bit=True, ...)`, `"8bit"` → `BitsAndBytesConfig(load_in_8bit=True)`, **mọi giá trị khác (`bf16`, `fp8`, `awq`) trả `None`**.
- Xóa hẳn khối `if "Qwen3-VL" in model_id: raise RuntimeError(...)` (`hf_captioner.py:145-150`).
- Xóa `from transformers import ... Qwen2VLForConditionalGeneration` và khối `try/except ImportError` bao quanh nó ở đầu hàm.
- Xóa import `VRAM_PROFILES` và nhánh fallback profile.
- Trong `caption_image`, thay `self._pixel_config` nếu còn tham chiếu.
- Cập nhật docstring đầu file: bỏ "UltraLow (4GB) / LowVRAM / NormalVRAM / HighVRAM profiles", thay bằng "nhận quant + n_ctx + max_pixels do utils/vram_plan tính".

- [ ] **Step 3: Sửa `GGUFCaptioner.load_model`**

Đổi chữ ký và bỏ phần đọc profile:

```python
    def load_model(
        self,
        model_path: str = "",
        mmproj_path: str = "",
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        device: str = "auto",
        n_batch: int = 512,
        image_max_tokens: int = 4096,
        **kwargs,
    ) -> None:
```

Trong thân hàm:
- Xóa khối `try: from models_catalog import GGUF_VL_MODELS, VRAM_PROFILES ... except ImportError` và biến `profile`, `model_info`, `gguf_defaults`.
- Thay khối tính `n_gpu_layers` bằng: `if device_kind != "cuda": n_gpu_layers = 0` (CPU luôn 0; GPU dùng đúng giá trị truyền vào).
- Thay các lời gọi `_coerce_runtime_value(kwargs.get(...), gguf_defaults.get(...))` bằng chính tham số mới: `n_ctx`, `n_batch`, `image_max_tokens`. Giữ `image_min_tokens = max(1024, kwargs.get("image_min_tokens") or 1024)`, `top_k = int(kwargs.get("top_k") or 0)`, `pool_size = int(kwargs.get("pool_size") or 4194304)`.
- Thay `self._vram_profile = vram_profile` bằng
  `self._runtime_desc = f"ctx={n_ctx} gpu_layers={n_gpu_layers} device={device_kind}"`.
- Trong `unload_model()`, thay `self._vram_profile = None` bằng `self._runtime_desc = None`.
- **Giữ nguyên không đụng**: `_build_chat_handler`, khối `try/except` fallback CUDA→CPU (`gguf_captioner.py:224-243`), `_filter_kwargs_for_callable`, `_cpu_cuda_hidden_env`, `_looks_like_cuda_oom_or_init_error`.

- [ ] **Step 4: Kiểm tra import và chữ ký**

```bash
"D:/001_Personal_Proj/Comfy/.venv/Scripts/python.exe" -c "
import inspect
from captioner.hf_captioner import HFCaptioner
from captioner.gguf_captioner import GGUFCaptioner
for cls in (HFCaptioner, GGUFCaptioner):
    params = list(inspect.signature(cls.load_model).parameters)
    print(cls.__name__, params)
    assert 'vram_profile' not in params, 'còn sót vram_profile'
import captioner.hf_captioner as m, inspect as i
src = i.getsource(m)
assert 'Qwen2VLForConditionalGeneration' not in src, 'còn dùng class Qwen2-VL'
assert 'AutoModelForImageTextToText' in src
assert 'VRAM_PROFILES' not in src
print('OK')
"
```

Expected: hai dòng chữ ký không chứa `vram_profile`, rồi `OK`

- [ ] **Step 5: Nạp thật một model GGUF để xác nhận không vỡ**

```bash
"D:/001_Personal_Proj/Comfy/.venv/Scripts/python.exe" -c "
from captioner.gguf_captioner import GGUFCaptioner
import inspect
print(inspect.signature(GGUFCaptioner.load_model))
"
```

Expected: in ra chữ ký mới. Nạp model thật sẽ làm ở Task 7 Step 7 qua WebUI — ở đây chỉ cần import sạch.

- [ ] **Step 6: Commit**

```bash
git add captioner/base.py captioner/hf_captioner.py captioner/gguf_captioner.py
git commit -m "feat: captioners take explicit runtime params, unblock Qwen3-VL on HF"
```

---

### Task 7: `app.py` — thẻ phần cứng, dropdown động, preflight khi load

**Files:**
- Modify: `app.py:19-21` (import), `app.py:57` (`VRAM_PROFILE_NAMES`), `app.py:107-181` (nhóm hàm profile), `app.py:205-330` (resolve/render/refresh), `app.py:379-446` (`load_model`), `app.py:595-597` (`on_vram_change`), `app.py:735-775` (UI Setup), `app.py:816-850` (markdown tham chiếu + wire events), `app.py:975-999` (Tab Model Library)
- Modify: `utils/system_info.py:75-113` (`get_gpu_info`)

**Interfaces:**
- Consumes: toàn bộ API của Task 1–6
- Produces: WebUI chạy được, không còn radio VRAM Profile

**Xóa hẳn** (spec §8): `_vram_profile_bucket`, `_available_vram_for_profile`, `_hf_quant_key_for_profile`, `_estimate_hf_required_vram`, `_pick_gguf_variant`, `_extract_vram_from_label`, `_estimate_gguf_required_vram`, `_hf_model_supported_by_loader`, `_is_model_compatible`, `_extract_model_name`, `_model_status_icon`, `_model_status_rank`, `VRAM_PROFILE_NAMES`, `on_vram_change`, `vram_radio`, `vram_desc`, và bảng markdown "📊 VRAM / RAM Reference" + "🟢 4GB-Friendly Models" ở cột phải tab Setup.

**Giữ, sửa nhẹ**: `_hf_storage_dir`, `_hf_is_available`, `_gguf_file_index`, `_download_hf_model`, `_badge`, `_expected_download_target`.

- [ ] **Step 1: Sửa import và state ở đầu `app.py`**

```python
from models_catalog import HF_VL_MODELS, GGUF_VL_MODELS, CATALOG_VERIFIED
from utils import hardware
from utils.vram_plan import PlanOption, plan_options, preflight, VramPlanError
```

Xóa `VRAM_PROFILES`, `get_model_info_html` khỏi import. Thay `VRAM_PROFILE_NAMES = ...` bằng:

```python
# label → PlanOption. Gradio dropdown chỉ trả về chuỗi label; tra ngược qua dict
# này thay vì parse chuỗi (lỗi cũ: _extract_vram_from_label bóc số bằng regex).
_OPTION_BY_LABEL: dict = {}

MODEL_STATUS_LEGEND = (
    "🟢 thoải mái trong ngân sách &nbsp;|&nbsp; "
    "🟡 sát ngưỡng &nbsp;|&nbsp; "
    "🔴 vượt ngân sách (vẫn chọn được — preflight sẽ hạ cấp hoặc chặn) &nbsp;|&nbsp; "
    "⬇️ chưa có sẵn trên máy, sẽ tải khi Load"
)
```

- [ ] **Step 2: Thêm nhóm hàm mới (thay chỗ nhóm hàm profile đã xóa)**

```python
def _backend_kind(backend: str) -> str:
    return "hf" if "HuggingFace" in backend else "gguf"


def _catalog_for(backend: str) -> dict:
    return HF_VL_MODELS if _backend_kind(backend) == "hf" else GGUF_VL_MODELS


def _device_kind(device_choice: str) -> str:
    return {"CPU": "cpu", "GPU": "cuda", "Auto": "auto"}.get(device_choice, "auto")


def _current_budget(device_choice: str, budget_override) -> float:
    """Advanced để trống = tự suy; điền số = ghi đè."""
    try:
        override = float(budget_override)
    except (TypeError, ValueError):
        override = 0.0
    if override > 0:
        return override
    return hardware.budget(_device_kind(device_choice))


def _hardware_html(device_choice: str, budget_override) -> str:
    budget = _current_budget(device_choice, budget_override)
    devices = hardware.get_devices()
    if not devices or _device_kind(device_choice) == "cpu":
        return _badge(
            f"Chế độ CPU — ngân sách RAM cho model: <b>{budget:.2f} GiB</b>", "info")
    d = devices[0]
    return _badge(
        f"GPU {d['index']} — {d['name']}<br>"
        f"VRAM {d['vram_free']:.2f} / {d['vram_total']:.2f} GiB trống · "
        f"compute {d['compute'][0]}.{d['compute'][1]}<br>"
        f"Ngân sách model: <b>{budget:.2f} GiB</b>",
        "info",
    )


def _resolve_gguf_local_assets(option: PlanOption, llm_dir: Path) -> dict:
    """Tìm file .gguf đã có trên máy cho đúng cấu hình đã chọn."""
    info = GGUF_VL_MODELS[option.model_name]
    model_filename = info["model_files"][option.quant][0]
    mmproj_filename = info["mmproj_files"][option.mmproj_quant][0]
    file_index = _gguf_file_index(llm_dir)
    model_path = file_index.get(model_filename.lower())
    mmproj_path = file_index.get(mmproj_filename.lower())
    return {
        "model_filename": model_filename,
        "mmproj_filename": mmproj_filename,
        "model_path": model_path,
        "mmproj_path": mmproj_path,
        "available": bool(model_path and mmproj_path),
    }


def _is_option_available(option: PlanOption, llm_dir: Path) -> bool:
    if option.backend == "hf":
        return _hf_is_available(option.model_name, llm_dir)
    return _resolve_gguf_local_assets(option, llm_dir)["available"]


def _refresh_options(backend, device_choice, budget_override, current_label):
    """Dựng lại dropdown từ ngân sách hiện tại."""
    global _OPTION_BY_LABEL
    budget = _current_budget(device_choice, budget_override)
    options = plan_options(
        budget, _backend_kind(backend), _catalog_for(backend),
        supports_fp8=hardware.supports_fp8(),
    )
    _OPTION_BY_LABEL = {o.label: o for o in options}
    labels = list(_OPTION_BY_LABEL)
    if not labels:
        return (
            gr.update(choices=[], value=None),
            _badge("Không tìm thấy cấu hình nào cho backend này.", "warning"),
            _hardware_html(device_choice, budget_override),
        )
    selected = current_label if current_label in _OPTION_BY_LABEL else labels[0]
    return (
        gr.update(choices=labels, value=selected),
        _render_option_info(selected),
        _hardware_html(device_choice, budget_override),
    )


def _render_option_info(label: str) -> str:
    option = _OPTION_BY_LABEL.get(label)
    if option is None:
        return _badge("Chưa chọn cấu hình nào.", "warning")
    catalog = HF_VL_MODELS if option.backend == "hf" else GGUF_VL_MODELS
    info = catalog[option.model_name]
    available = _is_option_available(option, LLM_DIR)

    if option.backend == "hf":
        detail = f"Thư mục lưu: {_hf_storage_dir(LLM_DIR, option.model_name)}"
    else:
        assets = _resolve_gguf_local_assets(option, LLM_DIR)
        detail = (
            f"Model: {assets['model_filename']}<br>"
            f"MMProj: {assets['mmproj_filename']} ({option.mmproj_quant})<br>"
            f"Thư mục tải về: {_expected_download_target(option.backend, option.model_name, LLM_DIR)}"
        )

    offload = "toàn bộ layer" if option.gpu_layers < 0 else f"{option.gpu_layers} layer trên GPU"
    return (
        f'<div style="font-size:13px;color:#cbd5e1;line-height:1.6;">'
        f'<div style="margin-bottom:6px;">{option.label}</div>'
        f'<div style="color:#94a3b8;">Quant {option.quant} · ctx {option.n_ctx} · '
        f'{offload} · ước lượng {option.est_gib:.2f} GiB</div>'
        f'<div style="margin-top:8px;color:#94a3b8;">'
        f'{"Đã có trên máy" if available else "⬇️ Sẽ tải về khi bấm Load"}</div>'
        f'<div style="margin-top:8px;color:#64748b;">{detail}</div>'
        f'<div style="margin-top:8px;color:#64748b;">Repo: {info["repo_id"]} '
        f'(đã verify {CATALOG_VERIFIED["checked"]})</div>'
        f'<div style="margin-top:8px;">'
        f'<a href="{info["hf_url"]}" target="_blank" style="color:#38bdf8;font-size:12px;">'
        f'📦 Xem repo gốc ↗</a></div></div>'
    )
```

Sửa `_expected_download_target` cho khớp chữ ký mới:

```python
def _expected_download_target(backend_kind: str, model_name: str, llm_dir: Path) -> Path:
    if backend_kind == "hf":
        return _hf_storage_dir(llm_dir, model_name)
    return llm_dir / "GGUF" / model_name
```

Sửa `_download_gguf_assets` nhận `PlanOption`:

```python
def _download_gguf_assets(option: PlanOption, llm_dir: Path) -> dict:
    from huggingface_hub import hf_hub_download

    info = GGUF_VL_MODELS[option.model_name]
    assets = _resolve_gguf_local_assets(option, llm_dir)
    target_dir = _expected_download_target("gguf", option.model_name, llm_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    for key, filename in (("model_path", assets["model_filename"]),
                          ("mmproj_path", assets["mmproj_filename"])):
        if not assets[key]:
            assets[key] = Path(hf_hub_download(
                repo_id=info["repo_id"],
                filename=filename,
                local_dir=str(target_dir),
            ))
    assets["available"] = bool(assets["model_path"] and assets["mmproj_path"])
    return assets
```

- [ ] **Step 3: Viết lại `load_model`**

```python
def _apply_overrides(option: PlanOption, ctx_override, layers_override, pixels_override) -> PlanOption:
    """Advanced để trống = giữ giá trị tự suy."""
    import dataclasses

    changes = {}
    for field, value, cast in (("n_ctx", ctx_override, int),
                               ("gpu_layers", layers_override, int),
                               ("max_pixels", pixels_override, int)):
        try:
            parsed = cast(value)
        except (TypeError, ValueError):
            continue
        if field == "max_pixels" and parsed > 0:
            changes[field] = parsed * 28 * 28
        elif field == "gpu_layers":
            changes[field] = parsed
        elif parsed > 0:
            changes[field] = parsed
    return dataclasses.replace(option, **changes) if changes else option


def load_model(backend, selected_label, auto_downgrade, device_choice,
               budget_override, ctx_override, layers_override, pixels_override):
    global _captioner
    try:
        option = _OPTION_BY_LABEL.get(selected_label)
        if option is None:
            yield _badge("Chọn một cấu hình model trước.", "warning")
            return

        option = _apply_overrides(option, ctx_override, layers_override, pixels_override)
        catalog = _catalog_for(backend)

        # ── Preflight: đọc LẠI VRAM ngay lúc này ────────────────────────────
        if _device_kind(device_choice) != "cpu":
            budget_now = _current_budget(device_choice, budget_override)
            try:
                option, note = preflight(option, budget_now, catalog, bool(auto_downgrade))
            except VramPlanError as e:
                yield _badge(str(e), "error")
                return
            if note:
                yield _badge(note, "warning")

        available = _is_option_available(option, LLM_DIR)
        if not available:
            yield _badge(f"⏳ Chưa có trên máy — đang tải {option.model_name} về {LLM_DIR}…", "loading")

        if _captioner is not None:
            _captioner.unload_model()
            _captioner = None

        device_param = _device_kind(device_choice)

        if option.backend == "hf":
            if not available:
                _download_hf_model(option.model_name, LLM_DIR)
            yield _badge(f"⏳ Đang nạp model HuggingFace trên {device_choice}…", "loading")
            c = HFCaptioner()
            c.load_model(
                model_id=str(_hf_storage_dir(LLM_DIR, option.model_name)),
                quant=option.quant,
                n_ctx=option.n_ctx,
                max_pixels=option.max_pixels,
                device=device_param,
                use_flash_attn=False,
            )
        else:
            assets = _resolve_gguf_local_assets(option, LLM_DIR)
            if not available:
                assets = _download_gguf_assets(option, LLM_DIR)
            yield _badge(f"⏳ Đang nạp model GGUF trên {device_choice}…", "loading")
            c = GGUFCaptioner()
            c.load_model(
                model_path=str(assets["model_path"]),
                mmproj_path=str(assets["mmproj_path"]),
                n_ctx=option.n_ctx,
                n_gpu_layers=option.gpu_layers,
                device=device_param,
            )

        _captioner = c
        if getattr(c, "runtime_device", "") == "cpu-fallback":
            reason = getattr(c, "fallback_reason", "")
            yield _badge(f"GPU lỗi → tự chuyển sang CPU{f' ({reason})' if reason else ''}", "warning")
            yield _badge(f"Đã nạp — {option.model_name} {option.quant} | Device: CPU (fallback)", "success")
        else:
            yield _badge(
                f"Đã nạp — {option.model_name} {option.quant} · ctx {option.n_ctx} · "
                f"~{option.est_gib:.2f} GiB | Device: {device_choice}", "success")
    except Exception as e:
        _captioner = None
        yield _badge(f"Nạp thất bại: {e}", "error")
```

- [ ] **Step 4: Sửa UI tab Setup**

Thay khối `vram_radio` / `vram_desc` / `flash_cb` (`app.py:768-780`) bằng:

```python
                auto_downgrade_cb = gr.Checkbox(
                    label="Tự hạ cấp khi thiếu VRAM",
                    value=True,
                    info="Tắt thì app sẽ chặn thay vì tự đổi sang cấu hình nhỏ hơn.",
                )

                with gr.Accordion("Advanced", open=False):
                    gr.Markdown("Để trống = tự suy từ ngân sách. Điền số để ghi đè.")
                    budget_box = gr.Number(label="Ngân sách VRAM (GiB)", value=None)
                    ctx_box = gr.Number(label="n_ctx", value=None, precision=0)
                    layers_box = gr.Number(
                        label="GPU layers (-1 = toàn bộ)", value=None, precision=0)
                    pixels_box = gr.Number(
                        label="max_pixels (×28×28)", value=None, precision=0)
```

Sửa phần khởi tạo (`app.py:735-740`):

```python
                initial_backend = "GGUF (llama-cpp)"
                initial_device = "Auto"
                _initial_budget = hardware.budget(_device_kind(initial_device))
                _initial_options = plan_options(
                    _initial_budget, _backend_kind(initial_backend),
                    _catalog_for(initial_backend),
                    supports_fp8=hardware.supports_fp8(),
                )
                _OPTION_BY_LABEL = {o.label: o for o in _initial_options}
                initial_labels = list(_OPTION_BY_LABEL)
                initial_label = initial_labels[0] if initial_labels else None
```

Thêm thẻ phần cứng ngay dưới `device_radio`, thay `system_info_panel`:

```python
                    hardware_panel = gr.HTML(value=_hardware_html(initial_device, None))
                    refresh_btn = gr.Button("🔄 Làm mới ngân sách", size="sm")
                    system_info_panel = gr.HTML(
                        value=get_system_info_html(initial_device),
                        elem_classes=["system-info-panel"],
                    )
```

Đổi `model_dd` dùng `initial_labels` / `initial_label`, và `model_info` dùng
`_render_option_info(initial_label)`.

Thay toàn bộ markdown cột phải (`app.py:817-834`) bằng:

```python
            with gr.Column(scale=2):
                gr.Markdown(f"""
### 📊 Cách app chọn model

Ngân sách = VRAM **trống thật** × 0.90 − 0.8 GiB.
Mọi cấu hình được ước lượng gồm trọng số + mmproj + KV cache + overhead,
rồi xếp hạng theo ngân sách đó.

- 🟢 dưới 80% ngân sách
- 🟡 vừa khít
- 🔴 vượt — preflight sẽ hạ cấp hoặc chặn

Số dung lượng lấy từ HuggingFace API, verify ngày **{CATALOG_VERIFIED["checked"]}**,
chỉ nhận repo của org `{CATALOG_VERIFIED["org"]}`.

### 🖥️ Chế độ CPU
- GGUF: toàn bộ layer trên CPU (`n_gpu_layers=0`)
- HF: `device_map="cpu"`, float32, **không có bitsandbytes**
- Nên dùng GGUF Q4_K_M model 2B/4B
""")
```

- [ ] **Step 5: Nối lại event**

Thay khối wire events (`app.py:840-858`) bằng:

```python
        _refresh_inputs = [backend_radio, device_radio, budget_box, model_dd]
        _refresh_outputs = [model_dd, model_info, hardware_panel]

        device_radio.change(on_device_change, [device_radio], [system_info_panel])
        device_radio.change(_refresh_options, _refresh_inputs, _refresh_outputs)
        backend_radio.change(_refresh_options, _refresh_inputs, _refresh_outputs)
        budget_box.change(_refresh_options, _refresh_inputs, _refresh_outputs)
        refresh_btn.click(_refresh_options, _refresh_inputs, _refresh_outputs)
        model_dd.change(_render_option_info, [model_dd], [model_info])

        load_btn.click(
            load_model,
            inputs=[backend_radio, model_dd, auto_downgrade_cb, device_radio,
                    budget_box, ctx_box, layers_box, pixels_box],
            outputs=[model_status],
        )
        unload_btn.click(unload_model, outputs=[model_status])
```

- [ ] **Step 6: Sửa Tab 4 Model Library và `utils/system_info.py`**

Tab 4 (`app.py:975-999`) — đang đọc `minfo["vram"]` và `minfo["min_vram_4gb"]` vừa bị xóa, sẽ `KeyError` ngay khi khởi động. Thay bằng:

```python
        rows_hf = []
        for mname, minfo in HF_VL_MODELS.items():
            rows_hf.append([mname, minfo["series"], minfo["size"],
                            f"{minfo['weights_gib']:.2f}",
                            minfo["native_quant"] or ", ".join(minfo["quants"]),
                            minfo["repo_id"]])
        gr.Dataframe(
            headers=["Tên", "Series", "Size", "Trọng số (GiB)", "Quant", "HF Repo ID"],
            value=rows_hf, interactive=False, wrap=True,
        )

        gr.Markdown("## GGUF Models")
        rows_gguf = []
        for mname, minfo in GGUF_VL_MODELS.items():
            quants = ", ".join(f"{q} {gib:.2f}GiB"
                               for q, (_fn, gib) in minfo["model_files"].items())
            rows_gguf.append([mname, minfo["series"], minfo["size"],
                              quants, minfo["repo_id"]])
        gr.Dataframe(
            headers=["Tên", "Series", "Size", "Quant có sẵn (GiB)", "HF Repo ID"],
            value=rows_gguf, interactive=False, wrap=True,
        )
```

`utils/system_info.py` — thay thân `get_gpu_info()` bằng:

```python
def get_gpu_info() -> List[Dict[str, Any]]:
    """
    Thông tin GPU cho panel HTML. VRAM trống lấy từ utils.hardware
    (driver thật), không dùng torch.cuda.memory_reserved() nữa — số đó chỉ
    thấy allocator của chính tiến trình này nên luôn báo card gần như trống.
    """
    from utils.hardware import get_devices

    gpus = []
    for d in get_devices():
        total = d["vram_total"]
        used = d["vram_used"]
        gpus.append({
            "index": d["index"],
            "name": d["name"],
            "vram_total": round(total, 2),
            "vram_free": round(d["vram_free"], 2),
            "vram_used": round(used, 2),
            "vram_percent": round(used / total * 100, 1) if total > 0 else 0.0,
            "compute": f"{d['compute'][0]}.{d['compute'][1]}",
        })
    return gpus
```

- [ ] **Step 7: Chạy app và kiểm tra thủ công**

```bash
"D:/001_Personal_Proj/Comfy/.venv/Scripts/python.exe" app.py --llm-dir llm
```

Kiểm tra theo thứ tự, mỗi mục phải đạt:
1. App khởi động, không traceback (đặc biệt Tab 📦 Model Library — chỗ hay `KeyError` nhất).
2. Tab Setup hiện thẻ GPU với VRAM trống thật và ngân sách, KHÔNG còn radio "VRAM Profile".
3. Dropdown mở đầu bằng `🟢 Qwen3-VL-8B-Instruct-GGUF · Q4_K_M · 6.54 GiB`.
4. Mở ComfyUI (hoặc chương trình khác chiếm VRAM) → bấm **🔄 Làm mới ngân sách** → ngân sách tụt, thứ tự dropdown đổi.
5. Chọn một dòng 🔴 → bấm Load → hiện badge vàng "đã chuyển sang…" rồi nạp được.
6. Bỏ tick "Tự hạ cấp" → chọn lại dòng 🔴 đó → Load → hiện badge đỏ có đủ con số cần/còn.
7. Load `🟢 ... Q4_K_M` → so `nvidia-smi` với `est_gib`. Lệch > 15% thì chỉnh `OVERHEAD_GIB` / `QUANT_FACTOR` trong `utils/vram_plan.py` rồi chạy lại `test_vram_plan.py` (các assert có biên ±0.01 sẽ phải cập nhật theo).
8. Đổi backend sang HuggingFace → chọn `Qwen3-VL-8B-Instruct · 4bit` → Load. **Đây là đường code chưa từng chạy thành công** (trước đây bị `raise` chặn). Nếu lỗi, đọc kỹ traceback: thiếu `accelerate`/`bitsandbytes` là vấn đề môi trường, khác với lỗi API.
9. Tab 🖼️ Single Image → caption một ảnh → có chữ ra.

- [ ] **Step 8: Chạy lại toàn bộ test**

```bash
"D:/001_Personal_Proj/Comfy/.venv/Scripts/python.exe" test_vram_plan.py
```

Expected: `OK — 0 lỗi`

- [ ] **Step 9: Commit**

```bash
git add app.py utils/system_info.py
git commit -m "feat: drive model selection from real VRAM budget in the UI"
```

---

### Task 8: `test_caption.py` và tài liệu

**Files:**
- Modify: `test_caption.py:114-124` (chữ ký `caption_image`), `:190-260` (`_run_backend`), `:329-341` (`--vram-profile`), `:372-400` (`main`)
- Modify: `app.py` Tab 📖 Help (`app.py:1000-1063`)
- Modify: `README.md`

**Interfaces:**
- Consumes: API captioner mới (Task 6), `vram_plan` + `hardware` (Task 1–5)
- Produces: CLI chạy được không cần WebUI, tài liệu khớp hành vi mới

- [ ] **Step 1: Sửa `test_caption.py`**

Trong `caption_image()`: bỏ tham số `vram_profile`, thêm `quant: str = ""`, `n_ctx: int = 0`.
Khi `quant`/`n_ctx` để trống, tự suy:

```python
    from utils import hardware
    from utils.vram_plan import derive_runtime

    budget_gib = hardware.budget(device)
    rt = derive_runtime(budget_gib)
    n_ctx = int(n_ctx) or rt["n_ctx"]
    max_pixels = rt["max_pixels"]
    if not quant:
        quant = "Q4_K_M" if backend == "gguf" else "4bit"
    logger.info("Ngân sách %.2f GiB → quant=%s ctx=%d", budget_gib, quant, n_ctx)
```

Trong `_run_backend()`: thay `vram_profile=vram_profile` ở hai lời gọi `cap.load_model()` bằng tham số mới:

```python
        cap.load_model(
            model_path=model_path,
            mmproj_path=mmproj_path,
            n_ctx=n_ctx,
            n_gpu_layers=0 if device == "cpu" else -1,
            device=device,
        )
```

và cho HF:

```python
        cap.load_model(
            model_id=model_id,
            quant=quant,
            n_ctx=n_ctx,
            max_pixels=max_pixels,
            device=device,
        )
```

Trong `_build_parser()`: xóa hẳn `--vram-profile`, thêm:

```python
    p.add_argument(
        "--quant",
        default="",
        metavar="Q",
        help=("Mức quantize. GGUF: Q4_K_M | Q8_0 | F16. "
              "HF: bf16 | 8bit | 4bit. Để trống = tự suy từ VRAM trống."),
    )
    p.add_argument(
        "--n-ctx",
        type=int,
        default=0,
        metavar="N",
        help="Độ dài context. 0 = tự suy từ VRAM trống.",
    )
```

Trong `main()`: xóa hẳn dict `vram_map` và biến `vram_profile`, truyền `quant=args.quant, n_ctx=args.n_ctx` vào `caption_image()`.

- [ ] **Step 2: Chạy thử CLI**

```bash
"D:/001_Personal_Proj/Comfy/.venv/Scripts/python.exe" test_caption.py --help
```

Expected: có `--quant` và `--n-ctx`, KHÔNG còn `--vram-profile`.

Rồi chạy thật trên một ảnh có sẵn (đổi đường dẫn cho đúng máy):

```bash
"D:/001_Personal_Proj/Comfy/.venv/Scripts/python.exe" test_caption.py --image <đường-dẫn-ảnh> --backend gguf
```

Expected: log in ra dòng `Ngân sách ... GiB → quant=Q4_K_M ctx=4096`, rồi ra caption.

- [ ] **Step 3: Sửa Tab Help trong `app.py`**

Trong markdown Tab 📖 Help:
- Mục "3. Load a Model": bỏ "choose ... VRAM profile"; đổi chú giải icon thành 🟢 thoải mái / 🟡 sát ngưỡng / 🔴 vượt ngân sách / ⬇️ chưa tải.
- Bảng "Device Selection": dòng GPU đổi `n_gpu_layers from profile` thành `n_gpu_layers do vram_plan tính`.
- **Xóa hẳn** bảng "### VRAM Profiles" (4 dòng UltraLow/LowVRAM/NormalVRAM/HighVRAM).
- Thêm mục mới thay vào:

```markdown
### Ngân sách VRAM
App đọc VRAM **trống thật** từ driver, trừ headroom (10% + 0.8 GiB), rồi xếp hạng
mọi cấu hình model theo con số đó. Không còn profile cố định.

Bấm **🔄 Làm mới ngân sách** sau khi đóng ComfyUI hoặc game để app tính lại.
Accordion **Advanced** cho phép ghi đè ngân sách, `n_ctx`, GPU layers, `max_pixels`.
```

- Mục "Standalone Test": đổi ví dụ `--vram-profile` thành `--quant Q8_0 --n-ctx 8192`.
- Troubleshooting: dòng "CUDA OOM" đổi thành "bật 'Tự hạ cấp khi thiếu VRAM', hoặc bấm Làm mới ngân sách rồi chọn dòng 🟢". **Xóa hẳn** dòng "HF Qwen3-VL error: Use GGUF backend..." — chặn đó không còn.

- [ ] **Step 4: Sửa `README.md`**

- Bỏ mọi câu quảng cáo "4GB VRAM" trong tiêu đề/mô tả; thay bằng "tự chọn model theo VRAM trống thật, từ 4GB tới 24GB+".
- Thay bảng VRAM Profiles (nếu có) bằng mô tả ngân sách và 3 mức 🟢🟡🔴.
- Cập nhật bảng model: bỏ cột "4GB OK", thêm cột dung lượng GiB đã verify.
- Ghi rõ catalog verify ngày 2026-09-20, chỉ nhận repo org `Qwen/`, và 30B/32B GGUF bị loại vì file split.

- [ ] **Step 5: Sửa dòng hero trong `app.py`**

`app.py:721` đang ghi `4 GB VRAM support`. Đổi thành:

```python
      <p>Qwen3-VL · Qwen2.5-VL &nbsp;|&nbsp; HuggingFace &amp; GGUF &nbsp;|&nbsp; CPU &amp; GPU &nbsp;|&nbsp; tự chọn model theo VRAM trống</p>
```

Và docstring đầu file `app.py:1-4`: bỏ `4GB VRAM`.

- [ ] **Step 6: Quét sạch tham chiếu còn sót**

```bash
grep -rn "vram_profile\|VRAM_PROFILES\|min_vram_4gb\|UltraLow\|LowVRAM\|NormalVRAM\|HighVRAM\|_extract_vram_from_label\|Qwen2VLForConditionalGeneration" --include=*.py --include=*.md . | grep -v "docs/superpowers/"
```

Expected: không có dòng nào. Mọi kết quả còn lại là chỗ sót, phải sửa.

- [ ] **Step 7: Chạy lại toàn bộ test + khởi động app**

```bash
"D:/001_Personal_Proj/Comfy/.venv/Scripts/python.exe" test_vram_plan.py
```

Expected: `OK — 0 lỗi`

```bash
"D:/001_Personal_Proj/Comfy/.venv/Scripts/python.exe" -c "import app; print('app import OK')"
```

Expected: `app import OK` (import được nghĩa là toàn bộ UI dựng được, kể cả Tab Model Library).

- [ ] **Step 8: Commit**

```bash
git add test_caption.py app.py README.md
git commit -m "docs: replace 4GB profile docs with dynamic VRAM budget"
```

---

### Task 9: Đồng bộ spec với ba sai lệch đã quyết

**Files:**
- Modify: `docs/superpowers/specs/2026-09-20-vram-aware-model-selection-design.md`

**Interfaces:**
- Consumes: kết quả thực tế của Task 4–5
- Produces: spec khớp code, để lần đọc sau không hiểu nhầm là code sai

- [ ] **Step 1: Sửa §6.5 — sàn offload**

Đổi câu `Chỉ giữ nếu gpu_layers ≥ 8 (dưới mức đó CPU gánh quá nhiều...)` thành:

```markdown
  Chỉ giữ nếu `gpu_layers ≥ MIN_OFFLOAD_FRAC × total` với `MIN_OFFLOAD_FRAC = 0.75`
  (dùng tỉ lệ chứ không phải số tuyệt đối: 8 trên 28 layer khác hẳn 8 trên 36).
  Không đạt sàn thì vẫn trả cấu hình full-offload đánh `over` để người dùng
  thấy nó tồn tại (🔴), chứ không biến mất khỏi danh sách.
```

- [ ] **Step 2: Sửa §6.5 — bỏ bậc thang ctx khỏi `plan_options`**

Xóa gạch đầu dòng `Nếu vượt ngân sách vì KV cache, thử lại với n_ctx giảm một bậc (8192 → 4096 → 2048).` và thay bằng:

```markdown
- `n_ctx` do `derive_runtime(budget)` quyết định, KHÔNG giảm tiếp trong
  `plan_options` (sẽ sinh hai lựa chọn cùng model khác ctx trong cùng dropdown,
  gây rối). Bậc thang ctx chỉ dùng trong `preflight` — xem §7 bước 3.
```

- [ ] **Step 3: Sửa §7 và §10.7 — con số ví dụ**

Trong khối lỗi ở §7, đổi `2.10 GiB` thành `1.00 GiB`:

```markdown
> ❌ Qwen3-VL-8B-Instruct-GGUF Q4_K_M cần 6.54 GiB, chỉ còn 1.00 GiB.
> Đã thử mọi mức quant và ctx thấp hơn. Hãy đóng bớt ứng dụng đang dùng GPU,
> hoặc chuyển Device sang CPU.
```

Trong §10 mục 7, đổi `budget_now=2.10` thành `budget_now=1.00` và `2.10` thành `1.00`,
kèm ghi chú: `(2.10 GiB vẫn hạ cấp thành công được — 2B Q4_K_M offload 24/28 layer = 2.09 GiB)`.

- [ ] **Step 4: Bổ sung §8 — hai chỗ spec bỏ sót**

Thêm vào cuối danh sách "Xóa" của §8:

```markdown
Ngoài ra, Tab 4 "📦 Model Library" (`app.py:975-999`) đọc `minfo["vram"]` và
`minfo["min_vram_4gb"]` — phải viết lại theo schema mới, nếu không app `KeyError`
ngay khi khởi động.
```

Thêm vào §9, trước §9.1:

```markdown
### 9.0 `captioner/base.py`

Abstract `load_model(self, vram_profile: str, **kwargs)` đổi thành
`load_model(self, **kwargs)`. State `_vram_profile` và property `vram_profile`
đổi thành `_runtime_desc` / `runtime_desc` (chuỗi mô tả thông số thật đang chạy).
```

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-09-20-vram-aware-model-selection-design.md
git commit -m "docs: sync spec with offload floor, ctx ladder scope, and missed call sites"
```

---

## Self-Review

**1. Spec coverage** — từng mục spec ánh xạ sang task:

| Spec | Task |
|---|---|
| §1 Vấn đề (1.1–1.4) | 1 (1.1), 2 (1.2), 3–4 (1.3), 6 (1.4) |
| §3.1 Đơn vị GiB | 2 (`test_catalog_dung_don_vi_gib`) |
| §4 `hardware.py` | 1 |
| §5 Catalog verify | 2 |
| §5.4 Model bị loại | 2 (`test_catalog_khong_co_file_split` + docstring) |
| §6.1–6.4 Ước lượng | 3 |
| §6.5–6.6 Xếp hạng | 4 |
| §7 Preflight + hạ cấp | 5 |
| §8 `app.py` | 7 |
| §9 `captioner/` | 6 |
| §10 Kiểm thử (8 assert) | 1–5 (32 assert, phủ cả 8) |
| §11 Ngoài phạm vi | tôn trọng — không có task nào thêm smoke test hay HF OOM fallback |
| §12 Tóm tắt file | bảng File Structure |

Không có mục spec nào thiếu task. Ngược lại, plan thêm hai chỗ spec bỏ sót (Tab Model Library, `base.py`) và một task đồng bộ ngược lại spec (Task 9).

**2. Placeholder scan** — đã quét, không còn "TBD"/"TODO"/"tương tự Task N"/"thêm xử lý lỗi phù hợp". Mọi step có code đều kèm code chạy được, không có step nào chỉ mô tả việc cần làm.

**3. Type consistency** — đã đối chiếu:
- `PlanOption.mmproj_quant` là `Optional[str]` (None với HF) — `_resolve_gguf_local_assets` chỉ gọi khi `backend == "gguf"` nên không bao giờ `None` ✓
- `model_files[q]` là tuple `(filename, gib)` ở Task 2 → `estimate_gguf` lấy `[1]` và `_resolve_gguf_local_assets` lấy `[0]` ✓
- `hardware.budget(device_kind)` nhận `"cpu"|"cuda"|"auto"` → `app._device_kind()` trả đúng ba giá trị đó ✓
- `preflight(option, budget_now, catalog, auto_downgrade)` — thứ tự tham số khớp giữa Task 5 và lời gọi ở Task 7 ✓
- `_expected_download_target(backend_kind, ...)` đổi sang nhận `"hf"|"gguf"` — mọi lời gọi ở Task 7 đã dùng dạng mới ✓
- `BaseCaptioner._runtime_desc` đặt ở Task 6 Step 1, được cả `HFCaptioner` (Step 2) và `GGUFCaptioner` (Step 3) gán ✓

---

## Execution Handoff

Plan hoàn tất, lưu ở `docs/superpowers/plans/2026-09-20-vram-aware-model-selection.md`.
