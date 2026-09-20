# Spec: Chọn model theo VRAM thật (bỏ hard-code 4GB)

- **Ngày**: 2026-09-20
- **Nhánh**: `2026-05-28_review-program-and-fix-gpu-errors`
- **Trạng thái**: đã triển khai trên nhánh `Ver_0_2`; mục này đã đồng bộ với code thực tế

---

## 1. Vấn đề

Chương trình được viết quanh giả định máy 4GB VRAM. Trên máy 10–12GB nó vừa không đề xuất
được model tốt hơn, vừa đề xuất sai model sẽ OOM. Có bốn lỗi độc lập:

**1.1 — App chưa bao giờ đọc VRAM thật.**
`app.py:119` `_available_vram_for_profile()` map tên profile sang một con số **giả định**
(`{ultralow: 4, low: 8, normal: 16, high: 24}`). `utils/system_info.py:get_gpu_info()` có
detect GPU, nhưng dùng `torch.cuda.memory_reserved()` — đó là bộ nhớ torch allocator đang
giữ, bằng 0 khi chưa load gì, nên `vram_free` báo gần như toàn bộ card kể cả khi ComfyUI
đang chiếm VRAM.

**1.2 — Số VRAM trong catalog bị bịa, sai nặng nhất ở nhánh FP8.**
Đã đối chiếu với HuggingFace API ngày 2026-09-20:

| Model | `models_catalog.py` ghi | Thực tế |
|---|---|---|
| Qwen3-VL-8B-Instruct-FP8 | 7.5 GB | **9.86 GiB** |
| Qwen3-VL-4B-Instruct-FP8 | 2.5 GB | **5.61 GiB** |
| Qwen3-VL-2B-Instruct-FP8 | 2.5 GB | **3.23 GiB** |
| Qwen3-VL-8B-Instruct (bf16) | 12.0 GB | **16.33 GiB** |
| Qwen2.5-VL-7B-Instruct (bf16) | 15.0 GB | **15.44 GiB** |

Hệ quả trực tiếp: trên máy 12GB, app mời người dùng load 8B-FP8 vì tưởng nó 7.5GB → OOM.
Ngoài ra ước lượng GGUF bỏ qua hoàn toàn mmproj (0.42–1.08 GiB) và KV cache (tới 1.13 GiB
ở ctx 8192).

**1.3 — Ước lượng GGUF bóc số từ chuỗi hiển thị.**
`app.py:159` `_extract_vram_from_label()` chạy regex `~([0-9.]+)\s*GB` trên nhãn tiếng Anh
do người viết tay (`"Q4_K_M (recommended, ~2.5GB)"`). Sửa chữ trong nhãn là đổi logic tính
VRAM. Trả `999.0` khi không khớp.

**1.4 — Backend HuggingFace đang chết.**
`captioner/hf_captioner.py:145` raise cứng nếu `"Qwen3-VL" in model_id`, và `app.py:170`
`_hf_model_supported_by_loader()` chỉ cho qua `Qwen2.5-VL` → 14/16 entry HF trong catalog
không load được. Tệ hơn, loader dùng `Qwen2VLForConditionalGeneration` cho cả Qwen2.5-VL
(sai class, model thật là `Qwen2_5_VLForConditionalGeneration`).

Chặn này đã lỗi thời. Môi trường hiện tại (`D:\001_Personal_Proj\Comfy\.venv`) có
transformers **5.17.0**, torch **2.14.0+cu130**, bitsandbytes **0.50.2**,
llama-cpp-python **0.3.39-preview**; `AutoModelForImageTextToText`,
`Qwen3VLForConditionalGeneration`, `Qwen2_5_VLForConditionalGeneration` và
`Qwen3VLChatHandler` đều có sẵn.

**Máy tham chiếu**: RTX 5070 Ti Laptop, 11.91 GiB tổng / 10.78 GiB trống, compute 12.0.

---

## 2. Quyết định đã chốt

| # | Quyết định |
|---|---|
| D1 | Sửa backend HF để chạy cả Qwen3-VL lẫn Qwen2.5-VL |
| D2 | **Bỏ khái niệm tier/profile.** Lọc và xếp hạng liên tục theo VRAM thật |
| D3 | Verify catalog một lần qua HF API, ghi cứng số đã verify kèm ngày |
| D4 | An toàn = preflight với VRAM **trống** thật + tự hạ cấp. **Không** smoke test, **không** fallback OOM→CPU cho HF |
| D5 | Bỏ radio "VRAM Profile"; mọi thông số tự suy từ ngân sách + accordion Advanced |
| D6 | Headroom cân bằng: `budget = free × 0.90 − 0.8` |
| D7 | Chỉ lấy model từ org `Qwen/` trên HuggingFace |

---

## 3. Kiến trúc

Ba đơn vị mới, ranh giới rõ, test được độc lập:

```
utils/hardware.py     dữ liệu phần cứng thật  → ngân sách VRAM/RAM (GiB)
models_catalog.py     DỮ LIỆU THUẦN đã verify (không còn logic)
utils/vram_plan.py    ngân sách + catalog     → danh sách cấu hình xếp hạng
```

Luồng: `hardware.budget()` → `vram_plan.plan_options()` → dropdown UI → chọn →
`vram_plan.preflight()` (đọc lại VRAM) → captioner nhận **thông số cụ thể**, không nhận
tên profile.

Nguyên tắc: `models_catalog.py` chỉ chứa dữ liệu, `vram_plan.py` chỉ chứa phép tính thuần
(không I/O, không torch), `hardware.py` là chỗ duy nhất chạm phần cứng. Nhờ đó toàn bộ
logic chọn model test được mà không cần GPU hay tải model.

### 3.1 Đơn vị đo

**Toàn bộ dung lượng trong hệ thống dùng GiB (`bytes / 2**30`).** HF API trả bytes;
chuyển đổi xảy ra **đúng một lần** lúc verify, và số GiB được ghi cứng vào catalog.
`torch.cuda.mem_get_info()` cũng trả bytes. Không có chỗ nào trộn GB thập phân với GiB.

---

## 4. `utils/hardware.py` (mới)

```python
def get_devices() -> list[dict]
```
Mỗi GPU trả: `index`, `name`, `vram_total`, `vram_free`, `compute` (tuple `(major, minor)`).
VRAM trống lấy từ **`torch.cuda.mem_get_info(idx)`** — số của driver, phản ánh cả tiến
trình khác (ComfyUI, game). Không thêm dependency. Nếu torch/CUDA không có, trả `[]`.

```python
HEADROOM_RATIO = 0.10   # hiệu chỉnh: KV cache phình khi ảnh độ phân giải cao
HEADROOM_FIXED = 0.8    # GiB, chừa cho desktop compositor / driver

def vram_budget(device_index: int = 0) -> float
def ram_budget() -> float
def budget(device_kind: str, device_index: int = 0) -> float
```
- GPU: `max(0.0, free × (1 − HEADROOM_RATIO) − HEADROOM_FIXED)`
- CPU: `psutil.virtual_memory().available / 2**30 × 0.6` (không có psutil → 4.0 GiB)
- Máy tham chiếu: `10.78 × 0.90 − 0.8 = 8.90 GiB`

Hai hằng số headroom là **núm hiệu chỉnh**, đặt ở module level kèm comment. Số lý thuyết
không bao giờ khớp thực tế đo được.

```python
def supports_fp8(device_index: int = 0) -> bool
```
`True` khi compute capability ≥ (8, 9). Máy tham chiếu là 12.0 → `True`; card 4GB đời
Pascal/Turing → `False` và mọi entry FP8 bị loại khỏi danh sách.

`utils/system_info.py` giữ nguyên vai trò render HTML, nhưng `get_gpu_info()` chuyển sang
gọi `hardware.get_devices()` thay vì tự tính bằng `memory_reserved()`.

---

## 5. `models_catalog.py` — viết lại thành dữ liệu đã verify

Xóa: `min_vram_4gb`, dict `vram`, `VRAM_PROFILES`, `get_hf_model_names()`,
`get_gguf_model_names()`, `get_vram_profile_names()`, `get_model_info_html()`.

### 5.1 Metadata verify

```python
CATALOG_VERIFIED = {
    "checked": "2026-09-20",
    "source": "https://huggingface.co/api/models/{repo_id}?blobs=true",
    "org": "Qwen",
}
```

### 5.2 Schema HF

```python
"Qwen3-VL-8B-Instruct": {
    "repo_id": "Qwen/Qwen3-VL-8B-Instruct",
    "arch": "Qwen3VLForConditionalGeneration",
    "series": "Qwen3-VL",
    "size": "8B",
    "weights_gib": 16.33,          # verified 2026-09-20
    "native_quant": None,          # None | "fp8" | "awq"
    "quants": ["bf16", "8bit", "4bit"],
    "kv": {"layers": 36, "kv_heads": 8, "head_dim": 128},
    "quality": 8.0,                # ~số tham số (tỷ), dùng để xếp hạng
    "hf_url": "https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct",
}
```

`native_quant` khác `None` ⇒ `quants == [native_quant]`; **không chồng bitsandbytes lên
checkpoint đã quantize sẵn**. Entry `fp8` chỉ hiện khi `hardware.supports_fp8()`.

Trọng số đã verify (GiB):

| Model | `weights_gib` | `native_quant` | `kv` (L/kvh/hd) |
|---|---|---|---|
| Qwen3-VL-2B-Instruct | 3.97 | – | 28 / 8 / 128 |
| Qwen3-VL-2B-Thinking | 3.97 | – | 28 / 8 / 128 |
| Qwen3-VL-2B-Instruct-FP8 | 3.23 | fp8 | 28 / 8 / 128 |
| Qwen3-VL-4B-Instruct | 8.27 | – | 36 / 8 / 128 |
| Qwen3-VL-4B-Thinking | 8.27 | – | 36 / 8 / 128 |
| Qwen3-VL-4B-Instruct-FP8 | 5.61 | fp8 | 36 / 8 / 128 |
| Qwen3-VL-8B-Instruct | 16.33 | – | 36 / 8 / 128 |
| Qwen3-VL-8B-Thinking | 16.33 | – | 36 / 8 / 128 |
| Qwen3-VL-8B-Instruct-FP8 | 9.86 | fp8 | 36 / 8 / 128 |
| Qwen2.5-VL-3B-Instruct | 6.99 | – | 36 / 2 / 128 |
| Qwen2.5-VL-3B-Instruct-AWQ | 3.17 | awq | 36 / 2 / 128 |
| Qwen2.5-VL-7B-Instruct | 15.44 | – | 28 / 4 / 128 |
| Qwen2.5-VL-7B-Instruct-AWQ | 6.44 | awq | 28 / 4 / 128 |

Bổ sung mới so với catalog cũ: hai entry AWQ (lựa chọn chất lượng tốt cho phân khúc 4–8GB
mà catalog cũ không có). Các entry `*-Thinking-FP8` giữ nguyên nếu đã có, cùng
`weights_gib` với bản Instruct-FP8 tương ứng.

### 5.3 Schema GGUF

```python
"Qwen3-VL-8B-Instruct-GGUF": {
    "repo_id": "Qwen/Qwen3-VL-8B-Instruct-GGUF",
    "series": "Qwen3-VL",
    "size": "8B",
    "kv": {"layers": 36, "kv_heads": 8, "head_dim": 128},
    "quality": 8.0,
    "model_files": {                       # quant → (filename, gib)
        "Q4_K_M": ("Qwen3VL-8B-Instruct-Q4_K_M.gguf", 4.68),
        "Q8_0":   ("Qwen3VL-8B-Instruct-Q8_0.gguf",   8.11),
        "F16":    ("Qwen3VL-8B-Instruct-F16.gguf",   15.26),
    },
    "mmproj_files": {                      # quant → (filename, gib)
        "Q8_0": ("mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf", 0.70),
        "F16":  ("mmproj-Qwen3VL-8B-Instruct-F16.gguf",  1.08),
    },
    "hf_url": "https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF",
}
```

Hai thay đổi cấu trúc so với hiện tại:
- `model_files` chuyển từ `{"nhãn tiếng Anh": "filename"}` sang `{quant: (filename, gib)}`.
  Nhãn hiển thị được **sinh ra** từ dữ liệu này, không phải nguồn của dữ liệu.
- `mmproj_files` thay `mmproj_file` (một chuỗi F16 cứng). Bản Q8_0 tiết kiệm 0.34–0.38 GiB
  mà catalog cũ không dùng tới.

Dung lượng đã verify (GiB):

| Repo | Q4_K_M | Q8_0 | F16 | mmproj Q8_0 | mmproj F16 |
|---|---|---|---|---|---|
| Qwen3-VL-2B-Instruct-GGUF | 1.03 | 1.70 | 3.21 | 0.42 | 0.76 |
| Qwen3-VL-2B-Thinking-GGUF | 1.03 | 1.70 | 3.21 | 0.42 | 0.76 |
| Qwen3-VL-4B-Instruct-GGUF | 2.33 | 3.99 | 7.50 | 0.42 | 0.78 |
| Qwen3-VL-4B-Thinking-GGUF | 2.33 | 3.99 | 7.50 | 0.42 | 0.78 |
| Qwen3-VL-8B-Instruct-GGUF | 4.68 | 8.11 | 15.26 | 0.70 | 1.08 |
| Qwen3-VL-8B-Thinking-GGUF | 4.68 | 8.11 | 15.26 | 0.70 | 1.08 |

Hai repo 2B là **bổ sung mới** — catalog cũ thiếu hoàn toàn, dù đây đúng là lựa chọn cho
phân khúc 4GB.

### 5.4 Model bị loại và lý do

| Model | Lý do loại |
|---|---|
| Qwen3-VL-30B-A3B-*-GGUF | File split 2 phần (`-split-00001-of-00002.gguf`). `hf_hub_download()` một filename không tải được; Q4_K_M đã 17.3 GiB |
| Qwen3-VL-32B-*-GGUF | Như trên, Q4_K_M 18.4 GiB |
| Qwen3-VL-32B-Instruct / -Thinking (HF) | 62.1 GiB trọng số; 4-bit vẫn ~19.9 GiB, vượt mọi cấu hình một GPU trong tầm ngắm |
| Qwen3-VL-235B-A22B-* | Ngoài phạm vi hoàn toàn |
| Qwen2-VL, Qwen-VL (đời cũ) | Bị Qwen2.5-VL thay thế |

Nếu sau này cần 30B/32B: phải thêm hỗ trợ tải & load file GGUF split. Ghi lại ở đây để
quyết định này không bị đọc nhầm thành "không tồn tại".

---

## 6. `utils/vram_plan.py` (mới) — tính và xếp hạng

Hàm thuần: nhận số, trả số/dataclass. Không torch, không network, không đọc đĩa.

### 6.1 KV cache

```python
def kv_cache_gib(kv: dict, n_ctx: int, bytes_per_elem: int = 2) -> float:
    return (2 * kv["layers"] * kv["kv_heads"] * kv["head_dim"]
            * n_ctx * bytes_per_elem) / 2**30
```

Số thật (fp16): Qwen3-VL-8B ở ctx 8192 tốn **1.125 GiB**, ở 4096 tốn 0.563 GiB. Đây chính
là phần mà ước lượng hiện tại bỏ sót hoàn toàn.

### 6.2 Núm hiệu chỉnh

```python
# Hệ số VRAM thực tế / trọng số gốc bf16. Số đo được, không phải lý thuyết —
# chỉnh ở đây khi thấy lệch so với nvidia-smi.
QUANT_FACTOR = {"bf16": 1.00, "8bit": 0.55, "4bit": 0.32, "fp8": 1.00, "awq": 1.00}

OVERHEAD_GIB = 0.60         # CUDA context + buffer của runtime
ACT_PER_MPIXEL_GIB = 0.35   # activation của vision tower, theo max_pixels
```

### 6.3 Ước lượng

```python
def estimate_hf(entry, quant, n_ctx, max_pixels) -> float:
    return (entry["weights_gib"] * QUANT_FACTOR[quant]
            + kv_cache_gib(entry["kv"], n_ctx)
            + ACT_PER_MPIXEL_GIB * max_pixels / 1e6
            + OVERHEAD_GIB)

def estimate_gguf(entry, quant, mmproj_quant, n_ctx, gpu_layers) -> float:
    model_gib = entry["model_files"][quant][1]
    total = entry["kv"]["layers"]
    frac = 1.0 if gpu_layers < 0 else min(gpu_layers, total) / total
    return (model_gib * frac
            + entry["mmproj_files"][mmproj_quant][1]     # mmproj luôn trên GPU
            + kv_cache_gib(entry["kv"], n_ctx) * frac
            + OVERHEAD_GIB)
```

### 6.4 Thông số suy từ ngân sách

```python
def derive_runtime(budget: float) -> dict:   # → {"n_ctx", "max_pixels", "mmproj_quant"}
```

| Ngân sách | `n_ctx` | `max_pixels` | `mmproj_quant` |
|---|---|---|---|
| < 3.0 GiB | 2048 | 512 × 28 × 28 | Q8_0 |
| < 6.0 GiB | 4096 | 768 × 28 × 28 | Q8_0 |
| < 10.0 GiB | 4096 | 1280 × 28 × 28 | Q8_0 |
| < 16.0 GiB | 8192 | 1280 × 28 × 28 | F16 |
| ≥ 16.0 GiB | 8192 | 2560 × 28 × 28 | F16 |

Đây là thay thế trực tiếp cho `pixel_config` / `gguf_ctx` / `gguf_layers` của
`VRAM_PROFILES` cũ, khác ở chỗ đầu vào là số GiB thật chứ không phải tên profile.

### 6.5 Cấu hình và xếp hạng

```python
@dataclass
class PlanOption:
    backend: str            # "hf" | "gguf"
    model_name: str
    quant: str
    n_ctx: int
    gpu_layers: int         # -1 = toàn bộ; chỉ GGUF
    max_pixels: int
    mmproj_quant: str | None
    est_gib: float
    fit: str                # "comfortable" | "tight" | "over"
    quality: float
    label: str              # nhãn hiển thị, SINH RA từ các trường trên

def plan_options(budget: float, backend: str, catalog,
                 supports_fp8: bool = True) -> list[PlanOption]
```

- Sinh một `PlanOption` cho mỗi cặp (model × quant hợp lệ), với `n_ctx`/`max_pixels`/
  `mmproj_quant` từ `derive_runtime(budget)`, `gpu_layers = -1`.
- Nếu một cấu hình GGUF vượt ngân sách với `gpu_layers = -1`, thử **offload một phần**:
  với `per_layer_gib = (model_gib + kv_cache_gib) / entry["kv"]["layers"]`,
  `gpu_layers = floor((budget − mmproj_gib − OVERHEAD_GIB) / per_layer_gib)`.
  Chỉ giữ nếu `gpu_layers ≥ MIN_OFFLOAD_FRAC × total` với `MIN_OFFLOAD_FRAC = 0.75`
  (dùng tỉ lệ chứ không phải số tuyệt đối: 8 trên 28 layer khác hẳn 8 trên 36).
  Không đạt sàn thì vẫn trả cấu hình full-offload đánh `over` để người dùng
  thấy nó tồn tại (🔴), chứ không biến mất khỏi danh sách.
- `n_ctx` do `derive_runtime(budget)` quyết định, KHÔNG giảm tiếp trong
  `plan_options` (sẽ sinh hai lựa chọn cùng model khác ctx trong cùng dropdown,
  gây rối). Bậc thang ctx chỉ dùng trong `preflight` — xem §7 bước 3.
- Loại entry `fp8` khi `supports_fp8` là `False`.
- Phân loại `fit`: `est_gib ≤ 0.80 × budget` → `comfortable`; `≤ budget` → `tight`;
  còn lại → `over`.
- Sắp xếp: `comfortable` trước `tight` trước `over`; trong mỗi nhóm, `quality` giảm dần,
  rồi `est_gib` tăng dần.

Nhãn sinh ra: `🟢 Qwen3-VL-8B-Instruct-GGUF · Q4_K_M · 6.54 GiB`

### 6.6 Kết quả kỳ vọng trên máy tham chiếu

Ngân sách 8.90 GiB, `derive_runtime` → `n_ctx=4096`, `max_pixels=1280·28·28`,
`mmproj_quant=Q8_0`:

| Cấu hình | Ước lượng | `fit` |
|---|---|---|
| GGUF 8B-Instruct Q4_K_M | 6.54 GiB | 🟢 comfortable |
| HF 8B-Instruct 4-bit | 6.74 GiB | 🟢 comfortable |
| GGUF 4B-Instruct Q8_0 | 5.57 GiB | 🟢 comfortable |
| HF 8B-Instruct-FP8 | 11.37 GiB | 🔴 over |
| GGUF 8B-Instruct Q8_0 (full offload) | 9.97 GiB | 🔴 over → thử offload một phần |

App hiện tại sẽ chọn **8B-FP8** (tưởng 7.5GB) và OOM. Sau thay đổi, lựa chọn mặc định là
GGUF 8B Q4_K_M hoặc HF 8B 4-bit.

Và với máy 4GB (`free ≈ 3.7` → `budget = 2.53 GiB`, `n_ctx=2048`,
`max_pixels=512·28·28`): GGUF 2B Q4_K_M ≈ 2.27 GiB 🟢, GGUF 4B Q4_K_M ≈ 3.63 GiB 🔴 →
phân khúc 4GB vẫn được phục vụ đúng, chỉ là không còn hard-code.

---

## 7. An toàn khi load

```python
def preflight(option: PlanOption, budget_now: float,
              catalog) -> tuple[PlanOption, str | None]
```

Chạy **ngay trước khi load**, không phải lúc dựng UI — VRAM trống thay đổi khi người dùng
mở ComfyUI hoặc game giữa chừng.

1. Gọi lại `hardware.budget()` để có `budget_now`.
2. Tính lại `est_gib` của cấu hình đã chọn.
3. Nếu `est_gib ≤ budget_now`: trả `(option, None)`.
4. Nếu vượt và **checkbox "Tự hạ cấp" bật**: hạ theo đúng thứ tự dưới đây, dừng ở bước đầu
   tiên vừa ngân sách, trả `(option_mới, thông_báo)`:
   1. quant thấp hơn của **cùng model** (F16 → Q8_0 → Q4_K_M; bf16 → 8bit → 4bit)
   2. giảm `gpu_layers` (chỉ GGUF, sàn 8 layer)
   3. giảm `n_ctx` một bậc (sàn 2048)
   4. model nhỏ hơn cùng series, bắt đầu lại từ bước 1
5. Nếu vượt và **checkbox tắt**: raise với con số cụ thể.
6. Nếu hạ hết cách mà vẫn không vừa: raise, bất kể checkbox.

Thông báo phải nêu cả ba con số — cần bao nhiêu, còn bao nhiêu, đã đổi thành gì:

> ⚠️ Qwen3-VL-8B-Instruct-GGUF Q8_0 cần 9.97 GiB nhưng chỉ còn 8.90 GiB
> → đã chuyển sang Q4_K_M (6.54 GiB)

Lỗi khi không hạ được:

> ❌ Qwen3-VL-8B-Instruct-GGUF Q4_K_M cần 6.54 GiB, chỉ còn 1.00 GiB.
> Đã thử mọi mức quant và ctx thấp hơn. Hãy đóng bớt ứng dụng đang dùng GPU,
> hoặc chuyển Device sang CPU.

**Ngoài phạm vi** (theo quyết định D4): smoke test sau load; fallback OOM→CPU cho backend
HF. Fallback CUDA→CPU sẵn có của GGUF (`gguf_captioner.py:224`) **giữ nguyên**, không đụng.

---

## 8. Thay đổi trong `app.py`

**Xóa** — toàn bộ nhóm hàm xoay quanh profile:
`_vram_profile_bucket`, `_available_vram_for_profile`, `_hf_quant_key_for_profile`,
`_estimate_hf_required_vram`, `_pick_gguf_variant`, `_extract_vram_from_label`,
`_estimate_gguf_required_vram`, `_hf_model_supported_by_loader`, `_is_model_compatible`,
`VRAM_PROFILE_NAMES`, `on_vram_change`, `vram_radio`, `vram_desc`, và bảng
"VRAM / RAM Reference" tĩnh trong markdown cột phải.

Ngoài ra, Tab 4 "📦 Model Library" (`app.py:975-999`) đọc `minfo["vram"]` và
`minfo["min_vram_4gb"]` — phải viết lại theo schema mới, nếu không app `KeyError`
ngay khi khởi động.

**Giữ, sửa nhẹ** — nhóm phát hiện file local vẫn đúng và vẫn cần:
`_hf_storage_dir`, `_hf_is_available`, `_gguf_file_index`, `_resolve_gguf_local_assets`
(nhận `PlanOption` thay vì `vram_profile`), `_download_hf_model`, `_download_gguf_assets`.

**UI mới** ở tab Setup, thay chỗ radio profile:

```
┌─ Phần cứng ────────────────────────────────────────┐
│ GPU 0 — NVIDIA GeForce RTX 5070 Ti Laptop GPU      │
│ VRAM 10.78 / 11.91 GiB trống  ·  compute 12.0      │
│ Ngân sách model: 8.90 GiB   [🔄 Làm mới]           │
└────────────────────────────────────────────────────┘

Select Model  ▼  🟢 Qwen3-VL-8B-Instruct-GGUF · Q4_K_M · 6.54 GiB

☑ Tự hạ cấp khi thiếu VRAM

▸ Advanced
    Ngân sách VRAM (GiB)  [ 8.90 ]   ← ghi đè giá trị tự tính
    n_ctx                 [ 4096 ]
    GPU layers            [  -1  ]
    max_pixels (×28×28)   [ 1280 ]
```

Chú thích trạng thái: 🟢 thoải mái · 🟡 sát ngưỡng · 🔴 vượt ngân sách (vẫn chọn được,
preflight sẽ hạ cấp hoặc chặn) · ⬇️ chưa tải về máy.

Nút **Làm mới** gọi lại `hardware.budget()` và dựng lại dropdown — dùng khi vừa đóng
ComfyUI. Giá trị trong Advanced để trống nghĩa là "tự suy"; điền số là ghi đè.

`load_model()` đổi chữ ký từ
`(backend, vram_profile, selected_model, flash_attn, device_choice)` sang
`(backend, selected_option_label, auto_downgrade, device_choice, advanced_overrides)`.

---

## 9. Thay đổi trong `captioner/`

### 9.0 `captioner/base.py`

Abstract `load_model(self, vram_profile: str, **kwargs)` đổi thành
`load_model(self, **kwargs)`. State `_vram_profile` và property `vram_profile`
đổi thành `_runtime_desc` / `runtime_desc` (chuỗi mô tả thông số thật đang chạy).

### 9.1 `hf_captioner.py`

- Thay `Qwen2VLForConditionalGeneration` bằng **`AutoModelForImageTextToText`** — chạy cả
  `Qwen3VLForConditionalGeneration` lẫn `Qwen2_5_VLForConditionalGeneration` trên
  transformers 5.17.
- **Xóa** khối `raise` chặn Qwen3-VL (`:145`).
- Chữ ký đổi: `load_model(model_id, quant, n_ctx, max_pixels, device, use_flash_attn)`;
  không còn tham số `vram_profile`, không còn import `VRAM_PROFILES`.
- `native_quant` khác `None` ⇒ **không truyền `quantization_config`**. Nạp checkpoint
  FP8/AWQ như hiện trạng của nó.
- Dùng `dtype=` thay `torch_dtype=` (đã deprecated trong transformers 5.x).
- Giữ nguyên: nhánh CPU ép `quant="none"` + float32, `_configure_torch_runtime()`,
  `unload_model()`.

### 9.2 `gguf_captioner.py`

- Chữ ký đổi: `load_model(model_path, mmproj_path, n_ctx, n_gpu_layers, n_batch, device)`;
  không còn `vram_profile`, không còn đọc `VRAM_PROFILES`/`gguf_defaults` từ catalog —
  mọi thông số do `vram_plan` quyết định và truyền vào.
- `_build_chat_handler()`, fallback CUDA→CPU, `_filter_kwargs_for_callable()`,
  `_cpu_cuda_hidden_env()`: **giữ nguyên**, đang hoạt động đúng.
- `gguf_defaults` trong catalog bị xóa (đã bị `derive_runtime()` thay thế).

---

## 10. Kiểm thử

`test_vram_plan.py` (mới) — hàm thuần, chạy không cần GPU, không tải model, không mạng.
Dùng `assert` + `if __name__ == "__main__"`, không framework:

1. **Ngân sách**: `free=10.78` → `budget == 8.90` (±0.01); `free=0.5` → `budget == 0.0`.
2. **KV cache**: `kv_cache_gib({"layers":36,"kv_heads":8,"head_dim":128}, 8192) == 1.125`.
3. **Máy 12GB**: `plan_options(8.90, "gguf")[0]` là 8B Q4_K_M, `fit == "comfortable"`.
4. **Máy 4GB**: `plan_options(2.53, "gguf")[0]` là 2B Q4_K_M; không cấu hình 8B nào được
   xếp `comfortable`.
5. **FP8 bị gate**: `plan_options(8.90, "hf", supports_fp8=False)` không chứa entry nào có
   `quant == "fp8"`.
6. **Hạ cấp**: `preflight(8B Q8_0, budget_now=8.90)` trả về cấu hình Q4_K_M kèm thông báo
   khác `None`.
7. **Hết đường**: `preflight(8B Q4_K_M, budget_now=1.00)` raise, thông báo chứa cả
   `6.54` và `1.00`. (2.10 GiB vẫn hạ cấp thành công được — 2B Q4_K_M offload
   24/28 layer = 2.09 GiB, nên không raise.)
8. **Không hồi quy đơn vị**: mọi `est_gib` trong catalog nằm trong `(0, 70)` — bắt lỗi
   trộn bytes/GB/GiB.

Kiểm tra thủ công sau khi code xong, trên máy tham chiếu:
- Load GGUF 8B Q4_K_M → so `nvidia-smi` với `est_gib`; lệch > 15% thì chỉnh
  `OVERHEAD_GIB` / `QUANT_FACTOR`.
- Load HF Qwen3-VL-8B 4-bit → xác nhận `AutoModelForImageTextToText` chạy được (đây là
  đường code chưa từng chạy thành công).
- Mở ComfyUI chiếm VRAM → bấm Làm mới → xác nhận ngân sách tụt và dropdown đổi thứ tự.

---

## 11. Ngoài phạm vi

Không đụng tới: `captioner/prompts.py`, vòng lặp batch caption, CSS/theme,
`utils/file_utils.py`, `utils/image_utils.py`, các file `.bat`.

Không làm trong đợt này: smoke test sau load, fallback OOM→CPU cho HF, hỗ trợ GGUF file
split, repo ngoài org `Qwen/`, nút "Verify online" cập nhật catalog lúc chạy, multi-GPU
(chỉ đọc `device_index=0`).

---

## 12. Tóm tắt file

| File | Thay đổi |
|---|---|
| `utils/hardware.py` | **mới** — đọc VRAM/RAM thật, tính ngân sách, gate FP8 |
| `utils/vram_plan.py` | **mới** — ước lượng, `derive_runtime`, `plan_options`, `preflight` |
| `models_catalog.py` | viết lại: dữ liệu đã verify, bỏ toàn bộ logic và `VRAM_PROFILES` |
| `app.py` | bỏ 11 hàm/biến theo profile, thẻ phần cứng + Advanced, preflight khi load |
| `captioner/hf_captioner.py` | `AutoModelForImageTextToText`, bỏ chặn Qwen3-VL, nhận thông số cụ thể |
| `captioner/gguf_captioner.py` | nhận thông số cụ thể thay `vram_profile` |
| `utils/system_info.py` | `get_gpu_info()` gọi `hardware.get_devices()` |
| `test_vram_plan.py` | **mới** — 8 assert, không cần GPU |
| `test_caption.py` | cập nhật theo chữ ký `load_model()` mới |
| `README.md` | bỏ mô tả "4GB VRAM", tả cơ chế ngân sách động |
