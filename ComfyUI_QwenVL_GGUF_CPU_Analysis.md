# ComfyUI-QwenVL: Phần hỗ trợ chạy GGUF trên CPU

Repo: `https://github.com/1038lab/ComfyUI-QwenVL`

Nội dung dưới đây tổng hợp các đoạn code quan trọng trong repo liên quan đến việc load và chạy model **GGUF** bằng **CPU**, đặc biệt thông qua backend `llama-cpp-python`.

---

## 1. File chính: `AILab_QwenVL_GGUF.py`

Đây là file quan trọng nhất cho node QwenVL GGUF, bao gồm các node:

```python
AILab_QwenVL_GGUF
AILab_QwenVL_GGUF_Advanced
```

File này dùng `llama_cpp.Llama` để load model GGUF, nghĩa là model GGUF không chạy qua Transformers mà chạy thông qua backend `llama.cpp`.

---

## 2. Hàm chọn device: `_pick_device(device_choice)`

Đoạn code quan trọng:

```python
def _pick_device(device_choice: str) -> str:
    if device_choice == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if device_choice.startswith("cuda") and torch.cuda.is_available():
        return "cuda"
    if device_choice == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
```

### Ý nghĩa

Flow xử lý device:

```text
device = "auto"
→ ưu tiên CUDA nếu có
→ nếu không có CUDA thì thử MPS
→ nếu không có MPS thì fallback về CPU
```

Nếu người dùng chọn trực tiếp:

```text
device = "cpu"
```

thì hàm sẽ trả về:

```python
"cpu"
```

Nếu người dùng chọn CUDA nhưng máy không có CUDA, hàm cũng fallback về:

```python
"cpu"
```

---

## 3. Đoạn ép GGUF chạy CPU bằng `n_gpu_layers = 0`

Trong hàm `_load_model(...)` của `AILab_QwenVL_GGUF.py`, logic quan trọng là:

```python
device_kind = _pick_device(device)

...

if device_kind == "cuda":
    n_gpu_layers = int(gpu_layers) if gpu_layers is not None else resolved.gpu_layers
else:
    n_gpu_layers = 0
```

### Ý nghĩa

Với `llama.cpp` hoặc `llama-cpp-python`:

```python
n_gpu_layers = 0
```

có nghĩa là:

```text
Không offload layer nào lên GPU.
Toàn bộ model GGUF chạy trên CPU.
```

Đây là đoạn quan trọng nhất để xác định repo có hỗ trợ chạy model GGUF trên CPU.

---

## 4. Đoạn truyền tham số vào `Llama(...)`

Sau khi xác định `n_gpu_layers`, repo truyền tham số vào `llama_cpp.Llama`:

```python
llm_kwargs = {
    "model_path": str(model_path),
    "n_ctx": n_ctx,
    "n_gpu_layers": n_gpu_layers,
    "n_batch": n_batch_val,
    "swa_full": True,
    "verbose": False,
    "pool_size": pool_size_val,
    "top_k": top_k_val,
}

...

self.llm = Llama(**llm_kwargs_filtered)
```

### Flow chạy CPU

```text
device = "cpu"
→ _pick_device("cpu") returns "cpu"
→ device_kind != "cuda"
→ n_gpu_layers = 0
→ Llama(..., n_gpu_layers=0)
→ GGUF chạy trên CPU
```

---

## 5. Node Advanced có option chọn CPU trong UI

Trong class `AILab_QwenVL_GGUF_Advanced`, phần `INPUT_TYPES()` có đoạn:

```python
num_gpus = torch.cuda.device_count()
gpu_list = [f"cuda:{i}" for i in range(num_gpus)]
device_options = ["auto", "cpu", "mps"] + gpu_list
```

Sau đó input node có:

```python
"device": (device_options, {"default": "auto"})
```

### Ý nghĩa

Trong UI của ComfyUI, node **QwenVL Advanced (GGUF)** có thể chọn trực tiếp:

```text
device = cpu
```

Đây là phần UI cho phép người dùng chủ động ép model GGUF chạy bằng CPU.

---

## 6. File phụ: `AILab_QwenVL_GGUF_PromptEnhancer.py`

File này cũng hỗ trợ GGUF chạy CPU, nhưng dùng cho node **Prompt Enhancer GGUF**, tức là text-only prompt enhancer, không phải vision-language node chính.

---

## 7. Device option trong Prompt Enhancer

Trong `INPUT_TYPES()`:

```python
"device": (["auto", "cuda", "cpu", "mps"], {
    "default": "auto",
    "tooltip": "Select device; auto prefers GPU when available."
})
```

### Ý nghĩa

Node này cũng cho phép chọn:

```text
auto
cuda
cpu
mps
```

Trong đó `cpu` là lựa chọn rõ ràng để chạy GGUF bằng CPU.

---

## 8. Load GGUF CPU trong Prompt Enhancer

Trong hàm `_load_model(self, model_name, device)`:

```python
if device == "auto":
    device_choice = "cuda" if torch.cuda.is_available() else (
        "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
    )
else:
    device_choice = device

auto_gpu_layers = -1 if device_choice == "cuda" else 0

threads = None
if device_choice == "cpu":
    threads = max(os.cpu_count() or 1, 1)

kwargs = {
    "model_path": str(resolved),
    "n_ctx": context_length,
    "n_gpu_layers": auto_gpu_layers,
    "n_threads": None if threads == 0 else threads,
    "n_batch": 1024,
    "verbose": False,
    "chat_format": "qwen",
}

self.llm = Llama(**kwargs)
```

### Ý nghĩa

Khi chọn CPU:

```text
device_choice == "cpu"
→ auto_gpu_layers = 0
→ n_threads = os.cpu_count()
→ Llama(..., n_gpu_layers=0, n_threads=<CPU threads>)
```

So với file `AILab_QwenVL_GGUF.py`, file này còn set thêm số thread CPU:

```python
n_threads = os.cpu_count()
```

Điều này giúp tận dụng nhiều CPU thread hơn khi chạy GGUF.

---

## 9. Bảng tổng hợp

| File | Chức năng | Đoạn quan trọng |
|---|---|---|
| `AILab_QwenVL_GGUF.py` | QwenVL GGUF vision-language node | `_pick_device()`, `_load_model()`, `n_gpu_layers = 0`, `Llama(...)` |
| `AILab_QwenVL_GGUF.py` | UI Advanced node | `device_options = ["auto", "cpu", "mps"] + gpu_list` |
| `AILab_QwenVL_GGUF_PromptEnhancer.py` | GGUF prompt enhancer text-only | `device=["auto","cuda","cpu","mps"]`, `auto_gpu_layers = 0`, `n_threads=os.cpu_count()` |

---

## 10. Kết luận

Repo `ComfyUI-QwenVL` hỗ trợ chạy GGUF trên CPU thông qua `llama-cpp-python`.

Điểm cốt lõi là tham số:

```python
n_gpu_layers = 0
```

Trong `llama.cpp` / `llama-cpp-python`, tham số này có nghĩa là:

```text
Không offload layer nào lên GPU.
Model GGUF chạy hoàn toàn bằng CPU.
```

Do đó, các đoạn code quan trọng nhất cần chú ý là:

```python
if device_kind == "cuda":
    n_gpu_layers = int(gpu_layers) if gpu_layers is not None else resolved.gpu_layers
else:
    n_gpu_layers = 0
```

và:

```python
auto_gpu_layers = -1 if device_choice == "cuda" else 0
```

Nếu muốn ép chạy CPU trong ComfyUI, chọn:

```text
device = cpu
```

trong node GGUF tương ứng.

---

## 11. Ghi chú kỹ thuật

### Với `llama-cpp-python`

- `n_gpu_layers = 0`: chạy CPU hoàn toàn.
- `n_gpu_layers = -1`: thường dùng để offload tối đa lên GPU nếu backend hỗ trợ.
- `n_threads`: số CPU threads dùng cho inference.
- `n_batch`: batch token processing size, ảnh hưởng tốc độ và RAM.
- `n_ctx`: context length, càng lớn càng tốn RAM.

### Với ComfyUI node

Nếu máy không có CUDA/MPS, chọn `auto` cũng sẽ fallback về CPU.

Tuy nhiên, để chắc chắn chạy CPU, nên chọn trực tiếp:

```text
device = cpu
```
