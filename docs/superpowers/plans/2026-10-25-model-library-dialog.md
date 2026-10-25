# Model Library Dialog and HF Safetensors Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Setup tab có dialog "Add or test model" để kiểm tra nhanh rồi thêm model Qwen3-VL (GGUF hoặc safetensors) vào thư viện lưu trong `models_library.json`; app kiểm tra thư viện khi khởi động; backend HuggingFace chạy được Qwen3-VL, kể cả file safetensors single-file của ComfyUI sau khi chuyển đổi.

**Architecture:** `model_registry.py` đọc header và đưa ra kết luận tĩnh cho từng file. `model_library.py` đọc/ghi thư viện và đăng ký entry vào catalog `GGUF_VL_MODELS`/`HF_VL_MODELS` có sẵn, nên dropdown và hàm Load gần như giữ nguyên. `captioner/comfy_convert.py` chuyển single-file fp8 thành thư mục HF bf16. `model_tester.py` nạp model bằng chính captioner của app, caption một ảnh tổng hợp và đo VRAM qua `nvidia-smi`. `app.py` chỉ thêm banner, dialog và các handler mỏng.

**Tech Stack:** Python 3.13, Gradio 6.28, transformers 5.17, torch + torchvision bản CUDA (Blackwell), bitsandbytes 0.50, llama-cpp-python 0.3.39, safetensors 0.8, huggingface_hub 1.32, Pillow.

**Spec:** `docs/superpowers/specs/2026-10-25-model-library-dialog-design.md`

## Global Constraints

- Interpreter: `.venv\Scripts\python.exe` (không dùng `D:\Comfy\.venv`). Mọi lệnh chạy ở thư mục gốc repo `D:\001_Personal_Proj\Captioner_QWENVL`.
- Models root mặc định: `D:\001_Personal_Proj\Comfy\ComfyUI\models\text_encoders` (`model_registry.DEFAULT_LLM_DIR`).
- Test theo kiểu sẵn có của repo: file `test_*.py` ở gốc, hàm `test_*` dùng `assert`, khối `__main__` gọi từng hàm và in `... OK`. Không dùng pytest.
- Đường dẫn trong `models_library.json` lưu **tương đối với models root** (dấu `/`), file ngoài root lưu tuyệt đối.
- Ngân sách VRAM theo profile: UltraLow 4, LowVRAM 8, NormalVRAM 16, HighVRAM 24 GB.
- Bảng Qwen3-VL: hidden 2048 → `Qwen/Qwen3-VL-2B-Instruct`, 28 layer, tied; 2560 → `Qwen/Qwen3-VL-4B-Instruct`, 36, tied; 4096 → `Qwen/Qwen3-VL-8B-Instruct`, 36, không tied; 5120 → `Qwen/Qwen3-VL-32B-Instruct`, 64, không tied.
- Chỉ hỗ trợ định dạng `comfy_quant` = `float8_e4m3fn`. nvfp4, int8 bị từ chối.
- Ảnh test: 512×512, nền xanh dương, tròn đỏ, vuông vàng. Prompt `Describe the shapes and colors in this image in one sentence.`, tối đa 60 token. Đạt khi điểm thị giác ≥ 3/5 trên các từ red, circle, yellow, square, blue.
- Commit message kết thúc bằng dòng `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## File Structure

| File | Trách nhiệm |
|------|-------------|
| `requirements.txt` (sửa) | Ghi lệnh cài torch/torchvision CUDA đã kiểm chứng |
| `captioner/comfy_convert.py` (mới) | Chuyển single-file ComfyUI → thư mục HF |
| `test_comfy_convert.py` (mới) | Test chuyển đổi bằng tensor nhỏ |
| `captioner/hf_captioner.py` (sửa) | Nạp Qwen3-VL qua `AutoModelForImageTextToText` |
| `model_registry.py` (sửa) | Đọc header, `inspect_gguf`, `inspect_safetensors`, `inspect_path`, `list_candidates`; bỏ `refresh_registry`/`apply_to_catalog` |
| `test_model_registry.py` (sửa) | Test kết luận tĩnh |
| `models_catalog.py` (sửa) | Thêm `budget_gb` vào `VRAM_PROFILES` |
| `model_library.py` (mới) | Đọc/ghi thư viện, trạng thái, đăng ký vào catalog, khôi phục |
| `test_model_library.py` (mới) | Test thư viện |
| `model_tester.py` (mới) | Quick test + CLI |
| `test_model_tester.py` (mới) | Test các hàm thuần của quick test |
| `models_library.json` (mới, commit) | Thư viện ban đầu: 4 GGUF có sẵn |
| `app.py` (sửa) | Khởi động theo thư viện, banner, dialog |
| `.claude/launch.json`, `.gitignore`, `README.md` (sửa) | Cấu hình chạy preview, dọn file cũ, tài liệu |

---

### Task 1: Môi trường CUDA cho backend HuggingFace

**Cần người dùng đồng ý trước khi chạy** (tải vài GB, thay torch trong `.venv`).

**Files:**
- Modify: `requirements.txt:4-7`

**Interfaces:**
- Consumes: không có.
- Produces: `.venv` có `torch.cuda.is_available() == True` và `import torchvision.io` chạy được. Các task 3, 7, 8, 10 cần điều này.

- [ ] **Step 1: Ghi lại trạng thái hiện tại để so sánh**

Run:
```bash
.venv/Scripts/python.exe -c "import torch, torchvision; print(torch.__version__, torch.version.cuda, torchvision.__version__)"
```
Expected: `2.14.0 None 0.2.0`

- [ ] **Step 2: Cài torch + torchvision bản CUDA**

Thử lần lượt các index, dừng ở cái đầu tiên cài được cho Python 3.13 Windows:
```bash
.venv/Scripts/python.exe -m pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu132
```
Nếu báo không tìm thấy bản phù hợp, chạy lại với `cu130`, rồi `cu128`.

- [ ] **Step 3: Kiểm tra torch thấy GPU và torchvision đúng**

Run:
```bash
.venv/Scripts/python.exe -c "import torch, torchvision, torchvision.io; print(torch.__version__, torch.version.cuda, torchvision.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0)); from transformers import AutoProcessor, AutoModelForImageTextToText; print('transformers ok')"
```
Expected: dòng có `True NVIDIA GeForce RTX 5070 Ti Laptop GPU`, rồi `transformers ok`.

- [ ] **Step 4: Kiểm tra GGUF vẫn chạy cùng torch CUDA**

Run:
```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe test_caption.py --image "D:/001_Personal_Proj/Comfy/srcs/260914/Image/Screenshot 2026-09-20 100205.png" --max-tokens 40
```
Expected: log `GGUF model loaded successfully. device=cuda` và một caption tiếng Anh. Nếu llama.cpp lỗi DLL CUDA sau khi đổi torch, dừng lại và báo người dùng, không đi tiếp.

- [ ] **Step 5: Cập nhật requirements.txt**

Thay khối torch hiện tại bằng index đã cài được ở Step 2 (ví dụ cu132):
```text
# PyTorch - install separately with CUDA support (RTX 50xx / Blackwell needs CUDA 12.8+).
# The CPU wheel from PyPI breaks the HuggingFace backend (no CUDA, torchvision.io missing).
#   .venv\Scripts\python.exe -m pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu132
#torch>=2.0
#torchvision
```

- [ ] **Step 6: Commit**

```bash
git add requirements.txt
git commit -m "requirements: document the CUDA torch install the HF backend needs" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Bộ chuyển đổi ComfyUI single-file → thư mục HF

**Files:**
- Create: `captioner/comfy_convert.py`
- Test: `test_comfy_convert.py`

**Interfaces:**
- Consumes: không có.
- Produces:
  - `hf_key(key: str) -> str`
  - `to_bf16(tensor, scale=None) -> torch.Tensor`
  - `fetch_small_files(base_repo: str, dst: Path) -> None`
  - `convert(src, dst, base_repo: str, shard_gb: float = 2.0, log=logger.info, fetch=fetch_small_files) -> Path` — trả `dst`; nếu `dst/config.json` đã có thì bỏ qua; lỗi giữa chừng không để lại `dst` hay `<dst>.partial`.

- [ ] **Step 1: Viết test hỏng**

`test_comfy_convert.py`:
```python
"""Self-checks for captioner.comfy_convert. Run: .venv\\Scripts\\python.exe test_comfy_convert.py"""
import json
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from captioner.comfy_convert import convert, hf_key


def _fake_config(repo, dst):
    (Path(dst) / "config.json").write_text("{}", encoding="utf-8")


def _comfy_file(path: Path, with_scale=True):
    blob = torch.tensor(list(b'{"format": "float8_e4m3fn"}'), dtype=torch.uint8)
    tensors = {
        "model.layers.0.mlp.down_proj.weight": torch.tensor([[1.0, 2.0]]).to(torch.float8_e4m3fn),
        "model.layers.0.mlp.down_proj.comfy_quant": blob,
        "model.embed_tokens.weight": torch.tensor([[0.5, 0.25]], dtype=torch.bfloat16),
        "visual.blocks.0.attn.proj.weight": torch.tensor([[3.0]], dtype=torch.bfloat16),
    }
    if with_scale:
        tensors["model.layers.0.mlp.down_proj.weight_scale"] = torch.tensor(0.5, dtype=torch.float32)
    save_file(tensors, str(path))


def test_hf_key():
    assert hf_key("model.layers.3.self_attn.q_proj.weight") == "model.language_model.layers.3.self_attn.q_proj.weight"
    assert hf_key("model.embed_tokens.weight") == "model.language_model.embed_tokens.weight"
    assert hf_key("model.norm.weight") == "model.language_model.norm.weight"
    assert hf_key("visual.blocks.0.attn.proj.weight") == "model.visual.blocks.0.attn.proj.weight"
    for already in ("model.language_model.layers.0.mlp.up_proj.weight", "model.visual.merger.norm.weight", "lm_head.weight"):
        assert hf_key(already) == already


def test_convert_dequantizes_and_renames():
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "m.safetensors", Path(tmp) / "out" / "My-HF"
        _comfy_file(src)
        assert convert(src, dst, "Qwen/Qwen3-VL-4B-Instruct", fetch=_fake_config, log=lambda m: None) == dst
        index = json.loads((dst / "model.safetensors.index.json").read_text(encoding="utf-8"))
        assert set(index["weight_map"]) == {
            "model.language_model.layers.0.mlp.down_proj.weight",
            "model.language_model.embed_tokens.weight",
            "model.visual.blocks.0.attn.proj.weight",
        }
        weights = {}
        for shard in set(index["weight_map"].values()):
            weights.update(load_file(str(dst / shard)))
        down = weights["model.language_model.layers.0.mlp.down_proj.weight"]
        assert down.dtype == torch.bfloat16 and down.tolist() == [[0.5, 1.0]]
        assert (dst / "config.json").is_file() and not dst.with_name("My-HF.partial").exists()
        # second call is a no-op on an already converted folder
        assert convert(src, dst, "Qwen/Qwen3-VL-4B-Instruct", fetch=_fake_config, log=lambda m: None) == dst


def test_convert_failure_leaves_nothing():
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "m.safetensors", Path(tmp) / "out" / "Bad-HF"
        _comfy_file(src, with_scale=False)
        try:
            convert(src, dst, "Qwen/Qwen3-VL-4B-Instruct", fetch=_fake_config, log=lambda m: None)
        except ValueError:
            pass
        else:
            raise AssertionError("fp8 tensor without a scale must fail")
        assert not dst.exists() and not dst.with_name("Bad-HF.partial").exists()


if __name__ == "__main__":
    test_hf_key()
    test_convert_dequantizes_and_renames()
    test_convert_failure_leaves_nothing()
    print("test_comfy_convert OK")
```

- [ ] **Step 2: Chạy test để thấy hỏng**

Run: `.venv/Scripts/python.exe test_comfy_convert.py`
Expected: `ModuleNotFoundError: No module named 'captioner.comfy_convert'`

- [ ] **Step 3: Viết bộ chuyển đổi**

`captioner/comfy_convert.py`:
```python
"""
Convert a ComfyUI single-file Qwen3-VL checkpoint into a HuggingFace model folder.

ComfyUI text-encoder files hold fp8 weights with one float32 scale per tensor
(<layer>.weight_scale) plus a <layer>.comfy_quant JSON blob, use ComfyUI key
names, and ship no config, tokenizer or processor. convert() writes a normal
HF folder: bf16 shards + index, with the small files taken from the base repo.
"""
import json
import logging
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

logger = logging.getLogger(__name__)

SMALL_FILES = (
    "config.json", "generation_config.json", "chat_template.json", "tokenizer.json",
    "tokenizer_config.json", "vocab.json", "merges.txt", "preprocessor_config.json",
    "video_preprocessor_config.json",
)
_RENAMES = (
    ("model.layers.", "model.language_model.layers."),
    ("model.embed_tokens.", "model.language_model.embed_tokens."),
    ("model.norm.", "model.language_model.norm."),
    ("visual.", "model.visual."),
)
_DROP_SUFFIXES = (".comfy_quant", ".weight_scale")


def hf_key(key: str) -> str:
    """Map a ComfyUI tensor name to the transformers Qwen3-VL name. HF-style names pass through."""
    if key.startswith(("model.language_model.", "model.visual.", "lm_head.")):
        return key
    for old, new in _RENAMES:
        if key.startswith(old):
            return new + key[len(old):]
    return key


def to_bf16(tensor, scale=None):
    if tensor.dtype == torch.float8_e4m3fn:
        if scale is None:
            raise ValueError("fp8 tensor without a weight_scale")
        return (tensor.to(torch.float32) * scale.to(torch.float32)).to(torch.bfloat16)
    return tensor.to(torch.bfloat16) if tensor.is_floating_point() else tensor


def fetch_small_files(base_repo: str, dst: Path) -> None:
    """Config, tokenizer and processor files from the base repo (a few MB, cached by huggingface_hub)."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    for name in SMALL_FILES:
        try:
            hf_hub_download(base_repo, name, local_dir=str(dst))
        except EntryNotFoundError:
            pass


def convert(src, dst, base_repo: str, shard_gb: float = 2.0, log=logger.info, fetch=fetch_small_files) -> Path:
    src, dst = Path(src), Path(dst)
    if (dst / "config.json").is_file():
        log(f"{dst.name} is already converted")
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    need = src.stat().st_size * 2          # fp8 -> bf16 roughly doubles the size
    free = shutil.disk_usage(dst.parent).free
    if free < need:
        raise OSError(f"converting needs ~{need / 1e9:.1f} GB free in {dst.parent}, only {free / 1e9:.1f} GB left")

    tmp = dst.with_name(dst.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        fetch(base_repo, tmp)
        weight_map, shard = {}, {}
        state = {"bytes": 0, "total": 0, "index": 0}

        def flush():
            if not shard:
                return
            state["index"] += 1
            fname = f"model-{state['index']:05d}.safetensors"
            save_file(shard, str(tmp / fname), metadata={"format": "pt"})
            weight_map.update({k: fname for k in shard})
            log(f"wrote {fname} ({state['bytes'] / 1e9:.1f} GB)")
            shard.clear()
            state["bytes"] = 0

        with safe_open(str(src), framework="pt") as f:
            keys = set(f.keys())
            for key in sorted(keys):
                if key.endswith(_DROP_SUFFIXES):
                    continue
                scale_key = key[: -len(".weight")] + ".weight_scale" if key.endswith(".weight") else ""
                scale = f.get_tensor(scale_key) if scale_key in keys else None
                tensor = to_bf16(f.get_tensor(key), scale).contiguous()
                shard[hf_key(key)] = tensor
                size = tensor.numel() * tensor.element_size()
                state["bytes"] += size
                state["total"] += size
                if state["bytes"] >= shard_gb * 1e9:
                    flush()
        flush()
        (tmp / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": state["total"]}, "weight_map": weight_map}, indent=2),
            encoding="utf-8",
        )
        tmp.replace(dst)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    log(f"converted {src.name} -> {dst}")
    return dst
```

- [ ] **Step 4: Chạy test để thấy qua**

Run: `.venv/Scripts/python.exe test_comfy_convert.py`
Expected: `test_comfy_convert OK`

- [ ] **Step 5: Commit**

```bash
git add captioner/comfy_convert.py test_comfy_convert.py
git commit -m "Add comfy_convert: ComfyUI fp8 single-file Qwen3-VL to a bf16 HF folder" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Backend HF nạp được Qwen3-VL, kiểm chứng bằng model chuyển đổi thật

**Files:**
- Modify: `captioner/hf_captioner.py` (import loader ở `load_model`, đoạn chặn Qwen3-VL, `from_pretrained`, `_user_content`, `caption_image`)
- Modify: `app.py:222-223` (`_hf_model_supported_by_loader`)

**Interfaces:**
- Consumes: Task 1 (torch CUDA), Task 2 `convert(src, dst, base_repo, log=...)`.
- Produces: `HFCaptioner().load_model(vram_profile, model_id=<thư mục HF>, device="cuda")` chạy được với Qwen3-VL và Qwen2.5-VL; `caption_image(path, prompt, max_new_tokens, temperature=0)` trả caption. Thư mục `<root>\HuggingFace\qwen3-vl-4b-heretic-HF` đã chuyển đổi (Task 10 dùng lại).

- [ ] **Step 1: Chuyển đổi bản 4B heretic thật**

Run (vài phút, ghi ~8.9 GB):
```bash
.venv/Scripts/python.exe -c "from captioner.comfy_convert import convert; convert(r'D:\001_Personal_Proj\Comfy\ComfyUI\models\text_encoders\qwen3-vl-4b-heretic_fp8_e4m3fn.safetensors', r'D:\001_Personal_Proj\Comfy\ComfyUI\models\text_encoders\HuggingFace\qwen3-vl-4b-heretic-HF', 'Qwen/Qwen3-VL-4B-Instruct', log=print)"
```
Expected: vài dòng `wrote model-0000N.safetensors`, rồi `converted ... -> ...HuggingFace\qwen3-vl-4b-heretic-HF`.

- [ ] **Step 2: Chạy caption HF để thấy lỗi hiện tại**

Run:
```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe test_caption.py --backend hf --model-id "D:/001_Personal_Proj/Comfy/ComfyUI/models/text_encoders/HuggingFace/qwen3-vl-4b-heretic-HF" --vram-profile "LowVRAM (6–8GB)" --image "D:/001_Personal_Proj/Comfy/srcs/260914/Image/Screenshot 2026-09-20 100205.png" --max-tokens 40
```
Expected: FAIL. Lớp `Qwen2_5_VLForConditionalGeneration` không nạp được thư mục Qwen3-VL (lỗi `model_type` hoặc thiếu trọng số). Ghi lại thông báo lỗi để so sánh.

- [ ] **Step 3: Đổi loader sang Auto class**

Trong `captioner/hf_captioner.py`, `load_model`:

Thay
```python
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
```
bằng
```python
            from transformers import AutoModelForImageTextToText, AutoProcessor
```

Xoá nguyên khối bắt đầu bằng `if "Qwen3-VL" in model_id:` và `raise RuntimeError(...)` của nó.

Thay `"torch_dtype": ...` trong `model_kwargs` bằng `"dtype": ...` (transformers 5 dùng `dtype`).

Thay
```python
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id, **model_kwargs
            )
```
bằng
```python
            self.model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
```
và đổi chuỗi lỗi ngay dưới thành `f"Failed to load HuggingFace model '{model_id}'. Original error: {e}"`.

Thay khối nạp processor:
```python
            self.processor = AutoProcessor.from_pretrained(
                model_id,
                min_pixels=self._pixel_config["min_pixels"],
                max_pixels=self._pixel_config["max_pixels"],
            )
```
bằng
```python
            self.processor = AutoProcessor.from_pretrained(
                model_id,
                size={"shortest_edge": self._pixel_config["min_pixels"],
                      "longest_edge": self._pixel_config["max_pixels"]},
            )
```

Sửa docstring lớp thành `Image captioner using HuggingFace Transformers (Qwen3-VL and Qwen2.5-VL).`

- [ ] **Step 4: Cho processor tự đọc ảnh**

Trong `_user_content`, đổi mục ảnh thành đường dẫn thường (transformers đọc được, không cần `file://`):
```python
        return [{"type": "image", "image": str(Path(p).resolve())} for p in paths] + [
            {"type": "text", "text": user_prompt}
        ]
```

Trong `caption_image`, xoá dòng `from qwen_vl_utils import process_vision_info` và thay đoạn từ `text = self.processor.apply_chat_template(` đến hết `).to(target_device)` bằng:
```python
        target_device = "cuda" if (self._device_kind == "cuda" and torch_mod.cuda.is_available()) else "cpu"
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(target_device)
```
Phần `generate` và decode phía dưới giữ nguyên.

- [ ] **Step 5: Mở khoá Qwen3-VL trong app**

`app.py`, thay thân hàm `_hf_model_supported_by_loader`:
```python
def _hf_model_supported_by_loader(model_name: str) -> bool:
    return "Qwen3-VL" in model_name or "Qwen2.5-VL" in model_name or bool(HF_VL_MODELS.get(model_name, {}).get("library"))
```

- [ ] **Step 6: Chạy lại caption HF ở hai profile**

Chạy lại lệnh Step 2 (LowVRAM, 4-bit), rồi chạy với `--vram-profile "NormalVRAM (12–16GB)"` (8-bit).
Expected: cả hai in một câu caption tiếng Anh hợp lý về ảnh. Nếu `AutoProcessor.from_pretrained` báo lỗi về `size`, đổi lại thành `min_pixels=..., max_pixels=...` như cũ và chạy lại.

- [ ] **Step 7: Chạy test cũ để chắc không vỡ gì**

Run: `.venv/Scripts/python.exe test_briefs_prompts.py && .venv/Scripts/python.exe test_model_registry.py && .venv/Scripts/python.exe batch_caption.py --self-test`
Expected: cả ba in OK.

- [ ] **Step 8: Commit**

```bash
git add captioner/hf_captioner.py app.py
git commit -m "HF backend: load Qwen3-VL through AutoModelForImageTextToText" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Kết luận tĩnh cho từng file trong `model_registry.py`

**Files:**
- Modify: `model_registry.py`
- Test: `test_model_registry.py`

**Interfaces:**
- Consumes: hàm có sẵn `read_gguf_meta`, `_pick_mmproj`, `_QUANT_RE`, `VL_ARCHS`.
- Produces:
  - `QWEN3VL_BY_HIDDEN: dict[int, tuple[str, int, bool, str]]` — `(base_repo, layers, tied, size_label)`
  - `read_safetensors_header(path) -> tuple[dict, int]` — header (gồm `__metadata__`) và offset dữ liệu
  - `inspect_safetensors(path) -> dict | None`
  - `inspect_gguf(path, mmprojs: list[tuple[Path, dict]]) -> dict | None`
  - `inspect_path(path, root) -> dict`
  - `list_candidates(root, known: set[str]) -> list[dict]`
  - Dict kết luận ("candidate") luôn có: `kind` (`"gguf"`/`"safetensors"`/`"?"`), `path`, `name`, `usable: bool`, `reason: str`. GGUF thêm `quant`, `size`, `size_gb`, `mmproj`, `mmproj_options`. Safetensors thêm `hidden`, `layers`, `formats`, `bf16_gb`, và khi biết cỡ: `base_repo`, `size`.

- [ ] **Step 1: Viết test hỏng**

Thêm vào `test_model_registry.py` (giữ test cũ), ngay trên khối `if __name__ == "__main__":`:
```python
import json as _json

import torch
from safetensors.torch import save_file

from model_registry import inspect_path, inspect_safetensors, list_candidates


def _st(path: Path, hidden: int, layers: int, lm_head=False, fmt="float8_e4m3fn", hf_names=False):
    """Tiny Qwen3-VL-shaped safetensors: one fp8 weight per layer plus a comfy_quant blob."""
    prefix = "model.language_model." if hf_names else "model."
    t = {f"{prefix}embed_tokens.weight": torch.zeros(4, hidden, dtype=torch.bfloat16),
         "model.visual.deepstack_merger_list.0.linear_fc1.weight": torch.zeros(2, 2, dtype=torch.bfloat16)}
    blob = torch.tensor(list(_json.dumps({"format": fmt}).encode()), dtype=torch.uint8)
    for i in range(layers):
        t[f"{prefix}layers.{i}.mlp.down_proj.weight"] = torch.zeros(2, 2).to(torch.float8_e4m3fn)
        t[f"{prefix}layers.{i}.mlp.down_proj.weight_scale"] = torch.tensor(1.0)
        t[f"{prefix}layers.{i}.mlp.down_proj.comfy_quant"] = blob
    if lm_head:
        t["lm_head.weight"] = torch.zeros(4, hidden, dtype=torch.bfloat16)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(t, str(path))


def test_inspect_safetensors_verdicts():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _st(root / "ok-4b_fp8_scaled.safetensors", 2560, 36)
        _st(root / "hf-names-4b.safetensors", 2560, 36, hf_names=True)
        _st(root / "cut-32b.safetensors", 5120, 50)
        _st(root / "no-head-8b.safetensors", 4096, 36)
        _st(root / "nvfp4-4b.safetensors", 2560, 36, fmt="nvfp4")
        save_file({"model.layers.0.mlp.down_proj.weight": torch.zeros(2, 2)}, str(root / "text-only.safetensors"))

        ok = inspect_safetensors(root / "ok-4b_fp8_scaled.safetensors")
        assert ok["usable"] and ok["base_repo"] == "Qwen/Qwen3-VL-4B-Instruct" and ok["size"] == "4B", ok
        assert inspect_safetensors(root / "hf-names-4b.safetensors")["usable"]
        cut = inspect_safetensors(root / "cut-32b.safetensors")
        assert not cut["usable"] and "50 of 64 layers" in cut["reason"], cut
        assert "no lm_head" in inspect_safetensors(root / "no-head-8b.safetensors")["reason"]
        assert "nvfp4" in inspect_safetensors(root / "nvfp4-4b.safetensors")["reason"]
        assert inspect_safetensors(root / "text-only.safetensors") is None


def test_list_candidates_and_inspect_path():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _fake_gguf(root / "A" / "Qwen3-VL-8B-A.Q4_K_M.gguf", model("A", 4096))
        _fake_gguf(root / "A" / "mmproj-A-F16.gguf", mmproj("A", 4096))
        _fake_gguf(root / "Qwen3-8B-Q4_K_M.gguf",
                   {"general.architecture": "qwen3", "general.type": "model", "qwen3.embedding_length": 4096})
        _st(root / "ok-4b_fp8_scaled.safetensors", 2560, 36)
        _st(root / "HuggingFace" / "Done-HF" / "model-00001.safetensors", 2560, 36)
        (root / "HuggingFace" / "Done-HF" / "config.json").write_text("{}", encoding="utf-8")

        names = sorted(c["name"] for c in list_candidates(root, set()))
        assert names == ["Qwen3-VL-8B-A.Q4_K_M.gguf", "ok-4b_fp8_scaled.safetensors"], names
        gguf = next(c for c in list_candidates(root, set()) if c["kind"] == "gguf")
        assert gguf["usable"] and gguf["quant"] == "Q4_K_M" and gguf["mmproj"].endswith("mmproj-A-F16.gguf")

        known = {str((root / "A" / "Qwen3-VL-8B-A.Q4_K_M.gguf").resolve())}
        assert [c["kind"] for c in list_candidates(root, known)] == ["safetensors"]

        assert not inspect_path(root / "missing.gguf", root)["usable"]
        assert "not a Qwen3-VL" in inspect_path(root / "Qwen3-8B-Q4_K_M.gguf", root)["reason"]
        assert inspect_path(root / "A" / "Qwen3-VL-8B-A.Q4_K_M.gguf", root)["usable"]
```
Và thêm vào khối `__main__`, trước dòng `print`:
```python
    test_inspect_safetensors_verdicts()
    test_list_candidates_and_inspect_path()
```

- [ ] **Step 2: Chạy test để thấy hỏng**

Run: `.venv/Scripts/python.exe test_model_registry.py`
Expected: `ImportError: cannot import name 'inspect_path' from 'model_registry'`

- [ ] **Step 3: Tách hàm thu thập GGUF**

Trong `model_registry.py`, thêm `import math` vào nhóm import. Thêm hàm sau `_pick_mmproj`:
```python
def _collect_gguf(root):
    """Qwen3-VL model GGUFs and qwen3vl mmprojs under root, plus unreadable files."""
    models, mmprojs, skipped = [], [], []
    for p in sorted(Path(root).rglob("*.gguf")):
        try:
            meta = read_gguf_meta(p)
        except (OSError, struct.error, KeyError) as e:
            skipped.append({"path": str(p), "reason": f"unreadable GGUF header: {e}"})
            continue
        if str(meta.get("clip.projector_type", "")).startswith("qwen3vl"):
            mmprojs.append((p, meta))
        elif meta.get("general.architecture") in VL_ARCHS:
            models.append((p, meta))
    return models, mmprojs, skipped
```
Trong `scan_models`, thay toàn bộ khối từ `models, mmprojs = [], []` đến hết vòng `for p in sorted(root.rglob("*.gguf")):` (kết thúc ở `models.append((p, meta))`) bằng:
```python
    models, mmprojs, skipped = _collect_gguf(root)
    registry["skipped"] += skipped
```

- [ ] **Step 4: Thêm đọc header và kết luận**

Thêm vào `model_registry.py`, sau `GGUF_DEFAULTS`:
```python
# hidden size -> (base repo, text layers, tied word embeddings, size label), from the official config.json files
QWEN3VL_BY_HIDDEN = {
    2048: ("Qwen/Qwen3-VL-2B-Instruct", 28, True, "2B"),
    2560: ("Qwen/Qwen3-VL-4B-Instruct", 36, True, "4B"),
    4096: ("Qwen/Qwen3-VL-8B-Instruct", 36, False, "8B"),
    5120: ("Qwen/Qwen3-VL-32B-Instruct", 64, False, "32B"),
}
SUPPORTED_COMFY_QUANT = {"float8_e4m3fn"}
_LAYER_RE = re.compile(r"^model\.(?:language_model\.)?layers\.(\d+)\.")
_SCALE_SUFFIXES = (".comfy_quant", ".weight_scale", ".weight_scale_2", ".pre_quant_scale", ".input_scale")
```
Thay hàm `_is_qwen3vl_safetensors` bằng:
```python
def read_safetensors_header(path):
    """(header dict incl. __metadata__, byte offset where tensor data starts)."""
    with open(path, "rb") as f:
        n, = struct.unpack("<Q", f.read(8))
        if n > 100_000_000:
            raise ValueError(f"safetensors header too large: {n} bytes")
        return json.loads(f.read(n)), 8 + n


def _comfy_formats(path, header: dict, data_start: int) -> set:
    formats = set()
    with open(path, "rb") as f:
        for key, info in header.items():
            if key.endswith(".comfy_quant"):
                start, end = info["data_offsets"]
                f.seek(data_start + start)
                formats.add(json.loads(f.read(end - start)).get("format", "?"))
    return formats


def inspect_safetensors(path):
    """Static verdict for a single-file safetensors; None when it is not Qwen3-VL."""
    path = Path(path)
    header, data_start = read_safetensors_header(path)
    header.pop("__metadata__", None)
    if not any("deepstack" in k for k in header):
        return None
    embed = next((v["shape"] for k, v in header.items() if k.endswith("embed_tokens.weight")), [0, 0])
    hidden = embed[1]
    layers = len({m.group(1) for k in header if (m := _LAYER_RE.match(k))})
    formats = _comfy_formats(path, header, data_start)
    params = sum(math.prod(v["shape"]) for k, v in header.items() if not k.endswith(_SCALE_SUFFIXES))
    verdict = {"kind": "safetensors", "path": str(path), "name": path.name, "hidden": hidden,
               "layers": layers, "formats": sorted(formats), "bf16_gb": round(params * 2 / 1e9, 1),
               "usable": False, "reason": ""}
    ref = QWEN3VL_BY_HIDDEN.get(hidden)
    if ref is None:
        verdict["reason"] = f"unknown Qwen3-VL size (hidden {hidden})"
        return verdict
    base_repo, n_layers, tied, size = ref
    verdict.update(base_repo=base_repo, size=size)
    unsupported = formats - SUPPORTED_COMFY_QUANT
    if layers != n_layers:
        verdict["reason"] = f"not usable: {layers} of {n_layers} layers (cut text encoder)"
    elif not tied and not any(k.startswith("lm_head.") for k in header):
        verdict["reason"] = "not usable: no lm_head (text encoder only)"
    elif unsupported:
        verdict["reason"] = f"not usable: unsupported quant {', '.join(sorted(unsupported))}"
    else:
        verdict.update(usable=True, reason=f"safetensors, converts once to bf16 (~{verdict['bf16_gb']} GB)")
    return verdict


def inspect_gguf(path, mmprojs):
    """Static verdict for a GGUF file; None when it is not a Qwen3-VL model (text-only, mmproj...)."""
    path = Path(path)
    meta = read_gguf_meta(path)
    if meta.get("general.architecture") not in VL_ARCHS:
        return None
    dim = meta.get(f"{meta['general.architecture']}.embedding_length")
    best = _pick_mmproj(path, meta, mmprojs)
    quant = _QUANT_RE.search(path.stem)
    return {
        "kind": "gguf", "path": str(path), "name": path.name,
        "quant": quant.group(1).upper() if quant else "unknown",
        "size": meta.get("general.size_label", ""),
        "size_gb": round(path.stat().st_size / 1e9, 1),
        "mmproj": str(best) if best else "",
        "mmproj_options": sorted({str(p) for p, m in mmprojs if dim and m.get("clip.vision.projection_dim") == dim}),
        "usable": best is not None,
        "reason": f"GGUF, {dim}-dim mmproj found" if best else f"no mmproj with projection_dim {dim} found",
    }


def inspect_path(path, root) -> dict:
    """Verdict for a pasted path. mmprojs are looked up under root and next to the file."""
    path = Path(path)
    unusable = {"kind": "?", "path": str(path), "name": path.name, "usable": False}
    if not path.is_file():
        return dict(unusable, reason=f"file not found: {path}")
    verdict = None
    try:
        if path.suffix.lower() == ".gguf":
            mmprojs = _collect_gguf(root)[1] + _collect_gguf(path.parent)[1]
            verdict = inspect_gguf(path, mmprojs)
        elif path.suffix.lower() == ".safetensors":
            verdict = inspect_safetensors(path)
    except (OSError, ValueError, KeyError, struct.error) as e:
        return dict(unusable, reason=f"unreadable file: {e}")
    return verdict or dict(unusable, reason="not a Qwen3-VL model (text-only, mmproj or another architecture)")


def list_candidates(root, known: set) -> list:
    """Qwen3-VL model files under root not in `known` (resolved path strings), each with a verdict.
    Text-only models, mmprojs and shards of converted HF folders are left out."""
    root = Path(root)
    out = []
    models, mmprojs, _ = _collect_gguf(root)
    for p, _meta in models:
        if str(p.resolve()) not in known:
            out.append(inspect_gguf(p, mmprojs))
    for p in sorted(root.rglob("*.safetensors")):
        if (p.parent / "config.json").exists() or str(p.resolve()) in known:
            continue
        try:
            verdict = inspect_safetensors(p)
        except (OSError, ValueError, KeyError, struct.error):
            verdict = None
        if verdict:
            out.append(verdict)
    return out
```
Trong `scan_models`, thay vòng safetensors single-file (khối `for p in sorted(root.rglob("*.safetensors")):` gọi `_is_qwen3vl_safetensors`) bằng:
```python
    for p in sorted(root.rglob("*.safetensors")):
        if (p.parent / "config.json").exists():
            continue   # shard of an HF folder, already listed above
        try:
            verdict = inspect_safetensors(p)
        except (OSError, struct.error, ValueError, KeyError):
            verdict = None
        if verdict:
            registry["safetensors"].append({"name": p.stem, "format": "single-file", "path": str(p),
                                            "loadable": verdict["usable"], "reason": verdict["reason"]})
```

- [ ] **Step 5: Chạy test để thấy qua**

Run: `.venv/Scripts/python.exe test_model_registry.py`
Expected: `test_model_registry OK`

- [ ] **Step 6: Chạy trên thư mục thật**

Run:
```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -c "from model_registry import list_candidates, DEFAULT_LLM_DIR; [print(c['usable'], c['name'], '|', c['reason']) for c in list_candidates(DEFAULT_LLM_DIR, set())]"
```
Expected: 5 GGUF `True`, ba file 4B safetensors `True ... converts once`, hai file 32B `False ... 50 of 64 layers`. Không có `Qwen3-8B-Q4_K_M.gguf`, `qwen_3_4b_fp4_mixed`, hay shard trong `HuggingFace\`.

- [ ] **Step 7: Commit**

```bash
git add model_registry.py test_model_registry.py
git commit -m "model_registry: static verdicts for GGUF and ComfyUI safetensors, candidate listing" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: `model_library.py` và ngân sách VRAM dùng chung

**Files:**
- Create: `model_library.py`
- Modify: `models_catalog.py` (`VRAM_PROFILES`)
- Test: `test_model_library.py`

**Interfaces:**
- Consumes: `model_registry._QUANT_RE`; dict candidate của Task 4.
- Produces:
  - `LIBRARY_PATH: Path`, `GGUF_DEFAULTS: dict`, `GGUF_LOAD_KWARGS: dict`
  - `load(path=LIBRARY_PATH) -> tuple[list[dict], str | None]`
  - `save(entries, path=LIBRARY_PATH) -> None`
  - `to_stored(path, root) -> str`, `abs_path(stored, root) -> Path`
  - `status(entry, root) -> "found" | "downloadable" | "missing"`
  - `find(entries, model_key) -> dict | None`, `upsert(entries, entry) -> list`, `remove(entries, model_key) -> list` — khoá là trường `"model"`
  - `known_paths(entries, root) -> set[str]`
  - `suggest_name(model_path, kind, gguf_catalog) -> str`
  - `new_entry(candidate, name, root, mmproj=None, base_repo=None, source_repo="") -> dict`
  - `record(entry, result) -> dict`
  - `apply_to_catalogs(entries, root, gguf_catalog, hf_catalog) -> dict[str, str]` — khoá là `entry["model"]`
  - `unregister(entry, root, gguf_catalog, hf_catalog, official_names) -> None`
  - `restore(entry, root, log=logger.info) -> None`
  - `VRAM_PROFILES[name]["budget_gb"]`: 4.0 / 8.0 / 16.0 / 24.0

- [ ] **Step 1: Viết test hỏng**

`test_model_library.py`:
```python
"""Self-checks for model_library. Run: .venv\\Scripts\\python.exe test_model_library.py"""
import tempfile
from pathlib import Path

import model_library as ml

GGUF_CAND = {"kind": "gguf", "path": "", "name": "Qwen3-VL-8B-X.Q4_K_M.gguf", "quant": "Q4_K_M", "size": "8B",
             "size_gb": 5.0, "mmproj": "", "usable": True, "reason": ""}


def _files(root: Path):
    (root / "GGUF").mkdir(parents=True)
    model, mm = root / "GGUF" / "Qwen3-VL-8B-X.Q4_K_M.gguf", root / "GGUF" / "mmproj-X.gguf"
    model.write_bytes(b"x")
    mm.write_bytes(b"x")
    return model, mm


def test_paths_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        assert ml.to_stored(root / "GGUF" / "a.gguf", root) == "GGUF/a.gguf"
        outside = Path(tempfile.gettempdir()) / "elsewhere.gguf"
        assert ml.to_stored(outside, root) == str(outside.resolve())
        assert ml.abs_path("GGUF/a.gguf", root) == root / "GGUF" / "a.gguf"
        assert ml.abs_path(str(outside), root) == outside


def test_save_load_and_bad_json():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "lib.json"
        assert ml.load(path) == ([], None)
        ml.save([{"name": "a", "model": "GGUF/a.gguf"}], path)
        assert ml.load(path) == ([{"name": "a", "model": "GGUF/a.gguf"}], None)
        path.write_text("{broken", encoding="utf-8")
        entries, warning = ml.load(path)
        assert entries == [] and "lib.json.bak" in warning and path.with_suffix(".json.bak").exists()


def test_status_new_entry_and_catalogs():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        model, mm = _files(root)
        cand = dict(GGUF_CAND, path=str(model), mmproj=str(mm))
        entry = ml.new_entry(cand, "Qwen3-VL-8B-X-GGUF", root, source_repo="someone/x-gguf")
        assert entry["model"] == "GGUF/Qwen3-VL-8B-X.Q4_K_M.gguf" and entry["mmproj"] == "GGUF/mmproj-X.gguf"
        assert entry["source"] == {"repo_id": "someone/x-gguf", "files": ["Qwen3-VL-8B-X.Q4_K_M.gguf", "mmproj-X.gguf"]}
        assert ml.status(entry, root) == "found"

        tested = ml.record(entry, {"date": "2026-10-25", "gpu": "GPU", "load_s": 1.0, "caption_s": 1.0,
                                   "vision_score": 5, "peak_vram_gb": 6.4, "profile": "LowVRAM (6–8GB)"})
        assert tested["test"]["peak_vram_gb"] == {"*": 6.4}

        gguf_cat, hf_cat = {}, {}
        assert ml.apply_to_catalogs([tested], root, gguf_cat, hf_cat) == {entry["model"]: "found"}
        info = gguf_cat["Qwen3-VL-8B-X-GGUF"]
        assert info["model_files"] == {"Q4_K_M (tested, ~6.4GB)": str(model)}
        assert info["mmproj_file"] == str(mm) and info["library"] is True

        model.unlink()
        assert ml.status(entry, root) == "downloadable"
        assert ml.status(dict(entry, source=None), root) == "missing"

        ml.unregister(tested, root, gguf_cat, hf_cat, official_names=set())
        assert "Qwen3-VL-8B-X-GGUF" not in gguf_cat


def test_hf_entry_and_merge_into_official():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        src = root / "x_fp8_scaled.safetensors"
        src.write_bytes(b"x")
        cand = {"kind": "safetensors", "path": str(src), "name": src.name, "base_repo": "Qwen/Qwen3-VL-4B-Instruct",
                "size": "4B", "bf16_gb": 8.9, "usable": True, "reason": ""}
        entry = ml.new_entry(cand, "x-HF", root)
        assert (entry["backend"], entry["model"], entry["origin"]) == ("hf", "HuggingFace/x-HF", "x_fp8_scaled.safetensors")
        assert ml.status(entry, root) == "downloadable"          # origin present, folder not converted yet
        (root / "HuggingFace" / "x-HF").mkdir(parents=True)
        (root / "HuggingFace" / "x-HF" / "config.json").write_text("{}", encoding="utf-8")
        assert ml.status(entry, root) == "found"

        hf_cat = {}
        tested = ml.record(entry, {"date": "d", "gpu": "g", "load_s": 1, "caption_s": 1, "vision_score": 5,
                                   "peak_vram_gb": 3.4, "profile": "LowVRAM (6–8GB)"})
        ml.apply_to_catalogs([tested], root, {}, hf_cat)
        assert hf_cat["x-HF"]["vram"]["4bit"] == 3.4 and hf_cat["x-HF"]["local_dir"] == str(root / "HuggingFace" / "x-HF")

        # a GGUF entry whose file belongs to an official catalog entry joins that entry
        model, mm = _files(root / "r2")
        official = {"Qwen3-VL-8B-X-GGUF": {"model_files": {"Q4_K_M (recommended, ~5GB)": "Qwen3-VL-8B-X.Q4_K_M.gguf",
                                                           "Q8_0 (~9GB)": "Qwen3-VL-8B-X.Q8_0.gguf"},
                                           "mmproj_file": "mmproj-X-F16.gguf", "min_vram_4gb": False}}
        root2 = root / "r2"
        name = ml.suggest_name(model, "gguf", official)
        assert name == "Qwen3-VL-8B-X-GGUF"
        g = ml.new_entry(dict(GGUF_CAND, path=str(model), mmproj=str(mm)), name, root2)
        ml.apply_to_catalogs([g], root2, official, {})
        files = official[name]["model_files"]
        assert len(files) == 2 and "Q4_K_M (untested, ~5.0GB)" in files and "Q8_0 (~9GB)" in files
        assert ml.suggest_name(root / "Huihui-Qwen3-VL-4B-abliterated-fp8_scaled.safetensors", "safetensors", {}) \
            == "Huihui-Qwen3-VL-4B-abliterated-HF"


def test_upsert_remove_known():
    a = {"name": "a", "backend": "gguf", "model": "GGUF/a.gguf"}
    b = {"name": "b", "backend": "hf", "model": "HuggingFace/b", "origin": "b.safetensors"}
    entries = ml.upsert(ml.upsert([], a), b)
    entries = ml.upsert(entries, dict(a, quant="Q8_0"))
    assert len(entries) == 2 and ml.find(entries, "GGUF/a.gguf")["quant"] == "Q8_0"
    root = Path(tempfile.gettempdir())
    assert ml.known_paths(entries, root) == {str((root / "GGUF/a.gguf").resolve()), str((root / "b.safetensors").resolve())}
    assert ml.remove(entries, "GGUF/a.gguf") == [b]


if __name__ == "__main__":
    test_paths_round_trip()
    test_save_load_and_bad_json()
    test_status_new_entry_and_catalogs()
    test_hf_entry_and_merge_into_official()
    test_upsert_remove_known()
    print("test_model_library OK")
```

- [ ] **Step 2: Chạy test để thấy hỏng**

Run: `.venv/Scripts/python.exe test_model_library.py`
Expected: `ModuleNotFoundError: No module named 'model_library'`

- [ ] **Step 3: Thêm `budget_gb` vào profile**

`models_catalog.py`, trong `VRAM_PROFILES` thêm một dòng vào mỗi profile, ngay dưới `"label"`:
- `"UltraLow (4GB)"`: `"budget_gb": 4.0,`
- `"LowVRAM (6–8GB)"`: `"budget_gb": 8.0,`
- `"NormalVRAM (12–16GB)"`: `"budget_gb": 16.0,`
- `"HighVRAM (20GB+)"`: `"budget_gb": 24.0,`

- [ ] **Step 4: Viết `model_library.py`**

```python
"""
The model library: models_library.json lists the models that passed a quick test.

Each entry is one model file (GGUF + its mmproj) or one HuggingFace folder,
with paths stored relative to the models root so the list works on another
machine. At startup the app checks every entry and registers it into the
catalog dicts (GGUF_VL_MODELS / HF_VL_MODELS) that the dropdown already reads.
"""
import json
import logging
import re
from pathlib import Path

from model_registry import _QUANT_RE

logger = logging.getLogger(__name__)

LIBRARY_PATH = Path(__file__).with_name("models_library.json")
GGUF_DEFAULTS = {
    "context_length": 8192,
    "image_max_tokens": 4096,
    "n_batch": 512,
    "gpu_layers": -1,
    "top_k": 0,
    "pool_size": 4194304,
}
# the same settings spelled the way GGUFCaptioner.load_model takes them as kwargs
GGUF_LOAD_KWARGS = {"gpu_layers": -1, "n_ctx": 8192, "n_batch": 512, "image_max_tokens": 4096,
                    "top_k": 0, "pool_size": 4194304}
_HF_QUANT_KEY = {"4bit": "4bit", "8bit": "8bit", "none": "full"}
_DTYPE_SUFFIX_RE = re.compile(r"(?i)[-_.](fp8\w*|nvfp4\w*|bf16|fp16)$")


def load(path: Path = LIBRARY_PATH):
    """(entries, warning). A broken file is moved to .json.bak so the app still starts."""
    if not path.exists():
        return [], None
    try:
        return list(json.loads(path.read_text(encoding="utf-8"))["models"]), None
    except (ValueError, KeyError, TypeError) as e:
        bak = path.with_suffix(".json.bak")
        path.replace(bak)
        return [], f"{path.name} could not be read ({e}). It was moved to {bak.name} and the library starts empty."


def save(entries, path: Path = LIBRARY_PATH) -> None:
    path.write_text(json.dumps({"version": 1, "models": entries}, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def to_stored(path, root) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return str(path)


def abs_path(stored: str, root) -> Path:
    p = Path(stored)
    return p if p.is_absolute() else Path(root) / p


def status(entry, root) -> str:
    model = abs_path(entry["model"], root)
    if entry["backend"] == "gguf":
        present = model.is_file() and abs_path(entry["mmproj"], root).is_file()
    else:
        present = (model / "config.json").is_file()
    if present:
        return "found"
    origin = entry.get("origin")
    if entry.get("source") or (origin and abs_path(origin, root).is_file()):
        return "downloadable"
    return "missing"


def find(entries, model_key):
    return next((e for e in entries if e["model"] == model_key), None)


def upsert(entries, entry) -> list:
    return [e for e in entries if e["model"] != entry["model"]] + [entry]


def remove(entries, model_key) -> list:
    return [e for e in entries if e["model"] != model_key]


def known_paths(entries, root) -> set:
    """Resolved source files already in the library (GGUF model, or the safetensors an HF entry came from)."""
    out = set()
    for e in entries:
        source = e.get("origin") if e["backend"] == "hf" else e["model"]
        if source:
            out.add(str(abs_path(source, root).resolve()))
    return out


def suggest_name(model_path, kind: str, gguf_catalog: dict) -> str:
    p = Path(model_path)
    if kind == "gguf":
        for key, info in gguf_catalog.items():
            if any(Path(f).name.lower() == p.name.lower() for f in info.get("model_files", {}).values()):
                return key
        return f"{_QUANT_RE.sub('', p.stem)}-GGUF"
    return f"{_DTYPE_SUFFIX_RE.sub('', p.stem)}-HF"


def new_entry(candidate, name, root, mmproj=None, base_repo=None, source_repo="") -> dict:
    if candidate["kind"] == "gguf":
        mm = Path(mmproj or candidate["mmproj"])
        source = ({"repo_id": source_repo, "files": [Path(candidate["path"]).name, mm.name]}
                  if source_repo else None)
        return {"name": name, "backend": "gguf", "model": to_stored(candidate["path"], root),
                "mmproj": to_stored(mm, root), "quant": candidate["quant"], "size": candidate.get("size", ""),
                "size_gb": candidate["size_gb"], "source": source}
    return {"name": name, "backend": "hf", "model": f"HuggingFace/{name}",
            "origin": to_stored(candidate["path"], root), "base_repo": base_repo or candidate["base_repo"],
            "quant": "bf16", "size": candidate.get("size", ""), "size_gb": candidate["bf16_gb"], "source": None}


def record(entry, result) -> dict:
    """Entry with the quick-test result merged in. GGUF VRAM applies to every profile ("*")."""
    test = dict(entry.get("test") or {})
    peaks = dict(test.get("peak_vram_gb") or {})
    if result.get("peak_vram_gb") is not None:
        peaks["*" if entry["backend"] == "gguf" else result["profile"]] = result["peak_vram_gb"]
    test.update({k: result[k] for k in ("date", "gpu", "load_s", "caption_s", "vision_score")})
    test["peak_vram_gb"] = peaks
    return dict(entry, test=test)


def _gguf_label(entry) -> str:
    peak = ((entry.get("test") or {}).get("peak_vram_gb") or {}).get("*")
    gb = peak if peak is not None else entry.get("size_gb", 0)
    return f"{entry['quant']} ({'tested' if peak is not None else 'untested'}, ~{gb}GB)"


def _hf_vram(entry) -> dict:
    from models_catalog import VRAM_PROFILES

    size = entry.get("size_gb", 0)
    # ponytail: rough bf16 / int8 / nf4 ratios, replaced by measured values once a profile is tested
    vram = {"full": size + 1.0, "8bit": size * 0.55 + 1.0, "4bit": size * 0.35 + 1.0}
    for profile, gb in ((entry.get("test") or {}).get("peak_vram_gb") or {}).items():
        if profile in VRAM_PROFILES:
            vram[_HF_QUANT_KEY[VRAM_PROFILES[profile]["hf_quant"]]] = gb
    return {k: round(v, 1) for k, v in vram.items()}


def apply_to_catalogs(entries, root, gguf_catalog, hf_catalog) -> dict:
    """Register entries into the catalog dicts the app reads. Returns {entry['model']: status}."""
    statuses = {}
    for e in entries:
        statuses[e["model"]] = status(e, root)
        model = str(abs_path(e["model"], root))
        repo_id = (e.get("source") or {}).get("repo_id", "")
        if e["backend"] == "gguf":
            info = gguf_catalog.setdefault(e["name"], {
                "repo_id": repo_id, "series": "Qwen3-VL", "size": e.get("size", ""),
                "gguf_defaults": dict(GGUF_DEFAULTS), "model_files": {}, "min_vram_4gb": False,
                "hf_url": "", "description": f"Library model ({e['quant']})",
            })
            name = Path(model).name.lower()
            info["model_files"] = {l: f for l, f in info["model_files"].items() if Path(f).name.lower() != name}
            info["model_files"][_gguf_label(e)] = model
            info["mmproj_file"] = str(abs_path(e["mmproj"], root))
            peak = ((e.get("test") or {}).get("peak_vram_gb") or {}).get("*", e.get("size_gb", 99))
            info["min_vram_4gb"] = info.get("min_vram_4gb", False) or peak <= 4.0
        else:
            origin = Path(e.get("origin", "")).name
            info = hf_catalog.setdefault(e["name"], {
                "repo_id": repo_id, "series": "Qwen3-VL", "size": e.get("size", ""), "quantized": False,
                "min_vram_4gb": False, "hf_url": "",
                "description": f"Library model converted from {origin}" if origin else "Library model",
            })
            info["local_dir"] = model
            info["vram"] = _hf_vram(e)
            info["min_vram_4gb"] = info["vram"]["4bit"] <= 4.0
        info["library"] = True
    return statuses


def unregister(entry, root, gguf_catalog, hf_catalog, official_names) -> None:
    """Undo apply_to_catalogs for one removed entry (official catalog entries only lose that file)."""
    catalog = gguf_catalog if entry["backend"] == "gguf" else hf_catalog
    info = catalog.get(entry["name"])
    if info is None:
        return
    if entry["name"] not in official_names:
        catalog.pop(entry["name"])
    elif entry["backend"] == "gguf":
        name = abs_path(entry["model"], root).name.lower()
        info["model_files"] = {l: f for l, f in info["model_files"].items() if Path(f).name.lower() != name}


def restore(entry, root, log=logger.info) -> None:
    """Bring back a missing entry: re-convert from its origin file, or download from its source repo."""
    origin = entry.get("origin")
    if entry["backend"] == "hf" and origin and abs_path(origin, root).is_file():
        from captioner.comfy_convert import convert
        convert(abs_path(origin, root), abs_path(entry["model"], root), entry["base_repo"], log=log)
        return
    source = entry.get("source") or {}
    if not source.get("repo_id"):
        raise FileNotFoundError(f"{entry['name']}: {entry['model']} is missing and has no download source")
    from huggingface_hub import hf_hub_download, snapshot_download
    if entry["backend"] == "hf":
        snapshot_download(source["repo_id"], local_dir=str(abs_path(entry["model"], root)))
        return
    for stored, remote in zip((entry["model"], entry["mmproj"]), source["files"]):
        dst = abs_path(stored, root)
        if not dst.is_file():
            log(f"downloading {remote} from {source['repo_id']}")
            got = Path(hf_hub_download(source["repo_id"], remote, local_dir=str(dst.parent)))
            if got.resolve() != dst.resolve():
                got.replace(dst)
```

- [ ] **Step 5: Chạy test để thấy qua**

Run: `.venv/Scripts/python.exe test_model_library.py`
Expected: `test_model_library OK`

- [ ] **Step 6: Commit**

```bash
git add model_library.py test_model_library.py models_catalog.py
git commit -m "Add model_library: models_library.json with relative paths, status, catalog registration" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: App khởi động theo thư viện, banner model thiếu, khôi phục khi Load

**Files:**
- Modify: `app.py` (import dòng 72; khối khởi động dòng 88-94; `_available_vram_for_profile`; `_hf_storage_dir`; `_build_model_choice_label`; `_get_candidate_model_names`; `_render_model_info`; `load_model`; Setup tab)
- Modify: `model_registry.py` (xoá `REGISTRY_PATH`, `GGUF_DEFAULTS`, `_norm`, `refresh_registry`, `apply_to_catalog`, `SAFETENSORS_REASON`; sửa `__main__`)
- Modify: `test_model_registry.py` (bỏ phần `apply_to_catalog`)
- Modify: `.gitignore` (bỏ dòng `models_registry.json`)

**Interfaces:**
- Consumes: Task 5 toàn bộ `model_library`; Task 4 `scan_models`.
- Produces (biến toàn cục trong `app.py`, Task 9 dùng): `LIBRARY: list[dict]`, `LIBRARY_WARNING: str | None`, `LIBRARY_STATUS: dict[str, str]` (khoá `entry["model"]`), `OFFICIAL_MODEL_NAMES: set[str]`, `LOCAL_GGUF_FILES: set[str]`; hàm `_library_banner() -> str`, `_library_entry_for(model_name, vram_profile) -> dict | None`; component `library_banner` (gr.HTML).

- [ ] **Step 1: Dọn model_registry**

Trong `model_registry.py`: xoá `REGISTRY_PATH`, `SAFETENSORS_REASON`, `GGUF_DEFAULTS`, hàm `_norm`, `refresh_registry`, `apply_to_catalog`. Trong `scan_models`, khối HF folder đang dùng `SAFETENSORS_REASON`: đổi `"loadable": False, "reason": SAFETENSORS_REASON` thành `"loadable": True, "reason": "HuggingFace folder"`. Thay khối `if __name__ == "__main__":` bằng:
```python
if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_LLM_DIR
    for c in list_candidates(root, set()):
        print(f"{'OK ' if c['usable'] else 'NO '} {c['kind']:<11} {c['path']}\n     {c['reason']}")
```
Sửa docstring đầu file: bỏ câu nói về việc ghi `models_registry.json`. Giữ `import time` (còn dùng trong `scan_models`); bỏ `import logging`/`logger` nếu không còn chỗ nào dùng.

Trong `test_model_registry.py`: xoá `apply_to_catalog` khỏi dòng import và xoá đoạn cuối `test_filters_and_pairs` bắt đầu từ `catalog = {"Qwen3-VL-8B-Instruct-GGUF": ...` đến hết hai dòng assert về `known`.

Trong `.gitignore` xoá dòng `models_registry.json`, rồi xoá file cũ: `rm -f models_registry.json`.

- [ ] **Step 2: Kiểm tra app hiện tại vỡ import (để chắc đã gỡ đúng chỗ)**

Run: `.venv/Scripts/python.exe -c "import sys; sys.argv=['app.py']; import app"`
Expected: `ImportError: cannot import name 'refresh_registry'`

- [ ] **Step 3: Khởi động theo thư viện**

`app.py`, thay dòng import
```python
from model_registry import DEFAULT_LLM_DIR, refresh_registry, apply_to_catalog
```
bằng
```python
import model_library
import model_registry
from model_registry import DEFAULT_LLM_DIR, scan_models
```
Thay khối từ `# Scan LLM_DIR for Qwen3-VL models before the UI is built` đến hết dòng `LOCAL_GGUF_FILES = ...` bằng:
```python
# Model library (models_library.json): the models that passed a quick test, paths relative to LLM_DIR.
# Every entry is checked before the UI is built; missing ones are reported, never dropped.
OFFICIAL_MODEL_NAMES = set(GGUF_VL_MODELS) | set(HF_VL_MODELS)
LIBRARY, LIBRARY_WARNING = model_library.load()
LIBRARY_STATUS = model_library.apply_to_catalogs(LIBRARY, LLM_DIR, GGUF_VL_MODELS, HF_VL_MODELS)
for _e in LIBRARY:
    if LIBRARY_STATUS[_e["model"]] != "found":
        logger.warning("Library model %s is %s: %s", _e["name"], LIBRARY_STATUS[_e["model"]], _e["model"])
LOCAL_GGUF_FILES = {Path(m["model_path"]).name.lower() for m in scan_models(LLM_DIR)["gguf"]} | {
    Path(e["model"]).name.lower() for e in LIBRARY
    if e["backend"] == "gguf" and LIBRARY_STATUS[e["model"]] == "found"
}
```

- [ ] **Step 4: Dùng ngân sách và thư mục HF từ catalog**

Thay thân `_available_vram_for_profile`:
```python
def _available_vram_for_profile(vram_profile: str) -> float:
    return VRAM_PROFILES.get(vram_profile, {}).get("budget_gb", 8.0)
```
Thay thân `_hf_storage_dir`:
```python
def _hf_storage_dir(llm_dir: Path, model_name: str) -> Path:
    local = HF_VL_MODELS.get(model_name, {}).get("local_dir")
    return Path(local) if local else llm_dir / "HuggingFace" / model_name
```

- [ ] **Step 5: Tìm entry thư viện, banner, nhãn dropdown**

Thêm sau `_hf_storage_dir`:
```python
def _library_entry_for(model_name: str, vram_profile: str):
    """The library entry behind a dropdown choice (GGUF: the variant this profile picks)."""
    if model_name in HF_VL_MODELS and HF_VL_MODELS[model_name].get("library"):
        return next((e for e in LIBRARY if e["backend"] == "hf" and e["name"] == model_name), None)
    _, filename = _pick_gguf_variant(model_name, vram_profile)
    if not filename:
        return None
    target = Path(filename).name.lower()
    return next((e for e in LIBRARY if e["backend"] == "gguf" and e["name"] == model_name
                 and Path(e["model"]).name.lower() == target), None)


def _library_banner() -> str:
    lines = [LIBRARY_WARNING] if LIBRARY_WARNING else []
    for e in LIBRARY:
        state = LIBRARY_STATUS.get(e["model"])
        if state == "downloadable":
            lines.append(f"{e['name']} is not on this machine yet. Loading it restores it from its source.")
        elif state == "missing":
            lines.append(f"{e['name']} is missing ({e['model']}) and has no download source. "
                         f"Copy the file into {LLM_DIR}, or remove it in Add or test model.")
    return _badge("<br>".join(lines), "warning") if lines else ""
```
Trong `_build_model_choice_label`, thay khối
```python
        if variant_label:
            suffix = f" [{variant_label.split(' ', 1)[0]}]"   # labels start with the quant
```
bằng
```python
        if variant_label:
            quant, _, rest = variant_label.partition(" (")
            tested = f" · tested {_extract_vram_from_label(variant_label)} GB" if rest.startswith("tested") else ""
            suffix = f" [{quant}{tested}]"
```
Trong `_get_candidate_model_names`, ngay đầu thân vòng `for model_name in catalog:` thêm:
```python
        entry = _library_entry_for(model_name, vram_profile)
        if entry and LIBRARY_STATUS.get(entry["model"]) == "missing":
            candidates.append((f"❌ {model_name}", "❌", 999.0, model_name))
            continue
```
Trong `_render_model_info`, ngay trước `suitability = ...`, thêm:
```python
    entry = _library_entry_for(model_name, vram_profile)
    if entry:
        test = entry.get("test") or {}
        peaks = ", ".join(f"{'all profiles' if k == '*' else k}: {v} GB"
                          for k, v in (test.get("peak_vram_gb") or {}).items()) or "not measured"
        detail += (f"<br>Library: {LIBRARY_STATUS.get(entry['model'], '?')} · tested {test.get('date', 'never')}"
                   f" on {test.get('gpu', '?')} · peak VRAM {peaks}")
```

- [ ] **Step 6: Khôi phục model thiếu khi Load**

Trong `load_model`, ngay sau dòng `compatible = _is_model_compatible(...)`, thêm:
```python
        entry = _library_entry_for(model_name, vram_profile)
        if entry and LIBRARY_STATUS.get(entry["model"]) != "found":
            if LIBRARY_STATUS.get(entry["model"]) == "missing":
                yield _badge(f"{model_name} is missing on this machine: {entry['model']}", "error")
                return
            yield _badge(f"⏳ Restoring {model_name} into {LLM_DIR}…", "loading")
            model_library.restore(entry, LLM_DIR)
            LIBRARY_STATUS[entry["model"]] = "found"
            if entry["backend"] == "gguf":
                LOCAL_GGUF_FILES.add(Path(entry["model"]).name.lower())
            available = True
```

- [ ] **Step 7: Banner trong Setup tab**

Ngay dưới `with gr.Tab("⚙️ Setup"):` và trước `with gr.Row(equal_height=False):`, thêm:
```python
        library_banner = gr.HTML(_library_banner())
```

- [ ] **Step 8: Kiểm tra import và dropdown**

Run:
```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -c "import sys; sys.argv=['app.py']; import app; print(app.LIBRARY, app.LIBRARY_WARNING); [print(p, app._get_candidate_model_names('GGUF (llama-cpp)', p, app.LLM_DIR)) for p in app.VRAM_PROFILE_NAMES]" 2>&1 | grep -v httpx | tail -6
```
Expected: `[] None` (chưa có thư viện), rồi các danh sách GGUF giống trước Task 6 cho model catalog chính thức (4B/8B Instruct, NSFW V4.5 🟢 ở NormalVRAM). File GGUF V4 mới chưa hiện, vì không còn tự thêm.

- [ ] **Step 9: Chạy toàn bộ test**

Run:
```bash
.venv/Scripts/python.exe test_model_registry.py && .venv/Scripts/python.exe test_model_library.py && .venv/Scripts/python.exe test_comfy_convert.py && .venv/Scripts/python.exe test_briefs_prompts.py && .venv/Scripts/python.exe batch_caption.py --self-test
```
Expected: tất cả in OK.

- [ ] **Step 10: Commit**

```bash
git add app.py model_registry.py test_model_registry.py .gitignore
git commit -m "App reads models_library.json at startup, flags missing models, restores on load" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Quick test (`model_tester.py`)

**Files:**
- Create: `model_tester.py`
- Test: `test_model_tester.py`

**Interfaces:**
- Consumes: `model_library.abs_path`, `GGUF_LOAD_KWARGS`, `new_entry`, `record`, `suggest_name`, `load`, `save`, `upsert`, `LIBRARY_PATH`; `model_registry.inspect_path`, `DEFAULT_LLM_DIR`; `utils.system_info._gpu_info_from_nvidia_smi`; `GGUFCaptioner`, `HFCaptioner`; `captioner.comfy_convert.convert`; `VRAM_PROFILES[...]["budget_gb"]`.
- Produces:
  - `make_test_image(path) -> Path`, `vision_score(caption) -> int`, `looks_like_text(caption) -> bool`, `fitting_profiles(peak_gb) -> dict[str, bool]`, `VramSampler(read, interval)` (context manager, thuộc tính `peak`)
  - `run_quick_test(entry, root, vram_profile, log=logger.info) -> dict` — kết quả có `ok, reason, profile, date, gpu, load_s, caption_s, caption, vision_score, peak_vram_gb, fits`
  - `prepare_and_test(candidate, name, root, vram_profile, mmproj=None, base_repo=None, source_repo="", log=logger.info) -> tuple[dict, dict]` — `(entry đã ghi kết quả, result)`
  - CLI `model_tester.py PATH [--mmproj P] [--name N] [--profile Normal] [--source owner/repo] [--root R] [--add]`

- [ ] **Step 1: Viết test hỏng**

`test_model_tester.py`:
```python
"""Self-checks for model_tester's pure parts. Run: .venv\\Scripts\\python.exe test_model_tester.py"""
import tempfile
from pathlib import Path

from PIL import Image

from model_tester import VramSampler, fitting_profiles, looks_like_text, make_test_image, vision_score


def test_image_and_scoring():
    with tempfile.TemporaryDirectory() as tmp:
        img = Image.open(make_test_image(Path(tmp) / "s.png"))
        assert img.size == (512, 512)
        assert img.getpixel((5, 5)) == (40, 90, 200)        # blue background
        assert img.getpixel((150, 240)) == (220, 30, 30)    # red circle
        assert img.getpixel((370, 240)) == (245, 210, 20)   # yellow square
    assert vision_score("A red circle and a yellow square on a blue background.") == 5
    assert vision_score("A dog.") == 0
    assert looks_like_text("A red circle next to a yellow square.")
    assert not looks_like_text("")
    assert not looks_like_text("!!!! #### ???? 1234 ....")


def test_fitting_profiles():
    fits = fitting_profiles(6.4)
    assert fits == {"UltraLow (4GB)": False, "LowVRAM (6–8GB)": True,
                    "NormalVRAM (12–16GB)": True, "HighVRAM (20GB+)": True}, fits
    assert fitting_profiles(None) == {}


def test_sampler_keeps_peak():
    values = iter([1.0, 5.0, 3.0] + [2.0] * 1000)
    with VramSampler(read=lambda: next(values), interval=0.01) as s:
        import time
        time.sleep(0.1)
    assert s.peak == 5.0


if __name__ == "__main__":
    test_image_and_scoring()
    test_fitting_profiles()
    test_sampler_keeps_peak()
    print("test_model_tester OK")
```

- [ ] **Step 2: Chạy test để thấy hỏng**

Run: `.venv/Scripts/python.exe test_model_tester.py`
Expected: `ModuleNotFoundError: No module named 'model_tester'`

- [ ] **Step 3: Viết `model_tester.py`**

```python
"""
Quick test: does a model run with this program, and how much VRAM does it need?

Loads the model with the app's own captioner classes, captions a generated
image (red circle, yellow square, blue background), checks the caption names
what is in it, and samples nvidia-smi for peak VRAM.

CLI: .venv\\Scripts\\python.exe model_tester.py PATH [--mmproj P] [--name N]
     [--profile Normal] [--source owner/repo] [--root DIR] [--add]
"""
import argparse
import gc
import json
import logging
import sys
import tempfile
import threading
import time
from datetime import date
from pathlib import Path

from PIL import Image, ImageDraw

import model_library
from model_registry import DEFAULT_LLM_DIR, inspect_path
from models_catalog import GGUF_VL_MODELS, VRAM_PROFILES
from utils.system_info import _gpu_info_from_nvidia_smi

logger = logging.getLogger(__name__)

TEST_PROMPT = "Describe the shapes and colors in this image in one sentence."
VISION_WORDS = ("red", "circle", "yellow", "square", "blue")
PASS_VISION = 3


def make_test_image(path) -> Path:
    img = Image.new("RGB", (512, 512), (40, 90, 200))
    draw = ImageDraw.Draw(img)
    draw.ellipse((60, 150, 240, 330), fill=(220, 30, 30))
    draw.rectangle((290, 160, 450, 320), fill=(245, 210, 20))
    img.save(path)
    return Path(path)


def vision_score(caption: str) -> int:
    text = caption.lower()
    return sum(word in text for word in VISION_WORDS)


def looks_like_text(caption: str) -> bool:
    letters = sum(c.isalpha() for c in caption)
    return len(caption.strip()) >= 10 and letters / max(len(caption), 1) >= 0.6


def fitting_profiles(peak_gb) -> dict:
    if peak_gb is None:
        return {}
    return {name: peak_gb <= p["budget_gb"] for name, p in VRAM_PROFILES.items()}


def gpu_used_gb():
    gpus = _gpu_info_from_nvidia_smi()
    return gpus[0]["vram_used"] if gpus else None


def gpu_name() -> str:
    gpus = _gpu_info_from_nvidia_smi()
    return f"{gpus[0]['name']}, {round(gpus[0]['vram_total'])} GB" if gpus else "no NVIDIA GPU found"


class VramSampler:
    """Reads a value every `interval` seconds in a thread and keeps the highest one."""

    def __init__(self, read=gpu_used_gb, interval: float = 0.25):
        self.read, self.interval, self.peak = read, interval, None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while True:
            value = self.read()
            if value is not None:
                self.peak = value if self.peak is None else max(self.peak, value)
            if self._stop.wait(self.interval):
                break

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


def _is_oom(e: Exception) -> bool:
    text = str(e).lower()
    return "out of memory" in text or "did not fit" in text or "failed to create llama_context" in text


def run_quick_test(entry, root, vram_profile, log=logger.info) -> dict:
    from captioner.gguf_captioner import GGUFCaptioner
    from captioner.hf_captioner import HFCaptioner

    model = model_library.abs_path(entry["model"], root)
    result = {"ok": False, "reason": "", "profile": vram_profile, "date": date.today().isoformat(),
              "gpu": gpu_name(), "load_s": None, "caption_s": None, "caption": "",
              "vision_score": 0, "peak_vram_gb": None}
    baseline = gpu_used_gb()
    captioner = None
    with tempfile.TemporaryDirectory() as tmp, VramSampler() as sampler:
        image = make_test_image(Path(tmp) / "shapes.png")
        try:
            t0 = time.perf_counter()
            if entry["backend"] == "gguf":
                captioner = GGUFCaptioner()
                captioner.load_model(vram_profile, model_path=str(model),
                                     mmproj_path=str(model_library.abs_path(entry["mmproj"], root)),
                                     model_name=entry["name"], device="cuda", **model_library.GGUF_LOAD_KWARGS)
                if captioner.runtime_device == "cpu-fallback":
                    raise MemoryError(f"did not fit on the GPU ({captioner.fallback_reason})")
            else:
                captioner = HFCaptioner()
                captioner.load_model(vram_profile, model_id=str(model), device="cuda")
            result["load_s"] = round(time.perf_counter() - t0, 1)
            log(f"loaded in {result['load_s']} s, captioning the test image")
            t1 = time.perf_counter()
            caption = captioner.caption_image(str(image), TEST_PROMPT, max_new_tokens=60, temperature=0)
            result["caption_s"] = round(time.perf_counter() - t1, 1)
            result["caption"] = caption
            result["vision_score"] = vision_score(caption)
        except Exception as e:
            result["reason"] = f"out of memory: {e}" if _is_oom(e) else f"{type(e).__name__}: {e}"
        finally:
            if captioner is not None:
                try:
                    captioner.unload_model()
                except Exception as e:
                    log(f"unload failed: {e}")
            gc.collect()
    if baseline is not None and sampler.peak is not None:
        result["peak_vram_gb"] = round(max(sampler.peak - baseline, 0.0), 1)
    if not result["reason"]:
        if not looks_like_text(result["caption"]):
            result["reason"] = "the caption is empty or garbage, the mmproj may not match"
        elif result["vision_score"] < PASS_VISION:
            result["reason"] = f"vision check {result['vision_score']}/5, the model does not see the test image"
        else:
            result["ok"] = True
    result["fits"] = fitting_profiles(result["peak_vram_gb"])
    return result


def prepare_and_test(candidate, name, root, vram_profile, mmproj=None, base_repo=None, source_repo="",
                     log=logger.info):
    """Build the library entry, convert safetensors once if needed, run the quick test."""
    entry = model_library.new_entry(candidate, name, root, mmproj=mmproj, base_repo=base_repo,
                                    source_repo=source_repo)
    if entry["backend"] == "hf":
        from captioner.comfy_convert import convert
        log("converting to a HuggingFace folder (one time, a few minutes)")
        convert(candidate["path"], model_library.abs_path(entry["model"], root), entry["base_repo"], log=log)
    result = run_quick_test(entry, root, vram_profile, log=log)
    return model_library.record(entry, result), result


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass
    p = argparse.ArgumentParser(description="Quick-test a Qwen3-VL model and optionally add it to the library.")
    p.add_argument("path")
    p.add_argument("--mmproj", default=None)
    p.add_argument("--name", default=None)
    p.add_argument("--profile", default="Normal", help="profile name or its start, e.g. Low, Normal")
    p.add_argument("--source", default="", help="owner/repo to restore the files from on another machine")
    p.add_argument("--root", default=str(DEFAULT_LLM_DIR))
    p.add_argument("--add", action="store_true", help="add to models_library.json when the test passes")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    root = Path(a.root)
    profile = next((n for n in VRAM_PROFILES if n.lower().startswith(a.profile.lower())), None)
    if profile is None:
        print(f"unknown profile {a.profile!r}; use one of {list(VRAM_PROFILES)}")
        return 2
    candidate = inspect_path(a.path, root)
    print(f"check: {candidate['reason']}")
    if not candidate["usable"]:
        return 1
    name = a.name or model_library.suggest_name(candidate["path"], candidate["kind"], GGUF_VL_MODELS)
    entry, result = prepare_and_test(candidate, name, root, profile, mmproj=a.mmproj, source_repo=a.source, log=print)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["ok"] and a.add:
        entries, warning = model_library.load()
        if warning:
            print(warning)
        model_library.save(model_library.upsert(entries, entry))
        print(f"added {name} to {model_library.LIBRARY_PATH.name}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Chạy test để thấy qua**

Run: `.venv/Scripts/python.exe test_model_tester.py`
Expected: `test_model_tester OK`

- [ ] **Step 5: Chạy thật một GGUF, không thêm vào thư viện**

Run:
```bash
.venv/Scripts/python.exe model_tester.py "D:/001_Personal_Proj/Comfy/ComfyUI/models/text_encoders/GGUF/Qwen3-VL-4B-Instruct-GGUF/Qwen3VL-4B-Instruct-Q4_K_M.gguf" --profile Low
```
Expected: `check: GGUF, 2560-dim mmproj found`, JSON có `"ok": true`, `vision_score` ≥ 3, `peak_vram_gb` khoảng 3–5, exit code 0. `models_library.json` chưa được tạo.

- [ ] **Step 6: Chạy thật bản HF đã chuyển đổi ở Task 3**

Run:
```bash
.venv/Scripts/python.exe model_tester.py "D:/001_Personal_Proj/Comfy/ComfyUI/models/text_encoders/qwen3-vl-4b-heretic_fp8_e4m3fn.safetensors" --profile Low
```
Expected: `qwen3-vl-4b-heretic-HF is already converted`, JSON `"ok": true`, `peak_vram_gb` khoảng 3–4.

- [ ] **Step 7: Commit**

```bash
git add model_tester.py test_model_tester.py
git commit -m "Add model_tester: quick GPU test with a synthetic image and peak VRAM sampling" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Thư viện ban đầu từ 4 GGUF đang chạy tốt

**Files:**
- Create: `models_library.json` (do CLI ghi)

**Interfaces:**
- Consumes: CLI của Task 7.
- Produces: `models_library.json` có 4 entry GGUF đã test, commit vào git.

- [ ] **Step 1: Test và thêm từng model**

Run lần lượt (mỗi lệnh khoảng 30 giây):
```bash
.venv/Scripts/python.exe model_tester.py "D:/001_Personal_Proj/Comfy/ComfyUI/models/text_encoders/GGUF/Qwen3-VL-4B-Instruct-GGUF/Qwen3VL-4B-Instruct-Q4_K_M.gguf" --source Qwen/Qwen3-VL-4B-Instruct-GGUF --add
.venv/Scripts/python.exe model_tester.py "D:/001_Personal_Proj/Comfy/ComfyUI/models/text_encoders/GGUF/Qwen3VL-4B-Instruct-Q8_0.gguf" --source Qwen/Qwen3-VL-4B-Instruct-GGUF --add
.venv/Scripts/python.exe model_tester.py "D:/001_Personal_Proj/Comfy/ComfyUI/models/text_encoders/GGUF/Qwen3-VL-8B-Instruct-GGUF/Qwen3VL-8B-Instruct-Q4_K_M.gguf" --source Qwen/Qwen3-VL-8B-Instruct-GGUF --add
.venv/Scripts/python.exe model_tester.py "D:/001_Personal_Proj/Comfy/ComfyUI/models/text_encoders/GGUF/Qwen3-VL-8B-NSFW-Caption-V4.5.Q6_K.gguf" --source mradermacher/Qwen3-VL-8B-NSFW-Caption-V4.5-GGUF --add
```
Expected: mỗi lệnh kết thúc bằng `added <tên> to models_library.json`. Tên lấy theo catalog: `Qwen3-VL-4B-Instruct-GGUF` (hai entry, Q4_K_M và Q8_0), `Qwen3-VL-8B-Instruct-GGUF`, `Qwen3-VL-8B-NSFW-Caption-V4.5-GGUF`. Nếu một lệnh không đạt, dừng lại và báo người dùng kết quả JSON.

- [ ] **Step 2: Kiểm tra nội dung và tính di động**

Run:
```bash
.venv/Scripts/python.exe -c "import json; d=json.load(open('models_library.json',encoding='utf-8')); [print(e['name'], e['model'], e['mmproj'], e['test']['peak_vram_gb']) for e in d['models']]"
```
Expected: 4 dòng, mọi đường dẫn `model` và `mmproj` là tương đối, bắt đầu bằng `GGUF/`, và trỏ tới file có thật. Mỗi dòng có `{'*': <số GB>}`.

- [ ] **Step 3: Commit**

```bash
git add models_library.json
git commit -m "Seed models_library.json with the four GGUF models that pass the quick test" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Dialog "Add or test model" trong Setup tab

**Files:**
- Modify: `app.py` (import; handler mới đặt sau `unload_model`; Setup tab; wiring)

**Interfaces:**
- Consumes: Task 6 (`LIBRARY`, `LIBRARY_STATUS`, `OFFICIAL_MODEL_NAMES`, `LOCAL_GGUF_FILES`, `_refresh_model_selector`, `_badge`); Task 4 `model_registry.list_candidates`, `inspect_path`; Task 5 `model_library.*`; Task 7 `model_tester.prepare_and_test`.
- Produces: handler `md_open`, `md_close`, `md_check`, `md_test`, `md_add`, `md_remove`; các component `md_*` trong Setup tab.

- [ ] **Step 1: Import**

Thêm cạnh các import của app:
```python
import html
import model_tester
```

- [ ] **Step 2: Handler của dialog**

Thêm ngay sau hàm `unload_model()`:
```python
# ── Model library dialog ─────────────────────────────────────────────────────
def _md_candidate_label(c: dict) -> str:
    return f"{'●' if c['usable'] else '✕'} {c['name']}  ·  {c['reason']}"


def _md_library_html() -> str:
    if not LIBRARY:
        return _badge("Your library is empty. Pick a model above and run the quick test.", "info")
    colors = {"found": "#4ade80", "downloadable": "#fbbf24", "missing": "#f87171"}
    rows = []
    for e in LIBRARY:
        state = LIBRARY_STATUS.get(e["model"]) or model_library.status(e, LLM_DIR)
        peaks = (e.get("test") or {}).get("peak_vram_gb") or {}
        vram = ", ".join(f"{'all profiles' if k == '*' else k} {v} GB" for k, v in peaks.items()) or "not measured"
        rows.append(f'<div style="display:flex;justify-content:space-between;padding:3px 0;">'
                    f'<span>{html.escape(e["name"])} · {e["quant"]} · {vram}</span>'
                    f'<span style="color:{colors.get(state, "#94a3b8")}">{state}</span></div>')
    return '<div style="font-size:13px;color:#cbd5e1;">' + "".join(rows) + "</div>"


def _md_library_choices() -> list:
    return [(f"{e['name']} · {e['quant']}", e["model"]) for e in LIBRARY]


def md_open():
    cands = model_registry.list_candidates(LLM_DIR, model_library.known_paths(LIBRARY, LLM_DIR))
    return (gr.update(visible=True), cands,
            gr.update(choices=[_md_candidate_label(c) for c in cands], value=None),
            _md_library_html(), gr.update(choices=_md_library_choices(), value=None))


def md_close():
    return gr.update(visible=False)


def md_check(choice, cands, pasted):
    """Static check of the picked candidate or the pasted path; fills mmproj, base repo and name."""
    hidden = gr.update(visible=False)
    if pasted and pasted.strip():
        c = model_registry.inspect_path(pasted.strip().strip('"'), LLM_DIR)
    else:
        labels = [_md_candidate_label(x) for x in cands or []]
        c = cands[labels.index(choice)] if choice in labels else None
    if c is None:
        return (None, _badge("Pick a model above or paste a file path.", "warning"),
                gr.update(choices=[], value=None, visible=False), hidden, gr.update(value=""),
                gr.update(interactive=False))
    name = model_library.suggest_name(c["path"], c["kind"], GGUF_VL_MODELS) if c["usable"] else ""
    return (c, _badge(html.escape(c["reason"]), "success" if c["usable"] else "error"),
            gr.update(choices=c.get("mmproj_options", []), value=c.get("mmproj") or None, visible=c["kind"] == "gguf"),
            gr.update(value=c.get("base_repo", ""), visible=c["kind"] == "safetensors"),
            gr.update(value=name), gr.update(interactive=False))


def _md_result_html(r: dict) -> str:
    head = _badge("Quick test passed." if r["ok"] else f"Quick test failed: {html.escape(r['reason'])}",
                  "success" if r["ok"] else "error")
    chips = " ".join(
        f'<span style="padding:2px 8px;border-radius:6px;background:{"#0a1f10" if ok else "#2a0a0a"};'
        f'color:{"#4ade80" if ok else "#f87171"};">{p} {"✓" if ok else "✕"}</span>'
        for p, ok in r["fits"].items())
    peak = f"{r['peak_vram_gb']} GB" if r["peak_vram_gb"] is not None else "not measured"
    return (head + '<div style="font-size:13px;color:#cbd5e1;line-height:1.7;margin-top:6px;">'
            f'Load {r["load_s"]} s · caption {r["caption_s"]} s · peak VRAM {peak} · vision {r["vision_score"]}/5<br>'
            f'<i>{html.escape(r["caption"][:240])}</i><div style="margin-top:6px;">{chips}</div></div>')


def md_test(current, name, mmproj, base_repo, source_repo, profile):
    """Unload the current model, convert if needed, run the quick test. Yields progress."""
    global _captioner
    off = gr.update(interactive=False)
    if not current or not current.get("usable"):
        yield None, _badge("Run Check on a usable model first.", "warning"), off, gr.update()
        return
    if not (name or "").strip():
        yield None, _badge("Enter a name for the list.", "warning"), off, gr.update()
        return
    if _captioner is not None:
        _captioner.unload_model()
        _captioner = None
    note = " Safetensors are converted first, which takes a few minutes." if current["kind"] == "safetensors" else ""
    yield None, _badge(f"⏳ Testing on the GPU. The loaded model was unloaded.{note}", "loading"), off, \
        _badge("No model loaded.", "info")
    try:
        entry, result = model_tester.prepare_and_test(
            current, name.strip(), LLM_DIR, profile, mmproj=mmproj or None,
            base_repo=(base_repo or "").strip() or None, source_repo=(source_repo or "").strip())
    except Exception as e:
        yield None, _badge(f"Test failed: {html.escape(str(e))}", "error"), off, gr.update()
        return
    yield (entry if result["ok"] else None), _md_result_html(result), gr.update(interactive=result["ok"]), gr.update()


def md_add(entry, backend, profile, current_choice):
    global LIBRARY
    if not entry:
        return (_badge("Run a passing quick test first.", "warning"), gr.update(), gr.update(),
                _md_library_html(), gr.update(), gr.update(interactive=False))
    LIBRARY = model_library.upsert(LIBRARY, entry)
    model_library.save(LIBRARY)
    LIBRARY_STATUS.update(model_library.apply_to_catalogs([entry], LLM_DIR, GGUF_VL_MODELS, HF_VL_MODELS))
    if entry["backend"] == "gguf":
        LOCAL_GGUF_FILES.add(Path(entry["model"]).name.lower())
    dd, info, _ = _refresh_model_selector(backend, profile, current_choice)
    return (_badge(f"Added {html.escape(entry['name'])} to {model_library.LIBRARY_PATH.name}.", "success"),
            dd, info, _md_library_html(), gr.update(choices=_md_library_choices(), value=None),
            gr.update(interactive=False))


def md_remove(model_key, backend, profile, current_choice):
    global LIBRARY
    entry = model_library.find(LIBRARY, model_key)
    if entry is None:
        return _badge("Pick a library entry to remove.", "warning"), gr.update(), gr.update(), _md_library_html(), gr.update()
    LIBRARY = model_library.remove(LIBRARY, model_key)
    model_library.save(LIBRARY)
    model_library.unregister(entry, LLM_DIR, GGUF_VL_MODELS, HF_VL_MODELS, OFFICIAL_MODEL_NAMES)
    LIBRARY_STATUS.pop(model_key, None)
    dd, info, _ = _refresh_model_selector(backend, profile, current_choice)
    return (_badge(f"Removed {html.escape(entry['name'])} from the list. Files stay on disk.", "success"),
            dd, info, _md_library_html(), gr.update(choices=_md_library_choices(), value=None))
```

- [ ] **Step 3: Nút mở dialog**

Trong Setup tab, ngay dưới khối `model_info = gr.HTML(...)` (vẫn trong `gr.Group` "Select Model"), thêm:
```python
                    md_open_btn = gr.Button("➕ Add or test model", size="sm")
```

- [ ] **Step 4: Panel dialog**

Trong Setup tab, ngay trước `with gr.Row():` chứa `load_btn`, thêm:
```python
        with gr.Group(visible=False) as md_panel:
            gr.Markdown("### Add a model to your library")
            md_cands = gr.State([])
            md_current = gr.State(None)
            md_entry = gr.State(None)
            md_cand_radio = gr.Radio(label=f"Found in {LLM_DIR.name}, not in your library yet", choices=[])
            with gr.Row():
                md_path = gr.Textbox(label="Or paste a file path", placeholder=r"E:\models\my-qwen3vl.gguf", scale=4)
                md_check_btn = gr.Button("🔍 Check", size="sm", scale=1)
            md_check_html = gr.HTML()
            with gr.Row():
                md_mmproj = gr.Dropdown(label="mmproj (matching projection size)", choices=[], visible=False)
                md_base_repo = gr.Textbox(label="Base repo for tokenizer and processor", visible=False)
            with gr.Row():
                md_name = gr.Textbox(label="Name in list")
                md_source = gr.Textbox(label="Download source (optional)", placeholder="owner/repo")
                md_profile = gr.Dropdown(label="Profile to test", choices=VRAM_PROFILE_NAMES, value=initial_profile)
            md_test_btn = gr.Button("🧪 Quick test on GPU")
            md_result = gr.HTML()
            with gr.Row():
                md_add_btn = gr.Button("✅ Add to library", variant="primary", interactive=False)
                md_close_btn = gr.Button("Close")
            md_status = gr.HTML()
            gr.Markdown(f"#### Your library ({model_library.LIBRARY_PATH.name})")
            md_lib_html = gr.HTML()
            with gr.Row():
                md_remove_dd = gr.Dropdown(label="Remove from list (files stay on disk)", choices=[], scale=4)
                md_remove_btn = gr.Button("🗑️ Remove", size="sm", scale=1)
```

- [ ] **Step 5: Nối sự kiện**

Ngay sau dòng `unload_btn.click(unload_model, outputs=[model_status])`, thêm:
```python
        md_open_btn.click(md_open, None, [md_panel, md_cands, md_cand_radio, md_lib_html, md_remove_dd], queue=False)
        md_close_btn.click(md_close, None, [md_panel], queue=False)
        md_check_outputs = [md_current, md_check_html, md_mmproj, md_base_repo, md_name, md_add_btn]
        md_cand_radio.change(md_check, [md_cand_radio, md_cands, md_path], md_check_outputs, queue=False)
        md_check_btn.click(md_check, [md_cand_radio, md_cands, md_path], md_check_outputs, queue=False)
        md_test_btn.click(md_test, [md_current, md_name, md_mmproj, md_base_repo, md_source, md_profile],
                          [md_entry, md_result, md_add_btn, model_status])
        md_add_btn.click(md_add, [md_entry, backend_radio, vram_radio, model_dd],
                         [md_status, model_dd, model_info, md_lib_html, md_remove_dd, md_add_btn])
        md_remove_btn.click(md_remove, [md_remove_dd, backend_radio, vram_radio, model_dd],
                            [md_status, model_dd, model_info, md_lib_html, md_remove_dd])
        vram_radio.change(lambda p: gr.update(value=p), [vram_radio], [md_profile], queue=False)
```

- [ ] **Step 6: Kiểm tra import và chạy test**

Run:
```bash
.venv/Scripts/python.exe -c "import sys; sys.argv=['app.py']; import app; print(len(app.LIBRARY), 'library entries')" 2>&1 | tail -1
.venv/Scripts/python.exe test_model_library.py && .venv/Scripts/python.exe test_model_tester.py && .venv/Scripts/python.exe test_model_registry.py
```
Expected: `4 library entries`, rồi ba dòng OK.

- [ ] **Step 7: Commit**

```bash
git add app.py
git commit -m "Setup tab: Add or test model dialog with check, GPU quick test and library list" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Kiểm chứng trong browser, tài liệu

**Files:**
- Modify: `.claude/launch.json` (bỏ `--llm-dir llm`)
- Modify: `README.md` (mục mới "Model library")

**Interfaces:**
- Consumes: mọi task trước.
- Produces: bằng chứng chạy thật, README cập nhật.

- [ ] **Step 1: Sửa cấu hình preview**

`.claude/launch.json`: đổi `"runtimeArgs": ["app.py", "--llm-dir", "llm"]` thành `"runtimeArgs": ["app.py"]`.

- [ ] **Step 2: Mở app và kiểm tra Setup tab**

Dùng `preview_start` với name `captioner`, đợi log có `Running on local URL`. Kiểm tra bằng `read_page`: không có banner cảnh báo; dropdown GGUF có `[Q4_K_M · tested ... GB]`; có nút `Add or test model`. `read_console_messages` không có lỗi.

- [ ] **Step 3: Thêm GGUF V4 mới qua dialog**

Bấm `Add or test model`. Trong danh sách ứng viên phải có `Qwen3-VL-8B-NSFW-Caption-V4.Q4_K_M.gguf` (●), ba file safetensors 4B (●), hai file 32B (✕, "50 of 64 layers"). Chọn file V4, kiểm tra mmproj tự chọn là `...V4.5.mmproj-Q8_0.gguf`, tên gợi ý `Qwen3-VL-8B-NSFW-Caption-V4-GGUF`, bấm `Quick test on GPU`, đợi kết quả. Nếu đạt, bấm `Add to library`; dropdown phải có model mới. Nếu không đạt (mmproj V4.5 không khớp chất lượng), ghi lại kết quả và báo người dùng, không ép thêm.

- [ ] **Step 4: Thêm hai safetensors 4B qua dialog**

Lặp lại Step 3 cho `Huihui-Qwen3-VL-4B-Instruct-abliterated-fp8_scaled` (chuyển đổi mới) và `qwen3-vl-4b-heretic_fp8_e4m3fn` (đã chuyển đổi ở Task 3, phải báo "already converted"), profile LowVRAM. Sau khi thêm, chuyển backend sang HuggingFace: hai model hiện 🟢 trong dropdown. Bấm `Load Model` với một model, sang Single Image tab caption một ảnh thật, xác nhận có caption.

- [ ] **Step 5: Khởi động lại và kiểm tra thư viện còn nguyên**

`preview_stop`, rồi `preview_start` lại. Setup tab vẫn có các model vừa thêm (cả GGUF và HF), mở dialog thấy chúng trong "Your library" với trạng thái `found`.

- [ ] **Step 6: Mô phỏng máy khác thiếu model**

Đổi tên tạm `GGUF\Qwen3-VL-8B-NSFW-Caption-V4.Q4_K_M.gguf` thành `.gguf.off`, khởi động lại app. Banner vàng phải nêu model này thiếu và không có nguồn; dropdown hiện `❌ Qwen3-VL-8B-NSFW-Caption-V4-GGUF`; bấm Load báo `missing on this machine`. Đổi tên file lại như cũ, khởi động lại, banner biến mất.

- [ ] **Step 7: README**

Thêm vào `README.md`, sau phần hướng dẫn `--llm-dir`:
```markdown
## Model library

The model dropdown lists the official catalog plus `models_library.json`, the models that
passed a quick test on this machine. Paths in it are relative to `--llm-dir`, so the file
can be committed and reused on another PC.

- **Add a model:** Setup tab → **Add or test model**. Pick a file found under `--llm-dir`
  or paste a path, press **Check**, then **Quick test on GPU**. The test loads the model with
  the app's own backend, captions a generated test image and measures peak VRAM. Only a
  passing model can be added.
- **Safetensors:** ComfyUI single-file Qwen3-VL checkpoints (fp8) are converted once into
  `--llm-dir\HuggingFace\<name>` (bf16, about twice the file size) and then run on the
  HuggingFace backend. nvfp4 files and cut text encoders (for example 32B with 50 of 64
  layers) cannot caption and are rejected.
- **From the command line:**
  `.venv\Scripts\python.exe model_tester.py PATH --profile Low --add`
- **Missing models:** at startup every library entry is checked. A missing one shows a
  warning; it is restored on load when it has a download source or its original
  safetensors file is still there.

The HuggingFace backend needs the CUDA build of torch and torchvision (see `requirements.txt`).
```

- [ ] **Step 8: Chạy toàn bộ test lần cuối**

Run:
```bash
.venv/Scripts/python.exe test_model_registry.py && .venv/Scripts/python.exe test_model_library.py && .venv/Scripts/python.exe test_model_tester.py && .venv/Scripts/python.exe test_comfy_convert.py && .venv/Scripts/python.exe test_briefs_prompts.py && .venv/Scripts/python.exe batch_caption.py --self-test
```
Expected: tất cả OK.

- [ ] **Step 9: Commit**

```bash
git add .claude/launch.json README.md models_library.json
git commit -m "Document the model library; add the models tested through the dialog" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
