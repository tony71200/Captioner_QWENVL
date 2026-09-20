# QwenVL Image Captioner

A local image captioning tool powered by **Qwen3-VL** and **Qwen2.5-VL** vision-language models.  
Supports both a **Gradio Web UI** and a **standalone CLI** — no cloud, no API key required.

- **Backends**: HuggingFace Transformers · GGUF via `llama-cpp-python`
- **Devices**: CPU (no GPU required) · CUDA GPU
- **VRAM**: tự chọn model theo VRAM trống thật — từ 4 GB tới 24 GB+
- **Modes**: Single image · Batch folder processing

---

## Table of Contents

1. [Features](#features)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Usage — Web UI](#usage--web-ui)
5. [Usage — Standalone CLI](#usage--standalone-cli)
6. [Device Selection Guide](#device-selection-guide)
7. [Model Catalog](#model-catalog)
8. [Prompt Templates](#prompt-templates)
9. [Project Structure](#project-structure)
10. [Troubleshooting](#troubleshooting)

---

## Features

| Feature | Detail |
|---------|--------|
| 🖥️ CPU inference | GGUF: `n_gpu_layers=0`, HF: `device_map="cpu"` |
| ⚡ GPU inference | GGUF: profile-based GPU layers, HF: BitsAndBytes 4/8-bit |
| 🧠 GGUF backend | `llama-cpp-python` — low memory, runs on CPU or CUDA |
| 🤗 HF backend | HuggingFace Transformers — 4-bit / 8-bit / BF16 |
| 📊 System info panel | Live CPU/GPU specs displayed in UI |
| 🖼️ Single image mode | Upload → Caption → Save |
| 📁 Batch mode | Process entire folder, skip existing, recursive scan |
| 🔧 7 prompt templates | Short, Detailed, Booru Tags, Structured, SD Training, Scene Analysis, Custom |
| 💻 Standalone CLI | `test_caption.py` — no WebUI dependency |
| 📦 Auto model download | Downloads from HuggingFace Hub on first load |

---

## Requirements

- **Python** 3.10+
- **OS**: Windows (primary), Linux (should work)
- **GPU (optional)**: CUDA-capable GPU for GPU mode
- **RAM**: 8 GB+ recommended for CPU mode with 4B model

### Python packages

```
gradio>=4.0
Pillow
transformers>=4.45.0
accelerate>=0.26.0
bitsandbytes>=0.43.0      # GPU only — HF 4bit/8bit quant
qwen-vl-utils>=0.0.8      # HF backend
llama-cpp-python>=0.3.0   # GGUF backend
psutil                     # System info panel (optional but recommended)
huggingface_hub            # Model download
```

---

## Installation

### Option A — Quick setup (Windows)

```bat
setup.bat
```

`setup.bat` creates a `.venv`, installs dependencies, and configures the environment.

### Option B — Manual

```bash
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# 2. Install PyTorch (with CUDA 12.x)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 3. Install GGUF backend (choose one)
# CPU-only build:
pip install llama-cpp-python

# CUDA build (faster):
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

# 4. Install remaining packages
pip install -r requirements.txt

# 5. Install psutil for system info (recommended)
pip install psutil
```

---

## Usage — Web UI

### Launch

```bat
run_independent_env.bat
```

Or manually:

```bash
.venv\Scripts\activate
python app.py --llm-dir "D:\Comfy\ComfyUI\models\llm"
```

Open browser at **http://localhost:7860**

### Setup Tab Walkthrough

1. **Inference Backend** — Choose `GGUF (llama-cpp)` (recommended) or `HuggingFace (Transformers)`
2. **Device Selection** — Choose where to run:
   - `Auto` — use GPU if available, fallback to CPU (shows both CPU + GPU specs)
   - `CPU` — force CPU mode (shows CPU name, cores, RAM usage)
   - `GPU` — force CUDA (shows GPU name, VRAM total/free with usage bar)
3. **Select Model** — danh sách cấu hình đã xếp hạng theo ngân sách VRAM
   (🟢 dưới 80% ngân sách · 🟡 vừa khít · 🔴 vượt · ⬇️ chưa có trên máy)
4. **Tự hạ cấp khi thiếu VRAM** — bật thì app tự đổi sang cấu hình nhỏ hơn thay vì báo lỗi
5. **🚀 Load Model** — đọc lại VRAM trống, chạy preflight, tải nếu cần rồi nạp

---

## Usage — Standalone CLI

`test_caption.py` runs without any WebUI — input is an image path, output is a caption string.

### Basic

```bash
# Auto-detect device and model
python test_caption.py --image photo.jpg

# Force CPU
python test_caption.py --image photo.jpg --device cpu

# Force GPU
python test_caption.py --image photo.jpg --device cuda
```

### Specify model explicitly

```bash
python test_caption.py \
  --image photo.jpg \
  --model-path D:\models\llm\Qwen3VL-4B-Instruct-Q4_K_M.gguf \
  --mmproj-path D:\models\llm\mmproj-Qwen3VL-4B-Instruct-F16.gguf \
  --device cpu
```

### Change prompt template

```bash
# Use a named template (default: "Detailed Description")
python test_caption.py --image photo.jpg --prompt "Booru Tags"
python test_caption.py --image photo.jpg --prompt "Short Caption"
python test_caption.py --image photo.jpg --prompt "Training Caption (SD/Flux)"

# Use raw prompt text
python test_caption.py --image photo.jpg --prompt "List every object visible in this image."
```

### Save output to file

```bash
python test_caption.py --image photo.jpg --output caption.txt
```

### HuggingFace backend

```bash
python test_caption.py \
  --image photo.jpg \
  --backend hf \
  --model-id Qwen/Qwen2.5-VL-3B-Instruct \
  --device cpu
```

### All options

```
--image PATH              Input image file (required)
--backend {gguf,hf}       Inference backend (default: gguf)
--device {auto,cpu,cuda}  Device (default: auto)
--model-path PATH         GGUF model file path (auto-detected if omitted)
--mmproj-path PATH        GGUF mmproj file path (auto-detected if omitted)
--model-id ID             HuggingFace model ID (HF backend only)
--llm-dir PATH            Directory to scan for GGUF files
--quant Q                 GGUF: Q4_K_M|Q8_0|F16 · HF: bf16|8bit|4bit (mặc định: tự suy)
--n-ctx N                 Độ dài context (mặc định 0 = tự suy từ VRAM trống)
--prompt TEMPLATE_OR_TEXT Prompt template name or raw text (default: Detailed Description)
--max-tokens N            Max tokens to generate (default: 512)
--output PATH             Save caption to file (optional)
--quiet                   Suppress logs, print caption only
```

---

## Device Selection Guide

### CPU Mode

| Backend | Behavior |
|---------|----------|
| GGUF | `n_gpu_layers=0` — all inference on CPU. `n_threads=os.cpu_count()` |
| HF | `device_map="cpu"`, `torch_dtype=float32`. BnB quantization disabled (not supported on CPU) |

**Recommended for CPU**: GGUF backend with Q4_K_M quantized 4B model.  
Expected speed: ~1–5 tokens/sec depending on CPU cores and model size.

> ⚠️ HF backend on CPU loads float32 weights — expect **high RAM usage** (8–16 GB for a 4B model).  
> Use GGUF for CPU inference whenever possible.

### GPU Mode

| Backend | Behavior |
|---------|----------|
| GGUF | `n_gpu_layers` do `utils/vram_plan` tính (−1 = toàn bộ layer trên GPU) |
| HF | `device_map="auto"`, BitsAndBytes 4-bit/8-bit quantization |

**System info panel** shows live VRAM usage (used/free/total with color-coded bar).

### Auto Mode

Detects CUDA availability at runtime:
- CUDA available → GPU
- No CUDA → CPU fallback

---

## Model Catalog

### HuggingFace Models

| Model | Size | Trọng số (GiB) | Quant |
|-------|------|----------------|-------|
| Qwen3-VL-2B-Instruct | 2B | 3.97 | bf16 / 8bit / 4bit |
| Qwen3-VL-2B-Instruct-FP8 | 2B | 3.23 | fp8 |
| Qwen3-VL-4B-Instruct | 4B | 8.27 | bf16 / 8bit / 4bit |
| Qwen3-VL-4B-Instruct-FP8 | 4B | 5.61 | fp8 |
| Qwen3-VL-8B-Instruct | 8B | 16.33 | bf16 / 8bit / 4bit |
| Qwen3-VL-8B-Instruct-FP8 | 8B | 9.86 | fp8 |
| Qwen2.5-VL-3B-Instruct | 3B | 6.99 | bf16 / 8bit / 4bit |
| Qwen2.5-VL-3B-Instruct-AWQ | 3B | 3.17 | awq |
| Qwen2.5-VL-7B-Instruct | 7B | 15.44 | bf16 / 8bit / 4bit |
| Qwen2.5-VL-7B-Instruct-AWQ | 7B | 6.44 | awq |

Các bản `-Thinking` có cùng dung lượng với bản `-Instruct` tương ứng.
Backend HF chạy được **cả Qwen3-VL lẫn Qwen2.5-VL** qua `AutoModelForImageTextToText`.

### GGUF Models

| Repo | Q4_K_M | Q8_0 | F16 | mmproj Q8_0 | mmproj F16 |
|------|--------|------|-----|-------------|------------|
| Qwen3-VL-2B-Instruct-GGUF | 1.03 | 1.70 | 3.21 | 0.42 | 0.76 |
| Qwen3-VL-2B-Thinking-GGUF | 1.03 | 1.70 | 3.21 | 0.42 | 0.76 |
| Qwen3-VL-4B-Instruct-GGUF | 2.33 | 3.99 | 7.50 | 0.42 | 0.78 |
| Qwen3-VL-4B-Thinking-GGUF | 2.33 | 3.99 | 7.50 | 0.42 | 0.78 |
| Qwen3-VL-8B-Instruct-GGUF | 4.68 | 8.11 | 15.26 | 0.70 | 1.08 |
| Qwen3-VL-8B-Thinking-GGUF | 4.68 | 8.11 | 15.26 | 0.70 | 1.08 |

Mọi con số tính bằng **GiB**, lấy từ HuggingFace API và verify ngày **2026-09-20**.
Catalog chỉ nhận repo của org [`Qwen`](https://huggingface.co/Qwen).

**Model bị loại**: 30B-A3B và 32B GGUF dùng file split 2 phần
(`-split-00001-of-00002.gguf`) nên `hf_hub_download` một filename không tải được;
32B bản HF nặng 62.1 GiB. Muốn thêm thì phải hỗ trợ file GGUF split trước.

### Ngân sách VRAM

Không còn profile cố định. App đọc VRAM **trống thật** từ driver
(`torch.cuda.mem_get_info`), trừ headroom, rồi ước lượng và xếp hạng từng cấu hình:

```
ngân sách = VRAM trống × 0.90 − 0.8 GiB
GGUF      = file × (gpu_layers/tổng layer) + mmproj + KV cache + overhead
HF        = trọng số × hệ số quant + KV cache + activation + overhead
```

Các hằng số hiệu chỉnh nằm ở `utils/vram_plan.py` và `utils/hardware.py`.

---

## Prompt Templates

| Template | Description |
|----------|-------------|
| **Detailed Description** *(default)* | Full scene description: subjects, colors, composition, mood |
| Short Caption | 1–2 sentence summary |
| Booru Tags | Comma-separated tags (Danbooru style) |
| Structured Caption | Labeled fields: Subject / Appearance / Action / Setting / Lighting / Style |
| Training Caption (SD/Flux) | Prose optimized for LoRA/fine-tune datasets |
| Object & Scene Analysis | Numbered list: objects, scene type, spatial relations, colors, text |
| Custom | Enter any prompt in the textbox |

---

## Project Structure

```
caption_QWENVL/
├── app.py                    # Gradio Web UI (main entry point)
├── test_caption.py           # Standalone CLI test (no WebUI)
├── models_catalog.py         # Dữ liệu model đã verify (GiB), không chứa logic
├── requirements.txt
├── setup.bat                 # One-click setup (Windows)
├── run.bat                   # Launch WebUI
├── run_independent_env.bat   # Launch with isolated .venv
│
├── captioner/
│   ├── base.py               # BaseCaptioner abstract class
│   ├── gguf_captioner.py     # GGUF backend (llama-cpp-python)
│   ├── hf_captioner.py       # HuggingFace backend (Transformers)
│   └── prompts.py            # Prompt templates
│
└── utils/
    ├── image_utils.py        # Image resizing, scanning
    ├── file_utils.py         # Caption file I/O
    └── system_info.py        # CPU/GPU hardware info for UI
```

---

## Troubleshooting

### CUDA out of memory
- Bật **Tự hạ cấp khi thiếu VRAM**, hoặc bấm **🔄 Làm mới ngân sách** rồi chọn dòng 🟢
- Dùng model **2B** hoặc **4B Q4_K_M**
- Unload model before loading a new one

### CPU inference is very slow
- GGUF Q4_K_M 4B is the fastest CPU option (~1–5 tok/s)
- Avoid HF backend on CPU (float32 is much slower than GGUF quantized)
- Set `--max-tokens 256` for faster results

### bitsandbytes error (HF backend)
```bash
pip install bitsandbytes --upgrade
# or on Windows:
pip install bitsandbytes-windows
```

### GGUF handler import error
The app tries these handlers in order:
1. `Qwen3VLChatHandler`
2. `Qwen25VLChatHandler`
3. `Llava15ChatHandler`

If all fail, upgrade `llama-cpp-python`:
```bash
pip install llama-cpp-python --upgrade
```

### HF Qwen3-VL load error
The HF backend is wired to the **Qwen2-VL API** — use **GGUF backend** for Qwen3-VL models.

### Model not found / auto-detection fails
Pass paths explicitly:
```bash
python test_caption.py --image photo.jpg \
  --model-path path/to/model.gguf \
  --mmproj-path path/to/mmproj.gguf
```

### psutil not installed (no system info in UI)
```bash
pip install psutil
```

---

## References

- [ComfyUI-QwenVL](https://github.com/1038lab/ComfyUI-QwenVL) — reference for CPU/GPU GGUF loading
- [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF) — GGUF models
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) — GGUF inference backend
"# Captioner_QWENVL" 
