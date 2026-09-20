# QwenVL Image Captioner

A local image captioning tool powered by **Qwen3-VL** and **Qwen2.5-VL** vision-language models.  
Supports both a **Gradio Web UI** and a **standalone CLI** — no cloud, no API key required.

- **Backends**: HuggingFace Transformers · GGUF via `llama-cpp-python`
- **Devices**: CPU (no GPU required) · CUDA GPU
- **VRAM**: Works on 4 GB VRAM with GGUF Q4_K_M models
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

# 3. Install GGUF backend - PREBUILT WHEEL ONLY (see Troubleshooting)
pip install llama-cpp-python --only-binary :all: --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# ...or from a wheel you downloaded (matching your Python version):
pip install path/to/llama_cpp_python-0.3.39rc0+cu131-cp313-cp313-win_amd64.whl

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
3. **VRAM Profile** — Select profile matching your hardware
4. **Select Model** — Pick from catalog (🟢 = local + compatible, 🟡 = will download, 🔴 = local but not recommended)
5. **🚀 Load Model** — Downloads (if needed) and loads into memory

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

### Batch a whole folder

`batch_caption.py` loads the GGUF model once and captions every image in a
folder, writing `<image>.txt` beside each one. Defaults are already set for the
Qwen3-VL-8B dataset run, so this is enough:

```bash
python batch_caption.py
```

Windows one-click: `run_batch.bat` (passes any extra flags straight through).

```bash
# Smoke test on 2 images first
python batch_caption.py --limit 2

# Different folder, captions into their own directory
python batch_caption.py --images D:/pics --out D:/pics/captions

# Different opening phrase and template
python batch_caption.py --prefix "A 30 year-old woman" --template "Training Caption (SD/Flux)"

# Re-caption images that already have a .txt
python batch_caption.py --overwrite

# Verify the prefix/prompt logic without loading a model
python batch_caption.py --self-test
```

Every caption is forced to start with `--prefix` (default:
`A 25 year-old toned young man`): the phrase is pinned in the prompt, and any
caption that still drifts is corrected after generation. Images that already
have a `.txt` are skipped, so an interrupted run resumes where it stopped.

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
--vram-profile PROFILE    VRAM profile key (default: LowVRAM (6-8GB))
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
| GGUF | `n_gpu_layers` set by VRAM profile (−1 = all layers on GPU) |
| HF | `device_map="auto"`, BitsAndBytes 4-bit/8-bit quantization |

**System info panel** shows live VRAM usage (used/free/total with color-coded bar).

### Auto Mode

Detects CUDA availability at runtime:
- CUDA available → GPU
- No CUDA → CPU fallback

---

## Model Catalog

### HuggingFace Models

| Model | Size | Min VRAM (4-bit) | CPU RAM (float32) |
|-------|------|-----------------|-------------------|
| Qwen3-VL-2B-Instruct | 2B | ~1.5 GB | ~8 GB |
| Qwen3-VL-2B-Instruct-FP8 | 2B | ~2.5 GB | — |
| Qwen3-VL-4B-Instruct | 4B | ~2.0 GB | ~16 GB |
| Qwen3-VL-4B-Instruct-FP8 | 4B | ~2.5 GB | — |
| Qwen3-VL-8B-Instruct | 8B | ~4.5 GB | ~32 GB |
| Qwen2.5-VL-3B-Instruct | 3B | ~2.0 GB | ~12 GB |
| Qwen2.5-VL-7B-Instruct | 7B | ~5.0 GB | ~28 GB |

> **Note**: HF backend currently supports **Qwen2.5-VL** only. Use GGUF backend for Qwen3-VL.

### GGUF Models

| Model | Quant | File Size | Min VRAM | CPU OK |
|-------|-------|-----------|----------|--------|
| Qwen3-VL-4B-Instruct-GGUF | Q4_K_M | ~2.5 GB | ~3 GB | ✅ |
| Qwen3-VL-4B-Instruct-GGUF | Q8_0 | ~4.5 GB | ~5 GB | ✅ |
| Qwen3-VL-4B-Thinking-GGUF | Q4_K_M | ~2.5 GB | ~3 GB | ✅ |
| Qwen3-VL-8B-Instruct-GGUF | Q4_K_M | ~5.0 GB | ~6 GB | ✅ (slow) |
| Qwen3-VL-8B-Thinking-GGUF | Q4_K_M | ~5.0 GB | ~6 GB | ✅ (slow) |

All GGUF models are downloaded from [Qwen HuggingFace](https://huggingface.co/Qwen) on first load.

### VRAM Profiles

| Profile | HF Quant | GGUF GPU Layers | Recommended GPU |
|---------|----------|-----------------|-----------------|
| UltraLow (4GB) | 4-bit NF4 | 5 | GTX 1650, RTX 3050 |
| LowVRAM (6–8GB) | 4-bit NF4 | 10 | RTX 3060, RTX 4060 |
| NormalVRAM (12–16GB) | 8-bit int8 | 25 | RTX 3080, RTX 4070 |
| HighVRAM (20GB+) | BF16 full | All | RTX 3090, RTX 4090 |

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
├── models_catalog.py         # Model definitions, VRAM profiles
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

### `pip install llama-cpp-python` fails with "No such file or directory" (Windows)

```
OSError: [Errno 2] No such file or directory:
'C:\Users\...\Temp\pip-install-xxxx\llama-cpp-python_xxxx\vendor\llama.cpp\tools\ui\src\lib\...svelte'
```

pip fell back to the source distribution. Unpacking `vendor/llama.cpp` creates
paths longer than Windows' 260-char `MAX_PATH` limit and extraction dies
partway through - it is not a network or permissions problem, and retrying
will not help.

Install a **prebuilt wheel** instead; nothing is unpacked, nothing is compiled:

```bash
pip install llama-cpp-python --only-binary :all: --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

`--only-binary :all:` is the important part - without it pip silently retries
the source build. If no wheel exists for your Python version, download one
matching your interpreter (`cp313` for Python 3.13) and install it by path, or
point `LLAMA_WHEEL` in `setup.bat` at it.

Qwen3-VL needs **llama-cpp-python >= 0.3.39** for `Qwen3VLChatHandler`.

### `OMP: Error #15: Initializing libomp140.x86_64.dll`

torch ships Intel's OpenMP (`libiomp5md.dll`) and llama.cpp ships LLVM's
(`libomp140`); the second one to initialise aborts the process. The GGUF path
no longer imports torch at all, and `KMP_DUPLICATE_LIB_OK=TRUE` is set
automatically when torch really is needed in the same process (the Web UI).
If you hit this from your own script, import order is the cause - keep torch
out of GGUF-only processes.

### CUDA out of memory
- Switch to **UltraLow (4GB)** profile
- Use a **2B** or **4B Q4_K_M** GGUF model
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
