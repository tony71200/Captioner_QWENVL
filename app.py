"""
QwenVL Image Captioning — Gradio Web UI (v3)
Dark premium design · Model catalog · CPU + GPU support · 4GB VRAM
"""
import os
import argparse
import logging
import re
import threading
from pathlib import Path
from typing import Optional

import gradio as gr

from captioner.hf_captioner import HFCaptioner
from captioner.gguf_captioner import GGUFCaptioner
from captioner.prompts import PROMPT_TEMPLATES, get_prompt_names, resolve_prompt
from models_catalog import HF_VL_MODELS, GGUF_VL_MODELS, CATALOG_VERIFIED
from utils import hardware
from utils.vram_plan import PlanOption, plan_options, preflight, VramPlanError
from utils.image_utils import scan_folder, resize_image
from utils.file_utils import save_caption, caption_exists
from utils.system_info import get_system_info_html

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_LLM_DIR = Path(r"D:\Comfy\ComfyUI\models\llm")


def _parse_cli_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--llm-dir",
        default=os.environ.get("QWENVL_LLM_DIR", str(DEFAULT_LLM_DIR)),
        help="Root directory used to store and scan local LLM/VLM models.",
    )
    return parser.parse_known_args()[0]


APP_ARGS = _parse_cli_args()
LLM_DIR = Path(APP_ARGS.llm_dir).expanduser()
LLM_DIR_DISPLAY = str(LLM_DIR)

MODEL_STATUS_LEGEND = (
    "🟢 thoải mái trong ngân sách &nbsp;|&nbsp; "
    "🟡 sát ngưỡng &nbsp;|&nbsp; "
    "🔴 vượt ngân sách (vẫn chọn được — preflight sẽ hạ cấp hoặc chặn) &nbsp;|&nbsp; "
    "⬇️ chưa có sẵn trên máy, sẽ tải khi Load"
)

# ── Global state ──────────────────────────────────────────────────────────────
_captioner: Optional[object] = None
_stop_event = threading.Event()
PROMPT_NAMES = get_prompt_names()

# label → PlanOption. Gradio dropdown chỉ trả về chuỗi label; tra ngược qua dict
# này thay vì parse chuỗi (lỗi cũ: _extract_vram_from_label bóc số bằng regex).
_OPTION_BY_LABEL: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
#  UI helpers
# ─────────────────────────────────────────────────────────────────────────────
def _badge(msg: str, kind: str = "info") -> str:
    pal = {
        "info":    ("#38bdf8", "#0c1a2e"),
        "success": ("#4ade80", "#0a1f10"),
        "error":   ("#f87171", "#2a0a0a"),
        "warning": ("#fbbf24", "#2a1a00"),
        "loading": ("#a78bfa", "#170f2e"),
    }
    border, bg = pal.get(kind, pal["info"])
    icons = {"info": "ℹ️", "success": "✅", "error": "❌", "warning": "⚠️", "loading": "⏳"}
    icon = icons.get(kind, "ℹ️")
    return (
        f'<div style="padding:10px 16px;border-left:4px solid {border};'
        f'background:{bg};border-radius:8px;font-size:13.5px;'
        f'color:#e2e8f0;margin:4px 0;line-height:1.5;">'
        f'{icon}&nbsp; {msg}</div>'
    )


def _backend_kind(backend: str) -> str:
    return "hf" if "HuggingFace" in backend else "gguf"


def _catalog_for(backend: str) -> dict:
    return HF_VL_MODELS if _backend_kind(backend) == "hf" else GGUF_VL_MODELS


def _device_kind(device_choice: str) -> str:
    return {"CPU": "cpu", "GPU": "cuda", "Auto": "auto"}.get(device_choice, "auto")


def _current_budget(device_choice: str, budget_override) -> float:
    """Advanced de trong = tu suy; dien so = ghi de."""
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


def _hf_storage_dir(llm_dir: Path, model_name: str) -> Path:
    return llm_dir / "HuggingFace" / model_name


def _hf_is_available(model_name: str, llm_dir: Path) -> bool:
    model_dir = _hf_storage_dir(llm_dir, model_name)
    return model_dir.exists() and any(
        (model_dir / filename).exists()
        for filename in ("config.json", "preprocessor_config.json", "tokenizer_config.json")
    )


def _gguf_file_index(llm_dir: Path) -> dict:
    index = {}
    if not llm_dir.exists():
        return index
    for path in llm_dir.rglob("*.gguf"):
        index.setdefault(path.name.lower(), path)
    return index


def _resolve_gguf_local_assets(option: PlanOption, llm_dir: Path) -> dict:
    """Tim file .gguf da co tren may cho dung cau hinh da chon."""
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


def _expected_download_target(backend_kind: str, model_name: str, llm_dir: Path) -> Path:
    if backend_kind == "hf":
        return _hf_storage_dir(llm_dir, model_name)
    return llm_dir / "GGUF" / model_name


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
            f"Thư mục tải về: "
            f"{_expected_download_target(option.backend, option.model_name, LLM_DIR)}"
        )

    offload = ("toàn bộ layer" if option.gpu_layers < 0
               else f"{option.gpu_layers} layer trên GPU")
    presence = ("Đã có trên máy" if available
                else "⬇️ Sẽ tải về khi bấm Load")
    return (
        f'<div style="font-size:13px;color:#cbd5e1;line-height:1.6;">'
        f'<div style="margin-bottom:6px;">{option.label}</div>'
        f'<div style="color:#94a3b8;">Quant {option.quant} · ctx {option.n_ctx} · '
        f'{offload} · ước lượng {option.est_gib:.2f} GiB</div>'
        f'<div style="margin-top:8px;color:#94a3b8;">{presence}</div>'
        f'<div style="margin-top:8px;color:#64748b;">{detail}</div>'
        f'<div style="margin-top:8px;color:#64748b;">Repo: {info["repo_id"]} '
        f'(đã verify {CATALOG_VERIFIED["checked"]})</div>'
        f'<div style="margin-top:8px;">'
        f'<a href="{info["hf_url"]}" target="_blank" style="color:#38bdf8;font-size:12px;">'
        f'📦 Xem repo gốc ↗</a></div></div>'
    )


def _refresh_options(backend, device_choice, budget_override, current_label):
    """Dung lai dropdown tu ngan sach hien tai."""
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

def _download_hf_model(model_name: str, llm_dir: Path) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise RuntimeError("huggingface_hub is required to download HuggingFace models.") from e

    target_dir = _hf_storage_dir(llm_dir, model_name)
    target_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=HF_VL_MODELS[model_name]["repo_id"],
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
    )
    return target_dir


def _download_gguf_assets(option: PlanOption, llm_dir: Path) -> dict:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise RuntimeError("Can huggingface_hub de tai model GGUF.") from e

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


# ─────────────────────────────────────────────────────────────────────────────
#  Backend logic
# ─────────────────────────────────────────────────────────────────────────────
def _apply_overrides(option: PlanOption, ctx_override, layers_override, pixels_override) -> PlanOption:
    """Advanced de trong = giu gia tri tu suy."""
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

        # Preflight: doc LAI VRAM ngay luc nay, khong phai luc dung UI
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
            yield _badge(
                f"⏳ Chưa có trên máy — đang tải "
                f"{option.model_name} về {LLM_DIR}…", "loading")

        if _captioner is not None:
            _captioner.unload_model()
            _captioner = None

        device_param = _device_kind(device_choice)

        if option.backend == "hf":
            if not available:
                _download_hf_model(option.model_name, LLM_DIR)
            yield _badge(
                f"⏳ Đang nạp model HuggingFace trên {device_choice}…", "loading")
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
            yield _badge(
                f"⏳ Đang nạp model GGUF trên {device_choice}…", "loading")
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
            detail = f" ({reason})" if reason else ""
            yield _badge(f"GPU lỗi → tự chuyển sang CPU{detail}", "warning")
            yield _badge(
                f"Đã nạp — {option.model_name} {option.quant} "
                f"| Device: CPU (fallback)", "success")
        else:
            yield _badge(
                f"Đã nạp — {option.model_name} {option.quant} · "
                f"ctx {option.n_ctx} · ~{option.est_gib:.2f} GiB "
                f"| Device: {device_choice}", "success")
    except Exception as e:
        _captioner = None
        yield _badge(f"Nạp thất bại: {e}", "error")


def unload_model():
    global _captioner
    if _captioner is None:
        return _badge("No model loaded.", "warning")
    try:
        _captioner.unload_model()
        _captioner = None
        return _badge("Model unloaded and memory freed.", "success")
    except Exception as e:
        return _badge(f"Unload error: {e}", "error")


def _save_resized_temp_image(image_obj, resize_mode, resize_width, resize_height):
    import tempfile

    prepared_image = resize_image(image_obj, resize_width, resize_height, resize_mode)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    prepared_image.save(tmp.name)
    return tmp.name


def _save_resized_temp_image_from_path(image_path, resize_mode, resize_width, resize_height):
    from PIL import Image as PILImage

    with PILImage.open(image_path) as image_obj:
        return _save_resized_temp_image(image_obj, resize_mode, resize_width, resize_height)


def caption_single(image_obj, template_name, custom_prompt, max_tokens,
                   resize_mode, resize_width, resize_height):
    if _captioner is None:
        return "", _badge("Load a model first.", "warning")
    if image_obj is None:
        return "", _badge("Upload an image.", "warning")
    tmp_path = None
    try:
        from PIL import Image as PILImage
        if not isinstance(image_obj, PILImage.Image):
            return "", _badge("Invalid image.", "error")
        tmp_path = _save_resized_temp_image(image_obj, resize_mode, resize_width, resize_height)
        prompt = resolve_prompt(template_name, custom_prompt)
        caption = _captioner.caption_image(tmp_path, prompt, int(max_tokens))
        return caption, _badge("Caption generated!", "success")
    except Exception as e:
        return "", _badge(f"Error: {e}", "error")
    finally:
        if tmp_path and Path(tmp_path).exists():
            os.unlink(tmp_path)


def save_single_caption(image_obj, caption_text):
    if not caption_text.strip():
        return _badge("Nothing to save.", "warning")
    if image_obj is None:
        return _badge("No image selected.", "warning")
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image_obj.save(tmp.name)
            tmp_path = tmp.name
        desktop = Path.home() / "Desktop"
        desktop.mkdir(exist_ok=True)
        out = desktop / "caption_output.txt"
        out.write_text(caption_text, encoding="utf-8")
        os.unlink(tmp_path)
        return _badge(f"Saved → {out}", "success")
    except Exception as e:
        return _badge(f"Save failed: {e}", "error")


def start_batch(folder_path, output_folder, template_name, custom_prompt,
                max_tokens, resize_mode, resize_width, resize_height,
                skip_existing, recursive, progress=gr.Progress()):
    global _stop_event
    _stop_event.clear()
    if _captioner is None:
        yield _badge("Load a model first.", "warning"), ""
        return
    folder = (folder_path or "").strip()
    if not folder or not Path(folder).is_dir():
        yield _badge("Invalid folder path.", "error"), ""
        return
    out_dir = (output_folder or "").strip() or None
    images = scan_folder(folder, recursive=recursive)
    if not images:
        yield _badge("No images found.", "warning"), ""
        return

    prompt = resolve_prompt(template_name, custom_prompt)
    log_lines = []
    total = len(images)
    yield _badge(f"Starting — {total} images found.", "info"), ""

    for idx, img_path in enumerate(images, 1):
        if _stop_event.is_set():
            msg = f"⏹ Stopped at {idx-1}/{total}."
            log_lines.append(msg)
            yield _badge(msg, "warning"), "\n".join(log_lines)
            return

        progress(idx / total, desc=f"[{idx}/{total}] {Path(img_path).name}")
        fname = Path(img_path).name

        if skip_existing and caption_exists(img_path, out_dir):
            log_lines.append(f"⏭  [{idx}/{total}] {fname} — skipped")
            yield _badge(f"Processing {idx}/{total}…", "loading"), "\n".join(log_lines)
            continue

        prepared_path = None
        try:
            prepared_path = _save_resized_temp_image_from_path(
                img_path, resize_mode, resize_width, resize_height
            )
            caption = _captioner.caption_image(prepared_path, prompt, int(max_tokens))
            saved_path = save_caption(img_path, caption, out_dir, overwrite=True)
            log_lines.append(f"✅ [{idx}/{total}] {fname} → {Path(saved_path).name}")
        except Exception as e:
            log_lines.append(f"❌ [{idx}/{total}] {fname} — {e}")

        if prepared_path and Path(prepared_path).exists():
            os.unlink(prepared_path)

        yield _badge(f"Processing {idx}/{total}…", "loading"), "\n".join(log_lines)

    done = f"Done! {total} images processed."
    log_lines.append(f"\n🏁 {done}")
    yield _badge(done, "success"), "\n".join(log_lines)


def stop_batch():
    _stop_event.set()
    return _badge("Stop requested — halting after current image.", "warning")


# ─────────────────────────────────────────────────────────────────────────────
#  Dynamic UI helpers
# ─────────────────────────────────────────────────────────────────────────────
def on_prompt_change(name):
    tmpl = next((t for t in PROMPT_TEMPLATES if t["name"] == name), None)
    if name == "Custom":
        desc = "Use only the custom prompt below. If left blank, the app falls back to a basic description prompt."
    else:
        desc = tmpl["description"] if tmpl else ""
    return gr.update(value=desc)


def on_device_change(device_choice):
    """Update the system info panel when device radio changes."""
    return gr.update(value=get_system_info_html(device_choice))


# ─────────────────────────────────────────────────────────────────────────────
#  CSS — Dark slate / teal-accent design
# ─────────────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

*, body { font-family: 'Inter', sans-serif !important; box-sizing: border-box; }

/* ── Background ── */
.gradio-container {
    background: linear-gradient(145deg, #0d1117 0%, #0f1923 50%, #0d1117 100%) !important;
    min-height: 100vh;
}

/* ── Header ── */
.app-hero {
    text-align: center;
    padding: 2.2rem 1rem 0.4rem;
}
.app-hero h1 {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    letter-spacing: -0.5px;
}
.app-hero p {
    color: #64748b;
    font-size: 0.9rem;
    margin: 6px 0 0;
}

/* ── Tabs ── */
.tab-nav { background: rgba(255,255,255,0.03) !important; border-radius: 12px !important; }
.tab-nav button { color: #94a3b8 !important; font-weight: 500 !important; }
.tab-nav button.selected { color: #38bdf8 !important; border-bottom: 2px solid #38bdf8 !important; }

/* ── Cards / panels ── */
.gr-group, .gr-box {
    background: rgba(15,25,38,0.8) !important;
    border: 1px solid rgba(56,189,248,0.12) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(8px);
}

/* ── Labels ── */
label span, .gr-form > label { color: #94a3b8 !important; font-size: 13px !important; }

/* ── Inputs ── */
textarea, input[type=text], input[type=number], select {
    background: rgba(15,25,38,0.9) !important;
    border: 1px solid rgba(56,189,248,0.18) !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
    font-size: 13.5px !important;
}
textarea:focus, input:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 2px rgba(56,189,248,0.15) !important;
    outline: none !important;
}

/* ── Dropdowns ── */
.wrap { background: rgba(15,25,38,0.95) !important; border-color: rgba(56,189,248,0.2) !important; }
.item { color: #cbd5e1 !important; }
.item:hover, .item.selected { background: rgba(56,189,248,0.12) !important; color: #38bdf8 !important; }

/* ── Buttons ── */
button.primary {
    background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
    border: none !important; color: #fff !important; font-weight: 600 !important;
    border-radius: 8px !important; transition: all 0.2s !important;
    box-shadow: 0 2px 12px rgba(14,165,233,0.25) !important;
}
button.primary:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }
button.secondary {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
    color: #cbd5e1 !important; font-weight: 500 !important; border-radius: 8px !important;
}
button.secondary:hover { background: rgba(255,255,255,0.1) !important; }

/* ── Radio buttons — device selection ── */
.device-radio label { color: #94a3b8 !important; }
.device-radio .wrap { gap: 8px !important; }

/* ── Sliders ── */
input[type=range] { accent-color: #38bdf8 !important; }

/* ── Markdown ── */
.prose, .md-text, .gr-markdown { color: #94a3b8 !important; font-size: 13px !important; }
.prose h3, .gr-markdown h3 { color: #38bdf8 !important; font-size: 14px !important; font-weight: 600; }
.prose table { border-collapse: collapse; width: 100%; }
.prose th { background: rgba(56,189,248,0.08) !important; color: #38bdf8 !important; padding: 6px 10px; }
.prose td { padding: 5px 10px; border-bottom: 1px solid rgba(255,255,255,0.05); color: #94a3b8; }

/* ── Image upload area ── */
.image-container { border-radius:10px !important; overflow:hidden !important; }

/* ── System info panel ── */
.system-info-panel { max-height: 320px; overflow-y: auto; }
"""


# ─────────────────────────────────────────────────────────────────────────────
#  Build Gradio UI
# ─────────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="QwenVL Image Captioner", css=None) as demo:

    # ── Header ──────────────────────────────────────────────────────────────
    gr.HTML("""
    <div class="app-hero">
      <h1>🖼️ QwenVL Image Captioner</h1>
      <p>Qwen3-VL · Qwen2.5-VL &nbsp;|&nbsp; HuggingFace &amp; GGUF &nbsp;|&nbsp; CPU &amp; GPU &nbsp;|&nbsp; 4 GB VRAM support</p>
    </div>
    """)

    # ════════════════════════════════════════════════════════════════════════
    #  Tab 1 — Configuration
    # ════════════════════════════════════════════════════════════════════════
    with gr.Tab("⚙️ Setup"):
        with gr.Row(equal_height=False):

            # ── Left: controls ──
            with gr.Column(scale=3):

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

                # ── Inference Backend ──
                backend_radio = gr.Radio(
                    choices=["HuggingFace (Transformers)", "GGUF (llama-cpp)"],
                    value=initial_backend,
                    label="Inference Backend",
                    elem_id="backend_radio",
                )

                # ── Device Selection ──
                with gr.Group():
                    gr.HTML(
                        '<div style="font-size:13px;font-weight:600;color:#38bdf8;'
                        'padding:8px 0 4px;">🖥️ Device Selection</div>'
                    )
                    device_radio = gr.Radio(
                        choices=["Auto", "CPU", "GPU"],
                        value=initial_device,
                        label="Run on",
                        info="Auto: uu tien GPU, khong co thi CPU  |  CPU: ep CPU (GGUF: n_gpu_layers=0)  |  GPU: ep CUDA",
                        elem_id="device_radio",
                        elem_classes=["device-radio"],
                    )
                    hardware_panel = gr.HTML(value=_hardware_html(initial_device, None))
                    refresh_btn = gr.Button("🔄 Làm mới ngân sách", size="sm")
                    system_info_panel = gr.HTML(
                        value=get_system_info_html(initial_device),
                        elem_id="system_info_panel",
                        elem_classes=["system-info-panel"],
                    )

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

                # ── Model chooser ──
                with gr.Group():
                    gr.Markdown("### Select Model")
                    llm_dir_box = gr.Textbox(
                        label="Local Path (--llm-dir)",
                        value=LLM_DIR_DISPLAY,
                        interactive=False,
                    )
                    model_dd = gr.Dropdown(
                        choices=initial_labels,
                        value=initial_label,
                        label="Select Model",
                    )
                    gr.HTML(
                        f'<div style="font-size:12px;color:#94a3b8;line-height:1.5;">{MODEL_STATUS_LEGEND}</div>'
                    )
                    model_info = gr.HTML(
                        _render_option_info(initial_label) if initial_label
                        else _badge("Không tìm thấy cấu hình nào.", "warning")
                    )
            # ── Right: reference card ──
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

        with gr.Row():
            load_btn = gr.Button("🚀 Load Model", variant="primary", scale=3)
            unload_btn = gr.Button("🗑️ Unload", variant="secondary", scale=1)
        model_status = gr.HTML(_badge("Chưa nạp model nào.", "info"))

        # ── Wire events ──
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


    # ════════════════════════════════════════════════════════════════════════
    #  Tab 2 — Single Image
    # ════════════════════════════════════════════════════════════════════════
    with gr.Tab("🖼️ Single Image"):
        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                single_img = gr.Image(type="pil", label="Upload Image", height=340)
            with gr.Column(scale=1):
                s_template = gr.Dropdown(choices=PROMPT_NAMES,
                                         value="Detailed Description",
                                         label="Prompt Template")
                s_tmpl_desc = gr.Markdown("Describe image in detail including subjects, colors, composition, and mood.")
                s_custom = gr.Textbox(label="Custom Prompt", lines=3,
                                      placeholder="If filled, this overrides the selected template and is sent as the full prompt.")
                s_tokens = gr.Slider(64, 1024, value=512, step=64,
                                     label="Max New Tokens")
                s_resize_mode = gr.Dropdown(
                    choices=["Adaptive image", "Fit image"],
                    value="Adaptive image",
                    label="Resize Mode",
                )
                with gr.Row():
                    s_resize_width = gr.Slider(32, 2048, value=512, step=32,
                                               label="Resize Width")
                    s_resize_height = gr.Slider(32, 2048, value=512, step=32,
                                                label="Resize Height")
                s_gen_btn = gr.Button("▶ Generate Caption", variant="primary")

        s_output = gr.Textbox(label="Generated Caption", lines=7, interactive=True)
        s_status = gr.HTML()
        s_save_btn = gr.Button("💾 Save Caption (.txt) to Desktop", variant="secondary")

        s_template.change(on_prompt_change, [s_template], [s_tmpl_desc])
        s_gen_btn.click(caption_single,
                        inputs=[single_img, s_template, s_custom, s_tokens,
                                s_resize_mode, s_resize_width, s_resize_height],
                        outputs=[s_output, s_status])
        s_save_btn.click(save_single_caption,
                         inputs=[single_img, s_output], outputs=[s_status])

    # ════════════════════════════════════════════════════════════════════════
    #  Tab 3 — Batch Processing
    # ════════════════════════════════════════════════════════════════════════
    with gr.Tab("📁 Batch Processing"):
        with gr.Row():
            with gr.Column(scale=1):
                b_folder = gr.Textbox(label="Input Folder Path",
                                      placeholder="H:/images/")
                b_out_dir = gr.Textbox(label="Output Folder (blank = same as input)",
                                       placeholder="H:/captions/  (optional)")
                b_recursive = gr.Checkbox(label="Scan subfolders recursively", value=False)
                b_skip = gr.Checkbox(label="Skip if .txt already exists", value=True)
                with gr.Row():
                    b_merge_prompt = gr.Checkbox(label="Collection all generated prompt to 1 file and separate by '/n/n'", value=False)
                    b_merge_filename = gr.Textbox(label="Directory of txt output file")
            with gr.Column(scale=1):
                b_template = gr.Dropdown(choices=PROMPT_NAMES,
                                         value="Detailed Description",
                                         label="Prompt Template")
                b_tmpl_desc = gr.Markdown("Describe image in detail including subjects, colors, composition, and mood.")
                b_custom = gr.Textbox(label="Custom Prompt", lines=3,
                                      placeholder="If filled, this overrides the selected template and is sent as the full prompt.")
                b_tokens = gr.Slider(64, 1024, value=512, step=64,
                                     label="Max New Tokens")
                b_resize_mode = gr.Dropdown(
                    choices=["Adaptive image", "Fit image"],
                    value="Adaptive image",
                    label="Resize Mode",
                )
                with gr.Row():
                    b_resize_width = gr.Slider(32, 2048, value=512, step=32,
                                               label="Resize Width")
                    b_resize_height = gr.Slider(32, 2048, value=512, step=32,
                                                label="Resize Height")

        with gr.Row():
            b_start = gr.Button("▶ Start Batch", variant="primary", scale=3)
            b_stop = gr.Button("⏹ Stop", variant="secondary", scale=1)

        b_status = gr.HTML()
        b_log = gr.Textbox(label="Processing Log", lines=14, interactive=False)

        b_template.change(on_prompt_change, [b_template], [b_tmpl_desc])
        b_start.click(start_batch,
                      inputs=[b_folder, b_out_dir, b_template, b_custom,
                               b_tokens, b_resize_mode, b_resize_width, b_resize_height,
                               b_skip, b_recursive],
                      outputs=[b_status, b_log])
        b_stop.click(stop_batch, outputs=[b_status])

    # ════════════════════════════════════════════════════════════════════════
    #  Tab 4 — Models
    # ════════════════════════════════════════════════════════════════════════
    with gr.Tab("📦 Model Library"):
        gr.Markdown("## HuggingFace Models")

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

    # ════════════════════════════════════════════════════════════════════════
    #  Tab 5 — Help
    # ════════════════════════════════════════════════════════════════════════
    with gr.Tab("📖 Help"):
        gr.Markdown("""
## Quick Start

### 1. Run via `run_independent_env.bat` (recommended)
Double-click **`run_independent_env.bat`** — it activates the `.venv` and launches the app automatically.

### 2. Manual
```bash
.venv\\Scripts\\activate
python app.py --llm-dir "D:\\Comfy\\ComfyUI\\models\\llm"
```

### 3. Load a Model
- Go to **⚙️ Setup** → choose backend, device, and VRAM profile → pick a model → **🚀 Load**
- `🟢` = available locally and suitable
- `🟡` = suitable, will be downloaded when you click Load
- `🔴` = available locally but not recommended for the current backend/profile

### Device Selection
| Choice | Behavior |
|--------|----------|
| Auto | Prefer GPU (CUDA) if available, fallback to CPU |
| CPU | Force CPU — GGUF: `n_gpu_layers=0`, HF: `device_map="cpu"` |
| GPU | Force CUDA — GGUF: `n_gpu_layers` from profile, HF: `device_map="auto"` |

> **CPU note**: bitsandbytes 4bit/8bit quantization is not supported on CPU.
> The HF backend falls back to float32 — expect higher RAM usage and slower inference.
> GGUF CPU mode is recommended for CPU inference (Q4_K_M 4B model ≈ 2.5GB RAM).

### Standalone Test (no WebUI)
```bash
python test_caption.py --image path/to/image.jpg --device cpu
python test_caption.py --image path/to/image.jpg --device cuda
python test_caption.py --image path/to/image.jpg --prompt "Detailed Description" --max-tokens 512
```

### VRAM Profiles
| Profile | HF Quantization | GGUF Layers | Best For |
|---------|----------------|-------------|----------|
| UltraLow (4GB) | 4-bit NF4 + small pixels | 5 | GTX 1650, RTX 3050 |
| LowVRAM (6–8GB) | 4-bit NF4 | 10 | RTX 3060, RTX 4060 |
| NormalVRAM (12–16GB) | 8-bit int8 | 25 | RTX 3080, RTX 4070 |
| HighVRAM (20GB+) | BF16 full | All | RTX 3090, RTX 4090 |

### Prompt Templates
| Template | Best For |
|----------|----------|
| Short Caption | Quick tag / overview |
| Detailed Description | Full scene understanding (default) |
| Booru Tags | SD/anime dataset training |
| Structured Caption | Multi-field organized output |
| Training Caption (SD/Flux) | LoRA / fine-tune datasets |
| Object & Scene Analysis | Scene parsing tasks |
| Custom | Use only the textbox below |

### Troubleshooting
- **CUDA OOM**: Switch to UltraLow or LowVRAM profile, or use a 2B model
- **bitsandbytes error**: Run `setup.bat` again or `pip install bitsandbytes`
- **CPU slow**: Use GGUF backend with Q4_K_M — much faster than HF float32 on CPU
- **GGUF handler import error**: App tries `Qwen3VLChatHandler` → `Qwen25VLChatHandler` → `Llava15ChatHandler`
- **HF Qwen3-VL error**: Use GGUF backend for Qwen3-VL — HF backend targets Qwen2-VL API only
""")

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.queue(max_size=5).launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        inbrowser=True,
    )
