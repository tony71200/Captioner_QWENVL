"""
QwenVL Image Captioning — Gradio Web UI (v3)
Dark premium design · Model catalog · CPU + GPU support · 4GB VRAM
"""
import os
import argparse
import logging
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import gradio as gr

from captioner.hf_captioner import HFCaptioner
from captioner.gguf_captioner import GGUFCaptioner
from captioner.prompts import (
    PROMPT_TEMPLATES,
    get_prompt_default_name,
    get_prompt_name_label,
    get_prompt_names,
    prompt_needs_name,
    resolve_prompt,
)
from models_catalog import (
    HF_VL_MODELS, GGUF_VL_MODELS, VRAM_PROFILES,
    get_model_info_html,
)
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
    "🟢 available and suitable for the selected backend/profile &nbsp;|&nbsp; "
    "🟡 suitable but not downloaded yet &nbsp;|&nbsp; "
    "🔴 already available locally but not recommended for the selected backend/profile"
)

# ── Global state ──────────────────────────────────────────────────────────────
_captioner: Optional[object] = None
_stop_event = threading.Event()
PROMPT_NAMES = get_prompt_names()
VRAM_PROFILE_NAMES = list(VRAM_PROFILES.keys())


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


def _model_status_icon(available: bool, compatible: bool) -> str:
    if available and compatible:
        return "🟢"
    if compatible and not available:
        return "🟡"
    if available and not compatible:
        return "🔴"
    return ""


def _model_status_rank(icon: str) -> int:
    return {"🟢": 0, "🟡": 1, "🔴": 2}.get(icon, 3)


def _extract_model_name(choice_label: str) -> str:
    if not choice_label:
        return ""
    name = choice_label.split(" ", 1)[1] if " " in choice_label else choice_label
    return name.split(" [", 1)[0].strip()


def _selected_backend_kind(backend: str) -> str:
    return "hf" if "HuggingFace" in backend else "gguf"


def _vram_profile_bucket(vram_profile: str) -> str:
    if vram_profile.startswith("UltraLow"):
        return "ultralow"
    if vram_profile.startswith("LowVRAM"):
        return "low"
    if vram_profile.startswith("NormalVRAM"):
        return "normal"
    if vram_profile.startswith("HighVRAM"):
        return "high"
    return "low"


def _available_vram_for_profile(vram_profile: str) -> float:
    bucket = _vram_profile_bucket(vram_profile)
    return {"ultralow": 4.0, "low": 8.0, "normal": 16.0, "high": 24.0}.get(bucket, 8.0)


def _hf_quant_key_for_profile(vram_profile: str) -> str:
    quant_mode = VRAM_PROFILES.get(vram_profile, {}).get("hf_quant", "4bit")
    return {"4bit": "4bit", "8bit": "8bit", "none": "full"}.get(quant_mode, "full")


def _estimate_hf_required_vram(model_name: str, vram_profile: str) -> float:
    info = HF_VL_MODELS.get(model_name, {})
    vram_info = info.get("vram", {})
    quant_key = _hf_quant_key_for_profile(vram_profile)
    if quant_key in vram_info:
        return float(vram_info[quant_key])
    if vram_info:
        return float(min(vram_info.values()))
    return 999.0


def _pick_gguf_variant(model_name: str, vram_profile: str):
    info = GGUF_VL_MODELS.get(model_name, {})
    model_files = info.get("model_files", {})
    bucket = _vram_profile_bucket(vram_profile)
    priorities = {
        "ultralow": ["Q4_K_M", "Q8_0", "F16"],
        "low": ["Q4_K_M", "Q8_0", "F16"],
        "normal": ["Q8_0", "Q4_K_M", "F16"],
        "high": ["F16", "Q8_0", "Q4_K_M"],
    }.get(bucket, ["Q4_K_M", "Q8_0", "F16"])
    for quant_name in priorities:
        for label, filename in model_files.items():
            if quant_name in label:
                return label, filename
    for label, filename in model_files.items():
        return label, filename
    return "", ""


def _extract_vram_from_label(label: str) -> float:
    match = re.search(r"~([0-9]+(?:\.[0-9]+)?)\s*GB", label)
    if match:
        return float(match.group(1))
    return 999.0


def _estimate_gguf_required_vram(model_name: str, vram_profile: str) -> float:
    variant_label, _ = _pick_gguf_variant(model_name, vram_profile)
    return _extract_vram_from_label(variant_label)


def _hf_model_supported_by_loader(model_name: str) -> bool:
    return "Qwen2.5-VL" in model_name


def _is_model_compatible(backend: str, model_name: str, vram_profile: str) -> bool:
    available_vram = _available_vram_for_profile(vram_profile)
    if _selected_backend_kind(backend) == "hf":
        if not _hf_model_supported_by_loader(model_name):
            return False
        return _estimate_hf_required_vram(model_name, vram_profile) <= available_vram
    return _estimate_gguf_required_vram(model_name, vram_profile) <= available_vram


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


def _resolve_gguf_local_assets(model_name: str, llm_dir: Path, vram_profile: str) -> dict:
    info = GGUF_VL_MODELS.get(model_name, {})
    variant_label, model_filename = _pick_gguf_variant(model_name, vram_profile)
    mmproj_filename = info.get("mmproj_file", "")
    file_index = _gguf_file_index(llm_dir)
    model_path = file_index.get(model_filename.lower()) if model_filename else None
    mmproj_path = file_index.get(mmproj_filename.lower()) if mmproj_filename else None
    return {
        "variant_label": variant_label,
        "model_filename": model_filename,
        "mmproj_filename": mmproj_filename,
        "model_path": model_path,
        "mmproj_path": mmproj_path,
        "available": bool(model_path and mmproj_path),
    }


def _is_model_available(backend: str, model_name: str, llm_dir: Path, vram_profile: str) -> bool:
    if _selected_backend_kind(backend) == "hf":
        return _hf_is_available(model_name, llm_dir)
    return _resolve_gguf_local_assets(model_name, llm_dir, vram_profile)["available"]


def _build_model_choice_label(backend: str, model_name: str, vram_profile: str, llm_dir: Path) -> str:
    available = _is_model_available(backend, model_name, llm_dir, vram_profile)
    compatible = _is_model_compatible(backend, model_name, vram_profile)
    icon = _model_status_icon(available, compatible)
    suffix = ""
    if _selected_backend_kind(backend) == "gguf":
        variant_label, _ = _pick_gguf_variant(model_name, vram_profile)
        quant_match = re.search(r"(Q4_K_M|Q8_0|F16)", variant_label)
        if quant_match:
            suffix = f" [{quant_match.group(1)}]"
    return f"{icon} {model_name}{suffix}".strip()


def _get_candidate_model_names(backend: str, vram_profile: str, llm_dir: Path) -> list:
    catalog = HF_VL_MODELS if _selected_backend_kind(backend) == "hf" else GGUF_VL_MODELS
    candidates = []
    for model_name in catalog:
        available = _is_model_available(backend, model_name, llm_dir, vram_profile)
        compatible = _is_model_compatible(backend, model_name, vram_profile)
        if available or compatible:
            label = _build_model_choice_label(backend, model_name, vram_profile, llm_dir)
            icon = _model_status_icon(available, compatible)
            required_vram = (
                _estimate_hf_required_vram(model_name, vram_profile)
                if _selected_backend_kind(backend) == "hf"
                else _estimate_gguf_required_vram(model_name, vram_profile)
            )
            candidates.append((label, icon, required_vram, model_name))
    candidates.sort(key=lambda item: (_model_status_rank(item[1]), item[2], item[3]))
    return [item[0] for item in candidates]


def _expected_download_target(backend: str, model_name: str, llm_dir: Path) -> Path:
    if _selected_backend_kind(backend) == "hf":
        return _hf_storage_dir(llm_dir, model_name)
    return llm_dir / "GGUF" / model_name


def _render_model_info(backend: str, choice_label: str, vram_profile: str, llm_dir: Path) -> str:
    model_name = _extract_model_name(choice_label)
    if not model_name:
        return _badge("No model detected for the current backend/profile.", "warning")

    compatible = _is_model_compatible(backend, model_name, vram_profile)
    available = _is_model_available(backend, model_name, llm_dir, vram_profile)
    icon = _model_status_icon(available, compatible) or "⚪"
    catalog = HF_VL_MODELS if _selected_backend_kind(backend) == "hf" else GGUF_VL_MODELS
    info = catalog.get(model_name, {})
    repo_id = info.get("repo_id", "")
    repo_url = info.get("hf_url", "")

    if _selected_backend_kind(backend) == "hf":
        local_target = _hf_storage_dir(llm_dir, model_name)
        detail = f"Storage target: {local_target}"
    else:
        assets = _resolve_gguf_local_assets(model_name, llm_dir, vram_profile)
        detail = (
            f"Variant: {assets['variant_label']}<br>"
            f"Model file: {assets['model_filename']}<br>"
            f"MMProj file: {assets['mmproj_filename']}<br>"
            f"Download target: {_expected_download_target(backend, model_name, llm_dir)}"
        )

    suitability = "Suitable" if compatible else "Not recommended"
    presence = "Available locally" if available else "Will be downloaded to local storage when loaded"
    repo_link = (
        f'<a href="{repo_url}" target="_blank" style="color:#38bdf8;font-size:12px;">'
        f'📦 View source model ↗</a>'
    ) if repo_url else ""

    return (
        f'<div style="font-size:13px;color:#cbd5e1;line-height:1.6;">'
        f'<div style="margin-bottom:6px;">{icon}&nbsp; <strong>{model_name}</strong></div>'
        f'<div style="color:#94a3b8;">{info.get("description", "")}</div>'
        f'<div style="margin-top:8px;color:#94a3b8;">{suitability} • {presence}</div>'
        f'<div style="margin-top:8px;color:#64748b;">{detail}</div>'
        f'<div style="margin-top:8px;color:#64748b;">Repo ID: {repo_id}</div>'
        f'<div style="margin-top:8px;">{repo_link}</div>'
        f'</div>'
    )


def _refresh_model_selector(backend: str, vram_profile: str, current_choice: str):
    choices = _get_candidate_model_names(backend, vram_profile, LLM_DIR)
    if not choices:
        return (
            gr.update(choices=[], value=None),
            _badge("No compatible or locally available models were found for this backend/profile.", "warning"),
            gr.update(visible="HuggingFace" in backend),
        )

    selected = current_choice if current_choice in choices else choices[0]
    return (
        gr.update(choices=choices, value=selected),
        _render_model_info(backend, selected, vram_profile, LLM_DIR),
        gr.update(visible="HuggingFace" in backend),
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


def _download_gguf_assets(model_name: str, llm_dir: Path, vram_profile: str) -> dict:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise RuntimeError("huggingface_hub is required to download GGUF models.") from e

    info = GGUF_VL_MODELS[model_name]
    assets = _resolve_gguf_local_assets(model_name, llm_dir, vram_profile)
    target_dir = _expected_download_target("GGUF (llama-cpp)", model_name, llm_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if not assets["model_path"]:
        assets["model_path"] = Path(
            hf_hub_download(
                repo_id=info["repo_id"],
                filename=assets["model_filename"],
                local_dir=str(target_dir),
                local_dir_use_symlinks=False,
            )
        )
    if not assets["mmproj_path"]:
        assets["mmproj_path"] = Path(
            hf_hub_download(
                repo_id=info["repo_id"],
                filename=assets["mmproj_filename"],
                local_dir=str(target_dir),
                local_dir_use_symlinks=False,
            )
        )
    assets["available"] = bool(assets["model_path"] and assets["mmproj_path"])
    return assets


# ─────────────────────────────────────────────────────────────────────────────
#  Backend logic
# ─────────────────────────────────────────────────────────────────────────────
def load_model(backend, vram_profile, selected_model, flash_attn, device_choice):
    global _captioner
    try:
        model_name = _extract_model_name(selected_model)
        if not model_name:
            yield _badge("Select a model first.", "warning")
            return

        available = _is_model_available(backend, model_name, LLM_DIR, vram_profile)
        compatible = _is_model_compatible(backend, model_name, vram_profile)
        if not compatible:
            yield _badge(
                f"Selected model is not recommended for {backend} with profile {vram_profile}, attempting load anyway…",
                "warning",
            )
        if not available:
            yield _badge(f"⏳ Model not found locally. Downloading {model_name} to {LLM_DIR}…", "loading")

        if _captioner is not None:
            _captioner.unload_model()
            _captioner = None

        # Map UI device_choice → backend device param
        device_map = {"CPU": "cpu", "GPU": "cuda", "Auto": "auto"}
        device_param = device_map.get(device_choice, "auto")

        if "HuggingFace" in backend:
            if not available:
                _download_hf_model(model_name, LLM_DIR)
            model_id = str(_hf_storage_dir(LLM_DIR, model_name))
            yield _badge(f"⏳ Loading HuggingFace model on {device_choice} — please wait…", "loading")
            c = HFCaptioner()
            c.load_model(
                vram_profile=vram_profile,
                model_id=model_id,
                use_flash_attn=flash_attn,
                device=device_param,
            )
        else:
            assets = _resolve_gguf_local_assets(model_name, LLM_DIR, vram_profile)
            if not available:
                assets = _download_gguf_assets(model_name, LLM_DIR, vram_profile)
            yield _badge(f"⏳ Loading GGUF model on {device_choice} — please wait…", "loading")
            c = GGUFCaptioner()
            c.load_model(
                vram_profile=vram_profile,
                model_path=str(assets["model_path"]),
                mmproj_path=str(assets["mmproj_path"]),
                model_name=model_name,
                device=device_param,
            )

        _captioner = c
        runtime_device = getattr(c, "runtime_device", "")
        if runtime_device == "cpu-fallback":
            reason = getattr(c, "fallback_reason", "")
            detail = f" ({reason})" if reason else ""
            yield _badge(
                f"GPU failed → auto switched to CPU fallback{detail}",
                "warning",
            )
            yield _badge(f"Model loaded — {backend} | {vram_profile} | Device: CPU (fallback)", "success")
        else:
            yield _badge(f"Model loaded — {backend} | {vram_profile} | Device: {device_choice}", "success")
    except Exception as e:
        _captioner = None
        yield _badge(f"Load failed: {e}", "error")


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


def caption_single(image_obj, template_name, custom_prompt, subject_name, max_tokens,
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
        system_prompt, user_prompt = resolve_prompt(template_name, custom_prompt, subject_name)
        caption = _captioner.caption_image(
            tmp_path, user_prompt, int(max_tokens), system_prompt=system_prompt
        )
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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = desktop / f"caption_output_{timestamp}.txt"
        out.write_text(caption_text, encoding="utf-8")
        os.unlink(tmp_path)
        return _badge(f"Saved → {out}", "success")
    except Exception as e:
        return _badge(f"Save failed: {e}", "error")


def start_batch(folder_path, output_folder, template_name, custom_prompt,
                subject_name, max_tokens, resize_mode, resize_width, resize_height,
                skip_existing, recursive, merge_prompt, merge_output_folder,
                progress=gr.Progress()):
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

    merge_path = None
    merged_count = 0
    if merge_prompt:
        merge_dir_text = (merge_output_folder or "").strip()
        if not merge_dir_text:
            yield _badge("Merge output folder is required when merge is enabled.", "error"), ""
            return
        try:
            merge_dir = Path(merge_dir_text).expanduser()
            merge_dir.mkdir(parents=True, exist_ok=True)
            merge_path = merge_dir / f"sample_{datetime.now().strftime('%Y_%m_%d')}.txt"
        except Exception as e:
            yield _badge(f"Cannot prepare merge output folder: {e}", "error"), ""
            return

    system_prompt, user_prompt = resolve_prompt(template_name, custom_prompt, subject_name)
    log_lines = []
    total = len(images)
    processed_count = 0
    skipped_count = 0
    failed_count = 0
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
            skipped_count += 1
            log_lines.append(f"⏭  [{idx}/{total}] {fname} — skipped")
            yield _badge(f"Processing {idx}/{total}…", "loading"), "\n".join(log_lines)
            continue

        prepared_path = None
        try:
            prepared_path = _save_resized_temp_image_from_path(
                img_path, resize_mode, resize_width, resize_height
            )
            caption = _captioner.caption_image(
                prepared_path, user_prompt, int(max_tokens), system_prompt=system_prompt
            )
            saved_path = save_caption(img_path, caption, out_dir, overwrite=True)
            processed_count += 1
            if merge_path is not None:
                separator = "\n\n" if merge_path.exists() and merge_path.stat().st_size > 0 else ""
                with open(merge_path, "a", encoding="utf-8") as f:
                    f.write(separator + caption)
                merged_count += 1
            log_lines.append(f"✅ [{idx}/{total}] {fname} → {Path(saved_path).name}")
        except Exception as e:
            log_lines.append(f"❌ [{idx}/{total}] {fname} — {e}")

        if prepared_path and Path(prepared_path).exists():
            os.unlink(prepared_path)

        yield _badge(f"Processing {idx}/{total}…", "loading"), "\n".join(log_lines)

    failed_count = total - processed_count - skipped_count
    merge_detail = f", merged: {merged_count}" if merge_path is not None else ""
    done = (
        f"Done! processed: {processed_count}, skipped: {skipped_count}, "
        f"failed: {failed_count}{merge_detail}."
    )
    if merge_path is not None:
        log_lines.append(f"Merge file: {merge_path}")
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
    needs_name = prompt_needs_name(name)
    default_name = get_prompt_default_name(name)
    placeholder = f"Default: {default_name}" if default_name else ""
    return (
        gr.update(value=desc),
        gr.update(
            label=get_prompt_name_label(name),
            placeholder=placeholder,
            visible=True if needs_name else "hidden",
            value="",
        ),
    )



def on_preview_prompt(template_name, custom_prompt, subject_name):
    system_prompt, user_prompt = resolve_prompt(template_name, custom_prompt, subject_name)
    return system_prompt, user_prompt

def on_merge_prompt_change(enabled):
    is_enabled = bool(enabled)
    return gr.update(visible=True if is_enabled else "hidden", interactive=is_enabled)


def on_vram_change(profile):
    info = VRAM_PROFILES.get(profile, {})
    return gr.update(value=info.get("description", ""))


def on_device_change(device_choice):
    """Update the system info panel when device radio changes."""
    return gr.update(value=get_system_info_html(device_choice))


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

            # ── Left: controls ──────────────────────────────────────────────
            with gr.Column(scale=3):

                initial_backend = "GGUF (llama-cpp)"
                initial_profile = VRAM_PROFILE_NAMES[1]
                initial_device = "Auto"
                initial_model_choices = _get_candidate_model_names(initial_backend, initial_profile, LLM_DIR)
                initial_model = initial_model_choices[0] if initial_model_choices else None

                # ── Inference Backend ─────────────────────────────────────
                backend_radio = gr.Radio(
                    choices=["HuggingFace (Transformers)", "GGUF (llama-cpp)"],
                    value=initial_backend,
                    label="Inference Backend",
                )

                # ── Device Selection ──────────────────────────────────────
                with gr.Group():
                    gr.HTML(
                        '<div style="font-size:13px;font-weight:600;color:#38bdf8;'
                        'padding:8px 0 4px;">🖥️ Device Selection</div>'
                    )
                    device_radio = gr.Radio(
                        choices=["Auto", "CPU", "GPU"],
                        value=initial_device,
                        label="Run on",
                        info="Auto: prefer GPU, fallback to CPU  |  CPU: force CPU (GGUF: n_gpu_layers=0)  |  GPU: force CUDA",
                    )
                    system_info_panel = gr.HTML(
                        value=get_system_info_html(initial_device),
                    )

                # ── VRAM / RAM Profile ────────────────────────────────────
                vram_radio = gr.Radio(
                    choices=VRAM_PROFILE_NAMES,
                    value=initial_profile,
                    label="VRAM Profile (GPU) / RAM Profile (CPU)",
                )
                vram_desc = gr.Markdown(VRAM_PROFILES[initial_profile]["description"])

                flash_cb = gr.Checkbox(
                    label="⚡ Flash Attention 2  (HighVRAM only · Ampere+ GPU)",
                    value=False,
                    visible=False,
                )

                # ── Model chooser ─────────────────────────────────────────
                with gr.Group():
                    gr.Markdown("### Select Model")
                    llm_dir_box = gr.Textbox(
                        label="Local Path (--llm-dir)",
                        value=LLM_DIR_DISPLAY,
                        interactive=False,
                    )
                    model_dd = gr.Dropdown(
                        choices=initial_model_choices,
                        value=initial_model,
                        label="Select Model",
                    )
                    gr.HTML(
                        f'<div style="font-size:12px;color:#94a3b8;line-height:1.5;">{MODEL_STATUS_LEGEND}</div>'
                    )
                    model_info = gr.HTML(
                        _render_model_info(initial_backend, initial_model, initial_profile, LLM_DIR)
                        if initial_model else _badge("No model found in the current llm-dir.", "warning")
                    )

            # ── Right: reference card ────────────────────────────────────
            with gr.Column(scale=2):
                gr.Markdown("""
### 📊 VRAM / RAM Reference

| Profile | HF Quant | GGUF Layers | Est. VRAM |
|---------|----------|-------------|-----------|
| UltraLow (4GB) | 4-bit NF4 | 5 | ~2–4 GB |
| LowVRAM (6–8GB) | 4-bit NF4 | 10 | ~4–6 GB |
| NormalVRAM (12–16GB) | 8-bit int8 | 25 | ~8–14 GB |
| HighVRAM (20GB+) | BF16 full | All | ~15–28 GB |

### 🖥️ CPU Mode Notes
- GGUF backend: all layers run on CPU (`n_gpu_layers=0`)
- CPU threads set to `os.cpu_count()` automatically
- HF backend: `device_map="cpu"`, float32, **no BnB quantization**
- Expect slower inference vs GPU — 2B/4B models recommended

### 🟢 4GB-Friendly Models (HF)
| Model | Min VRAM |
|-------|---------|
| Qwen3-VL-2B-Instruct | ~1.5 GB (4-bit) |
| Qwen3-VL-2B-Instruct-FP8 | ~2.5 GB |
| Qwen3-VL-4B-Instruct-FP8 | ~2.5 GB |
| Qwen2.5-VL-3B-Instruct | ~2 GB (4-bit) |

### 🧠 4GB-Friendly Models (GGUF)
| Model | Quant | VRAM |
|-------|-------|------|
| Qwen3-VL-4B Q4_K_M | Q4_K_M | ~2.5 GB |
| Qwen3-VL-4B Thinking Q4_K_M | Q4_K_M | ~2.5 GB |
""")

        with gr.Row():
            load_btn = gr.Button("🚀 Load Model", variant="primary", scale=3)
            unload_btn = gr.Button("🗑️ Unload", variant="secondary", scale=1)
        model_status = gr.HTML(_badge("No model loaded.", "info"))

        # ── Wire events ─────────────────────────────────────────────────────
        device_radio.change(
            on_device_change,
            inputs=[device_radio],
            outputs=[system_info_panel],
            queue=False,
        )
        vram_radio.change(on_vram_change, [vram_radio], [vram_desc], queue=False)
        backend_radio.change(
            _refresh_model_selector,
            [backend_radio, vram_radio, model_dd],
            [model_dd, model_info, flash_cb],
        )
        vram_radio.change(
            _refresh_model_selector,
            [backend_radio, vram_radio, model_dd],
            [model_dd, model_info, flash_cb],
        )
        model_dd.change(
            lambda backend, profile, choice: _render_model_info(backend, choice, profile, LLM_DIR),
            [backend_radio, vram_radio, model_dd],
            [model_info],
            queue=False,
        )

        load_btn.click(
            load_model,
            inputs=[backend_radio, vram_radio, model_dd, flash_cb, device_radio],
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
                s_subject_name = gr.Textbox(
                    label="Character/Object name",
                    placeholder="Default: Ivan_Ryo",
                    visible="hidden",
                )
                with gr.Accordion("Prompt Preview", open=False):
                    s_system_preview = gr.Textbox(
                        label="System Prompt",
                        lines=4,
                        value=resolve_prompt("Detailed Description", "", "")[0],
                        interactive=False,
                    )
                    s_user_preview = gr.Textbox(
                        label="User Prompt",
                        lines=8,
                        value=resolve_prompt("Detailed Description", "", "")[1],
                        interactive=False,
                    )
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

        s_template.change(on_prompt_change, [s_template], [s_tmpl_desc, s_subject_name], queue=False)
        for prompt_input in (s_template, s_custom, s_subject_name):
            prompt_input.change(
                on_preview_prompt,
                [s_template, s_custom, s_subject_name],
                [s_system_preview, s_user_preview],
                queue=False,
            )
        s_custom.input(
            on_preview_prompt,
            [s_template, s_custom, s_subject_name],
            [s_system_preview, s_user_preview],
            queue=False,
        )
        s_subject_name.input(
            on_preview_prompt,
            [s_template, s_custom, s_subject_name],
            [s_system_preview, s_user_preview],
            queue=False,
        )
        s_gen_btn.click(caption_single,
                        inputs=[single_img, s_template, s_custom, s_subject_name, s_tokens,
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
                    b_merge_prompt = gr.Checkbox(label="Merge generated captions", value=False)
                    b_merge_filename = gr.Textbox(
                        label="Merge output folder",
                        placeholder="D:/captions/merged",
                        visible="hidden",
                        interactive=False,
                    )
            with gr.Column(scale=1):
                b_template = gr.Dropdown(choices=PROMPT_NAMES,
                                         value="Detailed Description",
                                         label="Prompt Template")
                b_tmpl_desc = gr.Markdown("Describe image in detail including subjects, colors, composition, and mood.")
                b_custom = gr.Textbox(label="Custom Prompt", lines=3,
                                      placeholder="If filled, this overrides the selected template and is sent as the full prompt.")
                b_subject_name = gr.Textbox(
                    label="Character/Object name",
                    placeholder="Default: Ivan_Ryo",
                    visible="hidden",
                )
                with gr.Accordion("Prompt Preview", open=False):
                    b_system_preview = gr.Textbox(
                        label="System Prompt",
                        lines=4,
                        value=resolve_prompt("Detailed Description", "", "")[0],
                        interactive=False,
                    )
                    b_user_preview = gr.Textbox(
                        label="User Prompt",
                        lines=8,
                        value=resolve_prompt("Detailed Description", "", "")[1],
                        interactive=False,
                    )
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

        b_template.change(on_prompt_change, [b_template], [b_tmpl_desc, b_subject_name], queue=False)
        for prompt_input in (b_template, b_custom, b_subject_name):
            prompt_input.change(
                on_preview_prompt,
                [b_template, b_custom, b_subject_name],
                [b_system_preview, b_user_preview],
                queue=False,
            )
        b_custom.input(
            on_preview_prompt,
            [b_template, b_custom, b_subject_name],
            [b_system_preview, b_user_preview],
            queue=False,
        )
        b_subject_name.input(
            on_preview_prompt,
            [b_template, b_custom, b_subject_name],
            [b_system_preview, b_user_preview],
            queue=False,
        )
        b_merge_prompt.change(on_merge_prompt_change, [b_merge_prompt], [b_merge_filename], queue=False)
        b_start.click(start_batch,
                      inputs=[b_folder, b_out_dir, b_template, b_custom,
                               b_subject_name, b_tokens, b_resize_mode, b_resize_width, b_resize_height,
                               b_skip, b_recursive, b_merge_prompt, b_merge_filename],
                      outputs=[b_status, b_log])
        b_stop.click(stop_batch, outputs=[b_status])

    # ════════════════════════════════════════════════════════════════════════
    #  Tab 4 — Models
    # ════════════════════════════════════════════════════════════════════════
    with gr.Tab("📦 Model Library"):
        gr.Markdown("## HuggingFace Models")

        rows_hf = []
        for mname, minfo in HF_VL_MODELS.items():
            vram_d = minfo.get("vram", {})
            vram_str = " / ".join(f"{k}:{v}GB" for k, v in vram_d.items()) if vram_d else "—"
            ok4 = "🟢" if minfo.get("min_vram_4gb") else "🔴"
            rows_hf.append([mname, minfo["series"], minfo["size"],
                             ok4, vram_str, minfo["repo_id"]])

        gr.Dataframe(
            headers=["Name", "Series", "Size", "4GB OK", "VRAM (full/8bit/4bit)", "HF Repo ID"],
            value=rows_hf,
            interactive=False,
            wrap=True,
        )

        gr.Markdown("## GGUF Models")
        rows_gguf = []
        for mname, minfo in GGUF_VL_MODELS.items():
            ok4 = "🟢" if minfo.get("min_vram_4gb") else "🔴"
            files = ", ".join(minfo.get("model_files", {}).keys())
            rows_gguf.append([mname, minfo["series"], minfo["size"],
                               ok4, files, minfo["repo_id"]])
        gr.Dataframe(
            headers=["Name", "Series", "Size", "4GB OK", "Available Quants", "HF Repo ID"],
            value=rows_gguf,
            interactive=False,
            wrap=True,
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
