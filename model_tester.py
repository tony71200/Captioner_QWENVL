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
