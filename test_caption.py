"""
Standalone test script for QwenVL Image Captioning.
No WebUI dependency — runs directly from the command line.

Usage:
    python test_caption.py --image path/to/image.jpg
    python test_caption.py --image path/to/image.jpg --device cpu
    python test_caption.py --image path/to/image.jpg --device cuda --prompt "Detailed Description"
    python test_caption.py --image path/to/image.jpg --backend hf --model-id Qwen/Qwen2.5-VL-3B-Instruct

Input:  image file path
Output: caption string (printed to stdout, optionally saved to file)
"""
import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

# ── Allow running from project root ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

# Windows consoles default to cp1252; captions (and this script's own box
# drawing) are UTF-8, so print() would die with UnicodeEncodeError.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_caption")

from model_registry import DEFAULT_LLM_DIR, scan_models  # noqa: E402


def _auto_pick_gguf(llm_dir: Path):
    """
    Pick a Qwen3-VL model .gguf and its mmproj from llm_dir (via model_registry,
    so text-only GGUFs in the same folder are never picked).
    Prefers Q4_K_M > Q8_0 > F16, then the smaller file.

    Returns:
        (model_path, mmproj_path) or (None, None)
    """
    found = scan_models(llm_dir)["gguf"]
    if not found:
        return None, None
    order = ["Q4_K_M", "Q8_0", "F16"]
    best = min(found, key=lambda m: (order.index(m["quant"]) if m["quant"] in order else len(order), m["size_gb"]))
    return Path(best["model_path"]), Path(best["mmproj_path"])


# ─────────────────────────────────────────────────────────────────────────────
#  Core caption function (standalone, no WebUI)
# ─────────────────────────────────────────────────────────────────────────────

def caption_image(
    image_path: str,
    backend: str = "gguf",
    device: str = "auto",
    model_path: str = "",
    mmproj_path: str = "",
    model_id: str = "",
    vram_profile: str = "LowVRAM (6–8GB)",
    prompt: str = "Detailed Description",
    max_tokens: int = 512,
    llm_dir: Path = DEFAULT_LLM_DIR,
) -> str:
    """
    Generate a caption for an image. Completely standalone — no Gradio required.

    Args:
        image_path:   Absolute or relative path to the input image
        backend:      'gguf' or 'hf'
        device:       'auto' | 'cpu' | 'cuda'
        model_path:   GGUF model file path (auto-detected if empty)
        mmproj_path:  GGUF mmproj file path (auto-detected if empty)
        model_id:     HuggingFace model ID or local path (HF backend)
        vram_profile: VRAM profile key from models_catalog
        prompt:       Prompt template name OR raw prompt text
        max_tokens:   Maximum tokens to generate
        llm_dir:      Directory to scan for GGUF models

    Returns:
        Generated caption string
    """
    from captioner.prompts import get_prompt_names, resolve_prompt

    # ── Resolve image path ────────────────────────────────────────────────────
    img_path = Path(image_path).expanduser().resolve()
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")
    logger.info("Image: %s", img_path)

    # ── Resolve prompt ────────────────────────────────────────────────────────
    prompt_names = get_prompt_names()
    if prompt in prompt_names:
        system_prompt, final_prompt = resolve_prompt(prompt, "")
        logger.info("Prompt template: '%s'", prompt)
    else:
        # Treat as raw prompt text
        system_prompt, final_prompt = resolve_prompt("Detailed Description", prompt.strip())
        logger.info("Custom prompt: %s", final_prompt[:80])

    # ── Prepare image (resize to reasonable size) ─────────────────────────────
    from PIL import Image as PILImage
    from utils.image_utils import resize_image

    with PILImage.open(img_path) as img_obj:
        resized = resize_image(img_obj, 768, 768, "Adaptive image")
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        resized.save(tmp.name)
        tmp_path = tmp.name

    try:
        caption = _run_backend(
            backend=backend,
            device=device,
            tmp_path=tmp_path,
            model_path=model_path,
            mmproj_path=mmproj_path,
            model_id=model_id,
            vram_profile=vram_profile,
            final_prompt=final_prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            llm_dir=llm_dir,
        )
    finally:
        if Path(tmp_path).exists():
            os.unlink(tmp_path)

    return caption


def _run_backend(
    backend, device, tmp_path,
    model_path, mmproj_path, model_id,
    vram_profile, final_prompt, system_prompt, max_tokens, llm_dir,
) -> str:
    """Internal: load the captioner and generate caption."""

    if backend.lower() == "gguf":
        from captioner.gguf_captioner import GGUFCaptioner

        # Auto-detect GGUF files if not provided
        if not model_path or not mmproj_path:
            auto_model, auto_mmproj = _auto_pick_gguf(llm_dir)
            if not model_path:
                model_path = str(auto_model) if auto_model else ""
            if not mmproj_path:
                mmproj_path = str(auto_mmproj) if auto_mmproj else ""

        if not model_path:
            raise FileNotFoundError(
                f"No GGUF model found in {llm_dir}. "
                "Pass --model-path explicitly or place .gguf files in --llm-dir."
            )
        if not mmproj_path:
            raise FileNotFoundError(
                f"No mmproj .gguf found in {llm_dir}. "
                "Pass --mmproj-path explicitly."
            )

        logger.info("GGUF model: %s", model_path)
        logger.info("MMProj:     %s", mmproj_path)
        logger.info("Device:     %s", device)

        cap = GGUFCaptioner()
        cap.load_model(
            vram_profile=vram_profile,
            model_path=model_path,
            mmproj_path=mmproj_path,
            device=device,
        )
        result = cap.caption_image(tmp_path, final_prompt, max_tokens, system_prompt=system_prompt)
        cap.unload_model()
        return result

    elif backend.lower() == "hf":
        from captioner.hf_captioner import HFCaptioner

        if not model_id:
            model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
            logger.info("No model-id given; using default HF model: %s", model_id)

        logger.info("HF model:   %s", model_id)
        logger.info("Device:     %s", device)

        cap = HFCaptioner()
        cap.load_model(
            vram_profile=vram_profile,
            model_id=model_id,
            device=device,
        )
        result = cap.caption_image(tmp_path, final_prompt, max_tokens, system_prompt=system_prompt)
        cap.unload_model()
        return result

    else:
        raise ValueError(f"Unknown backend '{backend}'. Use 'gguf' or 'hf'.")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "QwenVL Standalone Caption Test\n"
            "Generates a caption for an image without the WebUI.\n\n"
            "Examples:\n"
            "  python test_caption.py --image photo.jpg\n"
            "  python test_caption.py --image photo.jpg --device cpu\n"
            "  python test_caption.py --image photo.jpg --prompt 'Booru Tags' --max-tokens 256\n"
            "  python test_caption.py --image photo.jpg --backend gguf "
            "--model-path model.gguf --mmproj-path mmproj.gguf\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--image", "-i",
        required=True,
        metavar="PATH",
        help="Path to the input image file.",
    )
    p.add_argument(
        "--backend",
        default="gguf",
        choices=["gguf", "hf"],
        help="Inference backend: 'gguf' (llama-cpp, default) or 'hf' (HuggingFace Transformers).",
    )
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help=(
            "Device to run on. "
            "'auto'=prefer GPU (default), 'cpu'=force CPU, 'cuda'=force GPU."
        ),
    )
    p.add_argument(
        "--model-path",
        default="",
        metavar="PATH",
        help="Path to GGUF model file. Auto-detected from --llm-dir if not set.",
    )
    p.add_argument(
        "--mmproj-path",
        default="",
        metavar="PATH",
        help="Path to mmproj GGUF file. Auto-detected from --llm-dir if not set.",
    )
    p.add_argument(
        "--model-id",
        default="",
        metavar="ID",
        help="HuggingFace model ID or local path (HF backend only).",
    )
    p.add_argument(
        "--llm-dir",
        default=os.environ.get("QWENVL_LLM_DIR", str(DEFAULT_LLM_DIR)),
        metavar="PATH",
        help=(
            f"Directory to scan for GGUF model files "
            f"(default: {DEFAULT_LLM_DIR})."
        ),
    )
    p.add_argument(
        "--vram-profile",
        default="LowVRAM (6–8GB)",
        choices=[
            "UltraLow (4GB)",
            "LowVRAM (6–8GB)",
            "NormalVRAM (12-16GB)",
            "HighVRAM (20GB+)",
        ],
        help="VRAM/RAM profile (affects quantization and GPU layers). Default: LowVRAM.",
    )
    p.add_argument(
        "--prompt",
        default="Detailed Description",
        metavar="TEMPLATE_OR_TEXT",
        help=(
            "Prompt template name OR raw prompt text. "
            "Templates: 'Short Caption', 'Detailed Description' (default), "
            "'Booru Tags', 'Structured Caption', 'Training Caption (SD/Flux)', "
            "'Object & Scene Analysis'. "
            "If the value is not a template name, it is used as a raw prompt."
        ),
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        metavar="N",
        help="Maximum number of tokens to generate (default: 512).",
    )
    p.add_argument(
        "--output", "-o",
        default="",
        metavar="PATH",
        help="Optional output .txt file path. If not set, caption is printed to stdout only.",
    )
    p.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress info logs; print only the caption.",
    )
    return p


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    # Map vram-profile arg to catalog key (handle dash vs en-dash)
    vram_map = {
        "UltraLow (4GB)": "UltraLow (4GB)",
        "LowVRAM (6–8GB)": "LowVRAM (6\u20138GB)",
        "NormalVRAM (12-16GB)": "NormalVRAM (12\u201316GB)",
        "HighVRAM (20GB+)": "HighVRAM (20GB+)",
    }
    vram_profile = vram_map.get(args.vram_profile, args.vram_profile)

    try:
        caption = caption_image(
            image_path=args.image,
            backend=args.backend,
            device=args.device,
            model_path=args.model_path,
            mmproj_path=args.mmproj_path,
            model_id=args.model_id,
            vram_profile=vram_profile,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            llm_dir=Path(args.llm_dir).expanduser(),
        )
    except FileNotFoundError as e:
        logger.error("%s", e)
        sys.exit(1)
    except Exception as e:
        logger.error("Caption failed: %s", e, exc_info=True)
        sys.exit(2)

    # ── Output ────────────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(caption)
    print("─" * 60 + "\n")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(caption, encoding="utf-8")
        logger.info("Caption saved → %s", out_path)


if __name__ == "__main__":
    main()
