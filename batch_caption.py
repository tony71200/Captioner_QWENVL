"""
Batch caption a folder of images with a local GGUF Qwen-VL model.

Standalone — no Gradio, no WebUI. The model is loaded ONCE and reused for
every image, which is the whole point of running a batch from the CLI.

Usage:
    python batch_caption.py                       # all defaults below
    python batch_caption.py --limit 2             # smoke test on 2 images
    python batch_caption.py --images D:\\pics --out D:\\pics\\captions
    python batch_caption.py --prefix "A 30 year-old woman"
    python batch_caption.py --self-test           # no model load, just asserts

Output: one <image-stem>.txt next to each image (or in --out).
"""
import argparse
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# ── Defaults for this dataset / model ────────────────────────────────────────
DEFAULT_MODEL = Path(
    r"D:\001_Personal_Proj\Comfy\ComfyUI\models\llm\GGUF"
    r"\Qwen3-VL-8B-Instruct-GGUF\Qwen3VL-8B-Instruct-Q4_K_M.gguf"
)
DEFAULT_IMAGES = Path(r"D:\001_Personal_Proj\Comfy\srcs\260914\Image")
DEFAULT_PREFIX = "A 25 year-old toned young man"
DEFAULT_TEMPLATE = "Detailed Description"
DEFAULT_MODEL_NAME = "Qwen3-VL-8B-Instruct-GGUF"   # key into models_catalog.GGUF_VL_MODELS
DEFAULT_PROFILE = "NormalVRAM (12–16GB)"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("batch_caption")


def pick_mmproj(model_path: Path) -> Path:
    """Find the mmproj .gguf sitting next to the model. Prefers F16 over Q8_0."""
    candidates = [p for p in model_path.parent.glob("*.gguf") if "mmproj" in p.name.lower()]
    if not candidates:
        raise FileNotFoundError(f"No mmproj*.gguf found in {model_path.parent}")
    candidates.sort(key=lambda p: (0 if "f16" in p.name.lower() else 1, p.name))
    return candidates[0]


def build_prompt(template: str, prefix: str) -> tuple[str, str]:
    """Resolve the template, then pin the opening words onto the user prompt."""
    from captioner.prompts import resolve_prompt

    system_prompt, user_prompt = resolve_prompt(template, "")
    if prefix:
        user_prompt += (
            f' Begin the description with exactly these words: "{prefix}". '
            "Write one continuous paragraph with no line breaks, no preamble "
            "and no bullet points. Output only the description."
        )
    return system_prompt, user_prompt


def enforce_prefix(caption: str, prefix: str) -> str:
    """Guarantee the caption opens with `prefix`, normalising its casing."""
    text = " ".join(caption.split())
    if not prefix:
        return text
    if not text:
        return prefix
    if text.lower().startswith(prefix.lower()):
        return prefix + text[len(prefix):]
    # ponytail: blunt splice — if the model opened with its own subject phrase the
    # result reads slightly redundant. Only fires when the model ignores the
    # instruction; swap for a single re-ask if that turns out to be common.
    return f"{prefix}, {text[:1].lower()}{text[1:]}"


def resize_to_temp(image_path: str, width: int, height: int, mode: str) -> str:
    """Write a resized copy to a temp .png and return its path."""
    from PIL import Image
    from utils.image_utils import resize_image

    with Image.open(image_path) as img:
        resized = resize_image(img, width, height, mode)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        resized.save(tmp.name)
        return tmp.name


def run_batch(args) -> int:
    from captioner.gguf_captioner import GGUFCaptioner
    from utils.file_utils import caption_exists, save_caption
    from utils.image_utils import scan_folder

    model_path = Path(args.model).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(f"GGUF model not found: {model_path}")
    mmproj_path = Path(args.mmproj).expanduser() if args.mmproj else pick_mmproj(model_path)

    images = scan_folder(str(Path(args.images).expanduser()), recursive=args.recursive)
    if not images:
        logger.error("No images found in %s", args.images)
        return 1
    if args.limit:
        images = images[: args.limit]

    out_dir = args.out or None
    system_prompt, user_prompt = build_prompt(args.template, args.prefix)

    logger.info("Model:   %s", model_path.name)
    logger.info("MMProj:  %s", mmproj_path.name)
    logger.info("Images:  %d from %s", len(images), args.images)
    logger.info("Output:  %s", out_dir or "<next to each image>")
    logger.info("Prompt:  %s...", user_prompt[:120])

    cap = GGUFCaptioner()
    cap.load_model(
        vram_profile=args.profile,
        model_path=str(model_path),
        mmproj_path=str(mmproj_path),
        model_name=args.model_name,
        device=args.device,
    )
    logger.info("Loaded on device=%s", cap.runtime_device)

    done = skipped = failed = 0
    started = time.perf_counter()
    try:
        for idx, img in enumerate(images, 1):
            name = Path(img).name
            if args.skip_existing and caption_exists(img, out_dir):
                skipped += 1
                logger.info("[%d/%d] %s - skipped (caption exists)", idx, len(images), name)
                continue

            tmp_path = None
            try:
                tmp_path = resize_to_temp(img, args.width, args.height, args.resize_mode)
                t0 = time.perf_counter()
                caption = cap.caption_image(
                    tmp_path, user_prompt, args.max_tokens, system_prompt=system_prompt
                )
                caption = enforce_prefix(caption, args.prefix)
                save_caption(img, caption, out_dir, overwrite=True)
                done += 1
                logger.info(
                    "[%d/%d] %s - %.1fs - %s...",
                    idx, len(images), name, time.perf_counter() - t0, caption[:70],
                )
            except Exception as e:
                failed += 1
                logger.error("[%d/%d] %s - FAILED: %s", idx, len(images), name, e)
            finally:
                if tmp_path and Path(tmp_path).exists():
                    os.unlink(tmp_path)
    except KeyboardInterrupt:
        logger.warning("Interrupted - stopping after %d captions.", done)
    finally:
        cap.unload_model()

    logger.info(
        "Done: %d captioned, %d skipped, %d failed in %.1f min",
        done, skipped, failed, (time.perf_counter() - started) / 60,
    )
    return 1 if failed else 0


def self_test() -> int:
    """Runnable check for the only non-obvious logic here - no model needed."""
    p = DEFAULT_PREFIX
    assert enforce_prefix(f"{p} stands in a gym.", p) == f"{p} stands in a gym."
    assert enforce_prefix(f"{p.lower()} stands.", p) == f"{p} stands."          # casing fixed
    assert enforce_prefix("  A man\n stands. ", p) == f"{p}, a man stands."     # spliced + flattened
    assert enforce_prefix("A man stands.", "") == "A man stands."               # prefix off
    assert enforce_prefix("", p) == p

    _, up = build_prompt(DEFAULT_TEMPLATE, p)
    assert p in up and "Describe this image in detail" in up
    assert p not in build_prompt(DEFAULT_TEMPLATE, "")[1]

    assert pick_mmproj(DEFAULT_MODEL).name.lower().startswith("mmproj")
    print("self-test OK")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Batch caption images with a GGUF Qwen-VL model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--images", default=str(DEFAULT_IMAGES), help="Folder of images")
    p.add_argument("--out", default="", help="Caption output folder (default: next to image)")
    p.add_argument("--model", default=str(DEFAULT_MODEL), help="Path to the .gguf model")
    p.add_argument("--mmproj", default="", help="Path to mmproj .gguf (default: auto-pick)")
    p.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="models_catalog key")
    p.add_argument("--prefix", default=DEFAULT_PREFIX, help="Words every caption must start with")
    p.add_argument("--template", default=DEFAULT_TEMPLATE, help="Prompt template name")
    p.add_argument("--profile", default=DEFAULT_PROFILE, help="VRAM profile")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--height", type=int, default=768)
    p.add_argument("--resize-mode", default="Adaptive image", choices=["Adaptive image", "Fit image"])
    p.add_argument("--recursive", action="store_true", help="Include subfolders")
    p.add_argument("--limit", type=int, default=0, help="Only process the first N images")
    p.add_argument("--overwrite", dest="skip_existing", action="store_false",
                   help="Re-caption images that already have a .txt")
    p.add_argument("--self-test", action="store_true", help="Run internal asserts and exit")
    p.set_defaults(skip_existing=True)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    sys.exit(self_test() if args.self_test else run_batch(args))
