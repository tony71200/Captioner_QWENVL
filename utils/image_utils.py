"""
Image utility functions: scanning folders, validating image files.
"""
import os
from pathlib import Path
from typing import List, Set

from PIL import Image, ImageOps


# Supported image file extensions
SUPPORTED_EXTENSIONS: Set[str] = {
    ".jpg", ".jpeg", ".png", ".webp",
    ".bmp", ".tiff", ".tif", ".gif",
}

# HEIC/HEIF (iPhone photos) need pillow-heif to teach Pillow the format. The
# extensions are only advertised once the opener is registered, so scanning a
# folder never hands back a file Image.open() would choke on.
try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
    SUPPORTED_EXTENSIONS |= {".heic", ".heif"}
except ImportError:
    pass


def get_supported_extensions() -> Set[str]:
    """Return the set of supported image file extensions."""
    return SUPPORTED_EXTENSIONS.copy()


def validate_image(path: str) -> bool:
    """
    Check that path exists and has a supported image extension.
    Does NOT do deep format validation (PIL open).
    """
    p = Path(path)
    return p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS


def scan_folder(folder_path: str, recursive: bool = False,
                extensions: Set[str] = SUPPORTED_EXTENSIONS) -> List[str]:
    """
    Scan a folder for image files.

    Args:
        folder_path: Path to folder to scan
        recursive:   If True, scan subdirectories recursively
        extensions:  Lower-case suffixes to collect (default: all supported)

    Returns:
        Sorted list of absolute image file paths
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        return []

    images: List[str] = []

    if recursive:
        for root, _dirs, files in os.walk(folder):
            for fname in files:
                fpath = Path(root) / fname
                if fpath.suffix.lower() in extensions:
                    images.append(str(fpath.resolve()))
    else:
        for fpath in folder.iterdir():
            if fpath.is_file() and fpath.suffix.lower() in extensions:
                images.append(str(fpath.resolve()))

    return sorted(images)


def get_image_name(image_path: str) -> str:
    """Return just the filename (with extension) of an image path."""
    return Path(image_path).name


def normalize_resize_dimensions(width: int | float, height: int | float) -> tuple[int, int]:
    """Validate resize dimensions and return normalized integer values."""
    width = int(width)
    height = int(height)

    if width <= 0 or height <= 0:
        raise ValueError("Resize width and height must be positive integers.")
    if width % 32 != 0 or height % 32 != 0:
        raise ValueError("Resize width and height must be multiples of 32.")

    return width, height


def resize_image(image: Image.Image, width: int | float, height: int | float, mode: str) -> Image.Image:
    """
    Resize an image for captioning.

    Modes:
    - Fit image: direct resize to the target width/height
    - Adaptive image: keep aspect ratio, center on black canvas
    """
    width, height = normalize_resize_dimensions(width, height)
    source = image.convert("RGB")

    if mode == "Fit image":
        return source.resize((width, height), Image.Resampling.LANCZOS)

    if mode == "Adaptive image":
        contained = ImageOps.contain(source, (width, height), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (width, height), (0, 0, 0))
        offset_x = (width - contained.width) // 2
        offset_y = (height - contained.height) // 2
        canvas.paste(contained, (offset_x, offset_y))
        return canvas

    raise ValueError(f"Unsupported resize mode: {mode}")


# Formats the Convert tab turns into JPG/PNG.
CONVERT_EXTENSIONS: Set[str] = {".heic", ".heif", ".tif", ".tiff", ".webp"}


def convert_image(path: str, fmt: str = "JPG", quality: int = 95) -> tuple[str | None, tuple[int, int]]:
    """
    Convert one image to JPG or PNG next to the original, same file name.

    Images with transparency are saved as PNG even when JPG is requested, so the
    alpha channel is never flattened. The original file is left untouched, and an
    existing target is never overwritten: returns (None, size) in that case.
    """
    src = Path(path)
    with Image.open(src) as opened:
        icc = opened.info.get("icc_profile")
        has_alpha = opened.mode in ("RGBA", "LA", "PA") or "transparency" in opened.info
        # iPhone HEICs are stored sideways and rely on the EXIF orientation tag.
        img = ImageOps.exif_transpose(opened)
        img = img.convert("RGBA" if has_alpha else "RGB")

    ext = ".png" if fmt.upper() == "PNG" or has_alpha else ".jpg"
    out = src.with_suffix(ext)
    if out.exists():
        return None, img.size

    params = {"icc_profile": icc} if icc else {}
    if ext == ".jpg":
        img.save(out, "JPEG", quality=int(quality), **params)
    else:
        img.save(out, "PNG", **params)
    return str(out), img.size


if __name__ == "__main__":
    # Self-check: python -m utils.image_utils
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        Image.new("RGB", (40, 20), "red").save(f"{d}/a.webp")
        Image.new("RGBA", (8, 8), (0, 0, 0, 0)).save(f"{d}/b.tiff")
        Image.new("RGB", (8, 8)).save(f"{d}/c.png")
        found = scan_folder(d, extensions=CONVERT_EXTENSIONS)
        assert [Path(f).name for f in found] == ["a.webp", "b.tiff"], found
        out, size = convert_image(f"{d}/a.webp", "JPG")
        assert out.endswith("a.jpg") and size == (40, 20)
        assert convert_image(f"{d}/a.webp", "JPG")[0] is None          # never overwrites
        assert convert_image(f"{d}/b.tiff", "JPG")[0].endswith("b.png")  # alpha -> PNG
        assert Path(f"{d}/a.webp").exists()                             # original kept
    print("image_utils self-check OK")
