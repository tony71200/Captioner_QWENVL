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


def scan_folder(folder_path: str, recursive: bool = False) -> List[str]:
    """
    Scan a folder for supported image files.

    Args:
        folder_path: Path to folder to scan
        recursive:   If True, scan subdirectories recursively

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
                if fpath.suffix.lower() in SUPPORTED_EXTENSIONS:
                    images.append(str(fpath.resolve()))
    else:
        for fpath in folder.iterdir():
            if fpath.is_file() and fpath.suffix.lower() in SUPPORTED_EXTENSIONS:
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
