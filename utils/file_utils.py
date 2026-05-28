"""
File utility functions: saving captions, checking for existing files.
"""
from pathlib import Path
from typing import Optional


def get_txt_path(image_path: str, output_dir: Optional[str] = None) -> str:
    """
    Compute the .txt output path for a given image file.

    Args:
        image_path: Path to the source image
        output_dir: Optional override output directory.
                    If None, uses the same directory as the image.

    Returns:
        Absolute path to the .txt file (same stem, .txt extension)
    """
    img = Path(image_path)
    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        return str(out_dir / (img.stem + ".txt"))
    return str(img.parent / (img.stem + ".txt"))


def caption_exists(image_path: str, output_dir: Optional[str] = None) -> bool:
    """Return True if the corresponding .txt caption file already exists."""
    return Path(get_txt_path(image_path, output_dir)).exists()


def save_caption(
    image_path: str,
    caption: str,
    output_dir: Optional[str] = None,
    overwrite: bool = True,
) -> str:
    """
    Save a caption string to a .txt file with the same stem as image_path.

    Args:
        image_path: Path to the source image
        caption:    Caption text to write
        output_dir: Optional directory to write to (default: same as image)
        overwrite:  If False, skip writing if file already exists

    Returns:
        Path to the written .txt file, or empty string if skipped
    """
    txt_path = get_txt_path(image_path, output_dir)

    if not overwrite and Path(txt_path).exists():
        return ""

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(caption)

    return txt_path


def load_caption(image_path: str, output_dir: Optional[str] = None) -> Optional[str]:
    """
    Load an existing caption file for the given image, if it exists.

    Returns:
        Caption string, or None if the .txt file doesn't exist
    """
    txt_path = get_txt_path(image_path, output_dir)
    p = Path(txt_path)
    if p.exists():
        return p.read_text(encoding="utf-8")
    return None
