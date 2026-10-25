"""Self-checks for model_tester's pure parts. Run: .venv\\Scripts\\python.exe test_model_tester.py"""
import tempfile
from pathlib import Path

from PIL import Image

from models_catalog import VRAM_PROFILES
from model_tester import VramSampler, fitting_profiles, looks_like_text, make_test_image, prepare_and_test, vision_score


def test_image_and_scoring():
    with tempfile.TemporaryDirectory() as tmp:
        img = Image.open(make_test_image(Path(tmp) / "s.png"))
        assert img.size == (512, 512)
        assert img.getpixel((5, 5)) == (40, 90, 200)        # blue background
        assert img.getpixel((150, 240)) == (220, 30, 30)    # red circle
        assert img.getpixel((370, 240)) == (245, 210, 20)   # yellow square
    assert vision_score("A red circle and a yellow square on a blue background.") == 5
    assert vision_score("A dog.") == 0
    assert looks_like_text("A red circle next to a yellow square.")
    assert not looks_like_text("")
    assert not looks_like_text("!!!! #### ???? 1234 ....")


def test_fitting_profiles():
    fits = fitting_profiles(6.4)
    assert fits == {"UltraLow (4GB)": False, "LowVRAM (6–8GB)": True,
                    "NormalVRAM (12–16GB)": True, "HighVRAM (20GB+)": True}, fits
    assert fitting_profiles(None) == {}


def test_sampler_keeps_peak():
    values = iter([1.0, 5.0, 3.0] + [2.0] * 1000)
    with VramSampler(read=lambda: next(values), interval=0.01) as s:
        import time
        time.sleep(0.1)
    assert s.peak == 5.0


def test_failed_results_have_no_fits_and_conversion_errors_are_results():
    # convert() fails on the missing source before any network fetch, so this runs offline and without a GPU.
    with tempfile.TemporaryDirectory() as tmp:
        candidate = {"kind": "safetensors", "path": str(Path(tmp) / "missing.safetensors"), "name": "missing",
                     "base_repo": "Qwen/Qwen3-VL-4B-Instruct", "size": "4B", "bf16_gb": 1.0,
                     "usable": True, "reason": ""}
        entry, result = prepare_and_test(candidate, "missing-test", Path(tmp), next(iter(VRAM_PROFILES)),
                                         log=lambda *_: None)
    assert result["ok"] is False, result
    assert result["reason"].startswith("conversion failed"), result
    assert result["fits"] == {}, result


if __name__ == "__main__":
    test_image_and_scoring()
    test_fitting_profiles()
    test_sampler_keeps_peak()
    test_failed_results_have_no_fits_and_conversion_errors_are_results()
    print("test_model_tester OK")
