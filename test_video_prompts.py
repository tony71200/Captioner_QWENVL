"""Self-checks for the Video Reel prompt helpers. Run: .venv/Scripts/python.exe test_video_prompts.py"""
from captioner.prompts import get_prompt_names


def test_simple_app_templates_removed():
    leftover = [n for n in get_prompt_names() if "Simple App" in n]
    assert not leftover, leftover


if __name__ == "__main__":
    tests = [fn for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for fn in tests:
        fn()
        print("ok", fn.__name__)
    print(f"{len(tests)} checks passed")
