"""Self-checks for the Video Reel prompt helpers. Run: .venv/Scripts/python.exe test_video_prompts.py"""
import json
import tempfile
from pathlib import Path

import captioner.prompts as prompts
from captioner.prompts import (
    VIDEO_PROMPTS,
    get_prompt_names,
    get_video_system_prompt,
    load_video_prompt_overrides,
    parse_ideas,
    resolve_video_prompt,
    save_video_prompt_override,
)


def test_simple_app_templates_removed():
    leftover = [n for n in get_prompt_names() if "Simple App" in n]
    assert not leftover, leftover


def test_video_prompts_shape():
    assert set(VIDEO_PROMPTS) == {"pose", "ideas", "reel"}
    for step, tmpl in VIDEO_PROMPTS.items():
        assert tmpl["system_prompt"].strip() and tmpl["user_prompt"].strip(), step
    assert "{duration}" in VIDEO_PROMPTS["ideas"]["user_prompt"]
    assert "|||" in VIDEO_PROMPTS["ideas"]["system_prompt"]
    for placeholder in ("{pose}", "{idea}", "{duration}"):
        assert placeholder in VIDEO_PROMPTS["reel"]["user_prompt"], placeholder


def test_resolve_video_prompt_fills_placeholders():
    system, user = resolve_video_prompt("reel", "  my rules  ", pose="He stands.", idea="He waves.", duration=7)
    assert system == "my rules"
    assert "Character: He stands." in user
    assert "Goal: He waves." in user
    assert "lasts 7 seconds" in user
    assert "{" not in user


def test_resolve_video_prompt_is_forgiving():
    # missing field stays as a visible placeholder instead of raising
    assert "{pose}" in resolve_video_prompt("reel", "r", idea="x", duration=5)[1]
    # system prompt may use placeholders too
    assert resolve_video_prompt("ideas", "Use {duration}s", duration=5)[0] == "Use 5s"
    # a lone brace typed by the user must not blow up str.format
    assert resolve_video_prompt("ideas", "lone { brace", duration=5)[0] == "lone { brace"


def test_parse_ideas():
    text = "He waves ||| Anh ấy vẫy tay\n1. He smiles ||| Anh ấy cười\n- He sits\n\n"
    assert parse_ideas(text) == [
        ("He waves", "Anh ấy vẫy tay"),
        ("He smiles", "Anh ấy cười"),
        ("He sits", ""),
    ]
    assert parse_ideas("") == []
    assert parse_ideas(None) == []
    five = "\n".join(f"idea {i} ||| ý {i}" for i in range(5))
    assert len(parse_ideas(five)) == 3


def _with_temp_override_file(fn):
    original = prompts.VIDEO_PROMPT_OVERRIDE_FILE
    with tempfile.TemporaryDirectory() as tmp:
        prompts.VIDEO_PROMPT_OVERRIDE_FILE = Path(tmp) / "prompts_override.json"
        try:
            fn(prompts.VIDEO_PROMPT_OVERRIDE_FILE)
        finally:
            prompts.VIDEO_PROMPT_OVERRIDE_FILE = original


def test_override_roundtrip():
    def body(path):
        default = VIDEO_PROMPTS["reel"]["system_prompt"]
        assert get_video_system_prompt("reel") == default
        assert load_video_prompt_overrides() == {}

        save_video_prompt_override("reel", "  custom rules  ")
        assert get_video_system_prompt("reel") == "custom rules"
        assert json.loads(path.read_text(encoding="utf-8")) == {"reel": "custom rules"}
        assert get_video_system_prompt("pose") == VIDEO_PROMPTS["pose"]["system_prompt"]

        save_video_prompt_override("reel", "")
        assert get_video_system_prompt("reel") == default
        assert not path.exists()

    _with_temp_override_file(body)


def test_override_ignores_broken_file():
    def body(path):
        path.write_text("{not json", encoding="utf-8")
        assert load_video_prompt_overrides() == {}
        path.write_text(json.dumps(["a", "list"]), encoding="utf-8")
        assert load_video_prompt_overrides() == {}
        path.write_text(json.dumps({"reel": 5, "bogus": "x", "pose": "ok"}), encoding="utf-8")
        assert load_video_prompt_overrides() == {"pose": "ok"}

    _with_temp_override_file(body)


if __name__ == "__main__":
    tests = [fn for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for fn in tests:
        fn()
        print("ok", fn.__name__)
    print(f"{len(tests)} checks passed")
