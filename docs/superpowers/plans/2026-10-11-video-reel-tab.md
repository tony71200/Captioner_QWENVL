# Video Reel Tab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "🎬 Video Reel" Gradio tab that chains Pose caption → Video ideas → Video reel on one image, with per-step editable system prompts persisted to `prompts_override.json` and an Export txt button.

**Architecture:** The three Simple App prompts move out of `PROMPT_TEMPLATES` (so they vanish from the Single Image / Batch dropdowns) into a new `VIDEO_PROMPTS` dict in `captioner/prompts.py`, where system prompt = editable rules and user prompt = fixed placeholders (`{duration}`, `{idea}`, `{pose}`). `app.py` gets pure handler functions (`vr_pose`, `vr_ideas`, `vr_reel`, `vr_export`) that call the already-loaded `_captioner`, plus a `_vr_section` helper that builds each numbered block with its ⚙ toggle. Overrides are a flat JSON file read on demand.

**Tech Stack:** Python 3.13, Gradio 6.28 (`.venv`), existing `BaseCaptioner.caption_image(image_path, user_prompt, max_new_tokens, system_prompt)` on both HF and GGUF backends. Tests are plain `assert` scripts, no pytest.

**Spec:** `docs/superpowers/specs/2026-10-11-video-reel-tab-design.md`

## Global Constraints

- Python interpreter for every command: `.venv/Scripts/python.exe` (run from the project root `D:\001_Personal_Proj\Captioner_QWENVL`).
- Gradio 6: hidden-but-present components use `visible="hidden"` elsewhere in `app.py`; for the gear panel use plain `visible=False/True` toggled by a `gr.State(bool)`.
- Gradio 6 removed `Textbox(show_copy_button=...)`; the copy icon is `Textbox(buttons=["copy"])` (verified against the installed 6.28.0 signature).
- Dropdown choices are `(label, value)` tuples; the value is the English idea sentence.
- Idea line format demanded from the model: `<English sentence> ||| <Vietnamese description>`.
- `duration` is coerced with `max(1, int(value or 5))` before it reaches any prompt.
- Export file: `~/Desktop/video_reel_<YYYYmmdd_HHMMSS>.txt`, UTF-8, sections joined by `\n\n`, empty sections skipped.
- Override file: `<project root>/prompts_override.json`, git-ignored, keys are `pose` / `ideas` / `reel`, values are system prompt strings.
- Commit after every task. Commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- `batch_caption.py --self-test` must print `self-test OK` from Task 1 onward (it currently fails on the raw templates).

---

## File map

| File | Change | Responsibility |
|------|--------|----------------|
| `captioner/prompts.py` | Modify | Remove 3 Simple App templates and `raw_output` plumbing. Add `VIDEO_PROMPTS`, `resolve_video_prompt`, `parse_ideas`, override load/save/get. |
| `test_video_prompts.py` | Create | Plain-assert checks for everything in `prompts.py` and `utils/file_utils.py` added by this plan. |
| `utils/file_utils.py` | Modify | Add `build_video_export(pose, idea, reel)`. |
| `app.py` | Modify | Import new names, add `vr_*` handlers, `_vr_section` helper, the new tab, Help text. |
| `README.md` | Modify | Drop 3 template rows, add "Video Reel tab" section. |
| `.gitignore` | Modify | Add `prompts_override.json`. |

---

### Task 1: Remove the Simple App templates and the `raw_output` plumbing

**Files:**
- Modify: `captioner/prompts.py:230-290` (the 3 template dicts), `captioner/prompts.py` `_with_output_rules` and `resolve_prompt`
- Create: `test_video_prompts.py`

**Interfaces:**
- Consumes: `captioner.prompts.get_prompt_names()` (existing).
- Produces: `PROMPT_TEMPLATES` without any "Simple App" entry; `_with_output_rules(user_prompt, training_caption=False)` (no `raw` parameter).

- [ ] **Step 1: Create the test script with the first check**

Create `test_video_prompts.py`:

```python
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/Scripts/python.exe test_video_prompts.py`
Expected: `AssertionError: ['Video Reel (Simple App)', 'Video Ideas (Simple App)', 'Pose Caption (Simple App)']`

- [ ] **Step 3: Delete the three templates**

In `captioner/prompts.py`, delete everything from the comment line
`    # Simple App (Wan2GP) video prompts. "raw_output": ...` (line 230) through the closing `    }` of the
"Pose Caption (Simple App)" dict (line 289), so the list ends as it did before commit 5fc7492:

```python
            "cooking in a bright modern kitchen. ect\n"
            "Output only the final prompt text."
        ),
    }]
```

- [ ] **Step 4: Remove the `raw` parameter**

Replace `_with_output_rules`:

```python
def _with_output_rules(user_prompt: str, training_caption: bool = False) -> str:
    """Append the appropriate output contract once to a resolved user prompt."""
    rules = TRAINING_OUTPUT_RULES if training_caption else OUTPUT_RULES
    if not user_prompt or rules.strip() in user_prompt:
        return user_prompt
    return user_prompt.rstrip() + rules
```

In `resolve_prompt`, delete both `raw=bool(... .get("raw_output")),` lines so the two calls read:

```python
        return system_prompt, _with_output_rules(
            _format(custom_prompt, name=name, underwear=underwear),
            training_caption=bool(tmpl and tmpl.get("training_caption")),
        )
```

and

```python
    return system_prompt, _with_output_rules(
        user_prompt,
        training_caption=bool(tmpl.get("training_caption")),
    )
```

- [ ] **Step 5: Run both checks**

Run: `.venv/Scripts/python.exe test_video_prompts.py`
Expected: `ok test_simple_app_templates_removed` and `1 checks passed`

Run: `.venv/Scripts/python.exe batch_caption.py --self-test`
Expected: last line `self-test OK` (this was failing before with `AssertionError: Video Reel (Simple App)`).

Run: `grep -n "raw_output\|raw=" captioner/prompts.py`
Expected: no output.

- [ ] **Step 6: Commit**

```bash
git add captioner/prompts.py test_video_prompts.py
git commit -m "Remove Simple App templates from the caption dropdowns

They move to a dedicated Video Reel tab next; the raw_output flag has no
user left, and the batch self-test passes again.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: `VIDEO_PROMPTS`, `resolve_video_prompt`, `parse_ideas`

**Files:**
- Modify: `captioner/prompts.py` (add after the `PROMPT_TEMPLATES` list, before `PEOPLE_MARKER`; also harden `_format`)
- Modify: `test_video_prompts.py`

**Interfaces:**
- Produces:
  - `VIDEO_PROMPTS: dict[str, dict[str, str]]` with keys `"pose"`, `"ideas"`, `"reel"`, each `{"system_prompt": str, "user_prompt": str}`.
  - `resolve_video_prompt(step: str, system_prompt: str, **fields) -> tuple[str, str]` returns `(system, user)`, both formatted with `_format` (unknown placeholders kept, a lone `{` leaves the text untouched).
  - `parse_ideas(text: str, limit: int = 3) -> list[tuple[str, str]]`.

- [ ] **Step 1: Add the failing tests**

In `test_video_prompts.py`, replace the single import line at the top with the one below, and add the
four test functions above the `if __name__` block:

```python
from captioner.prompts import VIDEO_PROMPTS, get_prompt_names, parse_ideas, resolve_video_prompt


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
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe test_video_prompts.py`
Expected: `ImportError: cannot import name 'VIDEO_PROMPTS'`

- [ ] **Step 3: Add `VIDEO_PROMPTS` after the `PROMPT_TEMPLATES` list**

Insert directly after the `    }]` that closes `PROMPT_TEMPLATES` and before the
`# Rules appended to every template` comment:

```python
# Simple App (Wan2GP) Video Reel tab. Each step is system prompt (rules, editable in
# the UI) + user prompt (fixed, only placeholders). None of these get OUTPUT_RULES:
# the video model reads the multi-line output verbatim.
VIDEO_PROMPTS = {
    "pose": {
        "system_prompt": (
            "You are an AI vision model describing the end pose of a video segment for an image-to-video model.\n"
            "Describe the provided image in ONE English sentence: the person's pose, where they are in the frame, "
            "what they wear or hold, and the camera framing (distance and angle).\n"
            "No commentary, no list, no meta-text. Output only that sentence."
        ),
        "user_prompt": "Describe the provided image.",
    },
    "ideas": {
        "system_prompt": (
            "You are a creative director for short vertical social videos. Look at the image and propose exactly 3 "
            "different short video ideas that start from this exact image.\n"
            "Each idea must work as one continuous shot of the requested duration in the same place: no cuts, no change "
            "of location, lighting or time of day. Keep the same person, face, hairstyle and setting, and prefer small "
            "natural actions over big ones. The person keeps wearing exactly what they wear in the image.\n"
            "Output rules:\n"
            "Write exactly 3 lines, one idea per line, no numbering, no bullets, no extra text.\n"
            "Each line is: <one English sentence describing the idea> ||| <the same idea described in Vietnamese>.\n"
            "The separator is exactly three vertical bars with a space on each side."
        ),
        "user_prompt": "The clip lasts {duration} seconds.",
    },
    "reel": {
        "system_prompt": (
            "You are an AI vision model writing a prompt for an image-to-video model. The image is the first frame of a clip; "
            "analyze it, the character description and the user's goal.\n"
            "Describe ONE continuous shot that starts exactly from the image: no cuts, no scene change, no change of location, "
            "lighting, weather or time of day.\n"
            "The person must stay the same person for the whole video: same face, hairstyle, hair color, body, skin tone, outfit "
            "and accessories as in the image and the character description, unless the goal says they put on, take off or pick up "
            "something. Never describe a different look.\n"
            "Keep the motion natural and modest so the face stays recognizable. The camera is static or moves slowly.\n"
            "The camera keeps the same distance and framing as the image: no zoom, no push-in, no close-up on the face; "
            "the person may walk closer by themselves.\n"
            "Each line describes exactly one main movement of the person and one concrete camera move (static, pan, tilt, dolly, "
            "slow push): say how things move, do not re-describe the image.\n"
            "Build the motion progressively: start subtle and grow through the clip.\n"
            "The camera is static on the first line and on the last line.\n"
            "Output rules:\n"
            "Write exactly one line for each second from 0 to the last whole second before the clip ends, in order. "
            "No other lines, none at or after the clip end.\n"
            "Each line must begin with \"(at X seconds:\" where X is that second number, followed by a dense, comma-separated "
            "visual description of at most 25 words, all inside a single set of parentheses.\n"
            "Do not include commentary, explanations, or meta-text."
        ),
        "user_prompt": "Character: {pose}\nGoal: {idea}\nThe clip lasts {duration} seconds.",
    },
}
```

- [ ] **Step 4: Harden `_format` and add the two helpers**

Replace the existing `_format`:

```python
def _format(text: str, **kwargs) -> str:
    if not text:
        return text
    try:
        return text.format_map(_SafeDict(**kwargs))
    except ValueError:  # a lone "{" or "}" typed into an editable prompt
        return text
```

Add after `resolve_prompt` (end of file):

```python
# ── Video Reel tab helpers ────────────────────────────────────────────────────
_IDEA_PREFIX = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s*")


def resolve_video_prompt(step: str, system_prompt: str, **fields) -> tuple[str, str]:
    """(system, user) for one Video Reel step. `fields`: duration, idea, pose."""
    user = _format(VIDEO_PROMPTS[step]["user_prompt"], **fields)
    return _format((system_prompt or "").strip(), **fields), user


def parse_ideas(text: str, limit: int = 3) -> list[tuple[str, str]]:
    """'EN ||| VI' lines → [(en, vi)]. Numbering/bullets dropped, missing VI → ''."""
    ideas = []
    for raw in (text or "").splitlines():
        line = _IDEA_PREFIX.sub("", raw).strip()
        if not line:
            continue
        en, _, vi = line.partition("|||")
        ideas.append((en.strip(), vi.strip()))
    return ideas[:limit]
```

- [ ] **Step 5: Run the checks**

Run: `.venv/Scripts/python.exe test_video_prompts.py`
Expected: five `ok ...` lines and `5 checks passed`

Run: `.venv/Scripts/python.exe batch_caption.py --self-test`
Expected: `self-test OK`

- [ ] **Step 6: Commit**

```bash
git add captioner/prompts.py test_video_prompts.py
git commit -m "Add VIDEO_PROMPTS with resolve_video_prompt and parse_ideas

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: System prompt overrides persisted to `prompts_override.json`

**Files:**
- Modify: `captioner/prompts.py` (imports at top; functions after `parse_ideas`)
- Modify: `test_video_prompts.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `VIDEO_PROMPTS` from Task 2.
- Produces:
  - `VIDEO_PROMPT_OVERRIDE_FILE: Path` (module global, tests re-point it).
  - `load_video_prompt_overrides() -> dict[str, str]`
  - `save_video_prompt_override(step: str, system_prompt: str) -> None` (empty text removes the key; no keys left removes the file).
  - `get_video_system_prompt(step: str) -> str`

- [ ] **Step 1: Add the failing tests**

In `test_video_prompts.py`, replace the import block at the top with the one below (the stdlib
imports and the `captioner.prompts` imports), then add the helper and two tests above the `if __name__` block:

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe test_video_prompts.py`
Expected: `ImportError: cannot import name 'get_video_system_prompt'`

- [ ] **Step 3: Implement**

At the top of `captioner/prompts.py` change the imports to:

```python
import json
import logging
import re
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)
```

Append after `parse_ideas`:

```python
VIDEO_PROMPT_OVERRIDE_FILE = Path(__file__).resolve().parent.parent / "prompts_override.json"


def load_video_prompt_overrides() -> dict[str, str]:
    """{step: system_prompt} saved from the UI; a missing or broken file counts as empty."""
    try:
        data = json.loads(VIDEO_PROMPT_OVERRIDE_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        logger.warning("Ignoring unreadable %s: %s", VIDEO_PROMPT_OVERRIDE_FILE.name, e)
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if k in VIDEO_PROMPTS and isinstance(v, str) and v.strip()}


def save_video_prompt_override(step: str, system_prompt: str) -> None:
    """Persist one step's system prompt; empty text restores the default (drops the key)."""
    overrides = load_video_prompt_overrides()
    text = (system_prompt or "").strip()
    if text:
        overrides[step] = text
    else:
        overrides.pop(step, None)
    if overrides:
        VIDEO_PROMPT_OVERRIDE_FILE.write_text(
            json.dumps(overrides, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    elif VIDEO_PROMPT_OVERRIDE_FILE.exists():
        VIDEO_PROMPT_OVERRIDE_FILE.unlink()


def get_video_system_prompt(step: str) -> str:
    return load_video_prompt_overrides().get(step) or VIDEO_PROMPTS[step]["system_prompt"]
```

- [ ] **Step 4: Git-ignore the override file**

The current `.gitignore` has no trailing newline, so prepend one:

```bash
printf '\nprompts_override.json\n' >> .gitignore
```

Run: `git check-ignore prompts_override.json`
Expected: prints `prompts_override.json`

- [ ] **Step 5: Run the checks**

Run: `.venv/Scripts/python.exe test_video_prompts.py`
Expected: `7 checks passed`

- [ ] **Step 6: Commit**

```bash
git add captioner/prompts.py test_video_prompts.py .gitignore
git commit -m "Persist Video Reel system prompt overrides in prompts_override.json

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: `build_video_export`

**Files:**
- Modify: `utils/file_utils.py` (append at end)
- Modify: `test_video_prompts.py`

**Interfaces:**
- Produces: `build_video_export(pose: str, idea: str, reel: str) -> str`.

- [ ] **Step 1: Add the failing test**

Add to `test_video_prompts.py`:

```python
from utils.file_utils import build_video_export


def test_build_video_export():
    assert build_video_export("P", "I", "R") == "P\n\nIdea: I\n\nR"
    assert build_video_export("", " I ", " R\n") == "Idea: I\n\nR"
    assert build_video_export("P", "", "") == "P"
    assert build_video_export("", "", "") == ""
    assert build_video_export(None, None, None) == ""
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe test_video_prompts.py`
Expected: `ImportError: cannot import name 'build_video_export'`

- [ ] **Step 3: Implement**

Append to `utils/file_utils.py`:

```python
def build_video_export(pose: str, idea: str, reel: str) -> str:
    """Video Reel tab export: pose caption, 'Idea: …', reel lines, blank-line separated; empty parts skipped."""
    idea = (idea or "").strip()
    parts = [(pose or "").strip(), f"Idea: {idea}" if idea else "", (reel or "").strip()]
    return "\n\n".join(p for p in parts if p)
```

- [ ] **Step 4: Run the checks**

Run: `.venv/Scripts/python.exe test_video_prompts.py`
Expected: `8 checks passed`

- [ ] **Step 5: Commit**

```bash
git add utils/file_utils.py test_video_prompts.py
git commit -m "Add build_video_export for the Video Reel txt file

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Video Reel tab in `app.py` (handlers + UI)

**Files:**
- Modify: `app.py:18-33` (imports), `app.py` after `convert_tags_file_to_caption_file` and before `on_prompt_change` (~line 795: handlers), `app.py` just before `with gr.Blocks(` (~line 840: `_vr_section`), `app.py` after the `text2Text` tab and before `# Tab 5 — Models` (~line 1253: the tab).

**Interfaces:**
- Consumes: `VIDEO_PROMPTS`, `VIDEO_PROMPT_OVERRIDE_FILE`, `get_video_system_prompt`, `parse_ideas`, `resolve_video_prompt`, `save_video_prompt_override` (Tasks 2–3); `build_video_export` (Task 4); existing `_captioner`, `_badge`, `_save_resized_temp_image`.
- Produces: Gradio handlers `vr_pose`, `vr_ideas`, `vr_reel`, `vr_export`, `vr_toggle_gear`, `vr_save_system`, `vr_restore_system`; UI helper `_vr_section(title, step, run_label) -> (system_box, run_btn)`.

- [ ] **Step 1: Extend the imports**

Replace the `from captioner.prompts import (...)` block with:

```python
from captioner.prompts import (
    PROMPT_TEMPLATES,
    VIDEO_PROMPTS,
    VIDEO_PROMPT_OVERRIDE_FILE,
    get_prompt_default_name,
    get_prompt_name_label,
    get_prompt_names,
    get_video_system_prompt,
    parse_ideas,
    prompt_needs_name,
    prompt_needs_underwear,
    resolve_prompt,
    resolve_video_prompt,
    save_video_prompt_override,
    UNDERWEAR_NAMES,
)
```

and the file_utils line with:

```python
from utils.file_utils import save_caption, caption_exists, normalize_caption, build_video_export
```

- [ ] **Step 2: Add the handlers**

Insert before `def on_prompt_change(name):`:

```python
# ── Video Reel (Simple App) tab ──────────────────────────────────────────────
def _vr_duration(value) -> int:
    try:
        return max(1, int(value or 5))
    except (TypeError, ValueError):
        return 5


def _vr_prepare(image_obj, resize_mode, resize_width, resize_height):
    """Shared guards. Returns (tmp_path, None) or (None, error_badge)."""
    if _captioner is None:
        return None, _badge("Load a model first.", "warning")
    if image_obj is None:
        return None, _badge("Upload an image.", "warning")
    return _save_resized_temp_image(image_obj, resize_mode, resize_width, resize_height), None


def _vr_cleanup(tmp_path):
    if tmp_path and Path(tmp_path).exists():
        os.unlink(tmp_path)


def _vr_generate(step, system_prompt, tmp_path, max_tokens, **fields) -> str:
    system, user = resolve_video_prompt(step, system_prompt, **fields)
    return _captioner.caption_image(tmp_path, user, int(max_tokens), system_prompt=system or None).strip()


def _vr_ideas_update(ideas):
    """Dropdown update for [(en, vi)]: label shows both, value is the English sentence, first one selected."""
    return gr.update(choices=[(f"{en} | {vi}" if vi else en, en) for en, vi in ideas], value=ideas[0][0])


def vr_pose(image_obj, pose_system, max_tokens, resize_mode, resize_width, resize_height):
    tmp_path, err = _vr_prepare(image_obj, resize_mode, resize_width, resize_height)
    if err:
        return gr.update(), err
    try:
        text = _vr_generate("pose", pose_system, tmp_path, max_tokens)
        return " ".join(text.split()), _badge("Pose caption generated.", "success")
    except Exception as e:
        return gr.update(), _badge(f"Error: {e}", "error")
    finally:
        _vr_cleanup(tmp_path)


def vr_ideas(image_obj, duration, ideas_system, max_tokens, resize_mode, resize_width, resize_height):
    tmp_path, err = _vr_prepare(image_obj, resize_mode, resize_width, resize_height)
    if err:
        return gr.update(), err
    try:
        raw = _vr_generate("ideas", ideas_system, tmp_path, max_tokens, duration=_vr_duration(duration))
        ideas = parse_ideas(raw)
        if not ideas:
            return gr.update(), _badge("The model returned no ideas. Try again.", "warning")
        return _vr_ideas_update(ideas), _badge(f"{len(ideas)} ideas suggested.", "success")
    except Exception as e:
        return gr.update(), _badge(f"Error: {e}", "error")
    finally:
        _vr_cleanup(tmp_path)


def vr_reel(image_obj, duration, pose_text, idea_custom, idea_choice,
            pose_system, ideas_system, reel_system, max_tokens, resize_mode, resize_width, resize_height):
    """Step 3. Runs step 1 / step 2 first when their boxes are empty and fills them in."""
    tmp_path, err = _vr_prepare(image_obj, resize_mode, resize_width, resize_height)
    if err:
        return gr.update(), gr.update(), gr.update(), err
    pose_out, ideas_out, auto = gr.update(), gr.update(), []
    try:
        secs = _vr_duration(duration)
        pose = (pose_text or "").strip()
        if not pose:
            pose = " ".join(_vr_generate("pose", pose_system, tmp_path, max_tokens).split())
            pose_out, auto = pose, auto + ["pose"]
        idea = (idea_custom or "").strip() or (idea_choice or "").strip()
        if not idea:
            ideas = parse_ideas(_vr_generate("ideas", ideas_system, tmp_path, max_tokens, duration=secs))
            if not ideas:
                return pose_out, ideas_out, gr.update(), _badge(
                    "The model returned no ideas; write one in the textbox.", "warning")
            ideas_out, idea, auto = _vr_ideas_update(ideas), ideas[0][0], auto + ["ideas → idea 1"]
        reel = _vr_generate("reel", reel_system, tmp_path, max_tokens, pose=pose, idea=idea, duration=secs)
        reel = "\n".join(line for line in reel.splitlines() if line.strip())
        note = f" (auto: {', '.join(auto)})" if auto else ""
        return pose_out, ideas_out, reel, _badge(f"Reel generated{note}.", "success")
    except Exception as e:
        return pose_out, ideas_out, gr.update(), _badge(f"Error: {e}", "error")
    finally:
        _vr_cleanup(tmp_path)


def vr_export(pose_text, idea_custom, idea_choice, reel_text):
    idea = (idea_custom or "").strip() or (idea_choice or "").strip()
    content = build_video_export(pose_text, idea, reel_text)
    if not content:
        return _badge("Nothing to export.", "warning")
    try:
        desktop = Path.home() / "Desktop"
        desktop.mkdir(exist_ok=True)
        out = desktop / f"video_reel_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        out.write_text(content, encoding="utf-8")
        return _badge(f"Saved → {out}", "success")
    except Exception as e:
        return _badge(f"Export failed: {e}", "error")


def vr_toggle_gear(visible):
    return (not visible), gr.update(visible=not visible)


def vr_save_system(step, text):
    try:
        save_video_prompt_override(step, text)
        return _badge(f"Saved to {VIDEO_PROMPT_OVERRIDE_FILE.name}.", "success")
    except Exception as e:
        return _badge(f"Save failed: {e}", "error")


def vr_restore_system(step):
    try:
        save_video_prompt_override(step, "")
        return VIDEO_PROMPTS[step]["system_prompt"], _badge("Default restored.", "success")
    except Exception as e:
        return gr.update(), _badge(f"Restore failed: {e}", "error")
```

- [ ] **Step 3: Syntax check**

Run: `.venv/Scripts/python.exe -m py_compile app.py && echo compiled`
Expected: `compiled`

- [ ] **Step 4: Add the section helper**

Insert directly above `with gr.Blocks(title="QwenVL Image Captioner", css=None) as demo:`:

```python
def _vr_section(title: str, step: str, run_label: str):
    """One numbered block of the Video Reel tab: header + ⚙ toggle, hidden system-prompt editor, ▶ button.

    Must be called inside the gr.Blocks context. `step` is bound as a parameter so the
    lambdas below do not suffer late binding.
    """
    with gr.Row():
        gr.Markdown(f"### {title}")
        gear = gr.Button("⚙", size="sm", scale=0, min_width=48)
    gear_open = gr.State(False)
    with gr.Column(visible=False) as gear_panel:
        system_box = gr.Textbox(label="System prompt", lines=10, value=get_video_system_prompt(step))
        with gr.Row():
            save_btn = gr.Button("💾 Save", size="sm")
            restore_btn = gr.Button("↺ Restore default", size="sm")
        gear_status = gr.HTML()
    run_btn = gr.Button(run_label, variant="primary")
    gear.click(vr_toggle_gear, [gear_open], [gear_open, gear_panel], queue=False)
    save_btn.click(lambda text: vr_save_system(step, text), [system_box], [gear_status], queue=False)
    restore_btn.click(lambda: vr_restore_system(step), None, [system_box, gear_status], queue=False)
    return system_box, run_btn
```

- [ ] **Step 5: Add the tab**

Insert after the `t2t_file_run.click(...)` call that ends the `text2Text` tab and before the
`#  Tab 5 — Models` banner comment:

```python
    # ════════════════════════════════════════════════════════════════════════
    #  Tab — Video Reel (Simple App / Wan2GP)
    # ════════════════════════════════════════════════════════════════════════
    with gr.Tab("🎬 Video Reel"):
        gr.Markdown(
            "Three chained steps on one image for the **Reels** tab of Simple App (Wan2GP). "
            "**Generate reel** runs steps 1 and 2 by itself when their boxes are empty. "
            "⚙ opens each step's system prompt; edits apply at once, **Save** keeps them for next time."
        )
        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                vr_img = gr.Image(type="pil", label="Upload Image", height=340)
                vr_duration = gr.Number(value=5, precision=0, minimum=1, label="Video duration (seconds)")
                vr_tokens = gr.Slider(64, 1024, value=512, step=64, label="Max New Tokens")
                with gr.Accordion("Resize options", open=False):
                    vr_resize_mode = gr.Dropdown(
                        choices=["Adaptive image", "Fit image"], value="Adaptive image", label="Resize Mode"
                    )
                    with gr.Row():
                        vr_resize_width = gr.Slider(32, 2048, value=512, step=32, label="Resize Width")
                        vr_resize_height = gr.Slider(32, 2048, value=512, step=32, label="Resize Height")
                vr_export_btn = gr.Button("💾 Export txt", variant="secondary")
                vr_export_status = gr.HTML()
            with gr.Column(scale=2):
                vr_pose_system, vr_pose_btn = _vr_section("1) Pose caption", "pose", "▶ Analyze character")
                vr_pose_out = gr.Textbox(label="Character / pose caption", lines=3,
                                         interactive=True, buttons=["copy"])
                vr_pose_status = gr.HTML()

                vr_ideas_system, vr_ideas_btn = _vr_section("2) Video ideas", "ideas", "▶ Suggest 3 ideas")
                vr_idea_choice = gr.Dropdown(choices=[], value=None, label="Ideas (EN | VI)",
                                             info="The first idea is selected by default")
                vr_idea_custom = gr.Textbox(label="Your own description (overrides dropdown)", lines=2,
                                            placeholder="Write your own idea in English...")
                vr_ideas_status = gr.HTML()

                vr_reel_system, vr_reel_btn = _vr_section("3) Video reel", "reel", "▶ Generate reel")
                vr_reel_out = gr.Textbox(label="Reel prompt, one line per second", lines=8,
                                         interactive=True, buttons=["copy"])
                vr_reel_status = gr.HTML()

        vr_resize = [vr_resize_mode, vr_resize_width, vr_resize_height]
        vr_pose_btn.click(vr_pose, [vr_img, vr_pose_system, vr_tokens, *vr_resize],
                          [vr_pose_out, vr_pose_status])
        vr_ideas_btn.click(vr_ideas, [vr_img, vr_duration, vr_ideas_system, vr_tokens, *vr_resize],
                           [vr_idea_choice, vr_ideas_status])
        vr_reel_btn.click(vr_reel,
                          [vr_img, vr_duration, vr_pose_out, vr_idea_custom, vr_idea_choice,
                           vr_pose_system, vr_ideas_system, vr_reel_system, vr_tokens, *vr_resize],
                          [vr_pose_out, vr_idea_choice, vr_reel_out, vr_reel_status])
        vr_export_btn.click(vr_export, [vr_pose_out, vr_idea_custom, vr_idea_choice, vr_reel_out],
                            [vr_export_status])
```

- [ ] **Step 6: Launch and verify in the browser**

Start the dev server with the `captioner` entry of `.claude/launch.json` (preview_start `{name: "captioner"}`; it runs `.venv/Scripts/python.exe app.py --llm-dir llm` on port 7860). Then check, in order:

1. The tab strip shows `🎬 Video Reel` between `text2Text` and `📦 Model Library`; the Single Image "Prompt Template" dropdown no longer lists any "(Simple App)" entry.
2. On the Video Reel tab, click ⚙ next to "1) Pose caption": the System prompt textbox appears with the default pose rules; click ⚙ again: it hides.
3. Open ⚙ on "3) Video reel", append the line `TEST OVERRIDE` to the system prompt, click 💾 Save. Expected: badge `Saved to prompts_override.json.`, and `prompts_override.json` appears in the project root containing `"reel"`. Click ↺ Restore default. Expected: textbox loses `TEST OVERRIDE`, badge `Default restored.`, file is gone.
4. With no model loaded, click ▶ Generate reel with no image. Expected badge: `Load a model first.` Upload any image and click again. Expected: still `Load a model first.` (model guard comes first).
5. Click 💾 Export txt with everything empty. Expected: `Nothing to export.` Type `hello` in the pose box and export. Expected: `Saved → …\Desktop\video_reel_<ts>.txt`; the file contains exactly `hello`.
6. Only if a GGUF model is available under the `llm` dir: load it on the Setup tab, upload a photo of a person, click ▶ Generate reel with steps 1 and 2 empty. Expected: pose box fills, dropdown gets up to 3 `EN | VI` entries with the first selected, reel box shows `(at 0 seconds: …)` lines, badge `Reel generated (auto: pose, ideas → idea 1).` Then type a sentence in "Your own description" and generate again: badge is just `Reel generated.` and the dropdown is unchanged.
7. `read_console_messages` / `preview_logs` show no Python traceback.

Fix anything that fails before committing. If no model is available, note that step 6 was skipped in the commit message body.

- [ ] **Step 7: Run the checks and commit**

Run: `.venv/Scripts/python.exe test_video_prompts.py` → `8 checks passed`
Run: `.venv/Scripts/python.exe batch_caption.py --self-test` → `self-test OK`

```bash
git add app.py
git commit -m "Add Video Reel tab: pose caption, ideas, reel chain with editable system prompts

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: README and in-app Help

**Files:**
- Modify: `README.md:335-348` (Prompt Templates table) and add a section after it
- Modify: `app.py` Help tab, the `### Prompt Templates` table (~line 1337)

**Interfaces:** none (docs only).

- [ ] **Step 1: README table and new section**

In `README.md`, delete these three rows from the Prompt Templates table:

```
| Video Reel (Simple App) | ... |
| Video Ideas (Simple App) | ... |
| Pose Caption (Simple App) | ... |
```

Then, after the `---` that follows the table, insert:

```markdown
## Video Reel tab (Simple App / Wan2GP)

Three chained steps on one image. Each step has its own ▶ button and a ⚙ button that opens its editable system prompt.

| Step | Output |
|------|--------|
| 1) Pose caption | One English sentence: pose, position in frame, clothing or props, camera framing. Locks the character for Wan. |
| 2) Video ideas | 3 one-shot ideas returned as `English ||| Tiếng Việt` lines and shown in a dropdown (first one selected). The textbox below overrides the dropdown. |
| 3) Video reel | `(at N seconds: …)` lines for the Reels tab. Uses step 1 as `Character:` and step 2 (textbox, else dropdown) as `Goal:`. If either is empty it runs that step first and fills its box. |

**Export txt** writes `Desktop/video_reel_<timestamp>.txt`: pose caption, blank line, `Idea: …`, blank line, reel lines. Empty parts are skipped.

Edited system prompts apply immediately. **Save** stores them in `prompts_override.json` in the project root (git-ignored) so they survive a restart; **Restore default** removes the override.

---
```

- [ ] **Step 2: Help tab**

In `app.py`, inside the `📖 Help` tab Markdown, directly after the `| Custom | Use only the textbox below |` row, insert:

```markdown

### Video Reel tab (Simple App / Wan2GP)
- **1) Pose caption** → one sentence that locks the character. **2) Video ideas** → 3 `EN | VI` ideas in a dropdown, or type your own below it. **3) Video reel** → `(at N seconds: …)` lines; it runs steps 1 and 2 first when their boxes are empty.
- ⚙ opens each step's system prompt. Edits apply at once; **Save** keeps them in `prompts_override.json`, **Restore default** drops them.
- **Export txt** saves pose + `Idea:` + reel to `Desktop/video_reel_<timestamp>.txt`.
```

- [ ] **Step 3: Verify**

Run: `grep -n "Simple App" README.md`
Expected: only lines inside the new "Video Reel tab" section (no table rows ending in `the name box holds the duration`).

Run: `.venv/Scripts/python.exe -m py_compile app.py && echo compiled`
Expected: `compiled`

Reload the preview, open `📖 Help`, confirm the new section renders.

- [ ] **Step 4: Commit**

```bash
git add README.md app.py
git commit -m "Document the Video Reel tab in README and Help

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Self-review notes

- Spec coverage: layout (Task 5 step 5), prompt split + removal from dropdowns (Tasks 1–2), `|||` ideas format + dropdown tuples (Tasks 2, 5), chain fallback with auto note (Task 5 `vr_reel`), gear toggle / live edit / Save / Restore (Task 5), override file + gitignore (Task 3), export format + Desktop path (Tasks 4–5), duration coercion (Task 5 `_vr_duration`), error handling via badges (Task 5), README + Help (Task 6), tests (Tasks 1–4), batch self-test green (Task 1).
- Names used across tasks: `VIDEO_PROMPTS`, `VIDEO_PROMPT_OVERRIDE_FILE`, `resolve_video_prompt`, `parse_ideas`, `load_video_prompt_overrides`, `save_video_prompt_override`, `get_video_system_prompt`, `build_video_export` — identical in definition, tests, and `app.py` imports.
- Deviation from spec wording: `resolve_video_prompt` formats the system prompt too (SafeDict, so plain text is unchanged). This lets an edited system prompt mention `{duration}`; the spec's "returns `system_prompt.strip()`" still holds for prompts without placeholders.
