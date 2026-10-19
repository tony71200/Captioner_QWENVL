# Prompt Video Minimax H3 Tab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Gradio tab "🎞️ Prompt Video Minimax H3" that turns a character picture (+ up to 3 role-tagged extra pictures) and a story into the two MiniMax H3 prompt parts, with a 3-idea → detailed-story path when the story is empty.

**Architecture:** Pure text logic (shot plans, picture references, output parsing, idea parsing, file paths, prompt store) lives in a new `captioner/h3_prompts.py` with its editable prompts in `captioner/system_prompt.json`. The two captioner backends gain multi-image input and an optional `temperature`. `app.py` gets thin handlers (`h3_*`) and the tab UI, reusing `_badge`, `_save_resized_temp_image`, `_vr_cleanup`, `vr_toggle_gear` and `captioner.prompts._format`.

**Tech Stack:** Python 3 (venv `D:\Comfy\.venv`), Gradio 6.14, Pillow, tkinter (stdlib), llama-cpp-python / transformers backends already in the repo.

**Spec:** `docs/superpowers/specs/2026-10-19-prompt-video-minimax-h3-design.md`

## Global Constraints

- Python: always `D:\Comfy\.venv\Scripts\python.exe` (the only venv; `run_independent_env.bat` uses it).
- No new dependencies. No import from, or file copied out of, ComfyUI (`H:\002_code\Comfy\...`).
- Max 12 s per output shot; 12 < T ≤ 15 → compressed to 12 s; T > 15 → split into `ceil(T/12)` integer shots summing to T.
- `temperature=None` must leave every existing tab's sampling unchanged (HF greedy; GGUF `temperature` 0.6 image / 0.4 text, `seed=1`).
- `captioner/system_prompt.json` is committed, edited by ⚙ Save, independent of `prompts_override.json`.
- Saved files: `<Picture 1 folder>/<stem>_H3_<sfw|nsfw>.txt` and `..._ideas.txt`; no Picture 1 path → `~/Desktop/h3_<YYYYmmdd_HHMMSS>_<sfw|nsfw>[_ideas].txt`; utf-8, overwrite.
- Logger: `[HH:MM:SS] msg` lines, last 200 kept, each also `logger.info`.
- Tests are plain `assert` scripts with a `__main__` runner, same style as `test_video_prompts.py`.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## File Structure

| File | Responsibility |
|------|----------------|
| `captioner/h3_prompts.py` (new) | Pure helpers: shot plans, picture references, output split/check, idea parse/format, output paths, prompt store. No gradio/torch. |
| `captioner/system_prompt.json` (new) | 7 editable steps: `bible`, `ideas_sfw/nsfw`, `expand_sfw/nsfw`, `shots_sfw/nsfw`, each `{system, user, temperature, max_tokens}`. |
| `captioner/base.py` | Abstract signatures gain `image_path: str | list[str]` and `temperature`. |
| `captioner/gguf_captioner.py` | `_user_content`, `_sampling` helpers; multi-image + temperature. |
| `captioner/hf_captioner.py` | `_user_content`, `_generate_kwargs` helpers; multi-image + temperature. |
| `app.py` | `h3_*` handlers, `_h3_section`, the tab, Help text. |
| `test_h3_prompts.py` (new) | Fast checks for `h3_prompts.py` and the JSON. |
| `test_h3_app.py` (new) | Handler chain with a fake model + captioner helper checks (imports `app`, ~16 s). |
| `README.md` | New tab section + project structure lines. |

---

### Task 1: Shot plans and picture references

**Files:**
- Create: `captioner/h3_prompts.py`
- Test: `test_h3_prompts.py` (create)

**Interfaces:**
- Produces:
  - `MAX_SHOT: int = 12`
  - `clamp_durations(total: int) -> list[int]`
  - `story_segments(story: str) -> list[tuple[int, int]]`
  - `build_shot_plan(segments: list[tuple[int, int]]) -> tuple[list[tuple[int, int, int]], list[str]]` — items are `(story_start_s, story_end_s, duration_s)`, second value is log notes.
  - `shot_plan_from_story(story: str) -> tuple[plan, notes]`
  - `shot_plan_from_total(total: int) -> tuple[plan, notes]`
  - `format_shot_plan(plan) -> str`
  - `ROLE_LABELS: dict[str, str]` = `{"Face sheet / góc mặt": "face", "Bối cảnh": "scene", "Trang phục / đạo cụ": "outfit"}`
  - `build_references(slots: list[str | None]) -> list[dict]` — `slots[0]` is Picture 1 (`"character"`), `None` = empty slot; dict keys `n, ref, role, slot`.
  - `describe_references(refs) -> str`
  - `BIBLE_FIELDS: tuple[str, ...]`, `bible_field_count(bible: str) -> int`

- [ ] **Step 1: Write the failing tests**

Create `test_h3_prompts.py`:

```python
"""Self-checks for the Prompt Video Minimax H3 helpers. Run: D:\\Comfy\\.venv\\Scripts\\python.exe test_h3_prompts.py"""
from captioner.h3_prompts import (
    bible_field_count,
    build_references,
    clamp_durations,
    describe_references,
    format_shot_plan,
    shot_plan_from_story,
    shot_plan_from_total,
)


def test_clamp_durations():
    assert clamp_durations(10) == [10]
    assert clamp_durations(12) == [12]
    assert clamp_durations(14) == [12]
    assert clamp_durations(15) == [12]
    assert clamp_durations(16) == [8, 8]
    assert clamp_durations(25) == [9, 8, 8]
    assert clamp_durations(30) == [10, 10, 10]
    for total in range(16, 121):
        pieces = clamp_durations(total)
        assert sum(pieces) == total and max(pieces) <= 12, (total, pieces)


def test_shot_plan_from_story():
    plan, notes = shot_plan_from_story("00:00–00:20 — Anh đi bộ.\n00:20-00:26 — Anh dừng lại.")
    assert plan == [(0, 10, 10), (10, 20, 10), (20, 26, 6)]
    assert notes == ["story 00:00–00:20 (20s) → 2 shot: 10s + 10s"]
    plan, notes = shot_plan_from_story("00:00–00:14 — Anh đi bộ.")
    assert plan == [(0, 14, 12)]
    assert notes == ["story 00:00–00:14 (14s) → nén còn 12s"]
    assert shot_plan_from_story("Anh đi bộ ra ban công.") == ([], [])
    assert shot_plan_from_story("00:10–00:05 — ngược") == ([], [])


def test_shot_plan_from_total():
    assert shot_plan_from_total(24) == ([(0, 12, 12), (12, 24, 12)], ["story 00:00–00:24 (24s) → 2 shot: 12s + 12s"])
    assert shot_plan_from_total(0) == ([], [])
    assert shot_plan_from_total(None) == ([], [])


def test_format_shot_plan():
    assert format_shot_plan([(0, 10, 10), (10, 20, 10)]) == (
        "SHOT 01: story 00:00–00:10 (10s)\nSHOT 02: story 00:10–00:20 (10s)")
    assert format_shot_plan([(0, 14, 12)]) == "SHOT 01: story 00:00–00:14 (12s) — compress the action into 12s"
    assert format_shot_plan([]) == "Choose 5 to 12 seconds per shot; never more than 12 seconds."


def test_build_references_numbers_only_filled_slots():
    assert build_references(["character", None, "scene", "face"]) == [
        {"n": 1, "ref": "ref_image_0", "role": "character", "slot": 0},
        {"n": 2, "ref": "ref_image_1", "role": "scene", "slot": 2},
        {"n": 3, "ref": "ref_image_2", "role": "face", "slot": 3},
    ]


def test_describe_references():
    only_character = describe_references(build_references(["character", None, None, None]))
    assert only_character == (
        "<Picture 1>, ref_image_0 = main character: identity, body, outfit. "
        "Also the setting, because no scene picture is given.")
    with_scene = describe_references(build_references(["character", None, "scene", "face"]))
    assert with_scene == (
        "<Picture 1>, ref_image_0 = main character: identity, body, outfit.\n"
        "<Picture 2>, ref_image_1 = scene / environment. Ignore any person visible in it.\n"
        "<Picture 3>, ref_image_2 = face reference sheet of the same character as <Picture 1>; "
        "use it for face details only.")
    outfit = describe_references(build_references(["character", "outfit"]))
    assert "<Picture 2>, ref_image_1 = outfit / props worn or held by the main character." in outfit


def test_bible_field_count():
    assert bible_field_count("NAME_HINT: a\nFACE: b\nfoo: c\n") == 2
    assert bible_field_count("") == 0


if __name__ == "__main__":
    tests = [fn for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for fn in tests:
        fn()
        print("ok", fn.__name__)
    print(f"{len(tests)} checks passed")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `D:\Comfy\.venv\Scripts\python.exe test_h3_prompts.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'captioner.h3_prompts'`

- [ ] **Step 3: Write the implementation**

Create `captioner/h3_prompts.py` (the imports cover Tasks 2–3 too):

```python
"""Pure helpers for the Prompt Video Minimax H3 tab. No gradio/torch imports, so test_h3_prompts.py runs standalone."""
import json
import math
import re
import unicodedata
from datetime import datetime
from pathlib import Path

from .prompts import _format

MAX_SHOT = 12

# ── Shot plans ───────────────────────────────────────────────────────────────
TIMERANGE = re.compile(r"(\d{1,2}):(\d{2})\s*[–—-]\s*(\d{1,2}):(\d{2})")


def _mmss(seconds: int) -> str:
    s = int(round(seconds))
    return f"{s // 60:02d}:{s % 60:02d}"


def clamp_durations(total: int) -> list[int]:
    """One clip ≤12s; 13–15s is compressed to 12s; longer is split into ceil(T/12) integer clips summing to T."""
    total = int(round(total))
    if total <= MAX_SHOT:
        return [total]
    if total <= 15:
        return [MAX_SHOT]
    n = math.ceil(total / MAX_SHOT)
    q, r = divmod(total, n)
    return [q + 1] * r + [q] * (n - r)


def story_segments(story: str) -> list[tuple[int, int]]:
    """Every 'MM:SS–MM:SS' range in the story, in order, as seconds. Empty or reversed ranges are skipped."""
    out = []
    for m in TIMERANGE.finditer(story or ""):
        start, end = int(m[1]) * 60 + int(m[2]), int(m[3]) * 60 + int(m[4])
        if end > start:
            out.append((start, end))
    return out


def build_shot_plan(segments):
    """[(start, end)] → ([(story_start, story_end, duration)], log notes) with the 12s rule applied."""
    plan, notes = [], []
    for a, b in segments:
        pieces = clamp_durations(b - a)
        if len(pieces) == 1:
            plan.append((a, b, pieces[0]))
            if pieces[0] != b - a:
                notes.append(f"story {_mmss(a)}–{_mmss(b)} ({b - a}s) → nén còn {pieces[0]}s")
            continue
        start = a
        for d in pieces:
            plan.append((start, start + d, d))
            start += d
        notes.append(f"story {_mmss(a)}–{_mmss(b)} ({b - a}s) → {len(pieces)} shot: "
                     + " + ".join(f"{d}s" for d in pieces))
    return plan, notes


def shot_plan_from_story(story: str):
    return build_shot_plan(story_segments(story))


def shot_plan_from_total(total):
    total = int(total or 0)
    return build_shot_plan([(0, total)]) if total > 0 else ([], [])


def format_shot_plan(plan) -> str:
    if not plan:
        return "Choose 5 to 12 seconds per shot; never more than 12 seconds."
    lines = []
    for i, (s, e, d) in enumerate(plan, 1):
        line = f"SHOT {i:02d}: story {_mmss(s)}–{_mmss(e)} ({d}s)"
        if e - s != d:
            line += f" — compress the action into {d}s"
        lines.append(line)
    return "\n".join(lines)


# ── Picture references ───────────────────────────────────────────────────────
ROLE_LABELS = {"Face sheet / góc mặt": "face", "Bối cảnh": "scene", "Trang phục / đạo cụ": "outfit"}
ROLE_TEXT = {
    "character": "main character: identity, body, outfit",
    "face": "face reference sheet of the same character as <Picture 1>; use it for face details only",
    "scene": "scene / environment. Ignore any person visible in it",
    "outfit": "outfit / props worn or held by the main character",
}


def build_references(slots):
    """slots[0] = Picture 1 role ('character'); later items are extra-slot roles or None when empty."""
    refs = []
    for slot, role in enumerate(slots):
        if role:
            n = len(refs) + 1
            refs.append({"n": n, "ref": f"ref_image_{n - 1}", "role": role, "slot": slot})
    return refs


def describe_references(refs) -> str:
    has_scene = any(r["role"] == "scene" for r in refs)
    lines = []
    for r in refs:
        text = ROLE_TEXT[r["role"]]
        if r["role"] == "character" and not has_scene:
            text += ". Also the setting, because no scene picture is given"
        lines.append(f"<Picture {r['n']}>, {r['ref']} = {text}.")
    return "\n".join(lines)


BIBLE_FIELDS = ("NAME_HINT", "AGE_APPEARANCE", "FACE", "HAIR", "BODY", "OUTFIT", "ACCESSORIES",
                "SETTING", "LIGHTING", "CAMERA_FRAMING", "STYLE")


def bible_field_count(bible: str) -> int:
    return sum(1 for line in (bible or "").splitlines() if line.split(":", 1)[0].strip() in BIBLE_FIELDS)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `D:\Comfy\.venv\Scripts\python.exe test_h3_prompts.py`
Expected: `7 checks passed`

- [ ] **Step 5: Commit**

```bash
git add captioner/h3_prompts.py test_h3_prompts.py
git commit -m "Add H3 shot-plan and picture-reference helpers"
```

---

### Task 2: Output split/check and idea parsing

**Files:**
- Modify: `captioner/h3_prompts.py` (append)
- Test: `test_h3_prompts.py` (append tests above the `__main__` block, extend the import)

**Interfaces:**
- Consumes: `clamp_durations`, `build_shot_plan`, `MAX_SHOT` (Task 1)
- Produces:
  - `clean_output(text: str) -> str`
  - `split_h3_output(text: str) -> tuple[str, str]` — raises `ValueError("no [SHOT ...] block")`
  - `check_shots(part2: str) -> tuple[str, list[str]]`
  - `parse_h3_ideas(text: str, limit: int = 3) -> list[dict]` — dict keys `name: str, logline: str, shots: list[int]`
  - `idea_label(n: int, idea: dict) -> str` (n is 1-based)
  - `idea_text(n: int, idea: dict) -> str`
  - `idea_plan(idea: dict) -> list[tuple[int, int, int]]`

- [ ] **Step 1: Write the failing tests**

Add to the import list in `test_h3_prompts.py`: `check_shots, idea_label, idea_plan, idea_text, parse_h3_ideas, split_h3_output`. Then add above `if __name__ == "__main__":`

```python
FENCE = "`" * 3  # a code fence the model may wrap its answer in
RAW_SHOTS = f"""Here you go:
{FENCE}
# CHARACTER & SCENE DEFINITION
subject_definitions:
<Subject 1> man, based on <Picture 1>, ref_image_0: oval face.
summary:
[reference generation] He walks.
**SHOT STRUCTURE — MINIMAX H3**
[SHOT 01 | DURATION: 8s | PRIMARY CAMERA SETUP: wide]
summary:
He walks.
{FENCE}
"""


def test_split_h3_output():
    part1, part2 = split_h3_output(RAW_SHOTS)
    assert part1 == ("subject_definitions:\n<Subject 1> man, based on <Picture 1>, ref_image_0: oval face.\n"
                     "summary:\n[reference generation] He walks.")
    assert part2 == "[SHOT 01 | DURATION: 8s | PRIMARY CAMERA SETUP: wide]\nsummary:\nHe walks."
    try:
        split_h3_output("I cannot help with that.")
    except ValueError as e:
        assert "no [SHOT" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_check_shots_renumbers_clamps_and_completes_tail():
    part2 = ("[SHOT 03 | DURATION: 15s | PRIMARY CAMERA SETUP: wide]\nsummary:\nA.\noverall_soundscape:\nWind.\n\n"
             "[SHOT 7 | DURATION: 6s]\nsummary:\nB.\noverall_soundscape:\nSteps.\nnon_diegetic_music:\nNONE")
    fixed, warnings = check_shots(part2)
    assert fixed == (
        "[SHOT 01 | DURATION: 12s | PRIMARY CAMERA SETUP: wide]\nsummary:\nA.\noverall_soundscape:\nWind.\n"
        "non_diegetic_music:\nNONE\n\n"
        "[SHOT 02 | DURATION: 6s]\nsummary:\nB.\noverall_soundscape:\nSteps.\nnon_diegetic_music:\nNONE")
    assert warnings == ["SHOT 01: DURATION 15s → 12s", "SHOT 01: thêm non_diegetic_music mặc định"]
    _, warnings = check_shots("[SHOT 01 | PRIMARY CAMERA SETUP: wide]\nsummary:\nA.")
    assert warnings[0] == "SHOT 01: thiếu DURATION"
    assert "SHOT 01: thêm overall_soundscape mặc định" in warnings


IDEAS_RAW = """Đây là 3 ý tưởng:
**Ý TƯỞNG 1: Buổi sáng ban công**
Logline: Anh pha cà phê rồi ra ban công.
Tổng thời lượng: 20s
Số shot: 2
Thời lượng shot: 10s + 10s

## Ý tưởng 2: Gương soi
Logline: Chỉnh áo, nháy mắt.
Tổng thời lượng: 12s

Ý TƯỞNG 3: Đi dạo
Logline: Bước chậm, quay lại cười.
Thời lượng shot: 20s + 4s

Ý TƯỞNG 4: Thừa
Logline: x
"""


def test_parse_h3_ideas():
    ideas = parse_h3_ideas(IDEAS_RAW)
    assert ideas == [
        {"name": "Buổi sáng ban công", "logline": "Anh pha cà phê rồi ra ban công.", "shots": [10, 10]},
        {"name": "Gương soi", "logline": "Chỉnh áo, nháy mắt.", "shots": [12]},
        {"name": "Đi dạo", "logline": "Bước chậm, quay lại cười.", "shots": [10, 10, 4]},
    ]
    assert parse_h3_ideas("") == []
    assert parse_h3_ideas("no ideas here") == []


def test_idea_formatting_and_plan():
    idea = {"name": "Buổi sáng ban công", "logline": "Anh pha cà phê rồi ra ban công.", "shots": [10, 10]}
    assert idea_label(1, idea) == "Ý tưởng 1: Buổi sáng ban công — Anh pha cà phê rồi ra ban công. · 20s · 2 shot: 10s + 10s"
    assert idea_text(1, idea) == ("Ý TƯỞNG 1: Buổi sáng ban công\nLogline: Anh pha cà phê rồi ra ban công.\n"
                                  "Tổng thời lượng: 20s\nSố shot: 2\nThời lượng shot: 10s + 10s")
    assert idea_plan(idea) == [(0, 10, 10), (10, 20, 10)]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `D:\Comfy\.venv\Scripts\python.exe test_h3_prompts.py`
Expected: FAIL with `ImportError: cannot import name 'check_shots'`

- [ ] **Step 3: Write the implementation**

Append to `captioner/h3_prompts.py`:

```python
# ── Model output ─────────────────────────────────────────────────────────────
PART_TITLES = re.compile(
    r"(?mi)^[ \t]*(?:PART\s*\d+\s*[—–:-]\s*)?"
    r"(?:CHARACTER\s*&\s*SCENE\s*DEFINITION|SHOT\s*STRUCTURE\s*[—–-]+\s*MINIMAX\s*H3)[ \t]*:?[ \t]*$\n?")
FIRST_SHOT = re.compile(r"(?m)^[ \t]*\[\s*SHOT\b")
SHOT_LINE = re.compile(r"(?mi)^[ \t]*\[\s*SHOT\s*0*\d+[^\n]*$")
DURATION_RE = re.compile(r"(?i)DURATION\s*:\s*(\d+(?:[.,]\d+)?)")
SHOT_TAIL_DEFAULTS = (("overall_soundscape", "Natural ambient sound, no dialogue."),
                      ("non_diegetic_music", "NONE"))


def clean_output(text: str) -> str:
    """NFC, no code fences, no '**', no leading '#' heading marks."""
    text = unicodedata.normalize("NFC", text or "").replace("**", "")
    text = "\n".join(line for line in text.splitlines() if not line.strip().startswith("```"))
    return re.sub(r"(?m)^[ \t]*#+[ \t]*", "", text).strip()


def split_h3_output(text: str) -> tuple[str, str]:
    """→ (CHARACTER & SCENE DEFINITION body, SHOT STRUCTURE body). Preamble and part titles are dropped."""
    text = clean_output(text)
    m = FIRST_SHOT.search(text)
    if not m:
        raise ValueError("no [SHOT ...] block")
    head = PART_TITLES.sub("", text[:m.start()])
    start = re.search(r"(?mi)^subject_definitions:", head)
    if start:
        head = head[start.start():]
    return head.strip(), text[m.start():].strip()


def check_shots(part2: str) -> tuple[str, list[str]]:
    """Renumber SHOT 01..NN, cap DURATION at 12s, add a missing soundscape/music tail to each shot."""
    heads = list(SHOT_LINE.finditer(part2))
    out, warnings = [], []
    for i, m in enumerate(heads, 1):
        end = heads[i].start() if i < len(heads) else len(part2)
        header = re.sub(r"(?i)SHOT\s*0*\d+", f"SHOT {i:02d}", m.group(0).strip(), count=1)
        d = DURATION_RE.search(header)
        if not d:
            warnings.append(f"SHOT {i:02d}: thiếu DURATION")
        elif float(d[1].replace(",", ".")) > MAX_SHOT:
            warnings.append(f"SHOT {i:02d}: DURATION {d[1]}s → {MAX_SHOT}s")
            header = header[:d.start(1)] + str(MAX_SHOT) + header[d.end(1):]
        body = part2[m.end():end].strip()
        for field, default in SHOT_TAIL_DEFAULTS:
            if not re.search(rf"(?mi)^{field}:", body):
                body += f"\n{field}:\n{default}"
                warnings.append(f"SHOT {i:02d}: thêm {field} mặc định")
        out.append(f"{header}\n{body.strip()}")
    return "\n\n".join(out), warnings


# ── Ideas ────────────────────────────────────────────────────────────────────
IDEA_SPLIT = re.compile(r"(?mi)^[ \t>]*Ý\s*TƯỞNG\s*\d+\s*[:.\-–]?[ \t]*")


def _field(block: str, name: str) -> str:
    m = re.search(rf"(?mi)^[ \t-]*{name}\s*:\s*(.+)$", block)
    return m.group(1).strip() if m else ""


def _seconds(text: str) -> list[int]:
    return [int(round(float(x.replace(",", ".")))) for x in re.findall(r"(\d+(?:[.,]\d+)?)\s*s\b", text)]


def parse_h3_ideas(text: str, limit: int = 3) -> list[dict]:
    """'Ý TƯỞNG N' blocks → [{name, logline, shots}]. Shots >12s are split; a missing shot list falls back to the total."""
    ideas = []
    for part in IDEA_SPLIT.split(clean_output(text))[1:limit + 1]:
        name, _, rest = part.partition("\n")
        shots = _seconds(_field(rest, "Thời lượng shot"))
        if not shots:
            total = _seconds(_field(rest, "Tổng thời lượng"))
            shots = [total[0] if total else MAX_SHOT]
        ideas.append({"name": name.strip(), "logline": _field(rest, "Logline"),
                      "shots": [p for s in shots for p in clamp_durations(s)]})
    return ideas


def _shots_text(idea) -> str:
    return " + ".join(f"{s}s" for s in idea["shots"])


def idea_label(n: int, idea: dict) -> str:
    return (f"Ý tưởng {n}: {idea['name']} — {idea['logline']} · {sum(idea['shots'])}s · "
            f"{len(idea['shots'])} shot: {_shots_text(idea)}")


def idea_text(n: int, idea: dict) -> str:
    return (f"Ý TƯỞNG {n}: {idea['name']}\nLogline: {idea['logline']}\n"
            f"Tổng thời lượng: {sum(idea['shots'])}s\nSố shot: {len(idea['shots'])}\n"
            f"Thời lượng shot: {_shots_text(idea)}")


def idea_plan(idea: dict):
    segments, start = [], 0
    for s in idea["shots"]:
        segments.append((start, start + s))
        start += s
    return build_shot_plan(segments)[0]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `D:\Comfy\.venv\Scripts\python.exe test_h3_prompts.py`
Expected: `11 checks passed`

- [ ] **Step 5: Commit**

```bash
git add captioner/h3_prompts.py test_h3_prompts.py
git commit -m "Add H3 output split/check and idea parsing helpers"
```

---

### Task 3: system_prompt.json, prompt store and output paths

**Files:**
- Create: `captioner/system_prompt.json` (generated once by the script in Step 3; the script is not committed)
- Modify: `captioner/h3_prompts.py` (append)
- Test: `test_h3_prompts.py` (append, extend imports)

**Interfaces:**
- Produces:
  - `H3_PROMPT_FILE: Path` (module global, tests reassign it)
  - `H3_KEYS = ("bible", "ideas_sfw", "ideas_nsfw", "expand_sfw", "expand_nsfw", "shots_sfw", "shots_nsfw")`
  - `H3_FIELDS = ("system", "user", "temperature", "max_tokens")`
  - `step_key(step: str, nsfw: bool) -> str`
  - `load_h3_prompts() -> dict` — raises `ValueError("system_prompt.json: ...")`
  - `get_h3_step(step: str, nsfw: bool) -> dict`
  - `save_h3_system(step: str, nsfw: bool, text: str) -> None`
  - `render_user(template: str, **fields) -> str`
  - `clean_path(text: str) -> str`
  - `h3_output_path(picture1_path: str, nsfw: bool, ideas: bool = False, now: datetime | None = None) -> Path`
  - `build_h3_file(part1: str, part2: str) -> str`

- [ ] **Step 1: Write the failing tests**

Add to the imports of `test_h3_prompts.py`:

```python
import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import captioner.h3_prompts as h3
```

and to the `from captioner.h3_prompts import (...)` list: `H3_FIELDS, H3_KEYS, build_h3_file, clean_path, get_h3_step, h3_output_path, load_h3_prompts, render_user, save_h3_system, step_key`. Then add:

```python
STEP_PLACEHOLDERS = {
    "bible": ["{references}"],
    "ideas": ["{bible}", "{references}", "{duration_rule}"],
    "expand": ["{bible}", "{references}", "{idea}", "{shot_plan}"],
    "shots": ["{bible}", "{references}", "{story}", "{shot_plan}"],
}


def test_system_prompt_json_shape():
    data = load_h3_prompts()
    assert set(data) == set(H3_KEYS), sorted(data)
    for key, cfg in data.items():
        assert set(cfg) == set(H3_FIELDS), key
        assert cfg["system"].strip() and cfg["user"].strip(), key
        assert 0 <= cfg["temperature"] <= 1.5 and cfg["max_tokens"] >= 256, key
        for placeholder in STEP_PLACEHOLDERS[key.split("_")[0]]:
            assert placeholder in cfg["user"], (key, placeholder)
    for key in ("ideas_nsfw", "expand_nsfw", "shots_nsfw"):
        assert "minor" in data[key]["system"], key
    for key in ("shots_sfw", "shots_nsfw"):
        for marker in ("CHARACTER & SCENE DEFINITION", "SHOT STRUCTURE — MINIMAX H3",
                       "overall_soundscape", "non_diegetic_music", "ref_image_0"):
            assert marker in data[key]["system"], (key, marker)
    for key in ("ideas_sfw", "ideas_nsfw"):
        assert "Thời lượng shot:" in data[key]["system"], key


def test_step_key():
    assert step_key("bible", True) == "bible"
    assert step_key("ideas", False) == "ideas_sfw"
    assert step_key("shots", True) == "shots_nsfw"


def test_render_user():
    assert render_user("A {x} {y}", x=1) == "A 1 {y}"


def _with_temp_prompt_file(fn):
    original = h3.H3_PROMPT_FILE
    with tempfile.TemporaryDirectory() as tmp:
        h3.H3_PROMPT_FILE = Path(tmp) / "system_prompt.json"
        shutil.copy(original, h3.H3_PROMPT_FILE)
        try:
            fn(h3.H3_PROMPT_FILE)
        finally:
            h3.H3_PROMPT_FILE = original


def test_save_h3_system_roundtrip():
    def body(path):
        save_h3_system("shots", True, "  new rules  ")
        assert get_h3_step("shots", True)["system"] == "new rules"
        assert get_h3_step("shots", False)["system"] != "new rules"
        assert json.loads(path.read_text(encoding="utf-8"))["shots_nsfw"]["system"] == "new rules"
    _with_temp_prompt_file(body)


def test_broken_prompt_file_raises_clear_error():
    def body(path):
        path.write_text("{not json", encoding="utf-8")
        try:
            get_h3_step("bible", False)
        except ValueError as e:
            assert "system_prompt.json" in str(e)
        else:
            raise AssertionError("expected ValueError")
        path.write_text(json.dumps({"bible": {"system": "x"}}), encoding="utf-8")
        try:
            get_h3_step("bible", False)
        except ValueError as e:
            assert "bible" in str(e)
        else:
            raise AssertionError("expected ValueError")
    _with_temp_prompt_file(body)


def test_output_paths():
    with tempfile.TemporaryDirectory() as tmp:
        pic = Path(tmp) / "hero.png"
        pic.write_bytes(b"x")
        assert h3_output_path(str(pic), False) == Path(tmp) / "hero_H3_sfw.txt"
        assert h3_output_path(f' "{pic}" ', True, ideas=True) == Path(tmp) / "hero_H3_nsfw_ideas.txt"
    now = datetime(2026, 10, 19, 10, 15, 0)
    assert h3_output_path("", False, now=now) == Path.home() / "Desktop" / "h3_20261019_101500_sfw.txt"
    assert h3_output_path(r"Z:\missing.png", True, True, now) == Path.home() / "Desktop" / "h3_20261019_101500_nsfw_ideas.txt"
    assert clean_path('  "D:\\a b.png" ') == "D:\\a b.png"
    assert clean_path(None) == ""
    assert build_h3_file("A", "B") == "CHARACTER & SCENE DEFINITION\n\nA\n\nSHOT STRUCTURE — MINIMAX H3\n\nB\n"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `D:\Comfy\.venv\Scripts\python.exe test_h3_prompts.py`
Expected: FAIL with `ImportError: cannot import name 'H3_FIELDS'`

- [ ] **Step 3: Generate `captioner/system_prompt.json`**

Save this script as `<scratchpad>/make_h3_prompts.py` (NOT in the repo) and run
`D:\Comfy\.venv\Scripts\python.exe <scratchpad>/make_h3_prompts.py D:\006_Captioner_QWENVL\captioner\system_prompt.json`.
From then on the JSON is the source of truth; the script is thrown away.

```python
import json
import sys

SFW = """Content rules (SFW):
- Safe for social media: no nudity, no sexual acts, no graphic violence. Light charm, romance or humor is fine."""

NSFW = """Content rules (NSFW, adults only; fictional, AI-generated characters):
- Sensual, erotic and explicit adult content is allowed when the story or idea asks for it. Describe bodies, touch and intimacy directly and visually.
- Hard limits, never break them:
  - Every character is an adult and is described as a man or a woman. Never use words that suggest youth (boy, girl, teen, kid, schoolboy, schoolgirl, student, young-looking).
  - If AGE_APPEARANCE in the bible says "minor" or the person may be under 18, ignore the adult permission and write SFW content only.
  - Fictional characters only. Never refer to a real or famous person.
  - Clear, enthusiastic consent between adults. No violence, coercion, intoxication, sleep, or anything non-consensual. Nothing illegal."""

BIBLE = """You are a visual analyst for a video pre-production tool. You receive one or more reference pictures, attached in the order of the REFERENCE PICTURES list, each with a role. Write a CHARACTER BIBLE that a video model will use to keep the character, outfit and setting consistent across many shots.

Rules:
- Describe only what is visible. Do not guess names, real identities or personal data. Never say the person resembles a real or famous person.
- Use each picture only for its role. Face and body come from <Picture 1> and the face sheet, if one is listed. Outfit comes from <Picture 1> and the outfit picture, if one is listed. The setting comes from the scene pictures if listed, otherwise from <Picture 1>.
- Ignore any person who appears in a scene picture.
- Plain English, short concrete phrases, no opinions, no story.
- If something is not visible, write "not visible".
- Output exactly these 11 fields, one per line, in this order. Nothing before the first field, nothing after the last field, no markdown.

NAME_HINT: a short neutral label for the character, e.g. "man with short black hair"
AGE_APPEARANCE: apparent age range, then "adult" if the person clearly looks 18 or older, or "minor" if the person may be under 18
FACE: face shape, eyes, eyebrows, nose, lips, skin tone, facial hair, expression
HAIR: color, length, style
BODY: build, proportions, notable features
OUTFIT: every visible clothing item with color, material and fit
ACCESSORIES: jewelry, props, footwear
SETTING: location, architecture, furniture, floor, walls, what is on the left and right of the frame, and which picture it comes from
LIGHTING: direction, softness, color temperature
CAMERA_FRAMING: shot size, angle, pose, aspect ratio
STYLE: photographic or illustrated, color palette, mood"""

IDEAS = """You are a creative director who writes short-video story ideas for the MiniMax Hailuo H3 image-to-video model. You receive a CHARACTER BIBLE taken from the reference pictures and the REFERENCE PICTURES list. Propose exactly 3 different story ideas starring the main character in the setting from the bible.

{content}

Idea rules:
- Every idea must work with this character, this outfit and this setting. Change the location only to a place a scene picture shows.
- The 3 ideas must be genuinely different in genre or mood (for example slice of life, humor, cinematic, emotional), not rewrites of the same action.
- Follow the DURATION RULE. Each shot is at most 12 seconds. The shot seconds must add up to the total.
- Keep each shot to one or two simple, physically believable actions.
- Write everything in Vietnamese.

Output format. Write exactly this block three times (Ý TƯỞNG 1, 2, 3), separated by one blank line. Nothing before the first line, nothing after the last line. No markdown, no bold, no # headings.

Ý TƯỞNG 1: <tên ngắn>
Logline: <1-2 câu>
Tổng thời lượng: <N>s
Số shot: <K>
Thời lượng shot: <a>s + <b>s"""

IDEAS_USER = """CHARACTER BIBLE:
{bible}

REFERENCE PICTURES:
{references}

DURATION RULE: {duration_rule}

TASK: Propose exactly 3 story ideas now."""

EXPAND = """You are a screenwriter for the MiniMax Hailuo H3 image-to-video model. You receive a CHARACTER BIBLE, the REFERENCE PICTURES list, one chosen story IDEA and a SHOT PLAN. Write the detailed story of that idea.

{content}

Rules:
- Follow the SHOT PLAN exactly: one line per shot, in order, starting with the story timecodes from the plan (MM:SS–MM:SS).
- Each line says what the character does, how the facial expression changes, the camera (shot size, angle, movement) and the sounds.
- Stay in the setting and outfit from the bible. Every action must fit in its seconds, including transitions: standing up, turning and walking take time.
- Spoken lines are optional. When the character speaks, write the line in Vietnamese inside double quotes.
- Write in Vietnamese. Output only the story lines: nothing before or after, no markdown.

Format:
00:00–00:10 — <...>
00:10–00:20 — <...>"""

EXPAND_USER = """CHARACTER BIBLE:
{bible}

REFERENCE PICTURES:
{references}

IDEA:
{idea}

SHOT PLAN:
{shot_plan}

Write the detailed story now."""

SHOTS = """You are the MiniMax Hailuo H3 video director: cinematographer, motion choreographer and prompt engineer. You receive a CHARACTER BIBLE taken from the reference pictures, the REFERENCE PICTURES list, a STORY (Vietnamese or English) and a SHOT PLAN. Turn the story into an H3 production prompt.

{content}

Reference rules:
- Refer to people and places as <Subject N>, and to images as <Picture N>, ref_image_N exactly as listed in REFERENCE PICTURES. Never mention a picture that is not listed.
- <Subject 1> is the main character from <Picture 1>, ref_image_0. <Subject 2> is the main location: from the scene picture if one is listed, otherwise from <Picture 1>, ref_image_0.
- A person visible in a scene picture is ignored; <Picture 1> controls identity.
- A place that no picture shows is a coherent_extension in retention_analysis, not fully_preserved.
- Never mix the subject's own left/right with the frame's left/right.

Timing rules:
- Follow the SHOT PLAN exactly: same number of shots, same order, same durations. If the plan says to choose, use 5 to 12 seconds per shot. Never more than 12 seconds per shot.
- Inside each shot the timecodes start at 00:00, are contiguous, do not overlap and end exactly at that shot's duration.
- Plan time for transitions: standing up, turning and walking take time. Do not overload a shot.

Writing rules:
- English for all descriptions. Keep spoken lines exactly as written in the STORY, in their original language, inside double quotes, each spoken once, only in the shot where they are spoken. Never invent dialogue.
- Observable actions, camera and physics, in present tense. One coherent camera plan per shot.
- The reference picture fixes identity, not the expression: the face changes naturally (gaze shifts, blinks, small smiles, brow movement) in step with the action.
- Keep the exact clothing design, fit and coverage unless the story changes it. Correct limbs and hands; feet carry the weight; no floating or duplicated props.
- Sound follows the action. Name the diegetic sounds of each shot and count repeated events exactly. non_diegetic_music is NONE unless the story asks for music.
- No text, captions, subtitles, logos or watermarks on screen.
- Output only the prompt below. No introduction, no explanation, no markdown, no code fences.

Output format (follow exactly):

CHARACTER & SCENE DEFINITION
subject_definitions:
<Subject 1> <short label>, based on <Picture 1>, ref_image_0:
<fixed face, hair, body and outfit; expressions may change>
<Subject 2> <location label>, based on <Picture K>, ref_image_<K-1>:
<architecture, layout (left/right of frame), colors, lighting>
summary:
[reference generation] <the whole video in 1-3 sentences: setting, subject, order of actions>
retention_analysis:
<Subject 1> (appears throughout): fully_preserved — <locked traits; allowed motion>
<Subject 2> (appears throughout): fully_preserved — <locked scene traits>
overall_soundscape:
<shared ambience of the whole video>
non_diegetic_music:
NONE

SHOT STRUCTURE — MINIMAX H3
[SHOT 01 | DURATION: <N>s | PRIMARY CAMERA SETUP: <shot size, angle, movement, framing>]
subject_definitions:
<Subject 1> <label>, based on <Picture 1>, ref_image_0: <short recap of identity and outfit>
<Subject 2> <label>, based on <Picture K>, ref_image_<K-1>: <short recap of the setting>
summary:
<what happens in this shot, with the visual style>
retention_analysis:
<what stays the same and what changes in this shot>
detailed_description:
00:00–00:0X — <action, expression, camera, physics>
00:0X–00:<N> — <...>
overall_soundscape:
<the sounds of this shot, synced to its actions>
non_diegetic_music:
NONE

Repeat the whole [SHOT NN] block for every line of the SHOT PLAN (SHOT 02, SHOT 03, ...)."""

SHOTS_USER = """CHARACTER BIBLE:
{bible}

REFERENCE PICTURES:
{references}

STORY:
{story}

SHOT PLAN:
{shot_plan}

REMINDERS: start with CHARACTER & SCENE DEFINITION; one [SHOT NN] block per SHOT PLAN line; every shot ends with its own overall_soundscape and non_diegetic_music."""


def step(system, user, temperature, max_tokens):
    return {"system": system, "user": user, "temperature": temperature, "max_tokens": max_tokens}


def fill(text, content):
    return text.replace("{content}", content)  # .replace keeps every other brace literal


data = {
    "bible": step(BIBLE, "REFERENCE PICTURES (attached in this order):\n{references}\n\nWrite the CHARACTER BIBLE now.", 0.3, 768),
    "ideas_sfw": step(fill(IDEAS, SFW), IDEAS_USER, 0.7, 1024),
    "ideas_nsfw": step(fill(IDEAS, NSFW), IDEAS_USER, 0.7, 1024),
    "expand_sfw": step(fill(EXPAND, SFW), EXPAND_USER, 0.6, 1024),
    "expand_nsfw": step(fill(EXPAND, NSFW), EXPAND_USER, 0.6, 1024),
    "shots_sfw": step(fill(SHOTS, SFW), SHOTS_USER, 0.4, 3072),
    "shots_nsfw": step(fill(SHOTS, NSFW), SHOTS_USER, 0.4, 3072),
}
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
    f.write("\n")
print("wrote", sys.argv[1])
```

- [ ] **Step 4: Write the store and path helpers**

Append to `captioner/h3_prompts.py`:

```python
# ── Prompt store (captioner/system_prompt.json) ──────────────────────────────
H3_PROMPT_FILE = Path(__file__).resolve().parent / "system_prompt.json"
H3_KEYS = ("bible", "ideas_sfw", "ideas_nsfw", "expand_sfw", "expand_nsfw", "shots_sfw", "shots_nsfw")
H3_FIELDS = ("system", "user", "temperature", "max_tokens")


def step_key(step: str, nsfw: bool) -> str:
    return step if step == "bible" else f"{step}_{'nsfw' if nsfw else 'sfw'}"


def load_h3_prompts() -> dict:
    """Read the file on every call so ⚙ Save and hand edits apply without a restart."""
    try:
        data = json.loads(H3_PROMPT_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ValueError(f"{H3_PROMPT_FILE.name}: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"{H3_PROMPT_FILE.name}: top level must be an object")
    return data


def get_h3_step(step: str, nsfw: bool) -> dict:
    key = step_key(step, nsfw)
    cfg = load_h3_prompts().get(key)
    if not isinstance(cfg, dict) or any(f not in cfg for f in H3_FIELDS):
        raise ValueError(f"{H3_PROMPT_FILE.name}: '{key}' needs {', '.join(H3_FIELDS)}")
    return cfg


def save_h3_system(step: str, nsfw: bool, text: str) -> None:
    data = load_h3_prompts()
    get_h3_step(step, nsfw)  # validates the key before writing
    data[step_key(step, nsfw)]["system"] = (text or "").strip()
    H3_PROMPT_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def render_user(template: str, **fields) -> str:
    return _format(template, **fields)


# ── Output files ─────────────────────────────────────────────────────────────
def clean_path(text) -> str:
    """Accept Explorer's 'Copy as path' (quoted) input."""
    return (text or "").strip().strip('"').strip()


def h3_output_path(picture1_path, nsfw: bool, ideas: bool = False, now=None) -> Path:
    preset = "nsfw" if nsfw else "sfw"
    suffix = "_ideas" if ideas else ""
    path = clean_path(picture1_path)
    if path and Path(path).is_file():
        p = Path(path)
        return p.with_name(f"{p.stem}_H3_{preset}{suffix}.txt")
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return Path.home() / "Desktop" / f"h3_{stamp}_{preset}{suffix}.txt"


def build_h3_file(part1: str, part2: str) -> str:
    return (f"CHARACTER & SCENE DEFINITION\n\n{(part1 or '').strip()}\n\n"
            f"SHOT STRUCTURE — MINIMAX H3\n\n{(part2 or '').strip()}\n")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `D:\Comfy\.venv\Scripts\python.exe test_h3_prompts.py`
Expected: `17 checks passed`

- [ ] **Step 6: Commit**

```bash
git add captioner/h3_prompts.py captioner/system_prompt.json test_h3_prompts.py
git commit -m "Add editable H3 system_prompt.json, prompt store and output paths"
```

---

### Task 4: Multi-image input and optional temperature in both backends

**Files:**
- Modify: `captioner/base.py:25-51`
- Modify: `captioner/gguf_captioner.py:315-410` (`caption_image`, `caption_text`, new static helpers next to `_image_to_url` at line 437)
- Modify: `captioner/hf_captioner.py:196-318`
- Test: `test_h3_app.py` (create)

**Interfaces:**
- Produces:
  - `caption_image(image_path: str | list[str], user_prompt, max_new_tokens=512, system_prompt=None, temperature: float | None = None) -> str`
  - `caption_text(user_prompt, max_new_tokens=512, system_prompt=None, temperature: float | None = None) -> str`
  - `GGUFCaptioner._user_content(image_path, user_prompt) -> list[dict]` (classmethod)
  - `GGUFCaptioner._sampling(temperature, default_temperature) -> dict` (staticmethod)
  - `HFCaptioner._user_content(image_path, user_prompt) -> list[dict]` (staticmethod)
  - `HFCaptioner._generate_kwargs(temperature) -> dict` (staticmethod)

- [ ] **Step 1: Write the failing tests**

Create `test_h3_app.py`:

```python
"""Self-checks for the H3 tab handlers (fake model) and captioner helpers. Run: D:\\Comfy\\.venv\\Scripts\\python.exe test_h3_app.py"""
from captioner.gguf_captioner import GGUFCaptioner
from captioner.hf_captioner import HFCaptioner


def test_gguf_user_content_keeps_single_image_shape_and_accepts_lists():
    one = GGUFCaptioner._user_content("a.png", "hi")
    assert one[0] == {"type": "text", "text": "hi"}
    assert len(one) == 2 and one[1]["type"] == "image_url"
    many = GGUFCaptioner._user_content(["a.png", "b.png"], "hi")
    assert [c["type"] for c in many] == ["text", "image_url", "image_url"]
    assert many[1]["image_url"]["url"].endswith("a.png") and many[2]["image_url"]["url"].endswith("b.png")


def test_gguf_sampling():
    assert GGUFCaptioner._sampling(None, 0.6) == {"temperature": 0.6, "seed": 1}
    assert GGUFCaptioner._sampling(0.7, 0.6) == {"temperature": 0.7, "seed": None}


def test_hf_helpers():
    one = HFCaptioner._user_content("a.png", "hi")
    assert [c["type"] for c in one] == ["image", "text"] and one[0]["image"].startswith("file://")
    many = HFCaptioner._user_content(["a.png", "b.png"], "hi")
    assert [c["type"] for c in many] == ["image", "image", "text"]
    assert HFCaptioner._generate_kwargs(None) == {"do_sample": False}
    assert HFCaptioner._generate_kwargs(0) == {"do_sample": False}
    assert HFCaptioner._generate_kwargs(0.7) == {"do_sample": True, "temperature": 0.7, "top_p": 0.9}


if __name__ == "__main__":
    tests = [fn for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for fn in tests:
        fn()
        print("ok", fn.__name__)
    print(f"{len(tests)} checks passed")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `D:\Comfy\.venv\Scripts\python.exe test_h3_app.py`
Expected: FAIL with `AttributeError: type object 'GGUFCaptioner' has no attribute '_user_content'`

- [ ] **Step 3: Update `captioner/base.py`**

Replace the two abstract methods (lines 25-51):

```python
    @abstractmethod
    def caption_image(
        self,
        image_path: Union[str, Sequence[str]],
        user_prompt: str,
        max_new_tokens: int = 512,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate a caption for the given image, or for several images sent together in one message.
        temperature=None keeps the backend's default sampling.
        Returns the generated text string.
        """
        ...


    @abstractmethod
    def caption_text(
        self,
        user_prompt: str,
        max_new_tokens: int = 512,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate text from a text-only prompt. temperature=None keeps the backend's default sampling.
        Returns the generated text string.
        """
        ...
```

and change line 5 to `from typing import Optional, Sequence, Union`.

- [ ] **Step 4: Update `captioner/gguf_captioner.py`**

Change the import on line 13 to `from typing import Optional, Sequence, Union`.

In `caption_image`: signature becomes

```python
    def caption_image(
        self,
        image_path: Union[str, Sequence[str]],
        user_prompt: str,
        max_new_tokens: int = 512,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
```

delete `image_url = self._image_to_url(image_path)`, and replace the user message plus the `create_chat_completion` call with:

```python
        messages = [
            {"role": "system", "content": effective_system},
            {"role": "user", "content": self._user_content(image_path, user_prompt)},
        ]

        start = time.perf_counter()
        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_new_tokens,
            top_p=0.9,
            repeat_penalty=1.2,
            stop=["<|im_end|>", "<|im_start|>"],
            **self._sampling(temperature, 0.6),
        )
```

In `caption_text`: add `temperature: Optional[float] = None,` after `system_prompt`, and replace its call with:

```python
        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_new_tokens,
            top_p=0.9,
            repeat_penalty=1.15,
            stop=["<|im_end|>", "<|im_start|>"],
            **self._sampling(temperature, 0.4),
        )
```

Add right after `_image_to_url`:

```python
    @classmethod
    def _user_content(cls, image_path, user_prompt: str) -> list:
        """Text first, then one image_url per picture in order (same shape as before for a single path)."""
        paths = [image_path] if isinstance(image_path, (str, Path)) else list(image_path)
        return [{"type": "text", "text": user_prompt}] + [
            {"type": "image_url", "image_url": {"url": cls._image_to_url(str(p))}} for p in paths
        ]

    @staticmethod
    def _sampling(temperature: Optional[float], default_temperature: float) -> dict:
        """None → the old fixed sampling (seed 1). A value → that temperature and a fresh seed each call."""
        if temperature is None:
            return {"temperature": default_temperature, "seed": 1}
        return {"temperature": float(temperature), "seed": None}
```

- [ ] **Step 5: Update `captioner/hf_captioner.py`**

Change line 10 to `from typing import Optional, Sequence, Union`.

`caption_image` signature: `image_path: Union[str, Sequence[str]]` and add `temperature: Optional[float] = None,` after `system_prompt`. Delete the line `image_path = str(Path(image_path).resolve())` and replace the user message with:

```python
        messages.append({"role": "user", "content": self._user_content(image_path, user_prompt)})
```

In both `caption_image` and `caption_text` replace

```python
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
```

with

```python
                max_new_tokens=max_new_tokens,
                use_cache=True,
                **self._generate_kwargs(temperature),
```

and add `temperature: Optional[float] = None,` to the `caption_text` signature after `system_prompt`.

Add these methods to `HFCaptioner` right before `caption_image`:

```python
    @staticmethod
    def _user_content(image_path, user_prompt: str) -> list:
        """One image entry per picture in order, then the text (same shape as before for a single path)."""
        paths = [image_path] if isinstance(image_path, (str, Path)) else list(image_path)
        return [{"type": "image", "image": f"file://{Path(p).resolve()}"} for p in paths] + [
            {"type": "text", "text": user_prompt}
        ]

    @staticmethod
    def _generate_kwargs(temperature: Optional[float]) -> dict:
        """None or ≤0 → greedy, as before. A positive value → sampling at that temperature."""
        if temperature is None or temperature <= 0:
            return {"do_sample": False}
        return {"do_sample": True, "temperature": float(temperature), "top_p": 0.9}
```

- [ ] **Step 6: Run tests to verify they pass, and that nothing else broke**

Run: `D:\Comfy\.venv\Scripts\python.exe test_h3_app.py`
Expected: `3 checks passed`
Run: `D:\Comfy\.venv\Scripts\python.exe test_video_prompts.py`
Expected: `8 checks passed`
Run: `D:\Comfy\.venv\Scripts\python.exe -c "import app"`
Expected: exits 0 (warnings about pynvml / Blocks css are pre-existing).

- [ ] **Step 7: Commit**

```bash
git add captioner/base.py captioner/gguf_captioner.py captioner/hf_captioner.py test_h3_app.py
git commit -m "Let both captioners take several images and an optional temperature"
```

---

### Task 5: H3 handlers in app.py

**Files:**
- Modify: `app.py` — imports (lines 5-40); new section after `vr_restore_system` (ends line 928)
- Test: `test_h3_app.py` (append)

**Interfaces:**
- Consumes: everything produced by Tasks 1–4; existing `app._badge`, `app._save_resized_temp_image(image, mode, w, h) -> str`, `app._vr_cleanup(path)`.
- Produces (all used by Task 6 wiring; `media` is always the 14-tuple
  `(img1, img2, img3, img4, role2, role3, role4, path1, path2, path3, path4, resize_mode, resize_w, resize_h)`):
  - `H3_ROLE_CHOICES: list[str]` (the 3 `ROLE_LABELS` keys, in order)
  - `h3_ideas(bible, bible_system, ideas_system, nsfw, total, log, *media)` → `(bible, ideas_radio_update, ideas_state, status_html, log_state, log_text)`
  - `h3_expand(choice, ideas, bible, bible_system, expand_system, nsfw, log, *media)` → `(bible, story, status_html, log_state, log_text)`
  - `h3_generate(story, bible, bible_system, shots_system, nsfw, total, log, *media)` → `(bible, part1, part2, saved_path, status_html, log_state, log_text)`
  - `h3_save(part1, part2, saved_path, path1, nsfw, log)` → `(saved_path, status_html, log_state, log_text)`
  - `h3_load_path(path, log)` → `(image_or_update, log_state, log_text)`
  - `h3_browse(current: str) -> str`
  - `h3_clear()` → `("", radio_update, [], "", "", "")`
  - `h3_reload_systems(nsfw)` → 3 system-prompt strings (ideas, expand, shots)
  - `h3_save_system(step, nsfw, text) -> str` (badge html)
  - `_h3_system_text(step, nsfw) -> str`

- [ ] **Step 1: Write the failing tests**

Append to `test_h3_app.py`, above `if __name__ == "__main__":`, and add the imports at the top of the file:

```python
import tempfile
from pathlib import Path

from PIL import Image

import app
from captioner.h3_prompts import parse_h3_ideas
```

```python
class FakeCaptioner:
    """Returns canned replies in order and records every call."""

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def caption_image(self, image_path, user_prompt, max_new_tokens=512, system_prompt=None, temperature=None):
        self.calls.append(("image", image_path, user_prompt, system_prompt, temperature))
        return self.replies.pop(0)

    def caption_text(self, user_prompt, max_new_tokens=512, system_prompt=None, temperature=None):
        self.calls.append(("text", None, user_prompt, system_prompt, temperature))
        return self.replies.pop(0)


BIBLE = "NAME_HINT: man with short black hair\nAGE_APPEARANCE: 30, adult\nFACE: oval face"
SHOTS = """Sure, here it is.
CHARACTER & SCENE DEFINITION
subject_definitions:
<Subject 1> man, based on <Picture 1>, ref_image_0: oval face.
summary:
[reference generation] He walks.
overall_soundscape:
Room tone.
non_diegetic_music:
NONE

SHOT STRUCTURE — MINIMAX H3
[SHOT 01 | DURATION: 12s | PRIMARY CAMERA SETUP: medium shot]
summary:
He walks.
detailed_description:
00:00–00:12 — He walks.
overall_soundscape:
Footsteps.
"""
IDEAS = """Ý TƯỞNG 1: Ban công
Logline: Ra ban công.
Tổng thời lượng: 20s
Số shot: 2
Thời lượng shot: 10s + 10s

Ý TƯỞNG 2: Gương
Logline: Soi gương.
Thời lượng shot: 12s

Ý TƯỞNG 3: Dạo
Logline: Đi dạo.
Thời lượng shot: 8s + 8s
"""


def _media(path1="", extra=None):
    img = Image.new("RGB", (64, 64), "white")
    return (img, extra, None, None, *app.H3_ROLE_CHOICES, path1, "", "", "", "Adaptive image", 512, 512)


def _run_with(fake, fn, *args):
    app._captioner = fake
    try:
        return fn(*args)
    finally:
        app._captioner = None


def test_generate_runs_bible_then_shots_and_saves_next_to_picture1():
    with tempfile.TemporaryDirectory() as tmp:
        pic = Path(tmp) / "hero.png"
        Image.new("RGB", (64, 64), "white").save(pic)
        fake = FakeCaptioner(BIBLE, SHOTS)
        bible, part1, part2, saved, status, log, log_text = _run_with(
            fake, app.h3_generate, "00:00–00:14 — Anh đi bộ.", "", "bible rules", "shot rules", False, 0, [],
            *_media(str(pic)))
        assert bible == BIBLE
        assert part1.startswith("subject_definitions:") and "Sure" not in part1
        assert part2.startswith("[SHOT 01 | DURATION: 12s")
        assert part2.rstrip().endswith("non_diegetic_music:\nNONE")
        assert saved == str(Path(tmp) / "hero_H3_sfw.txt")
        assert Path(saved).read_text(encoding="utf-8").startswith("CHARACTER & SCENE DEFINITION\n\nsubject_definitions:")
        kind, paths, user, system, temp = fake.calls[0]
        assert kind == "image" and len(paths) == 1 and system == "bible rules" and temp == 0.3
        assert "Also the setting" in user
        kind, _, user, system, temp = fake.calls[1]
        assert kind == "text" and system == "shot rules" and temp == 0.4
        assert "SHOT 01: story 00:00–00:14 (12s)" in user and BIBLE in user
        assert "nén còn 12s" in log_text and "Saved" in log_text
        assert "#4ade80" in status  # _badge(..., "success") colour
        assert len(log) == len(log_text.splitlines())


def test_generate_sends_extra_pictures_in_order():
    with tempfile.TemporaryDirectory() as tmp:
        pic = Path(tmp) / "hero.png"
        Image.new("RGB", (64, 64), "white").save(pic)
        fake = FakeCaptioner(BIBLE, SHOTS)
        extra = Image.new("RGB", (32, 32), "black")
        _run_with(fake, app.h3_generate, "Anh đi bộ.", "", "b", "s", False, 0, [], *_media(str(pic), extra))
    kind, paths, user, _, _ = fake.calls[0]
    assert len(paths) == 2
    assert "<Picture 2>, ref_image_1 = face reference sheet" in user


def test_generate_reuses_bible_and_keeps_raw_text_when_no_shot_block():
    fake = FakeCaptioner("I cannot do that.")
    bible, part1, part2, saved, status, log, log_text = _run_with(
        fake, app.h3_generate, "Anh đi bộ.", "NAME_HINT: x", "b", "s", False, 0, [], *_media())
    assert len(fake.calls) == 1 and fake.calls[0][0] == "text"
    assert part1 == "I cannot do that." and part2 == "" and not isinstance(saved, str)
    assert "Choose 5 to 12 seconds" in fake.calls[0][2]
    assert "no [SHOT" in log_text


def test_generate_guards():
    out = app.h3_generate("x", "", "b", "s", False, 0, [], *_media())
    assert "Load a model" in out[4]
    out = _run_with(FakeCaptioner(), app.h3_generate, "x", "", "b", "s", False, 0, [], None, *_media()[1:])
    assert "Picture 1" in out[4]
    out = _run_with(FakeCaptioner(), app.h3_generate, "  ", "", "b", "s", False, 0, [], *_media())
    assert "story" in out[4].lower()


def test_ideas_fill_radio_and_save_file():
    with tempfile.TemporaryDirectory() as tmp:
        pic = Path(tmp) / "hero.png"
        Image.new("RGB", (64, 64), "white").save(pic)
        fake = FakeCaptioner(IDEAS)
        bible, radio, ideas, status, log, log_text = _run_with(
            fake, app.h3_ideas, "NAME_HINT: x", "b", "idea rules", True, 0, [], *_media(str(pic)))
        assert [c[1] for c in radio["choices"]] == ["0", "1", "2"] and radio["value"] is None
        assert radio["choices"][0][0].startswith("Ý tưởng 1: Ban công")
        assert len(ideas) == 3 and ideas[2]["shots"] == [8, 8]
        saved = Path(tmp) / "hero_H3_nsfw_ideas.txt"
        assert saved.read_text(encoding="utf-8").startswith("Ý TƯỞNG 1: Ban công")
        kind, _, user, system, temp = fake.calls[0]
        assert kind == "text" and system == "idea rules" and temp == 0.7
        assert "Each idea lasts 10 to 36 seconds in total." in user
        fake = FakeCaptioner(IDEAS)
        _run_with(fake, app.h3_ideas, "NAME_HINT: x", "b", "i", False, 24, [], *_media(str(pic)))
        assert "Each idea lasts exactly 24 seconds in total." in fake.calls[0][2]


def test_ideas_warns_when_nothing_parsed():
    out = _run_with(FakeCaptioner("nope"), app.h3_ideas, "NAME_HINT: x", "b", "i", False, 0, [], *_media())
    assert "no ideas" in out[3]


def test_expand_follows_the_idea_plan():
    ideas = parse_h3_ideas(IDEAS)
    fake = FakeCaptioner("00:00–00:10 — Anh mở cửa.\n00:10–00:20 — Anh ra ban công.")
    bible, story, status, log, log_text = _run_with(
        fake, app.h3_expand, "0", ideas, "NAME_HINT: x", "b", "expand rules", False, [], *_media())
    assert story == "00:00–00:10 — Anh mở cửa.\n00:10–00:20 — Anh ra ban công."
    kind, _, user, system, temp = fake.calls[0]
    assert kind == "text" and system == "expand rules" and temp == 0.6
    assert "SHOT 02: story 00:10–00:20 (10s)" in user and "Ý TƯỞNG 1: Ban công" in user
    out = _run_with(FakeCaptioner(), app.h3_expand, None, ideas, "NAME_HINT: x", "b", "e", False, [], *_media())
    assert "Pick an idea" in out[2]


def test_save_writes_edited_text():
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "out.txt"
        saved, status, log, log_text = app.h3_save("P1", "P2", str(target), "", False, [])
        assert saved == str(target)
        assert target.read_text(encoding="utf-8") == "CHARACTER & SCENE DEFINITION\n\nP1\n\nSHOT STRUCTURE — MINIMAX H3\n\nP2\n"
    out = app.h3_save("", " ", "", "", False, [])
    assert "Nothing to save" in out[1]


def test_load_path():
    with tempfile.TemporaryDirectory() as tmp:
        pic = Path(tmp) / "hero.png"
        Image.new("RGB", (40, 30), "white").save(pic)
        img, log, log_text = app.h3_load_path(f'"{pic}"', [])
        assert img.size == (40, 30)
        img, log, log_text = app.h3_load_path(str(Path(tmp) / "missing.png"), [])
        assert not hasattr(img, "size") and "not found" in log_text
        img, log, log_text = app.h3_load_path("", [])
        assert not hasattr(img, "size") and log == []


def test_log_keeps_last_200_lines():
    log = []
    for i in range(205):
        log = app._h3_log(log, f"line {i}")
    assert len(log) == 200 and log[-1].endswith("line 204") and log[0].endswith("line 5")
```

Every test that reaches a save passes a real Picture 1 path inside a temp folder, so nothing is written to the real Desktop.

- [ ] **Step 2: Run tests to verify they fail**

Run: `D:\Comfy\.venv\Scripts\python.exe test_h3_app.py`
Expected: FAIL with `AttributeError: module 'app' has no attribute 'H3_ROLE_CHOICES'`

- [ ] **Step 3: Add imports to app.py**

Add `import time` after `import threading` (line 9). After the `from captioner.prompts import (...)` block add:

```python
from captioner.h3_prompts import (
    BIBLE_FIELDS,
    ROLE_LABELS,
    bible_field_count,
    build_h3_file,
    build_references,
    check_shots,
    clean_output,
    clean_path,
    describe_references,
    format_shot_plan,
    get_h3_step,
    h3_output_path,
    idea_label,
    idea_plan,
    idea_text,
    parse_h3_ideas,
    render_user,
    save_h3_system,
    shot_plan_from_story,
    shot_plan_from_total,
    split_h3_output,
    step_key,
)
```

- [ ] **Step 4: Add the handlers**

Insert after `vr_restore_system` (before the `# ─── Dynamic UI helpers` banner):

```python
# ── Prompt Video Minimax H3 tab ──────────────────────────────────────────────
H3_ROLE_CHOICES = list(ROLE_LABELS)


def _h3_log(lines, msg):
    msg = str(msg).replace("\n", " ⏎ ")  # one entry = one Logger line
    logger.info("H3: %s", msg)
    return ((lines or []) + [f"[{datetime.now():%H:%M:%S}] {msg}"])[-200:]


def _h3_tail(status, log):
    return status, log, "\n".join(log)


def _h3_unpack(media):
    """media = (img1..img4, role2..role4, path1..path4, resize_mode, resize_w, resize_h)."""
    return list(media[0:4]), list(media[4:7]), list(media[7:11]), media[11:14]


def _h3_refs(media):
    imgs, roles, _, _ = _h3_unpack(media)
    slots = ["character"] + [ROLE_LABELS.get(role) if img is not None else None
                             for img, role in zip(imgs[1:], roles)]
    return build_references(slots)


def _h3_guard(media):
    if _captioner is None:
        return _badge("Load a model first.", "warning")
    if media[0] is None:
        return _badge("Picture 1 (character) is required.", "warning")
    return None


def _h3_total(value) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _h3_run(step, nsfw, system, log, image_paths=None, **fields):
    """One model call configured by system_prompt.json; the system text comes from the ⚙ textbox."""
    key = step_key(step, nsfw)
    cfg = get_h3_step(step, nsfw)
    user = render_user(cfg["user"], **fields)
    log = _h3_log(log, f"{key}: start (temp {cfg['temperature']}, max {cfg['max_tokens']})")
    start = time.perf_counter()
    kwargs = {"system_prompt": (system or "").strip() or None, "temperature": cfg["temperature"]}
    if image_paths:
        text = _captioner.caption_image(image_paths, user, int(cfg["max_tokens"]), **kwargs)
    else:
        text = _captioner.caption_text(user, int(cfg["max_tokens"]), **kwargs)
    text = clean_output(text)
    log = _h3_log(log, f"{key}: done {time.perf_counter() - start:.1f}s, {len(text)} chars")
    return text, log


def _h3_bible(bible, bible_system, log, media):
    """The Bible box text when filled (the user may have edited it), else one vision call over every picture."""
    if (bible or "").strip():
        return bible.strip(), log
    imgs, _, paths, resize = _h3_unpack(media)
    refs = _h3_refs(media)
    for r in refs:
        log = _h3_log(log, f"Picture {r['n']} ({r['role']}): {clean_path(paths[r['slot']]) or '(uploaded, no path)'}")
    tmp_paths = []
    try:
        for r in refs:
            tmp_paths.append(_save_resized_temp_image(imgs[r["slot"]], *resize))
        text, log = _h3_run("bible", False, bible_system, log, image_paths=tmp_paths,
                            references=describe_references(refs))
    finally:
        for p in tmp_paths:
            _vr_cleanup(p)
    return text, _h3_log(log, f"bible: {bible_field_count(text)}/{len(BIBLE_FIELDS)} fields")


def _h3_write(path, text, log):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return _h3_log(log, f"Saved → {path}")


def h3_ideas(bible, bible_system, ideas_system, nsfw, total, log, *media):
    bible_out = gr.update()
    err = _h3_guard(media)
    if err:
        return (bible_out, gr.update(), gr.update(), *_h3_tail(err, log))
    try:
        bible, log = _h3_bible(bible, bible_system, log, media)
        bible_out = bible
        secs = _h3_total(total)
        rule = (f"Each idea lasts exactly {secs} seconds in total." if secs
                else "Each idea lasts 10 to 36 seconds in total.")
        raw, log = _h3_run("ideas", nsfw, ideas_system, log, bible=bible,
                           references=describe_references(_h3_refs(media)), duration_rule=rule)
        ideas = parse_h3_ideas(raw)
        if not ideas:
            log = _h3_log(log, f"ideas: none parsed; raw: {raw[:300]}")
            return (bible_out, gr.update(), gr.update(),
                    *_h3_tail(_badge("The model returned no ideas. Try again.", "warning"), log))
        log = _h3_log(log, f"ideas: {len(ideas)} parsed")
        log = _h3_write(h3_output_path(_h3_unpack(media)[2][0], nsfw, ideas=True),
                        "\n\n".join(idea_text(n, x) for n, x in enumerate(ideas, 1)) + "\n", log)
        radio = gr.update(choices=[(idea_label(n, x), str(n - 1)) for n, x in enumerate(ideas, 1)], value=None)
        return (bible_out, radio, ideas, *_h3_tail(_badge(f"{len(ideas)} ideas suggested.", "success"), log))
    except Exception as e:
        log = _h3_log(log, f"error: {e}")
        return (bible_out, gr.update(), gr.update(), *_h3_tail(_badge(f"Error: {e}", "error"), log))


def h3_expand(choice, ideas, bible, bible_system, expand_system, nsfw, log, *media):
    bible_out = gr.update()
    err = _h3_guard(media)
    if err:
        return (bible_out, gr.update(), *_h3_tail(err, log))
    if choice in (None, "") or not ideas:
        return (bible_out, gr.update(), *_h3_tail(_badge("Pick an idea first.", "warning"), log))
    try:
        bible, log = _h3_bible(bible, bible_system, log, media)
        bible_out = bible
        n = int(choice) + 1
        idea = ideas[n - 1]
        story, log = _h3_run("expand", nsfw, expand_system, log, bible=bible,
                             references=describe_references(_h3_refs(media)),
                             idea=idea_text(n, idea), shot_plan=format_shot_plan(idea_plan(idea)))
        if not story:
            return (bible_out, gr.update(), *_h3_tail(_badge("The model returned an empty story.", "warning"), log))
        return (bible_out, story, *_h3_tail(_badge(f"Idea {n} written into Story.", "success"), log))
    except Exception as e:
        log = _h3_log(log, f"error: {e}")
        return (bible_out, gr.update(), *_h3_tail(_badge(f"Error: {e}", "error"), log))


def h3_generate(story, bible, bible_system, shots_system, nsfw, total, log, *media):
    keep = (gr.update(), gr.update(), gr.update(), gr.update())
    err = _h3_guard(media)
    if err:
        return (*keep, *_h3_tail(err, log))
    story = (story or "").strip()
    if not story:
        return (*keep, *_h3_tail(_badge("Write a story or expand an idea first.", "warning"), log))
    bible_out = gr.update()
    try:
        bible, log = _h3_bible(bible, bible_system, log, media)
        bible_out = bible
        plan, notes = shot_plan_from_story(story)
        source = "story timecodes"
        if not plan:
            plan, notes = shot_plan_from_total(_h3_total(total))
            source = "total duration"
        if plan:
            log = _h3_log(log, f"shot plan ({source}): " + " + ".join(f"{d}s" for *_, d in plan))
        else:
            log = _h3_log(log, "shot plan: model chooses 5–12s per shot")
        for note in notes:
            log = _h3_log(log, note)
        raw, log = _h3_run("shots", nsfw, shots_system, log, bible=bible,
                           references=describe_references(_h3_refs(media)),
                           story=story, shot_plan=format_shot_plan(plan))
        try:
            part1, part2 = split_h3_output(raw)
        except ValueError as e:
            log = _h3_log(log, f"shots: {e}; raw: {raw[:300]}")
            return (bible_out, raw, "", gr.update(), *_h3_tail(
                _badge("Output has no [SHOT ...] block; the raw text is in the first box.", "warning"), log))
        part2, warnings = check_shots(part2)
        for w in warnings:
            log = _h3_log(log, w)
        path = h3_output_path(_h3_unpack(media)[2][0], nsfw)
        log = _h3_write(path, build_h3_file(part1, part2), log)
        return (bible_out, part1, part2, str(path), *_h3_tail(_badge("Prompt generated and saved.", "success"), log))
    except Exception as e:
        log = _h3_log(log, f"error: {e}")
        return (bible_out, gr.update(), gr.update(), gr.update(), *_h3_tail(_badge(f"Error: {e}", "error"), log))


def h3_save(part1, part2, saved_path, path1, nsfw, log):
    if not (part1 or "").strip() and not (part2 or "").strip():
        return (gr.update(), *_h3_tail(_badge("Nothing to save.", "warning"), log))
    try:
        path = Path(saved_path) if (saved_path or "").strip() else h3_output_path(path1, nsfw)
        log = _h3_write(path, build_h3_file(part1, part2), log)
        return (str(path), *_h3_tail(_badge(f"Saved → {path}", "success"), log))
    except Exception as e:
        log = _h3_log(log, f"save failed: {e}")
        return (gr.update(), *_h3_tail(_badge(f"Save failed: {e}", "error"), log))


def h3_load_path(path, log):
    """Path textbox → image. Accepts Explorer's quoted 'Copy as path'."""
    from PIL import Image as PILImage, ImageOps

    path = clean_path(path)
    if not path:
        return gr.update(), log, "\n".join(log or [])
    if not Path(path).is_file():
        log = _h3_log(log, f"image not found: {path}")
        return gr.update(), log, "\n".join(log)
    try:
        with PILImage.open(path) as im:
            img = ImageOps.exif_transpose(im).convert("RGB")
    except Exception as e:
        log = _h3_log(log, f"cannot open {path}: {e}")
        return gr.update(), log, "\n".join(log)
    log = _h3_log(log, f"loaded {path}")
    return img, log, "\n".join(log)


def h3_browse(current):
    """Native file dialog on the machine running the app (it runs locally)."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp *.heic"),
                                                     ("All files", "*.*")])
    finally:
        root.destroy()
    return path or current


def h3_clear():
    """New pictures make the old bible, ideas and prompts stale. The Story box is kept."""
    return "", gr.update(choices=[], value=None), [], "", "", ""


def _h3_system_text(step, nsfw):
    try:
        return get_h3_step(step, nsfw)["system"]
    except ValueError as e:
        logger.warning("H3: %s", e)
        return ""


def h3_reload_systems(nsfw):
    return tuple(_h3_system_text(step, nsfw) for step in ("ideas", "expand", "shots"))


def h3_save_system(step, nsfw, text):
    try:
        save_h3_system(step, nsfw, text)
        return _badge(f"Saved to system_prompt.json ({step_key(step, nsfw)}).", "success")
    except Exception as e:
        return _badge(f"Save failed: {e}", "error")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `D:\Comfy\.venv\Scripts\python.exe test_h3_app.py`
Expected: `13 checks passed`
Run: `D:\Comfy\.venv\Scripts\python.exe test_h3_prompts.py`
Expected: `17 checks passed`

- [ ] **Step 6: Commit**

```bash
git add app.py test_h3_app.py
git commit -m "Add Prompt Video Minimax H3 handlers"
```

---

### Task 6: Tab UI, wiring, docs and manual check

**Files:**
- Modify: `app.py` — `_h3_section` after `_vr_section` (ends line 998); the tab after the Video Reel tab block (ends line 1472); Help text after the "Video Reel tab" section (line 1571)
- Modify: `README.md` — new section after the Video Reel section (ends line 361); project structure (line 381)

**Interfaces:**
- Consumes: every `h3_*` handler and `H3_ROLE_CHOICES` (Task 5), `vr_toggle_gear` (existing).
- Produces: `_h3_section(title, step, nsfw_cb, run_label=None, run_variant="secondary", run_interactive=True) -> (system_box, run_btn | None)`.

- [ ] **Step 1: Add `_h3_section`**

Insert right after `_vr_section`:

```python
def _h3_section(title, step, nsfw_cb, run_label=None, run_variant="secondary", run_interactive=True):
    """Header + ⚙ toggle + hidden system-prompt editor (saved into system_prompt.json) + optional ▶ button."""
    with gr.Row():
        gr.Markdown(f"### {title}")
        gear = gr.Button("⚙", size="sm", scale=0, min_width=48)
    gear_open = gr.State(False)
    with gr.Column(visible=False) as gear_panel:
        system_box = gr.Textbox(label="System prompt", lines=12, value=lambda: _h3_system_text(step, False))
        save_btn = gr.Button("💾 Save to system_prompt.json", size="sm")
        gear_status = gr.HTML()
    run_btn = gr.Button(run_label, variant=run_variant, interactive=run_interactive) if run_label else None
    gear.click(vr_toggle_gear, [gear_open], [gear_open, gear_panel], queue=False)
    save_btn.click(lambda text, nsfw: h3_save_system(step, nsfw, text), [system_box, nsfw_cb], [gear_status],
                   queue=False)
    return system_box, run_btn
```

- [ ] **Step 2: Add the tab**

Insert after the Video Reel tab block (after the `vr_img.change(...)` call, before the `# Tab 5 — Models` banner):

```python
    # ════════════════════════════════════════════════════════════════════════
    #  Tab — Prompt Video Minimax H3
    # ════════════════════════════════════════════════════════════════════════
    with gr.Tab("🎞️ Prompt Video Minimax H3"):
        gr.Markdown(
            "Ảnh nhân vật (+ tối đa 3 ảnh phụ) và Story → hai phần prompt cho **MiniMax Hailuo H3**. "
            "Story trống: **Gợi ý 3 ý tưởng** → chọn → **Viết chi tiết** → **Generate**. "
            "⚙ sửa system prompt của từng bước trong `captioner/system_prompt.json`."
        )
        h3_log = gr.State([])
        h3_ideas_state = gr.State([])
        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                gr.Markdown("### Ảnh tham chiếu")
                h3_imgs = [gr.Image(type="pil", label="Picture 1 · Nhân vật (bắt buộc)", height=300)]
                with gr.Row():
                    h3_paths = [gr.Textbox(label="Đường dẫn Picture 1", placeholder=r"D:\photos\hero.png", scale=5)]
                    h3_browses = [gr.Button("📂", scale=0, min_width=48)]
                h3_roles = []
                with gr.Row():
                    for n, role in zip((2, 3, 4), H3_ROLE_CHOICES):
                        with gr.Column(min_width=120):
                            h3_imgs.append(gr.Image(type="pil", label=f"Ảnh phụ {n} (tùy chọn)", height=140))
                            h3_roles.append(gr.Dropdown(H3_ROLE_CHOICES, value=role, label="Vai trò"))
                            with gr.Row():
                                h3_paths.append(gr.Textbox(label="Đường dẫn", scale=4))
                                h3_browses.append(gr.Button("📂", scale=0, min_width=40))
                gr.Markdown("Không có ảnh phụ thì Picture 1 dùng cho cả nhân vật và bối cảnh.")
                with gr.Row():
                    h3_nsfw = gr.Checkbox(label="NSFW", value=False)
                    h3_total = gr.Number(value=0, precision=0, minimum=0, label="Tổng thời lượng (s), 0 = tự động")
                with gr.Accordion("Resize options", open=False):
                    h3_resize_mode = gr.Dropdown(
                        choices=["Adaptive image", "Fit image"], value="Adaptive image", label="Resize Mode"
                    )
                    with gr.Row():
                        h3_resize_width = gr.Slider(32, 2048, value=512, step=32, label="Resize Width")
                        h3_resize_height = gr.Slider(32, 2048, value=512, step=32, label="Resize Height")

                h3_bible_system, _ = _h3_section("1) Character bible", "bible", h3_nsfw)
                with gr.Accordion("Bible (tự tạo khi trống, sửa được)", open=False):
                    h3_bible = gr.Textbox(label="Character bible", lines=11, interactive=True)

                gr.Markdown("### 2) Story")
                h3_story = gr.Textbox(label="Story / kịch bản (VI hoặc EN)", lines=6,
                                      placeholder="Nhập nội dung video, hoặc để trống rồi bấm Gợi ý 3 ý tưởng")
                h3_ideas_system, h3_ideas_btn = _h3_section("Gợi ý ý tưởng", "ideas", h3_nsfw, "💡 Gợi ý 3 ý tưởng")
                h3_ideas_radio = gr.Radio(choices=[], label="3 ý tưởng")
                h3_ideas_status = gr.HTML()
                h3_expand_system, h3_expand_btn = _h3_section(
                    "Viết chi tiết ý tưởng", "expand", h3_nsfw, "✍ Viết chi tiết vào Story", run_interactive=False)
                h3_expand_status = gr.HTML()

            with gr.Column(scale=1):
                h3_shots_system, h3_gen_btn = _h3_section(
                    "3) Generate prompt", "shots", h3_nsfw, "▶ Generate prompt",
                    run_variant="primary", run_interactive=False)
                h3_gen_status = gr.HTML()
                h3_part1 = gr.Textbox(label="CHARACTER & SCENE DEFINITION", lines=12,
                                      interactive=True, buttons=["copy"])
                h3_part2 = gr.Textbox(label="SHOT STRUCTURE — MINIMAX H3", lines=20,
                                      interactive=True, buttons=["copy"])
                with gr.Row():
                    h3_saved = gr.Textbox(label="Saved file", interactive=False, scale=5)
                    h3_save_btn = gr.Button("💾 Lưu lại", scale=1)
                h3_log_box = gr.Textbox(label="Logger", lines=10, interactive=False, autoscroll=True)

        h3_media = [*h3_imgs, *h3_roles, *h3_paths, h3_resize_mode, h3_resize_width, h3_resize_height]
        h3_log_out = [h3_log, h3_log_box]
        h3_clear_out = [h3_bible, h3_ideas_radio, h3_ideas_state, h3_part1, h3_part2, h3_saved]

        h3_ideas_btn.click(h3_ideas,
                           [h3_bible, h3_bible_system, h3_ideas_system, h3_nsfw, h3_total, h3_log, *h3_media],
                           [h3_bible, h3_ideas_radio, h3_ideas_state, h3_ideas_status, *h3_log_out])
        h3_expand_btn.click(h3_expand,
                            [h3_ideas_radio, h3_ideas_state, h3_bible, h3_bible_system, h3_expand_system,
                             h3_nsfw, h3_log, *h3_media],
                            [h3_bible, h3_story, h3_expand_status, *h3_log_out])
        h3_gen_btn.click(h3_generate,
                         [h3_story, h3_bible, h3_bible_system, h3_shots_system, h3_nsfw, h3_total, h3_log, *h3_media],
                         [h3_bible, h3_part1, h3_part2, h3_saved, h3_gen_status, *h3_log_out])
        h3_save_btn.click(h3_save, [h3_part1, h3_part2, h3_saved, h3_paths[0], h3_nsfw, h3_log],
                          [h3_saved, h3_gen_status, *h3_log_out])

        h3_story.change(lambda s: gr.update(interactive=bool((s or "").strip())), [h3_story], [h3_gen_btn],
                        queue=False)
        h3_ideas_radio.change(lambda c: gr.update(interactive=c is not None), [h3_ideas_radio], [h3_expand_btn],
                              queue=False)
        h3_nsfw.change(h3_reload_systems, [h3_nsfw], [h3_ideas_system, h3_expand_system, h3_shots_system],
                       queue=False)
        for h3_img, h3_path, h3_browse_btn in zip(h3_imgs, h3_paths, h3_browses):
            h3_browse_btn.click(h3_browse, [h3_path], [h3_path])
            h3_path.change(h3_load_path, [h3_path, h3_log], [h3_img, *h3_log_out])
            h3_img.upload(lambda: "", None, [h3_path], queue=False)
            h3_img.clear(lambda: "", None, [h3_path], queue=False)
            h3_img.change(h3_clear, None, h3_clear_out, queue=False)
        for h3_role in h3_roles:
            h3_role.change(h3_clear, None, h3_clear_out, queue=False)
```

- [ ] **Step 3: Add the Help text**

In the Help tab markdown, after the three "Video Reel tab" bullet lines (before `### Troubleshooting`), add:

```markdown
### Prompt Video Minimax H3 tab
- **Picture 1** (character) is required; up to 3 extra pictures, each with a role (face sheet, scene, outfit/props). Paste a path or use 📂 so the result is saved next to Picture 1 as `<name>_H3_<sfw|nsfw>.txt`.
- Story empty → **Gợi ý 3 ý tưởng** (SFW/NSFW by the checkbox) → pick one → **Viết chi tiết** writes a timed story → **Generate prompt** fills CHARACTER & SCENE DEFINITION and SHOT STRUCTURE — MINIMAX H3.
- Max 12 s per shot: story timecodes or the total duration are turned into a shot plan by code (13–15 s compressed to 12 s, longer split). The Bible is built once from the pictures and can be edited.
- ⚙ edits each step's system prompt in `captioner/system_prompt.json` (temperature and max tokens live there too). The Logger shows every step, timing and saved file.
```

- [ ] **Step 4: Update README.md**

After the Video Reel section (after its "Edited system prompts apply immediately…" paragraph and before `---`), insert:

```markdown
---

## Prompt Video Minimax H3 tab

Character picture + story → the two parts of a MiniMax Hailuo H3 prompt.

| Step | Input | Output |
|------|-------|--------|
| 1) Character bible | Picture 1 + up to 3 role-tagged pictures (vision call) | 11 fields: NAME_HINT … STYLE. Built once, editable, cleared when a picture changes. |
| Gợi ý 3 ý tưởng | Bible (text only), SFW/NSFW | 3 ideas: name, logline, total, shot count, seconds per shot. Saved as `<name>_H3_<preset>_ideas.txt`. |
| Viết chi tiết | Chosen idea | A Vietnamese story with `MM:SS–MM:SS` lines written into the Story box. |
| Generate prompt | Bible + Story | `CHARACTER & SCENE DEFINITION` and `SHOT STRUCTURE — MINIMAX H3` (every shot with its own sound). Saved as `<name>_H3_<preset>.txt` next to Picture 1, or on the Desktop when Picture 1 has no path. |

Shots never exceed 12 s: the code builds the shot plan from story timecodes or the total duration (13–15 s compressed to 12 s, longer split into equal shots) and renumbers / caps the model's `[SHOT NN]` headers afterwards.

System prompts, user templates, temperature and max tokens for every step live in `captioner/system_prompt.json` (committed). ⚙ **Save** writes the edited system prompt back into that file; `git checkout captioner/system_prompt.json` restores the defaults.
```

In the project-structure tree, under `│   └── prompts.py            # Prompt templates`, change that line to `│   ├── prompts.py            # Prompt templates` and add after it:

```
│   ├── h3_prompts.py         # Prompt Video Minimax H3 helpers
│   └── system_prompt.json    # Editable H3 system prompts
```

(If the line above `prompts.py` in the tree already uses `└──`, keep the tree consistent: the last child of `captioner/` gets `└──`.)

- [ ] **Step 5: Run every self-check**

Run: `D:\Comfy\.venv\Scripts\python.exe test_h3_prompts.py`
Expected: `17 checks passed`
Run: `D:\Comfy\.venv\Scripts\python.exe test_h3_app.py`
Expected: `13 checks passed`
Run: `D:\Comfy\.venv\Scripts\python.exe test_video_prompts.py`
Expected: `8 checks passed`

- [ ] **Step 6: Manual check in the browser**

Start the app the usual way (`run_independent_env.bat`, or `D:\Comfy\.venv\Scripts\python.exe app.py --llm-dir D:\Comfy\ComfyUI\models\llm`), load a model in ⚙️ Setup (GGUF Qwen3-VL first, then an HF model), open the new tab and check:

1. Generate and "Viết chi tiết" start disabled; typing in Story enables Generate; clearing it disables it again.
2. 📂 on Picture 1 opens the Windows dialog; the picture appears; the Logger shows `loaded <path>`.
3. Picture 1 only, Story `00:00–00:14 — …`, Generate → Logger shows the bible step, `nén còn 12s`, `Saved → <folder>\<stem>_H3_sfw.txt`; both boxes are filled; the file sits next to the picture; every shot ends with `overall_soundscape` and `non_diegetic_music`.
4. Add a scene picture (role Bối cảnh) → outputs clear; Generate again → `<Subject 2>` uses `<Picture 2>, ref_image_1`.
5. Empty Story → Gợi ý → 3 ideas in the radio and `_ideas.txt` saved; Gợi ý again gives different ideas; pick one → Viết chi tiết fills Story with timecodes → Generate.
6. Tick NSFW → the ⚙ editors of ideas/expand/shots switch to the nsfw text; the saved file name uses `_nsfw`.
7. ⚙ Save on shots, then `git diff captioner/system_prompt.json` shows only that key changed; `git checkout captioner/system_prompt.json`.
8. Drag a picture straight into Picture 1 (no path) → Generate saves to `Desktop\h3_<timestamp>_sfw.txt`.
9. Edit a box, 💾 Lưu lại → the file content matches the boxes.
10. Single Image tab still captions as before (no temperature passed).

- [ ] **Step 7: Commit**

```bash
git add app.py README.md
git commit -m "Add the Prompt Video Minimax H3 tab UI and docs"
```
