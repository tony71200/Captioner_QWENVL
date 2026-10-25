# Text2Text Krea 2 / FLUX.2 Klein và tab H3 hai tab con (v4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tab Text2Text viết prompt đúng cấu trúc Krea 2 hoặc FLUX.2 Klein (có NSFW nam gợi cảm); tab Prompt Video MiniMax H3 có hai tab con (2a ảnh + element + gợi ý, 2b ảnh + video + element) cùng xuất prompt định dạng v4 của AutoShot Runner V5.3, có bảng kế hoạch shot và chuyển cảnh sửa được.

**Architecture:** Ba module thuần mới: `captioner/t2t_prompts.py` (prompt Text2Text), `captioner/h3_v4.py` (thời lượng, bảng kế hoạch, kiểm tra/sửa định dạng v4, cài đặt Runner), `captioner/video_segments.py` (PyAV + numpy: tìm điểm cắt, segment, gom shot, lấy frame). `captioner/system_prompt.json` có thêm bước `plan`, `segment` và bước `shots` viết lại theo v4. `app.py` chỉ là handler mỏng + giao diện `gr.Tabs` hai tab con.

**Tech Stack:** Python 3.13, Gradio 6.28 (`gr.Tabs`, `gr.Dataframe`, `gr.Video`), PyAV 18.1 (`av`), numpy, Pillow, llama-cpp-python / transformers qua `GGUFCaptioner` / `HFCaptioner` có sẵn.

**Spec:** `docs/superpowers/specs/2026-10-25-text2text-and-h3-v4-design.md`

## Global Constraints

- Interpreter: `.venv/Scripts/python.exe`, chạy mọi lệnh ở gốc repo `D:\001_Personal_Proj\Captioner_QWENVL`.
- Test theo kiểu repo: file `test_*.py` ở gốc, hàm `test_*` dùng `assert`, khối `__main__` chạy từng hàm và in `... OK`. Không pytest.
- Sửa `app.py` và `captioner/system_prompt.json` bằng công cụ sửa chính xác hoặc script Python có `encoding="utf-8"`, không dùng sed: các file có emoji, chữ Việt và dấu gạch ngang dài (U+2013, U+2014).
- Thời lượng shot hợp lệ: `4.4, 5, 5.8, 6.5, 7.2, 8, 8.7, 9.4, 10, 10.8, 11.5, 12` giây.
- Nhãn chuyển cảnh: `CONTINUE`, `OUTFIT CUT`, `SCENE CUT`.
- Header shot v4 chỉ có `DURATION`, `REF_VIDEO`, `REF_IMAGES`: `[SHOT 01 | DURATION: 5s | REF_VIDEO: none | REF_IMAGES: 1]`.
- Vai trò element: `Trang phục` → outfit, `Face sheet` → face, `Đạo cụ` → prop. Không có vai trò bối cảnh; bối cảnh lấy từ Picture 1.
- Phát hiện cắt: `d[i] > 20` và `d[i] > 3 × max(d trong ±8 frame trừ i)`, hai điểm cắt cách nhau ≥ 6 frame; setup mới khi `bg_diff ≥ 30`.
- Text2Text: một đoạn tiếng Anh 50–100 từ, bắt đầu bằng chủ thể, không quality tag, không cú pháp trọng số; giữ nguyên dòng negative ở chế độ file. NSFW: nam trưởng thành, gợi cảm, không lộ bộ phận sinh dục, không hành vi tình dục.
- Commit message kết thúc bằng dòng `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Không commit `models_library.json` và `run2.bat` (thay đổi của người dùng, đang chưa commit).

## File Structure

| File | Trách nhiệm |
|------|-------------|
| `captioner/t2t_prompts.py` (mới) | System/user prompt Text2Text cho Krea 2 và FLUX.2 Klein, khối NSFW |
| `test_t2t_prompts.py` (mới) | Test prompt Text2Text |
| `app.py` (sửa) | Handler + UI Text2Text; handler H3 2a/2b; UI hai tab con |
| `test_t2t_app.py` (mới) | Test handler Text2Text với model giả |
| `captioner/h3_v4.py` (mới) | Định dạng v4: thời lượng, bảng kế hoạch, `check_v4`, cài đặt Runner, bảng segment |
| `test_h3_v4.py` (mới) | Test `h3_v4` |
| `captioner/h3_prompts.py` (sửa) | Vai trò ảnh mới, khóa prompt mới; bỏ hàm định dạng cũ (Task 7) |
| `captioner/system_prompt.json` (sửa) | `bible` mới, `segment`, `plan_*`, `shots_*` v4 |
| `test_h3_prompts.py` (sửa) | Theo store mới; bỏ test định dạng cũ (Task 7) |
| `captioner/video_segments.py` (mới) | Đọc video, điểm cắt, segment, setup, gom shot, lấy frame |
| `test_video_segments.py` (mới) | Test với mảng tổng hợp và video tổng hợp |
| `test_h3_v4_app.py` (mới) | Test handler H3 2a/2b với model giả |
| `test_h3_app.py` (sửa) | Bỏ test handler cũ, cập nhật test ⚙ (Task 7) |
| `requirements.txt`, `README.md` (sửa) | Thêm `av`; tài liệu tab mới |

---

### Task 1: Prompt Text2Text cho Krea 2 / FLUX.2 Klein

**Files:**
- Create: `captioner/t2t_prompts.py`
- Test: `test_t2t_prompts.py`

**Interfaces:**
- Consumes: không có.
- Produces: `T2T_MODELS: tuple[str, str]` = `("Krea 2", "FLUX.2 Klein")`; `t2t_system(model: str, nsfw: bool) -> str`; `t2t_user(tags: str, model: str, nsfw: bool) -> str`. Model lạ → `ValueError`.

- [ ] **Step 1: Viết test hỏng**

`test_t2t_prompts.py`:
```python
"""Self-checks for the text2Text prompts. Run: .venv\\Scripts\\python.exe test_t2t_prompts.py"""
from captioner.t2t_prompts import T2T_MODELS, t2t_system, t2t_user


def _order(text, words):
    positions = [text.index(w) for w in words]
    return positions == sorted(positions)


def test_models():
    assert T2T_MODELS == ("Krea 2", "FLUX.2 Klein")


def test_krea_slot_order_and_rules():
    user = t2t_user(" 1boy, gym, sweat ", "Krea 2", False)
    assert _order(user, ["subject and what they are doing", "setting and spatial layout",
                         "framing and camera angle", "the light", "materials and textures", "mood and colour palette"])
    assert "50 to 100 words" in user and "(word:1.2)" in user and "masterpiece" in user
    assert user.endswith("Tags:\n1boy, gym, sweat")
    assert "Krea 2" in t2t_system("Krea 2", False) and "Adult male appeal" not in t2t_system("Krea 2", False)


def test_klein_slot_order():
    user = t2t_user("a", "FLUX.2 Klein", False)
    assert _order(user, ["subject with concrete details", "action or pose", "environment, foreground and background",
                         "the lighting", "style and mood", "camera, lens and composition"])
    assert "no prompt upsampling" in t2t_system("FLUX.2 Klein", False)


def test_nsfw_block():
    system = t2t_system("FLUX.2 Klein", True)
    assert "Adult male appeal" in system and "No exposed genitals, no sexual acts" in system
    assert "boy, teen" in system
    assert "adult male appeal rules" in t2t_user("a", "FLUX.2 Klein", True)
    assert "adult male appeal rules" not in t2t_user("a", "FLUX.2 Klein", False)


def test_unknown_model():
    for fn in (lambda: t2t_system("SDXL", False), lambda: t2t_user("a", "SDXL", False)):
        try:
            fn()
        except ValueError as e:
            assert "SDXL" in str(e)
        else:
            raise AssertionError("expected ValueError")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("test_t2t_prompts OK")
```

- [ ] **Step 2: Chạy test để thấy hỏng**

Run: `.venv/Scripts/python.exe test_t2t_prompts.py`
Expected: `ModuleNotFoundError: No module named 'captioner.t2t_prompts'`

- [ ] **Step 3: Viết module**

`captioner/t2t_prompts.py`:
```python
"""Prompts for the text2Text tab: Stable Diffusion tags → one prose prompt for Krea 2 or FLUX.2 Klein.

Pure (no gradio/torch) so test_t2t_prompts.py runs standalone. Slot orders follow the official
Krea 2 / BFL FLUX.2 guidance: subject first, one paragraph, no quality tags, no weights.
"""

T2T_MODELS = ("Krea 2", "FLUX.2 Klein")

_ORDER = {
    "Krea 2": (
        "the subject and what they are doing",
        "the setting and spatial layout",
        "the framing and camera angle",
        "the light: source, direction and quality",
        "materials and textures",
        "the mood and colour palette",
    ),
    "FLUX.2 Klein": (
        "the subject with concrete details",
        "the action or pose",
        "the environment, foreground and background",
        "the lighting: source, quality, direction and colour temperature",
        "the style and mood",
        "the camera, lens and composition",
    ),
}
_NOTES = {
    "Krea 2": "Krea 2 reads prose through a Qwen3-VL encoder: name the subject first and put style words early.",
    "FLUX.2 Klein": "FLUX.2 Klein has no prompt upsampling: write every detail it needs, concrete and specific.",
}
_NSFW = (
    "Adult male appeal (NSFW, sensual but not explicit):\n"
    "- The subject is an adult man. Never use words that suggest youth (boy, teen, schoolboy, young-looking).\n"
    "- Emphasise sensual male appeal: a toned or muscular physique, bare chest or an open shirt, defined abs, "
    "a light sheen of sweat, a confident seductive gaze, a relaxed provocative pose, warm intimate light.\n"
    "- Keep the clothing the tags give. If the tags name no clothing, he may be shirtless or wear fitted "
    "underwear or swimwear.\n"
    "- No exposed genitals, no sexual acts, no other person touching him."
)


def _check(model: str) -> None:
    if model not in _ORDER:
        raise ValueError(f"unknown target model: {model}")


def t2t_system(model: str, nsfw: bool) -> str:
    _check(model)
    text = (f"You turn Stable Diffusion tag prompts into one natural-language image prompt for {model}. "
            f"{_NOTES[model]} Return only the prompt.")
    return f"{text}\n\n{_NSFW}" if nsfw else text


def t2t_user(tags: str, model: str, nsfw: bool) -> str:
    _check(model)
    order = "\n".join(f"{i}. {slot}" for i, slot in enumerate(_ORDER[model], 1))
    lines = [
        f"Rewrite these tags as one {model} prompt.",
        "",
        "Order the content like this:",
        order,
        "",
        "Rules:",
        "- One English paragraph of 50 to 100 words, starting with the subject. Full sentences, no tag list, no markdown.",
        "- Use only details present in the tags. If the tags give no light, add one plausible light source.",
        "- Drop quality and meta tags such as masterpiece, best quality, highres, 8k, score tags and model or source tags.",
        "- No weighting syntax such as (word:1.2), no negative prompt, no quotes around the whole prompt.",
    ]
    if nsfw:
        lines.append("- Follow the adult male appeal rules from the system prompt.")
    lines += ["", "Tags:", (tags or "").strip()]
    return "\n".join(lines)
```

- [ ] **Step 4: Chạy test để thấy qua**

Run: `.venv/Scripts/python.exe test_t2t_prompts.py`
Expected: `test_t2t_prompts OK`

- [ ] **Step 5: Commit**

```bash
git add captioner/t2t_prompts.py test_t2t_prompts.py
git commit -m "Add text2Text prompts for Krea 2 and FLUX.2 Klein with an adult male NSFW block" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Tab Text2Text: icon, dropdown model, checkbox NSFW

**Files:**
- Modify: `app.py` (import; xóa `TEXT2TEXT_SYSTEM_PROMPT` và `_build_text2text_prompt` ~dòng 898-913; `convert_tags_to_caption` ~959; `convert_tags_file_to_caption_file` ~976; UI `with gr.Tab("text2Text")` ~1914-1965)
- Test: `test_t2t_app.py`

**Interfaces:**
- Consumes: Task 1 `T2T_MODELS`, `t2t_system`, `t2t_user`.
- Produces: `convert_tags_to_caption(tags_text, max_tokens, model="Krea 2", nsfw=False) -> (caption, status_html)`; `convert_tags_file_to_caption_file(file_obj, block_separator, positive_marker, negative_marker, max_tokens, model="Krea 2", nsfw=False) -> (path|None, status_html, log)`; tab label `📝 text2Text`; components `t2t_model`, `t2t_nsfw`.

- [ ] **Step 1: Viết test hỏng**

`test_t2t_app.py`:
```python
"""Self-checks for the text2Text handlers (fake model). Run: .venv\\Scripts\\python.exe test_t2t_app.py"""
import tempfile
from pathlib import Path

import app


class FakeCaptioner:
    def __init__(self, reply="A man stands in a gym."):
        self.reply, self.calls = reply, []

    def caption_text(self, user_prompt, max_new_tokens=512, system_prompt=None, temperature=None):
        self.calls.append((user_prompt, max_new_tokens, system_prompt))
        return self.reply


def _with(fake, fn, *args):
    app._captioner = fake
    try:
        return fn(*args)
    finally:
        app._captioner = None


def test_single_uses_model_and_nsfw():
    fake = FakeCaptioner()
    caption, status = _with(fake, app.convert_tags_to_caption, "1boy, gym", 256, "FLUX.2 Klein", True)
    assert caption == "A man stands in a gym." and "#4ade80" in status
    user, tokens, system = fake.calls[0]
    assert tokens == 256 and "FLUX.2 Klein" in system and "Adult male appeal" in system
    assert user.endswith("Tags:\n1boy, gym")


def test_single_defaults_to_krea_without_nsfw():
    fake = FakeCaptioner()
    _with(fake, app.convert_tags_to_caption, "1boy", 128)
    assert "Krea 2" in fake.calls[0][2] and "Adult male appeal" not in fake.calls[0][2]


def test_file_keeps_negative_line():
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "in.txt"
        src.write_text("Positive prompt: 1boy, gym\nNegative prompt: blurry\n\nPositive prompt: 1man, beach",
                       encoding="utf-8")
        fake = FakeCaptioner()
        out, status, log = _with(fake, app.convert_tags_file_to_caption_file, str(src), "\\n\\n",
                                 "Positive prompt:", "Negative prompt:", 256, "Krea 2", False)
        text = Path(out).read_text(encoding="utf-8")
    assert text.count("Positive prompt: A man stands in a gym.") == 2
    assert "Negative prompt: blurry" in text and len(fake.calls) == 2
    assert all("Krea 2" in c[2] for c in fake.calls)


def test_guard_and_tab_label():
    assert "Load a model" in app.convert_tags_to_caption("x", 64, "Krea 2", False)[1]
    labels = [getattr(b, "label", None) for b in app.demo.blocks.values()]
    assert "📝 text2Text" in labels and "text2Text" not in labels


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("test_t2t_app OK")
```

- [ ] **Step 2: Chạy test để thấy hỏng**

Run: `.venv/Scripts/python.exe test_t2t_app.py`
Expected: FAIL, `TypeError: convert_tags_to_caption() takes 2 positional arguments but 4 were given`.

- [ ] **Step 3: Sửa handler**

Trong `app.py`, thêm vào nhóm import của `captioner`:
```python
from captioner.t2t_prompts import T2T_MODELS, t2t_system, t2t_user
```
Xóa hằng `TEXT2TEXT_SYSTEM_PROMPT` và hàm `_build_text2text_prompt` (cả hai, ngay trên `_split_text2text_blocks`).

Thay `convert_tags_to_caption` bằng:
```python
def convert_tags_to_caption(tags_text, max_tokens, model="Krea 2", nsfw=False):
    if _captioner is None:
        return "", _badge("Load a model first.", "warning")
    tags_text = (tags_text or "").strip()
    if not tags_text:
        return "", _badge("Paste Stable Diffusion tags first.", "warning")
    try:
        caption = _captioner.caption_text(
            t2t_user(tags_text, model, bool(nsfw)),
            int(max_tokens),
            system_prompt=t2t_system(model, bool(nsfw)),
        )
        return caption, _badge(f"Tags converted to a {model} prompt.", "success")
    except Exception as e:
        return "", _badge(f"Text2Text error: {e}", "error")
```
Trong `convert_tags_file_to_caption_file`, đổi dòng định nghĩa thành
`def convert_tags_file_to_caption_file(file_obj, block_separator, positive_marker, negative_marker, max_tokens, model="Krea 2", nsfw=False):`
và thay lời gọi model trong vòng lặp bằng:
```python
            caption = _captioner.caption_text(
                t2t_user(positive_prompt, model, bool(nsfw)),
                int(max_tokens),
                system_prompt=t2t_system(model, bool(nsfw)),
            )
```

- [ ] **Step 4: Sửa giao diện**

Trong khối tab Text2Text:
- `with gr.Tab("text2Text"):` → `with gr.Tab("📝 text2Text"):`
- Markdown đầu tab thành:
```python
        gr.Markdown("""
## Tags → Krea 2 / FLUX.2 Klein prompt
Convert Stable Diffusion / booru-style tags into one natural-language prompt for the chosen model.
Load a model in **Setup** first, then use either direct paste or batch file conversion.
""")
```
- Ngay sau `t2t_tokens = gr.Slider(...)` thêm:
```python
                with gr.Row():
                    t2t_model = gr.Dropdown(list(T2T_MODELS), value=T2T_MODELS[0], label="Target model")
                    t2t_nsfw = gr.Checkbox(label="NSFW (male, sensual)", value=False)
```
- Wiring:
```python
        t2t_run.click(
            convert_tags_to_caption,
            inputs=[t2t_input, t2t_tokens, t2t_model, t2t_nsfw],
            outputs=[t2t_output, t2t_status],
        )
        t2t_file_run.click(
            convert_tags_file_to_caption_file,
            inputs=[t2t_file, t2t_block_separator, t2t_positive_marker, t2t_negative_marker, t2t_tokens,
                    t2t_model, t2t_nsfw],
            outputs=[t2t_file_output, t2t_file_status, t2t_file_log],
        )
```

- [ ] **Step 5: Chạy test để thấy qua, và test cũ không vỡ**

Run: `.venv/Scripts/python.exe test_t2t_app.py && .venv/Scripts/python.exe test_h3_app.py 2>&1 | grep -v httpx | tail -1`
Expected: `test_t2t_app OK`, rồi `16 checks passed`.

- [ ] **Step 6: Commit**

```bash
git add app.py test_t2t_app.py
git commit -m "text2Text tab: icon, Krea 2 / FLUX.2 Klein target model, NSFW checkbox" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Module định dạng v4 (`captioner/h3_v4.py`)

**Files:**
- Create: `captioner/h3_v4.py`
- Test: `test_h3_v4.py`

**Interfaces:**
- Consumes: `captioner.h3_prompts.clean_output(text) -> str` (đã có).
- Produces (mọi task sau dùng):
  - Hằng: `DURATIONS`, `LABELS`, `PLAN_COLUMNS = ["Shot","Duration","Source","REF_IMAGES","Into next","Seam pose","Content"]`, `SEGMENT_COLUMNS = ["#","Time","Seconds","Setup","Camera","Action","Outfit","Shot"]`, `SEGMENT_FIELDS = ("camera","action","outfit","location")`, `CHOOSE_SHOTS: str`, `PLAN_HELP: str`.
  - `snap_duration(x) -> float`, `fmt_duration(d) -> str` (`"4.4s"`, `"5s"`), `ts(seconds) -> str` (`"00:04.4"`, `"00:05"`).
  - Kế hoạch là `list[dict]`, mỗi dict có khóa `shot:int, duration:float, source:str, ref_images:str, next:str, seam:str, content:str`.
  - `new_row(shot, duration, source="", content="") -> dict`; `plan_from_story(segments: list[tuple[int,int,int]]) -> list[dict]`; `format_plan(plan) -> str`; `apply_model_plan(plan, text, n_pictures) -> (plan, warnings)`; `plan_to_rows(plan) -> list[list[str]]`; `rows_to_plan(rows, n_pictures) -> (plan, warnings)`; `norm_label(text) -> str`; `norm_ref_images(text, n_pictures) -> str`.
  - `runner_settings(plan, n_pictures, final_audio="", ref_video_hint="") -> str` (Markdown).
  - `check_v4(text, n_pictures, plan=None) -> (text, warnings)`; không có `[SHOT` → `ValueError("no [SHOT ...] block")`.
  - Segment (dict có `seg, start, end, seconds, setup` + `camera, action, outfit, location`): `parse_segment(text) -> dict`, `segments_to_rows(segs, shot_of) -> list[list]`, `apply_segment_rows(segs, rows) -> (segs, shot_of | None, warnings)`, `group_segments(segs, shot_of) -> list[list[dict]]`, `plan_from_video(groups) -> (plan, notes)`, `describe_segments(segs, shot_of) -> str`.

- [ ] **Step 1: Viết test hỏng**

`test_h3_v4.py`:
```python
"""Self-checks for the H3 v4 (Runner V5.3) format helpers. Run: .venv\\Scripts\\python.exe test_h3_v4.py"""
from captioner.h3_v4 import (
    DURATIONS, apply_model_plan, apply_segment_rows, check_v4, describe_segments, fmt_duration, format_plan,
    group_segments, new_row, norm_label, norm_ref_images, parse_segment, plan_from_story, plan_from_video,
    plan_to_rows, rows_to_plan, runner_settings, segments_to_rows, snap_duration, ts,
)


def test_durations_and_timecodes():
    assert snap_duration(2) == 4.4 and snap_duration(4.6) == 4.4 and snap_duration(4.7) == 5.0
    assert snap_duration(10.49) == 10.8 and snap_duration(30) == 12.0
    assert snap_duration(5.4) == 5.8  # tie (0.4 each side) → the longer one
    assert all(snap_duration(d) == d for d in DURATIONS)
    assert fmt_duration(4.4) == "4.4s" and fmt_duration(5.0) == "5s"
    assert ts(0) == "00:00" and ts(4.4) == "00:04.4" and ts(65) == "01:05" and ts(9.96) == "00:10"


def test_labels_and_refs():
    assert norm_label(" continue ") == "CONTINUE" and norm_label("outfit-cut") == "OUTFIT CUT"
    assert norm_label("scene cut") == "SCENE CUT" and norm_label("cut") == "SCENE CUT" and norm_label("??") == ""
    assert norm_ref_images("1, 3, 3, 9", 3) == "1,3" and norm_ref_images("", 2) == "1"


def test_plan_from_story_and_format():
    plan = plan_from_story([(0, 5, 5), (5, 7, 2)])
    assert [r["duration"] for r in plan] == [5.0, 4.4] and plan[1]["source"] == "story 00:05–00:07"
    plan[0].update(next="CONTINUE", seam="holds a smile", content="He smiles.")
    text = format_plan(plan)
    assert text.splitlines()[0] == ("SHOT 01 | DURATION: 5s | REF_IMAGES: 1 | SOURCE: story 00:00–00:05 | "
                                    "CONTENT: He smiles. | NEXT: CONTINUE | END POSE: holds a smile")
    assert text.splitlines()[1] == "SHOT 02 | DURATION: 4.4s | REF_IMAGES: 1 | SOURCE: story 00:05–00:07"


def test_apply_model_plan():
    plan = [new_row(1, 5), new_row(2, 5), new_row(3, 6.5)]
    raw = ("Here is the plan:\nSHOT 01 | REF_IMAGES: 1,2 | NEXT: continue | END: holds still | CONTENT: Walks in.\n"
           "SHOT 03 | REF_IMAGES: 1 | NEXT: SCENE CUT | CONTENT: Leaves.\n")
    out, warnings = apply_model_plan(plan, raw, 2)
    assert out[0]["ref_images"] == "1,2" and out[0]["next"] == "CONTINUE" and out[0]["seam"] == "holds still"
    assert out[1]["next"] == "SCENE CUT" and out[2]["next"] == "" and out[2]["content"] == "Leaves."
    assert "plan: model bỏ qua SHOT 02" in warnings and [r["duration"] for r in out] == [5.0, 5.0, 6.5]


def test_apply_model_plan_lets_the_model_choose_durations():
    out, warnings = apply_model_plan([], "SHOT 01 | DURATION: 6s | NEXT: CONTINUE\nSHOT 02 | CONTENT: Ends.", 1)
    assert [r["duration"] for r in out] == [5.8, 5.0] and out[0]["next"] == "CONTINUE"
    assert "SHOT 02: model không ghi DURATION → 5s" in warnings


def test_rows_round_trip_and_fixes():
    plan = [new_row(1, 5, "s1", "a"), new_row(2, 8, "s2", "b")]
    plan[0]["next"] = "OUTFIT CUT"
    rows = plan_to_rows(plan)
    assert rows[0] == ["SHOT 01", "5s", "s1", "1", "OUTFIT CUT", "", "a"]
    back, warnings = rows_to_plan(rows, 2)
    assert back == plan and warnings == []
    edited = [["SHOT 01", "6", "", "1, 5", "maybe", "", ""], ["", "", "", "", "", "", ""],
              ["SHOT 02", "x", "", "", "CONTINUE", "", ""]]
    fixed, warnings = rows_to_plan(edited, 2)
    assert [r["duration"] for r in fixed] == [5.8, 5.0] and fixed[0]["ref_images"] == "1"
    assert fixed[0]["next"] == "SCENE CUT" and fixed[1]["next"] == ""
    assert "SHOT 01: Duration 6s → 5.8s" in warnings and "SHOT 02: thiếu Duration → 5s" in warnings
    assert any("REF_IMAGES" in w for w in warnings) and any("SCENE CUT" in w for w in warnings)


def test_runner_settings():
    on = [new_row(1, 5), new_row(2, 5)]
    on[0]["next"] = "CONTINUE"
    text = runner_settings(on, 2, final_audio="clip.mp4", ref_video_hint="seg 1")
    assert "continuity_guide: ON" in text and "final_audio: clip.mp4" in text
    assert "Picture 1 → LoadImage #187" in text and "Picture 2 → LoadImage #188" in text
    assert "Tổng thời lượng: 10s (2 shot)" in text and "seg 1" in text
    mixed = [new_row(1, 5), new_row(2, 5), new_row(3, 5)]
    mixed[0]["next"], mixed[1]["next"] = "CONTINUE", "OUTFIT CUT"
    assert "continuity_guide: OFF — có cả CONTINUE" in runner_settings(mixed, 1)
    off = [new_row(1, 5), new_row(2, 5)]
    off[0]["next"] = "SCENE CUT"
    assert "continuity_guide: OFF\n" in runner_settings(off, 1) + "\n"


RAW = """Sure! Here is the prompt.
```text
subject_definitions:
<Subject 1> Ivan, an adult man: short black hair.
<Picture 1> is the identity, room, lighting and framing reference and the first outfit: grey tee.
style:
Photorealistic. No captions. He says "hi".
overall_soundscape:
Gym ambience.

[SHOT 1 | DURATION: 6s | PRIMARY CAMERA SETUP: medium, static]
00:02–00:04 — He waves at <Picture 2>.
00:05–00:09 — He turns.
00:09–00:11 — Too late.
overall_soundscape:
Footsteps.

[SHOT 7 | DURATION: 4.4s | REF_VIDEO: clip.mp4 | REF_IMAGES: 1]
Camera: close-up.
00:00–00:04.4 — He smiles.
non_diegetic_music:
Soft piano.
```"""


def test_check_v4_repairs():
    plan = [new_row(1, 5), new_row(2, 4.4)]
    text, warnings = check_v4(RAW, 2, plan)
    assert text.startswith("subject_definitions:") and "Sure" not in text and "```" not in text
    assert '"' not in text.split("[SHOT 01")[0]
    assert "[SHOT 01 | DURATION: 5s | REF_VIDEO: none | REF_IMAGES: 1,2]" in text
    assert "[SHOT 02 | DURATION: 4.4s | REF_VIDEO: none | REF_IMAGES: 1]" in text
    shot1 = text.split("[SHOT 01")[1].split("[SHOT 02")[0]
    assert "Camera: medium, static" in shot1
    assert "00:00–00:02 — He waves" in shot1 and "00:02–00:05 — He turns." in shot1 and "Too late" not in shot1
    assert text.count("overall_soundscape:") == 1 and text.count("non_diegetic_music:") == 1
    assert text.rstrip().endswith("overall_soundscape:\nFootsteps.\n\nnon_diegetic_music:\nSoft piano.")
    assert "GLOBAL: thiếu dòng vai trò <Picture 2>" in warnings
    assert "SHOT 01: thêm Picture 2 vào REF_IMAGES" in warnings
    assert any("DURATION 6s → 5s" in w for w in warnings) and any("ngoặc kép" in w for w in warnings)
    assert any("REF_VIDEO" in w for w in warnings)


def test_check_v4_without_plan_and_errors():
    text, warnings = check_v4("[SHOT 01 | DURATION: 7s]\n00:00–00:07 — Walks.", 1)
    assert "[SHOT 01 | DURATION: 7.2s | REF_VIDEO: none | REF_IMAGES: 1]" in text
    assert "00:00–00:07.2 — Walks." in text and "non_diegetic_music:\nNONE" in text
    assert "GLOBAL: thiếu subject_definitions" not in warnings  # an empty GLOBAL is only missing role lines
    try:
        check_v4("No shots here.", 1)
    except ValueError as e:
        assert "[SHOT" in str(e)
    else:
        raise AssertionError("expected ValueError")


SEGS = [
    {"seg": 1, "start": 0.0, "end": 1.73, "seconds": 1.73, "setup": "S1"},
    {"seg": 2, "start": 1.73, "end": 3.13, "seconds": 1.4, "setup": "S1"},
    {"seg": 3, "start": 3.13, "end": 9.13, "seconds": 6.0, "setup": "S2"},
]


def test_segments_parse_rows_and_plan():
    d = parse_segment("CAMERA: medium, static\n- ACTION: pulls shirt off\nOUTFIT: grey tee\n")
    assert d == {"camera": "medium, static", "action": "pulls shirt off", "outfit": "grey tee", "location": ""}
    segs = [dict(s, camera="c", action=f"a{s['seg']}", outfit="o", location="gym") for s in SEGS]
    rows = segments_to_rows(segs, [1, 1, 2])
    assert rows[0] == [1, "00:00–00:01.7", 1.73, "S1", "c", "a1", "o", 1]
    rows[1][5] = "sits down"
    rows[2][7] = ""
    edited, shot_of, warnings = apply_segment_rows(segs, rows)
    assert edited[1]["action"] == "sits down" and shot_of == [1, 1, 1] and len(warnings) == 1
    assert apply_segment_rows(segs, rows[:2])[1] is None
    plan, notes = plan_from_video(group_segments(segs, [1, 1, 2]))
    assert [r["duration"] for r in plan] == [4.4, 5.8]
    assert plan[0]["source"] == "seg 1 at 00:00, seg 2 at 00:01.7" and plan[0]["content"] == "a1; a2"
    assert any("tổng: video 9.13s" in n for n in notes)
    text = describe_segments(segs, [1, 1, 2])
    assert text.splitlines()[2].startswith("SEGMENT 3 (00:03.1–00:09.1, 6s, setup S2, shot 2): CAMERA: c")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("test_h3_v4 OK")
```

- [ ] **Step 2: Chạy test để thấy hỏng**

Run: `.venv/Scripts/python.exe test_h3_v4.py`
Expected: `ModuleNotFoundError: No module named 'captioner.h3_v4'`

- [ ] **Step 3: Viết module**

`captioner/h3_v4.py`:
```python
"""MiniMax H3 master prompts in the ComfyUI AutoShot Runner V5.3 format (GPT v4). Pure: no gradio/torch.

GLOBAL (everything before [SHOT 01) is copied into every shot by the Runner, so it holds only who and
where. Each shot header carries DURATION, REF_VIDEO and REF_IMAGES; one sound block follows the last shot.
"""
import re

from .h3_prompts import clean_output

DURATIONS = (4.4, 5.0, 5.8, 6.5, 7.2, 8.0, 8.7, 9.4, 10.0, 10.8, 11.5, 12.0)
LABELS = ("CONTINUE", "OUTFIT CUT", "SCENE CUT")
PLAN_COLUMNS = ["Shot", "Duration", "Source", "REF_IMAGES", "Into next", "Seam pose", "Content"]
SEGMENT_COLUMNS = ["#", "Time", "Seconds", "Setup", "Camera", "Action", "Outfit", "Shot"]
SEGMENT_FIELDS = ("camera", "action", "outfit", "location")
LOAD_IMAGE_NODES = ("#187", "#188", "#206", "#207")
SOUND_DEFAULTS = (("overall_soundscape", "Natural ambient sound. No dialogue."), ("non_diegetic_music", "NONE"))
CHOOSE_SHOTS = ("Choose the number of shots yourself. Add DURATION: <one of "
                + ", ".join(f"{d:g}s" for d in DURATIONS) + "> to every line, right after the shot number.")
PLAN_HELP = ("Into next: CONTINUE · OUTFIT CUT · SCENE CUT (để trống ở shot cuối). Duration (giây): "
             + " · ".join(f"{d:g}" for d in DURATIONS) + ". REF_IMAGES: ảnh 1 trước, vd 1,2.")

_NUM = re.compile(r"\d+(?:[.,]\d+)?")
PLAN_LINE = re.compile(r"(?mi)^[ \t>*-]*SHOT\s*0*(\d+)\s*\|(.*)$")
SHOT_HEAD = re.compile(r"(?mi)^[ \t]*\[\s*SHOT\s*\d+(?:\s*\|\s*([^\]\n]*))?\][ \t]*$")
SOUND_KEY = re.compile(r"(?i)^[ \t]*(overall_soundscape|non_diegetic_music)[ \t]*:[ \t]*(.*)$")
SECTION_KEY = re.compile(r"(?i)^[ \t]*(subject_definitions|style|scene_reference|summary|retention_analysis|"
                         r"detailed_description|overall_soundscape|non_diegetic_music)[ \t]*:")
TIMECODE = re.compile(r"^([ \t]*)(\d{1,2}):(\d{2}(?:\.\d+)?)\s*[–—-]\s*(\d{1,2}):(\d{2}(?:\.\d+)?)\s*[—–-]?\s*(.*)$")
PIC = re.compile(r"<Picture\s*(\d+)>")
QUOTES = re.compile(r"[\"“”]")
COPIED_PLACEHOLDER = re.compile(r"<(?!(?:Subject|Picture|Video|Audio)\s*\d+>)([^<>\n]{2,80})>")


# ── Durations and timecodes ──────────────────────────────────────────────────
def snap_duration(seconds) -> float:
    x = float(seconds)
    return min(DURATIONS, key=lambda d: (abs(d - x), -d))


def fmt_duration(d) -> str:
    return f"{float(d):g}s"


def ts(seconds) -> str:
    s = round(float(seconds), 1)
    m, rest = divmod(s, 60)
    rest = round(rest, 1)
    if rest == int(rest):
        return f"{int(m):02d}:{int(rest):02d}"
    return f"{int(m):02d}:{rest:04.1f}"


# ── Plan rows ────────────────────────────────────────────────────────────────
def new_row(shot, duration, source="", content="") -> dict:
    return {"shot": shot, "duration": snap_duration(duration), "source": source, "ref_images": "1",
            "next": "", "seam": "", "content": content}


def plan_from_story(segments) -> list:
    """[(story_start, story_end, duration)] from h3_prompts.shot_plan_from_* → plan rows with v4 durations."""
    return [new_row(i, d, source=f"story {ts(a)}–{ts(b)}") for i, (a, b, d) in enumerate(segments, 1)]


def format_plan(plan) -> str:
    lines = []
    for r in plan:
        line = f"SHOT {r['shot']:02d} | DURATION: {fmt_duration(r['duration'])} | REF_IMAGES: {r['ref_images']}"
        if r["source"]:
            line += f" | SOURCE: {r['source']}"
        if r["content"]:
            line += f" | CONTENT: {r['content']}"
        if r["next"]:
            line += f" | NEXT: {r['next']}"
        if r["seam"]:
            line += f" | END POSE: {r['seam']}"
        lines.append(line)
    return "\n".join(lines)


def norm_label(text) -> str:
    t = re.sub(r"[^A-Z]", "", str(text or "").upper())
    if not t:
        return ""
    if t.startswith("CONT"):
        return "CONTINUE"
    if t.startswith("OUT"):
        return "OUTFIT CUT"
    if t.startswith("SCENE") or t == "CUT":
        return "SCENE CUT"
    return ""


def norm_ref_images(text, n_pictures) -> str:
    nums = []
    for x in re.findall(r"\d+", str(text or "")):
        k = int(x)
        if 1 <= k <= n_pictures and k not in nums:
            nums.append(k)
    return ",".join(map(str, nums)) or "1"


def _fields(text) -> dict:
    out = {}
    for part in text.split("|"):
        key, sep, val = part.partition(":")
        if sep:
            out[key.strip().upper()] = val.strip()
    return out


def apply_model_plan(plan, text, n_pictures):
    """Fill REF_IMAGES / NEXT / END / CONTENT from the model's 'SHOT NN | KEY: value | …' lines.

    Durations and sources stay as the code planned them. An empty plan means the model chose the
    shots: rows are built from its lines, durations snapped to the v4 table.
    """
    lines = [(int(m.group(1)), _fields(m.group(2))) for m in PLAN_LINE.finditer(clean_output(text))]
    warnings = []
    if plan:
        out = [dict(r) for r in plan]
    else:
        out = []
        for i, (_, f) in enumerate(lines, 1):
            m = _NUM.search(f.get("DURATION", ""))
            if not m:
                warnings.append(f"SHOT {i:02d}: model không ghi DURATION → 5s")
            out.append(new_row(i, float(m.group(0).replace(",", ".")) if m else 5.0))
        lines = [(i, f) for i, (_, f) in enumerate(lines, 1)]
    seen = set()
    for k, f in lines:
        if not 1 <= k <= len(out) or k in seen:
            continue
        seen.add(k)
        r = out[k - 1]
        if "REF_IMAGES" in f:
            r["ref_images"] = norm_ref_images(f["REF_IMAGES"], n_pictures)
        if f.get("NEXT"):
            r["next"] = norm_label(f["NEXT"])
        r["seam"] = f.get("END") or f.get("END POSE") or r["seam"]
        r["content"] = f.get("CONTENT") or r["content"]
    missing = [f"SHOT {k:02d}" for k in range(1, len(out) + 1) if k not in seen]
    if missing:
        warnings.append("plan: model bỏ qua " + ", ".join(missing))
    for r in out[:-1]:
        if not r["next"]:
            r["next"] = "SCENE CUT"
            warnings.append(f"SHOT {r['shot']:02d}: thiếu NEXT → SCENE CUT")
    if out:
        out[-1]["next"] = ""
    return out, warnings


def plan_to_rows(plan) -> list:
    return [[f"SHOT {r['shot']:02d}", fmt_duration(r["duration"]), r["source"], r["ref_images"],
             r["next"], r["seam"], r["content"]] for r in plan]


def rows_to_plan(rows, n_pictures):
    """The edited plan table → plan dicts. Every fix is reported, never silent."""
    plan, warnings = [], []
    rows = [list(r) for r in (rows or []) if any(str(c).strip() for c in r)]
    for i, r in enumerate(rows, 1):
        r += [""] * (len(PLAN_COLUMNS) - len(r))
        tag = f"SHOT {i:02d}"
        m = _NUM.search(str(r[1]))
        raw = float(m.group(0).replace(",", ".")) if m else 5.0
        dur = snap_duration(raw)
        if not m:
            warnings.append(f"{tag}: thiếu Duration → 5s")
        elif abs(dur - raw) > 1e-6:
            warnings.append(f"{tag}: Duration {raw:g}s → {fmt_duration(dur)}")
        refs = norm_ref_images(r[3], n_pictures)
        if str(r[3]).strip() and refs != re.sub(r"\s", "", str(r[3])):
            warnings.append(f"{tag}: REF_IMAGES {r[3]} → {refs}")
        label = ""
        if i < len(rows):
            label = norm_label(r[4])
            if not label:
                label = "SCENE CUT"
                warnings.append(f"{tag}: nhãn nối '{r[4]}' không hợp lệ → SCENE CUT")
        plan.append({"shot": i, "duration": dur, "source": str(r[2]).strip(), "ref_images": refs,
                     "next": label, "seam": str(r[5]).strip(), "content": str(r[6]).strip()})
    return plan, warnings


def runner_settings(plan, n_pictures, final_audio="", ref_video_hint="") -> str:
    """Markdown shown next to the prompt. Never part of the prompt: the Runner copies GLOBAL into every shot."""
    labels = [r["next"] for r in plan[:-1]]
    if labels and all(label == "CONTINUE" for label in labels):
        guide = "ON (mọi mối nối là CONTINUE)"
    elif "CONTINUE" in labels:
        guide = "OFF — có cả CONTINUE và cắt cảnh: viết tư thế cuối/đầu khớp nhau, hoặc tách thành nhiều job"
    else:
        guide = "OFF"
    pictures = ", ".join(f"Picture {k} → LoadImage {LOAD_IMAGE_NODES[k - 1]}"
                         for k in range(1, min(n_pictures, len(LOAD_IMAGE_NODES)) + 1))
    lines = [
        "**Cài đặt Runner (AUTO SHOT RUNNER V5.3)**",
        "- split_mode: SHOT TAGS / AUTO PARAGRAPHS",
        f"- Ảnh: {pictures}",
        f"- continuity_guide: {guide}",
        f"- Tổng thời lượng: {sum(r['duration'] for r in plan):g}s ({len(plan)} shot)",
        f"- final_audio: {final_audio}" if final_audio else "- final_audio: (trống)",
        "- REF_VIDEO: none ở mọi shot",
    ]
    if ref_video_hint:
        lines.append(f"- Gợi ý REF_VIDEO 22f (nâng cao): {ref_video_hint}")
    return "\n".join(lines)


# ── check_v4 ─────────────────────────────────────────────────────────────────
def _cut_sound(text):
    """Remove overall_soundscape / non_diegetic_music sections → (text, {key: value})."""
    found, keep, current = {}, [], None
    for line in text.splitlines():
        m = SOUND_KEY.match(line)
        if m:
            current = m.group(1).lower()
            found[current] = [m.group(2).strip()] if m.group(2).strip() else []
            continue
        if current and not line.strip():
            current = None
            continue
        if current and not SECTION_KEY.match(line) and not SHOT_HEAD.match(line) and not TIMECODE.match(line):
            found[current].append(line.strip())
            continue
        current = None
        keep.append(line)
    return "\n".join(keep), {k: " ".join(v).strip() for k, v in found.items() if " ".join(v).strip()}


def _secs(minutes, seconds) -> float:
    return int(minutes) * 60 + float(seconds)


def _fix_timeline(body, duration, tag, warnings):
    lines = body.splitlines()
    marks = []
    for i, line in enumerate(lines):
        m = TIMECODE.match(line)
        if m:
            marks.append([i, _secs(m[2], m[3]), _secs(m[4], m[5]), m[6].strip(), m[1]])
    if not marks:
        return body
    offset = marks[0][1]
    if offset > 0:
        warnings.append(f"{tag}: timeline bắt đầu {ts(offset)} → 00:00")
        for mk in marks:
            mk[1] -= offset
            mk[2] -= offset
    drop = {mk[0] for mk in marks if mk[1] >= duration - 1e-6}
    if drop:
        warnings.append(f"{tag}: bỏ {len(drop)} dòng timeline sau {fmt_duration(duration)}")
    kept = [mk for mk in marks if mk[0] not in drop]
    for a, b in zip(kept, kept[1:]):
        if abs(b[1] - a[2]) > 1e-6:
            warnings.append(f"{tag}: nối timecode {ts(a[2])} → {ts(b[1])}")
            b[1] = a[2]
    if kept and abs(kept[-1][2] - duration) > 1e-6:
        warnings.append(f"{tag}: timeline kết thúc {ts(kept[-1][2])} → {ts(duration)}")
        kept[-1][2] = duration
    for i, a, b, text, indent in kept:
        lines[i] = f"{indent}{ts(a)}–{ts(b)} — {text}"
    return "\n".join(line for i, line in enumerate(lines) if i not in drop)


def _insert_camera(body, camera):
    lines = body.splitlines()
    at = 0
    while at < len(lines) and re.match(r"(?i)\s*(Outfit in this shot|Shot continues)", lines[at]):
        at += 1
    lines.insert(at, camera)
    return "\n".join(lines)


def check_v4(text, n_pictures, plan=None):
    """Repair a model answer into a Runner V5.3 master prompt → (text, warnings). See spec §5.5."""
    text = COPIED_PLACEHOLDER.sub(r"\1", clean_output(text))
    heads = list(SHOT_HEAD.finditer(text))
    if not heads:
        raise ValueError("no [SHOT ...] block")
    warnings = []
    head = text[:heads[0].start()]
    start = re.search(r"(?mi)^[ \t]*subject_definitions[ \t]*:", head)
    if start:
        head = head[start.start():]
    elif head.strip():
        warnings.append("GLOBAL: thiếu subject_definitions")
        head = ""
    head, sound = _cut_sound(head)
    if sound:
        warnings.append("GLOBAL: chuyển khối âm thanh xuống sau shot cuối")
    if QUOTES.search(head):
        head = QUOTES.sub("", head)
        warnings.append("GLOBAL: xóa dấu ngoặc kép (Runner coi là lời thoại)")
    for k in range(1, n_pictures + 1):
        if not re.search(rf"(?m)^[ \t]*<Picture {k}>", head):
            warnings.append(f"GLOBAL: thiếu dòng vai trò <Picture {k}>")
    use_plan = bool(plan) and len(plan) == len(heads)
    if plan and not use_plan:
        warnings.append(f"model viết {len(heads)} shot, kế hoạch có {len(plan)} shot")
    shots = []
    for i, m in enumerate(heads, 1):
        tag = f"SHOT {i:02d}"
        end = heads[i].start() if i < len(heads) else len(text)
        fields, extras = {}, []
        for part in (m.group(1) or "").split("|"):
            key, sep, val = part.partition(":")
            key = key.strip().upper()
            if key in ("DURATION", "REF_VIDEO", "REF_IMAGES"):
                fields[key] = val.strip()
            elif sep and val.strip():
                extras.append(val.strip())
        row = plan[i - 1] if use_plan else None
        given = _NUM.search(fields.get("DURATION", ""))
        given = float(given.group(0).replace(",", ".")) if given else None
        duration = row["duration"] if row else snap_duration(given if given is not None else 5.0)
        if given is not None and abs(given - duration) > 1e-6:
            warnings.append(f"{tag}: DURATION {given:g}s → {fmt_duration(duration)}")
        if fields.get("REF_VIDEO", "none").lower() != "none":
            warnings.append(f"{tag}: REF_VIDEO {fields['REF_VIDEO']} → none")
        refs = norm_ref_images(fields.get("REF_IMAGES") or (row["ref_images"] if row else "1"), n_pictures)
        body, shot_sound = _cut_sound(text[m.end():end])
        sound.update(shot_sound)  # the last block found wins
        body = body.strip()
        used = sorted({int(k) for k in PIC.findall(body)})
        unknown = [k for k in used if k > n_pictures or k < 1]
        if unknown:
            warnings.append(f"{tag}: nhắc Picture {', '.join(map(str, unknown))} không tồn tại")
        missing = [k for k in used if 1 <= k <= n_pictures and str(k) not in refs.split(",")]
        if missing:
            refs = ",".join(refs.split(",") + [str(k) for k in missing])
            warnings.append(f"{tag}: thêm Picture {', '.join(map(str, missing))} vào REF_IMAGES")
        if extras and not re.search(r"(?mi)^[ \t]*Camera\s*:", body):
            body = _insert_camera(body, "Camera: " + "; ".join(extras))
            warnings.append(f"{tag}: chuyển phần header thừa vào dòng Camera:")
        body = _fix_timeline(body, duration, tag, warnings)
        shots.append(f"[SHOT {i:02d} | DURATION: {fmt_duration(duration)} | REF_VIDEO: none | REF_IMAGES: {refs}]"
                     f"\n{body}")
    for key, default in SOUND_DEFAULTS:
        if key not in sound:
            warnings.append(f"thêm {key} mặc định")
            sound[key] = default
    tail = f"overall_soundscape:\n{sound['overall_soundscape']}\n\nnon_diegetic_music:\n{sound['non_diegetic_music']}"
    global_text = f"{head.strip()}\n\n" if head.strip() else ""
    return global_text + "\n\n".join(shots) + f"\n\n{tail}\n", warnings


# ── Video segments ───────────────────────────────────────────────────────────
def parse_segment(text) -> dict:
    text = clean_output(text)
    out = {}
    for f in SEGMENT_FIELDS:
        m = re.search(rf"(?mi)^[ \t*-]*{f}\s*:\s*(.+)$", text)
        out[f] = m.group(1).strip() if m else ""
    return out


def segments_to_rows(segs, shot_of) -> list:
    return [[s["seg"], f"{ts(s['start'])}–{ts(s['end'])}", s["seconds"], s["setup"], s.get("camera", ""),
             s.get("action", ""), s.get("outfit", ""), n] for s, n in zip(segs, shot_of)]


def apply_segment_rows(segs, rows):
    """Edited segment table → (segments with the edited text, shot number per segment, warnings).

    shot_of is None when the table no longer matches the video (rows added/removed): the caller regroups.
    A new shot starts wherever the Shot value changes, so shots are always contiguous runs.
    """
    segs = [dict(s) for s in segs]
    rows = [list(r) for r in (rows or [])]
    warnings = []
    if len(rows) != len(segs):
        warnings.append(f"bảng segment có {len(rows)} dòng, video có {len(segs)} segment: dùng cách gom mặc định")
        return segs, None, warnings
    nums, prev = [], None
    for i, (s, r) in enumerate(zip(segs, rows), 1):
        r += [""] * (len(SEGMENT_COLUMNS) - len(r))
        s["camera"], s["action"], s["outfit"] = (str(x).strip() for x in r[4:7])
        m = re.search(r"\d+", str(r[7]))
        if m:
            prev = int(m.group(0))
        elif prev is None:
            prev = 1
        else:
            warnings.append(f"segment {i}: thiếu Shot → theo segment trước")
        nums.append(prev)
    shot_of, shot = [], 0
    for i, n in enumerate(nums):
        if i == 0 or n != nums[i - 1]:
            shot += 1
        shot_of.append(shot)
    return segs, shot_of, warnings


def group_segments(segs, shot_of) -> list:
    groups = {}
    for s, n in zip(segs, shot_of):
        groups.setdefault(n, []).append(s)
    return [groups[k] for k in sorted(groups)]


def plan_from_video(groups):
    """Shots of grouped segments → (plan rows, notes). SOURCE lists each segment's start inside its shot."""
    plan, notes, video_total, shot_total = [], [], 0.0, 0.0
    for i, group in enumerate(groups, 1):
        secs = round(sum(s["seconds"] for s in group), 2)
        dur = snap_duration(secs)
        start = group[0]["start"]
        source = ", ".join(f"seg {s['seg']} at {ts(s['start'] - start)}" for s in group)
        content = "; ".join(s.get("action", "") for s in group if s.get("action"))
        plan.append(new_row(i, dur, source=source, content=content))
        if secs > 12:
            notes.append(f"SHOT {i:02d}: video {secs:.2f}s dài hơn 12s → 12s, nên tách shot")
        elif abs(dur - secs) > 0.05:
            notes.append(f"SHOT {i:02d}: video {secs:.2f}s → {fmt_duration(dur)}")
        video_total += secs
        shot_total += dur
    if groups:
        notes.append(f"tổng: video {video_total:.2f}s, shot {shot_total:g}s (lệch {shot_total - video_total:+.2f}s)")
    return plan, notes


def describe_segments(segs, shot_of) -> str:
    return "\n".join(
        f"SEGMENT {s['seg']} ({ts(s['start'])}–{ts(s['end'])}, {s['seconds']:g}s, setup {s['setup']}, shot {n}): "
        f"CAMERA: {s.get('camera', '')} | ACTION: {s.get('action', '')} | OUTFIT: {s.get('outfit', '')} | "
        f"LOCATION: {s.get('location', '')}"
        for s, n in zip(segs, shot_of))
```

- [ ] **Step 4: Chạy test để thấy qua**

Run: `.venv/Scripts/python.exe test_h3_v4.py`
Expected: `test_h3_v4 OK`. Nếu một assert về chuỗi cảnh báo cụ thể lệch do lỗi đánh máy trong test, sửa test cho khớp code (không đổi hành vi) và ghi vào report.

- [ ] **Step 5: Commit**

```bash
git add captioner/h3_v4.py test_h3_v4.py
git commit -m "Add h3_v4: Runner V5.3 durations, plan table, check_v4 repair, Runner settings, segment helpers" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Store prompt v4 và vai trò ảnh mới

**Files:**
- Modify: `captioner/h3_prompts.py` (`ROLE_LABELS`, `ROLE_TEXT`, `build_references`, `describe_references`, `H3_KEYS`, `step_key`)
- Modify: `captioner/system_prompt.json` (bằng script ở Step 3)
- Modify: `test_h3_prompts.py` (test vai trò, test JSON), `test_h3_app.py` (hai assert về chữ vai trò cũ)

**Interfaces:**
- Consumes: không có.
- Produces:
  - `ROLE_LABELS = {"Trang phục": "outfit", "Face sheet": "face", "Đạo cụ": "prop"}` (app dùng `list(ROLE_LABELS)` làm lựa chọn dropdown).
  - `build_references(slots) -> list[{"n","role","slot"}]` (bỏ khóa `ref`).
  - `describe_references(refs) -> str`, mỗi dòng `<Picture n> = <role text>.`.
  - `H3_KEYS = ("bible","segment","ideas_sfw","ideas_nsfw","expand_sfw","expand_nsfw","plan_sfw","plan_nsfw","shots_sfw","shots_nsfw")`; `NEUTRAL_STEPS = ("bible","segment")`; `step_key("segment", True) == "segment"`.
  - Bước JSON `segment` (placeholders `{segment} {start} {end} {seconds} {times}`), `plan_*` (`{bible} {references} {story} {shot_plan}`), `shots_*` (`{bible} {references} {story} {shot_plan}`).

- [ ] **Step 1: Sửa test trước**

Trong `test_h3_prompts.py`:
- Thay `test_build_references_numbers_only_filled_slots` và `test_describe_references` bằng:
```python
def test_build_references_numbers_only_filled_slots():
    assert build_references(["character", None, "outfit", "face"]) == [
        {"n": 1, "role": "character", "slot": 0},
        {"n": 2, "role": "outfit", "slot": 2},
        {"n": 3, "role": "face", "slot": 3},
    ]


def test_describe_references():
    text = describe_references(build_references(["character", "outfit", "face", "prop"]))
    assert text.splitlines() == [
        "<Picture 1> = the identity, room, lighting and framing reference and the first outfit; "
        "the location comes from this picture.",
        "<Picture 2> = a cropped garment reference only; copy only the garment, not the body, skin, pose, "
        "lighting or background.",
        "<Picture 3> = a face reference of the same person as <Picture 1>; copy only facial features.",
        "<Picture 4> = a prop reference only; copy only the object, not the hands, body or background.",
    ]
    assert "ref_image" not in text
```
- Thay `STEP_PLACEHOLDERS` bằng:
```python
STEP_PLACEHOLDERS = {
    "bible": ["{references}"],
    "segment": ["{segment}", "{start}", "{end}", "{seconds}", "{times}"],
    "ideas": ["{bible}", "{references}", "{duration_rule}"],
    "expand": ["{bible}", "{references}", "{idea}", "{shot_plan}"],
    "plan": ["{bible}", "{references}", "{story}", "{shot_plan}"],
    "shots": ["{bible}", "{references}", "{story}", "{shot_plan}"],
}
```
- Trong `test_system_prompt_json_shape`, thay hai vòng `for key in ("ideas_nsfw", ...)` và `for key in ("shots_sfw", ...)` bằng:
```python
    for key in ("ideas_nsfw", "expand_nsfw", "plan_nsfw", "shots_nsfw"):
        assert "minor" in data[key]["system"], key
    for key in ("shots_sfw", "shots_nsfw"):
        for marker in ("[SHOT 01 | DURATION:", "REF_IMAGES", "Shot continues from the final frame of the previous shot:",
                       "Outfit in this shot:", "overall_soundscape", "non_diegetic_music", "hard cut to"):
            assert marker in data[key]["system"], (key, marker)
        assert "ref_image_0" not in data[key]["system"]
    for key in ("plan_sfw", "plan_nsfw"):
        for marker in ("CONTINUE", "OUTFIT CUT", "SCENE CUT", "SHOT 01 | REF_IMAGES: 1 | NEXT:"):
            assert marker in data[key]["system"], (key, marker)
    for marker in ("CAMERA:", "ACTION:", "OUTFIT:", "LOCATION:"):
        assert marker in data["segment"]["system"]
```
- Trong `test_step_key` thêm `assert step_key("segment", True) == "segment"` và `assert step_key("plan", False) == "plan_sfw"`.

Trong `test_h3_app.py`:
- Ở `test_generate_runs_bible_then_shots_and_saves_next_to_picture1`, đổi `assert "Also the setting" in user` thành `assert "the location comes from this picture" in user`.
- Ở `test_generate_sends_extra_pictures_in_order`, đổi `assert "<Picture 2>, ref_image_1 = face reference sheet" in user` thành `assert "<Picture 2> = a cropped garment reference only" in user` (ô phụ đầu tiên giờ mặc định là "Trang phục").

- [ ] **Step 2: Chạy test để thấy hỏng**

Run: `.venv/Scripts/python.exe test_h3_prompts.py`
Expected: FAIL ở `test_build_references_numbers_only_filled_slots` (còn khóa `ref`).

- [ ] **Step 3: Sửa `h3_prompts.py`**

Thay khối `ROLE_LABELS` … `describe_references` bằng:
```python
ROLE_LABELS = {"Trang phục": "outfit", "Face sheet": "face", "Đạo cụ": "prop"}
ROLE_TEXT = {
    "character": "the identity, room, lighting and framing reference and the first outfit; "
                 "the location comes from this picture",
    "outfit": "a cropped garment reference only; copy only the garment, not the body, skin, pose, "
              "lighting or background",
    "face": "a face reference of the same person as <Picture 1>; copy only facial features",
    "prop": "a prop reference only; copy only the object, not the hands, body or background",
}


def build_references(slots):
    """slots[0] = Picture 1 role ('character'); later items are element roles or None when empty."""
    refs = []
    for slot, role in enumerate(slots):
        if role:
            refs.append({"n": len(refs) + 1, "role": role, "slot": slot})
    return refs


def describe_references(refs) -> str:
    return "\n".join(f"<Picture {r['n']}> = {ROLE_TEXT[r['role']]}." for r in refs)
```
Thay `H3_KEYS` và `step_key`:
```python
H3_KEYS = ("bible", "segment", "ideas_sfw", "ideas_nsfw", "expand_sfw", "expand_nsfw",
           "plan_sfw", "plan_nsfw", "shots_sfw", "shots_nsfw")
NEUTRAL_STEPS = ("bible", "segment")
H3_FIELDS = ("system", "user", "temperature", "max_tokens")


def step_key(step: str, nsfw: bool) -> str:
    return step if step in NEUTRAL_STEPS else f"{step}_{'nsfw' if nsfw else 'sfw'}"
```

- [ ] **Step 4: Ghi prompt mới vào `system_prompt.json`**

Lưu script sau ra file tạm trong scratchpad (vd `$TEMP/h3_v4_prompts.py`) và chạy `.venv/Scripts/python.exe <file>` từ gốc repo. Script đọc khối nội dung SFW/NSFW đang có trong `shots_*` rồi ghi các bước mới:
```python
import json
from pathlib import Path

path = Path("captioner/system_prompt.json")
data = json.loads(path.read_text(encoding="utf-8"))


def block(key):
    s = data[key]["system"]
    a = s.index("Content rules")
    return s[a:s.index("Reference rules")].strip()


CONTENT = {"sfw": block("shots_sfw"), "nsfw": block("shots_nsfw")}

data["bible"]["system"] = """You are a visual analyst for a video pre-production tool. You receive one or more reference pictures, attached in the order of the REFERENCE PICTURES list, each with a role. Write a CHARACTER BIBLE that a video model will use to keep the character, outfit and setting consistent across many shots.

Rules:
- Describe only what is visible. Do not guess names, real identities or personal data. Never say the person resembles a real or famous person.
- Use each picture only for its role. Face and body come from <Picture 1> and the face reference, if one is listed. OUTFIT is what the person wears in <Picture 1>. Garment and prop references go under ACCESSORIES as "garment reference: ..." or "prop reference: ...". The setting always comes from <Picture 1>.
- Plain English, short concrete phrases, no opinions, no story.
- If something is not visible, write "not visible".
- Output exactly these 11 fields, one per line, in this order. Nothing before the first field, nothing after the last field, no markdown.

NAME_HINT: a short neutral label for the character, e.g. "man with short black hair"
AGE_APPEARANCE: apparent age range, then "adult" if the person clearly looks 18 or older, or "minor" if the person may be under 18
FACE: face shape, eyes, eyebrows, nose, lips, skin tone, facial hair, expression
HAIR: color, length, style
BODY: build, proportions, notable features
OUTFIT: every visible clothing item in <Picture 1> with color, material and fit
ACCESSORIES: jewelry, footwear, and what each garment or prop reference shows
SETTING: location in <Picture 1>, architecture, furniture, floor, walls, what is on the left and right of the frame
LIGHTING: direction, softness, color temperature
CAMERA_FRAMING: shot size, angle, pose, aspect ratio
STYLE: photographic or illustrated, color palette, mood"""

data["segment"] = {
    "system": """You describe one segment of a reference video for a video director. You receive 3 frames from the segment, in time order, with their timestamps. Describe only what is visible. The person in the video is a stand-in: never describe their face or identity, and ignore any text, sign, sticker, logo or censor bar.

Output exactly these 4 lines and nothing else:
CAMERA: shot size (extreme close-up, close-up, medium close-up, medium, medium-full, full, wide), height and angle (eye level, chest height, low, high; front, three-quarter, profile, from behind), movement (static, hand-held sway, pan, tilt, push-in, pull-back, tracking, orbit), subject position in frame
ACTION: the person's movements in order, with limbs and directions (camera-left / camera-right), one short clause per beat
OUTFIT: what the person wears
LOCATION: the place in a few words (indoor or outdoor, room type, main furniture, light)""",
    "user": "SEGMENT {segment}: {start}–{end} ({seconds}s). Frames attached in this order, at {times}.\n\nDescribe this segment now.",
    "temperature": 0.2,
    "max_tokens": 300,
}

PLAN = """You are the shot planner for the MiniMax H3 AutoShot Runner. You receive a CHARACTER BIBLE, the REFERENCE PICTURES, a STORY (story beats or analysed reference-video segments) and the SHOT LIST with fixed durations. For every shot decide which pictures it uses and how it joins the next shot.

{content}

Rules:
- REF_IMAGES: picture 1 first, then only the pictures this shot uses: a garment reference only in the shots where that garment is worn, a prop reference only where the prop appears, the face reference when the face is close to the camera. Examples: 1 / 1,2 / 1,3.
- NEXT, for every shot except the last:
  - CONTINUE: the same take goes on (same place, same outfit, the movement continues).
  - OUTFIT CUT: a new outfit by a cut, same place and framing.
  - SCENE CUT: a new place, time or camera setup.
- END: the pose at the last frame of this shot, so the next shot can start from it. Before CONTINUE: a clear held pose with the person fully visible, nothing mid-blur. Before OUTFIT CUT: a clean accent gesture (arm sweep, turn) still in the old outfit.
- CONTENT: what happens in this shot, one short English sentence.
- Keep the number of shots, their order and their durations.
- Output one line per shot and nothing else, exactly like this:
SHOT 01 | REF_IMAGES: 1 | NEXT: CONTINUE | END: <pose> | CONTENT: <sentence>
The last shot has no NEXT and no END field."""

PLAN_USER = ("CHARACTER BIBLE:\n{bible}\n\nREFERENCE PICTURES:\n{references}\n\nSTORY:\n{story}\n\n"
             "SHOT LIST:\n{shot_plan}\n\nWrite the plan lines now.")

SHOTS = """You are the MiniMax H3 shot director. You write the master prompt that the ComfyUI AutoShot Runner V5.3 renders shot by shot. You receive a CHARACTER BIBLE, the REFERENCE PICTURES, a STORY (story beats or analysed reference-video segments) and the SHOT PLAN with each shot's DURATION, REF_IMAGES, CONTENT, the transition into the next shot (NEXT) and its END POSE.

{content}

Format contract:
- GLOBAL is everything before [SHOT 01. The Runner copies it into every shot, so it says only who and where: subject_definitions, one role line per picture, scene_reference, style, retention_analysis. No actions, no story summary, no quotes, no sound sections in GLOBAL.
- subject_definitions describes the person without picture tags. Then one role line per picture, each starting with its tag, for example "<Picture 2> is a cropped garment reference only: <garment>. Copy only the <garment>, not the body, skin sheen, pose, lighting or background of <Picture 2>."
- The location always comes from <Picture 1>: write scene_reference from it and end it with "The room, furniture, perspective and light stay unchanged in every shot." unless the plan has a SCENE CUT to another place.
- style ends with "No captions, emojis, stickers, signs, logos, subtitles or watermarks."
- One [SHOT NN | DURATION: <from the plan> | REF_VIDEO: none | REF_IMAGES: <from the plan>] block per plan line, in order, with the plan's DURATION and REF_IMAGES. The header carries nothing else.
- A shot mentions only the pictures in its own REF_IMAGES.
- Shot body: "Outfit in this shot: ..." when outfits change, then "Camera: <shot size>, <height and angle>, <movement>, <subject position>.", then timecoded lines like "00:00–00:02 — <action>" that start at 00:00, have no gaps and end exactly at the DURATION. Use one decimal, such as 00:04.4, when the DURATION has one. One clear beat per 1–2 seconds.
- Transitions, from the plan's NEXT field:
  - Before CONTINUE: end on the END POSE, a clear held pose with the person fully visible. The next shot starts with "Shot continues from the final frame of the previous shot:" and the same pose, outfit and framing; its first beat continues that movement.
  - Before OUTFIT CUT: keep the last 1–2 seconds as one continuous take that ends on the END POSE gesture, still in the old outfit. The next shot starts with "Outfit in this shot: only <garment> from <Picture k>, from the first frame; <other outfits> never appear." and its first beat lands from that same gesture.
  - Before SCENE CUT: any natural end. The next shot opens with its own Camera line and an establishing beat and never refers to the previous frame.
- When a plan line lists several video segments ("seg 2 at 00:01.7"), the shot is a montage: write "At 00:01.7, hard cut to <new framing and action>" at each segment start.
- After the last shot write exactly one sound block: "overall_soundscape:" with the shared ambience and "No dialogue." unless the story has spoken lines, then "non_diegetic_music:" with NONE unless the story asks for music. Shot-specific sounds go inside the timecoded lines.
- Spoken lines only when the STORY has them, in double quotes, inside the shot where they are spoken. Never invent dialogue.
- When the STORY comes from a reference video, copy only its actions, timing and camera. Never copy the person, their face, any text, sign, sticker, logo or censor bar from the video. The person is always <Subject 1> from the pictures.
- English for everything in the prompt. Observable actions, camera and physics, present tense. Correct limbs and hands; feet carry the weight.
- Words in angle brackets below are instructions: replace them with real words and drop the brackets. Only <Subject 1> and <Picture N> keep angle brackets.
- Output only the master prompt. No introduction, no explanation, no markdown, no code fences.

Master prompt format:

subject_definitions:
<Subject 1> <short name>, an adult <description>: <face, hair, build, skin; fixed traits only>. Use only the person from the pictures; nobody else appears.
<Picture 1> is the identity, room, lighting and framing reference and the first outfit: <outfit>.

scene_reference:
<place, layout, light from Picture 1>. The room, furniture, perspective and light stay unchanged in every shot.

style:
<medium, lens, framing, colour>. No captions, emojis, stickers, signs, logos, subtitles or watermarks.

retention_analysis:
Keep the face, hair, body and location identical across every shot and every cut. Camera distance, height and exposure never change unless a Camera line says so.

[SHOT 01 | DURATION: 5s | REF_VIDEO: none | REF_IMAGES: 1]
Camera: <shot size>, <height and angle>, <movement>, <subject position>.
00:00–00:02 — <action>
00:02–00:05 — <action ending on the END POSE>

[SHOT 02 | DURATION: 4.4s | REF_VIDEO: none | REF_IMAGES: 1]
Shot continues from the final frame of the previous shot: <same pose>.
Camera: <same framing>.
00:00–00:02 — <action>
00:02–00:04.4 — <action>

overall_soundscape:
<shared ambience>. No dialogue.

non_diegetic_music:
NONE"""

SHOTS_USER = ("CHARACTER BIBLE:\n{bible}\n\nREFERENCE PICTURES:\n{references}\n\nSTORY:\n{story}\n\n"
              "SHOT PLAN:\n{shot_plan}\n\nREMINDERS: GLOBAL says only who and where; one [SHOT NN] block per "
              "SHOT PLAN line with its DURATION and REF_IMAGES; write both sides of every transition; "
              "one sound block after the last shot.")

for preset in ("sfw", "nsfw"):
    data[f"plan_{preset}"] = {"system": PLAN.replace("{content}", CONTENT[preset]), "user": PLAN_USER,
                              "temperature": 0.3, "max_tokens": 1024}
    data[f"shots_{preset}"] = {"system": SHOTS.replace("{content}", CONTENT[preset]), "user": SHOTS_USER,
                               "temperature": 0.4, "max_tokens": 3072}

order = ("bible", "segment", "ideas_sfw", "ideas_nsfw", "expand_sfw", "expand_nsfw",
         "plan_sfw", "plan_nsfw", "shots_sfw", "shots_nsfw")
path.write_text(json.dumps({k: data[k] for k in order}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("ok", list(order))
```
Expected: `ok ['bible', 'segment', ..., 'shots_nsfw']`.

- [ ] **Step 5: Chạy test**

Run: `.venv/Scripts/python.exe test_h3_prompts.py && .venv/Scripts/python.exe test_h3_app.py 2>&1 | grep -v httpx | tail -1`
Expected: `test_h3_prompts OK` (hoặc dòng OK cuối của file), rồi `16 checks passed`. Các test định dạng cũ (`split_h3_output`, `check_shots`, …) vẫn chạy vì hàm cũ chưa bị xóa (Task 7 xóa).

- [ ] **Step 6: Commit**

```bash
git add captioner/h3_prompts.py captioner/system_prompt.json test_h3_prompts.py test_h3_app.py
git commit -m "H3 prompts: element roles outfit/face/prop, plan and segment steps, v4 shot director prompt" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Phân tích video (`captioner/video_segments.py`)

**Files:**
- Create: `captioner/video_segments.py`
- Modify: `requirements.txt` (thêm `av`)
- Test: `test_video_segments.py`

**Interfaces:**
- Consumes: không có (numpy, PyAV, Pillow).
- Produces:
  - `read_video(path) -> {"fps": float, "frames": int, "duration": float, "width": int, "height": int, "tiny": np.ndarray (n, 36, 64) float32}`; video không đọc được → `ValueError`.
  - `frame_diffs(tiny) -> np.ndarray`; `detect_cuts(d, level=20.0, ratio=3.0, radius=8, min_gap=6) -> list[int]` (frame bắt đầu segment mới).
  - `make_segments(frames, fps, cuts) -> list[dict]` với khóa `seg, first, last, start, end, seconds`.
  - `add_setups(tiny, segs, threshold=30.0) -> segs` (thêm `bg_diff`, `setup` = `"S1"`, `"S2"`…).
  - `split_long(segs, fps, max_len=12.0) -> segs` (đánh số lại `seg`).
  - `group_shots(segs, min_len=4.4, max_len=12.0) -> list[int]` (số shot 1-based mỗi segment).
  - `sample_frames(path, segs, out_dir, width=512) -> {seg: [(seconds, Path), ...]}` (đầu, giữa, cuối).

- [ ] **Step 1: Viết test hỏng**

`test_video_segments.py`:
```python
"""Self-checks for the reference-video segmenter. Run: .venv\\Scripts\\python.exe test_video_segments.py"""
import tempfile
from pathlib import Path

import av
import numpy as np

from captioner.video_segments import (
    add_setups, detect_cuts, frame_diffs, group_shots, make_segments, read_video, sample_frames, split_long,
)

COLOURS = (((255, 0, 0), 2), ((0, 0, 255), 3), ((0, 255, 0), 2))  # 10 fps → cuts at frames 20 and 50


def write_video(path, colours=COLOURS, fps=10, size=(64, 48)):
    with av.open(str(path), "w") as c:
        st = c.add_stream("mpeg4", rate=fps)
        st.width, st.height, st.pix_fmt = size[0], size[1], "yuv420p"
        for colour, secs in colours:
            frame = np.full((size[1], size[0], 3), colour, np.uint8)
            for _ in range(int(secs * fps)):
                for pkt in st.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                    c.mux(pkt)
        for pkt in st.encode():
            c.mux(pkt)
    return Path(path)


def test_detect_cuts_on_synthetic_diffs():
    d = np.full(60, 2.0, np.float32)
    d[0], d[30] = 0, 50
    assert detect_cuts(d) == [30]
    steady = np.full(60, 25.0, np.float32)  # fast continuous motion is not a cut
    assert detect_cuts(steady) == []
    strobe = np.full(60, 2.0, np.float32)
    for start in (10, 30, 50):  # 3-frame flashes: high, but not isolated
        strobe[start:start + 3] = 60
    assert detect_cuts(strobe) == []
    close = np.full(40, 2.0, np.float32)
    close[10], close[13] = 50, 50  # within min_gap and each other's neighbourhood
    assert detect_cuts(close) == []


def test_read_segments_and_setups_from_a_real_file():
    with tempfile.TemporaryDirectory() as tmp:
        info = read_video(write_video(Path(tmp) / "v.mp4"))
    assert info["frames"] == 70 and round(info["fps"]) == 10 and (info["width"], info["height"]) == (64, 48)
    assert info["tiny"].shape == (70, 36, 64)
    cuts = detect_cuts(frame_diffs(info["tiny"]))
    assert cuts == [20, 50]
    segs = add_setups(info["tiny"], make_segments(info["frames"], info["fps"], cuts))
    assert [s["seconds"] for s in segs] == [2.0, 3.0, 2.0] and [s["start"] for s in segs] == [0.0, 2.0, 5.0]
    assert [s["setup"] for s in segs] == ["S1", "S2", "S3"] and segs[0]["bg_diff"] == 0.0


def test_unreadable_video():
    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "bad.mp4"
        bad.write_bytes(b"not a video")
        try:
            read_video(bad)
        except (ValueError, av.error.FFmpegError) as e:
            assert str(e)
        else:
            raise AssertionError("expected an error")


def _segs(seconds, setups=None):
    out, t = [], 0.0
    for i, s in enumerate(seconds):
        out.append({"seg": i + 1, "first": int(t * 10), "last": int((t + s) * 10) - 1, "start": t,
                    "end": t + s, "seconds": s, "setup": (setups or ["S1"] * len(seconds))[i], "bg_diff": 0.0})
        t += s
    return out


def test_group_shots():
    assert group_shots(_segs([1.73, 1.4, 1.44, 2.26, 1.7, 1.96])) == [1] * 6
    assert group_shots(_segs([5, 5, 5])) == [1, 1, 2]
    assert group_shots(_segs([5, 6], ["S1", "S2"])) == [1, 2]
    assert group_shots(_segs([5, 3], ["S1", "S2"])) == [1, 1]  # a short last shot folds back
    assert group_shots(_segs([3, 3], ["S1", "S2"])) == [1, 1]  # setup change before 4.4s stays together


def test_split_long():
    segs = split_long(_segs([30.0]), fps=10)
    assert [s["seconds"] for s in segs] == [10.0, 10.0, 10.0] and [s["seg"] for s in segs] == [1, 2, 3]
    assert segs[1]["start"] == 10.0 and segs[2]["last"] == 299
    assert group_shots(segs) == [1, 2, 3]


def test_sample_frames():
    with tempfile.TemporaryDirectory() as tmp:
        video = write_video(Path(tmp) / "v.mp4")
        info = read_video(video)
        segs = make_segments(info["frames"], info["fps"], [20, 50])
        frames = sample_frames(video, segs, Path(tmp) / "frames", width=32)
        assert [t for t, _ in frames[2]] == [2.0, 3.4, 4.9]
        assert all(p.is_file() and p.suffix == ".jpg" for f in frames.values() for _, p in f)
        from PIL import Image
        with Image.open(frames[1][0][1]) as im:
            assert im.size == (32, 24)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("test_video_segments OK")
```

- [ ] **Step 2: Chạy test để thấy hỏng**

Run: `.venv/Scripts/python.exe test_video_segments.py`
Expected: `ModuleNotFoundError: No module named 'captioner.video_segments'`

- [ ] **Step 3: Viết module**

`captioner/video_segments.py`:
```python
"""Reference video → cuts, segments, camera setups and H3 shots. numpy + PyAV only (no torch, no gradio).

Qwen3-VL cannot place cuts in time (tested: it splits the timeline evenly), so cut times always come
from this code; the model only describes each segment's frames.
"""
import math
from pathlib import Path

import av
import numpy as np

TINY = (64, 36)      # (width, height) grey thumbnails for cut detection
CUT_LEVEL = 20.0     # mean abs grey difference a cut must exceed
CUT_RATIO = 3.0      # ... and be this many times larger than every neighbour
CUT_RADIUS = 8       # neighbourhood in frames on each side; rejects steady motion and multi-frame strobes
MIN_GAP = 6          # frames between two cuts
SETUP_DIFF = 30.0    # border difference that starts a new camera setup (as video_probe.py)
MIN_SHOT, MAX_SHOT = 4.4, 12.0


def _fps(stream) -> float:
    rate = stream.average_rate or stream.guessed_rate
    return float(rate) if rate else 30.0


def read_video(path) -> dict:
    """One decode pass: metadata plus a grey 64x36 thumbnail of every frame."""
    tiny, size = [], None
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = _fps(stream)
        for frame in container.decode(stream):
            size = size or (frame.width, frame.height)
            tiny.append(np.asarray(frame.to_image().convert("L").resize(TINY), dtype=np.float32))
    if not tiny:
        raise ValueError(f"cannot decode any frame from {path}")
    return {"fps": fps, "frames": len(tiny), "duration": round(len(tiny) / fps, 2),
            "width": size[0], "height": size[1], "tiny": np.stack(tiny)}


def frame_diffs(tiny) -> np.ndarray:
    """d[i] = mean |frame i − frame i−1|; d[0] = 0."""
    d = np.zeros(len(tiny), dtype=np.float32)
    if len(tiny) > 1:
        d[1:] = np.abs(tiny[1:] - tiny[:-1]).mean(axis=(1, 2))
    return d


def detect_cuts(d, level=CUT_LEVEL, ratio=CUT_RATIO, radius=CUT_RADIUS, min_gap=MIN_GAP) -> list:
    """Frames that start a new segment: an isolated spike well above its neighbourhood."""
    cuts = []
    for i in range(1, len(d)):
        lo, hi = max(1, i - radius), min(len(d), i + radius + 1)
        neighbours = np.concatenate([d[lo:i], d[i + 1:hi]])
        isolated = neighbours.size == 0 or d[i] > ratio * float(neighbours.max())
        if d[i] > level and isolated and (not cuts or i - cuts[-1] >= min_gap):
            cuts.append(i)
    return cuts


def make_segments(frames, fps, cuts) -> list:
    bounds = [0, *cuts, frames]
    return [{"seg": k + 1, "first": a, "last": b - 1, "start": round(a / fps, 2), "end": round(b / fps, 2),
             "seconds": round((b - a) / fps, 2)} for k, (a, b) in enumerate(zip(bounds, bounds[1:]))]


def add_setups(tiny, segs, threshold=SETUP_DIFF) -> list:
    """bg_diff against the previous segment (median frame, top/left/right border) and a setup id."""
    h, w = tiny.shape[1:]

    def border(t):
        return np.concatenate([t[: h // 4].ravel(), t[:, : w // 5].ravel(), t[:, -(w // 5):].ravel()])

    sid, prev = 1, None
    for s in segs:
        med = np.median(tiny[s["first"]:s["last"] + 1], axis=0)
        s["bg_diff"] = round(float(np.abs(border(med) - border(prev)).mean()), 1) if prev is not None else 0.0
        if prev is not None and s["bg_diff"] >= threshold:
            sid += 1
        s["setup"] = f"S{sid}"
        prev = med
    return segs


def split_long(segs, fps, max_len=MAX_SHOT) -> list:
    """A segment longer than max_len becomes equal continuous parts; segments are renumbered."""
    out = []
    for s in segs:
        parts = max(1, math.ceil(s["seconds"] / max_len - 1e-9))
        span = s["last"] - s["first"] + 1
        for p in range(parts):
            a = s["first"] + span * p // parts
            b = s["first"] + span * (p + 1) // parts
            out.append({**s, "first": a, "last": b - 1, "start": round(a / fps, 2), "end": round(b / fps, 2),
                        "seconds": round((b - a) / fps, 2), "bg_diff": s.get("bg_diff", 0.0) if p == 0 else 0.0})
    for k, s in enumerate(out, 1):
        s["seg"] = k
    return out


def group_shots(segs, min_len=MIN_SHOT, max_len=MAX_SHOT) -> list:
    """Shot number per segment: fill a shot in order, start a new one past max_len or at a new camera
    setup once the shot already lasts min_len; fold a too-short last shot back when it fits."""
    groups, cur, cur_len = [], [], 0.0
    for i, s in enumerate(segs):
        new_setup = bool(cur) and s["setup"] != segs[cur[-1]]["setup"]
        if cur and (cur_len + s["seconds"] > max_len + 1e-6 or (new_setup and cur_len >= min_len)):
            groups.append(cur)
            cur, cur_len = [], 0.0
        cur.append(i)
        cur_len += s["seconds"]
    if cur:
        groups.append(cur)
    if len(groups) > 1:
        last = sum(segs[i]["seconds"] for i in groups[-1])
        prev = sum(segs[i]["seconds"] for i in groups[-2])
        if last < min_len and prev + last <= max_len + 1e-6:
            groups[-2] += groups.pop()
    shot_of = [0] * len(segs)
    for n, group in enumerate(groups, 1):
        for i in group:
            shot_of[i] = n
    return shot_of


def sample_frames(path, segs, out_dir, width=512) -> dict:
    """First, middle and last frame of every segment as JPEGs (second decode pass, stops early)."""
    want = {}
    for s in segs:
        for k in sorted({s["first"], (s["first"] + s["last"]) // 2, s["last"]}):
            want.setdefault(k, []).append(s["seg"])
    out = {s["seg"]: [] for s in segs}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = _fps(stream)
        for i, frame in enumerate(container.decode(stream)):
            if i in want:
                img = frame.to_image().convert("RGB")
                img = img.resize((width, max(1, round(width * img.height / img.width))))
                p = out_dir / f"f{i:06d}.jpg"
                img.save(p, quality=90)
                for seg in want[i]:
                    out[seg].append((round(i / fps, 2), p))
            if i >= max(want):
                break
    return out
```

- [ ] **Step 4: Thêm `av` vào requirements**

Trong `requirements.txt`, ngay dưới dòng `pillow-heif ...` thêm:
```text
av            # PyAV: reads reference videos for the H3 2b tab (cut detection, frames)
```

- [ ] **Step 5: Chạy test để thấy qua**

Run: `.venv/Scripts/python.exe test_video_segments.py`
Expected: `test_video_segments OK`. Nếu `test_sample_frames` lệch timestamp giữa (vd 3.4 so với 3.5) vì cách làm tròn, đổi kỳ vọng theo giá trị đúng của frame `(20+49)//2 = 34` → `3.4`, và ghi vào report.

- [ ] **Step 6: Chạy trên video thật (kiểm chứng, không commit file nào)**

Run:
```bash
.venv/Scripts/python.exe -c "from captioner.video_segments import *; i=read_video(r'D:\001_Personal_Proj\Comfy\ComfyUI\input\czx126__1792129994_4008998592057733464_34027619544.mp4'); c=detect_cuts(frame_diffs(i['tiny'])); s=split_long(add_setups(i['tiny'], make_segments(i['frames'], i['fps'], c)), i['fps']); print(round(i['fps']), i['duration'], [round(x/i['fps'],2) for x in c], [x['setup'] for x in s], group_shots(s))"
```
Expected: điểm cắt gần `1.73, 3.13, 4.57, 6.83, 8.53` (lệch ≤ 0.05 s). Nếu file không còn ở đó, ghi vào report và bỏ qua bước này.

- [ ] **Step 7: Commit**

```bash
git add captioner/video_segments.py test_video_segments.py requirements.txt
git commit -m "Add video_segments: PyAV cut detection, camera setups, shot grouping, frame sampling" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Handler H3 v4 cho 2a và 2b (chưa đổi giao diện)

**Files:**
- Modify: `app.py` (import; thêm handler mới ngay sau `h3_save_system`, trước dòng `# ─────` kế tiếp)
- Test: `test_h3_v4_app.py`

**Interfaces:**
- Consumes: Task 3 (`h3_v4`), Task 4 (store, `describe_references`, `step_key`), Task 5 (`video_segments`); trong app có sẵn `_h3_guard`, `_h3_bible`, `_h3_run`, `_h3_refs`, `_h3_unpack`, `_h3_total`, `_h3_log`, `_h3_tail`, `_h3_write`, `_badge`, `h3_output_path`, `shot_plan_from_story`, `shot_plan_from_total`.
- Produces (Task 7 nối vào UI):
  - `h3_plan(story, total, bible, bible_system, plan_system, nsfw, log, *media) -> (bible, plan_rows, status, log, log_text)`
  - `h3_write(plan_rows, story, bible, bible_system, shots_system, nsfw, log, *media) -> (bible, prompt, settings_md, saved_path, status, log, log_text)`
  - `h3v_analyse(video, segment_system, log, *media) -> (segment_rows, segments_state, status, log, log_text)`
  - `h3v_plan(segment_rows, segments, bible, bible_system, plan_system, nsfw, log, *media) -> (bible, plan_rows, status, log, log_text)`
  - `h3v_write(plan_rows, segment_rows, segments, video, keep_music, ref_video_hint, bible, bible_system, shots_system, nsfw, log, *media) -> (bible, prompt, settings_md, saved_path, status, log, log_text)`
  - `h3_save_text(prompt, saved_path, path1, nsfw, log) -> (saved_path, status, log, log_text)`
  - `media` giữ đúng thứ tự cũ: `(img1..img4, role2..role4, path1..path4, resize_mode, resize_w, resize_h)`.

- [ ] **Step 1: Viết test hỏng**

`test_h3_v4_app.py`:
```python
"""Self-checks for the H3 v4 handlers (2a and 2b) with a fake model. Run: .venv\\Scripts\\python.exe test_h3_v4_app.py"""
import tempfile
from pathlib import Path

from PIL import Image

import app
from test_video_segments import write_video


class FakeCaptioner:
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def caption_image(self, image_path, user_prompt, max_new_tokens=512, system_prompt=None, temperature=None):
        self.calls.append(("image", image_path, user_prompt, system_prompt, temperature))
        return self.replies.pop(0)

    def caption_text(self, user_prompt, max_new_tokens=512, system_prompt=None, temperature=None):
        self.calls.append(("text", None, user_prompt, system_prompt, temperature))
        return self.replies.pop(0)


BIBLE = "NAME_HINT: man with short black hair\nAGE_APPEARANCE: 30, adult"
PLAN = ("SHOT 01 | REF_IMAGES: 1 | NEXT: CONTINUE | END: holds a smile | CONTENT: He waves.\n"
        "SHOT 02 | REF_IMAGES: 1 | CONTENT: He bows.")
PROMPT = """subject_definitions:
<Subject 1> Ivan, an adult man: short black hair.
<Picture 1> is the identity, room, lighting and framing reference and the first outfit: grey tee.

[SHOT 01 | DURATION: 5s | REF_VIDEO: none | REF_IMAGES: 1]
Camera: medium, static.
00:00–00:05 — He waves and holds a smile.

[SHOT 02 | DURATION: 5s | REF_VIDEO: none | REF_IMAGES: 1]
Shot continues from the final frame of the previous shot: he holds a smile.
00:00–00:05 — He bows.

overall_soundscape:
Room tone. No dialogue.

non_diegetic_music:
NONE
"""
SEGMENT = "CAMERA: medium, static\nACTION: waves\nOUTFIT: tee\nLOCATION: room"


def _media(path1=""):
    img = Image.new("RGB", (64, 64), "white")
    return (img, None, None, None, *app.H3_ROLE_CHOICES, path1, "", "", "", "Adaptive image", 512, 512)


def _run(fake, fn, *args):
    app._captioner = fake
    try:
        return fn(*args)
    finally:
        app._captioner = None


STORY = "00:00–00:05 — Anh vẫy tay.\n00:05–00:10 — Anh cúi chào."


def test_2a_plan_uses_story_durations_and_model_fields():
    fake = FakeCaptioner(BIBLE, PLAN)
    bible, rows, status, log, log_text = _run(fake, app.h3_plan, STORY, 0, "", "b", "plan rules", False, [],
                                              *_media())
    assert bible == BIBLE and len(rows) == 2
    assert rows[0] == ["SHOT 01", "5s", "story 00:00–00:05", "1", "CONTINUE", "holds a smile", "He waves."]
    assert rows[1][4] == "" and "#4ade80" in status
    kind, _, user, system, temp = fake.calls[1]
    assert kind == "text" and system == "plan rules" and temp == 0.3
    assert "SHOT 01 | DURATION: 5s | REF_IMAGES: 1 | SOURCE: story 00:00–00:05" in user and STORY in user


def test_2a_plan_lets_the_model_choose_without_timecodes():
    fake = FakeCaptioner("SHOT 01 | DURATION: 8s | CONTENT: Walks.")
    _, rows, _, _, _ = _run(fake, app.h3_plan, "Anh đi dạo.", 0, BIBLE, "b", "p", False, [], *_media())
    assert rows == [["SHOT 01", "8s", "", "1", "", "", "Walks."]]
    assert "Choose the number of shots yourself" in fake.calls[0][2]


def test_2a_write_checks_saves_and_shows_runner_settings():
    with tempfile.TemporaryDirectory() as tmp:
        pic = Path(tmp) / "hero.png"
        Image.new("RGB", (64, 64), "white").save(pic)
        rows = [["SHOT 01", "5s", "", "1", "CONTINUE", "holds a smile", "He waves."],
                ["SHOT 02", "5s", "", "1", "", "", "He bows."]]
        fake = FakeCaptioner(PROMPT)
        bible, prompt, settings, saved, status, log, log_text = _run(
            fake, app.h3_write, rows, STORY, BIBLE, "b", "shot rules", False, [], *_media(str(pic)))
        assert prompt.startswith("subject_definitions:") and "[SHOT 02 | DURATION: 5s | REF_VIDEO: none" in prompt
        assert "continuity_guide: ON" in settings and "final_audio: (trống)" in settings
        assert saved == str(Path(tmp) / "hero_H3_sfw.txt") and Path(saved).read_text(encoding="utf-8") == prompt
    kind, _, user, system, temp = fake.calls[0]
    assert kind == "text" and system == "shot rules" and temp == 0.4
    assert "SHOT 01 | DURATION: 5s | REF_IMAGES: 1 | CONTENT: He waves. | NEXT: CONTINUE | END POSE: holds a smile" in user


def test_2a_write_keeps_raw_text_without_shots_and_guards():
    rows = [["SHOT 01", "5s", "", "1", "", "", ""]]
    out = _run(FakeCaptioner("I cannot."), app.h3_write, rows, STORY, BIBLE, "b", "s", False, [], *_media())
    assert out[1] == "I cannot." and not isinstance(out[3], str) and "no [SHOT" in out[6]
    assert "Load a model" in app.h3_write(rows, STORY, BIBLE, "b", "s", False, [], *_media())[4]
    assert "plan" in _run(FakeCaptioner(), app.h3_write, [], STORY, BIBLE, "b", "s", False, [], *_media())[4].lower()


def test_2b_analyse_plan_write():
    with tempfile.TemporaryDirectory() as tmp:
        video = write_video(Path(tmp) / "dance.mp4")
        pic = Path(tmp) / "hero.png"
        Image.new("RGB", (64, 64), "white").save(pic)
        fake = FakeCaptioner(SEGMENT, SEGMENT, SEGMENT)
        seg_rows, segs, status, log, log_text = _run(fake, app.h3v_analyse, str(video), "seg rules", [],
                                                     *_media(str(pic)))
        assert len(seg_rows) == 3 and [r[0] for r in seg_rows] == [1, 2, 3] and seg_rows[0][5] == "waves"
        assert [len(c[1]) for c in fake.calls] == [3, 3, 3] and fake.calls[0][3] == "seg rules"
        assert "cuts at 00:02, 00:05" in log_text

        fake = FakeCaptioner(BIBLE, "SHOT 01 | REF_IMAGES: 1 | CONTENT: Montage.")
        bible, plan_rows, status, log, log_text = _run(fake, app.h3v_plan, seg_rows, segs, "", "b", "p", False, [],
                                                       *_media(str(pic)))
        assert len(plan_rows) == max(r[7] for r in seg_rows)
        assert "SEGMENT 1 (00:00–00:02" in fake.calls[1][2]

        fake = FakeCaptioner(PROMPT)
        out = _run(fake, app.h3v_write, plan_rows, seg_rows, segs, str(video), True, True, BIBLE, "b", "s", False, [],
                   *_media(str(pic)))
        assert "final_audio: dance.mp4" in out[2] and "Gợi ý REF_VIDEO 22f" in out[2]
        assert "SEGMENT 3" in fake.calls[0][2]


def test_2b_guards():
    assert "Load a model" in app.h3v_analyse("x.mp4", "s", [], *_media())[2]
    assert "video" in _run(FakeCaptioner(), app.h3v_analyse, None, "s", [], *_media())[2].lower()
    out = _run(FakeCaptioner(), app.h3v_analyse, "missing.mp4", "s", [], *_media())
    assert "#f87171" in out[2]  # error badge


def test_save_text():
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "out.txt"
        saved, status, log, log_text = app.h3_save_text("[SHOT 01]\nx", str(target), "", False, [])
        assert saved == str(target) and target.read_text(encoding="utf-8") == "[SHOT 01]\nx"
    assert "Nothing" in app.h3_save_text("  ", "", "", False, [])[1]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("test_h3_v4_app OK")
```

- [ ] **Step 2: Chạy test để thấy hỏng**

Run: `.venv/Scripts/python.exe test_h3_v4_app.py`
Expected: `AttributeError: module 'app' has no attribute 'h3_plan'`

- [ ] **Step 3: Import**

Trong `app.py`, cạnh các import `captioner` khác:
```python
from captioner.h3_v4 import (
    CHOOSE_SHOTS, SEGMENT_FIELDS, apply_model_plan, apply_segment_rows, check_v4, describe_segments,
    fmt_duration, format_plan, group_segments, parse_segment, plan_from_story, plan_from_video, plan_to_rows,
    rows_to_plan, runner_settings, segments_to_rows, ts,
)
from captioner.video_segments import (
    add_setups, detect_cuts, frame_diffs, group_shots, make_segments, read_video, sample_frames, split_long,
)
```
`tempfile` cần ở cấp module: nếu `app.py` chưa có `import tempfile` ở đầu file thì thêm vào nhóm import chuẩn.

- [ ] **Step 4: Thêm handler**

Ngay sau hàm `h3_save_system` trong `app.py`:
```python
# ── H3 v4 (Runner V5.3): 2a story → plan → prompt, 2b video → plan → prompt ──
def _h3_plan_call(plan, story, bible, plan_system, nsfw, log, media, bible_out):
    """One model call fills REF_IMAGES / NEXT / END / CONTENT of the planned shots (or chooses them)."""
    refs = _h3_refs(media)
    raw, log = _h3_run("plan", nsfw, plan_system, log, bible=bible, references=describe_references(refs),
                       story=story, shot_plan=format_plan(plan) if plan else CHOOSE_SHOTS)
    plan, warnings = apply_model_plan(plan, raw, len(refs))
    for w in warnings:
        log = _h3_log(log, w)
    if not plan:
        log = _h3_log(log, f"plan: none parsed; raw: {raw[:300]}")
        return (bible_out, gr.update(), *_h3_tail(_badge("The model returned no plan lines. Try again.", "warning"),
                                                  log))
    return (bible_out, plan_to_rows(plan),
            *_h3_tail(_badge(f"Plan: {len(plan)} shot. Sửa bảng nếu cần rồi bấm Viết prompt v4.", "success"), log))


def h3_plan(story, total, bible, bible_system, plan_system, nsfw, log, *media):
    err = _h3_guard(media)
    if err:
        return (gr.update(), gr.update(), *_h3_tail(err, log))
    story = (story or "").strip()
    if not story:
        return (gr.update(), gr.update(), *_h3_tail(_badge("Write a story or expand an idea first.", "warning"), log))
    bible_out = gr.update()
    try:
        bible, log = _h3_bible(bible, bible_system, log, media)
        bible_out = bible
        segments, notes = shot_plan_from_story(story)
        source = "story timecodes"
        if not segments:
            segments, notes = shot_plan_from_total(_h3_total(total))
            source = "total duration"
        for note in notes:
            log = _h3_log(log, note)
        plan = plan_from_story(segments)
        log = _h3_log(log, f"shot plan ({source}): " + " + ".join(fmt_duration(r["duration"]) for r in plan)
                      if plan else "shot plan: model chooses")
        return _h3_plan_call(plan, story, bible, plan_system, nsfw, log, media, bible_out)
    except Exception as e:
        log = _h3_log(log, f"error: {e}")
        return (bible_out, gr.update(), *_h3_tail(_badge(f"Error: {e}", "error"), log))


def _h3_write_core(rows, story, bible, bible_system, shots_system, nsfw, final_audio, ref_video_hint, log, media):
    keep = (gr.update(), gr.update(), gr.update(), gr.update())
    err = _h3_guard(media)
    if err:
        return (*keep, *_h3_tail(err, log))
    refs = _h3_refs(media)
    plan, warnings = rows_to_plan(rows, len(refs))
    if not plan:
        return (*keep, *_h3_tail(_badge("Make a shot plan first.", "warning"), log))
    if not (story or "").strip():
        return (*keep, *_h3_tail(_badge("Write a story or analyse a video first.", "warning"), log))
    for w in warnings:
        log = _h3_log(log, w)
    bible_out = gr.update()
    try:
        bible, log = _h3_bible(bible, bible_system, log, media)
        bible_out = bible
        raw, log = _h3_run("shots", nsfw, shots_system, log, bible=bible, references=describe_references(refs),
                           story=story.strip(), shot_plan=format_plan(plan))
        try:
            text, fixes = check_v4(raw, len(refs), plan)
        except ValueError as e:
            log = _h3_log(log, f"shots: {e}; raw: {raw[:300]}")
            return (bible_out, raw, gr.update(), gr.update(), *_h3_tail(
                _badge("Output has no [SHOT ...] block; the raw text is shown.", "warning"), log))
        for w in fixes:
            log = _h3_log(log, w)
        settings = runner_settings(plan, len(refs), final_audio=final_audio, ref_video_hint=ref_video_hint)
        path = h3_output_path(_h3_unpack(media)[2][0], nsfw)
        log = _h3_write(path, text, log)
        return (bible_out, text, settings, str(path), *_h3_tail(_badge("v4 prompt written and saved.", "success"), log))
    except Exception as e:
        log = _h3_log(log, f"error: {e}")
        return (bible_out, gr.update(), gr.update(), gr.update(), *_h3_tail(_badge(f"Error: {e}", "error"), log))


def h3_write(rows, story, bible, bible_system, shots_system, nsfw, log, *media):
    return _h3_write_core(rows, story, bible, bible_system, shots_system, nsfw, "", "", log, media)


def h3v_analyse(video, segment_system, log, *media):
    keep = (gr.update(), gr.update())
    err = _h3_guard(media)
    if err:
        return (*keep, *_h3_tail(err, log))
    if not video:
        return (*keep, *_h3_tail(_badge("Add a reference video first.", "warning"), log))
    try:
        info = read_video(video)
        cuts = detect_cuts(frame_diffs(info["tiny"]))
        segs = split_long(add_setups(info["tiny"], make_segments(info["frames"], info["fps"], cuts)), info["fps"])
        log = _h3_log(log, f"video {info['duration']}s, {info['fps']:.0f} fps, {info['width']}x{info['height']}; "
                           "cuts at " + (", ".join(ts(c / info["fps"]) for c in cuts) or "none"))
        with tempfile.TemporaryDirectory() as tmp:
            frames = sample_frames(video, segs, tmp)
            for s in segs:
                picks = frames[s["seg"]]
                raw, log = _h3_run("segment", False, segment_system, log, image_paths=[str(p) for _, p in picks],
                                   segment=s["seg"], start=ts(s["start"]), end=ts(s["end"]),
                                   seconds=f"{s['seconds']:g}", times=", ".join(ts(t) for t, _ in picks))
                s.update(parse_segment(raw))
                missing = [f for f in SEGMENT_FIELDS if not s[f]]
                if missing:
                    log = _h3_log(log, f"segment {s['seg']}: thiếu {', '.join(missing)}")
        shot_of = group_shots(segs)
        segs = [{k: v for k, v in s.items() if k != "tiny"} for s in segs]
        return (segments_to_rows(segs, shot_of), segs, *_h3_tail(
            _badge(f"{len(segs)} segment → {max(shot_of)} shot. Sửa cột Shot nếu muốn gộp/tách.", "success"), log))
    except Exception as e:
        log = _h3_log(log, f"video error: {e}")
        return (*keep, *_h3_tail(_badge(f"Video error: {e}", "error"), log))


def _h3v_segments(seg_rows, segs, log):
    segs, shot_of, warnings = apply_segment_rows(segs or [], seg_rows)
    if shot_of is None:
        shot_of = group_shots(segs)
    for w in warnings:
        log = _h3_log(log, w)
    return segs, shot_of, log


def h3v_plan(seg_rows, segs, bible, bible_system, plan_system, nsfw, log, *media):
    err = _h3_guard(media)
    if err:
        return (gr.update(), gr.update(), *_h3_tail(err, log))
    if not segs:
        return (gr.update(), gr.update(), *_h3_tail(_badge("Analyse the video first.", "warning"), log))
    bible_out = gr.update()
    try:
        bible, log = _h3_bible(bible, bible_system, log, media)
        bible_out = bible
        segs, shot_of, log = _h3v_segments(seg_rows, segs, log)
        plan, notes = plan_from_video(group_segments(segs, shot_of))
        for note in notes:
            log = _h3_log(log, note)
        return _h3_plan_call(plan, describe_segments(segs, shot_of), bible, plan_system, nsfw, log, media, bible_out)
    except Exception as e:
        log = _h3_log(log, f"error: {e}")
        return (bible_out, gr.update(), *_h3_tail(_badge(f"Error: {e}", "error"), log))


def h3v_write(rows, seg_rows, segs, video, keep_music, ref_video_hint, bible, bible_system, shots_system, nsfw,
              log, *media):
    segs, shot_of, log = _h3v_segments(seg_rows, segs, log)
    hint = ""
    if ref_video_hint:
        usable = [f"seg {s['seg']} ({ts(s['start'])}–{ts(s['end'])})" for s in segs if s["seconds"] >= 0.92]
        hint = (", ".join(usable) + " — cắt 22 frame bằng video_probe.py --clip <frame> --blur <đầu người>"
                if usable else "không có segment nào dài ≥ 0.92s")
    final_audio = Path(video).name if (keep_music and video) else ""
    return _h3_write_core(rows, describe_segments(segs, shot_of) if segs else "", bible, bible_system,
                          shots_system, nsfw, final_audio, hint, log, media)


def h3_save_text(prompt, saved_path, path1, nsfw, log):
    if not (prompt or "").strip():
        return (gr.update(), *_h3_tail(_badge("Nothing to save.", "warning"), log))
    try:
        path = Path(saved_path) if (saved_path or "").strip() else h3_output_path(path1, nsfw)
        log = _h3_write(path, prompt, log)
        return (str(path), *_h3_tail(_badge(f"Saved → {path}", "success"), log))
    except Exception as e:
        log = _h3_log(log, f"save failed: {e}")
        return (gr.update(), *_h3_tail(_badge(f"Save failed: {e}", "error"), log))
```

- [ ] **Step 5: Chạy test để thấy qua**

Run: `.venv/Scripts/python.exe test_h3_v4_app.py && .venv/Scripts/python.exe test_h3_app.py 2>&1 | grep -v httpx | tail -1`
Expected: `test_h3_v4_app OK`, rồi `16 checks passed`. Nếu `test_2b_analyse_plan_write` thấy chuỗi log khác (vd `cuts at 00:02, 00:05` lệch định dạng), sửa kỳ vọng theo đầu ra đúng của `ts()` và ghi vào report.

- [ ] **Step 6: Commit**

```bash
git add app.py test_h3_v4_app.py
git commit -m "H3 v4 handlers: 2a story plan/write, 2b video analyse/plan/write, plain-text save" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Giao diện hai tab con, xóa đường cũ

**Files:**
- Modify: `app.py` (thay toàn bộ khối `with gr.Tab("🎞️ Prompt Video Minimax H3") as h3_tab:`; thêm hàm dựng UI cạnh `_h3_section`; xóa `h3_generate`, `h3_save`, `h3_reload_systems` cũ, `h3_clear` cũ; cập nhật `h3_refresh_systems`; bỏ import cũ)
- Modify: `captioner/h3_prompts.py` (xóa định dạng cũ: `PART_TITLES`, `FIRST_SHOT`, `SHOT_LINE`, `DURATION_RE`, `SHOT_TAIL_DEFAULTS`, `PART1_FIELDS`, `TIMELINE`, `COPIED_PLACEHOLDER`, `REPEAT_FILLER`, `_add_tail`, `_fix_timeline`, `split_h3_output`, `check_part1`, `check_shots`, `build_h3_file`)
- Modify: `test_h3_prompts.py` (xóa test của các hàm trên), `test_h3_app.py` (xóa test `h3_generate*`, `test_save_writes_edited_text`; cập nhật test ⚙)

**Interfaces:**
- Consumes: Task 6 handler; Task 3 `PLAN_COLUMNS`, `SEGMENT_COLUMNS`, `PLAN_HELP`.
- Produces: biến module cho test ⚙: `h3a_bible_system, h3a_ideas_system, h3a_expand_system, h3a_plan_system, h3a_shots_system, h3b_bible_system, h3b_segment_system, h3b_plan_system, h3b_shots_system`; `h3_tab`; `H3A_STEPS = ("bible","ideas","expand","plan","shots")`, `H3B_STEPS = ("bible","segment","plan","shots")`; `h3_refresh_systems(nsfw_a, nsfw_b) -> tuple[9 str]`.

- [ ] **Step 1: Cập nhật test ⚙ trước**

Trong `test_h3_app.py`:
- Thay `H3_GEAR_BOXES = (...)` bằng:
```python
H3_GEAR_BOXES = ("h3a_bible_system", "h3a_ideas_system", "h3a_expand_system", "h3a_plan_system",
                 "h3a_shots_system", "h3b_bible_system", "h3b_segment_system", "h3b_plan_system",
                 "h3b_shots_system")
```
- Trong `test_opening_a_tab_refreshes_its_gear_boxes`, thay dòng `assert app.h3_refresh_systems(True) == ...` bằng:
```python
    assert app.h3_refresh_systems(True, False) == (
        tuple(app._h3_system_text(s, True) for s in app.H3A_STEPS)
        + tuple(app._h3_system_text(s, False) for s in app.H3B_STEPS))
```
- Xóa các test: `test_generate_runs_bible_then_shots_and_saves_next_to_picture1`, `test_generate_sends_extra_pictures_in_order`, `test_generate_reuses_bible_and_keeps_raw_text_when_no_shot_block`, `test_generate_completes_part1_tail`, `test_generate_guards`, `test_save_writes_edited_text`, cùng hằng `SHOTS` (chỉ chúng dùng). Giữ `BIBLE`, `IDEAS`, `_media`, `_run_with` và các test ideas/expand/load_path/log.

Trong `test_h3_prompts.py`: xóa `test_split_h3_output`, `test_check_shots_renumbers_clamps_and_completes_tail`, `test_check_shots_repairs_timeline_to_shot_duration`, `test_check_shots_moves_global_timecodes_to_zero`, `test_check_shots_merges_same_range_lines_and_drops_repeat_filler`, `test_check_part1_completes_tail`, `test_split_h3_output_unwraps_copied_placeholders`, và trong `test_output_paths` bỏ phần dùng `build_h3_file` (nếu có); xóa `build_h3_file`, `check_part1`, `check_shots`, `split_h3_output` khỏi dòng import.

- [ ] **Step 2: Chạy test để thấy hỏng**

Run: `.venv/Scripts/python.exe test_h3_app.py 2>&1 | grep -v httpx | tail -3`
Expected: FAIL `AttributeError: module 'app' has no attribute 'h3a_bible_system'`.

- [ ] **Step 3: Hàm dựng khối ảnh và nối sự kiện ảnh**

Ngay sau hàm `_h3_section` trong `app.py` thêm:
```python
H3A_STEPS = ("bible", "ideas", "expand", "plan", "shots")
H3B_STEPS = ("bible", "segment", "plan", "shots")


def _h3_pictures():
    """Picture 1 + 3 element slots with role, path and 📂; returns (imgs, roles, paths, browses, resize)."""
    gr.Markdown("### Ảnh tham chiếu")
    imgs = [gr.Image(type="pil", label="Picture 1 · Nhân vật + bối cảnh (bắt buộc)", height=260)]
    with gr.Row():
        paths = [gr.Textbox(label="Đường dẫn Picture 1", placeholder=r"D:\photos\hero.png", scale=5)]
        browses = [gr.Button("📂", scale=0, min_width=48)]
    roles = []
    with gr.Row():
        for n, role in zip((2, 3, 4), H3_ROLE_CHOICES):
            with gr.Column(min_width=120):
                imgs.append(gr.Image(type="pil", label=f"Element {n} (tùy chọn)", height=130))
                roles.append(gr.Dropdown(H3_ROLE_CHOICES, value=role, label="Vai trò"))
                with gr.Row():
                    paths.append(gr.Textbox(label="Đường dẫn", scale=4))
                    browses.append(gr.Button("📂", scale=0, min_width=40))
    gr.Markdown("Bối cảnh lấy từ Picture 1. Element: Trang phục (ảnh cắt sát món đồ), Face sheet, Đạo cụ.")
    with gr.Accordion("Resize options", open=False):
        mode = gr.Dropdown(choices=["Adaptive image", "Fit image"], value="Adaptive image", label="Resize Mode")
        with gr.Row():
            width = gr.Slider(32, 2048, value=512, step=32, label="Resize Width")
            height = gr.Slider(32, 2048, value=512, step=32, label="Resize Height")
    return imgs, roles, paths, browses, [mode, width, height]


def _h3_wire_pictures(imgs, roles, paths, browses, log_out, clear_fn, clear_out):
    for img, path, browse in zip(imgs, paths, browses):
        browse.click(h3_browse, [path], [path])
        path.change(h3_load_path, [path, log_out[0]], [img, *log_out])
        img.upload(lambda: "", None, [path], queue=False)
        img.clear(lambda: "", None, [path], queue=False)
        img.change(clear_fn, None, clear_out, queue=False)
    for role in roles:
        role.change(clear_fn, None, clear_out, queue=False)
```
Thay `h3_clear`, `h3_reload_systems`, `h3_refresh_systems` cũ bằng:
```python
def h3a_clear():
    """New pictures make the old bible, ideas, plan and prompt stale. The Story box is kept."""
    return "", gr.update(choices=[], value=None), [], [], "", ""


def h3b_clear():
    return "", [], [], [], "", ""


def h3_reload_systems(nsfw, steps):
    return tuple(_h3_system_text(step, nsfw) for step in steps)


def h3_refresh_systems(nsfw_a, nsfw_b=False):
    return (tuple(_h3_system_text(s, nsfw_a) for s in H3A_STEPS)
            + tuple(_h3_system_text(s, nsfw_b) for s in H3B_STEPS))
```
Xóa hàm `h3_generate` và `h3_save` cũ.

- [ ] **Step 4: Thay khối giao diện tab H3**

Thay toàn bộ khối từ `with gr.Tab("🎞️ Prompt Video Minimax H3") as h3_tab:` đến hết vòng `for h3_role in h3_roles:` (ngay trước comment `Tab 5 — Models`) bằng:
```python
    with gr.Tab("🎞️ Prompt Video Minimax H3") as h3_tab:
        gr.Markdown("Prompt cho **MiniMax H3** theo định dạng **AutoShot Runner V5.3 (v4)**. "
                    "⚙ sửa system prompt của từng bước trong `captioner/system_prompt.json`.")
        with gr.Tabs():
            # ── 2a: pictures + elements → ideas → story → plan → prompt ──
            with gr.Tab("2a · Ảnh + element → gợi ý → prompt"):
                h3a_log = gr.State([])
                h3a_ideas_state = gr.State([])
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1):
                        h3a_imgs, h3a_roles, h3a_paths, h3a_browses, h3a_resize = _h3_pictures()
                        with gr.Row():
                            h3a_nsfw = gr.Checkbox(label="NSFW", value=False)
                            h3a_total = gr.Number(value=0, precision=0, minimum=0,
                                                  label="Tổng thời lượng (s), 0 = tự động")
                        h3a_bible_system, _ = _h3_section("1) Character bible", "bible", h3a_nsfw)
                        with gr.Accordion("Bible (tự tạo khi trống, sửa được)", open=False):
                            h3a_bible = gr.Textbox(label="Character bible", lines=11, interactive=True)
                        gr.Markdown("### 2) Story")
                        h3a_story = gr.Textbox(label="Story / kịch bản (VI hoặc EN)", lines=6,
                                               placeholder="00:00–00:05 — … hoặc để trống rồi bấm Gợi ý 3 ý tưởng")
                        h3a_ideas_system, h3a_ideas_btn = _h3_section("Gợi ý ý tưởng", "ideas", h3a_nsfw,
                                                                      "💡 Gợi ý 3 ý tưởng")
                        h3a_ideas_radio = gr.Radio(choices=[], label="3 ý tưởng")
                        h3a_ideas_status = gr.HTML()
                        h3a_expand_system, h3a_expand_btn = _h3_section(
                            "Viết chi tiết ý tưởng", "expand", h3a_nsfw, "✍ Viết chi tiết vào Story",
                            run_interactive=False)
                        h3a_expand_status = gr.HTML()
                    with gr.Column(scale=1):
                        h3a_plan_system, h3a_plan_btn = _h3_section(
                            "3) Kế hoạch shot & chuyển cảnh", "plan", h3a_nsfw, "🧭 Lập kế hoạch",
                            run_interactive=False)
                        h3a_plan_status = gr.HTML()
                        h3a_plan = gr.Dataframe(headers=PLAN_COLUMNS, type="array", interactive=True,
                                                label="Kế hoạch shot (sửa được)", wrap=True)
                        gr.Markdown(PLAN_HELP)
                        h3a_shots_system, h3a_write_btn = _h3_section(
                            "4) Viết prompt v4", "shots", h3a_nsfw, "▶ Viết prompt v4", run_variant="primary")
                        h3a_status = gr.HTML()
                        h3a_prompt = gr.Textbox(label="Master prompt (Runner V5.3)", lines=24, interactive=True,
                                                buttons=["copy"])
                        h3a_settings = gr.Markdown()
                        with gr.Row():
                            h3a_saved = gr.Textbox(label="Saved file", interactive=False, scale=5)
                            h3a_save_btn = gr.Button("💾 Lưu lại", scale=1)
                        h3a_log_box = gr.Textbox(label="Logger", lines=10, interactive=False, autoscroll=True)

            # ── 2b: pictures + video + elements → segments → plan → prompt ──
            with gr.Tab("2b · Ảnh + video + element → prompt"):
                h3b_log = gr.State([])
                h3b_segs = gr.State([])
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1):
                        h3b_imgs, h3b_roles, h3b_paths, h3b_browses, h3b_resize = _h3_pictures()
                        h3b_video = gr.Video(label="Video tham chiếu", sources=["upload"], height=260)
                        with gr.Row():
                            h3b_music = gr.Checkbox(label="Giữ nhạc gốc (final_audio = file video)", value=True)
                            h3b_nsfw = gr.Checkbox(label="NSFW", value=False)
                        h3b_ref_hint = gr.Checkbox(
                            label="Gợi ý clip REF_VIDEO 22f cho shot một segment (nâng cao)", value=False)
                        h3b_bible_system, _ = _h3_section("1) Character bible", "bible", h3b_nsfw)
                        with gr.Accordion("Bible (tự tạo khi trống, sửa được)", open=False):
                            h3b_bible = gr.Textbox(label="Character bible", lines=11, interactive=True)
                        h3b_segment_system, h3b_analyse_btn = _h3_section(
                            "2) Phân tích video", "segment", h3b_nsfw, "✂ Phân tích video")
                        gr.Markdown("Điểm cắt tìm bằng code; model chỉ mô tả từng đoạn qua 3 frame.")
                        h3b_analyse_status = gr.HTML()
                    with gr.Column(scale=1):
                        h3b_seg_table = gr.Dataframe(headers=SEGMENT_COLUMNS, type="array", interactive=True,
                                                     label="Segment → shot (sửa cột Shot để gộp/tách)", wrap=True)
                        h3b_plan_system, h3b_plan_btn = _h3_section(
                            "3) Kế hoạch shot & chuyển cảnh", "plan", h3b_nsfw, "🧭 Lập kế hoạch")
                        h3b_plan_status = gr.HTML()
                        h3b_plan = gr.Dataframe(headers=PLAN_COLUMNS, type="array", interactive=True,
                                                label="Kế hoạch shot (sửa được)", wrap=True)
                        gr.Markdown(PLAN_HELP)
                        h3b_shots_system, h3b_write_btn = _h3_section(
                            "4) Viết prompt v4", "shots", h3b_nsfw, "▶ Viết prompt v4", run_variant="primary")
                        h3b_status = gr.HTML()
                        h3b_prompt = gr.Textbox(label="Master prompt (Runner V5.3)", lines=24, interactive=True,
                                                buttons=["copy"])
                        h3b_settings = gr.Markdown()
                        with gr.Row():
                            h3b_saved = gr.Textbox(label="Saved file", interactive=False, scale=5)
                            h3b_save_btn = gr.Button("💾 Lưu lại", scale=1)
                        h3b_log_box = gr.Textbox(label="Logger", lines=10, interactive=False, autoscroll=True)

        h3_tab.select(h3_refresh_systems, [h3a_nsfw, h3b_nsfw],
                      [h3a_bible_system, h3a_ideas_system, h3a_expand_system, h3a_plan_system, h3a_shots_system,
                       h3b_bible_system, h3b_segment_system, h3b_plan_system, h3b_shots_system], queue=False)

        # 2a wiring
        h3a_media = [*h3a_imgs, *h3a_roles, *h3a_paths, *h3a_resize]
        h3a_log_out = [h3a_log, h3a_log_box]
        h3a_ideas_btn.click(h3_ideas,
                            [h3a_bible, h3a_bible_system, h3a_ideas_system, h3a_nsfw, h3a_total, h3a_log, *h3a_media],
                            [h3a_bible, h3a_ideas_radio, h3a_ideas_state, h3a_ideas_status, *h3a_log_out])
        h3a_expand_btn.click(h3_expand,
                             [h3a_ideas_radio, h3a_ideas_state, h3a_bible, h3a_bible_system, h3a_expand_system,
                              h3a_nsfw, h3a_log, *h3a_media],
                             [h3a_bible, h3a_story, h3a_expand_status, *h3a_log_out])
        h3a_plan_btn.click(h3_plan,
                           [h3a_story, h3a_total, h3a_bible, h3a_bible_system, h3a_plan_system, h3a_nsfw, h3a_log,
                            *h3a_media],
                           [h3a_bible, h3a_plan, h3a_plan_status, *h3a_log_out])
        h3a_write_btn.click(h3_write,
                            [h3a_plan, h3a_story, h3a_bible, h3a_bible_system, h3a_shots_system, h3a_nsfw, h3a_log,
                             *h3a_media],
                            [h3a_bible, h3a_prompt, h3a_settings, h3a_saved, h3a_status, *h3a_log_out])
        h3a_save_btn.click(h3_save_text, [h3a_prompt, h3a_saved, h3a_paths[0], h3a_nsfw, h3a_log],
                           [h3a_saved, h3a_status, *h3a_log_out])
        h3a_story.change(lambda s: gr.update(interactive=bool((s or "").strip())), [h3a_story], [h3a_plan_btn],
                         queue=False)
        h3a_ideas_radio.change(lambda c: gr.update(interactive=c is not None), [h3a_ideas_radio], [h3a_expand_btn],
                               queue=False)
        h3a_nsfw.change(lambda n: h3_reload_systems(n, ("ideas", "expand", "plan", "shots")), [h3a_nsfw],
                        [h3a_ideas_system, h3a_expand_system, h3a_plan_system, h3a_shots_system], queue=False)
        _h3_wire_pictures(h3a_imgs, h3a_roles, h3a_paths, h3a_browses, h3a_log_out, h3a_clear,
                          [h3a_bible, h3a_ideas_radio, h3a_ideas_state, h3a_plan, h3a_prompt, h3a_saved])

        # 2b wiring
        h3b_media = [*h3b_imgs, *h3b_roles, *h3b_paths, *h3b_resize]
        h3b_log_out = [h3b_log, h3b_log_box]
        h3b_analyse_btn.click(h3v_analyse, [h3b_video, h3b_segment_system, h3b_log, *h3b_media],
                              [h3b_seg_table, h3b_segs, h3b_analyse_status, *h3b_log_out])
        h3b_plan_btn.click(h3v_plan,
                           [h3b_seg_table, h3b_segs, h3b_bible, h3b_bible_system, h3b_plan_system, h3b_nsfw, h3b_log,
                            *h3b_media],
                           [h3b_bible, h3b_plan, h3b_plan_status, *h3b_log_out])
        h3b_write_btn.click(h3v_write,
                            [h3b_plan, h3b_seg_table, h3b_segs, h3b_video, h3b_music, h3b_ref_hint, h3b_bible,
                             h3b_bible_system, h3b_shots_system, h3b_nsfw, h3b_log, *h3b_media],
                            [h3b_bible, h3b_prompt, h3b_settings, h3b_saved, h3b_status, *h3b_log_out])
        h3b_save_btn.click(h3_save_text, [h3b_prompt, h3b_saved, h3b_paths[0], h3b_nsfw, h3b_log],
                           [h3b_saved, h3b_status, *h3b_log_out])
        h3b_video.change(lambda: ([], [], [], "", ""), None, [h3b_seg_table, h3b_segs, h3b_plan, h3b_prompt, h3b_saved],
                         queue=False)
        h3b_nsfw.change(lambda n: h3_reload_systems(n, ("plan", "shots")), [h3b_nsfw],
                        [h3b_plan_system, h3b_shots_system], queue=False)
        _h3_wire_pictures(h3b_imgs, h3b_roles, h3b_paths, h3b_browses, h3b_log_out, h3b_clear,
                          [h3b_bible, h3b_seg_table, h3b_segs, h3b_plan, h3b_prompt, h3b_saved])
```
Thêm `PLAN_COLUMNS`, `SEGMENT_COLUMNS`, `PLAN_HELP` vào dòng import `captioner.h3_v4` của Task 6.

- [ ] **Step 5: Bỏ định dạng cũ**

- Trong dòng import `captioner.h3_prompts` của `app.py`: xóa `build_h3_file`, `check_part1`, `check_shots`, `split_h3_output`.
- Trong `captioner/h3_prompts.py`: xóa khối `# ── Model output ──` từ `PART_TITLES = ...` đến hết `check_shots` (giữ `clean_output`), và hàm `build_h3_file` ở cuối file. `clean_output` phải còn nguyên.
- Kiểm tra không còn tham chiếu: `grep -n "split_h3_output\|check_part1\|check_shots\|build_h3_file\|h3_generate\|h3_imgs\|h3_part1" *.py captioner/*.py` không ra dòng nào.

- [ ] **Step 6: Chạy toàn bộ test**

Run:
```bash
for t in test_t2t_prompts.py test_t2t_app.py test_h3_v4.py test_video_segments.py test_h3_prompts.py test_h3_v4_app.py test_model_library.py test_model_registry.py test_model_tester.py test_comfy_convert.py test_briefs_prompts.py; do .venv/Scripts/python.exe $t 2>&1 | tail -1; done
.venv/Scripts/python.exe test_h3_app.py 2>&1 | grep -v httpx | tail -1
.venv/Scripts/python.exe batch_caption.py --self-test
```
Expected: mọi dòng kết thúc bằng OK / `checks passed`.

- [ ] **Step 7: Commit**

```bash
git add app.py captioner/h3_prompts.py test_h3_prompts.py test_h3_app.py
git commit -m "H3 tab: 2a and 2b sub-tabs on the v4 format; drop the old two-part format" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Kiểm chứng thật (GPU + trình duyệt), tài liệu

**Files:**
- Modify: `README.md` (mục Text2Text và H3)
- Modify: `docs/architecture/app-architecture.html` (dòng chú thích dưới sơ đồ: thêm `video_segments.py` vào node Prompt templates sublabel không cần; chỉ sửa đoạn `.note`)

**Interfaces:**
- Consumes: toàn bộ task trước.
- Produces: bằng chứng chạy thật trong report; README cập nhật.

- [ ] **Step 1: Mở app**

Dùng `preview_start` với name `captioner`. Ở Setup, nạp `Qwen3-VL-8B-Instruct-GGUF` (profile Normal). `read_console_messages` không có lỗi.

- [ ] **Step 2: Text2Text**

Tab `📝 text2Text`: dán `1boy, solo, muscular, gym, sweat, tank top, dumbbell, looking at viewer, best quality, masterpiece`. Chạy một lần với `Krea 2`, một lần với `FLUX.2 Klein` + NSFW. Ghi hai kết quả vào report. Kỳ vọng: một đoạn tiếng Anh, bắt đầu bằng chủ thể, không có "masterpiece", 50–100 từ; bản NSFW gợi cảm nhưng không lộ.

- [ ] **Step 3: Tab 2a**

Picture 1 = `D:\001_Personal_Proj\Comfy\srcs\260914\Image\Screenshot 2026-09-20 100205.png` (hoặc ảnh người thật bất kỳ trong thư mục đó). Story:
```
00:00–00:05 — Anh mỉm cười, vẫy tay chào.
00:05–00:10 — Anh quay người, nhìn ra cửa sổ.
```
Bấm Lập kế hoạch → bảng 2 shot `5s`; sửa nhãn nối thành `CONTINUE`; bấm Viết prompt v4. Kỳ vọng: prompt bắt đầu `subject_definitions:`, có `[SHOT 01 | DURATION: 5s | REF_VIDEO: none | REF_IMAGES: 1]`, shot 2 mở đầu bằng `Shot continues from the final frame of the previous shot:`, một khối âm thanh cuối; ô cài đặt có `continuity_guide: ON`; file `_H3_sfw.txt` được lưu.

- [ ] **Step 4: Tab 2b**

Cùng Picture 1; video `D:\001_Personal_Proj\Comfy\ComfyUI\input\czx126__1792129994_4008998592057733464_34027619544.mp4`. Bấm Phân tích video → log có `cuts at 00:01.7, 00:03.1, 00:04.6, 00:06.8, 00:08.5` (±0.1 s), bảng 6 segment có Camera/Action. Lập kế hoạch → Viết prompt v4. Kỳ vọng: shot montage có dòng `At 00:0x.x, hard cut to`, ô cài đặt có `final_audio: czx126_….mp4`. Chụp màn hình tab 2b.

- [ ] **Step 5: Thử với Runner (nếu ComfyUI đang chạy)**

Nếu `http://127.0.0.1:8188` trả lời: mở workflow `MINIMAX_H3_AUTO_SHOTS_V4_HIBERNATE.json`, dán prompt 2a vào master_prompt của node #189, bấm **PREVIEW SHOTS** (không bấm START). Kỳ vọng: không lỗi, đúng số shot và thời lượng. Nếu ComfyUI không chạy, ghi "không kiểm được" vào report, không tự khởi động ComfyUI.

- [ ] **Step 6: README**

Thêm vào `README.md`, sau mục Model library:
```markdown
## text2Text

Turns Stable Diffusion tags into one English prompt for **Krea 2** or **FLUX.2 Klein** (choose in
*Target model*): one paragraph, subject first, no quality tags or weights. **NSFW (male, sensual)** adds
adult male appeal without explicit content. File mode keeps each block's negative line unchanged.

## Prompt Video MiniMax H3 (Runner V5.3)

Both sub-tabs write the v4 master prompt read by the ComfyUI AutoShot Runner V5.3: GLOBAL (who and
where), `[SHOT NN | DURATION | REF_VIDEO | REF_IMAGES]` blocks, one sound block at the end.

- **2a · pictures + elements → ideas → prompt**: Picture 1 (character and location) plus up to three
  elements (garment, face sheet, prop). Write or generate a story, press **Lập kế hoạch** to get the
  shot table, edit durations, REF_IMAGES and the transition into each next shot
  (CONTINUE / OUTFIT CUT / SCENE CUT), then **Viết prompt v4**.
- **2b · pictures + video + elements → prompt**: **Phân tích video** finds the cuts in code (PyAV) and
  asks the model to describe each segment from three frames. Edit the Shot column to merge or split,
  then plan and write as in 2a. *Giữ nhạc gốc* puts the video file in the Runner's `final_audio` note.
- The **Cài đặt Runner** box under the prompt lists continuity_guide, the LoadImage order and
  final_audio. It is never part of the prompt, because the Runner copies GLOBAL into every shot.
```

- [ ] **Step 7: Sơ đồ**

Trong `docs/architecture/app-architecture.html`, sửa đoạn `<p class="note">` thành:
```html
    <p class="note">Tabs: Setup, Single Image, Batch Processing, Convert Image, text2Text, Video Reel, Prompt Video MiniMax H3 (2a and 2b), Model Library, Help. The H3 2b sub-tab also reads reference videos through captioner/video_segments.py (PyAV). The quick tester loads models through the same two backends; that link is left out to keep the drawing readable.</p>
```

- [ ] **Step 8: Chạy lại toàn bộ test và commit**

Chạy lại khối lệnh ở Task 7 Step 6. Expected: tất cả OK.
```bash
git add README.md docs/architecture/app-architecture.html
git commit -m "Document text2Text targets and the H3 v4 sub-tabs" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
