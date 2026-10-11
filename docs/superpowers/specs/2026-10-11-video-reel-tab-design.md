# Tab "Video Reel" cho Simple App (Wan2GP)

Ngày: 2026-10-11. Nhánh: `20260927_add_new_model`.

## Mục tiêu

Gom 3 prompt Simple App (Pose Caption, Video Ideas, Video Reel) vào một tab riêng
"🎬 Video Reel" với luồng 3 bước nối nhau trên cùng một ảnh. Mỗi bước có nút chạy
riêng và nút bánh răng để sửa system prompt (lưu qua file). Có nút Export txt gộp
kết quả 3 bước. Gỡ 3 prompt này khỏi dropdown của tab Single Image và Batch.

## Bố cục (đã chốt bằng mockup)

Tab mới nằm sau `text2Text`, trước `📦 Model Library`.

```
Row
├─ Column (scale 1)                          ├─ Column (scale 2)
│  Image (pil, height 340)                   │  ── 1) Pose caption ───────────── [⚙]
│  Number "Video duration (seconds)" = 5     │  [▶ Analyze character]
│  Slider "Max New Tokens" 64..1024 = 512    │  Textbox "Character / pose caption"
│  Accordion "Resize options" (đóng)         │     (interactive, copy button)
│     Dropdown mode / Slider W / Slider H    │  HTML status
│  ── Export ──                              │
│  [💾 Export txt]                           │  ── 2) Video ideas ────────────── [⚙]
│  HTML status                               │  [▶ Suggest 3 ideas]
│                                            │  Dropdown "Ideas (EN | VI)"
│                                            │  Textbox "Your own description
│                                            │     (overrides dropdown)"
│                                            │  HTML status
│                                            │
│                                            │  ── 3) Video reel ─────────────── [⚙]
│                                            │  [▶ Generate reel]
│                                            │  Textbox "Reel prompt" (lines 8,
│                                            │     interactive, copy button)
│                                            │  HTML status
```

Mỗi phần là một khối: `gr.Row` chứa `gr.Markdown("### 1) Pose caption")` và
`gr.Button("⚙", size="sm", scale=0)`; ngay dưới là `gr.Column(visible=False)`
chứa `gr.Textbox("System prompt", lines=10)`, `gr.Row([Save, Restore default])`,
`gr.HTML` status. Bánh răng bật tắt cột này qua một `gr.State(bool)`.

Resize mặc định giống tab Single Image: "Adaptive image", 512 x 512.

## Prompt (captioner/prompts.py)

### Tách khỏi PROMPT_TEMPLATES

3 template Simple App bị xóa khỏi `PROMPT_TEMPLATES`, nên tự biến mất khỏi
`PROMPT_NAMES` và mọi dropdown. Cờ `raw_output` và tham số `raw` của
`_with_output_rules` không còn ai dùng, xóa luôn.

Thay vào đó là một dict mới:

```python
VIDEO_PROMPTS = {
    "pose":  {"system_prompt": ..., "user_prompt": ...},
    "ideas": {"system_prompt": ..., "user_prompt": ...},
    "reel":  {"system_prompt": ..., "user_prompt": ...},
}
```

`system_prompt` là phần luật, người dùng sửa được qua bánh răng. `user_prompt`
cố định, chỉ chứa dữ liệu thay đổi qua placeholder.

| Bước | system_prompt | user_prompt |
|------|---------------|-------------|
| pose | Luật: một câu tiếng Anh, tả tư thế, vị trí trong khung, trang phục hoặc vật cầm, khung máy (khoảng cách, góc). Không bình luận, không liệt kê. | `Describe the provided image.` |
| ideas | Luật: đạo diễn video dọc ngắn, đúng 3 ý tưởng khác nhau bắt đầu từ ảnh, mỗi ý là một cú máy liên tục `{duration}` giây, cùng chỗ, cùng người, cùng trang phục, hành động nhỏ tự nhiên. **Định dạng: đúng 3 dòng, mỗi dòng `<câu tiếng Anh> ||| <mô tả tiếng Việt>`, không đánh số, không thêm chữ.** | `The clip lasts {duration} seconds.` |
| reel | Toàn bộ system prompt "Video Reel (Simple App)" hiện có (one continuous shot, same person, camera rules, output rules `(at X seconds: …)`). | `Character: {pose}` / `Goal: {idea}` / `The clip lasts {duration} seconds.` (3 dòng) |

Nội dung tiếng Anh của system prompt ideas và pose được viết lại từ user prompt
hiện có, giữ nguyên ý, thêm dòng định dạng `|||` cho ideas.

### Hàm mới

```python
VIDEO_PROMPT_OVERRIDE_FILE = Path(__file__).resolve().parent.parent / "prompts_override.json"

def load_video_prompt_overrides() -> dict[str, str]
    # Đọc file nếu có. JSON hỏng hoặc không phải dict → trả {} (không raise).

def save_video_prompt_override(step: str, system_prompt: str) -> None
    # Đọc file, đặt overrides[step] = text.strip(); text rỗng → xóa key.
    # Ghi lại file (ensure_ascii=False, indent=2). Không còn key nào → xóa file.

def get_video_system_prompt(step: str) -> str
    # override nếu có, ngược lại VIDEO_PROMPTS[step]["system_prompt"].

def resolve_video_prompt(step: str, system_prompt: str, **fields) -> tuple[str, str]
    # Trả (system_prompt.strip(), _format(user_prompt, **fields)).
    # fields: duration, idea, pose. Placeholder thiếu được giữ nguyên (_SafeDict).

def parse_ideas(text: str) -> list[tuple[str, str]]
    # Mỗi dòng không rỗng → (en, vi). Bỏ tiền tố đánh số "1." "1)" "-" "*".
    # Tách bằng "|||"; không có "|||" → (cả dòng, ""). Tối đa 3 mục.
```

`prompts_override.json` nằm ở thư mục gốc project, thêm vào `.gitignore`.

## Luồng chạy (app.py)

Hàm dùng chung:

```python
def _vr_generate(step, system_prompt, tmp_path, max_tokens, **fields) -> str
    # resolve_video_prompt → _captioner.caption_image(tmp_path, user, max_tokens, system_prompt=system)
    # Trả text.strip().
```

Ảnh được resize và ghi tạm một lần cho mỗi lần bấm nút, xóa ở `finally`
(dùng lại `_save_resized_temp_image`).

`duration` từ `gr.Number` được ép `max(1, int(duration or 5))` trước khi đưa vào
prompt, để không bao giờ ra "5.0 seconds" hay "0 seconds".

### Nút 1: Analyze character

Input: image, pose_system, max_tokens, resize (mode, w, h).
Output: pose_text, pose_status.
Post-process: `" ".join(text.split())` để ra đúng một dòng.

### Nút 2: Suggest 3 ideas

Input: image, duration, ideas_system, max_tokens, resize.
Output: `gr.update(choices=[(f"{en} | {vi}", en), ...], value=en_đầu)`, ideas_status.
Parse bằng `parse_ideas`. Nếu parse ra 0 mục → status cảnh báo, dropdown giữ nguyên.

Dropdown dùng choices dạng `(label, value)`: label hiện cả EN và VI, value là
câu tiếng Anh. `value` của dropdown chính là idea đưa vào bước 3.

### Nút 3: Generate reel

Input: image, duration, pose_text, idea_custom, idea_choice, pose_system,
ideas_system, reel_system, max_tokens, resize.
Output: pose_text, ideas dropdown update, reel_text, reel_status.

Thứ tự:
1. Chưa load model hoặc chưa có ảnh → cảnh báo, không đổi gì.
2. `pose = pose_text.strip()`; rỗng → chạy bước pose, điền vào ô.
3. `idea = idea_custom.strip() or (idea_choice or "")`; rỗng → chạy bước ideas,
   điền dropdown, lấy mục đầu. Vẫn rỗng (model không trả gì) → cảnh báo, dừng.
4. Chạy bước reel với `pose`, `idea`, `duration`.
5. Reel output: bỏ dòng trống, giữ nguyên phần còn lại.
6. Status ghi rõ bước nào được chạy tự động, ví dụ
   "Reel generated (auto: pose, ideas → idea 1)".

Ô pose và dropdown chỉ bị cập nhật khi bước đó được chạy tự động; ngược lại trả
`gr.update()` để giữ nguyên.

### Bánh răng (3 phần giống nhau)

- Click ⚙: `state = not state`, cột `visible=state`.
- Textbox system prompt khởi tạo bằng `get_video_system_prompt(step)`.
- Các nút chạy đọc system prompt **trực tiếp từ textbox** này, nên sửa là có hiệu
  lực ngay, chưa cần Save.
- Save: `save_video_prompt_override(step, text)` → badge "Saved to prompts_override.json".
- Restore default: `save_video_prompt_override(step, "")`, trả
  `VIDEO_PROMPTS[step]["system_prompt"]` vào textbox → badge "Default restored".

### Export txt

Input: pose_text, idea_custom, idea_choice, reel_text. Output: export_status.

```
idea = idea_custom.strip() or (idea_choice or "")
parts = [pose.strip(), f"Idea: {idea}" if idea else "", reel.strip()]
content = "\n\n".join(p for p in parts if p)
```

Rỗng hoàn toàn → cảnh báo "Nothing to export". Ngược lại ghi
`~/Desktop/video_reel_<YYYYmmdd_HHMMSS>.txt` (utf-8) → badge đường dẫn.

## Thay đổi khác

- `README.md`: bỏ 3 dòng Simple App khỏi bảng template, thêm mục "Video Reel tab"
  mô tả 3 bước, fallback, bánh răng và `prompts_override.json`.
- Tab Help trong `app.py`: thêm 4 dòng tương tự.
- `.gitignore`: thêm `prompts_override.json`.
- Kiểm tra `batch_caption.py` và `test_caption.py` không tham chiếu 3 tên template
  cũ (hiện chúng chỉ dùng `get_prompt_names`, nên không cần sửa).

## Xử lý lỗi

- Mọi handler bắt `Exception`, trả badge `error`, giữ nguyên các ô output.
- File override hỏng → coi như không có, in cảnh báo ra console một lần khi load.
- Model trả ideas không đúng định dạng → `parse_ideas` vẫn tận dụng được dòng không
  có `|||` (VI rỗng); 0 dòng → cảnh báo.

## Kiểm thử

`test_video_prompts.py` ở thư mục gốc, chạy `python test_video_prompts.py`, không
cần model, chỉ `assert`:

- `parse_ideas`: 3 dòng chuẩn; dòng có đánh số; dòng thiếu `|||`; input rỗng; hơn
  3 dòng bị cắt còn 3.
- `resolve_video_prompt("reel", ...)`: user prompt chứa đúng pose, idea, duration;
  system prompt là chuỗi truyền vào.
- Override round-trip với `VIDEO_PROMPT_OVERRIDE_FILE` trỏ tạm sang thư mục tmp:
  save → get trả override; save "" → get trả default và file bị xóa; file JSON
  hỏng → load trả {}.
- `get_prompt_names()` không còn tên nào chứa "Simple App".

Phần UI và chuỗi 3 bước kiểm tra tay trong trình duyệt với model đã load.

## Ngoài phạm vi

- Không sửa user prompt qua UI.
- Không batch nhiều ảnh.
- Không lưu lịch sử ideas hay reel.
