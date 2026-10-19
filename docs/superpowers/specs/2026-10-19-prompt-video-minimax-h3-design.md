# Tab "Prompt Video Minimax H3"

Ngày: 2026-10-19. Nhánh: `20261019_PromptVideoMiniMaxH3` (tạo từ `20261011_add_video_reel_tab`).

## Mục tiêu

Làm lại workflow ComfyUI `H3_STORY_PLANNER.json` thành một tab Gradio dễ kiểm tra:
ảnh nhân vật (+ ảnh phụ) và story → **CHARACTER & SCENE DEFINITION** và
**SHOT STRUCTURE — MINIMAX H3** theo `docs/SYSTEM_PROMPT_MINIMAX_H3.md` §10.2 / §10.3.
Story trống thì gợi ý 3 ý tưởng (SFW/NSFW theo checkbox), chọn một ý tưởng rồi viết
chi tiết vào ô Story, sau đó mới Generate.

Code độc lập hoàn toàn với ComfyUI: không import, không copy file từ custom node.
System prompt viết mới từ 3 tài liệu trong `docs/`.

## Bố cục (đã chốt bằng mockup 2 cột)

Tab mới nằm sau `🎬 Video Reel`, trước `📦 Model Library`.

```
Row
├─ Column (scale 1) — đầu vào                 ├─ Column (scale 1) — đầu ra
│  ── Ảnh tham chiếu ──                         │  [▶ Generate prompt] (khóa khi Story trống) [⚙ shots]
│  Image "Picture 1 · Nhân vật (bắt buộc)"     │  HTML status
│  Row: Textbox path | [📂]                     │  Textbox "CHARACTER & SCENE DEFINITION"
│  Row ×3 cột: mỗi cột = ảnh phụ N:             │     (lines 12, interactive, copy)
│     Image, Dropdown vai trò, Textbox path,[📂]│  Textbox "SHOT STRUCTURE — MINIMAX H3"
│  Row: Checkbox NSFW | Number "Tổng (s)" = 0   │     (lines 20, interactive, copy)
│  Accordion "Resize options" (đóng)            │  Row: Textbox "Saved file" (read-only) | [💾 Lưu lại]
│  ── 1) Character bible ── [⚙ bible]           │  Textbox "Logger" (lines 10, read-only)
│  Accordion (đóng): Textbox bible (sửa được)   │
│  ── 2) Story ──                               │
│  Textbox Story (lines 6)                      │
│  [💡 Gợi ý 3 ý tưởng] [⚙ ideas]               │
│  Radio ideas (3 lựa chọn, rỗng lúc đầu)       │
│  [✍ Viết chi tiết vào Story] (khóa khi chưa   │
│     chọn ý tưởng) [⚙ expand]                  │
```

Vai trò ảnh phụ: `Face sheet / góc mặt`, `Bối cảnh`, `Trang phục / đạo cụ`. Mặc định
lần lượt Face sheet, Bối cảnh, Trang phục. Nhiều ô được trùng vai trò.

Resize giống tab Video Reel: "Adaptive image", 512 × 512, áp dụng cho mọi ảnh.

## Luồng và trạng thái nút

```
ảnh ──► [Bible] (vision, tự chạy nếu ô Bible trống) ──► bible text
                                         │
Story trống ──► [Gợi ý] ──► 3 ý tưởng ──► chọn ──► [Viết chi tiết] ──► Story có timecode
                                                                         │
Story có nội dung ──────────────────────────────────────────► [Generate] ┘
   └─► shot plan (code) ──► [Shots] (text-only) ──► tách Part 1 / Part 2 ──► lưu .txt
```

- `Generate` có `interactive` = `bool(story.strip())`, cập nhật bằng `story.change`.
  Gõ tay vào Story cũng mở khóa.
- `Viết chi tiết` có `interactive` = có ý tưởng đang được chọn (`ideas_radio.change`).
- `Gợi ý`, `Viết chi tiết` và `Generate` tự chạy Bible trước nếu ô Bible trống. Nếu ô Bible có nội dung
  thì dùng nguyên văn (người dùng có thể đã sửa).
- Đổi bất kỳ ảnh nào, đường dẫn hay vai trò → xóa Bible, Radio ý tưởng (choices rỗng),
  2 ô output, ô Saved file. Story giữ nguyên.
- Chỉ bước Bible gửi ảnh. Ideas, Expand, Shots chỉ gửi chữ (`caption_text`).

## Ảnh và đường dẫn gốc

Trình duyệt không cho biết đường dẫn gốc của file kéo thả, nên mỗi ô ảnh có Textbox đường
dẫn và nút 📂.

- 📂 mở `tkinter.filedialog.askopenfilename` (cửa sổ `Tk()` ẩn, `-topmost`, lọc
  `*.png *.jpg *.jpeg *.webp *.bmp *.heic`), chạy trên máy chủ vì app chạy local.
  Chọn xong → điền Textbox path. Hủy → giữ nguyên.
- Textbox path `.change` → mở ảnh (`PIL.Image.open` + `ImageOps.exif_transpose`,
  `.strip().strip('"')` để nhận "Copy as path") và đặt vào ô Image. Đường dẫn không tồn
  tại → ô Image giữ nguyên, logger ghi cảnh báo.
- Kéo thả thẳng vào ô Image (không qua path) → path coi như rỗng.
- Đường dẫn Picture 1 dùng để đặt tên và vị trí file lưu (mục Lưu file).

### Đánh số Picture

Chỉ ô có ảnh mới được đánh số, liên tục, theo thứ tự ô: Picture 1 luôn là nhân vật,
các ảnh phụ có ảnh lần lượt là Picture 2, 3, 4 với `ref_image_{N-1}`.

Hàm thuần `build_references(roles: list[str | None]) -> list[dict]` (phần tử đầu là
Picture 1, `None` = ô trống) trả `[{"n": 1, "ref": "ref_image_0", "role": "character"}, ...]`
và `describe_references(refs) -> str` sinh khối `REFERENCE PICTURES:` cho prompt:

```
<Picture 1>, ref_image_0 = main character: identity, body, outfit. Also the setting, because no scene picture is given.
<Picture 2>, ref_image_1 = scene / environment. Ignore any person visible in it.
```

Câu "Also the setting…" chỉ có khi không có ảnh vai trò Bối cảnh (yêu cầu Q2: không có
ảnh phụ thì Picture 1 dùng cho cả nhân vật và bối cảnh). Face sheet: "same character,
use for face details only". Trang phục: "outfit / props worn or held by <Subject 1>".

## Model: nhiều ảnh và temperature

`BaseCaptioner`, `HFCaptioner`, `GGUFCaptioner`:

```python
def caption_image(self, image_path: str | list[str], user_prompt, max_new_tokens=512,
                  system_prompt=None, temperature: float | None = None) -> str
def caption_text(self, user_prompt, max_new_tokens=512, system_prompt=None,
                 temperature: float | None = None) -> str
```

- `image_path` là `str` → hành vi cũ. Là `list` → mỗi ảnh một phần tử content, theo thứ
  tự, rồi tới text. HF: `{"type": "image", "image": "file://..."}`. GGUF:
  `{"type": "image_url", ...}` (chat handler Qwen/Llava của llama-cpp nhận nhiều ảnh).
- `temperature is None` → giữ nguyên tham số sampling hiện tại (HF greedy; GGUF
  `temperature` 0.6/0.4, `seed=1`). Mọi tab cũ không đổi.
- `temperature` có giá trị → HF `do_sample=True, temperature=t, top_p=0.9` (t ≤ 0 thì
  greedy); GGUF `temperature=t`, `seed=None`, để bấm lại "Gợi ý" ra ý tưởng khác.

## System prompt: `captioner/system_prompt.json`

File riêng cho tab này, commit vào git, không liên quan `prompts_override.json`.

```json
{
  "bible":       {"system": "...", "user": "...", "temperature": 0.3, "max_tokens": 768},
  "ideas_sfw":   {"system": "...", "user": "...", "temperature": 0.7, "max_tokens": 1024},
  "ideas_nsfw":  {...},
  "expand_sfw":  {"system": "...", "user": "...", "temperature": 0.6, "max_tokens": 1024},
  "expand_nsfw": {...},
  "shots_sfw":   {"system": "...", "user": "...", "temperature": 0.4, "max_tokens": 3072},
  "shots_nsfw":  {...}
}
```

`user` là template `str.format_map` với placeholder (thiếu → giữ nguyên, như `_SafeDict`):

| Bước | Placeholder trong `user` |
|------|--------------------------|
| bible | `{references}` |
| ideas | `{bible}`, `{references}`, `{duration_rule}` |
| expand | `{bible}`, `{references}`, `{idea}`, `{shot_plan}` |
| shots | `{bible}`, `{references}`, `{story}`, `{shot_plan}` |

`duration_rule`: tổng > 0 → `"Each idea lasts exactly {T} seconds in total."`; = 0 →
`"Each idea lasts 10 to 36 seconds in total."`. `shot_plan`: xem mục Thời lượng.

### Nội dung prompt (viết mới, tiếng Anh, ngắn cho model 4B/8B)

- **bible**: phân tích ảnh theo vai trò trong `{references}`; đúng 11 dòng `NAME_HINT,
  AGE_APPEARANCE (kèm "adult"/"minor"), FACE, HAIR, BODY, OUTFIT, ACCESSORIES, SETTING,
  LIGHTING, CAMERA_FRAMING, STYLE`; chỉ mô tả cái nhìn thấy; SETTING lấy từ ảnh Bối cảnh
  nếu có, không thì từ Picture 1; người trong ảnh bối cảnh bị bỏ qua; không đoán danh tính.
- **ideas_sfw / ideas_nsfw**: đúng 3 ý tưởng khác thể loại/mood, dựa trên bible và bối
  cảnh; mỗi shot ≤ 12s; viết tiếng Việt; định dạng cố định:
  ```
  Ý TƯỞNG 1: <tên ngắn>
  Logline: <1-2 câu>
  Tổng thời lượng: <N>s
  Số shot: <K>
  Thời lượng shot: <a>s + <b>s + ...
  ```
- **expand_sfw / expand_nsfw**: viết story tiếng Việt chi tiết cho ý tưởng đã chọn, đúng
  theo `{shot_plan}`, mỗi đoạn một dòng `MM:SS–MM:SS — <hành động, camera, âm thanh>` với
  timecode toàn cục liên tục; lời thoại (nếu có) trong ngoặc kép; không thêm chữ ngoài story.
- **shots_sfw / shots_nsfw**: đạo diễn MiniMax H3; tiếng Anh cho mô tả, lời thoại giữ nguyên
  văn trong Story, không bịa thoại; dùng đúng `<Subject N>`, `<Picture N>`, `ref_image_N`
  theo `{references}`; ảnh không có thì không nhắc; đúng `{shot_plan}`; timecode mỗi shot
  bắt đầu 00:00 và kết thúc bằng duration của shot; nét mặt thay đổi tự nhiên; âm thanh
  đúng hành động. Output đúng 2 phần, không markdown:
  ```
  CHARACTER & SCENE DEFINITION
  subject_definitions: ...
  summary: [reference generation] ...
  retention_analysis: ...
  overall_soundscape: ...
  non_diegetic_music: ...

  SHOT STRUCTURE — MINIMAX H3
  [SHOT 01 | DURATION: <N>s | PRIMARY CAMERA SETUP: ...]
  subject_definitions: ...   (recap ngắn)
  summary: ...
  retention_analysis: ...
  detailed_description:
  00:00–00:0X — ...
  overall_soundscape: ...    (âm thanh riêng của shot này)
  non_diegetic_music: ...
  ```
- Bản **nsfw**: nội dung người lớn được phép; giới hạn cứng luôn giữ: mọi nhân vật là
  người lớn, không từ ngữ gợi tuổi nhỏ; `AGE_APPEARANCE` có "minor" → chỉ viết SFW; nhân
  vật hư cấu, không người thật; đồng thuận rõ ràng, không bạo lực/ép buộc/phi pháp.

### Hàm (captioner/h3_prompts.py, thuần, không import gradio/torch)

```python
H3_PROMPT_FILE = Path(__file__).resolve().parent / "system_prompt.json"

def step_key(step: str, nsfw: bool) -> str            # "bible" → "bible"; "shots", True → "shots_nsfw"
def load_h3_prompts() -> dict                          # đọc file mỗi lần gọi; lỗi → raise ValueError rõ ràng
def get_h3_step(step, nsfw) -> dict                    # {"system","user","temperature","max_tokens"}
def save_h3_system(step, nsfw, text) -> None           # ghi "system" của key vào file (ensure_ascii=False, indent=2)
def render_user(template: str, **fields) -> str        # format_map, placeholder thiếu giữ nguyên
```

⚙ của mỗi bước: Textbox system prompt (lines 12) + nút 💾 Save + HTML status, ẩn/hiện
bằng `gr.State` như `_vr_section`. Textbox khởi tạo bằng `get_h3_step(step, False)["system"]`;
checkbox NSFW đổi → nạp lại 3 Textbox ideas/expand/shots theo bản tương ứng. Các nút chạy
đọc `system` **từ Textbox** (sửa là có hiệu lực ngay), còn `user`, `temperature`,
`max_tokens` đọc từ file mỗi lần chạy. Không có Restore default: git là bản gốc.

## Thời lượng (luật 12s/shot do code ép)

```python
MAX_SHOT = 12
def clamp_durations(total: float) -> list[float]
    # ≤12 → [total]; 12<total≤15 → [12] (nén); >15 → n=ceil(total/12) shot bằng nhau,
    # làm tròn 0.5s, shot cuối bù phần dư để tổng đúng bằng total.
def shot_plan_from_story(story: str) -> list[tuple[float, float]]
    # Mọi "MM:SS–MM:SS" (– hoặc -) trong story, theo thứ tự; mỗi đoạn đi qua clamp_durations
    # → danh sách (start, duration) toàn cục. Không có timecode → [].
def shot_plan_from_total(total: float) -> list[tuple[float, float]]
def format_shot_plan(plan) -> str
    # "SHOT 01: story 00:00–00:10 (10s)\nSHOT 02: ..." ; plan rỗng →
    # "Choose 5 to 12 seconds per shot."
```

Thứ tự chọn plan ở bước Shots: timecode trong Story → `Tổng (s)` > 0 → để model tự chọn.
Ở bước Expand: plan lấy từ thời lượng shot của ý tưởng đã chọn (cộng dồn thành start).

Logger ghi plan và mọi chỗ bị nén (vd. "Story 00:00–00:14 (14s) → nén còn 12s").

## Hậu xử lý output

```python
def clean_output(text) -> str          # NFC, bỏ ``` và tiền tố "#" đầu dòng, bỏ "**"
def split_h3_output(text) -> tuple[str, str]
    # Part 1 = từ "subject_definitions:" đầu tiên tới trước "[SHOT" đầu tiên;
    #   bỏ dòng tiêu đề "CHARACTER & SCENE DEFINITION" / "SHOT STRUCTURE — MINIMAX H3".
    # Part 2 = từ "[SHOT" đầu tiên tới hết.
    # Không có "[SHOT" → raise ValueError("no [SHOT ...] block").
def check_shots(part2) -> tuple[str, list[str]]
    # Đánh số lại SHOT 01..NN; DURATION > 12 → sửa thành 12s; shot thiếu overall_soundscape
    # hoặc non_diegetic_music → thêm "overall_soundscape:\nNatural ambient sound, no dialogue."
    # / "non_diegetic_music:\nNONE". Trả (part2 đã sửa, danh sách cảnh báo cho logger).
def parse_h3_ideas(text) -> list[dict]
    # Tách theo "Ý TƯỞNG N"; mỗi ý: {"name","logline","shots":[float]}. Tổng tính lại =
    # sum(shots); shot > 12 được thay bằng clamp_durations(shot). Thiếu "Thời lượng shot"
    # → shots = clamp_durations(tổng ghi trong text, mặc định 12). Tối đa 3.
def idea_label(i, idea) -> str         # "Ý tưởng 1: <name> — <logline> · 20s · 2 shot: 10s + 10s"
def idea_text(i, idea) -> str          # khối 5 dòng như định dạng ở trên, dùng để lưu file và đưa vào expand
```

Radio ý tưởng: `choices=[(idea_label(i, x), str(i)) ...]`, danh sách ý tưởng giữ trong
`gr.State` để bước Expand lấy theo index.

## Lưu file

```python
def h3_output_path(picture1_path: str, nsfw: bool, ideas: bool = False) -> Path
    # Có path tồn tại → <thư mục ảnh>/<stem>_H3_<sfw|nsfw>[_ideas].txt
    # Không có → ~/Desktop/h3_<YYYYmmdd_HHMMSS>_<sfw|nsfw>[_ideas].txt
def build_h3_file(part1, part2) -> str
    # "CHARACTER & SCENE DEFINITION\n\n{part1}\n\nSHOT STRUCTURE — MINIMAX H3\n\n{part2}\n"
```

- Sau Generate thành công → ghi đè file shots, điền ô Saved file.
- Sau Gợi ý thành công → ghi file `_ideas.txt` (3 khối `idea_text`).
- 💾 Lưu lại → ghi nội dung hiện tại của 2 ô (đã sửa tay) vào đường dẫn trong ô Saved file;
  ô đó trống thì tính bằng `h3_output_path`.
- utf-8. Lỗi ghi → status error + logger, không mất nội dung ô.

## Logger

`H3Log` nhỏ trong app.py: danh sách dòng `"[HH:MM:SS] ..."`, giữ 200 dòng cuối, mỗi
dòng cũng gọi `logger.info`. Giữ trong `gr.State(list)` để mỗi phiên trình duyệt một log.
Mỗi handler nhận state log, trả lại state + `"\n".join(lines)` cho Textbox Logger.

Nội dung ghi: danh sách Picture N (đường dẫn hoặc "(uploaded, no path)", vai trò); mỗi
bước start (key prompt, temperature, max_tokens) / done (số giây, số ký tự); bible
`k/11 fields`; số ý tưởng; shot plan và chỗ nén; cảnh báo `check_shots`; file đã lưu; lỗi
(kèm 300 ký tự đầu của output thô khi tách thất bại).

## Xử lý lỗi

- Chưa load model / chưa có Picture 1 → badge warning, không gọi model.
- Mọi handler bắt `Exception` → badge error + logger, giữ nguyên các ô output.
- `system_prompt.json` hỏng/thiếu key → badge error "system_prompt.json: <lỗi>".
- Output Shots không tách được → toàn bộ output thô vào ô 1, ô 2 rỗng, badge warning,
  không lưu file.
- Ý tưởng parse ra 0 mục → badge warning, Radio giữ nguyên.

## Kiểm thử

`test_h3_prompts.py` ở thư mục gốc, chạy `python test_h3_prompts.py`, chỉ `assert`, không
cần model (cùng kiểu `test_video_prompts.py`):

- `build_references` / `describe_references`: chỉ Picture 1 → có câu "Also the setting";
  có ảnh Bối cảnh → không có câu đó, có "Ignore any person"; ô trống ở giữa không làm thủng
  số thứ tự; `ref_image_N` = N-1.
- `clamp_durations`: 10 → [10]; 12 → [12]; 14 → [12]; 16 → [8, 8]; 30 → tổng 30, mỗi shot ≤ 12.
- `shot_plan_from_story`: 2 đoạn timecode; đoạn 00:00–00:20 → 2 shot 10s; không timecode → [].
- `split_h3_output`: output chuẩn; có lời dẫn trước `subject_definitions:`; có tiêu đề và
  ```; không có `[SHOT` → ValueError.
- `check_shots`: đánh số lại; DURATION 15s → 12s + cảnh báo; shot thiếu soundscape được thêm.
- `parse_h3_ideas`: 3 khối chuẩn; có `**` và `#`; thiếu dòng thời lượng shot; shot 20s bị
  chia; hơn 3 khối bị cắt.
- `render_user`: điền placeholder, thiếu placeholder giữ nguyên.
- `system_prompt.json`: đủ 7 key, mỗi key đủ 4 trường; `save_h3_system` round-trip trên
  bản copy tạm (trỏ `H3_PROMPT_FILE` sang thư mục tmp).
- `h3_output_path`: có path → cạnh ảnh đúng tên; không path → Desktop.

Kiểm tra tay trong trình duyệt với model đã load (HF và GGUF): 1 ảnh; 1 ảnh + bối cảnh;
NSFW bật/tắt; luồng ý tưởng → viết chi tiết → Generate; nút 📂; đổi ảnh xóa output; file
txt nằm cạnh ảnh gốc; tab Single Image vẫn ra caption như cũ (temperature None).

## Thay đổi khác

- `README.md` và tab Help: thêm mục "Prompt Video Minimax H3".
- Commit 3 tài liệu `docs/SYSTEM_PROMPT_MINIMAX_H3.md`, `docs/sample_Identify.txt`,
  `docs/Sample_Scence.txt` cùng spec.

## Ngoài phạm vi

- Batch nhiều ảnh, lịch sử ý tưởng, nhân vật thứ hai (Subject khác).
- Chỉnh `user` template, temperature, max_tokens trên UI (sửa trực tiếp file JSON).
- Bản prompt rút gọn (§13), sửa từng phần riêng.
