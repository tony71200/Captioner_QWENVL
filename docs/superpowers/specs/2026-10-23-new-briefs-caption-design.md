# Template "New Briefs Caption"

Ngày: 2026-10-23. Nhánh: `20261023_newprompt_underwear` (tạo từ `20261019_PromptVideoMiniMaxH3`).

## Mục tiêu

Thêm một template caption vào `captioner/prompts.py` để sinh caption train LoRA
**kiểu cắt quần lót nam** trên Krea 2 (text encoder Qwen3-VL). Caption phải theo
`krea2_briefs_lora_guide.html` mục 4 "Caption đúng cách": trigger đúng 1 lần trong cụm
`wearing [color] trigger underwear`, không mô tả lại kiểu cắt, không chép chữ thương hiệu,
câu tiếng Anh tự nhiên. Thứ tự slot theo format người dùng đưa, có bổ sung slot mô tả người.

Chạy thử trên 66 ảnh chưa có `.txt` trong
`C:\Users\tonyn\Downloads\briefs_lora_dataset_captions\briefs_lora_dataset\dataset\*`
bằng `batch_caption.py`, rồi kiểm tra tự động bằng hàm `check_briefs_caption`.

## Nguồn tham chiếu

| File | Dùng cho |
|------|----------|
| `C:\Users\tonyn\Downloads\briefs_lora_dataset_captions\krea2_briefs_lora_guide.html` | Mục 2 (định nghĩa trigger), mục 3 (khung hình, ảnh mất đầu), mục 4 (quy tắc caption, system prompt gợi ý, danh sách từ cấm). |
| `C:\Users\tonyn\Downloads\briefs_lora_dataset_captions\briefs_report.html` | Tiêu chí phân loại 7 dạng briefs: độ rộng sườn, độ cao cắt chân, độ hở lưng, cạp, chất liệu. |
| `.../briefs_lora_dataset/dataset/` | 215 ảnh, 149 caption có sẵn (44–71 từ) làm chuẩn văn phong. |

## Quyết định đã chốt

| # | Câu hỏi | Chốt | Lý do |
|---|---------|------|-------|
| 1 | Vị trí trigger | **1 lần, trong cụm wearing.** Bỏ `<brief_trigger>,` ở đầu câu. | Guide mục 4: "Trigger xuất hiện đúng 1 lần, luôn ở mẫu wearing [color] trigger underwear". Ảnh đối chứng tự nhiên không có trigger. |
| 2 | Mô tả kiểu cắt | **Không.** Bỏ slot silhouette, rise, leg opening. Giữ màu, pouch structure, material/mesh, waistband. | Guide: "nếu viết ra thì trigger không học được gì". Pouch, embossed, sheer là thuộc tính tách ra caption. |
| 3 | Chữ thương hiệu | **Không chép.** Chỉ "logo waistband", "small printed text", "small captions", "headline text". | Guide cấm tên thương hiệu và chữ logo; 149 caption cũ cũng viết vậy. |
| 4 | 10 caption cũ `sporthc_brief` trong `01_tnga_brief` | **Đổi token thành `tnga_brief`.** | Thư mục 04 đã bị gộp vào 01; để nguyên sẽ train thêm một trigger không còn tồn tại. |
| 5 | Mô tả người | **Có:** build, màu da, tóc, râu, hình xăm; không có tuổi, không có hình dáng mặt. | Guide: mô tả người/pose trước để các token đó gánh phần mặt và dáng. Format người dùng chỉ có "an adult man", bổ sung theo guide. |
| 6 | Khung hình | **Đứng đầu câu**, có thêm "Cropped torso shot …, head out of frame" và góc nhìn từ sau. | Theo format người dùng; trùng ví dụ 3 của guide. Gần nửa dataset là ảnh mất đầu, guide mục 3 bắt buộc ghi rõ. |
| 7 | Độ dài | **50–90 từ** (guide: 40–80). | Thêm slot pouch, chất liệu, phụ kiện. Checker chấp nhận 40–100. |
| 8 | Ảnh đối chứng (`07_reg_classic`) | Ô trigger gõ `classic` → cụm "classic men's briefs", không trigger. | Guide: reg caption "classic men's briefs with a wide logo waistband". |
| 9 | Tên nhánh | `20261023_newprompt_underwear` | Git không cho khoảng trắng. |

## Trigger và định nghĩa kiểu cắt

Trigger chỉ mã hoá thứ bất biến: **kiểu cắt**. Mọi thứ thay đổi được nằm trong caption.
`sporthc_brief` bị bỏ; ảnh AIR MOVE nay thuộc `tnga_brief`.

| Trigger | Kiểu cắt (trigger học, caption KHÔNG viết) | Thuộc tính tách ra caption | Thư mục |
|---------|-------------------------------------------|---------------------------|---------|
| `tnga_brief` | Sườn hẹp 2–3 cm, cắt chân cao trên xương hông, lưng cheeky, chữ V sâu | màu, viền cạp ánh kim, logo waistband | `01_tnga_brief` |
| `strng_bikini` | Sườn dạng dây ~1 cm, cạp siêu thấp dưới xương chậu, thân tam giác nhỏ | lưới/mesh, khoá kim loại hông, vân kim loại | `02_strng_bikini` |
| `lrcontour_brief` | Cạp thấp, cạp mỏng dệt logo, ôm sát second-skin | dập nổi, lưng bán trong, màu, cỡ túi | `03_lrcontour_brief` |
| `hpstr_brief` | Cạp thấp–vừa, sườn rộng hơn bikini, cạp mảnh tone-on-tone | vải satin/ánh kim, màu | `05_hpstr_brief` |
| `dblstrap_bikini` | Sườn tách 2 dải tạo khe hở ở hông | slogan in cạp (ghi "printed waistband"), màu | `06_dblstrap_bikini` |
| *(không)* `classic` | Brief cạp bản to 1–1.5 inch, lưng kín | caption: "classic men's briefs with a wide logo waistband" | `07_reg_classic` |

## Định dạng caption

Ánh xạ từ format người dùng đưa:

```
<brief_trigger>, an adult man, [full-body | three-quarter | waist-up], [camera angle],
[pose and action], [face direction and expression],
wearing [color] [silhouette] [brief_trigger] with [rise], [leg opening], [pouch structure],
[material / mesh], [waistband details], [optional brand text if clearly legible],
[other visible clothing], [setting], [lighting], [composition].
```

| Slot người dùng | Trong template | Ghi chú |
|-----------------|----------------|---------|
| `<brief_trigger>,` đầu câu | bỏ | quyết định 1 |
| `an adult man` | giữ, thêm build / màu da / tóc / râu / xăm | quyết định 5 |
| `[full-body / three-quarter / waist-up]` | slot 1, đứng đầu: `Full-body shot` / `Medium shot from the knees up` / `Medium shot from the waist up` / `Cropped torso shot from chest to mid-thigh, head out of frame` | dùng đúng cụm từ của 149 caption cũ; "three-quarter" = knees up |
| `[camera angle]` | slot 2 | eye level, slightly low angle, high angle, side view, from behind, mirror selfie |
| `[pose and action]` | slot 4 | hướng thân (facing / three-quarter view / profile / seen from behind), tay, chân, vật cầm |
| `[face direction and expression]` | slot 5 | bỏ qua khi mặt ngoài khung |
| `wearing [color] [silhouette] [brief_trigger]` | `wearing [color] {garment}` | silhouette bỏ; `{garment}` = `tnga_brief underwear` hoặc `classic men's briefs` |
| `[rise]`, `[leg opening]` | bỏ | quyết định 2 |
| `[pouch structure]` | giữ | contoured, seamed, two-layer |
| `[material / mesh]` | giữ | smooth matte, satin sheen, fine mesh, embossed, ribbed |
| `[waistband details]` | giữ | width, color, metallic/contrast trim, logo waistband, small side buckle |
| `[optional brand text]` | bỏ | quyết định 3 |
| `[other visible clothing]` | slot 7 | hat, socks, watch; bỏ qua nếu không có |
| `[setting]`, `[lighting]` | slot 8, 9 | chữ quảng cáo và ảnh sản phẩm lồng chỉ ghi chung chung |
| `[composition]` | gộp vào slot 1 và 8 | tránh lặp khung hình |

### Ví dụ đầu ra mong đợi

Ảnh mất đầu, `tnga_brief`:

```
Cropped torso shot from chest to mid-thigh, head out of frame, side view of an adult muscular man with tanned skin, standing with hips turned slightly toward the camera and one hand resting on his thigh. He is wearing white tnga_brief underwear with a contoured seamed pouch in smooth matte fabric and a thin white logo waistband. Plain pale grey studio background, soft even studio lighting.
```

Ảnh thấy mặt, `hpstr_brief`:

```
Medium shot from the knees up at eye level of an adult lean East Asian man with short dark hair swept back and a faint stubble, standing facing the camera with one hand raised to his chin and the other on his hip, looking straight into the lens with a calm, serious expression. He is wearing bronze hpstr_brief underwear with a contoured pouch in satin-sheen fabric and a thin matching waistband with a small metal plate. Warm amber backdrop with a hard shadow behind him, single hard light from the upper left.
```

Ảnh đối chứng, `classic`:

```
Full-body shot at eye level of an adult athletic East Asian man with short black hair, kneeling with one arm raised behind his head, looking at the camera with a slight smile. He is wearing black classic men's briefs with a contoured pouch in smooth cotton-blend fabric and a wide white contrast logo waistband. Soft white-pink background with a large headline logo and small captions, soft even studio lighting.
```

Caption tồi (guide): `tnga_brief, sexy, bulge, high cut, white, man` → checker báo 4 lỗi.

## Prompt

Entry trong `PROMPT_TEMPLATES`, đặt sau "Train Lora Prompt (Following ChatGPT)":

```python
{
    "name": "New Briefs Caption",
    "description": "Caption train LoRA kiểu cắt quần lót nam (Krea 2 / Qwen3-VL): trigger đúng 1 lần trong cụm wearing, không mô tả kiểu cắt, không chép chữ brand. Gõ 'classic' cho ảnh đối chứng.",
    "name_field": True,
    "name_label": "Brief trigger word (classic = reference image)",
    "default_name": "tnga_brief",
    "training_caption": True,
    "system_prompt": ...,
    "user_prompt": ...,
}
```

`training_caption: True` → `_with_output_rules` nối `TRAINING_OUTPUT_RULES` (một dòng,
không PEOPLE marker, không negative prompt), giống template "Following ChatGPT".

### System prompt (luật cố định)

```
You caption training images for a LoRA that teaches men's underwear cuts. The garment in this image is labelled '{garment}'. The label alone carries the cut, so your words must carry everything else: the person, the pose, the framing, the color, the fabric and the waistband. Hard rules: write the label exactly once, verbatim, inside the phrase 'wearing [color] {garment}'. Never describe the cut: no rise, leg opening, side width or back coverage, and no cut words such as high-cut, low-rise, cheeky, narrow sides, string sides, thong, tanga, hipster or bikini outside the label itself. Never transcribe brand names, logo text, slogans or overlaid marketing text; call them 'logo waistband', 'small printed text', 'small captions' or 'headline text'. No sexual wording, never mention a bulge or anything under the garment, and never invent details that are not clearly visible. Output only the caption paragraph.
```

### User prompt (công thức slot, sửa được trong UI)

```
Describe this image as ONE flowing English paragraph of 50 to 90 words, in this order: 1) Framing: 'Full-body shot', 'Medium shot from the knees up', 'Medium shot from the waist up' or 'Cropped torso shot from chest to mid-thigh, head out of frame'; always say 'head out of frame' when the head is cut off. 2) Camera angle: eye level, slightly low angle, high angle, side view, from behind, or mirror selfie. 3) The person: 'an adult man' plus build (slim, lean, athletic, muscular, stocky), skin tone, hair style and color, facial hair and visible tattoos; skip whatever is not visible. 4) Pose and action: body orientation (facing the camera, three-quarter view, profile, seen from behind), arms, hands, legs, what he holds or leans on. 5) Face direction and expression: gaze, head turn, expression; skip when the face is out of frame. 6) Garment: 'wearing [color] {garment}' then pouch structure (contoured, seamed, two-layer), fabric or texture (smooth matte, satin sheen, fine mesh, embossed, ribbed) and waistband details (width, color, metallic or contrast trim, logo waistband, small side buckle); never the cut. 7) Other visible clothing or accessories (hat, socks, watch); skip if none. 8) Setting: background, surfaces, props; mention overlaid text or product insets only generically. 9) Lighting: direction and quality. Write flowing sentences, not a numbered list, and output only the caption.
```

Cả hai dùng placeholder `{garment}`; `{name}` vẫn nhận trigger thô nếu người dùng
tự gõ vào ô Custom.

## Thay đổi code

### `captioner/prompts.py`

- Hằng:
  - `BRIEF_REG_LABELS = frozenset({"classic", "reg", "none"})`
  - `BRIEF_REG_GARMENT = "classic men's briefs"`
  - `BRIEF_FORBIDDEN: tuple[str, ...]` — từ cấm, so khớp không phân biệt hoa thường, theo
    biên từ, sau khi đã xoá trigger khỏi text: `high-cut`, `high cut`, `low-rise`, `low rise`,
    `mid-rise`, `cheeky`, `narrow side`, `string side`, `side width`, `leg opening`,
    `back coverage`, `thong`, `tanga`, `hipster`, `bikini`, `bulge`, `sexy`, `crotch`,
    `genital`, `penis`, `attentionwear`, `attent!on`, `do_me`, `do-me`.
- `brief_garment(trigger: str) -> str`: `BRIEF_REG_GARMENT` nếu `trigger.strip().lower()`
  thuộc `BRIEF_REG_LABELS`, ngược lại `f"{trigger.strip()} underwear"`.
- `resolve_prompt`: tính `garment = brief_garment(name)` và truyền `garment=garment` vào cả
  3 lần gọi `_format` (system prompt và user prompt, cả nhánh custom). Template khác không
  dùng `{garment}` nên không đổi. `_SafeDict` giữ nguyên `[color]` vì không phải ngoặc nhọn.
- `check_briefs_caption(text: str, trigger: str) -> list[str]`: trả danh sách lỗi, rỗng là
  đạt. Luật:
  1. `multiline`: nhiều hơn 1 dòng không rỗng.
  2. `trigger x<n>`: trigger (hoặc `BRIEF_REG_GARMENT` với reg) xuất hiện khác 1 lần, so
     khớp theo biên từ.
  3. `stray trigger <tok>`: với reg, text chứa token dạng `\w+_(brief|bikini)`.
  4. `forbidden: <từ>`: mỗi từ cấm tìm thấy.
  5. `words <n>`: số từ ngoài 40–100.

### `batch_caption.py`

- `--name` (mặc định rỗng): trigger word hoặc tên nhân vật cho template có `name_field`.
- `build_prompt(template, prefix, name="")` → `resolve_prompt(template, "", name)`.
- `self_test`: vòng lặp kiểm tra output rules phân biệt template train bằng
  `get_prompt_by_name(tname).get("training_caption")` thay vì so tên; assert thêm cho
  template mới (có `tnga_brief underwear`, không còn `{garment}`, không `PEOPLE: <n>`;
  `classic` → `classic men's briefs`).
- Docstring usage thêm ví dụ `--template "New Briefs Caption" --name tnga_brief --prefix ""`.

### `test_briefs_prompts.py` (mới)

Assert thuần, chạy `.venv\Scripts\python.exe test_briefs_prompts.py`:

- `brief_garment("tnga_brief") == "tnga_brief underwear"`; `"classic"`, `" Reg "`, `"none"`
  → `BRIEF_REG_GARMENT`.
- `resolve_prompt("New Briefs Caption", "", "tnga_brief")`: cả system và user chứa
  `wearing [color] tnga_brief underwear`, không chứa `{garment}`, user kết thúc bằng
  `TRAINING_OUTPUT_RULES`, không chứa `PEOPLE: <n>`.
- Ô trống → default `tnga_brief`; `classic` → `classic men's briefs`.
- `check_briefs_caption` trả `[]` cho 3 ví dụ của guide (trigger `tnga_brief`,
  `dblstrap_bikini`, `strng_bikini`) và ví dụ reg (trigger `classic`).
- Caption tồi của guide với `tnga_brief` → có lỗi `forbidden: sexy`, `forbidden: bulge`,
  `forbidden: high cut`, `words 7`.
- Ví dụ reg trong test lấy từ dataset (`reg_mrb-ny_04.txt`, "black classic men's briefs"),
  vì ví dụ reg của guide viết "classic black men's briefs" không khớp `BRIEF_REG_GARMENT`.
- Trigger 2 lần → `trigger x2`; reg có `tnga_brief` → `stray trigger tnga_brief`;
  2 dòng → `multiline`.
- `strng_bikini` caption không bị báo `forbidden: bikini` (trigger đã xoá trước khi so).

### `README.md`

Mục "Batch a whole folder": thêm 1 ví dụ dùng `--template "New Briefs Caption" --name
tnga_brief --prefix ""`, và 1 câu nói template này ghi caption train LoRA, không PEOPLE.

### Không đổi

`app.py` (ô name đã hiện theo `name_field`), `hf_captioner.py`, `gguf_captioner.py`,
`utils/`. Không thêm dependency.

## Chạy thử nghiệm

Môi trường: `.venv\Scripts\python.exe` (llama-cpp-python 0.3.39), GPU RTX 5070 Ti 12 GB.
Model mặc định của `batch_caption.py`:
`D:\001_Personal_Proj\Comfy\ComfyUI\models\llm\GGUF\Qwen3-VL-8B-Instruct-GGUF\Qwen3VL-8B-Instruct-Q4_K_M.gguf`
+ mmproj F16 tự chọn, profile `NormalVRAM (12–16GB)`.

Bước 0: thay `sporthc_brief` → `tnga_brief` trong 10 file `01_tnga_brief/sporthc_brief_*.txt`.

Bước 1: mỗi thư mục một lần, trigger theo tên thư mục (bỏ tiền tố `NN_`; `07_reg_classic`
→ `classic`):

| Thư mục | `--name` | Ảnh thiếu `.txt` |
|---------|----------|------------------|
| `01_tnga_brief` | `tnga_brief` | 34 |
| `02_strng_bikini` | `strng_bikini` | 8 |
| `03_lrcontour_brief` | `lrcontour_brief` | 10 |
| `05_hpstr_brief` | `hpstr_brief` | 12 |
| `06_dblstrap_bikini` | `dblstrap_bikini` | 0 (bỏ qua) |
| `07_reg_classic` | `classic` | 2 |

```
python batch_caption.py --images "<dataset>\01_tnga_brief" --template "New Briefs Caption" --name tnga_brief --prefix "" --max-tokens 300
```

`--skip-existing` mặc định nên chỉ ảnh chưa có `.txt` được caption; `.txt` ghi cạnh ảnh
qua `save_caption` → `normalize_caption`. Chạy nền, log vào scratchpad. Ước 10–20 phút.

Bước 2: script tạm trong scratchpad gọi `check_briefs_caption` trên 66 file mới, in: số
đạt / tổng theo thư mục, lỗi theo loại, min / median / max số từ, 5 caption mẫu. Danh sách
66 đường dẫn đã tạo ghi vào scratchpad để xoá được nếu không ưng.

## Kiểm thử

- `test_briefs_prompts.py` và `batch_caption.py --self-test` xanh trước mỗi commit.
- `test_video_prompts.py`, `test_h3_prompts.py` vẫn xanh (không đụng, chạy để chắc).
- Kết quả thật: ≥ 90 % caption đạt checker; lỗi còn lại liệt kê trong báo cáo cuối.

## Ngoài phạm vi

- Không tự suy trigger từ tên thư mục trong `batch_caption.py`; một vòng lặp shell là đủ.
- Không thêm REVIEW marker nhiều người / mất mặt như template "Following ChatGPT".
- Không sửa 139 caption cũ khác; không outpaint ảnh mất đầu; không tạo mask.
- Không đổi `DEFAULT_PREFIX` của `batch_caption.py`.
