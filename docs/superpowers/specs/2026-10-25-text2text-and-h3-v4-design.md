# Text2Text cho Krea 2 / FLUX.2 Klein và tab H3 hai tab con theo định dạng Runner V5.3

Ngày: 2026-10-25. Nhánh: `20261025_model_registry_text_encoders`.
Mockup tab H3 đã duyệt trong hội thoại ngày 2026-10-25. Sơ đồ app hiện tại: `docs/architecture/app-architecture.html`.

## 1. Mục tiêu

1. **Text2Text**: tab có icon; chọn model đích Krea 2 hoặc FLUX.2 Klein để viết prompt đúng cấu trúc của model đó; ô NSFW thêm yếu tố nam gợi cảm.
2. **Tab Prompt Video MiniMax H3** chia hai tab con, cả hai xuất prompt **định dạng v4** mà ComfyUI AutoShot Runner V5.3 đọc được:
   - **2a**: ảnh + element (ảnh phụ) + gợi ý ý tưởng + prompt, như hiện tại, thêm bảng kế hoạch shot và chuyển cảnh giữa hai shot.
   - **2b**: ảnh + video tham chiếu + element phụ → prompt v4 chép hành động và góc máy của video.

## 2. Những gì đã kiểm tra trước khi thiết kế

| Câu hỏi | Kết quả |
|---|---|
| Định dạng Runner V5.3 | GLOBAL (trước `[SHOT 01`) được chép nguyên vào mọi shot. Header `[SHOT NN \| DURATION: Ns \| REF_VIDEO: … \| REF_IMAGES: 1,2]`, thân shot có `Outfit in this shot:`, `Camera:`, dòng timecode. Một khối `overall_soundscape:` + `non_diegetic_music:` sau shot cuối. Dấu ngoặc kép trong GLOBAL hoặc khối âm thanh chung làm Runner báo lỗi. Shot nhắc `<Picture k>` không có trong `REF_IMAGES` thì Runner báo lỗi ở PREVIEW. Nguồn: `ComfyUI-MiniMax-AutoShot-Runner/__init__.py`, `GPTs_MiniMax_H3_Video_Shots_v4/01_INSTRUCTIONS.md`. |
| Thời lượng | Mỗi shot 4.4–12 s, chỉ dùng các giá trị bảng §9 của knowledge v4: 4.4, 5, 5.8, 6.5, 7.2, 8, 8.7, 9.4, 10, 10.8, 11.5, 12. |
| Chuyển cảnh | 3 nhãn: CONTINUE, OUTFIT CUT, SCENE CUT (§7). `continuity_guide` là một công tắc cho cả job. |
| REF_VIDEO 22 frame | Đã render thành công 23–24/10. Lỗi duy nhất (24/10) do canvas có 4 LoadVideo, đã sửa. Chưa chạy lại sau bản sửa. Docs khuyên `REF_VIDEO: none` mặc định. |
| Qwen3-VL đọc video | Không tự tìm điểm cắt (chia đều thời gian). GGUF không nhận video, tối đa 7 frame/lần ở n_ctx 8192. HF cần torchcodec hoặc tự tách frame. Code phát hiện cắt (PyAV, frame difference, quy tắc "đỉnh cô lập") bắt đúng 5/5 điểm cắt trên video thử, 0 điểm sai trên video strobe. |
| Krea 2 | Một đoạn văn tiếng Anh, chủ thể trước. Thứ tự: chủ thể + hành động → bối cảnh → khung hình/góc máy → ánh sáng → chất liệu → mood/màu. Không quality tag, không trọng số. Turbo bỏ qua negative. |
| FLUX.2 Klein | Một đoạn văn, chủ thể trước. Thứ tự: chủ thể (chi tiết cụ thể) → hành động → môi trường → ánh sáng → style/mood → máy ảnh/ống kính. 40–120 từ, không quality tag, không `(word:1.2)`. Bản distilled bỏ qua negative. |

## 3. Quyết định đã chốt với người dùng

| # | Câu hỏi | Chốt |
|---|---|---|
| 1 | Định dạng H3 | Cả 2a và 2b dùng v4. Định dạng cũ (CHARACTER & SCENE DEFINITION / SHOT STRUCTURE) bị bỏ. |
| 2 | Chuyển cảnh | Model gợi ý nhãn và tư thế nối cho từng mối nối, người dùng sửa trong bảng trước khi viết prompt. |
| 3 | Vai trò video (2b) | Chép hành động + góc máy; giữ nhạc gốc (ghi chú `final_audio`). Clip REF_VIDEO 22f là tùy chọn nâng cao, **tắt mặc định**, và trong bản này chỉ ghi gợi ý vào ô cài đặt Runner, không cắt clip. |
| 4 | Element | Vai trò: Trang phục (garment), Face sheet, Đạo cụ (prop). **Không có element bối cảnh**: bối cảnh lấy từ mô tả trong ảnh chính (Picture 1). |
| 5 | Text2Text model | Dropdown Krea 2 / FLUX.2 Klein. |
| 6 | Negative prompt | Giữ nguyên dòng negative trong chế độ file như hiện tại. |
| 7 | NSFW Text2Text | Gợi cảm, không lộ: thân hình săn chắc, ngực trần, mồ hôi, đồ lót/đồ bơi, ánh mắt và tư thế quyến rũ, ánh sáng ấm. Không lộ bộ phận sinh dục, không hành vi tình dục. Luôn là người trưởng thành. |
| 8 | Sơ đồ | Archify không có trên máy; dùng skill `diagram-design`, xuất HTML. |

## 4. Text2Text

### 4.1 Giao diện

- Tên tab: `📝 text2Text`.
- Thêm cạnh ô Max New Tokens: dropdown **Target model** (`Krea 2`, `FLUX.2 Klein`, mặc định `Krea 2`) và checkbox **NSFW (male, sensual)**. Hai điều khiển dùng chung cho cả cột dán tag và cột file.
- Mọi thứ khác giữ nguyên, kể cả cách tách khối và giữ dòng negative.

### 4.2 Prompt

Module thuần mới `captioner/t2t_prompts.py` (không import gradio/torch):

- `T2T_MODELS = ("Krea 2", "FLUX.2 Klein")`
- `t2t_system(model, nsfw) -> str`, `t2t_user(tags, model, nsfw) -> str`.
- Chung cho cả hai model: một đoạn tiếng Anh 50–100 từ, bắt đầu bằng chủ thể, câu tự nhiên, không danh sách tag, không markdown, không quality tag (masterpiece, best quality, 8k, score tags), không cú pháp trọng số, không nhắc negative. Chỉ dùng chi tiết có trong tag; riêng ánh sáng được thêm một nguồn sáng hợp lý nếu tag không có.
- Thứ tự Krea 2: subject + action → setting/spatial layout → framing and camera angle → light (source, direction, quality) → materials/textures → mood/palette.
- Thứ tự FLUX.2 Klein: subject with concrete details → action/pose → environment (foreground/background) → lighting (source, quality, direction, colour temperature) → style/mood → camera/lens/composition.
- NSFW bật: thêm khối quy tắc "adult man, sensual but not explicit" theo quyết định 7; không mâu thuẫn trang phục tag đã ghi; nếu tag không ghi trang phục thì được chọn ngực trần hoặc đồ lót/đồ bơi; không dùng từ gợi tuổi nhỏ.

## 5. Định dạng v4 dùng chung (module `captioner/h3_v4.py`, thuần)

### 5.1 Thời lượng và timecode

- `DURATIONS = (4.4, 5, 5.8, 6.5, 7.2, 8, 8.7, 9.4, 10, 10.8, 11.5, 12)`; `snap_duration(x)` chọn giá trị gần nhất (hòa thì lấy lớn hơn).
- `ts(seconds)`: `00:05`, `00:04.4` (1 chữ số thập phân khi lẻ).

### 5.2 Ảnh tham chiếu

- Picture 1 luôn là nhân vật; đồng thời là nguồn bối cảnh, ánh sáng và trang phục đầu tiên.
- Ảnh phụ (tối đa 3, thành Picture 2–4 theo thứ tự ô có ảnh): vai trò `outfit`, `face`, `prop`.
- `describe_references(refs)` viết danh sách cho model theo v4 (không còn `ref_image_N`).
- Workflow V4 có 4 ô ảnh, khớp tối đa 4 picture.

### 5.3 Kế hoạch shot (bảng sửa được)

Mỗi dòng: `Shot`, `Duration` (giá trị bảng §5.1), `Source` (đoạn story hoặc segment video), `REF_IMAGES` (vd `1,2`), `Into next` (CONTINUE / OUTFIT CUT / SCENE CUT, trống ở shot cuối), `Seam pose` (tư thế cuối shot này để nối), `Content` (một dòng nội dung).
`plan_to_rows` / `rows_to_plan` chuyển qua lại với `gr.Dataframe` (kiểu `array`), chuẩn hóa nhãn, điền mặc định (`REF_IMAGES` = `1`, nhãn lạ → `SCENE CUT` kèm cảnh báo).

### 5.4 Ô cài đặt Runner

`runner_settings(rows, final_audio)` → văn bản tiếng Việt hiển thị riêng (KHÔNG nằm trong prompt, vì Runner chép mọi chữ trước `[SHOT 01` vào từng shot):
- `split_mode`: SHOT TAGS; số ảnh cần nối vào LoadImage #187, #188, #206, #207 theo thứ tự Picture 1–4.
- `continuity_guide`: ON nếu mọi mối nối là CONTINUE; OFF nếu không có CONTINUE; có cả hai thì OFF kèm nhắc "viết khớp tư thế đầu/cuối hoặc tách job".
- `final_audio`: tên file video (2b, khi bật giữ nhạc gốc).
- `REF_VIDEO`: none ở mọi shot; nếu bật tùy chọn nâng cao thì liệt kê segment dài ≥ 0.92 s (22 frame ở 24 fps) có thể cắt clip bằng `video_probe.py --clip`, kèm nhắc làm mờ đầu.

### 5.5 Kiểm tra và sửa đầu ra (`check_v4`)

Đầu vào: văn bản model, số picture, bảng kế hoạch. Trả về văn bản đã sửa và danh sách cảnh báo (log):
1. Bỏ lời dẫn trước `subject_definitions:`, code fence, `**`.
2. Không có `[SHOT` → lỗi rõ ràng, giữ văn bản thô cho người dùng.
3. Đánh số lại SHOT 01..NN. Header chỉ còn `DURATION`, `REF_VIDEO`, `REF_IMAGES`. Trường khác (vd `PRIMARY CAMERA SETUP`, `CAMERA`) chuyển thành dòng `Camera:` đầu thân shot nếu thân chưa có.
4. `DURATION`: dùng thời lượng của bảng kế hoạch khi số shot khớp; ngược lại snap về bảng §5.1. `REF_VIDEO` thiếu → `none`. `REF_IMAGES`: số ngoài 1..N bị bỏ; thiếu → lấy từ bảng hoặc `1`.
5. `<Picture k>` nhắc trong thân shot mà không có trong `REF_IMAGES` → thêm k vào `REF_IMAGES` (nếu k ≤ N), cảnh báo.
6. Timecode trong shot: dời về 00:00, nối liền, bỏ dòng bắt đầu sau DURATION, dòng cuối kết thúc đúng DURATION. Hỗ trợ giây lẻ.
7. Khối `overall_soundscape:` / `non_diegetic_music:` trong GLOBAL hoặc trong từng shot được gom về một khối sau shot cuối (giữ khối cuối cùng tìm thấy; thiếu thì mặc định `Natural ambient sound. No dialogue.` / `NONE`).
8. Dấu ngoặc kép trong GLOBAL bị xóa (Runner coi là lời thoại và báo lỗi), cảnh báo.
9. Cảnh báo (không sửa) khi thiếu dòng vai trò `<Picture k> is …` cho một picture.

## 6. Tab con 2a: ảnh + element → gợi ý → prompt

Luồng: ảnh (Picture 1 + tối đa 3 element) → Character bible (như hiện tại) → Story (gõ tay, hoặc Gợi ý 3 ý tưởng → Viết chi tiết, như hiện tại) → **3) Lập kế hoạch shot & chuyển cảnh** → bảng sửa được → **4) Viết prompt v4** → ô kết quả + ô cài đặt Runner + Lưu/Copy.

- Thời lượng shot: từ timecode trong Story (quy tắc 12 s hiện có), nếu không có thì từ Tổng thời lượng, rồi snap về bảng §5.1. Không có cả hai → model chọn, code snap.
- Bước lập kế hoạch: một lần gọi model mới `plan_{sfw,nsfw}` (chỉ văn bản), nhận bible, danh sách ảnh, story, các shot và thời lượng; trả mỗi shot một dòng `SHOT 01 | REF_IMAGES: 1 | NEXT: CONTINUE | END: <pose> | CONTENT: <…>`. Code parse vào bảng; dòng thiếu dùng mặc định.
- Bước viết prompt: `shots_{sfw,nsfw}` viết lại theo hợp đồng v4 (§2, `01_INSTRUCTIONS.md`), nhận bible, ảnh, story và bảng kế hoạch (kèm nhãn nối + tư thế nối). Quy tắc nối: CONTINUE → shot sau mở đầu `Shot continues from the final frame of the previous shot:` + cùng tư thế; OUTFIT CUT → `Outfit in this shot: only … from <Picture k>, from the first frame`; SCENE CUT → dòng `Camera:` mới và nhịp mở cảnh. Shot trước mối nối kết thúc bằng tư thế giữ yên, người thấy rõ.
- Hậu xử lý `check_v4`, lưu `<stem ảnh 1>_H3_<sfw|nsfw>.txt` cạnh Picture 1 (như hiện tại).

## 7. Tab con 2b: ảnh + video + element → prompt

Luồng: Picture 1 + video tham chiếu + tối đa 3 element → **✂ Phân tích video** → bảng segment (sửa được cột Shot) → **Lập kế hoạch** → bảng kế hoạch → **Viết prompt v4** → kết quả + cài đặt Runner + Lưu/Copy.

### 7.1 Phân tích video (module `captioner/video_segments.py`, numpy + PyAV)

1. Đọc video bằng PyAV, giữ mỗi frame ở 64×36 xám.
2. Điểm cắt: chênh lệch trung bình tuyệt đối giữa hai frame liền nhau `d[i]`; cắt khi `d[i] > 20` **và** `d[i] > 3 × max(d trong ±8 frame, trừ i)`; hai điểm cắt cách nhau ≥ 6 frame.
3. Segment giữa các điểm cắt; mỗi segment có `bg_diff` so với segment trước (trung vị frame, vùng viền trên/trái/phải) và mã setup (S2 khi `bg_diff ≥ 30`), như `video_probe.py`.
4. Segment dài > 12 s được chia đều thành các phần ≤ 12 s (nối nhau bằng CONTINUE).
5. Gom segment thành shot: cộng dồn theo thứ tự; tách shot mới khi cộng thêm vượt 12 s, hoặc khi đổi setup mà shot hiện tại đã ≥ 4.4 s; shot cuối < 4.4 s được gộp vào shot trước nếu tổng ≤ 12 s. Thời lượng shot = snap(tổng giây); độ lệch so với video được ghi log (quan trọng khi giữ nhạc gốc).
6. Mỗi segment lấy 3 frame (đầu, giữa, cuối; JPEG rộng 512 px) và gọi model một lần với bước mới `segment` (trung tính, không chia sfw/nsfw): trả 4 dòng `CAMERA:`, `ACTION:`, `OUTFIT:`, `LOCATION:`. Model không quyết định thời gian; thời gian luôn từ code.
7. Bảng segment: `#`, `Time`, `Seconds`, `Setup`, `Camera`, `Action`, `Outfit`, `Shot` (sửa được để gộp/tách khác).

### 7.2 Lập kế hoạch và viết

- Bảng kế hoạch dựng từ cột `Shot` của bảng segment; `Source` liệt kê segment và thời điểm trong shot (vd `seg 2 at 00:01.7`) để model viết dòng `At 00:01.7, hard cut to …` cho shot montage.
- Bước `plan` và `shots` giống 2a; "story" là danh sách segment đã mô tả. Prompt nhấn mạnh: không chép người, chữ, biển hiệu, sticker trong video; nhân vật từ Picture 1; bối cảnh từ Picture 1.
- Giữ nhạc gốc bật → ô cài đặt Runner ghi `final_audio = <tên file video>`, khối âm thanh ngắn.

## 8. Lưu trữ prompt

`captioner/system_prompt.json`:
- `bible`: cập nhật vai trò ảnh (outfit/face/prop; bối cảnh từ Picture 1).
- `shots_sfw`, `shots_nsfw`: viết lại theo hợp đồng v4.
- Thêm `plan_sfw`, `plan_nsfw`, `segment`.
- `H3_KEYS` cập nhật; `step_key` coi `bible` và `segment` là bước trung tính.
- Bản NSFW giữ khối giới hạn hiện có (chỉ người trưởng thành, đồng thuận, nhân vật hư cấu).

Các hàm định dạng cũ (`split_h3_output`, `check_part1`, `check_shots`, `build_h3_file`, `PART_TITLES`, …) và test của chúng bị xóa.

## 9. Giao diện (theo mockup đã duyệt)

- Trong tab `🎞️ Prompt Video Minimax H3`: `gr.Tabs` với `2a · Ảnh + element → gợi ý → prompt` và `2b · Ảnh + video + element → prompt`.
- Khối ảnh (Picture 1 + 3 element có dropdown vai trò Trang phục / Face sheet / Đạo cụ, ô đường dẫn, nút 📂) dựng bằng một hàm dùng chung, mỗi tab con có bộ riêng.
- Bảng kế hoạch là `gr.Dataframe` sửa được. Cột `Into next` là ô văn bản; giá trị hợp lệ ghi ngay dưới bảng; giá trị sai được chuẩn hóa và báo trong log.
- Kết quả là một ô văn bản (nút copy) + ô Markdown "Cài đặt Runner" + Lưu.
- Logger và các ⚙ system prompt giữ như hiện tại cho mỗi bước.

## 10. Xử lý lỗi

| Tình huống | Hành vi |
|---|---|
| Chưa nạp model / thiếu Picture 1 / thiếu video (2b) | Badge cảnh báo, không gọi model. |
| Video không đọc được (PyAV lỗi) | Badge lỗi kèm thông báo PyAV. |
| Video không có điểm cắt | Một segment; vẫn gom shot theo quy tắc độ dài. |
| Model trả dòng segment thiếu trường | Trường thiếu để trống, log cảnh báo. |
| Model trả prompt không có `[SHOT` | Hiển thị văn bản thô, badge cảnh báo, không lưu. |
| Bảng kế hoạch bị sửa sai (thời lượng, nhãn, REF_IMAGES) | Chuẩn hóa (snap, nhãn mặc định, bỏ số ảnh không tồn tại), log từng sửa. |

## 11. Kiểm thử

- `test_t2t_prompts.py`: thứ tự slot từng model có trong prompt, NSFW thêm/không thêm khối, không có chữ "masterpiece" trong yêu cầu đầu ra (chỉ trong quy tắc loại bỏ).
- `test_h3_v4.py`: `snap_duration`, `ts`, bảng kế hoạch qua lại, `runner_settings` (ON/OFF/mixed, final_audio), `check_v4` cho từng quy tắc §5.5 bằng văn bản mẫu.
- `test_video_segments.py`: `detect_cuts` trên mảng tổng hợp (đỉnh cô lập, chuyển động liên tục không cắt, strobe đều không cắt), `group_shots` (gộp ngắn, tách > 12 s, đổi setup), video tổng hợp bằng PyAV (3 đoạn màu) cho điểm cắt đúng frame.
- `test_h3_prompts.py`: cập nhật theo store mới (khóa, `step_key`), bỏ test định dạng cũ.
- `test_h3_app.py`: viết lại phần H3 với `FakeCaptioner`: 2a plan → write → save; 2b analyse (video tổng hợp) → plan → write; thứ tự gọi model và số ảnh gửi.
- Text2Text app: `convert_tags_to_caption` truyền đúng system/user theo model và NSFW (FakeCaptioner).
- Kiểm tra thật trên GPU + trình duyệt: 2a với ảnh thật, 2b với video `czx126_….mp4` (5 điểm cắt đã biết), dán prompt vào Runner PREVIEW SHOTS nếu ComfyUI đang chạy; Text2Text với một khối tag thật cho mỗi model.

## 12. Ngoài phạm vi

- Cắt clip REF_VIDEO 22f và làm mờ đầu trong app (chỉ gợi ý trong ô cài đặt Runner).
- Gửi prompt trực tiếp vào ComfyUI.
- Nhiều nhân vật (`<Subject 2>`).
- Tách job tự động khi trộn CONTINUE và cắt (chỉ nhắc trong ô cài đặt).

## 13. Thứ tự triển khai

1. Text2Text (module prompt + UI).
2. `h3_v4.py` + `system_prompt.json` v4, bỏ định dạng cũ.
3. `video_segments.py`.
4. Handler 2a, handler 2b.
5. Giao diện hai tab con.
6. Kiểm tra thật (GPU + trình duyệt), README.
