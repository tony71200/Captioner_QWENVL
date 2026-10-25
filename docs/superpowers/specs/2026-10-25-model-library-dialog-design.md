# Thư viện model, dialog "Add or test model" và backend safetensors

Ngày: 2026-10-25. Nhánh: `20261025_model_registry_text_encoders`.
Mockup giao diện: xem bản phác trong hội thoại ngày 2026-10-25 (Setup tab + dialog), mô tả lại ở mục 5.

## 1. Mục tiêu

1. Setup tab có nút **Add or test model** mở một dialog. Trong dialog, người dùng chọn một model
   encoder (GGUF hoặc safetensors) trong `models\text_encoders`, hoặc dán đường dẫn file ở chỗ khác.
   Dialog **kiểm tra nhanh** model có chạy được với chương trình không, đo VRAM thật, rồi mới cho
   **thêm vào danh sách**. Danh sách và mức VRAM đo được lưu vào file, mở lại app vẫn còn.
2. Mỗi lần khởi động, app **kiểm tra từng model trong danh sách còn tồn tại không**. Việc này cần
   khi người khác clone repo về máy khác: model thiếu được báo rõ, và tải lại được nếu có nguồn.
3. Kiểm tra và sửa đường chạy **safetensors (HuggingFace)**, vì tới nay app chỉ chạy GGUF.

## 2. Hiện trạng đã kiểm tra

### 2.1 Backend HuggingFace đang hỏng hoàn toàn

| Vấn đề | Bằng chứng |
|--------|-----------|
| torch trong `.venv` là bản CPU | `torch 2.14.0`, `torch.version.cuda = None`. GPU là RTX 5070 Ti Laptop 12 GB, compute 12.0 (Blackwell). |
| torchvision sai gói | `torchvision 0.2.0` (bản 2018), thiếu `torchvision.io`. |
| Không import nổi processor | `from transformers import AutoProcessor` → `ModuleNotFoundError ... torchvision.io`. Cả Qwen2.5-VL cũng không chạy được. |
| Loader chặn Qwen3-VL | `captioner/hf_captioner.py` chỉ dùng `Qwen2_5_VLForConditionalGeneration` và raise lỗi nếu tên có "Qwen3-VL". |

Các gói khác đã đủ: `transformers 5.17.0` (có Qwen3-VL), `accelerate 1.15.0`, `bitsandbytes 0.50.2`,
`qwen-vl-utils 0.0.14`, `safetensors 0.8.0`, `huggingface_hub 1.32.0`.

### 2.2 Các file safetensors trong `text_encoders`

Tất cả là checkpoint **single-file kiểu ComfyUI**: không có `config.json`, tokenizer hay processor,
không có `lm_head`, trọng số đã lượng tử hoá theo định dạng riêng của ComfyUI
(khoá `<layer>.comfy_quant` dạng U8 chứa JSON, `weight_scale` dạng F32).

| File | Cỡ | Layer | Định dạng | Kết luận |
|------|----|-------|-----------|----------|
| `Huihui-Qwen3-VL-4B-Instruct-abliterated-fp8_scaled` | 5.2 GB | 36/36 | fp8 e4m3 + scale theo tensor; vision bf16; khoá kiểu HF (`model.language_model.*`) | **Dùng được** sau khi chuyển đổi |
| `qwen3-vl-4b-heretic_fp8_e4m3fn` | 4.8 GB | 36/36 | fp8 e4m3 + scale theo tensor, cả vision cũng fp8; khoá kiểu ComfyUI (`model.layers.*`) | **Dùng được** sau khi chuyển đổi |
| `qwen3vl_4b_fp8_scaled` | 5.2 GB | 36/36 | như trên, vision bf16 | **Dùng được** sau khi chuyển đổi |
| `qwen3vl_32b_heretic_minimax_h3_nvfp4` | 15.7 GB | **50/64** | nvfp4 + int8 | **Không dùng được** |
| `qwen3vl_32b_minimax_h3_nvfp4_awq` | 15.7 GB | **50/64** | nvfp4 AWQ | **Không dùng được** |

Lý do 4B dùng được dù thiếu `lm_head`: config chính thức `Qwen/Qwen3-VL-4B-Instruct` có
`tie_word_embeddings: true`, nên `lm_head` chính là `embed_tokens`.
Lý do 32B không dùng được: bản gốc có 64 layer và `tie_word_embeddings: false`. File chỉ giữ
50 layer đầu và không có `lm_head`, tức là text encoder đã cắt cho mô hình sinh ảnh, không sinh chữ được.

### 2.3 GGUF mới

`GGUF\Qwen3-VL-8B-NSFW-Caption-V4.Q4_K_M.gguf` (kiến trúc `qwen3vl`, 36 layer, dim 4096).
Thư mục không có mmproj riêng của V4. Bộ quét hiện ghép nó với `NSFW-Caption-V4.5.mmproj-Q8_0`
vì cùng dim 4096. Ghép đúng về kích thước, nhưng chất lượng ảnh chỉ biết sau khi chạy thử.
Đây chính là trường hợp quick test cần bắt.

## 3. Quyết định (mặc định đề xuất, cần xác nhận)

| # | Câu hỏi | Đề xuất | Lý do |
|---|---------|---------|-------|
| 1 | Danh sách lưu ở đâu | `models_library.json` ở gốc repo, **commit vào git** | Người khác clone về là có danh sách và biết model nào cần tải. |
| 2 | Đường dẫn trong danh sách | **Tương đối với models root** (`--llm-dir`). File nằm ngoài root thì lưu đường dẫn tuyệt đối và đánh dấu "không di động". | Máy khác có ổ đĩa khác vẫn dùng được. |
| 3 | Dropdown hiển thị gì | Model trong thư viện + model chính thức trong `models_catalog.py` (tải được). **Bỏ việc tự thêm** mọi file quét được. File quét được chỉ hiện trong dialog dưới dạng ứng viên. | Người dùng muốn danh sách gồm model đã test. Tự thêm làm lộ model chưa kiểm. |
| 4 | Bốn GGUF đang chạy tốt | Khi triển khai, chạy quick test cho từng cái rồi commit vào thư viện ban đầu. | Không mất model đang dùng khi bỏ cơ chế tự thêm. |
| 5 | Safetensors single-file chạy kiểu gì | **Chuyển đổi một lần** sang thư mục HF chuẩn: giải lượng tử fp8 → bf16, lấy config/tokenizer/processor từ repo gốc. Sau đó nạp bằng `from_pretrained` như mọi model HF. | Dùng lại nguyên đường bitsandbytes 4/8-bit của các VRAM profile. Tốn đĩa (~8.9 GB cho 4B) nhưng code đơn giản nhất. |
| 6 | Sửa môi trường | Cài **torch + torchvision bản CUDA cho Blackwell** (cu128 trở lên) vào `.venv`. Cần người dùng đồng ý vì tải vài GB. | Không có bước này thì backend HF không chạy được dù sửa code. |
| 7 | Quick test chạy ở đâu | **Trong cùng tiến trình**, dùng đúng `GGUFCaptioner`/`HFCaptioner` của app. Model đang nạp sẽ bị unload trước, dialog báo trước điều này. | "Phù hợp với chương trình" phải được chứng minh bằng chính code của chương trình. |
| 8 | Ảnh test | Ảnh tổng hợp sinh lúc chạy: nền xanh dương, hình tròn đỏ, hình vuông vàng. | Không cần asset có bản quyền, và chấm điểm thị giác tự động được. |
| 9 | Hỗ trợ nvfp4 / 32B | **Không làm.** Dialog liệt kê và ghi lý do loại. | File 32B thiếu layer và lm_head, không có cách nào sinh caption. |

## 4. Kiến trúc

Mỗi đơn vị một việc, giao tiếp qua dict/dataclass đơn giản.

| Đơn vị | Việc | Phụ thuộc |
|--------|------|-----------|
| `model_registry.py` (có sẵn, mở rộng) | Quét root, đọc header GGUF/safetensors. Thêm `inspect_candidate(path) -> Verdict` cho cả safetensors: kiến trúc, số layer so với config gốc, `lm_head`/tied, định dạng lượng tử. Bỏ `apply_to_catalog` và file `models_registry.json`. | struct, json |
| `model_library.py` (mới) | Đọc/ghi `models_library.json`, giải đường dẫn tương đối, `validate()` trả trạng thái từng entry: `found` / `missing` / `downloadable`. | model_registry |
| `captioner/comfy_convert.py` (mới) | Chuyển single-file ComfyUI → thư mục HF: đổi tên khoá, giải lượng tử fp8 × scale → bf16, ghi shard 2 GB + index, tải file nhỏ của repo gốc. | safetensors, torch, huggingface_hub |
| `model_tester.py` (mới) | Quick test: unload model hiện tại, đo VRAM nền, nạp, caption ảnh tổng hợp, lấy mẫu VRAM, unload, trả `TestResult`. | captioner/*, utils/system_info |
| `captioner/hf_captioner.py` (sửa) | Dùng `AutoModelForImageTextToText` + `AutoProcessor`, bỏ chặn Qwen3-VL, giữ logic bitsandbytes theo profile. | transformers |
| `app.py` (sửa) | Banner model thiếu, nút + dialog, dropdown đọc từ thư viện, dùng VRAM đo được khi xét độ phù hợp. | các đơn vị trên |

### 4.1 Schema `models_library.json`

```json
{
  "version": 1,
  "models": [
    {
      "name": "Qwen3-VL-8B-NSFW-Caption-V4-GGUF",
      "backend": "gguf",
      "model": "GGUF/Qwen3-VL-8B-NSFW-Caption-V4.Q4_K_M.gguf",
      "mmproj": "GGUF/Qwen3-VL-8B-NSFW-Caption-V4.5.mmproj-Q8_0.gguf",
      "quant": "Q4_K_M",
      "size_gb": 5.8,
      "source": null,
      "test": {
        "date": "2026-10-25",
        "gpu": "NVIDIA GeForce RTX 5070 Ti Laptop GPU, 12 GB",
        "load_s": 14.0,
        "caption_s": 2.1,
        "vision_score": 4,
        "peak_vram_gb": {"*": 6.4}
      }
    },
    {
      "name": "Qwen3-VL-4B-Heretic-HF",
      "backend": "hf",
      "model": "HuggingFace/Qwen3-VL-4B-Heretic",
      "origin": "qwen3-vl-4b-heretic_fp8_e4m3fn.safetensors",
      "base_repo": "Qwen/Qwen3-VL-4B-Instruct",
      "quant": "bf16",
      "size_gb": 8.9,
      "source": null,
      "test": {"peak_vram_gb": {"LowVRAM (6–8GB)": 3.4, "NormalVRAM (12–16GB)": 5.6}}
    }
  ]
}
```

- `source` (tuỳ chọn): `{"repo_id": "...", "files": ["..."]}` để tải lại khi thiếu.
  Với model HF chuyển đổi từ single-file, `source` trỏ tới file gốc (`origin`), và app chuyển đổi lại khi thiếu thư mục HF.
- `peak_vram_gb`: GGUF dùng khoá `"*"` vì `gguf_defaults.gpu_layers = -1` nạp toàn bộ lên GPU ở mọi profile.
  HF lưu theo từng profile đã test, vì profile đổi mức lượng tử bitsandbytes.

### 4.2 Khởi động

1. `model_library.load()` đọc file. Không có file thì tạo danh sách rỗng.
2. `validate()` kiểm tra từng entry, ghi log mỗi model thiếu.
3. Setup tab hiện banner vàng nếu có model thiếu, nêu tên và việc app sẽ làm: tải khi Load nếu có `source`,
   không thì hướng dẫn mở dialog để thêm lại.
4. Dropdown: model thấy được đứng trước, rồi model tải được, rồi model thiếu (chọn được nhưng Load sẽ báo lỗi rõ ràng).

### 4.3 Ký hiệu độ phù hợp trong dropdown

Giữ bộ ký hiệu đang có, thêm một ký hiệu cho model thiếu. Model đã test hiện VRAM đo được
trong nhãn biến thể, ví dụ `[Q4_K_M · tested 6.4 GB]`.

| Ký hiệu | Nghĩa |
|---------|-------|
| 🟢 | Có trên máy, VRAM (đo được, hoặc ước tính nếu chưa test) ≤ ngân sách profile |
| 🟡 | Vừa ngân sách nhưng chưa có trên máy, Load sẽ tải hoặc chuyển đổi lại |
| 🔴 | Có trên máy nhưng vượt ngân sách profile |
| ❌ | Model trong thư viện bị thiếu file và không có nguồn để khôi phục |

Ngân sách profile chuyển vào `VRAM_PROFILES[...]["budget_gb"]` (4 / 8 / 16 / 24 GB, giá trị như hàm
`_available_vram_for_profile` hiện có) để app và quick test dùng chung.

Bảng tra cấu trúc Qwen3-VL (từ `config.json` chính thức), dùng cho kiểm tra tĩnh:

| hidden size | Repo gốc | Số layer | tied embeddings |
|-------------|----------|----------|-----------------|
| 2048 | `Qwen/Qwen3-VL-2B-Instruct` | 28 | có |
| 2560 | `Qwen/Qwen3-VL-4B-Instruct` | 36 | có |
| 4096 | `Qwen/Qwen3-VL-8B-Instruct` | 36 | không |
| 5120 | `Qwen/Qwen3-VL-32B-Instruct` | 64 | không |

## 5. Dialog "Add or test model"

Dialog là một `gr.Group` ẩn, bật/tắt bằng `gr.State`, giống pattern nút ⚙ ở Video Reel.
Gradio 6.28 không có modal sẵn, nên không thêm thư viện ngoài.

Thành phần, từ trên xuống:

1. **Ứng viên tìm thấy** trong root nhưng chưa có trong thư viện (`gr.Radio`), mỗi dòng kèm kết luận
   kiểm tra tĩnh: "GGUF · ready to test", "safetensors · converts once (8.9 GB)", hoặc lý do loại
   như "not usable: 50 of 64 layers, no lm_head". Ứng viên bị loại vẫn hiện nhưng không test được.
2. **Hoặc dán đường dẫn** file ở chỗ khác (`gr.Textbox`).
3. **mmproj** (chỉ GGUF): dropdown các mmproj có `projection_dim` khớp, chọn sẵn cặp tự ghép.
4. **Base repo** (chỉ safetensors): suy từ hidden size (2048 → 2B, 2560 → 4B, 4096 → 8B), sửa được.
5. **Tên trong danh sách** và **nguồn tải** (tuỳ chọn).
6. **Profile để test**: mặc định là profile đang chọn ở Setup.
7. Nút **Check** (đọc header, vài giây) và **Quick test on GPU** (nạp thật, khoảng 30 giây; safetensors chuyển đổi trước, vài phút).
8. **Kết quả**: thời gian nạp, thời gian caption, VRAM đỉnh, điểm thị giác, caption mẫu, và 4 chip profile ✓/✕.
9. **Add to library** chỉ bật khi quick test đạt. **Cancel** đóng dialog.
10. **Thư viện hiện tại**: danh sách entry với trạng thái found/missing và nút xoá entry
    (chỉ xoá khỏi danh sách, không xoá file).

## 6. Quick test

| Bước | Chi tiết |
|------|---------|
| Chuẩn bị | Unload model đang nạp. Đọc VRAM nền bằng `nvidia-smi` (hàm sẵn có trong `utils/system_info.py`). |
| Chuyển đổi (safetensors) | Nếu thư mục HF đích chưa có. Bỏ qua nếu đã có. |
| Nạp | `GGUFCaptioner.load_model` hoặc `HFCaptioner.load_model` với profile test, `device="cuda"`. Nếu GGUF tự lùi về CPU (`runtime_device == "cpu-fallback"`) thì coi là không vừa GPU. |
| Sinh | Ảnh tổng hợp 512×512, prompt "Describe the shapes and colors in this image in one sentence.", tối đa 60 token. |
| Đo VRAM | Thread lấy mẫu `nvidia-smi` mỗi 0.25 s từ lúc nạp đến lúc sinh xong. Đỉnh trừ nền. |
| Dọn | Unload, `gc.collect()`, `torch.cuda.empty_cache()` nếu có torch CUDA. |

**Đạt** khi: nạp không lỗi, caption không rỗng, tỉ lệ chữ cái ≥ 60% (chặn chuỗi rác),
điểm thị giác ≥ 3/5 (đếm các từ red, circle, yellow, square, blue), và không hết VRAM.
**Không đạt** thì hiện lỗi cụ thể, ví dụ "out of memory at 11.8 GB" hoặc "caption is garbage, mmproj may not match".

## 7. Chuyển đổi safetensors ComfyUI → HF

1. Đọc header, xác định base repo theo hidden size, tải `config.json`, tokenizer, `preprocessor_config.json`,
   `chat_template` từ base repo (chỉ file nhỏ, `allow_patterns`).
2. Từ chối nếu số layer khác config gốc, hoặc thiếu `lm_head` mà config không tied, hoặc gặp định dạng
   `comfy_quant` khác `float8_e4m3fn` (nvfp4, int8).
3. Đổi tên khoá kiểu ComfyUI sang kiểu HF: `model.layers.` → `model.language_model.layers.`,
   `model.embed_tokens.` → `model.language_model.embed_tokens.`, `model.norm.` → `model.language_model.norm.`,
   `visual.` → `model.visual.`. File đã đúng kiểu HF (Huihui) giữ nguyên.
4. Với mỗi tensor fp8 có `weight_scale`: `bf16 = weight.float() * scale`. Bỏ khoá `comfy_quant` và `weight_scale`.
5. Ghi từng shard 2 GB bằng `safe_open` đọc lần lượt, để RAM không vượt khoảng 3 GB. Ghi `model.safetensors.index.json`.
6. Đích: `<root>/HuggingFace/<name>/`. Dừng giữa chừng thì xoá thư mục tạm, không để lại thư mục hỏng.

## 8. Sửa backend HF

1. Cài torch + torchvision bản CUDA (cu128 trở lên) đúng cặp phiên bản. Ghi lệnh cài và index URL vào
   `requirements.txt`/README. Xác nhận: `torch.cuda.is_available()` là True, `import torchvision.io` chạy.
2. `HFCaptioner` dùng `AutoModelForImageTextToText.from_pretrained` + `AutoProcessor`. Bỏ đoạn chặn Qwen3-VL.
3. `_hf_model_supported_by_loader` trả True cho Qwen3-VL và Qwen2.5-VL.
4. Kiểm chứng bằng caption thật một ảnh với bản 4B chuyển đổi, ở profile LowVRAM (4-bit) và NormalVRAM (8-bit).

## 9. Xử lý lỗi

| Tình huống | Hành vi |
|-----------|---------|
| `models_library.json` hỏng JSON | Log lỗi, đổi tên thành `.bak`, chạy với danh sách rỗng, banner báo. |
| Model thiếu lúc Load | Có `source` thì tải/chuyển đổi lại. Không có thì báo "missing: <path>, open Add or test model". |
| Quick test hết VRAM | Bắt lỗi, unload, kết quả "out of memory at X GB", không thêm vào danh sách. |
| Đĩa không đủ khi chuyển đổi | Kiểm tra dung lượng trống trước, báo cần bao nhiêu GB. |
| Không có `nvidia-smi` | Vẫn test, VRAM ghi "not measured", dropdown dùng ước tính. |

## 10. Kiểm thử

- `test_model_library.py`: đọc/ghi, đường dẫn tương đối, `validate()` với file thiếu, JSON hỏng → `.bak`.
- `test_model_registry.py` (mở rộng): `inspect_candidate` cho safetensors giả: 4B tied hợp lệ, 32B cắt layer bị loại, nvfp4 bị loại.
- `test_comfy_convert.py`: tensor fp8 nhỏ + scale → bf16 đúng giá trị; đổi tên khoá; ghi shard và index.
- `model_tester` kiểm bằng chạy thật: quick test 4 GGUF có sẵn, GGUF V4 mới, và 3 safetensors 4B.
- Kiểm tra giao diện trong browser: mở dialog, test, thêm vào, khởi động lại app thấy entry còn; đổi tên một file để xem banner thiếu.

## 11. Ngoài phạm vi

- Hỗ trợ nvfp4, int8 tensorwise, model 32B.
- Giữ fp8 trên GPU (không giải lượng tử).
- Nhiều GPU, chỉnh sửa định nghĩa VRAM profile.
- Tự động tải model catalog chính thức để test.

## 12. Thứ tự triển khai

1. **A. Môi trường + backend HF** (mục 8, phần 1–3) và bộ chuyển đổi (mục 7). Kiểm chứng bằng caption thật bản 4B.
2. **B. Thư viện + kiểm tra khởi động** (mục 4.1–4.3), seed 4 GGUF có sẵn.
3. **C. Quick test** (mục 6) chạy được từ dòng lệnh.
4. **D. Dialog** (mục 5) nối mọi thứ, kiểm tra trong browser.
