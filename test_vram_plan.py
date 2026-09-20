"""
Test cho logic chọn model theo VRAM. Hàm thuần — KHÔNG cần GPU, không tải model.

Chạy:  D:\\001_Personal_Proj\\Comfy\\.venv\\Scripts\\python.exe test_vram_plan.py
"""
import sys

# Console Windows mặc định là cp1252, không in được tiếng Việt trong thông báo test.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

from utils.hardware import budget_from_free, HEADROOM_RATIO, HEADROOM_FIXED


# ── Task 1: ngân sách phần cứng ──────────────────────────────────────────────

def test_budget_may_tham_chieu():
    """RTX 5070 Ti Laptop: 10.78 GiB trống → 8.90 GiB ngân sách."""
    assert abs(budget_from_free(10.78) - 8.90) < 0.01, budget_from_free(10.78)


def test_budget_khong_bao_gio_am():
    """Card gần đầy phải trả 0.0, không phải số âm."""
    assert budget_from_free(0.5) == 0.0
    assert budget_from_free(0.0) == 0.0


def test_budget_headroom_dung_cong_thuc():
    assert HEADROOM_RATIO == 0.10
    assert HEADROOM_FIXED == 0.8
    assert abs(budget_from_free(24.0) - (24.0 * 0.9 - 0.8)) < 1e-9


# ── Task 2: schema catalog ───────────────────────────────────────────────────

from models_catalog import HF_VL_MODELS, GGUF_VL_MODELS, CATALOG_VERIFIED


def test_catalog_khong_con_truong_cu():
    """min_vram_4gb / vram / VRAM_PROFILES là gốc của giả định 4GB cứng."""
    import models_catalog
    assert not hasattr(models_catalog, "VRAM_PROFILES")
    for name, info in {**HF_VL_MODELS, **GGUF_VL_MODELS}.items():
        assert "min_vram_4gb" not in info, name
        assert "vram" not in info, name


def test_catalog_hf_du_truong():
    need = {"repo_id", "arch", "series", "size", "weights_gib",
            "native_quant", "quants", "kv", "quality", "hf_url"}
    for name, info in HF_VL_MODELS.items():
        assert need <= set(info), f"{name} thiếu {need - set(info)}"
        assert info["repo_id"].startswith("Qwen/"), name
        assert {"layers", "kv_heads", "head_dim"} <= set(info["kv"]), name


def test_catalog_hf_native_quant_la_don_le():
    """Không được chồng bitsandbytes lên checkpoint đã quantize sẵn."""
    for name, info in HF_VL_MODELS.items():
        if info["native_quant"] is not None:
            assert info["quants"] == [info["native_quant"]], name


def test_catalog_gguf_du_truong():
    need = {"repo_id", "series", "size", "kv", "quality",
            "model_files", "mmproj_files", "hf_url"}
    for name, info in GGUF_VL_MODELS.items():
        assert need <= set(info), f"{name} thiếu {need - set(info)}"
        assert info["mmproj_files"], f"{name} không có mmproj"
        for quant, value in info["model_files"].items():
            assert isinstance(value, tuple) and len(value) == 2, f"{name}/{quant}"
            filename, gib = value
            assert filename.endswith(".gguf"), f"{name}/{quant}"
            assert isinstance(gib, float), f"{name}/{quant}"


def test_catalog_khong_co_file_split():
    """File -split-NNNNN-of-NNNNN.gguf không tải được bằng hf_hub_download."""
    for name, info in GGUF_VL_MODELS.items():
        for quant, (filename, _gib) in info["model_files"].items():
            assert "-split-" not in filename, f"{name}/{quant}"


def test_catalog_dung_don_vi_gib():
    """Bắt lỗi trộn bytes / GB thập phân / GiB — mọi số phải trong (0, 70)."""
    for name, info in HF_VL_MODELS.items():
        assert 0.0 < info["weights_gib"] < 70.0, f"{name}={info['weights_gib']}"
    for name, info in GGUF_VL_MODELS.items():
        for quant, (_f, gib) in info["model_files"].items():
            assert 0.0 < gib < 70.0, f"{name}/{quant}={gib}"
        for quant, (_f, gib) in info["mmproj_files"].items():
            assert 0.0 < gib < 5.0, f"{name}/mmproj/{quant}={gib}"


def test_catalog_co_ngay_verify():
    assert CATALOG_VERIFIED["checked"] == "2026-09-20"
    assert CATALOG_VERIFIED["org"] == "Qwen"


# ── Task 3: ước lượng VRAM ───────────────────────────────────────────────────

from utils.vram_plan import (
    kv_cache_gib, estimate_hf, estimate_gguf, derive_runtime,
    OVERHEAD_GIB, MIN_OFFLOAD_FRAC,
)


def test_kv_cache_dung_so_that():
    """Qwen3-VL-8B (36 layer, 8 kv-head, head_dim 128) ở ctx 8192 tốn đúng 1.125 GiB."""
    kv8b = {"layers": 36, "kv_heads": 8, "head_dim": 128}
    assert abs(kv_cache_gib(kv8b, 8192) - 1.125) < 1e-9
    assert abs(kv_cache_gib(kv8b, 4096) - 0.5625) < 1e-9
    kv2b = {"layers": 28, "kv_heads": 8, "head_dim": 128}
    assert abs(kv_cache_gib(kv2b, 2048) - 0.21875) < 1e-9


def test_estimate_gguf_8b_q4_full_offload():
    """8B Q4_K_M + mmproj Q8_0 + ctx 4096 = 6.54 GiB. Catalog cũ bỏ sót mmproj và KV."""
    entry = GGUF_VL_MODELS["Qwen3-VL-8B-Instruct-GGUF"]
    est = estimate_gguf(entry, "Q4_K_M", "Q8_0", 4096, -1)
    assert abs(est - 6.54) < 0.01, est


def test_estimate_gguf_offload_mot_phan_re_hon():
    """gpu_layers < tổng số layer phải cho ước lượng nhỏ hơn full offload."""
    entry = GGUF_VL_MODELS["Qwen3-VL-8B-Instruct-GGUF"]
    full = estimate_gguf(entry, "Q8_0", "Q8_0", 4096, -1)
    part = estimate_gguf(entry, "Q8_0", "Q8_0", 4096, 31)
    assert abs(full - 9.97) < 0.01, full
    assert part < full, (part, full)
    assert abs(part - 8.77) < 0.02, part


def test_estimate_hf_8b_4bit():
    """8B 4-bit, ctx 4096, max_pixels 1280*28*28 = 6.74 GiB."""
    entry = HF_VL_MODELS["Qwen3-VL-8B-Instruct"]
    est = estimate_hf(entry, "4bit", 4096, 1280 * 28 * 28)
    assert abs(est - 6.74) < 0.01, est


def test_estimate_hf_fp8_that_su_khong_vua_may_12gb():
    """Lỗi gốc: catalog cũ ghi 8B-FP8 là 7.5GB nên app mời load rồi OOM."""
    entry = HF_VL_MODELS["Qwen3-VL-8B-Instruct-FP8"]
    est = estimate_hf(entry, "fp8", 4096, 1280 * 28 * 28)
    assert abs(est - 11.37) < 0.01, est
    assert est > 8.90


def test_derive_runtime_theo_ngan_sach():
    assert derive_runtime(8.90) == {
        "n_ctx": 4096, "max_pixels": 1280 * 28 * 28, "mmproj_quant": "Q8_0"}
    assert derive_runtime(2.53) == {
        "n_ctx": 2048, "max_pixels": 512 * 28 * 28, "mmproj_quant": "Q8_0"}
    assert derive_runtime(24.0) == {
        "n_ctx": 8192, "max_pixels": 2560 * 28 * 28, "mmproj_quant": "F16"}


def test_vram_plan_khong_import_torch():
    """
    vram_plan phải thuần — nếu nó kéo torch vào thì test chạy không cần GPU sẽ vỡ.

    Duyệt cây AST thay vì tìm chuỗi, để không khớp nhầm vào docstring/comment.
    """
    import ast
    import inspect
    import utils.vram_plan as vp

    banned = {"torch", "gradio", "huggingface_hub", "psutil", "requests", "urllib"}
    tree = ast.parse(inspect.getsource(vp))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not (imported & banned), imported & banned


# ── Task 4: sinh và xếp hạng cấu hình ────────────────────────────────────────

from utils.vram_plan import PlanOption, plan_options, classify_fit

BUDGET_12GB = 8.90   # RTX 5070 Ti Laptop: 10.78 trống
BUDGET_4GB = 2.53    # card 4GB: ~3.7 trống


def test_classify_fit():
    assert classify_fit(7.0, 8.90) == "comfortable"   # <= 0.80 * 8.90
    assert classify_fit(8.50, 8.90) == "tight"
    assert classify_fit(9.50, 8.90) == "over"
    assert classify_fit(1.0, 0.0) == "over"           # không có GPU


def test_plan_may_12gb_chon_8b_q4():
    """Kỳ vọng cốt lõi: máy 12GB được mời 8B Q4_K_M, không phải 8B FP8."""
    opts = plan_options(BUDGET_12GB, "gguf", GGUF_VL_MODELS)
    top = opts[0]
    assert top.model_name == "Qwen3-VL-8B-Instruct-GGUF", top.model_name
    assert top.quant == "Q4_K_M", top.quant
    assert top.fit == "comfortable", top.fit
    assert abs(top.est_gib - 6.54) < 0.01, top.est_gib


def test_plan_may_12gb_hf_chon_8b_4bit():
    opts = plan_options(BUDGET_12GB, "hf", HF_VL_MODELS, supports_fp8=True)
    top = opts[0]
    assert top.model_name == "Qwen3-VL-8B-Instruct", top.model_name
    assert top.quant == "4bit", top.quant
    assert top.fit == "comfortable", top.fit


def test_plan_may_4gb_chon_2b_q4():
    """Phân khúc 4GB vẫn được phục vụ đúng — chỉ là không còn hard-code."""
    opts = plan_options(BUDGET_4GB, "gguf", GGUF_VL_MODELS)
    top = opts[0]
    assert top.model_name == "Qwen3-VL-2B-Instruct-GGUF", top.model_name
    assert top.quant == "Q4_K_M", top.quant
    for o in opts:
        if "8B" in o.model_name:
            assert o.fit != "comfortable", (o.model_name, o.quant, o.est_gib)


def test_plan_4gb_khong_nhan_offload_qua_thap():
    """4B Q4_K_M trên máy 4GB chỉ chạy được 20/36 layer → phải bị đánh 'over'."""
    opts = plan_options(BUDGET_4GB, "gguf", GGUF_VL_MODELS)
    four_b = [o for o in opts
              if o.model_name == "Qwen3-VL-4B-Instruct-GGUF" and o.quant == "Q4_K_M"]
    assert four_b, "4B Q4_K_M phải xuất hiện trong danh sách, dù là 🔴"
    assert four_b[0].fit == "over", four_b[0].est_gib
    assert four_b[0].gpu_layers == -1, "không đạt sàn offload thì báo full-offload"


def test_plan_12gb_nhan_offload_mot_phan():
    """8B Q8_0 chạy 31/36 layer = 86% > sàn 0.75 → giữ, xếp 'tight'."""
    opts = plan_options(BUDGET_12GB, "gguf", GGUF_VL_MODELS)
    q8 = [o for o in opts
          if o.model_name == "Qwen3-VL-8B-Instruct-GGUF" and o.quant == "Q8_0"][0]
    assert q8.gpu_layers == 31, q8.gpu_layers
    assert q8.fit == "tight", (q8.fit, q8.est_gib)
    assert q8.gpu_layers >= MIN_OFFLOAD_FRAC * 36


def test_plan_gate_fp8_theo_compute_capability():
    """Card không có kernel FP8 thì không được mời model FP8."""
    opts = plan_options(BUDGET_12GB, "hf", HF_VL_MODELS, supports_fp8=False)
    assert not [o for o in opts if o.quant == "fp8"]
    opts_on = plan_options(BUDGET_12GB, "hf", HF_VL_MODELS, supports_fp8=True)
    assert [o for o in opts_on if o.quant == "fp8"]


def test_plan_thu_tu_comfortable_truoc_tight_truoc_over():
    rank = {"comfortable": 0, "tight": 1, "over": 2}
    for backend, catalog in (("gguf", GGUF_VL_MODELS), ("hf", HF_VL_MODELS)):
        opts = plan_options(BUDGET_12GB, backend, catalog)
        seq = [rank[o.fit] for o in opts]
        assert seq == sorted(seq), (backend, seq)


def test_plan_label_sinh_ra_tu_du_lieu():
    opts = plan_options(BUDGET_12GB, "gguf", GGUF_VL_MODELS)
    top = opts[0]
    assert top.label.startswith("🟢")
    assert top.model_name in top.label
    assert top.quant in top.label
    assert f"{top.est_gib:.2f}" in top.label
    assert len({o.label for o in opts}) == len(opts), "label phải là khóa duy nhất"


# ── Test runner ──────────────────────────────────────────────────────────────

def _run_all():
    failed = []
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            failed.append(name)
            print(f"FAIL  {name}: {e}")
        except Exception as e:
            failed.append(name)
            print(f"ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{'FAILED' if failed else 'OK'} — {len(failed)} lỗi")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
