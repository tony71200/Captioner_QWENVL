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
