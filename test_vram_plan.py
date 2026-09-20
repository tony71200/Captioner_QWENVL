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
