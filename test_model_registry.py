"""Self-checks for model_registry. Run: .venv\\Scripts\\python.exe test_model_registry.py"""
import struct
import tempfile
from pathlib import Path

from model_registry import apply_to_catalog, scan_models


def _fake_gguf(path: Path, meta: dict):
    """Write a GGUF v3 header with the given keys (str or uint32), then a tokenizer key."""
    def s(text):
        b = text.encode()
        return struct.pack("<Q", len(b)) + b

    def kv(key, val):
        if isinstance(val, int):
            return s(key) + struct.pack("<II", 4, val)
        return s(key) + struct.pack("<I", 8) + s(val)

    body = b"".join(kv(k, v) for k, v in meta.items()) + kv("tokenizer.ggml.model", "gpt2")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, len(meta) + 1) + body)


def model(name, dim, size="8B"):
    return {"general.architecture": "qwen3vl", "general.type": "model", "general.name": name,
            "general.size_label": size, "qwen3vl.embedding_length": dim}


def mmproj(name, dim):
    return {"general.architecture": "clip", "general.type": "mmproj", "general.name": name,
            "clip.projector_type": "qwen3vl_merger", "clip.vision.projection_dim": dim}


def test_filters_and_pairs():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # text-only GGUF next to VL models must be ignored
        _fake_gguf(root / "Qwen3-8B-Q4_K_M.gguf",
                   {"general.architecture": "qwen3", "general.type": "model", "qwen3.embedding_length": 4096})
        # unsloth-style folders sharing the mmproj file name -> paired by name + folder
        for variant in ("Instruct", "Thinking"):
            d = root / f"Qwen3-VL-8B-{variant}"
            _fake_gguf(d / f"Qwen3-VL-8B-{variant}-UD-Q4_K_XL.gguf", model(f"Qwen3-Vl-8B-{variant}", 4096))
            _fake_gguf(d / "mmproj-F16.gguf", mmproj(f"Qwen3-Vl-8B-{variant}", 4096))
        # a fine-tune with an arbitrary name borrows a 4096-dim mmproj, never the 2560 (4B) one
        _fake_gguf(root / "Brand-New-Tune.Q6_K.gguf", model("Brand New", 4096))
        _fake_gguf(root / "mmproj-Qwen3VL-4B-Instruct-F16.gguf", mmproj("Qwen3Vl 4b Instruct", 2560))

        reg = scan_models(root)
        by_file = {Path(m["model_path"]).name: m for m in reg["gguf"]}
        assert "Qwen3-8B-Q4_K_M.gguf" not in by_file and len(by_file) == 3, by_file
        for variant in ("Instruct", "Thinking"):
            m = by_file[f"Qwen3-VL-8B-{variant}-UD-Q4_K_XL.gguf"]
            assert Path(m["mmproj_path"]).parent.name == f"Qwen3-VL-8B-{variant}", m
            assert (m["name"], m["quant"]) == (f"Qwen3-VL-8B-{variant}", "UD-Q4_K_XL"), m
        assert "4B" not in by_file["Brand-New-Tune.Q6_K.gguf"]["mmproj_path"]

        catalog = {"Qwen3-VL-8B-Instruct-GGUF": {"model_files": {"Q8_0 (~9GB)": "Qwen3VL-8B-Instruct-Q8_0.gguf"},
                                                  "mmproj_file": "mmproj-Qwen3VL-8B-Instruct-F16.gguf",
                                                  "min_vram_4gb": False}}
        added = apply_to_catalog(reg, catalog)
        # a new quant of a known model joins it; unknown models become their own entries
        assert sorted(added) == ["Brand-New-Tune-GGUF", "Qwen3-VL-8B-Thinking-GGUF"], added
        known = catalog["Qwen3-VL-8B-Instruct-GGUF"]
        assert len(known["model_files"]) == 2 and Path(known["mmproj_file"]).is_absolute(), known


def test_skips_model_without_matching_mmproj():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _fake_gguf(root / "Other-8B.Q6_K.gguf", model("Other", 4096))
        _fake_gguf(root / "mmproj-Qwen3VL-4B-Instruct-F16.gguf", mmproj("Qwen3Vl 4b Instruct", 2560))
        reg = scan_models(root)
        assert reg["gguf"] == [] and [Path(s["path"]).name for s in reg["skipped"]] == ["Other-8B.Q6_K.gguf"]


if __name__ == "__main__":
    test_filters_and_pairs()
    test_skips_model_without_matching_mmproj()
    print("test_model_registry OK")
