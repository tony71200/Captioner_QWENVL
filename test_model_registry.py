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


import json as _json

import torch
from safetensors.torch import save_file

from model_registry import inspect_path, inspect_safetensors, list_candidates


def _st(path: Path, hidden: int, layers: int, lm_head=False, fmt="float8_e4m3fn", hf_names=False):
    """Tiny Qwen3-VL-shaped safetensors: one fp8 weight per layer plus a comfy_quant blob."""
    prefix = "model.language_model." if hf_names else "model."
    t = {f"{prefix}embed_tokens.weight": torch.zeros(4, hidden, dtype=torch.bfloat16),
         "model.visual.deepstack_merger_list.0.linear_fc1.weight": torch.zeros(2, 2, dtype=torch.bfloat16)}
    blob = torch.tensor(list(_json.dumps({"format": fmt}).encode()), dtype=torch.uint8)
    for i in range(layers):
        t[f"{prefix}layers.{i}.mlp.down_proj.weight"] = torch.zeros(2, 2).to(torch.float8_e4m3fn)
        t[f"{prefix}layers.{i}.mlp.down_proj.weight_scale"] = torch.tensor(1.0)
        t[f"{prefix}layers.{i}.mlp.down_proj.comfy_quant"] = blob.clone()   # save_file rejects shared storage
    if lm_head:
        t["lm_head.weight"] = torch.zeros(4, hidden, dtype=torch.bfloat16)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(t, str(path))


def test_inspect_safetensors_verdicts():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _st(root / "ok-4b_fp8_scaled.safetensors", 2560, 36)
        _st(root / "hf-names-4b.safetensors", 2560, 36, hf_names=True)
        _st(root / "cut-32b.safetensors", 5120, 50)
        _st(root / "no-head-8b.safetensors", 4096, 36)
        _st(root / "nvfp4-4b.safetensors", 2560, 36, fmt="nvfp4")
        save_file({"model.layers.0.mlp.down_proj.weight": torch.zeros(2, 2)}, str(root / "text-only.safetensors"))

        ok = inspect_safetensors(root / "ok-4b_fp8_scaled.safetensors")
        assert ok["usable"] and ok["base_repo"] == "Qwen/Qwen3-VL-4B-Instruct" and ok["size"] == "4B", ok
        assert inspect_safetensors(root / "hf-names-4b.safetensors")["usable"]
        cut = inspect_safetensors(root / "cut-32b.safetensors")
        assert not cut["usable"] and "50 of 64 layers" in cut["reason"], cut
        assert "no lm_head" in inspect_safetensors(root / "no-head-8b.safetensors")["reason"]
        assert "nvfp4" in inspect_safetensors(root / "nvfp4-4b.safetensors")["reason"]
        assert inspect_safetensors(root / "text-only.safetensors") is None


def test_list_candidates_and_inspect_path():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _fake_gguf(root / "A" / "Qwen3-VL-8B-A.Q4_K_M.gguf", model("A", 4096))
        _fake_gguf(root / "A" / "mmproj-A-F16.gguf", mmproj("A", 4096))
        _fake_gguf(root / "Qwen3-8B-Q4_K_M.gguf",
                   {"general.architecture": "qwen3", "general.type": "model", "qwen3.embedding_length": 4096})
        _st(root / "ok-4b_fp8_scaled.safetensors", 2560, 36)
        _st(root / "HuggingFace" / "Done-HF" / "model-00001.safetensors", 2560, 36)
        (root / "HuggingFace" / "Done-HF" / "config.json").write_text("{}", encoding="utf-8")

        names = sorted(c["name"] for c in list_candidates(root, set()))
        assert names == ["Qwen3-VL-8B-A.Q4_K_M.gguf", "ok-4b_fp8_scaled.safetensors"], names
        gguf = next(c for c in list_candidates(root, set()) if c["kind"] == "gguf")
        assert gguf["usable"] and gguf["quant"] == "Q4_K_M" and gguf["mmproj"].endswith("mmproj-A-F16.gguf")

        known = {str((root / "A" / "Qwen3-VL-8B-A.Q4_K_M.gguf").resolve())}
        assert [c["kind"] for c in list_candidates(root, known)] == ["safetensors"]

        assert not inspect_path(root / "missing.gguf", root)["usable"]
        assert "not a Qwen3-VL" in inspect_path(root / "Qwen3-8B-Q4_K_M.gguf", root)["reason"]
        assert inspect_path(root / "A" / "Qwen3-VL-8B-A.Q4_K_M.gguf", root)["usable"]


if __name__ == "__main__":
    test_filters_and_pairs()
    test_skips_model_without_matching_mmproj()
    test_inspect_safetensors_verdicts()
    test_list_candidates_and_inspect_path()
    print("test_model_registry OK")
