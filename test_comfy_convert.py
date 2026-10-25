"""Self-checks for captioner.comfy_convert. Run: .venv\\Scripts\\python.exe test_comfy_convert.py"""
import json
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from captioner.comfy_convert import convert, hf_key


def _fake_config(repo, dst):
    (Path(dst) / "config.json").write_text("{}", encoding="utf-8")


def _comfy_file(path: Path, with_scale=True, fmt="float8_e4m3fn"):
    blob = torch.tensor(list(json.dumps({"format": fmt}).encode()), dtype=torch.uint8)
    tensors = {
        "model.layers.0.mlp.down_proj.weight": torch.tensor([[1.0, 2.0]]).to(torch.float8_e4m3fn),
        "model.layers.0.mlp.down_proj.comfy_quant": blob,
        "model.embed_tokens.weight": torch.tensor([[0.5, 0.25]], dtype=torch.bfloat16),
        "visual.blocks.0.attn.proj.weight": torch.tensor([[3.0]], dtype=torch.bfloat16),
    }
    if with_scale:
        tensors["model.layers.0.mlp.down_proj.weight_scale"] = torch.tensor(0.5, dtype=torch.float32)
    save_file(tensors, str(path))


def test_hf_key():
    assert hf_key("model.layers.3.self_attn.q_proj.weight") == "model.language_model.layers.3.self_attn.q_proj.weight"
    assert hf_key("model.embed_tokens.weight") == "model.language_model.embed_tokens.weight"
    assert hf_key("model.norm.weight") == "model.language_model.norm.weight"
    assert hf_key("visual.blocks.0.attn.proj.weight") == "model.visual.blocks.0.attn.proj.weight"
    for already in ("model.language_model.layers.0.mlp.up_proj.weight", "model.visual.merger.norm.weight", "lm_head.weight"):
        assert hf_key(already) == already


def test_convert_dequantizes_and_renames():
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "m.safetensors", Path(tmp) / "out" / "My-HF"
        _comfy_file(src)
        assert convert(src, dst, "Qwen/Qwen3-VL-4B-Instruct", fetch=_fake_config, log=lambda m: None) == dst
        index = json.loads((dst / "model.safetensors.index.json").read_text(encoding="utf-8"))
        assert set(index["weight_map"]) == {
            "model.language_model.layers.0.mlp.down_proj.weight",
            "model.language_model.embed_tokens.weight",
            "model.visual.blocks.0.attn.proj.weight",
        }
        weights = {}
        for shard in set(index["weight_map"].values()):
            weights.update(load_file(str(dst / shard)))
        down = weights["model.language_model.layers.0.mlp.down_proj.weight"]
        assert down.dtype == torch.bfloat16 and down.tolist() == [[0.5, 1.0]]
        assert (dst / "config.json").is_file() and not dst.with_name("My-HF.partial").exists()
        # second call is a no-op on an already converted folder
        assert convert(src, dst, "Qwen/Qwen3-VL-4B-Instruct", fetch=_fake_config, log=lambda m: None) == dst


def test_convert_failure_leaves_nothing():
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "m.safetensors", Path(tmp) / "out" / "Bad-HF"
        _comfy_file(src, with_scale=False)
        try:
            convert(src, dst, "Qwen/Qwen3-VL-4B-Instruct", fetch=_fake_config, log=lambda m: None)
        except ValueError:
            pass
        else:
            raise AssertionError("fp8 tensor without a scale must fail")
        assert not dst.exists() and not dst.with_name("Bad-HF.partial").exists()


def test_convert_rejects_unsupported_format():
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "m.safetensors", Path(tmp) / "out" / "Fp4-HF"
        _comfy_file(src, fmt="nvfp4")
        try:
            convert(src, dst, "Qwen/Qwen3-VL-4B-Instruct", fetch=_fake_config, log=lambda m: None)
        except ValueError as e:
            assert "nvfp4" in str(e), str(e)
        else:
            raise AssertionError("nvfp4 comfy_quant must be rejected")
        assert not dst.exists() and not dst.with_name("Fp4-HF.partial").exists()


if __name__ == "__main__":
    test_hf_key()
    test_convert_dequantizes_and_renames()
    test_convert_failure_leaves_nothing()
    test_convert_rejects_unsupported_format()
    print("test_comfy_convert OK")
