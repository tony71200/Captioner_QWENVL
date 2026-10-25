"""
Convert a ComfyUI single-file Qwen3-VL checkpoint into a HuggingFace model folder.

ComfyUI text-encoder files hold fp8 weights with one float32 scale per tensor
(<layer>.weight_scale) plus a <layer>.comfy_quant JSON blob, use ComfyUI key
names, and ship no config, tokenizer or processor. convert() writes a normal
HF folder: bf16 shards + index, with the small files taken from the base repo.
"""
import json
import logging
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

logger = logging.getLogger(__name__)

SMALL_FILES = (
    "config.json", "generation_config.json", "chat_template.json", "tokenizer.json",
    "tokenizer_config.json", "vocab.json", "merges.txt", "preprocessor_config.json",
    "video_preprocessor_config.json",
)
_RENAMES = (
    ("model.layers.", "model.language_model.layers."),
    ("model.embed_tokens.", "model.language_model.embed_tokens."),
    ("model.norm.", "model.language_model.norm."),
    ("visual.", "model.visual."),
)
_DROP_SUFFIXES = (".comfy_quant", ".weight_scale")


def hf_key(key: str) -> str:
    """Map a ComfyUI tensor name to the transformers Qwen3-VL name. HF-style names pass through."""
    if key.startswith(("model.language_model.", "model.visual.", "lm_head.")):
        return key
    for old, new in _RENAMES:
        if key.startswith(old):
            return new + key[len(old):]
    return key


def to_bf16(tensor, scale=None):
    if tensor.dtype == torch.float8_e4m3fn:
        if scale is None:
            raise ValueError("fp8 tensor without a weight_scale")
        return (tensor.to(torch.float32) * scale.to(torch.float32)).to(torch.bfloat16)
    return tensor.to(torch.bfloat16) if tensor.is_floating_point() else tensor


def fetch_small_files(base_repo: str, dst: Path) -> None:
    """Config, tokenizer and processor files from the base repo (a few MB, cached by huggingface_hub)."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    for name in SMALL_FILES:
        try:
            hf_hub_download(base_repo, name, local_dir=str(dst))
        except EntryNotFoundError:
            pass


def convert(src, dst, base_repo: str, shard_gb: float = 2.0, log=logger.info, fetch=fetch_small_files) -> Path:
    src, dst = Path(src), Path(dst)
    if (dst / "config.json").is_file():
        log(f"{dst.name} is already converted")
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    need = src.stat().st_size * 2          # fp8 -> bf16 roughly doubles the size
    free = shutil.disk_usage(dst.parent).free
    if free < need:
        raise OSError(f"converting needs ~{need / 1e9:.1f} GB free in {dst.parent}, only {free / 1e9:.1f} GB left")

    tmp = dst.with_name(dst.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        fetch(base_repo, tmp)
        weight_map, shard = {}, {}
        state = {"bytes": 0, "total": 0, "index": 0}

        def flush():
            if not shard:
                return
            state["index"] += 1
            fname = f"model-{state['index']:05d}.safetensors"
            save_file(shard, str(tmp / fname), metadata={"format": "pt"})
            weight_map.update({k: fname for k in shard})
            log(f"wrote {fname} ({state['bytes'] / 1e9:.1f} GB)")
            shard.clear()
            state["bytes"] = 0

        with safe_open(str(src), framework="pt") as f:
            keys = set(f.keys())
            for key in sorted(keys):
                if key.endswith(_DROP_SUFFIXES):
                    continue
                scale_key = key[: -len(".weight")] + ".weight_scale" if key.endswith(".weight") else ""
                scale = f.get_tensor(scale_key) if scale_key in keys else None
                tensor = to_bf16(f.get_tensor(key), scale).contiguous()
                shard[hf_key(key)] = tensor
                size = tensor.numel() * tensor.element_size()
                state["bytes"] += size
                state["total"] += size
                if state["bytes"] >= shard_gb * 1e9:
                    flush()
        flush()
        (tmp / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": state["total"]}, "weight_map": weight_map}, indent=2),
            encoding="utf-8",
        )
        tmp.replace(dst)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    log(f"converted {src.name} -> {dst}")
    return dst
