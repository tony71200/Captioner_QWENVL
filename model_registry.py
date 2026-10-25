"""
Discover Qwen3-VL models under the models root and write models_registry.json.

GGUF        Filtered by header metadata, not file names: a model needs
            general.architecture == "qwen3vl"; an mmproj needs
            clip.projector_type == "qwen3vl_merger". They pair when the mmproj
            projection_dim equals the model embedding_length (2560 = 4B, 4096 = 8B...),
            preferring the same model name, then the same folder, then F16.
            Text-only GGUFs sharing the folder (Qwen3, flux2 encoders...) are ignored.
safetensors HF folders (config.json model_type qwen3_vl*) and single-file
            checkpoints with Qwen3-VL "deepstack" tensors. Listed for reference
            only: this app runs Qwen3-VL through the GGUF backend.

Adding a model: drop the .gguf and its mmproj .gguf anywhere under the root
(same sub-folder is safest) and restart the app. It appears in the model list.

Standalone: .venv\\Scripts\\python.exe model_registry.py [ROOT]
"""
import json
import logging
import math
import re
import struct
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_LLM_DIR = Path(r"D:\001_Personal_Proj\Comfy\ComfyUI\models\text_encoders")
REGISTRY_PATH = Path(__file__).with_name("models_registry.json")

VL_ARCHS = {"qwen3vl", "qwen3vlmoe"}
SAFETENSORS_REASON = (
    "Not loadable here: the HuggingFace backend only runs Qwen2.5-VL. "
    "Use a GGUF build of this model instead."
)
GGUF_DEFAULTS = {
    "context_length": 8192,
    "image_max_tokens": 4096,
    "n_batch": 512,
    "gpu_layers": -1,
    "top_k": 0,
    "pool_size": 4194304,
}
# hidden size -> (base repo, text layers, tied word embeddings, size label), from the official config.json files
QWEN3VL_BY_HIDDEN = {
    2048: ("Qwen/Qwen3-VL-2B-Instruct", 28, True, "2B"),
    2560: ("Qwen/Qwen3-VL-4B-Instruct", 36, True, "4B"),
    4096: ("Qwen/Qwen3-VL-8B-Instruct", 36, False, "8B"),
    5120: ("Qwen/Qwen3-VL-32B-Instruct", 64, False, "32B"),
}
SUPPORTED_COMFY_QUANT = {"float8_e4m3fn"}
_LAYER_RE = re.compile(r"^model\.(?:language_model\.)?layers\.(\d+)\.")
_SCALE_SUFFIXES = (".comfy_quant", ".weight_scale", ".weight_scale_2", ".pre_quant_scale", ".input_scale")

_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
           7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_QUANT_RE = re.compile(r"(?i)[-_.]((?:UD-)?(?:I?Q\d[A-Z0-9_]*)|BF16|F16|F32)$")


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def read_gguf_meta(path) -> dict:
    """Return the general.* and <architecture>.* metadata of a GGUF file
    ({} if it is not GGUF v2+). Stops before the large tokenizer arrays."""
    out = {}
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF" or struct.unpack("<I", f.read(4))[0] < 2:
            return out
        _, n_kv = struct.unpack("<QQ", f.read(16))

        def string():
            n, = struct.unpack("<Q", f.read(8))
            return f.read(n).decode("utf-8", "replace")

        def value(t):
            if t == 8:
                return string()
            if t == 9:
                elem_type, n = struct.unpack("<IQ", f.read(12))
                return [value(elem_type) for _ in range(n)]
            fmt = _SCALAR[t]
            return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]

        for _ in range(n_kv):
            key = string()
            arch = out.get("general.architecture", "")
            if not (key.startswith("general.") or (arch and key.startswith(arch + "."))):
                break
            out[key] = value(struct.unpack("<I", f.read(4))[0])
    return out


def read_safetensors_header(path):
    """(header dict incl. __metadata__, byte offset where tensor data starts)."""
    with open(path, "rb") as f:
        n, = struct.unpack("<Q", f.read(8))
        if n > 100_000_000:
            raise ValueError(f"safetensors header too large: {n} bytes")
        return json.loads(f.read(n)), 8 + n


def _comfy_formats(path, header: dict, data_start: int) -> set:
    formats = set()
    with open(path, "rb") as f:
        for key, info in header.items():
            if key.endswith(".comfy_quant"):
                start, end = info["data_offsets"]
                f.seek(data_start + start)
                formats.add(json.loads(f.read(end - start)).get("format", "?"))
    return formats


def inspect_safetensors(path):
    """Static verdict for a single-file safetensors; None when it is not Qwen3-VL."""
    path = Path(path)
    header, data_start = read_safetensors_header(path)
    header.pop("__metadata__", None)
    if not any("deepstack" in k for k in header):
        return None
    embed = next((v["shape"] for k, v in header.items() if k.endswith("embed_tokens.weight")), [0, 0])
    hidden = embed[1]
    layers = len({m.group(1) for k in header if (m := _LAYER_RE.match(k))})
    formats = _comfy_formats(path, header, data_start)
    params = sum(math.prod(v["shape"]) for k, v in header.items() if not k.endswith(_SCALE_SUFFIXES))
    verdict = {"kind": "safetensors", "path": str(path), "name": path.name, "hidden": hidden,
               "layers": layers, "formats": sorted(formats), "bf16_gb": round(params * 2 / 1e9, 1),
               "usable": False, "reason": ""}
    ref = QWEN3VL_BY_HIDDEN.get(hidden)
    if ref is None:
        verdict["reason"] = f"unknown Qwen3-VL size (hidden {hidden})"
        return verdict
    base_repo, n_layers, tied, size = ref
    verdict.update(base_repo=base_repo, size=size)
    unsupported = formats - SUPPORTED_COMFY_QUANT
    if layers != n_layers:
        verdict["reason"] = f"not usable: {layers} of {n_layers} layers (cut text encoder)"
    elif not tied and not any(k.startswith("lm_head.") for k in header):
        verdict["reason"] = "not usable: no lm_head (text encoder only)"
    elif unsupported:
        verdict["reason"] = f"not usable: unsupported quant {', '.join(sorted(unsupported))}"
    else:
        verdict.update(usable=True, reason=f"safetensors, converts once to bf16 (~{verdict['bf16_gb']} GB)")
    return verdict


def inspect_gguf(path, mmprojs):
    """Static verdict for a GGUF file; None when it is not a Qwen3-VL model (text-only, mmproj...)."""
    path = Path(path)
    meta = read_gguf_meta(path)
    if meta.get("general.architecture") not in VL_ARCHS:
        return None
    dim = meta.get(f"{meta['general.architecture']}.embedding_length")
    best = _pick_mmproj(path, meta, mmprojs)
    quant = _QUANT_RE.search(path.stem)
    return {
        "kind": "gguf", "path": str(path), "name": path.name,
        "quant": quant.group(1).upper() if quant else "unknown",
        "size": meta.get("general.size_label", ""),
        "size_gb": round(path.stat().st_size / 1e9, 1),
        "mmproj": str(best) if best else "",
        "mmproj_options": sorted({str(p) for p, m in mmprojs if dim and m.get("clip.vision.projection_dim") == dim}),
        "usable": best is not None,
        "reason": f"GGUF, {dim}-dim mmproj found" if best else f"no mmproj with projection_dim {dim} found",
    }


def inspect_path(path, root) -> dict:
    """Verdict for a pasted path. mmprojs are looked up under root and next to the file."""
    path = Path(path)
    unusable = {"kind": "?", "path": str(path), "name": path.name, "usable": False}
    if not path.is_file():
        return dict(unusable, reason=f"file not found: {path}")
    verdict = None
    try:
        if path.suffix.lower() == ".gguf":
            mmprojs = _collect_gguf(root)[1] + _collect_gguf(path.parent)[1]
            verdict = inspect_gguf(path, mmprojs)
        elif path.suffix.lower() == ".safetensors":
            verdict = inspect_safetensors(path)
    except (OSError, ValueError, KeyError, struct.error) as e:
        return dict(unusable, reason=f"unreadable file: {e}")
    return verdict or dict(unusable, reason="not a Qwen3-VL model (text-only, mmproj or another architecture)")


def list_candidates(root, known: set) -> list:
    """Qwen3-VL model files under root not in `known` (resolved path strings), each with a verdict.
    Text-only models, mmprojs and shards of converted HF folders are left out."""
    root = Path(root)
    out = []
    models, mmprojs, _ = _collect_gguf(root)
    for p, _meta in models:
        if str(p.resolve()) not in known:
            out.append(inspect_gguf(p, mmprojs))
    for p in sorted(root.rglob("*.safetensors")):
        if (p.parent / "config.json").exists() or str(p.resolve()) in known:
            continue
        try:
            verdict = inspect_safetensors(p)
        except (OSError, ValueError, KeyError, struct.error):
            verdict = None
        if verdict:
            out.append(verdict)
    return out


def _pick_mmproj(model: Path, meta: dict, mmprojs: list):
    """Only mmprojs whose projection_dim fits the model's embedding_length qualify."""
    dim = meta.get(f"{meta['general.architecture']}.embedding_length")
    name = meta.get("general.name")
    candidates = [(p, m) for p, m in mmprojs if dim and m.get("clip.vision.projection_dim") == dim]
    if not candidates:
        return None
    candidates.sort(key=lambda item: (
        not name or item[1].get("general.name") != name,   # same model name first
        item[0].parent != model.parent,                     # then the same folder
        "f16" not in item[0].name.lower(),                  # then full-precision mmproj
    ))
    return candidates[0][0]


def _collect_gguf(root):
    """Qwen3-VL model GGUFs and qwen3vl mmprojs under root, plus unreadable files."""
    models, mmprojs, skipped = [], [], []
    for p in sorted(Path(root).rglob("*.gguf")):
        try:
            meta = read_gguf_meta(p)
        except (OSError, struct.error, KeyError) as e:
            skipped.append({"path": str(p), "reason": f"unreadable GGUF header: {e}"})
            continue
        if str(meta.get("clip.projector_type", "")).startswith("qwen3vl"):
            mmprojs.append((p, meta))
        elif meta.get("general.architecture") in VL_ARCHS:
            models.append((p, meta))
    return models, mmprojs, skipped


def scan_models(root) -> dict:
    root = Path(root)
    registry = {"root": str(root), "scanned_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "gguf": [], "mmproj": [], "safetensors": [], "skipped": []}
    if not root.exists():
        registry["skipped"].append({"path": str(root), "reason": "models root does not exist"})
        return registry

    models, mmprojs, skipped = _collect_gguf(root)
    registry["skipped"] += skipped

    registry["mmproj"] = [str(p) for p, _ in mmprojs]
    for p, meta in models:
        mmproj = _pick_mmproj(p, meta, mmprojs)
        if mmproj is None:
            registry["skipped"].append({"path": str(p), "reason": "no matching mmproj .gguf found"})
            continue
        quant = _QUANT_RE.search(p.stem)
        registry["gguf"].append({
            "name": _QUANT_RE.sub("", p.stem),
            "arch": meta["general.architecture"],
            "size": meta.get("general.size_label", ""),
            "quant": quant.group(1).upper() if quant else "unknown",
            "model_path": str(p),
            "mmproj_path": str(mmproj),
            "size_gb": round((p.stat().st_size + mmproj.stat().st_size) / 1e9, 1),
        })

    for cfg in sorted(root.rglob("config.json")):
        try:
            model_type = json.loads(cfg.read_text(encoding="utf-8")).get("model_type", "")
        except (OSError, ValueError):
            continue
        if model_type.startswith("qwen3_vl"):
            registry["safetensors"].append({"name": cfg.parent.name, "format": "hf-folder",
                                            "path": str(cfg.parent), "loadable": False,
                                            "reason": SAFETENSORS_REASON})
    for p in sorted(root.rglob("*.safetensors")):
        if (p.parent / "config.json").exists():
            continue   # shard of an HF folder, already listed above
        try:
            verdict = inspect_safetensors(p)
        except (OSError, struct.error, ValueError, KeyError):
            verdict = None
        if verdict:
            registry["safetensors"].append({"name": p.stem, "format": "single-file", "path": str(p),
                                            "loadable": verdict["usable"], "reason": verdict["reason"]})
    return registry


def refresh_registry(root, path: Path = REGISTRY_PATH) -> dict:
    registry = scan_models(root)
    try:
        path.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError as e:
        logger.warning("Could not write %s: %s", path, e)
    logger.info("Model scan of %s: %d Qwen3-VL GGUF, %d safetensors (listed only), %d skipped -> %s",
                root, len(registry["gguf"]), len(registry["safetensors"]), len(registry["skipped"]), path.name)
    return registry


def apply_to_catalog(registry: dict, catalog: dict) -> list:
    """Make every discovered GGUF selectable in the app.

    A file already named in the catalog keeps its entry. A new quant of a known
    model joins that entry. Anything else becomes a new "<name>-GGUF" entry.
    Local entries store absolute paths, so identical file names in different
    folders (e.g. several "mmproj-F16.gguf") never get mixed up.
    Returns the names of entries added.
    """
    local_mmproj = {Path(p).name.lower() for p in registry["mmproj"]}
    by_file = {Path(f).name.lower(): key
               for key, info in catalog.items() for f in info.get("model_files", {}).values()}
    by_norm = {_norm(re.sub(r"(?i)-gguf$", "", key)): key for key in catalog}
    added = []
    for m in registry["gguf"]:
        key = by_file.get(Path(m["model_path"]).name.lower())
        if key is None:
            key = by_norm.get(_norm(m["name"])) or f"{m['name']}-GGUF"
            if key not in catalog:
                catalog[key] = {
                    "repo_id": "", "series": "Qwen3-VL", "size": m["size"],
                    "mmproj_file": m["mmproj_path"], "gguf_defaults": dict(GGUF_DEFAULTS),
                    "model_files": {}, "min_vram_4gb": False, "hf_url": "",
                    "description": f"Local {m['arch']} model, auto-discovered in "
                                   f"{Path(m['model_path']).parent.name}",
                }
                by_norm[_norm(m["name"])] = key
                added.append(key)
            catalog[key]["model_files"][f"{m['quant']} (local, ~{m['size_gb']}GB)"] = m["model_path"]
            catalog[key]["min_vram_4gb"] = catalog[key]["min_vram_4gb"] or m["size_gb"] <= 4.0
        info = catalog[key]
        if Path(info.get("mmproj_file", "")).name.lower() not in local_mmproj:
            info["mmproj_file"] = m["mmproj_path"]   # catalog's preferred mmproj is not on disk
    return added


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    reg = refresh_registry(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_LLM_DIR)
    for m in reg["gguf"]:
        print(f"GGUF  {m['quant']:<8} {m['model_path']}\n      mmproj {m['mmproj_path']}")
    for s in reg["safetensors"]:
        print(f"ST    {s['format']:<12} {s['path']}  (listed only)")
    for s in reg["skipped"]:
        print(f"SKIP  {s['path']}: {s['reason']}")
