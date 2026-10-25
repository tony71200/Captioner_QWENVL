"""
The model library: models_library.json lists the models that passed a quick test.

Each entry is one model file (GGUF + its mmproj) or one HuggingFace folder,
with paths stored relative to the models root so the list works on another
machine. At startup the app checks every entry and registers it into the
catalog dicts (GGUF_VL_MODELS / HF_VL_MODELS) that the dropdown already reads.
"""
import copy
import json
import logging
import re
from pathlib import Path

from model_registry import _QUANT_RE

logger = logging.getLogger(__name__)

LIBRARY_PATH = Path(__file__).with_name("models_library.json")
GGUF_DEFAULTS = {
    "context_length": 8192,
    "image_max_tokens": 4096,
    "n_batch": 512,
    "gpu_layers": -1,
    "top_k": 0,
    "pool_size": 4194304,
}
# the same settings spelled the way GGUFCaptioner.load_model takes them as kwargs
GGUF_LOAD_KWARGS = {"gpu_layers": -1, "n_ctx": 8192, "n_batch": 512, "image_max_tokens": 4096,
                    "top_k": 0, "pool_size": 4194304}
_HF_QUANT_KEY = {"4bit": "4bit", "8bit": "8bit", "none": "full"}
_DTYPE_SUFFIX_RE = re.compile(r"(?i)[-_.](fp8\w*|nvfp4\w*|bf16|fp16)$")


def load(path: Path = LIBRARY_PATH):
    """(entries, warning). A broken file is moved to .json.bak so the app still starts."""
    if not path.exists():
        return [], None
    try:
        return list(json.loads(path.read_text(encoding="utf-8"))["models"]), None
    except (ValueError, KeyError, TypeError) as e:
        bak = path.with_suffix(".json.bak")
        path.replace(bak)
        return [], f"{path.name} could not be read ({e}). It was moved to {bak.name} and the library starts empty."


def save(entries, path: Path = LIBRARY_PATH) -> None:
    path.write_text(json.dumps({"version": 1, "models": entries}, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def to_stored(path, root) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return str(path)


def abs_path(stored: str, root) -> Path:
    p = Path(stored)
    return p if p.is_absolute() else Path(root) / p


def status(entry, root) -> str:
    model = abs_path(entry["model"], root)
    if entry["backend"] == "gguf":
        present = model.is_file() and abs_path(entry["mmproj"], root).is_file()
    else:
        present = (model / "config.json").is_file()
    if present:
        return "found"
    origin = entry.get("origin")
    if entry.get("source") or (origin and abs_path(origin, root).is_file()):
        return "downloadable"
    return "missing"


def find(entries, model_key):
    return next((e for e in entries if e["model"] == model_key), None)


def upsert(entries, entry) -> list:
    return [e for e in entries if e["model"] != entry["model"]] + [entry]


def remove(entries, model_key) -> list:
    return [e for e in entries if e["model"] != model_key]


def known_paths(entries, root) -> set:
    """Resolved source files already in the library (GGUF model, or the safetensors an HF entry came from)."""
    out = set()
    for e in entries:
        source = e.get("origin") if e["backend"] == "hf" else e["model"]
        if source:
            out.add(str(abs_path(source, root).resolve()))
    return out


def suggest_name(model_path, kind: str, gguf_catalog: dict) -> str:
    p = Path(model_path)
    if kind == "gguf":
        for key, info in gguf_catalog.items():
            if any(Path(f).name.lower() == p.name.lower() for f in info.get("model_files", {}).values()):
                return key
        return f"{_QUANT_RE.sub('', p.stem)}-GGUF"
    return f"{_DTYPE_SUFFIX_RE.sub('', p.stem)}-HF"


def new_entry(candidate, name, root, mmproj=None, base_repo=None, source_repo="") -> dict:
    if candidate["kind"] == "gguf":
        mm = Path(mmproj or candidate["mmproj"])
        source = ({"repo_id": source_repo, "files": [Path(candidate["path"]).name, mm.name]}
                  if source_repo else None)
        return {"name": name, "backend": "gguf", "model": to_stored(candidate["path"], root),
                "mmproj": to_stored(mm, root), "quant": candidate["quant"], "size": candidate.get("size", ""),
                "size_gb": candidate["size_gb"], "source": source}
    return {"name": name, "backend": "hf", "model": f"HuggingFace/{name}",
            "origin": to_stored(candidate["path"], root), "base_repo": base_repo or candidate["base_repo"],
            "quant": "bf16", "size": candidate.get("size", ""), "size_gb": candidate["bf16_gb"], "source": None}


def record(entry, result) -> dict:
    """Entry with the quick-test result merged in. GGUF VRAM applies to every profile ("*")."""
    test = dict(entry.get("test") or {})
    peaks = dict(test.get("peak_vram_gb") or {})
    if result.get("peak_vram_gb") is not None:
        peaks["*" if entry["backend"] == "gguf" else result["profile"]] = result["peak_vram_gb"]
    test.update({k: result[k] for k in ("date", "gpu", "load_s", "caption_s", "vision_score")})
    test["peak_vram_gb"] = peaks
    return dict(entry, test=test)


def _gguf_label(entry) -> str:
    peak = ((entry.get("test") or {}).get("peak_vram_gb") or {}).get("*")
    gb = peak if peak is not None else entry.get("size_gb", 0)
    return f"{entry['quant']} ({'tested' if peak is not None else 'untested'}, ~{gb}GB)"


def _hf_vram(entry) -> dict:
    from models_catalog import VRAM_PROFILES

    size = entry.get("size_gb", 0)
    # ponytail: rough bf16 / int8 / nf4 ratios, replaced by measured values once a profile is tested
    vram = {"full": size + 1.0, "8bit": size * 0.55 + 1.0, "4bit": size * 0.35 + 1.0}
    for profile, gb in ((entry.get("test") or {}).get("peak_vram_gb") or {}).items():
        if profile in VRAM_PROFILES:
            vram[_HF_QUANT_KEY[VRAM_PROFILES[profile]["hf_quant"]]] = gb
    return {k: round(v, 1) for k, v in vram.items()}


def apply_to_catalogs(entries, root, gguf_catalog, hf_catalog) -> dict:
    """Register entries into the catalog dicts the app reads. Returns {entry['model']: status}."""
    statuses = {}
    for e in entries:
        statuses[e["model"]] = status(e, root)
        model = str(abs_path(e["model"], root))
        repo_id = (e.get("source") or {}).get("repo_id", "")
        if e["backend"] == "gguf":
            info = gguf_catalog.setdefault(e["name"], {
                "repo_id": repo_id, "series": "Qwen3-VL", "size": e.get("size", ""),
                "gguf_defaults": dict(GGUF_DEFAULTS), "model_files": {}, "min_vram_4gb": False,
                "hf_url": "", "description": f"Library model ({e['quant']})",
            })
            name = Path(model).name.lower()
            info["model_files"] = {l: f for l, f in info["model_files"].items() if Path(f).name.lower() != name}
            info["model_files"][_gguf_label(e)] = model
            info["mmproj_file"] = str(abs_path(e["mmproj"], root))
            peak = ((e.get("test") or {}).get("peak_vram_gb") or {}).get("*", e.get("size_gb", 99))
            info["min_vram_4gb"] = info.get("min_vram_4gb", False) or peak <= 4.0
        else:
            origin = Path(e.get("origin", "")).name
            info = hf_catalog.setdefault(e["name"], {
                "repo_id": repo_id, "series": "Qwen3-VL", "size": e.get("size", ""), "quantized": False,
                "min_vram_4gb": False, "hf_url": "",
                "description": f"Library model converted from {origin}" if origin else "Library model",
            })
            info["local_dir"] = model
            info["vram"] = _hf_vram(e)
            info["min_vram_4gb"] = info["vram"]["4bit"] <= 4.0
        info["library"] = True
    return statuses


def unregister(entry, remaining, root, gguf_catalog, hf_catalog, pristine_gguf, pristine_hf) -> None:
    """Undo apply_to_catalogs for one removed entry: reset its catalog entry from the untouched catalog snapshot (or drop it if it was not official), then re-apply the library entries that still share its name."""
    gguf = entry["backend"] == "gguf"
    catalog, pristine = (gguf_catalog, pristine_gguf) if gguf else (hf_catalog, pristine_hf)
    name = entry["name"]
    if name in pristine:
        catalog[name] = copy.deepcopy(pristine[name])
    else:
        catalog.pop(name, None)
    siblings = [e for e in remaining if e["name"] == name and e["backend"] == entry["backend"]]
    apply_to_catalogs(siblings, root, gguf_catalog, hf_catalog)


def restore(entry, root, log=logger.info) -> None:
    """Bring back a missing entry: re-convert from its origin file, or download from its source repo."""
    origin = entry.get("origin")
    if entry["backend"] == "hf" and origin and abs_path(origin, root).is_file():
        from captioner.comfy_convert import convert
        convert(abs_path(origin, root), abs_path(entry["model"], root), entry["base_repo"], log=log)
        return
    source = entry.get("source") or {}
    if not source.get("repo_id"):
        raise FileNotFoundError(f"{entry['name']}: {entry['model']} is missing and has no download source")
    from huggingface_hub import hf_hub_download, snapshot_download
    if entry["backend"] == "hf":
        snapshot_download(source["repo_id"], local_dir=str(abs_path(entry["model"], root)))
        return
    for stored, remote in zip((entry["model"], entry["mmproj"]), source["files"]):
        dst = abs_path(stored, root)
        if not dst.is_file():
            log(f"downloading {remote} from {source['repo_id']}")
            got = Path(hf_hub_download(source["repo_id"], remote, local_dir=str(dst.parent)))
            if got.resolve() != dst.resolve():
                got.replace(dst)
