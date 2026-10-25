"""Self-checks for model_library. Run: .venv\\Scripts\\python.exe test_model_library.py"""
import tempfile
from pathlib import Path

import model_library as ml

GGUF_CAND = {"kind": "gguf", "path": "", "name": "Qwen3-VL-8B-X.Q4_K_M.gguf", "quant": "Q4_K_M", "size": "8B",
             "size_gb": 5.0, "mmproj": "", "usable": True, "reason": ""}


def _files(root: Path):
    (root / "GGUF").mkdir(parents=True)
    model, mm = root / "GGUF" / "Qwen3-VL-8B-X.Q4_K_M.gguf", root / "GGUF" / "mmproj-X.gguf"
    model.write_bytes(b"x")
    mm.write_bytes(b"x")
    return model, mm


def test_paths_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        assert ml.to_stored(root / "GGUF" / "a.gguf", root) == "GGUF/a.gguf"
        outside = Path(tempfile.gettempdir()) / "elsewhere.gguf"
        assert ml.to_stored(outside, root) == str(outside.resolve())
        assert ml.abs_path("GGUF/a.gguf", root) == root / "GGUF" / "a.gguf"
        assert ml.abs_path(str(outside), root) == outside


def test_save_load_and_bad_json():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "lib.json"
        assert ml.load(path) == ([], None)
        ml.save([{"name": "a", "model": "GGUF/a.gguf"}], path)
        assert ml.load(path) == ([{"name": "a", "model": "GGUF/a.gguf"}], None)
        path.write_text("{broken", encoding="utf-8")
        entries, warning = ml.load(path)
        assert entries == [] and "lib.json.bak" in warning and path.with_suffix(".json.bak").exists()


def test_status_new_entry_and_catalogs():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        model, mm = _files(root)
        cand = dict(GGUF_CAND, path=str(model), mmproj=str(mm))
        entry = ml.new_entry(cand, "Qwen3-VL-8B-X-GGUF", root, source_repo="someone/x-gguf")
        assert entry["model"] == "GGUF/Qwen3-VL-8B-X.Q4_K_M.gguf" and entry["mmproj"] == "GGUF/mmproj-X.gguf"
        assert entry["source"] == {"repo_id": "someone/x-gguf", "files": ["Qwen3-VL-8B-X.Q4_K_M.gguf", "mmproj-X.gguf"]}
        assert ml.status(entry, root) == "found"

        tested = ml.record(entry, {"date": "2026-10-25", "gpu": "GPU", "load_s": 1.0, "caption_s": 1.0,
                                   "vision_score": 5, "peak_vram_gb": 6.4, "profile": "LowVRAM (6–8GB)"})
        assert tested["test"]["peak_vram_gb"] == {"*": 6.4}

        gguf_cat, hf_cat = {}, {}
        assert ml.apply_to_catalogs([tested], root, gguf_cat, hf_cat) == {entry["model"]: "found"}
        info = gguf_cat["Qwen3-VL-8B-X-GGUF"]
        assert info["model_files"] == {"Q4_K_M (tested, ~6.4GB)": str(model)}
        assert info["mmproj_file"] == str(mm) and info["library"] is True

        model.unlink()
        assert ml.status(entry, root) == "downloadable"
        assert ml.status(dict(entry, source=None), root) == "missing"

        ml.unregister(tested, root, gguf_cat, hf_cat, official_names=set())
        assert "Qwen3-VL-8B-X-GGUF" not in gguf_cat


def test_hf_entry_and_merge_into_official():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        src = root / "x_fp8_scaled.safetensors"
        src.write_bytes(b"x")
        cand = {"kind": "safetensors", "path": str(src), "name": src.name, "base_repo": "Qwen/Qwen3-VL-4B-Instruct",
                "size": "4B", "bf16_gb": 8.9, "usable": True, "reason": ""}
        entry = ml.new_entry(cand, "x-HF", root)
        assert (entry["backend"], entry["model"], entry["origin"]) == ("hf", "HuggingFace/x-HF", "x_fp8_scaled.safetensors")
        assert ml.status(entry, root) == "downloadable"          # origin present, folder not converted yet
        (root / "HuggingFace" / "x-HF").mkdir(parents=True)
        (root / "HuggingFace" / "x-HF" / "config.json").write_text("{}", encoding="utf-8")
        assert ml.status(entry, root) == "found"

        hf_cat = {}
        tested = ml.record(entry, {"date": "d", "gpu": "g", "load_s": 1, "caption_s": 1, "vision_score": 5,
                                   "peak_vram_gb": 3.4, "profile": "LowVRAM (6–8GB)"})
        ml.apply_to_catalogs([tested], root, {}, hf_cat)
        assert hf_cat["x-HF"]["vram"]["4bit"] == 3.4 and hf_cat["x-HF"]["local_dir"] == str(root / "HuggingFace" / "x-HF")

        # a GGUF entry whose file belongs to an official catalog entry joins that entry
        model, mm = _files(root / "r2")
        official = {"Qwen3-VL-8B-X-GGUF": {"model_files": {"Q4_K_M (recommended, ~5GB)": "Qwen3-VL-8B-X.Q4_K_M.gguf",
                                                           "Q8_0 (~9GB)": "Qwen3-VL-8B-X.Q8_0.gguf"},
                                           "mmproj_file": "mmproj-X-F16.gguf", "min_vram_4gb": False}}
        root2 = root / "r2"
        name = ml.suggest_name(model, "gguf", official)
        assert name == "Qwen3-VL-8B-X-GGUF"
        g = ml.new_entry(dict(GGUF_CAND, path=str(model), mmproj=str(mm)), name, root2)
        ml.apply_to_catalogs([g], root2, official, {})
        files = official[name]["model_files"]
        assert len(files) == 2 and "Q4_K_M (untested, ~5.0GB)" in files and "Q8_0 (~9GB)" in files
        assert ml.suggest_name(root / "Huihui-Qwen3-VL-4B-abliterated-fp8_scaled.safetensors", "safetensors", {}) \
            == "Huihui-Qwen3-VL-4B-abliterated-HF"


def test_upsert_remove_known():
    a = {"name": "a", "backend": "gguf", "model": "GGUF/a.gguf"}
    b = {"name": "b", "backend": "hf", "model": "HuggingFace/b", "origin": "b.safetensors"}
    entries = ml.upsert(ml.upsert([], a), b)
    entries = ml.upsert(entries, dict(a, quant="Q8_0"))
    assert len(entries) == 2 and ml.find(entries, "GGUF/a.gguf")["quant"] == "Q8_0"
    root = Path(tempfile.gettempdir())
    assert ml.known_paths(entries, root) == {str((root / "GGUF/a.gguf").resolve()), str((root / "b.safetensors").resolve())}
    assert ml.remove(entries, "GGUF/a.gguf") == [b]


if __name__ == "__main__":
    test_paths_round_trip()
    test_save_load_and_bad_json()
    test_status_new_entry_and_catalogs()
    test_hf_entry_and_merge_into_official()
    test_upsert_remove_known()
    print("test_model_library OK")
