"""
GGUF backend for Qwen-VL image captioning using llama-cpp-python.
Supports CPU and GPU (CUDA) inference via device selection.
"""
import contextlib
import gc
import inspect
import logging
import os
import time
from pathlib import Path
from typing import Optional

from .base import BaseCaptioner

logger = logging.getLogger(__name__)


def _coerce_runtime_value(explicit_value, default_value: int) -> int:
    if explicit_value is None:
        return int(default_value)
    return int(explicit_value)


def _looks_like_cuda_oom_or_init_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    hints = ("cuda error", "cublas", "out of memory", "failed to allocate", "ggml-cuda")
    return any(h in msg for h in hints)




@contextlib.contextmanager
def _cpu_cuda_hidden_env(enable: bool):
    if not enable:
        yield
        return
    old = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old


def _pick_device(device_choice: str) -> str:
    """
    Resolve device_choice to 'cuda' or 'cpu'.

    Logic (mirrors ComfyUI-QwenVL):
        'auto'  → prefer CUDA, fallback to CPU
        'cuda'  → CUDA if available, else CPU
        'cpu'   → CPU always
    """
    choice = (device_choice or "auto").strip().lower()
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
    except ImportError:
        cuda_ok = False

    if choice == "auto":
        return "cuda" if cuda_ok else "cpu"
    if choice.startswith("cuda"):
        return "cuda" if cuda_ok else "cpu"
    return "cpu"


class GGUFCaptioner(BaseCaptioner):
    """
    Image captioner using llama-cpp-python with GGUF model files.
    Requires: model .gguf file + mmproj .gguf file.
    Supports CPU (n_gpu_layers=0) and GPU (n_gpu_layers>0) inference.
    """

    def __init__(self):
        super().__init__()
        self.llm = None
        self.model_path: Optional[str] = None
        self.mmproj_path: Optional[str] = None
        self.current_signature: Optional[tuple] = None
        self.runtime_device: str = "unknown"
        self.fallback_reason: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_model(
        self,
        vram_profile: str,
        model_path: str = "",
        mmproj_path: str = "",
        model_name: str = "",
        device: str = "auto",
        **kwargs,
    ) -> None:
        """
        Load GGUF model.

        Args:
            vram_profile: VRAM profile name from models_catalog.VRAM_PROFILES
            model_path:   Path to the main GGUF model file
            mmproj_path:  Path to the multimodal projection GGUF file
            model_name:   Model name key in GGUF_VL_MODELS catalog
            device:       'auto' | 'cpu' | 'cuda'  (default: 'auto')
        """
        from llama_cpp import Llama

        model_path = str(Path(model_path).expanduser())
        mmproj_path = str(Path(mmproj_path).expanduser())

        if not model_path.strip():
            raise FileNotFoundError("GGUF model path is empty. Select a local .gguf model file.")
        if not mmproj_path.strip():
            raise FileNotFoundError("MMProj path is empty. Select the matching mmproj .gguf file.")
        if not Path(model_path).exists():
            raise FileNotFoundError(f"GGUF model file not found: {model_path}")
        if not Path(mmproj_path).exists():
            raise FileNotFoundError(f"MMProj file not found: {mmproj_path}")

        # ── Resolve device ───────────────────────────────────────────────────
        device_kind = _pick_device(device)
        logger.info("GGUF device selection: requested='%s' → resolved='%s'", device, device_kind)
        self.runtime_device = device_kind
        self.fallback_reason = ""

        try:
            from models_catalog import GGUF_VL_MODELS, VRAM_PROFILES
            profile = VRAM_PROFILES.get(vram_profile, {})
            model_info = GGUF_VL_MODELS.get(model_name, {})
        except ImportError:
            profile = {}
            model_info = {}

        gguf_defaults = model_info.get("gguf_defaults", {})

        # ── GPU layers — 0 forces full CPU inference ─────────────────────────
        if device_kind == "cuda":
            n_gpu_layers = _coerce_runtime_value(
                kwargs.get("gpu_layers"),
                gguf_defaults.get("gpu_layers", profile.get("gguf_layers", -1)),
            )
        else:
            # CPU mode: no GPU offload
            n_gpu_layers = 0

        # ── CPU threads — only meaningful in CPU mode ─────────────────────────
        if device_kind == "cpu":
            n_threads = int(kwargs.get("n_threads") or os.cpu_count() or 4)
        else:
            n_threads = None  # let llama.cpp decide when using GPU

        n_ctx = _coerce_runtime_value(
            kwargs.get("n_ctx"),
            gguf_defaults.get("context_length", profile.get("gguf_ctx", 8192)),
        )
        n_batch = _coerce_runtime_value(
            kwargs.get("n_batch"),
            gguf_defaults.get("n_batch", 512),
        )
        image_min_tokens = max(1024, _coerce_runtime_value(
            kwargs.get("image_min_tokens"),
            gguf_defaults.get("image_min_tokens", 1024),
        ))
        image_max_tokens = _coerce_runtime_value(
            kwargs.get("image_max_tokens"),
            gguf_defaults.get("image_max_tokens", 4096),
        )
        if image_max_tokens < image_min_tokens:
            image_max_tokens = image_min_tokens
        top_k = _coerce_runtime_value(
            kwargs.get("top_k"),
            gguf_defaults.get("top_k", 0),
        )
        pool_size = _coerce_runtime_value(
            kwargs.get("pool_size"),
            gguf_defaults.get("pool_size", 4194304),
        )

        signature = (
            model_path,
            mmproj_path,
            n_gpu_layers,
            n_ctx,
            n_batch,
            image_min_tokens,
            image_max_tokens,
            top_k,
            pool_size,
            device_kind,
        )

        logger.info(
            "Loading GGUF model: %s\n"
            "MMProj: %s\n"
            "Profile: %s | device=%s | gpu_layers=%d | ctx=%d | "
            "n_batch=%d | n_threads=%s | image_min_tokens=%d | image_max_tokens=%d",
            model_path, mmproj_path, vram_profile,
            device_kind, n_gpu_layers, n_ctx,
            n_batch, n_threads, image_min_tokens, image_max_tokens,
        )

        self.model_path = model_path
        self.mmproj_path = mmproj_path
        self._vram_profile = vram_profile

        if self.llm is not None and self.current_signature == signature:
            logger.info("GGUF model already loaded with matching runtime config.")
            self._loaded = True
            return

        if self._loaded:
            logger.info("GGUF model already loaded. Unloading first.")
            self.unload_model()

        chat_handler = self._build_chat_handler(
            model_path=model_path,
            mmproj_path=mmproj_path,
            image_min_tokens=image_min_tokens,
            image_max_tokens=image_max_tokens,
        )
        llm_kwargs = {
            "model_path": model_path,
            "chat_handler": chat_handler,
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "n_batch": n_batch,
            "swa_full": True,
            "pool_size": pool_size,
            "top_k": top_k,
            "image_min_tokens": image_min_tokens,
            "image_max_tokens": image_max_tokens,
            "offload_kqv": False if device_kind == "cpu" else True,
            "flash_attn": False if device_kind == "cpu" else True,
            "verbose": False,
        }
        # Add n_threads only for CPU mode (avoid confusing GPU builds)
        if n_threads is not None:
            llm_kwargs["n_threads"] = n_threads

        llm_kwargs = self._filter_kwargs_for_callable(getattr(Llama, "__init__", Llama), llm_kwargs)

        try:
            with _cpu_cuda_hidden_env(device_kind == "cpu"):
                self.llm = Llama(**llm_kwargs)
            loaded_signature = signature
        except Exception as e:
            if device_kind == "cuda" and _looks_like_cuda_oom_or_init_error(e):
                logger.warning("CUDA init failed (%s). Retrying GGUF on CPU with safe settings.", e)
                self.fallback_reason = str(e)
                cpu_kwargs = dict(llm_kwargs)
                cpu_kwargs["n_gpu_layers"] = 0
                cpu_kwargs["n_threads"] = int(kwargs.get("n_threads") or os.cpu_count() or 4)
                cpu_kwargs["n_batch"] = min(int(cpu_kwargs.get("n_batch", n_batch)), 128)
                with _cpu_cuda_hidden_env(True):
                    self.llm = Llama(**cpu_kwargs)
                device_kind = "cpu-fallback"
                self.runtime_device = device_kind
                n_gpu_layers = 0
                loaded_signature = (
                    model_path,
                    mmproj_path,
                    0,
                    n_ctx,
                    int(cpu_kwargs.get("n_batch", n_batch)),
                    image_min_tokens,
                    image_max_tokens,
                    top_k,
                    pool_size,
                    device_kind,
                )
            else:
                raise

        self.runtime_device = device_kind
        self._loaded = True
        self.current_signature = loaded_signature
        logger.info(
            "GGUF model loaded successfully. device=%s, gpu_layers=%d",
            device_kind, n_gpu_layers,
        )

    def caption_image(
        self,
        image_path: str,
        user_prompt: str,
        max_new_tokens: int = 512,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate a caption for the image at image_path.

        Args:
            image_path:     Absolute path to the image file
            user_prompt:    Text prompt
            max_new_tokens: Max tokens to generate
            system_prompt:  Optional system prompt to guide the model

        Returns:
            Generated caption string
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        image_url = self._image_to_url(image_path)

        effective_system = system_prompt or (
            "You are a helpful vision-language assistant. "
            "Answer directly with the final answer only. No <think> and no reasoning."
        )
        messages = [
            {"role": "system", "content": effective_system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ]

        start = time.perf_counter()
        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=0.6,
            top_p=0.9,
            repeat_penalty=1.2,
            seed=1,
            stop=["<|im_end|>", "<|im_start|>"],
        )
        elapsed = max(time.perf_counter() - start, 1e-6)
        usage = response.get("usage") or {}
        completion_tokens = usage.get("completion_tokens")
        if isinstance(completion_tokens, int) and completion_tokens > 0:
            logger.info(
                "GGUF completion: %s tokens in %.2fs (%.2f tok/s)",
                completion_tokens,
                elapsed,
                completion_tokens / elapsed,
            )

        return response["choices"][0]["message"]["content"].strip()

    def caption_text(
        self,
        user_prompt: str,
        max_new_tokens: int = 512,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate text from a text-only prompt."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        effective_system = system_prompt or (
            "You are a helpful assistant. Answer directly with the final answer only. "
            "No <think> and no reasoning."
        )
        messages = [
            {"role": "system", "content": effective_system},
            {"role": "user", "content": user_prompt},
        ]

        start = time.perf_counter()
        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=0.4,
            top_p=0.9,
            repeat_penalty=1.15,
            seed=1,
            stop=["<|im_end|>", "<|im_start|>"],
        )
        elapsed = max(time.perf_counter() - start, 1e-6)
        usage = response.get("usage") or {}
        completion_tokens = usage.get("completion_tokens")
        if isinstance(completion_tokens, int) and completion_tokens > 0:
            logger.info(
                "GGUF text completion: %s tokens in %.2fs (%.2f tok/s)",
                completion_tokens,
                elapsed,
                completion_tokens / elapsed,
            )

        return response["choices"][0]["message"]["content"].strip()

    def unload_model(self) -> None:
        """Release model from memory."""
        if self.llm is not None:
            del self.llm
            self.llm = None

        gc.collect()
        self._loaded = False
        self._vram_profile = None
        self.current_signature = None
        self.runtime_device = "unknown"
        self.fallback_reason = ""
        logger.info("GGUF model unloaded.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _image_to_url(image_path: str) -> str:
        """
        Convert a local image path to a file:// URL.

        Using a file URL avoids the extra Python-side base64 encode/decode step,
        which helps batch throughput and matches how llama-cpp already loads
        local media internally.
        """
        return Path(image_path).resolve().as_uri()

    @staticmethod
    def _filter_kwargs_for_callable(fn, kwargs: dict) -> dict:
        try:
            sig = inspect.signature(fn)
        except Exception:
            return dict(kwargs)

        params = list(sig.parameters.values())
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
            return dict(kwargs)

        allowed = {
            p.name
            for p in params
            if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        return {key: value for key, value in kwargs.items() if key in allowed}

    @staticmethod
    def _build_chat_handler(model_path: str, mmproj_path: str, image_min_tokens: int, image_max_tokens: int):
        """Instantiate the best available llama-cpp vision chat handler."""
        import llama_cpp.llama_chat_format as chat_format

        name_hint = f"{Path(model_path).name} {Path(mmproj_path).name}".lower()
        if "qwen3" in name_hint:
            handler_names = ["Qwen3VLChatHandler", "Qwen25VLChatHandler", "Llava15ChatHandler"]
        elif "qwen2.5" in name_hint or "qwen25" in name_hint:
            handler_names = ["Qwen25VLChatHandler", "Qwen3VLChatHandler", "Llava15ChatHandler"]
        else:
            handler_names = ["Qwen3VLChatHandler", "Qwen25VLChatHandler", "Llava15ChatHandler"]

        errors = []
        for handler_name in handler_names:
            handler_cls = getattr(chat_format, handler_name, None)
            if handler_cls is None:
                errors.append(f"{handler_name} not exported by llama_cpp.llama_chat_format")
                continue

            try:
                init_kwargs = {
                    "clip_model_path": mmproj_path,
                    "image_min_tokens": image_min_tokens,
                    "image_max_tokens": image_max_tokens,
                    "force_reasoning": False,
                    "verbose": False,
                    "add_vision_id": False,
                }
                init_kwargs = GGUFCaptioner._filter_kwargs_for_callable(
                    getattr(handler_cls, "__init__", handler_cls),
                    init_kwargs,
                )
                chat_handler = handler_cls(**init_kwargs)
                logger.info("Using %s for GGUF vision chat.", handler_name)
                return chat_handler
            except Exception as e:
                errors.append(f"{handler_name}: {e}")

        raise RuntimeError(
            "No compatible vision chat handler could be initialized for this GGUF model. "
            "Tried: "
            + " | ".join(errors)
        )
