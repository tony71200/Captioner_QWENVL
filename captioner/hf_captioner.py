"""
HuggingFace Transformers backend for Qwen-VL image captioning.
Supports UltraLow (4GB), LowVRAM (4-bit), NormalVRAM (8-bit), and HighVRAM (bf16) profiles.
Supports CPU and GPU (CUDA) inference via device selection.
Note: bitsandbytes quantization (4bit/8bit) is NOT supported on CPU — falls back to float32.
"""
import gc
import logging
from pathlib import Path
from typing import Optional

try:
    import torch
except ImportError:
    torch = None

from .base import BaseCaptioner

logger = logging.getLogger(__name__)

# Default model to use
DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"


def _require_torch():
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for the HuggingFace backend. Install torch before loading HF models."
        )
    return torch


def _pick_hf_device(device_choice: str) -> str:
    """
    Resolve device_choice to 'cuda' or 'cpu' for the HuggingFace backend.

    Args:
        device_choice: 'auto' | 'cpu' | 'cuda'

    Returns:
        'cuda' or 'cpu'
    """
    choice = (device_choice or "auto").strip().lower()
    cuda_ok = bool(torch and torch.cuda.is_available())

    if choice == "auto":
        return "cuda" if cuda_ok else "cpu"
    if choice.startswith("cuda"):
        return "cuda" if cuda_ok else "cpu"
    return "cpu"


class HFCaptioner(BaseCaptioner):
    """
    Image captioner using HuggingFace Transformers (Qwen2.5-VL).
    """

    def __init__(self):
        super().__init__()
        self.model = None
        self.processor = None
        self.model_id: Optional[str] = None
        self._pixel_config: dict = {}
        self._device_kind: str = "cuda"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_model(
        self,
        vram_profile: str,
        model_id: str = DEFAULT_MODEL_ID,
        use_flash_attn: bool = False,
        device: str = "auto",
        **kwargs,
    ) -> None:
        """
        Load model with the given VRAM profile.

        Args:
            vram_profile:  one of the keys in models_catalog.VRAM_PROFILES
            model_id:      HuggingFace model ID or local path
            use_flash_attn: Enable Flash Attention 2 (requires Ampere+ GPU)
            device:        'auto' | 'cpu' | 'cuda'  (default: 'auto')
        """
        torch_mod = _require_torch()
        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as e:
            raise RuntimeError(
                "The installed transformers package does not expose the Qwen2-VL loader required "
                "by this HuggingFace backend. Upgrade transformers/qwen-vl-utils to a version "
                "that supports your selected Qwen-VL model."
            ) from e

        # ── Resolve device ───────────────────────────────────────────────────
        device_kind = _pick_hf_device(device)
        self._device_kind = device_kind
        logger.info("HF device selection: requested='%s' → resolved='%s'", device, device_kind)

        # Import profile config from catalog
        try:
            from models_catalog import VRAM_PROFILES
            profile = VRAM_PROFILES.get(vram_profile, list(VRAM_PROFILES.values())[1])
        except ImportError:
            # Fallback defaults
            profile = {
                "hf_quant": "4bit",
                "pixel_config": {"min_pixels": 256 * 28 * 28, "max_pixels": 768 * 28 * 28},
            }

        if self._loaded:
            logger.info("Model already loaded. Unloading first.")
            self.unload_model()

        logger.info("Loading HF model: %s | Profile: %s | Device: %s", model_id, vram_profile, device_kind)
        self.model_id = model_id
        self._vram_profile = vram_profile
        self._pixel_config = profile["pixel_config"]
        quant_mode = profile["hf_quant"]

        # ── CPU mode: bitsandbytes is NOT supported on CPU ───────────────────
        if device_kind == "cpu":
            if quant_mode in ("4bit", "8bit"):
                logger.warning(
                    "bitsandbytes quantization (%s) is not supported on CPU. "
                    "Falling back to float32 (full precision). "
                    "Expect high RAM usage and slower inference.",
                    quant_mode,
                )
            quant_mode = "none"

        # Build quantization config
        bnb_config = self._build_bnb_config(quant_mode)

        # Build model kwargs
        model_kwargs = {
            "torch_dtype": torch_mod.float32 if device_kind == "cpu" else torch_mod.bfloat16,
            "device_map": "cpu" if device_kind == "cpu" else "auto",
            "low_cpu_mem_usage": True,
        }
        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config

        self._configure_torch_runtime()

        if device_kind == "cuda":
            if use_flash_attn and quant_mode == "none":
                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Flash Attention 2 enabled.")
            else:
                model_kwargs["attn_implementation"] = "sdpa"
                logger.info("Using SDPA attention backend.")
        # On CPU, leave attn_implementation unset (use default eager/sdpa if supported)

        if "Qwen3-VL" in model_id:
            raise RuntimeError(
                "The HuggingFace backend is still wired for the Qwen2-VL API, but the selected "
                f"model is '{model_id}'. Use the GGUF backend for Qwen3-VL right now, or upgrade "
                "the HuggingFace stack and loader implementation to a Qwen3-VL-compatible API."
            )

        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id, **model_kwargs
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load HuggingFace model '{model_id}'. "
                "This backend expects a Qwen2-VL-compatible model/API in the current transformers "
                f"environment. Original error: {e}"
            ) from e
        self.model.eval()
        if hasattr(self.model, "config"):
            self.model.config.use_cache = True
        if getattr(self.model, "generation_config", None) is not None:
            self.model.generation_config.use_cache = True

        try:
            self.processor = AutoProcessor.from_pretrained(
                model_id,
                min_pixels=self._pixel_config["min_pixels"],
                max_pixels=self._pixel_config["max_pixels"],
            )
        except Exception as e:
            self.unload_model()
            raise RuntimeError(
                f"Failed to load processor for HuggingFace model '{model_id}'. Original error: {e}"
            ) from e

        self._loaded = True
        logger.info("HF model loaded. Profile: %s | Quant: %s | Device: %s", vram_profile, quant_mode, device_kind)


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
            user_prompt:    Text prompt to send with the image
            max_new_tokens: Maximum number of new tokens to generate
            system_prompt:  Optional system prompt to guide the model

        Returns:
            Generated caption string
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        torch_mod = _require_torch()

        # Lazy import
        from qwen_vl_utils import process_vision_info

        image_path = str(Path(image_path).resolve())

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": user_prompt},
                ],
            }
        )

        # Prepare inputs
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        target_device = "cuda" if (self._device_kind == "cuda" and torch_mod.cuda.is_available()) else "cpu"
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(target_device)

        # Generate
        with torch_mod.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

        # Decode — strip the input tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return output_text[0].strip()

    def unload_model(self) -> None:
        """Delete model from memory and clear CUDA cache."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None

        gc.collect()
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        self._loaded = False
        self._vram_profile = None
        logger.info("HF model unloaded and VRAM freed.")

    def get_vram_usage_mb(self) -> float:
        """Return current GPU VRAM usage in MB (if CUDA available)."""
        if torch and torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_bnb_config(quant_mode: str):
        """
        Return a BitsAndBytesConfig for the given quant mode, or None.
        quant_mode: '4bit', '8bit', or 'none'
        """
        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            logger.warning("bitsandbytes not installed; skipping quantization.")
            return None

        if quant_mode == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=_require_torch().bfloat16,
            )
        elif quant_mode == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True)
        else:
            # none / full precision
            return None

    @staticmethod
    def _configure_torch_runtime() -> None:
        """Enable safe CUDA runtime optimizations similar to ComfyUI defaults."""
        if not torch or not torch.cuda.is_available():
            return
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass
        try:
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
