"""
Backend HuggingFace Transformers cho Qwen-VL.

Nhận thông số cụ thể (quant, n_ctx, max_pixels) do utils/vram_plan tính ra,
không nhận tên profile. Chạy được cả Qwen3-VL lẫn Qwen2.5-VL qua
AutoModelForImageTextToText.

Lưu ý: bitsandbytes (4bit/8bit) KHÔNG chạy trên CPU — tự rơi về float32.
"""
import gc
import logging
from pathlib import Path
from typing import Optional

import torch

from .base import BaseCaptioner

logger = logging.getLogger(__name__)

# Default model to use
DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"


def _pick_hf_device(device_choice: str) -> str:
    """
    Resolve device_choice to 'cuda' or 'cpu' for the HuggingFace backend.

    Args:
        device_choice: 'auto' | 'cpu' | 'cuda'

    Returns:
        'cuda' or 'cpu'
    """
    choice = (device_choice or "auto").strip().lower()
    cuda_ok = torch.cuda.is_available()

    if choice == "auto":
        return "cuda" if cuda_ok else "cpu"
    if choice.startswith("cuda"):
        return "cuda" if cuda_ok else "cpu"
    return "cpu"


class HFCaptioner(BaseCaptioner):
    """
    Captioner dùng HuggingFace Transformers — Qwen3-VL và Qwen2.5-VL.
    """

    def __init__(self):
        super().__init__()
        self.model = None
        self.processor = None
        self.model_id: Optional[str] = None
        self._device_kind: str = "cuda"
        self._max_pixels: int = 0
        self._n_ctx: int = 0
        # Cùng giao diện với GGUFCaptioner để app.py xử lý đồng nhất.
        # HF không có fallback OOM->CPU (ngoài phạm vi theo spec muc 7).
        self.runtime_device: str = "unknown"
        self.fallback_reason: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_model(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        quant: str = "4bit",
        n_ctx: int = 4096,
        max_pixels: int = 1280 * 28 * 28,
        device: str = "auto",
        use_flash_attn: bool = False,
        **kwargs,
    ) -> None:
        """
        Nạp model Qwen-VL qua HuggingFace Transformers.

        quant: 'bf16' | '8bit' | '4bit' | 'fp8' | 'awq'
               'fp8'/'awq' nghĩa là checkpoint đã quantize sẵn — nạp nguyên
               trạng, KHÔNG chồng bitsandbytes lên trên.
        """
        from transformers import AutoProcessor, AutoModelForImageTextToText

        device_kind = _pick_hf_device(device)
        self._device_kind = device_kind
        self.runtime_device = device_kind
        self.fallback_reason = ""
        logger.info("HF device: yeu cau='%s' -> thuc te='%s'", device, device_kind)

        if self._loaded:
            self.unload_model()

        self.model_id = model_id
        self._max_pixels = max_pixels
        self._n_ctx = n_ctx

        # bitsandbytes khong chay tren CPU
        if device_kind == "cpu" and quant in ("4bit", "8bit"):
            logger.warning(
                "bitsandbytes (%s) khong ho tro CPU - roi ve float32. "
                "RAM cao va cham hon nhieu.", quant,
            )
            quant = "bf16"

        model_kwargs = {
            "dtype": torch.float32 if device_kind == "cpu" else torch.bfloat16,
            "device_map": "cpu" if device_kind == "cpu" else "auto",
            "low_cpu_mem_usage": True,
        }

        # Chi ap bitsandbytes len checkpoint CHUA quantize.
        bnb_config = self._build_bnb_config(quant)
        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config

        self._configure_torch_runtime()

        if device_kind == "cuda":
            if use_flash_attn and quant == "bf16":
                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Bat Flash Attention 2.")
            else:
                model_kwargs["attn_implementation"] = "sdpa"

        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id, **model_kwargs
            )
        except Exception as e:
            raise RuntimeError(
                f"Khong nap duoc model HuggingFace '{model_id}'. Loi goc: {e}"
            ) from e

        self.model.eval()
        if hasattr(self.model, "config"):
            self.model.config.use_cache = True
        if getattr(self.model, "generation_config", None) is not None:
            self.model.generation_config.use_cache = True

        try:
            self.processor = AutoProcessor.from_pretrained(
                model_id,
                min_pixels=256 * 28 * 28,
                max_pixels=max_pixels,
            )
        except Exception as e:
            self.unload_model()
            raise RuntimeError(
                f"Khong nap duoc processor cho '{model_id}'. Loi goc: {e}"
            ) from e

        self._loaded = True
        self._runtime_desc = f"quant={quant} ctx={n_ctx} device={device_kind}"
        logger.info("Da nap model HF. %s", self._runtime_desc)


    def caption_image(
        self,
        image_path: str,
        prompt: str,
        max_new_tokens: int = 512,
    ) -> str:
        """
        Generate a caption for the image at image_path.

        Args:
            image_path:     Absolute path to the image file
            prompt:         Text prompt to send with the image
            max_new_tokens: Maximum number of new tokens to generate

        Returns:
            Generated caption string
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Lazy import
        from qwen_vl_utils import process_vision_info

        image_path = str(Path(image_path).resolve())

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Prepare inputs
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        target_device = "cuda" if (self._device_kind == "cuda" and torch.cuda.is_available()) else "cpu"
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(target_device)

        # Generate
        with torch.inference_mode():
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        self._loaded = False
        self._runtime_desc = None
        self.runtime_device = "unknown"
        logger.info("HF model unloaded and VRAM freed.")

    def get_vram_usage_mb(self) -> float:
        """Return current GPU VRAM usage in MB (if CUDA available)."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_bnb_config(quant_mode: str):
        """
        BitsAndBytesConfig cho quant mode, hoặc None.

        Trả None với 'bf16', 'fp8', 'awq' — fp8/awq là checkpoint đã quantize
        sẵn, chồng bitsandbytes lên trên sẽ hỏng.
        """
        if quant_mode not in ("4bit", "8bit"):
            return None

        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            logger.warning("Khong co bitsandbytes; bo qua quantization.")
            return None

        if quant_mode == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif quant_mode == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True)
        else:
            # none / full precision
            return None

    @staticmethod
    def _configure_torch_runtime() -> None:
        """Enable safe CUDA runtime optimizations similar to ComfyUI defaults."""
        if not torch.cuda.is_available():
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
