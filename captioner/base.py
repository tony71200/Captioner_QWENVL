"""
Abstract base class for all captioner backends.
"""
from abc import ABC, abstractmethod
from typing import Optional


class BaseCaptioner(ABC):
    """
    Abstract interface that all captioner backends must implement.
    """

    def __init__(self):
        self._loaded = False
        self._runtime_desc: Optional[str] = None

    @abstractmethod
    def load_model(self, **kwargs) -> None:
        """
        Nạp model. Backend nhận thông số cụ thể (quant, n_ctx, n_gpu_layers…),
        do utils/vram_plan tính ra — không nhận tên profile.
        """
        ...

    @abstractmethod
    def caption_image(
        self,
        image_path: str,
        prompt: str,
        max_new_tokens: int = 512,
    ) -> str:
        """
        Generate a caption for the given image.
        Returns the generated text string.
        """
        ...

    @abstractmethod
    def unload_model(self) -> None:
        """Release model from memory and free GPU VRAM."""
        ...

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def runtime_desc(self) -> Optional[str]:
        return self._runtime_desc

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(loaded={self._loaded}, {self._runtime_desc})"
