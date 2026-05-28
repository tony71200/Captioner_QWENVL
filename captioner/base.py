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
        self._vram_profile: Optional[str] = None

    @abstractmethod
    def load_model(self, vram_profile: str, **kwargs) -> None:
        """
        Load the model with the given VRAM profile.
        vram_profile: one of 'LowVRAM', 'NormalVRAM', 'HighVRAM'
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
    def vram_profile(self) -> Optional[str]:
        return self._vram_profile

    def __repr__(self) -> str:
        status = f"loaded={self._loaded}, profile={self._vram_profile}"
        return f"{self.__class__.__name__}({status})"
