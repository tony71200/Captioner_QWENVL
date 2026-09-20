"""
Các backend captioner.

HFCaptioner được nạp LAZY (PEP 562). Lý do: `captioner/hf_captioner.py` import
torch ở đầu module, mà torch kéo theo libiomp5md.dll (Intel OpenMP của MKL),
xung đột với libomp140 của llama-cpp-python và giết tiến trình khi caption GGUF
(OMP Error #15, exit 3).

Nếu import nó ở đây thì chỉ cần `import captioner.gguf_captioner` là torch đã
vào tiến trình — người dùng GGUF thuần vẫn dính lỗi. Xem
test_duong_gguf_khong_nap_torch.

`from captioner import HFCaptioner` vẫn hoạt động như cũ, chỉ là torch được nạp
đúng lúc ai đó thật sự cần backend HF.
"""
from .base import BaseCaptioner
from .prompts import PROMPT_TEMPLATES, get_prompt_names, get_prompt_by_name
from .gguf_captioner import GGUFCaptioner

__all__ = [
    "BaseCaptioner",
    "HFCaptioner",
    "GGUFCaptioner",
    "PROMPT_TEMPLATES",
    "get_prompt_names",
    "get_prompt_by_name",
]


def __getattr__(name):
    if name == "HFCaptioner":
        from .hf_captioner import HFCaptioner
        return HFCaptioner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
