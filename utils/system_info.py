"""
System information utilities for QwenVL Captioner.
Collects CPU and GPU hardware specs for display in the WebUI.
"""
import os
import platform
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  CPU Info
# ─────────────────────────────────────────────────────────────────────────────

def get_cpu_info() -> Dict[str, Any]:
    """
    Return a dict with CPU hardware information.

    Keys:
        name        (str)  — processor brand string
        physical    (int)  — physical core count
        logical     (int)  — logical thread count
        ram_total   (float) — total RAM in GB
        ram_free    (float) — available RAM in GB
        ram_used    (float) — used RAM in GB
        ram_percent (float) — used RAM percentage
    """
    info: Dict[str, Any] = {
        "name": platform.processor() or "Unknown CPU",
        "physical": os.cpu_count() or 1,
        "logical": os.cpu_count() or 1,
        "ram_total": 0.0,
        "ram_free": 0.0,
        "ram_used": 0.0,
        "ram_percent": 0.0,
    }

    try:
        import psutil
        info["physical"] = psutil.cpu_count(logical=False) or os.cpu_count() or 1
        info["logical"] = psutil.cpu_count(logical=True) or os.cpu_count() or 1
        vm = psutil.virtual_memory()
        info["ram_total"] = vm.total / (1024 ** 3)
        info["ram_free"] = vm.available / (1024 ** 3)
        info["ram_used"] = (vm.total - vm.available) / (1024 ** 3)
        info["ram_percent"] = vm.percent
        # Try to get a better CPU name on Windows
        if platform.system() == "Windows":
            try:
                import winreg
                key = winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE,
                    r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
                )
                name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
                winreg.CloseKey(key)
                info["name"] = name.strip()
            except Exception:
                pass
    except ImportError:
        logger.debug("psutil not installed; RAM info unavailable.")

    return info


# ─────────────────────────────────────────────────────────────────────────────
#  GPU Info
# ─────────────────────────────────────────────────────────────────────────────

def get_gpu_info() -> List[Dict[str, Any]]:
    """
    Return a list of dicts, one per CUDA GPU.

    Each dict has:
        index       (int)   — GPU index
        name        (str)   — GPU display name
        vram_total  (float) — total VRAM in GB
        vram_free   (float) — free VRAM in GB
        vram_used   (float) — used VRAM in GB
        vram_percent (float) — used VRAM percentage
        compute     (str)   — CUDA compute capability string
    """
    gpus: List[Dict[str, Any]] = []
    try:
        import torch
        if not torch.cuda.is_available():
            return gpus
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total = props.total_memory / (1024 ** 3)
            # allocated memory is current usage in PyTorch allocator
            alloc = torch.cuda.memory_allocated(i) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
            # Use reserved as a proxy for "used" (matches nvidia-smi more closely)
            used = reserved
            free = total - used
            percent = (used / total * 100) if total > 0 else 0.0
            gpus.append({
                "index": i,
                "name": props.name,
                "vram_total": round(total, 2),
                "vram_free": round(free, 2),
                "vram_used": round(used, 2),
                "vram_percent": round(percent, 1),
                "compute": f"{props.major}.{props.minor}",
            })
    except ImportError:
        logger.debug("torch not available; GPU info unavailable.")
    except Exception as e:
        logger.debug("GPU info error: %s", e)
    return gpus


# ─────────────────────────────────────────────────────────────────────────────
#  HTML rendering helpers
# ─────────────────────────────────────────────────────────────────────────────

def _bar_html(percent: float, color: str = "#38bdf8") -> str:
    """Render a small progress bar as HTML."""
    clamped = max(0.0, min(100.0, percent))
    # Color shifts: green → yellow → red
    if percent < 50:
        color = "#4ade80"
    elif percent < 80:
        color = "#fbbf24"
    else:
        color = "#f87171"
    return (
        f'<div style="background:rgba(255,255,255,0.08);border-radius:4px;height:6px;'
        f'width:100%;margin-top:4px;">'
        f'<div style="background:{color};border-radius:4px;height:6px;'
        f'width:{clamped:.1f}%;transition:width 0.4s;"></div>'
        f'</div>'
    )


def _info_row(label: str, value: str) -> str:
    return (
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'padding:3px 0;border-bottom:1px solid rgba(255,255,255,0.04);">'
        f'<span style="color:#64748b;font-size:12px;">{label}</span>'
        f'<span style="color:#cbd5e1;font-size:12px;font-weight:500;">{value}</span>'
        f'</div>'
    )


def _card(title: str, icon: str, body: str) -> str:
    return (
        f'<div style="background:rgba(15,25,38,0.85);border:1px solid rgba(56,189,248,0.15);'
        f'border-radius:10px;padding:12px 14px;margin-bottom:10px;">'
        f'<div style="font-size:13px;font-weight:600;color:#38bdf8;margin-bottom:8px;">'
        f'{icon}&nbsp; {title}</div>'
        f'{body}'
        f'</div>'
    )


def get_cpu_info_html() -> str:
    """Return an HTML card displaying CPU specifications."""
    cpu = get_cpu_info()
    body = "".join([
        _info_row("Processor", cpu["name"]),
        _info_row("Physical cores", str(cpu["physical"])),
        _info_row("Logical threads", str(cpu["logical"])),
        _info_row("RAM Total", f"{cpu['ram_total']:.1f} GB"),
        _info_row("RAM Used", f"{cpu['ram_used']:.1f} GB  ({cpu['ram_percent']:.0f}%)"),
        _info_row("RAM Free", f"{cpu['ram_free']:.1f} GB"),
        _bar_html(cpu["ram_percent"]),
        f'<div style="margin-top:8px;font-size:11px;color:#475569;">'
        f'⚙️ GGUF CPU mode: all layers on CPU (n_gpu_layers=0), threads={cpu["logical"]}'
        f'</div>',
    ])
    return _card("CPU Information", "🖥️", body)


def get_gpu_info_html() -> str:
    """Return an HTML card (or cards) displaying GPU specifications."""
    gpus = get_gpu_info()
    if not gpus:
        return _card(
            "GPU Information", "⚡",
            '<div style="color:#94a3b8;font-size:12px;">No CUDA GPU detected on this machine.</div>'
        )
    cards = []
    for g in gpus:
        body = "".join([
            _info_row(f"GPU {g['index']}", g["name"]),
            _info_row("VRAM Total", f"{g['vram_total']:.1f} GB"),
            _info_row("VRAM Used", f"{g['vram_used']:.1f} GB  ({g['vram_percent']:.0f}%)"),
            _info_row("VRAM Free", f"{g['vram_free']:.1f} GB"),
            _info_row("CUDA Compute", g["compute"]),
            _bar_html(g["vram_percent"]),
        ])
        cards.append(_card(f"GPU {g['index']} — {g['name']}", "⚡", body))
    return "".join(cards)


def get_system_info_html(device_choice: str) -> str:
    """
    Return the appropriate HTML system info panel for the given device_choice.

    Args:
        device_choice: one of 'CPU', 'GPU', 'Auto'

    Returns:
        HTML string for display in gr.HTML()
    """
    choice = (device_choice or "Auto").strip().upper()
    if choice == "CPU":
        return get_cpu_info_html()
    elif choice == "GPU":
        return get_gpu_info_html()
    else:
        # Auto — show both
        cpu_html = get_cpu_info_html()
        gpu_html = get_gpu_info_html()
        return cpu_html + gpu_html
