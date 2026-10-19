"""Pure helpers for the Prompt Video Minimax H3 tab. No gradio/torch imports, so test_h3_prompts.py runs standalone."""
import json
import math
import re
import unicodedata
from datetime import datetime
from pathlib import Path

from .prompts import _format

MAX_SHOT = 12

# ── Shot plans ───────────────────────────────────────────────────────────────
TIMERANGE = re.compile(r"(\d{1,2}):(\d{2})\s*[–—-]\s*(\d{1,2}):(\d{2})")


def _mmss(seconds: int) -> str:
    s = int(round(seconds))
    return f"{s // 60:02d}:{s % 60:02d}"


def clamp_durations(total: int) -> list[int]:
    """One clip ≤12s; 13–15s is compressed to 12s; longer is split into ceil(T/12) integer clips summing to T."""
    total = int(round(total))
    if total <= MAX_SHOT:
        return [total]
    if total <= 15:
        return [MAX_SHOT]
    n = math.ceil(total / MAX_SHOT)
    q, r = divmod(total, n)
    return [q + 1] * r + [q] * (n - r)


def story_segments(story: str) -> list[tuple[int, int]]:
    """Every 'MM:SS–MM:SS' range in the story, in order, as seconds. Empty or reversed ranges are skipped."""
    out = []
    for m in TIMERANGE.finditer(story or ""):
        start, end = int(m[1]) * 60 + int(m[2]), int(m[3]) * 60 + int(m[4])
        if end > start:
            out.append((start, end))
    return out


def build_shot_plan(segments):
    """[(start, end)] → ([(story_start, story_end, duration)], log notes) with the 12s rule applied."""
    plan, notes = [], []
    for a, b in segments:
        pieces = clamp_durations(b - a)
        if len(pieces) == 1:
            plan.append((a, b, pieces[0]))
            if pieces[0] != b - a:
                notes.append(f"story {_mmss(a)}–{_mmss(b)} ({b - a}s) → nén còn {pieces[0]}s")
            continue
        start = a
        for d in pieces:
            plan.append((start, start + d, d))
            start += d
        notes.append(f"story {_mmss(a)}–{_mmss(b)} ({b - a}s) → {len(pieces)} shot: "
                     + " + ".join(f"{d}s" for d in pieces))
    return plan, notes


def shot_plan_from_story(story: str):
    return build_shot_plan(story_segments(story))


def shot_plan_from_total(total):
    total = int(total or 0)
    return build_shot_plan([(0, total)]) if total > 0 else ([], [])


def format_shot_plan(plan) -> str:
    if not plan:
        return "Choose 5 to 12 seconds per shot; never more than 12 seconds."
    lines = []
    for i, (s, e, d) in enumerate(plan, 1):
        line = f"SHOT {i:02d}: story {_mmss(s)}–{_mmss(e)} ({d}s)"
        if e - s != d:
            line += f" — compress the action into {d}s"
        lines.append(line)
    return "\n".join(lines)


# ── Picture references ───────────────────────────────────────────────────────
ROLE_LABELS = {"Face sheet / góc mặt": "face", "Bối cảnh": "scene", "Trang phục / đạo cụ": "outfit"}
ROLE_TEXT = {
    "character": "main character: identity, body, outfit",
    "face": "face reference sheet of the same character as <Picture 1>; use it for face details only",
    "scene": "scene / environment. Ignore any person visible in it",
    "outfit": "outfit / props worn or held by the main character",
}


def build_references(slots):
    """slots[0] = Picture 1 role ('character'); later items are extra-slot roles or None when empty."""
    refs = []
    for slot, role in enumerate(slots):
        if role:
            n = len(refs) + 1
            refs.append({"n": n, "ref": f"ref_image_{n - 1}", "role": role, "slot": slot})
    return refs


def describe_references(refs) -> str:
    has_scene = any(r["role"] == "scene" for r in refs)
    lines = []
    for r in refs:
        text = ROLE_TEXT[r["role"]]
        if r["role"] == "character" and not has_scene:
            text += ". Also the setting, because no scene picture is given"
        lines.append(f"<Picture {r['n']}>, {r['ref']} = {text}.")
    return "\n".join(lines)


BIBLE_FIELDS = ("NAME_HINT", "AGE_APPEARANCE", "FACE", "HAIR", "BODY", "OUTFIT", "ACCESSORIES",
                "SETTING", "LIGHTING", "CAMERA_FRAMING", "STYLE")


def bible_field_count(bible: str) -> int:
    return sum(1 for line in (bible or "").splitlines() if line.split(":", 1)[0].strip() in BIBLE_FIELDS)
