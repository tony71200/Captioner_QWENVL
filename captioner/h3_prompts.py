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


# ── Model output ─────────────────────────────────────────────────────────────
PART_TITLES = re.compile(
    r"(?mi)^[ \t]*(?:PART\s*\d+\s*[—–:-]\s*)?"
    r"(?:CHARACTER\s*&\s*SCENE\s*DEFINITION|SHOT\s*STRUCTURE\s*[—–-]+\s*MINIMAX\s*H3)[ \t]*:?[ \t]*$\n?")
FIRST_SHOT = re.compile(r"(?m)^[ \t]*\[\s*SHOT\b")
SHOT_LINE = re.compile(r"(?mi)^[ \t]*\[\s*SHOT\s*0*\d+[^\n]*$")
DURATION_RE = re.compile(r"(?i)DURATION\s*:\s*(\d+(?:[.,]\d+)?)")
SHOT_TAIL_DEFAULTS = (("overall_soundscape", "Natural ambient sound, no dialogue."),
                      ("non_diegetic_music", "NONE"))


def clean_output(text: str) -> str:
    """NFC, no code fences, no '**', no leading '#' heading marks."""
    text = unicodedata.normalize("NFC", text or "").replace("**", "")
    text = "\n".join(line for line in text.splitlines() if not line.strip().startswith("```"))
    return re.sub(r"(?m)^[ \t]*#+[ \t]*", "", text).strip()


def split_h3_output(text: str) -> tuple[str, str]:
    """→ (CHARACTER & SCENE DEFINITION body, SHOT STRUCTURE body). Preamble and part titles are dropped."""
    text = clean_output(text)
    m = FIRST_SHOT.search(text)
    if not m:
        raise ValueError("no [SHOT ...] block")
    head = PART_TITLES.sub("", text[:m.start()])
    start = re.search(r"(?mi)^subject_definitions:", head)
    if start:
        head = head[start.start():]
    return head.strip(), text[m.start():].strip()


def check_shots(part2: str) -> tuple[str, list[str]]:
    """Renumber SHOT 01..NN, cap DURATION at 12s, add a missing soundscape/music tail to each shot."""
    heads = list(SHOT_LINE.finditer(part2))
    out, warnings = [], []
    for i, m in enumerate(heads, 1):
        end = heads[i].start() if i < len(heads) else len(part2)
        header = re.sub(r"(?i)SHOT\s*0*\d+", f"SHOT {i:02d}", m.group(0).strip(), count=1)
        d = DURATION_RE.search(header)
        if not d:
            warnings.append(f"SHOT {i:02d}: thiếu DURATION")
        elif float(d[1].replace(",", ".")) > MAX_SHOT:
            warnings.append(f"SHOT {i:02d}: DURATION {d[1]}s → {MAX_SHOT}s")
            header = header[:d.start(1)] + str(MAX_SHOT) + header[d.end(1):]
        body = part2[m.end():end].strip()
        for field, default in SHOT_TAIL_DEFAULTS:
            if not re.search(rf"(?mi)^{field}:", body):
                body += f"\n{field}:\n{default}"
                warnings.append(f"SHOT {i:02d}: thêm {field} mặc định")
        out.append(f"{header}\n{body.strip()}")
    return "\n\n".join(out), warnings


# ── Ideas ────────────────────────────────────────────────────────────────────
IDEA_SPLIT = re.compile(r"(?mi)^[ \t>]*Ý\s*TƯỞNG\s*\d+\s*[:.\-–]?[ \t]*")


def _field(block: str, name: str) -> str:
    m = re.search(rf"(?mi)^[ \t-]*{name}\s*:\s*(.+)$", block)
    return m.group(1).strip() if m else ""


def _seconds(text: str) -> list[int]:
    return [int(round(float(x.replace(",", ".")))) for x in re.findall(r"(\d+(?:[.,]\d+)?)\s*s\b", text)]


def parse_h3_ideas(text: str, limit: int = 3) -> list[dict]:
    """'Ý TƯỞNG N' blocks → [{name, logline, shots}]. Shots >12s are split; a missing shot list falls back to the total."""
    ideas = []
    for part in IDEA_SPLIT.split(clean_output(text))[1:limit + 1]:
        name, _, rest = part.partition("\n")
        shots = _seconds(_field(rest, "Thời lượng shot"))
        if not shots:
            total = _seconds(_field(rest, "Tổng thời lượng"))
            shots = [total[0] if total else MAX_SHOT]
        ideas.append({"name": name.strip(), "logline": _field(rest, "Logline"),
                      "shots": [p for s in shots for p in clamp_durations(s)]})
    return ideas


def _shots_text(idea) -> str:
    return " + ".join(f"{s}s" for s in idea["shots"])


def idea_label(n: int, idea: dict) -> str:
    return (f"Ý tưởng {n}: {idea['name']} — {idea['logline']} · {sum(idea['shots'])}s · "
            f"{len(idea['shots'])} shot: {_shots_text(idea)}")


def idea_text(n: int, idea: dict) -> str:
    return (f"Ý TƯỞNG {n}: {idea['name']}\nLogline: {idea['logline']}\n"
            f"Tổng thời lượng: {sum(idea['shots'])}s\nSố shot: {len(idea['shots'])}\n"
            f"Thời lượng shot: {_shots_text(idea)}")


def idea_plan(idea: dict):
    segments, start = [], 0
    for s in idea["shots"]:
        segments.append((start, start + s))
        start += s
    return build_shot_plan(segments)[0]
