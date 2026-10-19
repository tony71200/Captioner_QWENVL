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
PART1_FIELDS = ("subject_definitions", "summary", "retention_analysis")
TIMELINE = re.compile(r"^[ \t]*(\d{1,2}):(\d{2})(?:\.\d+)?\s*[–—-]\s*(\d{1,2}):(\d{2})(?:\.\d+)?\s*[—–-]?\s*(.*)$")
# The 4B model often copies "<short label>" placeholders verbatim; only <Subject N> / <Picture N> keep brackets.
COPIED_PLACEHOLDER = re.compile(r"<(?!(?:Subject|Picture)\s*\d+>)([^<>\n]{2,80})>")
REPEAT_FILLER = re.compile(r"[ \t]*\((?:repeated|repeats?)\b[^)\n]*\)", re.I)  # "(repeated once)", "(repeated 2x)"


def _add_tail(body, tag, warnings):
    for field, default in SHOT_TAIL_DEFAULTS:
        if not re.search(rf"(?mi)^{field}:", body):
            body += f"\n{field}:\n{default}"
            warnings.append(f"{tag}: thêm {field} mặc định")
    return body


def _fix_timeline(body, duration, tag, warnings):
    """Drop timeline lines starting at/after the shot end and make the last one end exactly at the duration."""
    lines = body.splitlines()
    marks = []  # (line index, start, end, text)
    for i, line in enumerate(lines):
        m = TIMELINE.match(line)
        if m:
            marks.append((i, int(m[1]) * 60 + int(m[2]), int(m[3]) * 60 + int(m[4]), m[5].strip()))
    if not marks:
        return body
    offset = marks[0][1]
    if offset > 0:  # the model kept the story's global timecodes
        warnings.append(f"{tag}: timeline toàn cục, dời về 00:00 (−{offset}s)")
        marks = [(i, a - offset, b - offset, t) for i, a, b, t in marks]
    drop = set()
    merged = [list(marks[0])]
    for i, a, b, t in marks[1:]:
        if (a, b) == tuple(merged[-1][1:3]):  # two descriptions of the same beat → one line
            merged[-1][3] += " " + t
            drop.add(i)
            warnings.append(f"{tag}: gộp dòng timeline trùng {_mmss(a)}–{_mmss(b)}")
        else:
            merged.append([i, a, b, t])
    late = [m for m in merged if m[1] >= duration]
    if late:
        warnings.append(f"{tag}: bỏ dòng timeline từ {_mmss(late[0][1])} (shot {duration}s)")
        drop.update(m[0] for m in late)
    kept = [m for m in merged if m[1] < duration]
    if kept and kept[-1][2] != duration:
        warnings.append(f"{tag}: timeline kết thúc {_mmss(kept[-1][2])} → {_mmss(duration)}")
        kept[-1][2] = duration
    for i, a, b, t in kept:
        lines[i] = f"{_mmss(a)}–{_mmss(b)} — {t}"
    return "\n".join(line for i, line in enumerate(lines) if i not in drop)


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
    return COPIED_PLACEHOLDER.sub(r"\1", head).strip(), COPIED_PLACEHOLDER.sub(r"\1", text[m.start():]).strip()


def check_part1(part1: str) -> tuple[str, list[str]]:
    """Warn on missing §10.2 fields and add the soundscape/music tail when the model skipped it."""
    warnings = [f"Part 1: thiếu {f}" for f in PART1_FIELDS if not re.search(rf"(?mi)^{f}:", part1)]
    summary = re.search(r"(?mi)^summary:[ \t]*(?:\n[ \t]*)?", part1)
    if summary and not part1[summary.end():].lower().startswith("[reference generation]"):
        part1 = part1[:summary.end()] + "[reference generation] " + part1[summary.end():]
        warnings.append("Part 1: thêm [reference generation]")
    return _add_tail(part1.strip(), "Part 1", warnings).strip(), warnings


def check_shots(part2: str) -> tuple[str, list[str]]:
    """Renumber SHOT 01..NN, cap DURATION at 12s, fit each timeline to its duration, add a missing sound tail."""
    heads = list(SHOT_LINE.finditer(part2))
    out, warnings = [], []
    for i, m in enumerate(heads, 1):
        end = heads[i].start() if i < len(heads) else len(part2)
        tag = f"SHOT {i:02d}"
        header = re.sub(r"(?i)SHOT\s*0*\d+", tag, m.group(0).strip(), count=1)
        d = DURATION_RE.search(header)
        duration = None
        if not d:
            warnings.append(f"{tag}: thiếu DURATION")
        else:
            duration = int(round(float(d[1].replace(",", "."))))
            if duration > MAX_SHOT:
                warnings.append(f"{tag}: DURATION {d[1]}s → {MAX_SHOT}s")
                header = header[:d.start(1)] + str(MAX_SHOT) + header[d.end(1):]
                duration = MAX_SHOT
        body = REPEAT_FILLER.sub("", part2[m.end():end]).strip()
        if duration:
            body = _fix_timeline(body, duration, tag, warnings)
        out.append(f"{header}\n{_add_tail(body, tag, warnings).strip()}")
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
                      "shots": _merge_short([p for s in shots for p in clamp_durations(s)])})
    return ideas


MIN_SHOT = 5


def _merge_short(shots: list[int]) -> list[int]:
    """Fold shots under 5s into a neighbour while the sum stays ≤12s (the 4B model likes 2-3s shots)."""
    out = []
    for s in shots:
        if out and (s < MIN_SHOT or out[-1] < MIN_SHOT) and out[-1] + s <= MAX_SHOT:
            out[-1] += s
        else:
            out.append(s)
    return out


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


# ── Prompt store (captioner/system_prompt.json) ──────────────────────────────
H3_PROMPT_FILE = Path(__file__).resolve().parent / "system_prompt.json"
H3_KEYS = ("bible", "ideas_sfw", "ideas_nsfw", "expand_sfw", "expand_nsfw", "shots_sfw", "shots_nsfw")
H3_FIELDS = ("system", "user", "temperature", "max_tokens")


def step_key(step: str, nsfw: bool) -> str:
    return step if step == "bible" else f"{step}_{'nsfw' if nsfw else 'sfw'}"


def load_h3_prompts() -> dict:
    """Read the file on every call so ⚙ Save and hand edits apply without a restart."""
    try:
        data = json.loads(H3_PROMPT_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ValueError(f"{H3_PROMPT_FILE.name}: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"{H3_PROMPT_FILE.name}: top level must be an object")
    return data


def get_h3_step(step: str, nsfw: bool) -> dict:
    key = step_key(step, nsfw)
    cfg = load_h3_prompts().get(key)
    if not isinstance(cfg, dict) or any(f not in cfg for f in H3_FIELDS):
        raise ValueError(f"{H3_PROMPT_FILE.name}: '{key}' needs {', '.join(H3_FIELDS)}")
    return cfg


def save_h3_system(step: str, nsfw: bool, text: str) -> None:
    data = load_h3_prompts()
    get_h3_step(step, nsfw)  # validates the key before writing
    data[step_key(step, nsfw)]["system"] = (text or "").strip()
    H3_PROMPT_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def render_user(template: str, **fields) -> str:
    return _format(template, **fields)


# ── Output files ─────────────────────────────────────────────────────────────
def clean_path(text) -> str:
    """Accept Explorer's 'Copy as path' (quoted) input."""
    return (text or "").strip().strip('"').strip()


def h3_output_path(picture1_path, nsfw: bool, ideas: bool = False, now=None) -> Path:
    preset = "nsfw" if nsfw else "sfw"
    suffix = "_ideas" if ideas else ""
    path = clean_path(picture1_path)
    if path and Path(path).is_file():
        p = Path(path)
        return p.with_name(f"{p.stem}_H3_{preset}{suffix}.txt")
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return Path.home() / "Desktop" / f"h3_{stamp}_{preset}{suffix}.txt"


def build_h3_file(part1: str, part2: str) -> str:
    return (f"CHARACTER & SCENE DEFINITION\n\n{(part1 or '').strip()}\n\n"
            f"SHOT STRUCTURE — MINIMAX H3\n\n{(part2 or '').strip()}\n")
