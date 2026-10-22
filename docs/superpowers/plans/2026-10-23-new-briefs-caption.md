# New Briefs Caption Template Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the "New Briefs Caption" prompt template (LoRA training captions for men's underwear cuts, trigger once inside the garment phrase, no cut words, no brand text), make `batch_caption.py` able to pass the trigger word, and run it on the 66 uncaptioned images of the briefs dataset.

**Architecture:** One new entry in `PROMPT_TEMPLATES` reuses the existing `name_field` as the trigger field and `training_caption: True` for the one-line output contract. A new `{garment}` placeholder is computed by `brief_garment()` inside `resolve_prompt` so the same template serves trigger classes (`tnga_brief underwear`) and reference images (`classic men's briefs`). `check_briefs_caption()` implements the guide's post-run checker. `batch_caption.py` gains `--name`. The dataset run is a shell loop, one folder per trigger.

**Tech Stack:** Python 3.13 (`.venv\Scripts\python.exe`, llama-cpp-python 0.3.39), Qwen3-VL-8B-Instruct Q4_K_M GGUF, plain `assert` test scripts.

**Spec:** `docs/superpowers/specs/2026-10-23-new-briefs-caption-design.md`

## Global Constraints

- Python: always `D:\001_Personal_Proj\Captioner_QWENVL\.venv\Scripts\python.exe` (`D:\Comfy\.venv` no longer exists on this machine).
- No new dependencies. Do not touch `app.py`, `hf_captioner.py`, `gguf_captioner.py`, `utils/`.
- Template name is exactly `New Briefs Caption`; default trigger `tnga_brief`; reference labels `classic`, `reg`, `none` → `classic men's briefs`.
- Trigger appears exactly once, inside `wearing [color] {garment}`. Target length 50–90 words; checker accepts 40–100.
- Forbidden words (verbatim): `high-cut`, `high cut`, `low-rise`, `low rise`, `mid-rise`, `cheeky`, `narrow side`, `string side`, `side width`, `leg opening`, `back coverage`, `thong`, `tanga`, `hipster`, `bikini`, `bulge`, `sexy`, `crotch`, `genital`, `penis`, `attentionwear`, `attent!on`, `do_me`, `do-me`.
- `training_caption: True` → `TRAINING_OUTPUT_RULES` appended, never `PEOPLE: <n>`.
- Dataset: `C:\Users\tonyn\Downloads\briefs_lora_dataset_captions\briefs_lora_dataset\dataset\` — only images with no sibling `.txt` get captioned; `.txt` written beside the image. Scripts that read dataset files run with `python -I`.
- Tests are plain `assert` scripts with a `__main__` runner, same style as `test_h3_prompts.py`.
- Commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## File Structure

| File | Responsibility |
|------|----------------|
| `captioner/prompts.py` | New template entry; `BRIEF_REG_LABELS`, `BRIEF_REG_GARMENT`, `BRIEF_FORBIDDEN`; `brief_garment()`, `check_briefs_caption()`; `{garment}` in `resolve_prompt`. |
| `batch_caption.py` | `--name` flag → `build_prompt(template, prefix, name)`; `self_test` uses the `training_caption` flag; usage example. |
| `test_briefs_prompts.py` (new) | Asserts for garment phrase, template resolution and the checker. |
| `README.md` | One batch example for the new template. |

---

### Task 1: Template entry, `brief_garment`, `{garment}` placeholder

**Files:**
- Modify: `captioner/prompts.py:234` (end of `PROMPT_TEMPLATES`), `:331-333` (after `UNDERWEAR_NAMES`), `:446-460` (`resolve_prompt`)
- Test: `test_briefs_prompts.py` (create)

**Interfaces:**
- Produces:
  - `BRIEF_REG_LABELS: frozenset[str]`, `BRIEF_REG_GARMENT: str = "classic men's briefs"`
  - `brief_garment(trigger: str) -> str`
  - `resolve_prompt(...)` unchanged signature; fills `{garment}` in system and user prompt.

- [ ] **Step 1: Write the failing tests**

Create `test_briefs_prompts.py`:

```python
"""Self-checks for the New Briefs Caption template. Run: .venv\\Scripts\\python.exe test_briefs_prompts.py"""
from captioner.prompts import (
    BRIEF_REG_GARMENT,
    TRAINING_OUTPUT_RULES,
    brief_garment,
    get_prompt_by_name,
    resolve_prompt,
)

TEMPLATE = "New Briefs Caption"


def test_brief_garment():
    assert brief_garment("tnga_brief") == "tnga_brief underwear"
    assert brief_garment("  strng_bikini ") == "strng_bikini underwear"
    for label in ("classic", " Reg ", "none", "CLASSIC"):
        assert brief_garment(label) == BRIEF_REG_GARMENT, label


def test_template_resolves_trigger():
    tmpl = get_prompt_by_name(TEMPLATE)
    assert tmpl and tmpl["name_field"] and tmpl["training_caption"]
    assert tmpl["default_name"] == "tnga_brief"

    system, user = resolve_prompt(TEMPLATE, "", "tnga_brief")
    for text in (system, user):
        assert "wearing [color] tnga_brief underwear" in text, text
        assert "{garment}" not in text and "{name}" not in text, text
    assert user.endswith(TRAINING_OUTPUT_RULES), user[-200:]
    assert "PEOPLE: <n>" not in user

    # blank field -> default trigger; reference label -> reference phrase, no trigger token
    assert "tnga_brief underwear" in resolve_prompt(TEMPLATE, "")[1]
    system, user = resolve_prompt(TEMPLATE, "", "classic")
    assert "wearing [color] classic men's briefs" in user and "classic men's briefs" in system
    assert "_brief underwear" not in user

    # a custom prompt may still use {garment}
    assert resolve_prompt(TEMPLATE, "Say {garment}.", "hpstr_brief")[1].startswith("Say hpstr_brief underwear.")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("test_briefs_prompts OK")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv\Scripts\python.exe test_briefs_prompts.py`
Expected: `ImportError: cannot import name 'BRIEF_REG_GARMENT'`

- [ ] **Step 3: Add the template entry**

In `captioner/prompts.py`, replace the closing `    }]` of `PROMPT_TEMPLATES` (line 234, right after the "Train Lora Prompt (Following ChatGPT)" entry) with:

```python
    },
    {
        "name": "New Briefs Caption",
        "description": (
            "Caption train LoRA kiểu cắt quần lót nam (Krea 2 / Qwen3-VL): trigger đúng 1 lần trong cụm "
            "wearing, không mô tả kiểu cắt, không chép chữ brand. Gõ 'classic' cho ảnh đối chứng."
        ),
        "name_field": True,
        "name_label": "Brief trigger word (classic = reference image)",
        "default_name": "tnga_brief",
        "training_caption": True,
        "system_prompt": (
            "You caption training images for a LoRA that teaches men's underwear cuts. The garment in "
            "this image is labelled '{garment}'. The label alone carries the cut, so your words must carry "
            "everything else: the person, the pose, the framing, the color, the fabric and the waistband. "
            "Hard rules: write the label exactly once, verbatim, inside the phrase 'wearing [color] {garment}'. "
            "Never describe the cut: no rise, leg opening, side width or back coverage, and no cut words such "
            "as high-cut, low-rise, cheeky, narrow sides, string sides, thong, tanga, hipster or bikini outside "
            "the label itself. Never transcribe brand names, logo text, slogans or overlaid marketing text; "
            "call them 'logo waistband', 'small printed text', 'small captions' or 'headline text'. No sexual "
            "wording, never mention a bulge or anything under the garment, and never invent details that are "
            "not clearly visible. Output only the caption paragraph."
        ),
        "user_prompt": (
            "Describe this image as ONE flowing English paragraph of 50 to 90 words, in this order: "
            "1) Framing: 'Full-body shot', 'Medium shot from the knees up', 'Medium shot from the waist up' or "
            "'Cropped torso shot from chest to mid-thigh, head out of frame'; always say 'head out of frame' "
            "when the head is cut off. "
            "2) Camera angle: eye level, slightly low angle, high angle, side view, from behind, or mirror selfie. "
            "3) The person: 'an adult man' plus build (slim, lean, athletic, muscular, stocky), skin tone, hair "
            "style and color, facial hair and visible tattoos; skip whatever is not visible. "
            "4) Pose and action: body orientation (facing the camera, three-quarter view, profile, seen from "
            "behind), arms, hands, legs, what he holds or leans on. "
            "5) Face direction and expression: gaze, head turn, expression; skip when the face is out of frame. "
            "6) Garment: 'wearing [color] {garment}' then pouch structure (contoured, seamed, two-layer), fabric "
            "or texture (smooth matte, satin sheen, fine mesh, embossed, ribbed) and waistband details (width, "
            "color, metallic or contrast trim, logo waistband, small side buckle); never the cut. "
            "7) Other visible clothing or accessories (hat, socks, watch); skip if none. "
            "8) Setting: background, surfaces, props; mention overlaid text or product insets only generically. "
            "9) Lighting: direction and quality. "
            "Write flowing sentences, not a numbered list, and output only the caption."
        ),
    }]
```

- [ ] **Step 4: Add the constants and `brief_garment`**

In `captioner/prompts.py`, right after `UNDERWEAR_NAMES = list(UNDERWEAR_OPTIONS)` (line 331) insert:

```python

# "New Briefs Caption": the trigger field holds a cut token such as tnga_brief, or one
# of BRIEF_REG_LABELS for a reference image that must carry no trigger at all.
BRIEF_REG_LABELS = frozenset({"classic", "reg", "none"})
BRIEF_REG_GARMENT = "classic men's briefs"


def brief_garment(trigger: str) -> str:
    """Garment phrase for the {garment} placeholder: '<trigger> underwear' or the reference phrase."""
    trigger = (trigger or "").strip()
    if trigger.lower() in BRIEF_REG_LABELS:
        return BRIEF_REG_GARMENT
    return f"{trigger} underwear"
```

- [ ] **Step 5: Fill `{garment}` in `resolve_prompt`**

In `captioner/prompts.py` `resolve_prompt` (lines 446-460), after `name = _resolve_subject_name(template_name, subject_name)` add `garment = brief_garment(name)`, and pass `garment=garment` to all three `_format` calls:

```python
    name = _resolve_subject_name(template_name, subject_name)
    garment = brief_garment(name)
    underwear = UNDERWEAR_OPTIONS.get(underwear, UNDERWEAR_OPTIONS["Normal"])

    if custom_prompt:
        system_prompt = _format(tmpl.get("system_prompt", "") if tmpl else "", name=name, garment=garment)
        return system_prompt, _with_output_rules(
            _format(custom_prompt, name=name, underwear=underwear, garment=garment),
            training_caption=bool(tmpl and tmpl.get("training_caption")),
        )

    if template_name == "Custom" or not tmpl:
        return "", _with_output_rules("Describe this image.")

    system_prompt = _format(tmpl.get("system_prompt", ""), name=name, garment=garment)
    user_prompt = _format(
        tmpl.get("user_prompt", tmpl.get("prompt_text", "")), name=name, underwear=underwear, garment=garment
    )
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv\Scripts\python.exe test_briefs_prompts.py`
Expected: `ok test_brief_garment`, `ok test_template_resolves_trigger`, `test_briefs_prompts OK`

Also run: `.venv\Scripts\python.exe test_video_prompts.py` and `.venv\Scripts\python.exe test_h3_prompts.py`
Expected: both still pass (the extra `garment` kwarg is ignored by templates that do not use it).

- [ ] **Step 7: Commit**

```bash
git add captioner/prompts.py test_briefs_prompts.py
git commit -m "Add the New Briefs Caption template with a {garment} trigger phrase

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: `check_briefs_caption` (the guide's post-run checker)

**Files:**
- Modify: `captioner/prompts.py` (after `brief_garment`)
- Test: `test_briefs_prompts.py`

**Interfaces:**
- Consumes: `brief_garment`, `BRIEF_REG_GARMENT` from Task 1.
- Produces: `BRIEF_FORBIDDEN: tuple[str, ...]`, `check_briefs_caption(text: str, trigger: str) -> list[str]` — problem strings `multiline`, `trigger x<n>`, `stray trigger <tok>`, `forbidden: <word>`, `words <n>`; `[]` means pass.

- [ ] **Step 1: Write the failing tests**

Append to `test_briefs_prompts.py` (above the `__main__` block) and extend the import:

```python
from captioner.prompts import BRIEF_FORBIDDEN, check_briefs_caption  # add to the import list

GUIDE_OK = {
    "tnga_brief": (
        "A photo of a lean East Asian man in his late twenties with short black hair and a relaxed half "
        "smile, standing in three-quarter view with his left hand resting on his hip and his weight on the "
        "right leg, looking directly at the camera. He is wearing white tnga_brief underwear with a thin "
        "silver-trimmed waistband. Plain light-grey studio backdrop, soft key light from the left, full-body "
        "shot from a slightly low angle."
    ),
    "dblstrap_bikini": (
        "A muscular man with a trimmed beard and curly brown hair, seen from behind and turning his head over "
        "his right shoulder toward the camera, arms relaxed. He is wearing sage-green dblstrap_bikini "
        "underwear with a narrow printed waistband. Warm indoor light, beige wall background, medium shot "
        "from the knees up."
    ),
    "strng_bikini": (
        "A cropped torso shot, head out of frame, of a man standing straight with both thumbs hooked in the "
        "waistband. He is wearing black strng_bikini underwear made of fine mesh with a small metal buckle on "
        "the side. White seamless background, even studio lighting, close-up from chest to mid-thigh."
    ),
    "classic": (
        "A photo of a slim athletic East Asian man with styled black hair, kneeling with one arm raised behind "
        "his head, looking at the camera. He is wearing black classic men's briefs with a wide white contrast "
        "waistband. Soft white-pink background with a large black headline logo and small captions, soft even "
        "studio lighting. Full-body shot."
    ),
}
GUIDE_BAD = "tnga_brief, sexy, bulge, high cut, white, man"


def test_check_briefs_caption():
    for trigger, text in GUIDE_OK.items():
        assert check_briefs_caption(text, trigger) == [], (trigger, check_briefs_caption(text, trigger))

    bad = check_briefs_caption(GUIDE_BAD, "tnga_brief")
    assert set(bad) == {"forbidden: sexy", "forbidden: bulge", "forbidden: high cut", "words 7"}, bad

    ok = GUIDE_OK["tnga_brief"]
    assert check_briefs_caption(ok + " Another tnga_brief here.", "tnga_brief") == ["trigger x2"]
    assert check_briefs_caption(ok.replace("tnga_brief", "briefs"), "tnga_brief") == ["trigger x0"]
    assert check_briefs_caption(ok.replace(". He is", ".\nHe is"), "tnga_brief") == ["multiline"]

    reg = GUIDE_OK["classic"].replace("black classic men's briefs", "black tnga_brief underwear")
    assert check_briefs_caption(reg, "classic") == ["trigger x0", "stray trigger tnga_brief"]

    # the trigger itself is stripped before the forbidden-word scan
    assert "bikini" in BRIEF_FORBIDDEN
    assert check_briefs_caption(GUIDE_OK["strng_bikini"], "strng_bikini") == []
    assert check_briefs_caption(GUIDE_OK["strng_bikini"] + " A bikini.", "strng_bikini") == ["forbidden: bikini"]
    # brand marks are caught however they are cased
    assert check_briefs_caption(ok.replace("logo", "ATTENT!ON").replace("thin silver-trimmed", "thin ATTENT!ON"), "tnga_brief") == ["forbidden: attent!on"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv\Scripts\python.exe test_briefs_prompts.py`
Expected: `ImportError: cannot import name 'BRIEF_FORBIDDEN'`

- [ ] **Step 3: Implement the checker**

In `captioner/prompts.py`, directly after `brief_garment` add:

```python
# Words the guide forbids in a briefs caption: cut descriptions the trigger must learn,
# sexual wording and brand marks. Matched case-insensitively on word boundaries, after
# the trigger token itself has been removed (strng_bikini must not trip "bikini").
BRIEF_FORBIDDEN = (
    "high-cut", "high cut", "low-rise", "low rise", "mid-rise", "cheeky", "narrow side",
    "string side", "side width", "leg opening", "back coverage", "thong", "tanga", "hipster",
    "bikini", "bulge", "sexy", "crotch", "genital", "penis", "attentionwear", "attent!on",
    "do_me", "do-me",
)


def check_briefs_caption(text: str, trigger: str) -> list[str]:
    """Problems with a briefs LoRA caption; [] means it passes the guide's checks."""
    problems: list[str] = []
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    if len(lines) != 1:
        problems.append("multiline")
    body = " ".join(lines)

    label = brief_garment(trigger)
    is_reg = label == BRIEF_REG_GARMENT
    token = label if is_reg else (trigger or "").strip()
    token_re = re.compile(rf"\b{re.escape(token)}\b", re.IGNORECASE)
    hits = len(token_re.findall(body))
    if hits != 1:
        problems.append(f"trigger x{hits}")
    stripped = token_re.sub(" ", body)
    if is_reg:
        problems += [f"stray trigger {t}" for t in re.findall(r"\b\w+_(?:brief|bikini)\b", stripped)]
    for word in BRIEF_FORBIDDEN:
        if re.search(rf"\b{re.escape(word)}\b", stripped, re.IGNORECASE):
            problems.append(f"forbidden: {word}")

    words = len(body.split())
    if not 40 <= words <= 100:
        problems.append(f"words {words}")
    return problems
```

Note: the `GUIDE_BAD` caption has 7 whitespace-separated words (`high cut` is two), so the expected problem is `words 7`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv\Scripts\python.exe test_briefs_prompts.py`
Expected: `ok test_check_briefs_caption` plus the two earlier tests, `test_briefs_prompts OK`

- [ ] **Step 5: Commit**

```bash
git add captioner/prompts.py test_briefs_prompts.py
git commit -m "Add check_briefs_caption: trigger once, no cut words, no brand text, 40-100 words

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: `batch_caption.py --name`, self-test fix, README

**Files:**
- Modify: `batch_caption.py:11` (usage), `:59-63` (`build_prompt`), `:127` (call), `:196` (import), `:224-231` (self-test loop), `:240` (extra asserts), `:255-256` (parser)
- Modify: `README.md:198-199` (batch examples)

**Interfaces:**
- Consumes: `resolve_prompt(template, custom_prompt, subject_name)` and `get_prompt_by_name` from `captioner.prompts`.
- Produces: `build_prompt(template: str, prefix: str, name: str = "") -> tuple[str, str]`; CLI flag `--name`.

- [ ] **Step 1: Extend the self-test (it is the failing test)**

In `batch_caption.py` `self_test`, change the import (line 196) to:

```python
    from captioner.prompts import (
        NEGATIVE_PROMPT, get_prompt_by_name, get_prompt_names, resolve_prompt,
    )
```

Replace the loop at lines 224-231 with:

```python
    # Normal templates retain people counting, while LoRA training captions
    # deliberately omit it so no negative-prompt text can enter training data.
    for tname in get_prompt_names():
        rules = resolve_prompt(tname, "")[1]
        assert "empty lines" in rules, tname
        if get_prompt_by_name(tname).get("training_caption"):
            assert "PEOPLE: <n>" not in rules, tname
            assert "Do not output a PEOPLE marker or a negative prompt" in rules, tname
        else:
            assert "PEOPLE: <n>" in rules, tname
```

Before `assert pick_mmproj(DEFAULT_MODEL)...` (line 240) add:

```python
    # --name reaches the template's trigger field; 'classic' means a reference image.
    briefs = "New Briefs Caption"
    _, up = build_prompt(briefs, "", "tnga_brief")
    assert "wearing [color] tnga_brief underwear" in up and "{garment}" not in up, up[:200]
    assert "classic men's briefs" in build_prompt(briefs, "", "classic")[1]
    assert "Begin the description" not in up  # empty prefix adds nothing
```

- [ ] **Step 2: Run the self-test to verify it fails**

Run: `.venv\Scripts\python.exe batch_caption.py --self-test`
Expected: `TypeError: build_prompt() takes 2 positional arguments but 3 were given`

- [ ] **Step 3: Add `--name` and thread it through**

`build_prompt` (lines 59-63):

```python
def build_prompt(template: str, prefix: str, name: str = "") -> tuple[str, str]:
    """Resolve the template (with its trigger / subject name), then pin the opening words onto the user prompt."""
    from captioner.prompts import resolve_prompt

    system_prompt, user_prompt = resolve_prompt(template, "", name)
```

Call site (line 127):

```python
    system_prompt, user_prompt = build_prompt(args.template, args.prefix, args.name)
```

Parser, after `--template` (line 256):

```python
    p.add_argument("--name", default="",
                   help="Trigger word / subject name for templates with a name field (blank = template default)")
```

Usage docstring, after line 11:

```python
    python batch_caption.py --images D:\\briefs\\01_tnga_brief --template "New Briefs Caption" --name tnga_brief --prefix ""
```

- [ ] **Step 4: Run the self-test and the other test scripts**

Run: `.venv\Scripts\python.exe batch_caption.py --self-test`
Expected: `self-test OK`

Run: `.venv\Scripts\python.exe test_briefs_prompts.py`, `test_video_prompts.py`, `test_h3_prompts.py`
Expected: all pass.

- [ ] **Step 5: README example**

In `README.md`, after the "# Different opening phrase and template" example (lines 198-199) add:

```bash
# LoRA training captions for briefs cuts: one trigger word per folder, no opening phrase
python batch_caption.py --images D:/briefs/01_tnga_brief --template "New Briefs Caption" --name tnga_brief --prefix ""
```

And after the paragraph that ends "...resumes where it stopped." add:

```markdown
`--name` fills the template's name / trigger field (templates such as "Description Character"
or "New Briefs Caption"). Training-caption templates write one line and never add the
`PEOPLE` marker or negative prompt.
```

- [ ] **Step 6: Commit**

```bash
git add batch_caption.py README.md
git commit -m "batch_caption: --name passes the trigger word; self-test keys on training_caption

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Dataset prep — retire `sporthc_brief`

**Files:**
- Modify (outside the repo, no commit): `C:\Users\tonyn\Downloads\briefs_lora_dataset_captions\briefs_lora_dataset\dataset\01_tnga_brief\sporthc_brief_*.txt` (10 files)

- [ ] **Step 1: Count before**

```bash
cd "/c/Users/tonyn/Downloads/briefs_lora_dataset_captions/briefs_lora_dataset/dataset" && grep -l "sporthc_brief" */*.txt | wc -l
```
Expected: `10`

- [ ] **Step 2: Replace the token**

```bash
cd "/c/Users/tonyn/Downloads/briefs_lora_dataset_captions/briefs_lora_dataset/dataset/01_tnga_brief" && sed -i 's/sporthc_brief/tnga_brief/g' sporthc_brief_*.txt
```

- [ ] **Step 3: Verify**

```bash
cd "/c/Users/tonyn/Downloads/briefs_lora_dataset_captions/briefs_lora_dataset/dataset" && grep -l "sporthc_brief" */*.txt | wc -l; grep -c "tnga_brief" 01_tnga_brief/sporthc_brief_air-move-black-sport-brief_00.txt
```
Expected: `0` then `1`

---

### Task 5: Trial run on the 66 uncaptioned images

**Files:**
- Create (scratchpad): `trial_targets.txt`, `trial_run.log`, `briefs_report.py`
- Writes: one `.txt` beside each uncaptioned image in folders `01, 02, 03, 05, 07`

**Interfaces:**
- Consumes: `batch_caption.py --name` (Task 3), `check_briefs_caption` (Task 2).

- [ ] **Step 1: Record the targets (the manifest of files the run will create)**

```bash
SCR="C:/Users/tonyn/AppData/Local/Temp/claude/D--001-Personal-Proj-Captioner-QWENVL/fc0c8abd-8e15-436f-a20c-1bd63887596f/scratchpad"
DS="C:/Users/tonyn/Downloads/briefs_lora_dataset_captions/briefs_lora_dataset/dataset"
cd "$DS" && find . -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' \) | while IFS= read -r f; do b="${f%.*}"; [ -f "$b.txt" ] || echo "$DS/${f#./}"; done | sort > "$SCR/trial_targets.txt" && wc -l < "$SCR/trial_targets.txt"
```
Expected: `66`

- [ ] **Step 2: Run the batch, one folder per trigger, in the background**

```bash
SCR="C:/Users/tonyn/AppData/Local/Temp/claude/D--001-Personal-Proj-Captioner-QWENVL/fc0c8abd-8e15-436f-a20c-1bd63887596f/scratchpad"
DS="C:/Users/tonyn/Downloads/briefs_lora_dataset_captions/briefs_lora_dataset/dataset"
PY="D:/001_Personal_Proj/Captioner_QWENVL/.venv/Scripts/python.exe"
cd /d/001_Personal_Proj/Captioner_QWENVL
for pair in 01_tnga_brief:tnga_brief 02_strng_bikini:strng_bikini 03_lrcontour_brief:lrcontour_brief 05_hpstr_brief:hpstr_brief 07_reg_classic:classic; do
  folder="${pair%%:*}"; trig="${pair##*:}"
  echo "=== $folder ($trig) ==="
  "$PY" batch_caption.py --images "$DS/$folder" --template "New Briefs Caption" --name "$trig" --prefix "" --max-tokens 300
done > "$SCR/trial_run.log" 2>&1
```

Run with `run_in_background`; the harness notifies on exit. Expected in the log: five `Done: N captioned, M skipped, 0 failed` lines with N = 34, 8, 10, 12, 2.

- [ ] **Step 3: Write the report script**

Create `<scratchpad>/briefs_report.py`:

```python
"""Trial report for the New Briefs Caption run.
Usage: python -I briefs_report.py <trial_targets.txt>"""
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, r"D:\001_Personal_Proj\Captioner_QWENVL")
from captioner.prompts import check_briefs_caption  # noqa: E402

TRIGGERS = {
    "01_tnga_brief": "tnga_brief", "02_strng_bikini": "strng_bikini",
    "03_lrcontour_brief": "lrcontour_brief", "05_hpstr_brief": "hpstr_brief",
    "06_dblstrap_bikini": "dblstrap_bikini", "07_reg_classic": "classic",
}

targets = [Path(p) for p in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines() if p.strip()]
per_folder, errors, lengths, samples, missing = Counter(), Counter(), [], [], []
for img in targets:
    txt = img.with_suffix(".txt")
    if not txt.exists():
        missing.append(txt.name)
        continue
    text = txt.read_text(encoding="utf-8")
    problems = check_briefs_caption(text, TRIGGERS[img.parent.name])
    per_folder[(img.parent.name, "pass" if not problems else "FAIL")] += 1
    for p in problems:
        errors["trigger" if p.startswith("trigger") else p] += 1
    if problems:
        print(f"FAIL {img.parent.name}/{txt.name}: {problems}\n    {text.strip()}\n")
    lengths.append(len(text.split()))
    if len(samples) < 6 and not problems:
        samples.append(f"[{img.parent.name}/{txt.stem}]\n{text.strip()}")

print("== per folder ==")
for key in sorted(per_folder):
    print(*key, per_folder[key])
print("missing .txt:", len(missing), missing)
print("== errors ==", dict(errors))
if lengths:
    print("== words == min", min(lengths), "median", statistics.median(lengths), "max", max(lengths))
print("\n== samples ==\n" + "\n\n".join(samples))
```

- [ ] **Step 4: Run the report**

```bash
SCR="C:/Users/tonyn/AppData/Local/Temp/claude/D--001-Personal-Proj-Captioner-QWENVL/fc0c8abd-8e15-436f-a20c-1bd63887596f/scratchpad"
"D:/001_Personal_Proj/Captioner_QWENVL/.venv/Scripts/python.exe" -I "$SCR/briefs_report.py" "$SCR/trial_targets.txt"
```

Expected: `missing .txt: 0`, ≥ 60 of 66 pass (≥ 90 %), word median inside 50–90. Every FAIL line is listed with its problems and text.

- [ ] **Step 5: Read the failures**

If one error type hits more than 10 % of captions (for example `trigger x0` because the model wrote "tnga briefs", or `words >100`), note the pattern in the final report. Do not loop on prompt edits inside this task: a wording change is a new commit the user decides on, with the failing files re-run via `--overwrite --limit` on copies.

Spot-check three captions against their images (one headless crop, one full-body, one banner with overlaid text) and confirm: framing phrase matches, "head out of frame" present when the head is cut, no transcribed brand text.

---

## Self-review notes

- Spec coverage: template + `{garment}` (Task 1), checker + forbidden list (Task 2), `--name` + self-test + README (Task 3), `sporthc_brief` retirement (Task 4), trial run + report + manifest (Task 5). Spec "Không đổi" list respected: `app.py` untouched.
- Names used consistently: `brief_garment`, `BRIEF_REG_LABELS`, `BRIEF_REG_GARMENT`, `BRIEF_FORBIDDEN`, `check_briefs_caption`, `build_prompt(template, prefix, name="")`, `--name`.
- `GUIDE_BAD` word count is 7, matching the checker test and the spec.
