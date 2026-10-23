"""Self-checks for the New Briefs Caption template. Run: .venv\\Scripts\\python.exe test_briefs_prompts.py"""
from captioner.prompts import (
    BRIEF_FORBIDDEN,
    BRIEF_REG_GARMENT,
    TRAINING_OUTPUT_RULES,
    brief_garment,
    check_briefs_caption,
    get_prompt_by_name,
    resolve_prompt,
)

TEMPLATE = "New Briefs Caption"

# The guide's own examples (section 4) plus a reference caption from the dataset
# (the guide's reference example says "classic black men's briefs", which the
# checker does not accept; the dataset's 20 reg captions all put the color first).
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
    assert check_briefs_caption(ok.replace("thin silver-trimmed", "thin ATTENT!ON"), "tnga_brief") == [
        "forbidden: attent!on"
    ]
    # quoted capitalised text is a transcribed logo, slogan or overlay caption
    assert check_briefs_caption(
        ok.replace("thin silver-trimmed waistband", "waistband reading 'ATTENTION'"), "tnga_brief"
    ) == ["transcribed text: ATTENTION"]
    assert check_briefs_caption(
        ok.replace("Plain light-grey studio backdrop", 'Poster reading "TAGLESS SKIN TOUCH" behind him'), "tnga_brief"
    ) == ["transcribed text: TAGLESS SKIN TOUCH"]


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
