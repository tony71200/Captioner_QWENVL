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
