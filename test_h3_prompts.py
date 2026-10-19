"""Self-checks for the Prompt Video Minimax H3 helpers. Run: D:\\Comfy\\.venv\\Scripts\\python.exe test_h3_prompts.py"""
from captioner.h3_prompts import (
    bible_field_count,
    build_references,
    clamp_durations,
    describe_references,
    format_shot_plan,
    shot_plan_from_story,
    shot_plan_from_total,
)


def test_clamp_durations():
    assert clamp_durations(10) == [10]
    assert clamp_durations(12) == [12]
    assert clamp_durations(14) == [12]
    assert clamp_durations(15) == [12]
    assert clamp_durations(16) == [8, 8]
    assert clamp_durations(25) == [9, 8, 8]
    assert clamp_durations(30) == [10, 10, 10]
    for total in range(16, 121):
        pieces = clamp_durations(total)
        assert sum(pieces) == total and max(pieces) <= 12, (total, pieces)


def test_shot_plan_from_story():
    plan, notes = shot_plan_from_story("00:00–00:20 — Anh đi bộ.\n00:20-00:26 — Anh dừng lại.")
    assert plan == [(0, 10, 10), (10, 20, 10), (20, 26, 6)]
    assert notes == ["story 00:00–00:20 (20s) → 2 shot: 10s + 10s"]
    plan, notes = shot_plan_from_story("00:00–00:14 — Anh đi bộ.")
    assert plan == [(0, 14, 12)]
    assert notes == ["story 00:00–00:14 (14s) → nén còn 12s"]
    assert shot_plan_from_story("Anh đi bộ ra ban công.") == ([], [])
    assert shot_plan_from_story("00:10–00:05 — ngược") == ([], [])


def test_shot_plan_from_total():
    assert shot_plan_from_total(24) == ([(0, 12, 12), (12, 24, 12)], ["story 00:00–00:24 (24s) → 2 shot: 12s + 12s"])
    assert shot_plan_from_total(0) == ([], [])
    assert shot_plan_from_total(None) == ([], [])


def test_format_shot_plan():
    assert format_shot_plan([(0, 10, 10), (10, 20, 10)]) == (
        "SHOT 01: story 00:00–00:10 (10s)\nSHOT 02: story 00:10–00:20 (10s)")
    assert format_shot_plan([(0, 14, 12)]) == "SHOT 01: story 00:00–00:14 (12s) — compress the action into 12s"
    assert format_shot_plan([]) == "Choose 5 to 12 seconds per shot; never more than 12 seconds."


def test_build_references_numbers_only_filled_slots():
    assert build_references(["character", None, "scene", "face"]) == [
        {"n": 1, "ref": "ref_image_0", "role": "character", "slot": 0},
        {"n": 2, "ref": "ref_image_1", "role": "scene", "slot": 2},
        {"n": 3, "ref": "ref_image_2", "role": "face", "slot": 3},
    ]


def test_describe_references():
    only_character = describe_references(build_references(["character", None, None, None]))
    assert only_character == (
        "<Picture 1>, ref_image_0 = main character: identity, body, outfit. "
        "Also the setting, because no scene picture is given.")
    with_scene = describe_references(build_references(["character", None, "scene", "face"]))
    assert with_scene == (
        "<Picture 1>, ref_image_0 = main character: identity, body, outfit.\n"
        "<Picture 2>, ref_image_1 = scene / environment. Ignore any person visible in it.\n"
        "<Picture 3>, ref_image_2 = face reference sheet of the same character as <Picture 1>; "
        "use it for face details only.")
    outfit = describe_references(build_references(["character", "outfit"]))
    assert "<Picture 2>, ref_image_1 = outfit / props worn or held by the main character." in outfit


def test_bible_field_count():
    assert bible_field_count("NAME_HINT: a\nFACE: b\nfoo: c\n") == 2
    assert bible_field_count("") == 0


if __name__ == "__main__":
    tests = [fn for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for fn in tests:
        fn()
        print("ok", fn.__name__)
    print(f"{len(tests)} checks passed")
