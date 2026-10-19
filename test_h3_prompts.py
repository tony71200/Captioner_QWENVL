"""Self-checks for the Prompt Video Minimax H3 helpers. Run: D:\\Comfy\\.venv\\Scripts\\python.exe test_h3_prompts.py"""
from captioner.h3_prompts import (
    bible_field_count,
    build_references,
    check_shots,
    clamp_durations,
    describe_references,
    format_shot_plan,
    idea_label,
    idea_plan,
    idea_text,
    parse_h3_ideas,
    shot_plan_from_story,
    shot_plan_from_total,
    split_h3_output,
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


FENCE = "`" * 3  # a code fence the model may wrap its answer in
RAW_SHOTS = f"""Here you go:
{FENCE}
# CHARACTER & SCENE DEFINITION
subject_definitions:
<Subject 1> man, based on <Picture 1>, ref_image_0: oval face.
summary:
[reference generation] He walks.
**SHOT STRUCTURE — MINIMAX H3**
[SHOT 01 | DURATION: 8s | PRIMARY CAMERA SETUP: wide]
summary:
He walks.
{FENCE}
"""


def test_split_h3_output():
    part1, part2 = split_h3_output(RAW_SHOTS)
    assert part1 == ("subject_definitions:\n<Subject 1> man, based on <Picture 1>, ref_image_0: oval face.\n"
                     "summary:\n[reference generation] He walks.")
    assert part2 == "[SHOT 01 | DURATION: 8s | PRIMARY CAMERA SETUP: wide]\nsummary:\nHe walks."
    try:
        split_h3_output("I cannot help with that.")
    except ValueError as e:
        assert "no [SHOT" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_check_shots_renumbers_clamps_and_completes_tail():
    part2 = ("[SHOT 03 | DURATION: 15s | PRIMARY CAMERA SETUP: wide]\nsummary:\nA.\noverall_soundscape:\nWind.\n\n"
             "[SHOT 7 | DURATION: 6s]\nsummary:\nB.\noverall_soundscape:\nSteps.\nnon_diegetic_music:\nNONE")
    fixed, warnings = check_shots(part2)
    assert fixed == (
        "[SHOT 01 | DURATION: 12s | PRIMARY CAMERA SETUP: wide]\nsummary:\nA.\noverall_soundscape:\nWind.\n"
        "non_diegetic_music:\nNONE\n\n"
        "[SHOT 02 | DURATION: 6s]\nsummary:\nB.\noverall_soundscape:\nSteps.\nnon_diegetic_music:\nNONE")
    assert warnings == ["SHOT 01: DURATION 15s → 12s", "SHOT 01: thêm non_diegetic_music mặc định"]
    _, warnings = check_shots("[SHOT 01 | PRIMARY CAMERA SETUP: wide]\nsummary:\nA.")
    assert warnings[0] == "SHOT 01: thiếu DURATION"
    assert "SHOT 01: thêm overall_soundscape mặc định" in warnings


IDEAS_RAW = """Đây là 3 ý tưởng:
**Ý TƯỞNG 1: Buổi sáng ban công**
Logline: Anh pha cà phê rồi ra ban công.
Tổng thời lượng: 20s
Số shot: 2
Thời lượng shot: 10s + 10s

## Ý tưởng 2: Gương soi
Logline: Chỉnh áo, nháy mắt.
Tổng thời lượng: 12s

Ý TƯỞNG 3: Đi dạo
Logline: Bước chậm, quay lại cười.
Thời lượng shot: 20s + 4s

Ý TƯỞNG 4: Thừa
Logline: x
"""


def test_parse_h3_ideas():
    ideas = parse_h3_ideas(IDEAS_RAW)
    assert ideas == [
        {"name": "Buổi sáng ban công", "logline": "Anh pha cà phê rồi ra ban công.", "shots": [10, 10]},
        {"name": "Gương soi", "logline": "Chỉnh áo, nháy mắt.", "shots": [12]},
        {"name": "Đi dạo", "logline": "Bước chậm, quay lại cười.", "shots": [10, 10, 4]},
    ]
    assert parse_h3_ideas("") == []
    assert parse_h3_ideas("no ideas here") == []


def test_idea_formatting_and_plan():
    idea = {"name": "Buổi sáng ban công", "logline": "Anh pha cà phê rồi ra ban công.", "shots": [10, 10]}
    assert idea_label(1, idea) == "Ý tưởng 1: Buổi sáng ban công — Anh pha cà phê rồi ra ban công. · 20s · 2 shot: 10s + 10s"
    assert idea_text(1, idea) == ("Ý TƯỞNG 1: Buổi sáng ban công\nLogline: Anh pha cà phê rồi ra ban công.\n"
                                  "Tổng thời lượng: 20s\nSố shot: 2\nThời lượng shot: 10s + 10s")
    assert idea_plan(idea) == [(0, 10, 10), (10, 20, 10)]


if __name__ == "__main__":
    tests = [fn for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for fn in tests:
        fn()
        print("ok", fn.__name__)
    print(f"{len(tests)} checks passed")
