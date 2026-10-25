"""Self-checks for the H3 tab handlers (fake model) and captioner helpers. Run: D:\\Comfy\\.venv\\Scripts\\python.exe test_h3_app.py"""
import tempfile
from pathlib import Path

from PIL import Image

import app
from captioner.gguf_captioner import GGUFCaptioner
from captioner.h3_prompts import parse_h3_ideas
from captioner.hf_captioner import HFCaptioner


def test_gguf_user_content_keeps_single_image_shape_and_accepts_lists():
    one = GGUFCaptioner._user_content("a.png", "hi")
    assert one[0] == {"type": "text", "text": "hi"}
    assert len(one) == 2 and one[1]["type"] == "image_url"
    many = GGUFCaptioner._user_content(["a.png", "b.png"], "hi")
    assert [c["type"] for c in many] == ["text", "image_url", "image_url"]
    assert many[1]["image_url"]["url"].endswith("a.png") and many[2]["image_url"]["url"].endswith("b.png")


def test_gguf_sampling():
    assert GGUFCaptioner._sampling(None, 0.6) == {"temperature": 0.6, "seed": 1}
    assert GGUFCaptioner._sampling(0.7, 0.6) == {"temperature": 0.7, "seed": None}


def test_hf_helpers():
    one = HFCaptioner._user_content("a.png", "hi")
    assert [c["type"] for c in one] == ["image", "text"] and one[0]["image"] == str(Path("a.png").resolve())
    many = HFCaptioner._user_content(["a.png", "b.png"], "hi")
    assert [c["type"] for c in many] == ["image", "image", "text"]
    assert HFCaptioner._generate_kwargs(None) == {"do_sample": False}
    assert HFCaptioner._generate_kwargs(0) == {"do_sample": False}
    assert HFCaptioner._generate_kwargs(0.7) == {"do_sample": True, "temperature": 0.7, "top_p": 0.9}


class FakeCaptioner:
    """Returns canned replies in order and records every call."""

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def caption_image(self, image_path, user_prompt, max_new_tokens=512, system_prompt=None, temperature=None):
        self.calls.append(("image", image_path, user_prompt, system_prompt, temperature))
        return self.replies.pop(0)

    def caption_text(self, user_prompt, max_new_tokens=512, system_prompt=None, temperature=None):
        self.calls.append(("text", None, user_prompt, system_prompt, temperature))
        return self.replies.pop(0)


BIBLE = "NAME_HINT: man with short black hair\nAGE_APPEARANCE: 30, adult\nFACE: oval face"
SHOTS = """Sure, here it is.
CHARACTER & SCENE DEFINITION
subject_definitions:
<Subject 1> man, based on <Picture 1>, ref_image_0: oval face.
summary:
[reference generation] He walks.
overall_soundscape:
Room tone.
non_diegetic_music:
NONE

SHOT STRUCTURE — MINIMAX H3
[SHOT 01 | DURATION: 12s | PRIMARY CAMERA SETUP: medium shot]
summary:
He walks.
detailed_description:
00:00–00:12 — He walks.
overall_soundscape:
Footsteps.
"""
IDEAS = """Ý TƯỞNG 1: Ban công
Logline: Ra ban công.
Tổng thời lượng: 20s
Số shot: 2
Thời lượng shot: 10s + 10s

Ý TƯỞNG 2: Gương
Logline: Soi gương.
Thời lượng shot: 12s

Ý TƯỞNG 3: Dạo
Logline: Đi dạo.
Thời lượng shot: 8s + 8s
"""


def _media(path1="", extra=None):
    img = Image.new("RGB", (64, 64), "white")
    return (img, extra, None, None, *app.H3_ROLE_CHOICES, path1, "", "", "", "Adaptive image", 512, 512)


def _run_with(fake, fn, *args):
    app._captioner = fake
    try:
        return fn(*args)
    finally:
        app._captioner = None


def test_generate_runs_bible_then_shots_and_saves_next_to_picture1():
    with tempfile.TemporaryDirectory() as tmp:
        pic = Path(tmp) / "hero.png"
        Image.new("RGB", (64, 64), "white").save(pic)
        fake = FakeCaptioner(BIBLE, SHOTS)
        bible, part1, part2, saved, status, log, log_text = _run_with(
            fake, app.h3_generate, "00:00–00:14 — Anh đi bộ.", "", "bible rules", "shot rules", False, 0, [],
            *_media(str(pic)))
        assert bible == BIBLE
        assert part1.startswith("subject_definitions:") and "Sure" not in part1
        assert part2.startswith("[SHOT 01 | DURATION: 12s")
        assert part2.rstrip().endswith("non_diegetic_music:\nNONE")
        assert saved == str(Path(tmp) / "hero_H3_sfw.txt")
        assert Path(saved).read_text(encoding="utf-8").startswith("CHARACTER & SCENE DEFINITION\n\nsubject_definitions:")
        kind, paths, user, system, temp = fake.calls[0]
        assert kind == "image" and len(paths) == 1 and system == "bible rules" and temp == 0.3
        assert "Also the setting" in user
        kind, _, user, system, temp = fake.calls[1]
        assert kind == "text" and system == "shot rules" and temp == 0.4
        assert "SHOT 01: story 00:00–00:14 (12s)" in user and BIBLE in user
        assert "nén còn 12s" in log_text and "Saved" in log_text
        assert "#4ade80" in status  # _badge(..., "success") colour
        assert len(log) == len(log_text.splitlines())


def test_generate_sends_extra_pictures_in_order():
    with tempfile.TemporaryDirectory() as tmp:
        pic = Path(tmp) / "hero.png"
        Image.new("RGB", (64, 64), "white").save(pic)
        fake = FakeCaptioner(BIBLE, SHOTS)
        extra = Image.new("RGB", (32, 32), "black")
        _run_with(fake, app.h3_generate, "Anh đi bộ.", "", "b", "s", False, 0, [], *_media(str(pic), extra))
    kind, paths, user, _, _ = fake.calls[0]
    assert len(paths) == 2
    assert "<Picture 2>, ref_image_1 = face reference sheet" in user


def test_generate_reuses_bible_and_keeps_raw_text_when_no_shot_block():
    fake = FakeCaptioner("I cannot do that.")
    bible, part1, part2, saved, status, log, log_text = _run_with(
        fake, app.h3_generate, "Anh đi bộ.", "NAME_HINT: x", "b", "s", False, 0, [], *_media())
    assert len(fake.calls) == 1 and fake.calls[0][0] == "text"
    assert part1 == "I cannot do that." and part2 == "" and not isinstance(saved, str)
    assert "Choose 5 to 12 seconds" in fake.calls[0][2]
    assert "no [SHOT" in log_text


def test_generate_completes_part1_tail():
    with tempfile.TemporaryDirectory() as tmp:
        pic = Path(tmp) / "hero.png"
        Image.new("RGB", (64, 64), "white").save(pic)
        raw = SHOTS.replace("non_diegetic_music:\nNONE\n\nSHOT STRUCTURE", "SHOT STRUCTURE")
        _, part1, _, _, _, _, log_text = _run_with(
            FakeCaptioner(raw), app.h3_generate, "Anh đi bộ.", "NAME_HINT: x", "b", "s", False, 0, [],
            *_media(str(pic)))
    assert part1.endswith("overall_soundscape:\nRoom tone.\nnon_diegetic_music:\nNONE")
    assert "Part 1: thêm non_diegetic_music mặc định" in log_text


def test_generate_guards():
    out = app.h3_generate("x", "", "b", "s", False, 0, [], *_media())
    assert "Load a model" in out[4]
    out = _run_with(FakeCaptioner(), app.h3_generate, "x", "", "b", "s", False, 0, [], None, *_media()[1:])
    assert "Picture 1" in out[4]
    out = _run_with(FakeCaptioner(), app.h3_generate, "  ", "", "b", "s", False, 0, [], *_media())
    assert "story" in out[4].lower()


def test_ideas_fill_radio_and_save_file():
    with tempfile.TemporaryDirectory() as tmp:
        pic = Path(tmp) / "hero.png"
        Image.new("RGB", (64, 64), "white").save(pic)
        fake = FakeCaptioner(IDEAS)
        bible, radio, ideas, status, log, log_text = _run_with(
            fake, app.h3_ideas, "NAME_HINT: x", "b", "idea rules", True, 0, [], *_media(str(pic)))
        assert [c[1] for c in radio["choices"]] == ["0", "1", "2"] and radio["value"] is None
        assert radio["choices"][0][0].startswith("Ý tưởng 1: Ban công")
        assert len(ideas) == 3 and ideas[2]["shots"] == [8, 8]
        saved = Path(tmp) / "hero_H3_nsfw_ideas.txt"
        assert saved.read_text(encoding="utf-8").startswith("Ý TƯỞNG 1: Ban công")
        kind, _, user, system, temp = fake.calls[0]
        assert kind == "text" and system == "idea rules" and temp == 0.7
        assert "Each idea lasts 10 to 36 seconds in total." in user
        fake = FakeCaptioner(IDEAS)
        _run_with(fake, app.h3_ideas, "NAME_HINT: x", "b", "i", False, 24, [], *_media(str(pic)))
        assert "Each idea lasts exactly 24 seconds in total." in fake.calls[0][2]


def test_ideas_warns_when_nothing_parsed():
    out = _run_with(FakeCaptioner("nope"), app.h3_ideas, "NAME_HINT: x", "b", "i", False, 0, [], *_media())
    assert "no ideas" in out[3]


def test_expand_follows_the_idea_plan():
    ideas = parse_h3_ideas(IDEAS)
    fake = FakeCaptioner("00:00–00:10 — Anh mở cửa.\n00:10–00:20 — Anh ra ban công.")
    bible, story, status, log, log_text = _run_with(
        fake, app.h3_expand, "0", ideas, "NAME_HINT: x", "b", "expand rules", False, [], *_media())
    assert story == "00:00–00:10 — Anh mở cửa.\n00:10–00:20 — Anh ra ban công."
    kind, _, user, system, temp = fake.calls[0]
    assert kind == "text" and system == "expand rules" and temp == 0.6
    assert "SHOT 02: story 00:10–00:20 (10s)" in user and "Ý TƯỞNG 1: Ban công" in user
    out = _run_with(FakeCaptioner(), app.h3_expand, None, ideas, "NAME_HINT: x", "b", "e", False, [], *_media())
    assert "Pick an idea" in out[2]


def test_save_writes_edited_text():
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "out.txt"
        saved, status, log, log_text = app.h3_save("P1", "P2", str(target), "", False, [])
        assert saved == str(target)
        assert target.read_text(encoding="utf-8") == "CHARACTER & SCENE DEFINITION\n\nP1\n\nSHOT STRUCTURE — MINIMAX H3\n\nP2\n"
    out = app.h3_save("", " ", "", "", False, [])
    assert "Nothing to save" in out[1]


def test_load_path():
    with tempfile.TemporaryDirectory() as tmp:
        pic = Path(tmp) / "hero.png"
        Image.new("RGB", (40, 30), "white").save(pic)
        img, log, log_text = app.h3_load_path(f'"{pic}"', [])
        assert img.size == (40, 30)
        img, log, log_text = app.h3_load_path(str(Path(tmp) / "missing.png"), [])
        assert not hasattr(img, "size") and "not found" in log_text
        img, log, log_text = app.h3_load_path("", [])
        assert not hasattr(img, "size") and log == []


VR_GEAR_BOXES = ("vr_pose_system", "vr_ideas_system", "vr_reel_system")
H3_GEAR_BOXES = ("h3_bible_system", "h3_ideas_system", "h3_expand_system", "h3_shots_system")


def _deps(trigger):
    return [d for d in app.demo.config["dependencies"] if any(t[1] == trigger for t in d.get("targets", []))]


def test_no_page_load_update_reaches_a_gear_box():
    # Gradio 6.14: any page-load value update (a callable `value=` or demo.load) into the ⚙ textboxes froze
    # the browser (svelte effect_update_depth_exceeded) as soon as their tab was opened.
    callable_values = [str(getattr(b, "label", "")) for b in app.demo.blocks.values()
                       if getattr(b, "load_event_to_attach", None)]
    assert not callable_values, callable_values
    gear_ids = {getattr(app, n)._id for n in VR_GEAR_BOXES + H3_GEAR_BOXES}
    assert not gear_ids & {o for d in _deps("load") for o in d["outputs"]}


def test_opening_a_tab_refreshes_its_gear_boxes():
    selects = {d["targets"][0][0]: d for d in _deps("select")}
    for tab, names in ((app.vr_tab, VR_GEAR_BOXES), (app.h3_tab, H3_GEAR_BOXES)):
        assert tab._id in selects, tab.label
        assert selects[tab._id]["outputs"] == [getattr(app, n)._id for n in names]
    assert app.h3_refresh_systems(True) == tuple(app._h3_system_text(s, True)
                                                 for s in ("bible", "ideas", "expand", "shots"))
    assert app.vr_refresh_systems() == tuple(app.get_video_system_prompt(s) for s in ("pose", "ideas", "reel"))


def test_log_keeps_last_200_lines():
    log = []
    for i in range(205):
        log = app._h3_log(log, f"line {i}")
    assert len(log) == 200 and log[-1].endswith("line 204") and log[0].endswith("line 5")


if __name__ == "__main__":
    tests = [fn for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for fn in tests:
        fn()
        print("ok", fn.__name__)
    print(f"{len(tests)} checks passed")
