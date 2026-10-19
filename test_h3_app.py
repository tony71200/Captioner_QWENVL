"""Self-checks for the H3 tab handlers (fake model) and captioner helpers. Run: D:\\Comfy\\.venv\\Scripts\\python.exe test_h3_app.py"""
from captioner.gguf_captioner import GGUFCaptioner
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
    assert [c["type"] for c in one] == ["image", "text"] and one[0]["image"].startswith("file://")
    many = HFCaptioner._user_content(["a.png", "b.png"], "hi")
    assert [c["type"] for c in many] == ["image", "image", "text"]
    assert HFCaptioner._generate_kwargs(None) == {"do_sample": False}
    assert HFCaptioner._generate_kwargs(0) == {"do_sample": False}
    assert HFCaptioner._generate_kwargs(0.7) == {"do_sample": True, "temperature": 0.7, "top_p": 0.9}


if __name__ == "__main__":
    tests = [fn for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for fn in tests:
        fn()
        print("ok", fn.__name__)
    print(f"{len(tests)} checks passed")
