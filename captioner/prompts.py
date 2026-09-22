"""
Prompt templates for image captioning with Qwen2.5-VL.
"""
import re
from typing import List, Optional

PROMPT_TEMPLATES = [
    {
        "name": "Short Caption",
        "description": "Mô tả ngắn gọn 1–2 câu.",
        "system_prompt": "",
        "user_prompt": (
            "Write a brief, concise caption for this image in 1-2 sentences. "
            "Focus on the most important subject and action."
        ),
    },
    {
        "name": "Detailed Description",
        "description": "Mô tả chi tiết toàn bộ nội dung ảnh.",
        "system_prompt": "",
        "user_prompt": (
            "Describe this image in detail. Include: main subjects, their appearance, "
            "actions or poses, setting and background, colors and lighting, "
            "composition, mood and atmosphere."
        ),
    },
    {
        "name": "Booru Tags",
        "description": "Tags kiểu Danbooru, phân cách bằng dấu phẩy.",
        "system_prompt": "",
        "user_prompt": (
            "Generate descriptive tags for this image in comma-separated booru tag format. "
            "Include tags for: subject (e.g. 1girl, 1boy, no humans), appearance, "
            "hair color/style, clothing, pose/action, setting/background, "
            "art style, and quality. Order from most important to least important. "
            "Output ONLY the tags, no explanation."
        ),
    },
    {
        "name": "Structured Caption",
        "description": "Mô tả có cấu trúc theo từng hạng mục.",
        "system_prompt": "",
        "user_prompt": (
            "Describe this image using the following structured format:\n"
            "Subject: [describe the main subject(s)]\n"
            "Appearance: [describe physical appearance, clothing, hair, etc.]\n"
            "Action/Pose: [describe what they are doing or their pose]\n"
            "Setting/Background: [describe the environment and background]\n"
            "Lighting/Mood: [describe lighting, colors, and overall mood]\n"
            "Style: [describe art style or photographic style]"
        ),
    },
    {
        "name": "Training Caption (SD/Flux)",
        "description": "Caption tối ưu cho training Stable Diffusion / Flux.",
        "system_prompt": "",
        "user_prompt": (
            "Write a detailed image caption optimized for AI image generation model training. "
            "Describe every visual element comprehensively: main subject(s), physical features, "
            "clothing and accessories, pose and expression, hand positions if visible, "
            "background and setting, lighting direction and quality, color palette, "
            "camera angle/perspective, and artistic style. "
            "Write in flowing descriptive prose without bullet points."
        ),
    },
    {
        "name": "Object & Scene Analysis",
        "description": "Phân tích đối tượng và cảnh vật trong ảnh.",
        "system_prompt": "",
        "user_prompt": (
            "Analyze this image and list:\n"
            "1. Main objects/subjects present\n"
            "2. Scene/environment type\n"
            "3. Spatial relationships between objects\n"
            "4. Notable colors and textures\n"
            "5. Any text or symbols visible\n"
            "6. Overall scene category (indoor/outdoor/abstract/etc.)"
        ),
    },
    {
        "name": "Custom",
        "description": "Nhập prompt tùy ý của bạn.",
        "system_prompt": "",
        "user_prompt": "",  # Filled by user input
    },
    {
        "name": "Description Character",
        "description": "Mô tả để tái hiện tạo nhân vật bằng Text 2 Image.",
        "name_field": True,
        "name_label": "Character/Object name",
        "default_name": "Ivan_Ryo",
        "system_prompt": "",
        "user_prompt": (
            "Analyze the image and rewrite it as a long detailed image prompt. "
            "Start with '{name}'."
            "Keep the same pose, actions, outfit, proportions, lighting, camera angle and style. "
            "Always describe the character's fixed features: face shape, body type."
            "Always describe the result as photorealistic; if the original image is anime, illustration, painting, or any drawn style, reinterpret it as a realistic photorealistic image, removing visible line art, cel shading, flat colors, and other stylized drawing effects."
            "If the subject is naked or nude, a full artistic and non-graphic description is required, including body posture, visible body condition, and penis status, while adding natural hand placement, fabric, shadow, pose angle, foreground objects, or other tasteful artistic elements to cover or obscure sensitive intimate parts when appropriate." 
            "If the subject is wearing underwear, the type, material, and color must be described. Write the output as a single continuous paragraph without any line breaks."
            "Output only the final prompt text."
        ),
    },
    {
        "name": "Description Underwear",
        "description": "Mô tả quần lót và nhân vật",
        "name_field": True,
        "name_label": "Character/Object name",
        "default_name": "Ivan_Undie",
        "system_prompt": "",
        "user_prompt": (
            "Analyze the image and rewrite it as a long, detailed image prompt for captioning. "
            "Start with '{name}'. "
            "Immediately after that, write the underwear style (for example: briefs, boxer briefs, boxers, trunks, jockstrap, thong, fundoshi, bikini briefs, or other visible type). "
            "If a brand name is clearly visible on the waistband, write the brand name in quotation marks immediately after the underwear style. "
            "Keep the same pose, body proportions, lighting, camera angle, framing, and overall visual style. Focus primarily on the underwear: describe the style, cut, rise, pouch shape, coverage, leg openings, waistband width, waistband design, visible brand text, color, fabric or material, texture, pattern, seams, trim, and how it fits on the body. Describe the wearer only in generic body-type terms such as slim, lean, athletic, muscular, average build, or stocky. "
            "Do not describe facial features, hairstyle, age, or identity. Do not invent a brand name, logo, fabric, or detail that is not clearly visible in the image. Write the result as one single continuous paragraph with no line breaks. "
            "Output only the final prompt text."
        ),
    },
    {
        "name": "Train Lora Prompt (Following Clause)",
        "description": "Mô tả để train Lora theo câu điều kiện.",
        "name_field": True,
        "name_label": "Character/Object name",
        "default_name": "IvanRyo3",
        "system_prompt": (
            "You are an expert image captioner preparing training data for a LoRA "
            "model of a specific real person named '{name}'."
        ),
        "user_prompt": (
            "Write ONE natural-language caption for this image, following these rules strictly:\n"
            "1. ALWAYS start the caption with '{name}'."
            "2. DO NOT describe permanent identity features that should stay tied to the trigger word — no eye color, face shape, skin tone, nose/lip shape, or general ethnicity. The model must learn these implicitly from '{name}', not from text."
            "3. DO describe everything that varies between images:"
            " - Shot framing (close-up portrait, upper body, medium shot, etc.)"
            " - Head/body angle and pose (looking at camera, looking away, tilted head, etc.)"
            " - Facial expression (smiling, neutral, serious, laughing, etc.)"
            " - Hair style/color IF it changes between photos in the dataset (skip if hair is identical in every image)"
            " - Clothing and accessories (jacket, glasses, jewelry, etc.)"
            " - Lighting (soft natural light, studio lighting, backlight, etc.)"
            "4. Mention the background/setting in no more than 4-5 words — just enough context, not a full scene description (e.g. 'in a cafe', 'outdoors, blurred background', 'plain white background')."
            "5. Write in plain fluent English, one or two sentences, no tag lists, no commas-only style, no markdown."
            "6. Do not mention 'photo', 'image', 'picture' — describe the subject directly as if narrating what is seen."
            "Output ONLY the caption text, nothing else."
        ),
    },
    {
        "name": "Train Lora Prompt (Following ChatGPT)",
        "description": "Caption ngắn, sạch cho FLUX.2 character LoRA; tự đánh dấu ảnh cần loại.",
        "name_field": True,
        "name_label": "Character trigger word",
        "default_name": "HongDong_00",
        "training_caption": True,
        "system_prompt": (
            "You create clean natural-language captions for training a FLUX.2 Klein "
            "character LoRA of one specific real adult man. The trigger word '{name}' "
            "represents only the man's identity. Describe visible attributes that should "
            "remain controllable at inference, while leaving permanent facial identity "
            "attached to the trigger word. Follow the requested output format exactly."
        ),
        "user_prompt": (
            "Analyze the image and follow these rules strictly:\n"
            "1. If two or more people are visible, output exactly "
            "'REVIEW_REQUIRED_MULTIPLE_PEOPLE' and nothing else.\n"
            "2. If the only person's face is absent, extremely small, heavily blurred, "
            "or fully hidden, output exactly 'REVIEW_REQUIRED_IDENTITY_NOT_CLEAR' and "
            "nothing else.\n"
            "3. Otherwise output exactly one English sentence of 20 to 45 words on one line.\n"
            "4. Start with exactly: '{name}, an adult East Asian man,' using the trigger "
            "once and preserving its spelling and capitalization.\n"
            "5. Briefly describe only visible, changeable attributes: shot framing, view "
            "or body angle, expression or action, current hairstyle and hair color, "
            "clothing, accessories, pose, and a short setting.\n"
            "6. Mention build or body visibility only when relevant, using neutral terms "
            "such as lean, athletic, shirtless, wearing briefs, or rear nude view. Keep "
            "nudity descriptions factual, non-graphic, and non-sexual.\n"
            "7. Do not describe permanent identity-bearing facial anatomy: face shape, "
            "eye shape or color, nose, lips, jawline, cheekbones, skin tone, ethnicity "
            "beyond the fixed class phrase, attractiveness, or resemblance to anyone.\n"
            "8. Do not add quality claims, photographic technique, detailed lighting, "
            "fabric micro-details, anatomy lists, stories, symbolism, inferred personality, "
            "visible text transcription, or details that are not clearly visible.\n"
            "9. Never output a negative prompt, PEOPLE marker, tag list, heading, markdown, "
            "explanation, quotation marks around the caption, or a second sentence.\n"
            "Example: {name}, an adult East Asian man, shown waist-up facing the camera "
            "with short tousled black hair, smiling in a gray sleeveless shirt while "
            "cooking in a bright modern kitchen."
        ),
    }]


# Rules appended to every template (including Custom). Kept in one place so the
# Web UI, test_caption.py and batch_caption.py cannot drift apart.
PEOPLE_MARKER = "PEOPLE:"

NEGATIVE_PROMPT = (
    "Negative prompt: identical faces, same face, duplicate face, cloned face, "
    "merged faces, fused faces, face swap, twins, repeated face"
)

TRAINING_OUTPUT_RULES = (
    " Formatting rules. Output one line only, with no empty lines. Do not output "
    "a PEOPLE marker or a negative prompt. Write nothing before or after the caption "
    "or review marker."
)

OUTPUT_RULES = (
    " Formatting rules. "
    "Never separate paragraphs with a blank line: the output must contain no "
    "empty lines at all. "
    f"End with one final line reading exactly '{PEOPLE_MARKER} <n>', where <n> "
    "is how many people are visible in the image. Write nothing after that "
    "line, and never write a negative prompt yourself."
)


def _with_output_rules(user_prompt: str, training_caption: bool = False) -> str:
    """Append the appropriate output contract once to a resolved user prompt."""
    rules = TRAINING_OUTPUT_RULES if training_caption else OUTPUT_RULES
    if not user_prompt or rules.strip() in user_prompt:
        return user_prompt
    return user_prompt.rstrip() + rules


def finalize_caption(text: str) -> str:
    """
    Turn a raw model caption into the final caption text.

    The model is asked only to count people; the negative prompt wording is
    ours, so it comes out byte-identical in every caption instead of being
    re-improvised per image - and it appears for two or more people only.
    A caption without the marker is returned untouched, so nothing is lost when
    the model ignores the instruction.
    """
    body = text.strip()
    # The marker is asked for on its own final line, but models routinely run it
    # on at the end of the last sentence instead - accept either.
    match = re.search(rf"{PEOPLE_MARKER}\s*(\d+)\s*[.]?\s*$", body, re.IGNORECASE)
    if match is None:
        return body

    people = int(match.group(1))
    body = body[: match.start()].rstrip().rstrip(",;:").rstrip()
    # Drop any negative prompt the model wrote anyway; ours is authoritative.
    body = re.sub(r"\s*Negative prompt:.*$", "", body, flags=re.IGNORECASE | re.DOTALL).strip()
    return f"{body}\n{NEGATIVE_PROMPT}" if people >= 2 else body


def get_prompt_names() -> List[str]:
    """Return list of all prompt template names."""
    return [t["name"] for t in PROMPT_TEMPLATES]


def get_prompt_by_name(name: str) -> Optional[dict]:
    """Return prompt template dict by name, or None if not found."""
    for t in PROMPT_TEMPLATES:
        if t["name"] == name:
            return t
    return None


def prompt_needs_name(template_name: str) -> bool:
    """Return True if the prompt template should expose the subject name field."""
    tmpl = get_prompt_by_name(template_name)
    return bool(tmpl and tmpl.get("name_field"))


def get_prompt_default_name(template_name: str) -> str:
    """Return the default subject name for a template, if configured."""
    tmpl = get_prompt_by_name(template_name)
    if not tmpl:
        return ""
    return str(tmpl.get("default_name", ""))


def get_prompt_name_label(template_name: str) -> str:
    """Return the UI label for the subject name field."""
    tmpl = get_prompt_by_name(template_name)
    if not tmpl:
        return "Character/Object name"
    return str(tmpl.get("name_label", "Character/Object name"))


def _resolve_subject_name(template_name: str, subject_name: str = "") -> str:
    name = (subject_name or "").strip()
    return name or get_prompt_default_name(template_name)


class _SafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def _format(text: str, **kwargs) -> str:
    if not text:
        return text
    return text.format_map(_SafeDict(**kwargs))


def resolve_prompt(template_name: str, custom_prompt: str = "", subject_name: str = "") -> tuple[str, str]:
    """
    Return the final (system_prompt, user_prompt) pair.

    - Custom text overrides only the user prompt and may use placeholders such as {name}.
    - Unknown placeholders are preserved instead of raising KeyError.
    - Templates without a system prompt return an empty system prompt.
    """
    custom_prompt = custom_prompt.strip()
    tmpl = get_prompt_by_name(template_name)
    name = _resolve_subject_name(template_name, subject_name)

    if custom_prompt:
        system_prompt = _format(tmpl.get("system_prompt", "") if tmpl else "", name=name)
        return system_prompt, _with_output_rules(
            _format(custom_prompt, name=name),
            training_caption=bool(tmpl and tmpl.get("training_caption")),
        )

    if template_name == "Custom" or not tmpl:
        return "", _with_output_rules("Describe this image.")

    system_prompt = _format(tmpl.get("system_prompt", ""), name=name)
    user_prompt = _format(tmpl.get("user_prompt", tmpl.get("prompt_text", "")), name=name)
    return system_prompt, _with_output_rules(
        user_prompt,
        training_caption=bool(tmpl.get("training_caption")),
    )
