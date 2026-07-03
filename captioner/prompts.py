"""
Prompt templates for image captioning with Qwen2.5-VL.
"""
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
        "description": "Mô tả để train Lora theo ChatGPT.",
        "name_field": True,
        "name_label": "Character/Object name",
        "default_name": "Rennoir",
        "system_prompt": (
            "You are generating concise natural-language captions for a realistic character "
            "LoRA training dataset. The goal is to help the LoRA learn a consistent "
            "adult male character identity named '{name}', while avoiding overfitting "
            "to temporary details such as outfit, background, pose, lighting, or camera angle."
        ),
        "user_prompt": (
            "Caption rules:\n"
            "1. Always start the caption with the trigger name '{name}'."
            "2. Describe '{name}' as an adult man naturally in the sentence."
            "3. Use 'adult Asian man' only when it is visually appropriate or clearly useful."
            "4. Prioritize stable identity traits:"
            " - hairstyle"
            " - hair color"
            " - face shape"
            " - facial structure"
            " - eyebrows, eyes, nose, lips, jawline, or other visible facial traits"
            "5. Mention body details only if clearly visible and useful, such as shirtless torso, chest, slight abs, lean build, or athletic build."
            "6. Mention clothing, pose, camera angle, lighting, and background only if they are clearly visible and important. Keep these details very brief."
            "7. Do not describe personality, story, mood symbolism, hidden meaning, or anything not visible."
            "8. Do not compare the subject to celebrities, fictional characters, or other people."
            "9. Do not use the words: girl, woman, childlike."
            "10. Do not write a tag list."
            "11. Output only one natural caption."
            "12. The caption must be 1 to 2 short sentences."
            "Preferred caption structure:"
            "'{name}' is a realistic adult man with [stable hairstyle and facial traits]. [Optional brief visible body/clothing detail if useful]."
            "Write a concise natural training caption for this image of '{name}'."
            "Focus mainly on visible identity features, especially hairstyle and facial characteristics. Describe him as a realistic adult man. Use 'adult Asian man' only if it is visually appropriate. Keep body details brief, and avoid describing outfit, pose, lighting, or background unless they are clearly important."
            "Do not use the words: girl, woman, childlike."
            "Return only the final caption."
        ),
    }
]


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
        return system_prompt, _format(custom_prompt, name=name)

    if template_name == "Custom" or not tmpl:
        return "", "Describe this image."

    system_prompt = _format(tmpl.get("system_prompt", ""), name=name)
    user_prompt = _format(tmpl.get("user_prompt", tmpl.get("prompt_text", "")), name=name)
    return system_prompt, user_prompt
