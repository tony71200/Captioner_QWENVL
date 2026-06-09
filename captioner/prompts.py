"""
Prompt templates for image captioning with Qwen2.5-VL.
"""
import re
from typing import List, Optional

PROMPT_TEMPLATES = [
    {
        "name": "Short Caption",
        "description": "Mô tả ngắn gọn 1–2 câu.",
        "prompt_text": (
            "Write a brief, concise caption for this image in 1-2 sentences. "
            "Focus on the most important subject and action."
        ),
    },
    {
        "name": "Detailed Description",
        "description": "Mô tả chi tiết toàn bộ nội dung ảnh.",
        "prompt_text": (
            "Describe this image in detail. Include: main subjects, their appearance, "
            "actions or poses, setting and background, colors and lighting, "
            "composition, mood and atmosphere."
        ),
    },
    {
        "name": "Booru Tags",
        "description": "Tags kiểu Danbooru, phân cách bằng dấu phẩy.",
        "prompt_text": (
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
        "prompt_text": (
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
        "prompt_text": (
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
        "prompt_text": (
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
        "prompt_text": "",  # Filled by user input
    },
    {
        "name": "Description Character",
        "description": "Mô tả để tái hiện tạo nhân vật bằng Text 2 Image.",
        "name_field": True,
        "name_label": "Character/Object name",
        "default_name": "Ivan_Ryo",
        "prompt_text": (
            "Analyze the image and rewrite it as a long detailed image prompt. "
            "Start with 'Ivan_Ryo'. "
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
        "prompt_text": (
            "Analyze the image and rewrite it as a long, detailed image prompt for captioning. "
            "Start with 'Ivan_Undie'. "
            "Immediately after that, write the underwear style (for example: briefs, boxer briefs, boxers, trunks, jockstrap, thong, fundoshi, bikini briefs, or other visible type). "
            "If a brand name is clearly visible on the waistband, write the brand name in quotation marks immediately after the underwear style. "
            "Keep the same pose, body proportions, lighting, camera angle, framing, and overall visual style. Focus primarily on the underwear: describe the style, cut, rise, pouch shape, coverage, leg openings, waistband width, waistband design, visible brand text, color, fabric or material, texture, pattern, seams, trim, and how it fits on the body. Describe the wearer only in generic body-type terms such as slim, lean, athletic, muscular, average build, or stocky. "
            "Do not describe facial features, hairstyle, age, or identity. Do not invent a brand name, logo, fabric, or detail that is not clearly visible in the image. Write the result as one single continuous paragraph with no line breaks. "
            "Output only the final prompt text."
        ),
    },
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


def _replace_start_name(prompt_text: str, default_name: str, subject_name: str) -> str:
    if not default_name or not subject_name or subject_name == default_name:
        return prompt_text
    quoted_default = re.escape(default_name)
    pattern = rf"(Start with\s+['\"]){quoted_default}(['\"]\.)"
    return re.sub(pattern, lambda match: f"{match.group(1)}{subject_name}{match.group(2)}", prompt_text, count=1)


def resolve_prompt(template_name: str, custom_prompt: str = "", subject_name: str = "") -> str:
    """
    Resolve the final prompt string.
    Match the ComfyUI-QwenVL behavior:
    - if custom_prompt is provided, it overrides the selected preset entirely
    - otherwise use the selected preset prompt
    - templates with a name field replace their default name unless custom text
      is provided without a {name} placeholder
    """
    custom_prompt = custom_prompt.strip()
    subject_name = _resolve_subject_name(template_name, subject_name)

    if custom_prompt:
        if "{name}" in custom_prompt:
            return custom_prompt.replace("{name}", subject_name)
        return custom_prompt

    if template_name == "Custom":
        return "Describe this image."

    tmpl = get_prompt_by_name(template_name)
    if tmpl:
        prompt_text = tmpl["prompt_text"]
        if tmpl.get("name_field"):
            prompt_text = _replace_start_name(
                prompt_text,
                str(tmpl.get("default_name", "")),
                subject_name,
            )
        return prompt_text

    return "Describe this image."
