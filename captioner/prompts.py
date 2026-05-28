"""
Prompt templates for image captioning with Qwen2.5-VL.
"""
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


def resolve_prompt(template_name: str, custom_prompt: str = "") -> str:
    """
    Resolve the final prompt string.
    Match the ComfyUI-QwenVL behavior:
    - if custom_prompt is provided, it overrides the selected preset entirely
    - otherwise use the selected preset prompt
    """
    custom_prompt = custom_prompt.strip()

    if custom_prompt:
        return custom_prompt

    if template_name == "Custom":
        return "Describe this image."

    tmpl = get_prompt_by_name(template_name)
    if tmpl:
        return tmpl["prompt_text"]

    return "Describe this image."
