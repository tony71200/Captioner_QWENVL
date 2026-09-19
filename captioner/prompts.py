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
    {
        "name": "Description Character",
        "description": "Analyze the image and rewrite it as a training caption for a character LoRA. Output a single continuous paragraph with no line breaks. Always start the result with the exact string Ivan_Ryo. Write the caption in a stable order: first describe the subject class and the character’s fixed identity traits that should remain consistent across the dataset, including face shape, body type or build, hairstyle, hair color, skin tone, and other clearly visible recurring features. Then describe the facial expression, pose, and action. After that, describe clothing, accessories, and any clearly visible body exposure details relevant to the image. Then describe the environment or background, lighting, framing, camera angle, and perspective. Finally, describe the actual visual medium or style shown in the image. Keep the caption faithful to the source image and describe only what is directly visible or reasonably inferable. If the source image is anime, illustration, painting, or another stylized medium, describe the actual medium instead of converting it into a photorealistic image. If the character is wearing underwear, clearly describe the type, material, and color. If the image contains high skin exposure or sensitive elements, describe them in tasteful, non-graphic, non-explicit language and mention only what is clearly visible, while naturally noting coverage created by pose, hands, hair, fabric, shadows, or foreground objects when relevant. Do not speculate about hidden details, exact age, ethnicity, identity, address, or personal information. Avoid negative-prompt phrasing, avoid restyling instructions, and avoid poetic filler. Keep the wording clear, concrete, and consistent so that recurring identity traits are described similarly across images. Output only the final caption text.",
        "prompt_text": (
            "Analyze the image and rewrite it as a training caption for a character LoRA. Output a single continuous paragraph with no line breaks. Always start the result with the exact string Ivan_Ryo. Write the caption in a stable order: first describe the subject class and the character’s fixed identity traits that should remain consistent across the dataset, including face shape, body type or build, hairstyle, hair color, skin tone, and other clearly visible recurring features. Then describe the facial expression, pose, and action. After that, describe clothing, accessories, and any clearly visible body exposure details relevant to the image. Then describe the environment or background, lighting, framing, camera angle, and perspective. Finally, describe the actual visual medium or style shown in the image. Keep the caption faithful to the source image and describe only what is directly visible or reasonably inferable. If the source image is anime, illustration, painting, or another stylized medium, describe the actual medium instead of converting it into a photorealistic image. If the character is wearing underwear, clearly describe the type, material, and color. If the image contains high skin exposure or sensitive elements, describe them in tasteful, non-graphic, non-explicit language and mention only what is clearly visible, while naturally noting coverage created by pose, hands, hair, fabric, shadows, or foreground objects when relevant. Do not speculate about hidden details, exact age, ethnicity, identity, address, or personal information. Avoid negative-prompt phrasing, avoid restyling instructions, and avoid poetic filler. Keep the wording clear, concrete, and consistent so that recurring identity traits are described similarly across images. Output only the final caption text."
        ),
    },
    {
        "name": "Description Underwear",
        "description": "Mô tả quần lót và nhân vật",
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
