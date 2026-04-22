import json
import os
from pathlib import Path

import google.generativeai as genai

from models.base_model import LLMModel
from schema.smell_detection_response_model import DetectionResult, LikertScaleOption, ModelEnum


class GeminiModel(LLMModel):
    name = ModelEnum.GEMINI.value

    def __init__(self):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.client = genai.GenerativeModel(self.name)
        super().__init__()

    def analyze(self, diagram_path: Path, prompt_text: str, smell_key: str) -> DetectionResult:
        with open(diagram_path, "rb") as f:
            image_bytes = f.read()

        media_type = "image/jpeg" if diagram_path.suffix.lower() == ".jpg" else "image/png"
        image_part = {"mime_type": media_type, "data": image_bytes}

        response = self.client.generate_content(
            [prompt_text, image_part],
            generation_config=genai.GenerationConfig(
                temperature=0,
                response_mime_type="application/json",
            ),
        )

        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        raw = json.loads(text.strip())
        return DetectionResult(
            smell=smell_key,
            model=ModelEnum.GEMINI,
            diagram=diagram_path.name,
            confidence=LikertScaleOption(str(raw["confidence"])),
            explanation=raw["explanation"],
            involved_services=raw.get("involved_services"),
        )
