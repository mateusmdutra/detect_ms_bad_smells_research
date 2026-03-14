import json
import os
from pathlib import Path

import anthropic

from models.base_model import LLMModel
from schema.smell_detection_response_model import DetectionResult, LikertScaleOption, ModelEnum
from utils.utils import encode_image


class ClaudeModel(LLMModel):
    name = ModelEnum.CLAUDE.value

    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        super().__init__()

    def analyze(self, diagram_path: Path, prompt_text: str, smell_key: str) -> DetectionResult:
        image_data = encode_image(diagram_path)
        media_type = "image/jpeg" if diagram_path.suffix.lower() == ".jpg" else "image/png"

        response = self.client.messages.create(
            model=self.name,
            max_tokens=1024,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data,
                            },
                        },
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ],
        )

        raw = json.loads(response.content[0].text)
        return DetectionResult(
            smell=smell_key,
            model=ModelEnum.CLAUDE,
            diagram=diagram_path.name,
            confidence=LikertScaleOption(str(raw["confidence"])),
            explanation=raw["explanation"],
            involved_services=raw.get("involved_services"),
        )
