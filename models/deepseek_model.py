import json
import os
from pathlib import Path

from openai import OpenAI

from models.base_model import LLMModel
from schema.smell_detection_response_model import DetectionResult, LikertScaleOption, ModelEnum
from utils.utils import encode_image


class DeepseekModel(LLMModel):
    name = ModelEnum.DEEPSEEK.value

    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
        super().__init__()

    def analyze(self, diagram_path: Path, prompt_text: str, smell_key: str) -> DetectionResult:
        image_data = encode_image(diagram_path)
        media_type = "image/jpeg" if diagram_path.suffix.lower() == ".jpg" else "image/png"

        response = self.client.chat.completions.create(
            model=self.name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{image_data}"},
                        },
                    ],
                }
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )

        raw = json.loads(response.choices[0].message.content)
        return DetectionResult(
            smell=smell_key,
            model=ModelEnum.DEEPSEEK,
            diagram=diagram_path.name,
            confidence=LikertScaleOption(str(raw["confidence"])),
            explanation=raw["explanation"],
            involved_services=raw.get("involved_services"),
        )
