import json
import os
from pathlib import Path

from openai import OpenAI

from models.base_model import LLMModel
from schema.smell_detection_response_model import DetectionResult, LikertScaleOption, ModelEnum
from utils.utils import encode_image


class GPTCodexModel(LLMModel):
    name = ModelEnum.OPENAI_CODEX.value

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        super().__init__()

    def analyze(self, diagram_path: Path, prompt_text: str, smell_key: str) -> DetectionResult:
        image_data = encode_image(diagram_path)
        media_type = "image/jpeg" if diagram_path.suffix.lower() == ".jpg" else "image/png"

        response = self.client.responses.create(
            model=self.name,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt_text},
                        {
                            "type": "input_image",
                            "image_url": f"data:{media_type};base64,{image_data}",
                        },
                    ],
                }
            ],
            text={"format": {"type": "json_object"}},
        )

        raw = json.loads(response.output_text)
        return DetectionResult(
            smell=smell_key,
            model=ModelEnum.OPENAI_CODEX,
            diagram=diagram_path.name,
            confidence=LikertScaleOption(str(raw["confidence"])),
            explanation=raw["explanation"],
            involved_services=raw.get("involved_services"),
        )
