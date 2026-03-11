from typing import List

from openai import BaseModel
from models.claude_model import ClaudeModel
from models.deepseek_model import DeepseekModel
from models.openai_model import GPTModel
from models.qwen_model import QwenModel
from schema.smell_detection_response_model import SmellsDetectionResponse

models: List['BaseModel'] = [
    GPTModel(),
    ClaudeModel(),
    DeepseekModel(),
    QwenModel()
]

architectures = []
results: List['SmellsDetectionResponse'] = []

for arch in architectures:
    for model in models:
        result = model.analyze(arch)
        results.append({
            "architecture": arch.id,
            "model": model.name,
            "result": result
        })