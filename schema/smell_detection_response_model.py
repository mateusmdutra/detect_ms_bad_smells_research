from pydantic import BaseModel
from enum import Enum
from typing import List, Optional

class ModelEnum(Enum):
    OPENAI = 'gpt-5.4'
    CLAUDE = 'claude-sonnet-4-6'
    DEEPSEEK = 'mmdutra14_3974/deepseek-ai/DeepSeek-OCR-2-b91bd70c'
    QWEN = 'qwen-vl-max'
    GEMINI = 'gemini-2.5-pro'

class LikertScaleOption(Enum):
    NOT_PRESENT = '1'
    UNLIKELY_PRESENT = '2'
    LIKELY_PRESENT = '3'
    CLEARLY_PRESENT = '4'

class SmellType(Enum):
    NO_API_GATEWAY = "no_api_gateway"
    NANO_SERVICE = "nano_service"
    SHARED_DATABASE = "shared_database"
    MICROSERVICE_GREEDY = "microservice_greedy"
    CYCLIC_DEPENDENCY = "cyclic_dependency"

class DetectionResult(BaseModel):
    smell: str
    model: ModelEnum
    diagram: str
    confidence: LikertScaleOption
    explanation: str
    involved_services: Optional[List[str]] = None