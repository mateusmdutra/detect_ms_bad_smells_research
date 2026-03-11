from pydantic import BaseModel
from enum import Enum
from typing import List, Optional, Dict

class ResponseConfidenceEnum(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class SmellType(Enum):
    NO_API_GATEWAY = "no_api_gateway"
    NANO_SERVICE = "nano_service"
    SHARED_DATABASE = "shared_database"
    MICROSERVICE_GREEDY = "microservice_greedy"
    CYCLIC_DEPENDENCY = "cyclic_dependency"

class DetectionResult(BaseModel):
    detected: bool
    confidence: ResponseConfidenceEnum
    justification: str
    affected_services: Optional[List[str]] = None
    cycle: Optional[List[str]] = None

class SmellsDetectionResponse(BaseModel):
    smells: Dict[SmellType, DetectionResult]