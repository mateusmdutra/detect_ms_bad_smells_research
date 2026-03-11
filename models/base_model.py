from abc import ABC, abstractmethod

from schema.smell_detection_response_model import SmellsDetectionResponse

class LLMModel(ABC):
    name: str

    @abstractmethod
    def analyze(self, architecture: str) -> SmellsDetectionResponse:
        pass