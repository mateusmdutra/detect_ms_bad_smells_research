from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import json

from schema.smell_detection_response_model import DetectionResult

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
DIAGRAMS_DIR = Path(__file__).parent.parent / "diagrams"
RESULTS_DIR = Path(__file__).parent.parent / "results"

PROMPT_FILES: Dict[str, str] = {
    "nano_service": "nano-service.txt",
    "no_api_gateway": "no-api-gateway.txt",
    "shared_database": "shared-database.txt",
    "microservice_greedy": "microservice-greedy.txt",
    "cyclic_dependency": "cyclic-dependency.txt",
}


class LLMModel(ABC):
    name: str

    def load_prompts(self) -> Dict[str, str]:
        prompts = {}
        for smell_key, filename in PROMPT_FILES.items():
            path = PROMPTS_DIR / filename
            with open(path, "r") as f:
                prompts[smell_key] = f.read()
        return prompts

    def load_diagrams(self) -> List[Path]:
        return sorted(DIAGRAMS_DIR.glob("*.png")) + sorted(DIAGRAMS_DIR.glob("*.jpg"))

    def run_experiment(self) -> List[DetectionResult]:
        prompts = self.load_prompts()
        diagrams = self.load_diagrams()

        RESULTS_DIR.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = RESULTS_DIR / f"{self.name}_{timestamp}.json"

        results: List[DetectionResult] = []
        total = len(diagrams) * len(prompts)
        count = 0

        for diagram_path in diagrams:
            for smell_key, prompt_text in prompts.items():
                count += 1
                print(f"[{count}/{total}] {diagram_path.name} | {smell_key}")
                try:
                    result = self.analyze(diagram_path, prompt_text, smell_key)
                    results.append(result)
                    print(f"  -> confidence: {result.confidence.name}")
                except Exception as e:
                    print(f"  ERROR: {e}")

                self._save(results, output_path)

        print(f"\nDone. {len(results)}/{total} results saved to {output_path}")
        return results

    def _save(self, results: List[DetectionResult], output_path: Path) -> None:
        data = [r.model_dump(mode="json") for r in results]
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @abstractmethod
    def analyze(self, diagram_path: Path, prompt_text: str, smell_key: str) -> DetectionResult:
        pass
