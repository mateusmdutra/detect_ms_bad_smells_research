import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.claude_model import ClaudeModel
from models.deepseek_model import DeepseekModel
from models.openai_model import GPTModel
from models.qwen_model import QwenModel

MODELS = {
    "openai": GPTModel,
    "claude": ClaudeModel,
    "deepseek": DeepseekModel,
    "qwen": QwenModel,
}


def main():
    parser = argparse.ArgumentParser(
        description="Run microservice bad smell detection experiment"
    )
    parser.add_argument(
        "model",
        choices=list(MODELS.keys()),
        help="Model to use for the experiment",
    )
    args = parser.parse_args()

    model = MODELS[args.model]()
    print(f"Starting experiment with model: {model.name}\n")
    results = model.run_experiment()
    print(f"\nExperiment complete. {len(results)} results collected.")


if __name__ == "__main__":
    main()
