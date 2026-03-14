# Detecting Microservice Bad Smells with LLMs

Research experiment that uses Large Language Models to detect bad smells in microservice architectures from architecture diagrams.

Previous study: [From Detection to Refactoring of Microservice Bad Smells: A Systematic Literature Review](https://journals-sol.sbc.org.br/index.php/jserd/article/view/5224)

---

## Bad Smells Detected

| Smell | Prompt file |
|---|---|
| No API Gateway | `prompts/no-api-gateway.txt` |
| Nano Service | `prompts/nano-service.txt` |
| Shared Database | `prompts/shared-database.txt` |
| Microservice Greedy | `prompts/microservice-greedy.txt` |
| Cyclic Dependency | `prompts/cyclic-dependency.txt` |

---

## Models

| Key | Model |
|---|---|
| `claude` | Claude Opus 4.6 |
| `openai` | GPT-5.4 |
| `deepseek` | DeepSeek Coder V2 |
| `qwen` | Qwen3 Coder Plus |

---

## How the Experiment Works

For each run, one model is selected via command-line. The model then analyzes all 9 architecture diagrams in `/diagrams` against each of the 5 smell-detection prompts — one diagram at a time — resulting in **45 API calls per model**.

```
for each diagram (9):
    for each smell prompt (5):
        call model API with diagram + prompt → DetectionResult
```

Each `DetectionResult` contains:
- `smell` — which bad smell was evaluated
- `model` — model used
- `diagram` — diagram file name
- `confidence` — Likert scale (1–4): `not_present` → `clearly_present`
- `explanation` — model's reasoning
- `involved_services` — services involved (when applicable)

Results are saved incrementally to `results/<model>_<timestamp>.json` after each API call, so progress is not lost if the run fails midway.

---

## Setup

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Configure API keys**

Create a `.env` file at the project root:

```env
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
DEEPSEEK_API_KEY=your_key_here
QWEN_API_KEY=your_key_here
```

---

## Running the Experiment

```bash
python run_experiment.py <model>
```

Where `<model>` is one of: `claude`, `openai`, `deepseek`, `qwen`.

**Examples:**

```bash
python run_experiment.py claude
python run_experiment.py openai
python run_experiment.py deepseek
python run_experiment.py qwen
```

Output is saved to:

```
results/<model-name>_<YYYYMMDD_HHMMSS>.json
```

---

## Project Structure

```
.
├── diagrams/          # Architecture diagram dataset (.png and .jpg)
├── models/            # One file per LLM; BaseModel handles shared logic
├── prompts/           # One .txt prompt per bad smell
├── schema/            # Pydantic output schema (DetectionResult)
├── utils/             # Shared utilities (e.g., image encoding)
├── results/           # Generated output — one JSON file per run
├── run_experiment.py  # Entrypoint
├── requirements.txt
└── .env               # API keys (not committed)
```

---

**Author:** Mateus Dutra <br/>
**Affiliation:** Federal University of Minas Gerais
