from models.base_model import LLMModel

class ClaudeModel(LLMModel):

    name = "claude-sonnet-4.5"

    def analyze(self, architecture: str):
        response = anthropic_client.messages.create(
            model=self.name,
            messages=[{"role": "user", "content": build_prompt(architecture)}]
        )

        return parse_json(response)