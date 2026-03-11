from base_model import LLMModel

class GPTModel(LLMModel):

    name = "gpt-5.2-codex"

    def analyze(self, architecture: str):

        response = client.responses.create(
            model=self.name,
            input=build_prompt(architecture),
            response_format={"type": "json_object"}
        )

        return response.output_parsed