from dataclasses import dataclass


@dataclass
class Prompt:
    prompt: str
    positive_answer:str
    max_new_tokens: int

    def insert_caption(self, caption: str) -> str:
        return self.prompt.replace('<caption>', caption)
    
    def answered_positive(self, out: str) -> float:
        return self.positive_answer in out[-12:].lower()

    @classmethod
    def get_prompts_from_dict(cls, prompt_dict):
        return {k: cls(v['prompt'], v['pos_ans'], v['max_new_tokens']) for k, v in prompt_dict.items()}