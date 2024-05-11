from dataclasses import dataclass


@dataclass
class Prompt:
    prompt: str
    positive_answer:str

    def insert_caption(self, caption: str) -> str:
        return self.prompt.replace('<caption>', caption)
    
    def answered_positive(self, out: str) -> float:
        return self.positive_answer in out[len(self.prompt):].lower()

    @classmethod
    def get_prompts_from_dict(cls, prompt_dict):
        return {k: cls(v['prompt'], v['pos_ans']) for k, v in prompt_dict.items()}