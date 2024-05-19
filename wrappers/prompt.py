from abc import ABC, abstractmethod
from typing import Union


class ZeroShotPrompt:
    '''
    '''
    def __init__(self, prompt, example_format, positive_answer) -> None: 
        self.prompt = prompt
        self.example_format = example_format
        self.pos_ans = positive_answer

    def insert_caption(self, captions: list[str], labels: list[str]) -> str:
        examples = '\n'.join([
            self.example_format.replace('<caption>', caption) + ("Yes" if label else "No")
            for caption, label in zip(captions, labels)
        ])
        return self.prompt.replace('<examples>', examples)
    
    def answered_positive(self, out: str) -> float:
        return self.pos_ans in out[-12:].lower()

    @classmethod
    def get_prompts_from_dict(cls, prompt_dict):
        return {k: cls(v['prompt'], v['pos_ans'], v['max_new_tokens']) for k, v in prompt_dict.items()}


class NShotPrompt:
    '''
    '''
    def __init__(self, prompt, example_format, positive_answer) -> None: 
        self.prompt = prompt
        self.example_format = example_format
        self.pos_ans = positive_answer

    def insert_caption(self, captions: list[str], labels: list[str]) -> str:
        examples = '\n'.join([
            self.example_format.replace('<caption>', caption) + ("Yes" if label else "No")
            for caption, label in zip(captions, labels)
        ])
        return self.prompt.replace('<examples>', examples)
    
    def answered_positive(self, out: str) -> float:
        return self.pos_ans in out[-12:].lower()

    @classmethod
    def get_prompts_from_dict(cls, prompt_dict):
        return {k: cls(v['prompt'], v['pos_ans'], v['max_new_tokens']) for k, v in prompt_dict.items()}


Prompt = Union[ZeroShotPrompt, NShotPrompt]
