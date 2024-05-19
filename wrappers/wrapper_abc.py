import torch
from torch import FloatTensor
from PIL.Image import Image
from wrappers.prompt import Prompt
from abc import ABC, abstractmethod


class VLMWrapper(ABC):
    '''
    The forward function of VLModelWrapper should return FloatTensor
    containing the logits per text, i.e. out: R^N_text
    '''
    @torch.no_grad()
    @abstractmethod
    def predict_match(self, images: list[Image], captions: list[str]) -> FloatTensor:
        raise NotImplmentedError("Implement this method")

    @torch.no_grad()
    @abstractmethod
    def predict_choice(self, images: list[Image], captions: list[str]) -> FloatTensor:
        raise NotImplmentedError("Implement this method")

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplmentedError("Implement this method")


class GenVLMWrapper(VLMWrapper):
    def set_prompt(self, prompt: Prompt) -> None:
        self.prompt = prompt


class Benchmark(ABC):
    @abstractmethod
    def evaluate(self, model) -> list[float]: 
        raise NotImplmentedError("Implement this method")
    
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplmentedError("Implement this method")
