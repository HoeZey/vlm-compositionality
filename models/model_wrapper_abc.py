import torch
from abc import ABC, abstractmethod
from PIL.Image import Image
from torch import FloatTensor
from typing import Union


class VLMWrapper(ABC, torch.nn.Module):
    '''
    The forward function of VLModelWrapper should return FloatTensor
    containing the logits per text, i.e. out: R^N_text
    '''
    @abstractmethod
    def predict(imgs: list[Union[Image, torch.Tensor]], texts: list[str]) -> Union[FloatTensor, float]:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass
