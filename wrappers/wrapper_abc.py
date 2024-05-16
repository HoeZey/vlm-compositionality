from abc import ABC, abstractmethod


class VLModelWrapper(ABC, torch.nn.Module):
    '''
    The forward function of VLModelWrapper should return FloatTensor
    containing the logits per text, i.e. out: R^N_text
    '''

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    

class Benchmark:
    def evaluate(self, model) -> list[float]: 
        pass
    
    @property
    def name(self) -> str:
        pass
