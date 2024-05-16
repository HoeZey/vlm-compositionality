from wrappers.prompt import ZeroShotPrompt


class ContrastiveVLMWrapper:
    '''
    The forward function of VLModelWrapper should return FloatTensor
    containing the logits per text, i.e. out: R^N_text
    '''
    def predict_binary():
        pass

    def predict_choice():
        pass


class GenerativeVLMWrapper:
    def predict_binary():
        pass

    def predict_choice():
        pass
    

class Benchmark:
    def evaluate(self, model) -> list[float]: 
        pass
    
    @property
    def name(self) -> str:
        pass
