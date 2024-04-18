import torch
import numpy as np
from tqdm import tqdm
from models.model_wrappers import VLModelWrapper
from .utils import *


class Benchmark:
    def evaluate(self, model) -> list[float]: 
        pass
    
    @property
    def name(self) -> str:
        pass


class WinoGroundWrapper:
    def __init__(self) -> None:
        self.data = get_winoground_datset()['test']

    def evaluate(self, model: VLModelWrapper) -> dict[str, float]:
        ''' 
        return:
        {
            'text_score': x,
            'image_score': y,
            'group_score': z
        }
        where 
        - text_score is the accuracy of the model on matching text with the correct image 
        - image_score is the accuracy of the model on matching image with the correct text
        - group_score is the accuracy of the model if both previous pairings are correct
        '''
        scores = []
        for d_i in tqdm(self.data):
            texts = [d_i['caption_0'], d_i['caption_1']]
            img0 = d_i['image_0']
            img1 = d_i['image_1']

            with torch.no_grad(), torch.cuda.amp.autocast():
                out_img0 = model(img0, texts)
                out_img1 = model(img1, texts)
            
            scores.append({'id': d_i['id'], 'out_img0': out_img0, 'out_img1': out_img1})

        return {
            'text_score': np.mean([text_correct(s['out_img0'], s['out_img1']) for s in scores]),
            'image_score': np.mean([image_correct(s['out_img0'], s['out_img1']) for s in scores]),
            'group_score': np.mean([group_correct(s['out_img0'], s['out_img1']) for s in scores])
        }


    @property
    def name(self) -> str:
        return 'winoground'


class VLCheckListWrapper:
    def __init__(self) -> None:
        pass

    def evaluate(self, model) -> list[float]:
        return 0.0
    @property
    def name(self) -> str:
        return 'vl-checklist'


class SugarCrepeWrapper:
    def __init__(self) -> None:
        pass

    def evaluate(self, model) -> list[float]:
        return 0.0

    @property
    def name(self) -> str:
        return 'sugar-crepe'


class AROWrapper:
    def __init__(self) -> None:
        pass

    def evaluate(self, model) -> list[float]:
        return 0.0

    @property
    def name(self) -> str:
        return 'aro'


BENCHMARKS = [WinoGroundWrapper, VLCheckListWrapper, SugarCrepeWrapper, AROWrapper]


def instantiate_benchmarks() -> list[Benchmark]:
    return [benchmark() for benchmark in BENCHMARKS]