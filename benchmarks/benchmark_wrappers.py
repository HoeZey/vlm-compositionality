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
            text0, text1 = d_i['caption_0'], d_i['caption_1']
            img0, img1 = d_i['image_0'], d_i['image_1']

            with torch.no_grad(), torch.cuda.amp.autocast():
                out_i0_t0 = model(img0, text0)
                out_i0_t1 = model(img0, text1)
                out_i1_t0 = model(img1, text0)
                out_i1_t1 = model(img1, text1)
            
            scores.append({
                'id': d_i['id'], 
                'i0_t0': out_i0_t0, 'i0_t1': out_i0_t1, 
                'i1_t0': out_i1_t0, 'i1_t1': out_i1_t1
            })

        text_correct_count = 0
        image_correct_count = 0
        group_correct_count = 0
        for result in scores:
            text_correct_count += 1 if text_correct(result) else 0
            image_correct_count += 1 if image_correct(result) else 0
            group_correct_count += 1 if group_correct(result) else 0

            denominator = len(scores)
        

        # return {
        #     'text_score': np.mean([text_correct(s) for s in scores]),
        #     'image_score': np.mean([image_correct(s) for s in scores]),
        #     'group_score': np.mean([group_correct(s) for s in scores])
        # }

        return {
            "text_score": text_correct_count/denominator, 
            "image_score": image_correct_count/denominator,
            "group_score": group_correct_count/denominator
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