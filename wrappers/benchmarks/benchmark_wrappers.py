import torch
import numpy as np
from tqdm import tqdm
from models.model_wrappers import VLModelWrapper
from .utils import *
from constants import HF_ACCESS_TOKEN




class WinoGroundWrapper:
    def __init__(self) -> None:
        self.data = load_dataset(
            'facebook/winoground', 
            token=HF_ACCESS_TOKEN, 
            trust_remote_code=True
        )['test']

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
                out_i0_t0 = model.predict(img0, text0)
                out_i0_t1 = model.predict(img0, text1)
                out_i1_t0 = model.predict(img1, text0)
                out_i1_t1 = model.predict(img1, text1)
            
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
        
        return {
            'text_score': text_correct_count/denominator, 
            'image_score': image_correct_count/denominator,
            'group_score': group_correct_count/denominator
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
        self.data = {
            'add-obj'    : load_dataset('HuggingFaceM4/SugarCrepe_add_obj', trust_remote_code=True)['test'],
            'add_att'    : load_dataset('HuggingFaceM4/SugarCrepe_add_att', trust_remote_code=True)['test'],
            'replace_obj': load_dataset('HuggingFaceM4/SugarCrepe_replace_obj', trust_remote_code=True)['test'],
            'replace_att': load_dataset('HuggingFaceM4/SugarCrepe_replace_att', trust_remote_code=True)['test'],
            'replace_rel': load_dataset('HuggingFaceM4/SugarCrepe_replace_rel', trust_remote_code=True)['test'],
            'swap_obj'   : load_dataset('HuggingFaceM4/SugarCrepe_swap_obj', trust_remote_code=True)['test'],
            'swap_att'   : load_dataset('HuggingFaceM4/SugarCrepe_swap_att', trust_remote_code=True)['test']
        }

    def evaluate(self, model: VLModelWrapper) -> list[float]:
        results = {}
        for name, data_dict in self.data.items():
            n_correct = 0
            for data in tqdm(data_dict, desc=f'Evaluating {name}'):
                out = model.predict([data['image']], data['tested_labels'])
                n_correct += int(out.argmax() == 0)
            count = len(data_dict)
            results[name] = n_correct / count
        return results

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