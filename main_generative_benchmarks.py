import argparse
import os
import json
from benchmarks.aro.main_evaluate_aro import ARO_evaluation
from benchmarks.sugarcrepe.sugarcrepe_evaluation import SugarCrepe_evaluation

from benchmarks.winoground.generativestuff.LLaVa import Winoground_generative_evaluation
import wandb



parser = argparse.ArgumentParser(description=__doc__)
_AA = parser.add_argument

_AA("--model_list", nargs='+', help="List of models to evaluate.")

# BENCHMARKS_LIST = ["aro", "sugarcrepe", "winoground", 'vlchecklist']

BENCHMARKS_LIST = ["winoground"]

# models = ['RN50', 'RN50-quickgelu', RN101-quickgelu, RN50x64, ViT-B-32, ViT-B-32, ViT-B-32,  ViT-B-32-quickgelu, ViT-B-16, ViT-B-16, ViT-B-16,

 
# pretrained = ['openai', 'openai', yfcc15m, openai, openai, laion2b_s34b_b79k, laion2b_e16, openai, openai, laion400m_e31, dfn2b, 



def main(_A: argparse.Namespace):

    print(f"Model list: {_A.model_list}")
    
    for model_name in _A.model_list:
        for benchmark in BENCHMARKS_LIST:
            if benchmark == "winoground":
                benchmark_module = Winoground_generative_evaluation(model_name)
                eval_results = benchmark_module.evaluate_winoground_LLava()

            else:
                raise ValueError(f"Unknown benchmark: {benchmark}")
    

    print(eval_results)




if __name__ == "__main__":
    _A = parser.parse_args()
    main(_A)