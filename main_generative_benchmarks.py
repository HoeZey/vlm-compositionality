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
_AA("--evaluation_type", help="Evaluation mode to activate. Accuracy overall or text/image/group scores.")


# BENCHMARKS_LIST = ["aro", "sugarcrepe", "winoground", 'vlchecklist']

BENCHMARKS_LIST = ["winoground"]




def main(_A: argparse.Namespace):

    print(f"Model list: {_A.model_list}")
    if _A.evaluation_type == "accuracy":
        PROMPT_LIST = ["gpt4", "gpt4-moretokens", "gpt4-shorterprompt","choices-first", "choices-first-numbers"]
    if _A.evaluation_type == "text_image_group_score":
        PROMPT_LIST = ["gpt4"]
    
    
    for model_name in _A.model_list:
        for benchmark in BENCHMARKS_LIST:
            if benchmark == "winoground":
                for prompt_name in PROMPT_LIST:
                    wandb.init(
                    # set the wandb project where this run will be logged
                    project="generative_models",
                    entity="fomo-vlm-comp",

                        # track hyperparameters and run metadata
                        config={
                        "model": model_name,
                        "prompt": prompt_name
                        }
                        )
                    benchmark_module = Winoground_generative_evaluation(model_name, prompt_name, _A.evaluation_type)
                    eval_results = benchmark_module.evaluate_winoground_LLava()
                    wandb.log({'Winoground' : benchmark_module.evaluate_winoground_LLava()})
                    # wandb.log({'Evaluation_type': _A.evaluation_type})
                    wandb.finish()
            else:
                raise ValueError(f"Unknown benchmark: {benchmark}")
    

    print(eval_results)




if __name__ == "__main__":
    _A = parser.parse_args()
    main(_A)