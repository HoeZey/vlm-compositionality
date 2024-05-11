import sys
import os
sys.path.append(os.getcwd())
import json
import warnings
warnings.filterwarnings("ignore")
from models.generative_model_wrappers import *
from prompts.prompt import Prompt
from benchmarks.benchmark_wrappers import instantiate_benchmarks
from collections import defaultdict

def main() -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open('./prompts/prompts_debug.json', 'r') as f:
        prompt_dict = Prompt.get_prompts_from_dict(json.load(f))
    
    models = [OpenFlamingoWrapper(device)]
    benchmarks = instantiate_benchmarks()

    results = defaultdict(dict)
    for model in models:
        for prompt_type, prompt in prompt_dict.items():
            # wandb.init(s
            #     project="generative_models",
            #     entity="fomo-vlm-comp",
            #     config={
            #         "model": model.name,
            #         "prompt": prompt_type
            #         }
            # )
            for benchmark in benchmarks:
                print(f'{model.name}\t{prompt_type}\t{benchmark.name}')
                model.set_prompt(prompt)
                result = benchmark.evaluate(model)
                results[model.name][benchmark.name] = result
                print(f'{model.name:<5} {benchmark.name:<13} {result}')
                break
    # print(results)

if __name__ == '__main__':
    main()