import sys
import os
sys.path.append(os.getcwd())
from models.generative_model_wrappers import *
from benchmarks.benchmark_wrappers import instantiate_benchmarks
from collections import defaultdict


if __name__ == '__main__':
    prompt = Prompt('Hello', 'Answer:', 'yes')
    models = [CogVLMWrapper(prompt=prompt)]
    benchmarks = instantiate_benchmarks()

    results = defaultdict(dict)
    for model in models:
        for benchmark in benchmarks:
            result = benchmark.evaluate(model)
            results[model.name][benchmark.name] = result
            print(f'{model.name:<5} {benchmark.name:<13} {result}')
    # print(results)