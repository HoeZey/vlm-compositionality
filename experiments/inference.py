import sys
import os
sys.path.append(os.getcwd())
from models.contrastive_model_wrappers import OpenClipWrapper
from benchmarks.benchmark_wrappers import instantiate_benchmarks
from collections import defaultdict


if __name__ == '__main__':
    models = [OpenClipWrapper()]
    benchmarks = instantiate_benchmarks()

    results = defaultdict(dict)
    for model in models:
        for benchmark in benchmarks:
            result = benchmark.evaluate(model)
            results[model.name][benchmark.name] = result
            print(f'{model.name:<5} {benchmark.name:<13} {result}')
    # print(results)