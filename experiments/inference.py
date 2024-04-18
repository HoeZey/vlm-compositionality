import numpy as np
import matplotlib.pyplot as plt
from models.model_wrappers import OpenClipWrapper, OpenFlamingoWrapper
from benchmarks.benchmark_wrappers import instantiate_benchmarks
from collections import defaultdict

models = [OpenClipWrapper()]
benchmarks = instantiate_benchmarks()

results = defaultdict(dict)
for model in models:
    for benchmark in benchmarks:
        result = benchmark.evaluate(model)
        print(f'{model.name:<5} {benchmark.name:<13} {result}')
        results[model.name][benchmark.name] = result

print(results)