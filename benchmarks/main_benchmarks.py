import argparse
import os
import json
from benchmarks.aro.main_evaluate_aro import ARO_evaluation
from benchmarks.sugarcrepe.sugarcrepe_evaluation import SugarCrepe_evaluation
from benchmarks.winoground.winoground_evaluation import Winoground_evaluation

parser = argparse.ArgumentParser(description=__doc__)
_AA = parser.add_argument
_AA("--benchmark", help="Benchmark to run.")
_AA("--model-name", help="OpenAI CLIP model name.")
_AA("--pretrained", help="Pretrained checkpoint for the selected model")
_AA("--output-dir", help="Output directory to save the results.")


def main(_A: argparse.Namespace):
    if _A.benchmark == "aro":
        benchmark_module = ARO_evaluation(_A.model_name, _A.pretrained)
        eval_results = benchmark_module.evaluate_open_clip_on_aro()
    elif _A.benchmark == "sugarcrepe":
        benchmark_module = SugarCrepe_evaluation(_A.model_name, _A.pretrained)
        eval_results = benchmark_module.evaluate_open_clip_on_sugarcrepe()
    elif _A.benchmark == "winoground":
        benchmark_module = Winoground_evaluation(_A.model_name, _A.pretrained)
        eval_results = benchmark_module.evaluate_open_clip_on_winoground()
    elif _A.benchmark == "vlchecklist":
        pass
    else:
        raise ValueError(f"Unknown benchmark: {_A.benchmark}")
    
    print(eval_results)

    if not os.path.exists(_A.output_DIR):
        os.makedirs(_A.output_DIR, exist_ok=True)
    
    with open(os.path.join(_A.output_DIR, f"{_A.benchmark}_{_A.model_name}_{_A.pretrained}_results.json"), "w") as f:
        json.dump(eval_results, f)
