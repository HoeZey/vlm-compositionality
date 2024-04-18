import argparse
from benchmarks.aro.main_evaluate_aro import evaluate_open_clip_on_aro
from benchmarks.sugarcrepe.sugarcrepe_evaluation import evaluate_open_clip_on_sugarcrepe
from benchmarks.winoground.winoground_evaluation import Winoground_evaluation

parser = argparse.ArgumentParser(description=__doc__)
_AA = parser.add_argument
_AA("--benchmark", help="Benchmark to run.")
_AA("--model-name", help="OpenAI CLIP model name.")
_AA("--pretrained", help="Pretrained checkpoint for the selected model")


def main(_A: argparse.Namespace):
    if _A.benchmark == "aro":
        eval_results = evaluate_open_clip_on_aro(_A.model_name, _A.pretrained)
    elif _A.benchmark == "sugarcrepe":
        eval_results = evaluate_open_clip_on_sugarcrepe(_A.model_name, _A.pretrained)
    elif _A.benchmark == "winoground":
        eval_results = evaluate_open_clip_on_winoground(_A.model_name, _A.pretrained)
    else:
        raise ValueError(f"Unknown benchmark: {_A.benchmark}")