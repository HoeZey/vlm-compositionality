import argparse
import os
import json
from benchmarks.aro.main_evaluate_aro import ARO_evaluation
from benchmarks.sugarcrepe.sugarcrepe_evaluation import SugarCrepe_evaluation
from benchmarks.winoground.winoground_evaluation import Winoground_evaluation
import wandb

parser = argparse.ArgumentParser(description=__doc__)
_AA = parser.add_argument

_AA("--model_list", nargs='+', help="List of models to evaluate.")
_AA("--pretrained_list", nargs='+', help="List of pretrained checkpoints for the selected models.")
_AA("--output-dir", help="Output directory to save the results.")

BENCHMARKS_LIST = ["aro", "sugarcrepe", "winoground", 'vlchecklist']

# models = ['RN50', 'RN50-quickgelu', RN101-quickgelu, RN50x64, ViT-B-32, ViT-B-32, ViT-B-32,  ViT-B-32-quickgelu, ViT-B-16, ViT-B-16, ViT-B-16,

 
# pretrained = ['openai', 'openai', yfcc15m, openai, openai, laion2b_s34b_b79k, laion2b_e16, openai, openai, laion400m_e31, dfn2b, 



def main(_A: argparse.Namespace):
    for model_name, pretrained in zip(_A.model_list, _A.pretrained_list):
        wandb.init(
            # set the wandb project where this run will be logged
            project="all_benchmarks",
            entity="fomo-vlm-comp",

            # track hyperparameters and run metadata
            config={
            "model": model_name,
            "pretrained": pretrained
            }
            )
        for benchmark in BENCHMARKS_LIST:
            if benchmark == "aro":
                benchmark_module = ARO_evaluation(model_name, pretrained)
                eval_results = benchmark_module.evaluate_open_clip_on_aro()
                wandb.log({'ARO_VG_R' : eval_results['ARO_accuracies']['VG_Relation' ]['Accuracy']})
                wandb.log({'ARO_VG_A' : eval_results['ARO_accuracies']['VG_Attribution']['Accuracy']})
                wandb.log({'ARO_coco': eval_results['ARO_accuracies' ]['COCO_Order']['Accuracy']})
                wandb.log({'ARO_Flickr' : eval_results['ARO_accuracies ']['Flickr30k_Order'] ['Accuracy']})
                
            elif benchmark == "sugarcrepe":
                benchmark_module = SugarCrepe_evaluation(model_name, pretrained)
                eval_results = benchmark_module.evaluate_open_clip_on_sugarcrepe()
                wandb.log({'Sugarcrepe_add-obj' : eval_results['SugarCrepe_accuracies']['add_obj']})
                wandb.log({'Sugarcrepe_add_att' : eval_results['SugarCrepe_accuracies']['add_att']})
                wandb.log({'Sugarcrepe_replace_obj' : eval_results['SugarCrepe_accuracies']['replace_obj']})
                wandb.log({'Sugarcrepe_replace_att' : eval_results['SugarCrepe_accuracies']['replace_att']})
                wandb.log({'Sugarcrepe_replace_rel' : eval_results['SugarCrepe_accuracies']['replace_rel']})
                wandb.log({'Sugarcrepe_swap_obj' : eval_results['SugarCrepe_accuracies']['swap_obj']}) 
                wandb.log({'Sugarcrepe_swap_att' : eval_results['SugarCrepe_accuracies']['swap_att']}) 

            elif benchmark == "winoground":
                benchmark_module = Winoground_evaluation(model_name, pretrained)
                eval_results = benchmark_module.evaluate_open_clip_on_winoground()
                wandb.log({'Winoground_text_score' : eval_results['Winoground_accuracies']['text_score']})
                wandb.log({'Winoground_image_score' : eval_results['Winoground_accuracies']['image_score']})
                wandb.log({'Winoground_group_score' : eval_results['Winoground_accuracies']['group_score']})
            elif benchmark == "vlchecklist":
                pass
            else:
                raise ValueError(f"Unknown benchmark: {benchmark}")

        wandb.finish()
    
    print(eval_results)

    if not os.path.exists(_A.output_DIR):
        os.makedirs(_A.output_DIR, exist_ok=True)
    
    with open(os.path.join(_A.output_DIR, f"{_A.benchmark}_{_A.model_name}_{_A.pretrained}_results.json"), "w") as f:
        json.dump(eval_results, f)


if __name__ == "__main__":
    _A = parser.parse_args()
    main(_A)