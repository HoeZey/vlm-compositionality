import argparse
import os
import json
from benchmarks.aro.main_evaluate_aro import ARO_evaluation, ARO_generative_evaluation
from benchmarks.sugarcrepe.sugarcrepe_evaluation import SugarCrepe_evaluation
from benchmarks.winoground.winoground_evaluation import Winoground_generative_evaluation
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import wandb
import torch


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
        # PROMPT_LIST = ["gpt4", "gpt4-smallerprompt"]
        PROMPT_LIST = ["gpt4-smallerprompt"]
    
    
    for model_name in _A.model_list:
        if model_name == "llava-hf/llava-1.5-7b-hf":
            model = LlavaForConditionalGeneration.from_pretrained(model_name)
            processor = AutoProcessor.from_pretrained(model_name)
        elif model_name == "Salesforce/blip2-opt-2.7b":            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
            model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-2.7b", device_map={"": 0}, torch_dtype=torch.float16
            )  # doctest: +IGNORE_RESULT
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
            for benchmark in BENCHMARKS_LIST:                                   
                if benchmark == "winoground":
                    benchmark_module = Winoground_generative_evaluation(model_name, model, processor, prompt_name, _A.evaluation_type)
                    eval_results = benchmark_module.evaluate_winoground_LLava()
                    if _A.evaluation_type == "accuracy":
                        wandb.log({'Winoground_accuracy' : eval_results["accuracy"]})
                    elif _A.evaluation_type == "text_image_group_score":
                        # print("Logging scores...")
                        wandb.log({'Winoground_text_score' : eval_results['text_score']})
                        wandb.log({'Winoground_image_score' : eval_results['image_score']})
                        wandb.log({'Winoground_group_score' : eval_results['group_score']})

                elif benchmark == "aro":
                    benchmark_module = ARO_generative_evaluation(model, processor, prompt_name)
                    eval_results = benchmark_module.evaluate_winoground_LLava()
                    wandb.log({'ARO_VG_R' : eval_results['ARO_accuracies']['VG_Relation' ]['Accuracy']})
                    wandb.log({'ARO_VG_A' : eval_results['ARO_accuracies']['VG_Attribution']['Accuracy']})
                    wandb.log({'ARO_coco': eval_results['ARO_accuracies' ]['COCO_Order']['Accuracy']})
                    wandb.log({'ARO_Flickr' : eval_results['ARO_accuracies']['Flickr30k_Order'] ['Accuracy']})
                
                elif benchmark == "sugarcrepe":
                    benchmark_module = SugarCrepe_evaluation(model, processor, prompt_name)
                    eval_results = benchmark_module.evaluate_open_clip_on_sugarcrepe()
                    wandb.log({'Sugarcrepe_add-obj' : eval_results['SugarCrepe_accuracies']['add_obj']})
                    wandb.log({'Sugarcrepe_add_att' : eval_results['SugarCrepe_accuracies']['add_att']})
                    wandb.log({'Sugarcrepe_replace_obj' : eval_results['SugarCrepe_accuracies']['replace_obj']})
                    wandb.log({'Sugarcrepe_replace_att' : eval_results['SugarCrepe_accuracies']['replace_att']})
                    wandb.log({'Sugarcrepe_replace_rel' : eval_results['SugarCrepe_accuracies']['replace_rel']})
                    wandb.log({'Sugarcrepe_swap_obj' : eval_results['SugarCrepe_accuracies']['swap_obj']}) 
                    wandb.log({'Sugarcrepe_swap_att' : eval_results['SugarCrepe_accuracies']['swap_att']}) 


                else:
                    raise ValueError(f"Unknown benchmark: {benchmark}")
                
            wandb.finish()


    print(eval_results)




if __name__ == "__main__":
    _A = parser.parse_args()
    main(_A)