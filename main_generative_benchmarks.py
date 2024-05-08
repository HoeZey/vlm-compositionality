import argparse
import os
import json
from benchmarks.aro.main_evaluate_aro import ARO_evaluation, ARO_generative_evaluation
from benchmarks.sugarcrepe.sugarcrepe_evaluation import SugarCrepe_generative_evaluation
from benchmarks.winoground.winoground_evaluation import Winoground_generative_evaluation
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import LlamaTokenizer, AutoModelForCausalLM
import wandb
import torch


parser = argparse.ArgumentParser(description=__doc__)
_AA = parser.add_argument

_AA("--model_list", nargs='+', help="List of models to evaluate.")
_AA("--evaluation_type", help="Evaluation mode to activate. Accuracy overall or text/image/group scores.")
_AA("--no_hard_negatives", help="Evaluation mode in which caption and image pairs are swapped with ones from different examples.")



# BENCHMARKS_LIST = ["aro", "sugarcrepe", "winoground", 'vlchecklist']

# BENCHMARKS_LIST = ["winoground"]
# BENCHMARKS_LIST = ["sugarcrepe"]
BENCHMARKS_LIST = ["aro"]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_TYPE = torch.bfloat16


def main(_A: argparse.Namespace):

    print(f"Model list: {_A.model_list}")
    if _A.evaluation_type == "accuracy_score":
        # PROMPT_LIST = ["gpt4", "gpt4-moretokens", "gpt4-shorterprompt","choices-first", "choices-first-numbers"]
        # PROMPT_LIST = ["gpt4-moretokens"]
        PROMPT_LIST = ["alignment"]
    if _A.evaluation_type == "text_image_group_score":
        # PROMPT_LIST = ["gpt4-evensmallerprompt"]
        PROMPT_LIST = ["gpt4-evensmallerprompt2"]
        # PROMPT_LIST = ["alignment"]
        # PROMPT_LIST = ["gpt4-smallerprompt"]
        # PROMPT_LIST = ["gpt4-shorterprompt"]
    
    
    for model_name in _A.model_list:
        if model_name == "llava-hf/llava-1.5-7b-hf":
            model = LlavaForConditionalGeneration.from_pretrained(model_name).to(DEVICE).eval()
            processor = AutoProcessor.from_pretrained(model_name)
            tokenizer = None
        elif model_name == "blip2_t5":
            model, processor, _ = load_model_and_preprocess(name=model_name, model_type="pretrain_flant5xxl", is_eval=True, device=DEVICE)

        elif model_name == "Salesforce/blip2-opt-2.7b":            
            model = Blip2ForConditionalGeneration.from_pretrained(
                model_name, load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16
            )
            processor = Blip2Processor.from_pretrained(model_name)
            tokenizer = None
        elif model_name == "THUDM/cogvlm-chat-hf":
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=TORCH_TYPE, low_cpu_mem_usage=True, trust_remote_code=True
            ).to(DEVICE).eval()
            processor = None
            tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
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
                    print(model_name)
                    # print(model)
                    benchmark_module = Winoground_generative_evaluation(model_name, model, processor, tokenizer, TORCH_TYPE, DEVICE, prompt_name, _A.evaluation_type, _A.no_hard_negatives)
                    eval_results = benchmark_module.evaluate_winoground()
                    
                    if _A.evaluation_type == "accuracy_score":
                        wandb.log({'Winoground_accuracy' : eval_results["accuracy_score"]})
                    elif _A.evaluation_type == "text_image_group_score":
                        # print("Logging scores...")
                        wandb.log({'Winoground_text_score' : eval_results['text_score']})
                        wandb.log({'Winoground_image_score' : eval_results['image_score']})
                        wandb.log({'Winoground_group_score' : eval_results['group_score']})

                elif benchmark == "aro":
                    benchmark_module = ARO_generative_evaluation(model_name, model, processor, tokenizer, TORCH_TYPE, DEVICE, prompt_name, _A.evaluation_type)
                    eval_results = benchmark_module.evaluate_aro()
                    wandb.log({'ARO_VG_R' : eval_results['ARO_accuracies']['VG_Relation' ]})
                    wandb.log({'ARO_VG_A' : eval_results['ARO_accuracies']['VG_Attribution']})
                    wandb.log({'ARO_coco': eval_results['ARO_accuracies' ]['COCO_Order']})
                    wandb.log({'ARO_Flickr' : eval_results['ARO_accuracies']['Flickr30k_Order']})
                
                elif benchmark == "sugarcrepe":
                    benchmark_module = SugarCrepe_generative_evaluation(model_name, model, processor, tokenizer, TORCH_TYPE, DEVICE, prompt_name, _A.evaluation_type)
                    eval_results = benchmark_module.evaluate_sugarcrepe()
                    wandb.log({'Sugarcrepe_add-obj' : eval_results['SugarCrepe_accuracies']['add_obj']})
                    wandb.log({'Sugarcrepe_add_att' : eval_results['SugarCrepe_accuracies']['add_att']})
                    wandb.log({'Sugarcrepe_replace_obj' : eval_results['SugarCrepe_accuracies']['replace_obj']})
                    wandb.log({'Sugarcrepe_replace_att' : eval_results['SugarCrepe_accuracies']['replace_att']})
                    wandb.log({'Sugarcrepe_replace_rel' : eval_results['SugarCrepe_accuracies']['replace_rel']})
                    wandb.log({'Sugarcrepe_swap_obj' : eval_results['SugarCrepe_accuracies']['swap_obj']}) 
                    wandb.log({'Sugarcrepe_swap_att' : eval_results['SugarCrepe_accuracies']['swap_att']})

                else:
                    raise ValueError(f"Unknown benchmark: {benchmark}")

            # wandb.log({'no-hard-negatives': _A.no_hard_negatives})
            wandb.finish()

    print(eval_results)


if __name__ == "__main__":
    _A = parser.parse_args()
    main(_A)