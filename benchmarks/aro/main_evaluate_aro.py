import os
import sys
sys.path.append("..")

import pandas as pd
import torch
from torch.utils.data import DataLoader, default_collate
import pandas as pd
import torch
from tqdm import tqdm
import re
from torch.utils.data import DataLoader
import open_clip
from torchvision import transforms

from benchmarks.aro.misc import seed_all, _default_collate
from benchmarks.aro.model_zoo.clip_models import CLIPWrapper

    
class ARO_evaluation:
    """
    This class is used to evaluate the OpenCLIP model on the ARO datasets
    """
    def __init__(self, model_name, pretrained) -> None:
        self.model_name = model_name
        self.pretrained = pretrained


    def load_model(self, model_name, pretrained, device):
        model, _, image_preprocess = open_clip.create_model_and_transforms(model_name, pretrained, device=device)
        model = model.eval()
        clip_model = CLIPWrapper(model_name, model, device) 
        return clip_model, image_preprocess


    def load_dataset(self, dataset_name, image_preprocess=None, text_perturb_fn=None, image_perturb_fn=None, download=False, *args, **kwargs):
        """
        Helper function that returns a dataset object with an evaluation function. 
        dataset_name: Name of the dataset.
        image_preprocess: Preprocessing function for images.
        text_perturb_fn: A function that takes in a string and returns a string. This is for perturbation experiments.
        image_perturb_fn: A function that takes in a PIL image and returns a PIL image. This is for perturbation experiments.
        download: Whether to allow downloading images if they are not found.
        """
        if dataset_name == "VG_Relation": 
            from benchmarks.aro.dataset_zoo.aro_datasets import get_visual_genome_relation
            return get_visual_genome_relation(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, 
                                            image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
        elif dataset_name == "VG_Attribution":
            from benchmarks.aro.dataset_zoo.aro_datasets import get_visual_genome_attribution
            return get_visual_genome_attribution(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, 
                                                image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
        elif dataset_name == "COCO_Order":
            from benchmarks.aro.dataset_zoo.aro_datasets import get_coco_order
            return get_coco_order(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, 
                                image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
        elif dataset_name == "Flickr30k_Order":
            from benchmarks.aro.dataset_zoo.aro_datasets import get_flickr30k_order
            return get_flickr30k_order(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, 
                                    image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
        else:
            raise ValueError(f"Unknown dataset {dataset_name}")


    def evaluate_open_clip_on_aro(self):
        seed = 1
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        seed_all(seed)
        dataset_names =["VG_Relation", "VG_Attribution", "COCO_Order", "Flickr30k_Order"]
        download = True
        batch_size = 32
        num_workers = 4
        
        model, image_preprocess = self.load_model(self.model_name, self.pretrained, device)
        results = {}

        for dataset_name in dataset_names:
            dataset = self.load_dataset(dataset_name, image_preprocess=image_preprocess, download=download)
            
            # For some models we just pass the PIL images, so we'll need to handle them in the collate_fn. 
            collate_fn = _default_collate if image_preprocess is None else None
            
            #batch
            joint_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn) 

            scores = model.get_retrieval_scores_batched(joint_loader)
            result_records = dataset.evaluate_scores(scores)
            df = pd.DataFrame(result_records)

            mean_acc = df['Accuracy'].mean()

            results[dataset_name] = {
                "Model": self.model_name,
                "Accuracy": mean_acc,
                "Seed": seed
            }   
    
        return {"ARO_accuracies": results}



class ARO_generative_evaluation:
    """
    This class is used to evaluate the Generative models on the ARO datasets
    """

    def __init__(self, 
                 model_name, 
                 model, 
                 processor=None, 
                 tokenizer=None,
                 torch_type=None,
                 device=None,
                 prompt_name=None, 
                 evaluation_type=None
                 ):
        self.model_name = model_name
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.torch_type = torch_type
        self.device = device
        self.prompt_name = prompt_name  
        self.evaluation_type = evaluation_type

        with open('./fewshot/captions_dalle.json', 'r') as f:
            captions_pairs = json.load(f)

        synthetic_examples = []
        for captions in captions_pairs.values():
            example = {}
            for i, (img_file, capts) in enumerate(captions.items()):
                example['image'] = Image.open(f'./fewshot/images/{img_file}')
                example['caption_A'] = capts[0]
                example['caption_B'] = capts[1]
            synthetic_examples.append(example)

        self.synthetic_examples = synthetic_examples


    def load_dataset(self, dataset_name, image_preprocess=None, text_perturb_fn=None, image_perturb_fn=None, download=False, *args, **kwargs):
        """
        Helper function that returns a dataset object with an evaluation function. 
        dataset_name: Name of the dataset.
        image_preprocess: Preprocessing function for images.
        text_perturb_fn: A function that takes in a string and returns a string. This is for perturbation experiments.
        image_perturb_fn: A function that takes in a PIL image and returns a PIL image. This is for perturbation experiments.
        download: Whether to allow downloading images if they are not found.
        """
        if dataset_name == "VG_Relation": 
            from benchmarks.aro.dataset_zoo.aro_datasets import get_visual_genome_relation
            return get_visual_genome_relation(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, 
                                            image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
        elif dataset_name == "VG_Attribution":
            from benchmarks.aro.dataset_zoo.aro_datasets import get_visual_genome_attribution
            return get_visual_genome_attribution(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, 
                                                image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
        elif dataset_name == "COCO_Order":
            from benchmarks.aro.dataset_zoo.aro_datasets import get_coco_order
            return get_coco_order(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, 
                                image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
        elif dataset_name == "Flickr30k_Order":
            from benchmarks.aro.dataset_zoo.aro_datasets import get_flickr30k_order
            return get_flickr30k_order(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, 
                                    image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
        else:
            raise ValueError(f"Unknown dataset {dataset_name}")


    @torch.no_grad()
    def llava_caption_choice(self, image, caption_0, caption_1):

        if self.prompt_name == "gpt4":
            prompt = "USER: <image>\n Given the image and two candidate captions, which caption is the better description of the given image? (Give a short explanation first, then change to a new line give the final answer in the exact format of: \"The answer is A/B.\")\n"
            prompt += "A. " + caption_0.strip() + "\n"
            prompt += "B. " + caption_1.strip() + "\n"
            prompt += "ASSISTANT:"
            max_new_tokens = 15
        
        elif self.prompt_name == "gpt4-moretokens":
            prompt = "USER: \n Given the image and two candidate captions, which caption is the better description of the given image? (Give a short explanation first, then change to a new line give the final answer in the exact format of: \"The answer is A/B.\")\n"
            prompt += "A. " + caption_0.strip() + "\n"
            prompt += "B. " + caption_1.strip() + "\n"
            prompt += "ASSISTANT:"
            max_new_tokens = 35

        elif self.prompt_name == "gpt4-shorterprompt":
            prompt = "USER: <image>\n Given this image and two candidate captions (A and B), which caption is the better description of the given image? Only give a single character answer - 'A' or 'B'.\n"
            prompt += "A." + caption_0 + "\n"
            prompt += "B." + caption_1 + "\n"  
            prompt += "ASSISTANT:"
            max_new_tokens = 35

        elif self.prompt_name == "choices-first":
            prompt = "USER: <image>\n There are two choices:\n"
            prompt += "A." + caption_0 + "\n"
            prompt += "B." + caption_1 + "\n"  
            prompt += "Given an image and two candidate captions, which caption is the better description of the given image? Give the final answer in the exact format of \"The answer is A/B.\"\n"
            prompt += "ASSISTANT:"
            max_new_tokens = 35
        
        elif self.prompt_name == "choices-first-numbers":
            prompt = "USER: <image>\n There are two choices:\n"
            prompt += "1." + caption_0 + "\n"
            prompt += "2." + caption_1 + "\n"  
            prompt += "Given an image and two candidate captions, which caption is the better description of the given image? Give the final answer in the exact format of \"The answer is 1./2..\"\n"
            prompt += "ASSISTANT:"
            max_new_tokens = 35

        elif self.prompt_name == "old-cot": #Chain of Thought Prompting (Old Version)
            prompt = "USER: <image>\n Given this image and two candidate captions (A and B), which caption is the better description of the given image? Pay close attention to the word order. Think step-by-step. Answer in the format of \"<A> or <B>\", then give a short explanation.\n"
            prompt += "A. " + caption_0 + "\n"
            prompt += "B. " + caption_1 + "\n"  
            prompt += "ASSISTANT:"
            max_new_tokens = 50

        elif self.prompt_name == "cot":  # Chain of Thought Prompting (Option 1)
            prompt = ("USER: <image>\nGiven this image and two candidate captions (A and B)," 
            "which caption is the better description of the given image? Analyze each caption against the image." 
            "Begin by describing the key elements visible in the image. Compare these elements with the details mentioned in each caption to determine which one matches better." 
            "Answer by stating your choice between 'A' or 'B', then explicitly print the chosen caption, and follow with a detailed explanation of why that caption fits best.\n")
            prompt += "A. " + caption_0.strip() + "\n"
            prompt += "B. " + caption_1.strip() + "\n"
            prompt += "ASSISTANT: Chosen Caption: {Chosen caption here}. Explanation: {Provide a detailed explanation here}."
            max_new_tokens = 500


        elif self.prompt_name == "auto-cot":  # Chain of Thought Prompting ( Option 2 : (Auto-CoT) Best/structure so far)
            prompt = ("USER: <image>\nGiven this image and two candidate captions (A and B), "
              "which caption is the better description of the given image? Think step-by-step "
              "and analyze each caption against the image. Begin by describing the key elements "
              "visible in the image. Then, compare these elements with the details mentioned in "
              "each caption to determine which one matches better. After providing a detailed "
              "explanation of your reasoning, clearly state your final answer as <A> or <B>.\n")
            prompt += "A. " + caption_0.strip() + "\n"
            prompt += "B. " + caption_1.strip() + "\n"
            prompt += "ASSISTANT:"
            max_new_tokens = 500


        elif self.prompt_name == "cbe-cot":  # Chain of Thought Prompting (Option 3: Criterion-Based Evaluation)
            prompt = ("USER: <image>\nGiven this image and two candidate captions (A and B), "
                    "which caption is the better description of the given image? Evaluate each caption "
                    "based on the following criteria: Relevance to the image, accuracy of the details, "
                    "and completeness of the description.\n"
                    "Start by describing the key elements visible in the image. Then proceed as follows:\n")
            prompt += "1. Relevance: How well does each caption relate to the key elements you have described? \n"
            prompt += "2. Accuracy: Are the details mentioned in each caption correct as per the image? \n"
            prompt += "3. Completeness: Does the caption cover all the important aspects of the image? \n"
            prompt += "Conclude with your assessment for each caption and state your final answer as '<A>' or '<B>', "
            prompt += "based on which caption scores better across these criteria.\n"
            prompt += "A. " + caption_0.strip() + "\n"
            prompt += "B. " + caption_1.strip() + "\n"
            prompt += "ASSISTANT: \n"
            max_new_tokens = 500

        elif self.prompt_name == "ltm-cot":  # Chain of Thought Prompting (Option 4: Least-to-Most Strategy)
            prompt = ("USER: <image>\nGiven this image and two candidate captions (A and B), "
                    "which caption is the better description of the given image? Begin your analysis by identifying "
                    "the most obvious elements and statements in the captions and image. Gradually move to more detailed "
                    "and subtle aspects as you compare each caption.\n"
                    "Start by commenting on the general accuracy and relevance of both captions: \n")
            prompt += "1. Initial Impressions: What are your first thoughts on each caption based on the visible elements? \n"
            prompt += "2. Detailed Analysis: Examine closer details and subtleties in the image. How do these influence the accuracy of each caption? \n"
            prompt += "3. Depth of Description: Consider which caption provides a deeper and more comprehensive description of the image. \n"
            prompt += "Conclude with your final analysis, synthesizing all points, and state your final answer as <A> or <B>.\n"
            prompt += "A. " + caption_0.strip() + "\n"
            prompt += "B. " + caption_1.strip() + "\n"
            prompt += "ASSISTANT: \n"
            max_new_tokens = 500

        elif self.prompt_name == "sc-cot":  # Chain of Thought Prompting (Option 5: Self-Consistency)
            prompt = ("USER: <image>\nGiven this image and two candidate captions (A and B), "
                    "which caption is the better description of the given image? Use a self-consistency approach by reasoning through the problem three times, "
                    "each time trying to verify your previous conclusions. Begin by identifying the key elements visible in the image, then evaluate each caption against these elements.\n")
            prompt += "Cycle 1: Provide your initial analysis and choose between 'A' or 'B'.\n"
            prompt += "Cycle 2: Re-examine the key elements and your previous decision. Provide any new insights or changes in your reasoning.\n"
            prompt += "Cycle 3: Final review and confirmation of your choice. Ensure consistency or revise if necessary.\n"
            prompt += "Conclude with your final, consistent decision on the best caption and a summary of your reasoning across all cycles and state your final answer as <A> or <B>.\n"
            prompt += "A. " + caption_0.strip() + "\n"
            prompt += "B. " + caption_1.strip() + "\n"
            prompt += "ASSISTANT: \n"
            max_new_tokens = 500

        elif self.prompt_name == "few-shot": #Inspect & Adjust this
            prompt = "USER: Does the image match the caption?. Answer in the format of: \"Yes or No.\"))\n"
            fewshot_images = []
            for x in self.fewshot_data:
                c0, c1 = x['caption_0'], x['caption_1']
                fewshot_images.append(x['image_0'])
                fewshot_images.append(x['image_1'])
                prompt += f"<image>. Caption: {c0.strip()}. ASSISTANT: <answer>\n"
                prompt += f"<image>. Caption: {c1.strip()}. ASSISTANT: <answer>\n"
            prompt += f"<image>. Caption: {caption.strip()}. ASSISTANT: "
            max_new_tokens = 1

            inputs = self.processor(text=prompt, images=fewshot_images + [image], return_tensors="pt").to(self.device)

        if self.prompt_name == "few-shot":
            inputs = self.processor(text=prompt, images=fewshot_images + [image], return_tensors="pt").to(self.device)
        else:
            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)

        
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)

        # Generate
        generate_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        output = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        output = output.split('ASSISTANT:')[1].strip()
        print(output)
        return output

    @torch.no_grad()
    def llava_caption_logits(self, image, caption_0, caption_1):
        if self.prompt_name == "gpt4-shorterprompt":
            prompt = "USER: <image>\n Given this image and two candidate captions (A and B), which caption is the better description of the given image? Only give a single character answer - 'A' or 'B'.\n"
            prompt += "A. " + caption_0 + "\n"
            prompt += "B. " + caption_1 + "\n"  
            prompt += "ASSISTANT:"
            max_new_tokens = 35

        elif self.prompt_name == "synth":
            prompt = "USER: Does the image match the caption?.\n"
            fewshot_images = []
            for x in self.synthetic_examples:
                c0 = x['caption_A']
                c1 = x['caption_B']
                fewshot_images.append(x['image'])
                prompt += "A. " + c0.strip() + "\n"
                prompt += "B. " + c1.strip() + "\n"
                prompt += f"<image>. Caption A. matches the image, the answer is <A.>.\n"

            prompt += ("USER: <image>\nGiven this image and two candidate captions (A and B), "
              "which caption is the better description of the given image? Think step-by-step "
              "and analyze each caption against the image. Begin by describing the key elements "
              "visible in the image. Then, compare these elements with the details mentioned in "
              "each caption to determine which one matches better. After providing a detailed "
              "explanation of your reasoning, clearly state your final answer as <A> or <B>.\n")
            prompt += "A. " + c0.strip() + "\n"
            prompt += "B. " + c1.strip() + "\n"
            prompt += f"<image> ASSISTANT: "
            max_new_tokens = 500
        else:
            print("Prompt type not supported!")

        
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)

        
        # Contrast logits
        outputs = self.model(**inputs)
        logits = outputs.logits.squeeze()
        a_logits = torch.mean(logits[:, 319]) ## 319 is the token id for 'A' based on llama2 tokenizer
        b_logits = torch.mean(logits[:, 350]) ## 350 is the token id for 'B' based on llama2 tokenizer
        a_logits = torch.mean(logits[:, 319]) ## 319 is the token id for 'A' based on llama2 tokenizer
        b_logits = torch.mean(logits[:, 350]) ## 350 is the token id for 'B' based on llama2 tokenizer

        return a_logits, b_logits 

    @torch.no_grad()
    def blip2_caption_choice(self, image, caption_0, caption_1): #same as sugarcrepe
        if self.prompt_name == "gpt4-shorterprompt":
            prompt = "USER: <image>\n Given this image and two candidate captions (A and B), which caption is the better description of the given image? Only give a single character answer - 'A' or 'B'.\n"
            prompt += "A. " + caption_0 + "\n"
            prompt += "B. " + caption_1 + "\n"  
            prompt += "ASSISTANT:"
            max_new_tokens = 35
        else:
            print("Prompt type not supported!")
        
        # prompts = [prompt] * image.size(0) #old version used when I tried the batching method 

        # inputs = self.processor(text=prompts, images=image, return_tensors="pt").to(self.device)
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)


        # print("Number of prompts:", len(prompts))
        # print("Text input shape:", inputs['input_ids'].shape)
        # print("Image input shape:", inputs['pixel_values'].shape)
        # Generate
        generate_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        output = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        print(output)
        return output

    @torch.no_grad()
    def blip2_caption_logits(self, image, caption_0, caption_1):
        if self.prompt_name == "gpt4-shorterprompt":
            prompt = "USER: Given this image and two candidate captions (A and B), which caption is the better description of the given image? Only give a single character answer - 'A' or 'B'.\n"
            prompt += "A. " + caption_0 + "\n"
            prompt += "B. " + caption_1 + "\n"  
            prompt += "ASSISTANT:"
            max_new_tokens = 35

        elif self.prompt_name == "synth":
            prompt = "USER: Does the image match the caption?.\n"
            fewshot_images = []
            for x in self.synthetic_examples:
                c0 = x['caption_A']
                c1 = x['caption_B']
                fewshot_images.append(x['image'])
                prompt += "A. " + c0.strip() + "\n"
                prompt += "B. " + c1.strip() + "\n"
                prompt += f"Caption A. matches the image, the answer is <A.>.\n"

            prompt += ("USER: Given this image and two candidate captions (A and B), "
              "which caption is the better description of the given image? Think step-by-step "
              "and analyze each caption against the image. Begin by describing the key elements "
              "visible in the image. Then, compare these elements with the details mentioned in "
              "each caption to determine which one matches better. After providing a detailed "
              "explanation of your reasoning, clearly state your final answer as <A> or <B>.\n")
            prompt += "A. " + c0.strip() + "\n"
            prompt += "B. " + c1.strip() + "\n"
            prompt += f"ASSISTANT: "
            max_new_tokens = 500
        else:
            print("Prompt type not supported!")

        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)

        outputs = self.model(**inputs)
        logits = outputs.logits.squeeze()

        # print("logits.shape", logits.shape)
        a_logits = torch.mean(logits[:, 1037]) ## 1037 is the token id for 'A' based on bert tokenizer
        b_logits = torch.mean(logits[:, 1038]) ## 1038 is the token id for 'B' based on bert tokenizer

        return a_logits, b_logits        

    @torch.no_grad()
    def cogvlm_caption_choice(self, image, caption_0, caption_1):
        if self.prompt_name == "gpt4-shorterprompt":
            prompt = "USER: <image>\n Given this image and two candidate captions (A and B), which caption is the better description of the given image? Only give a single character answer - 'A' or 'B'.\n"
            prompt += "A. " + caption_0 + "\n"
            prompt += "B. " + caption_1 + "\n"  
            prompt += "ASSISTANT:"
            max_new_tokens = 35
        else:
            print("Prompt type not supported!")
        
        input_by_model = self.model.build_conversation_input_ids(self.tokenizer, query=prompt, images=[image])
        inputs = {
            'input_ids': input_by_model['input_ids'].unsqueeze(0).to(self.device),
            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(self.device),
            'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(self.device),
            'images': [[input_by_model['images'][0].to(self.device).to(self.torch_type)]] if image is not None else None,
        }
        if 'cross_images' in input_by_model and input_by_model['cross_images']:
            inputs['cross_images'] = [[input_by_model['cross_images'][0].to(self.device).to(self.torch_type)]]

        # Generate
        gen_kwargs = {"max_length": 2048,
                      "do_sample": False} # "temperature": 0.9

        output = self.model.generate(**inputs, **gen_kwargs)
        output = output[:, inputs['input_ids'].shape[1]:]
        output = self.tokenizer.decode(output[0])

        print(output)
        output = output.split("</s>")[0]
        return output
    
    @torch.no_grad()
    def cogvlm_caption_logits(self, image, caption_0, caption_1):
        if self.prompt_name == "gpt4-shorterprompt":
            prompt = "USER: <image>\n Given this image and two candidate captions (A and B), which caption is the better description of the given image? Only give a single character answer - 'A' or 'B'.\n"
            prompt += "A. " + caption_0 + "\n"
            prompt += "B. " + caption_1 + "\n"  
            prompt += "ASSISTANT:"
            max_new_tokens = 35

        elif self.prompt_name == "synth":
            prompt = "USER: Does the image match the caption?.\n"
            fewshot_images = []
            for x in self.synthetic_examples:
                c0 = x['caption_A']
                c1 = x['caption_B']
                fewshot_images.append(x['image'])
                prompt += "A. " + c0.strip() + "\n"
                prompt += "B. " + c1.strip() + "\n"
                prompt += f"<image>. Caption A. matches the image, the answer is <A.>.\n"

            prompt += ("USER: <image>\nGiven this image and two candidate captions (A and B), "
              "which caption is the better description of the given image? Think step-by-step "
              "and analyze each caption against the image. Begin by describing the key elements "
              "visible in the image. Then, compare these elements with the details mentioned in "
              "each caption to determine which one matches better. After providing a detailed "
              "explanation of your reasoning, clearly state your final answer as <A> or <B>.\n")
            prompt += "A. " + c0.strip() + "\n"
            prompt += "B. " + c1.strip() + "\n"
            prompt += f"<image> ASSISTANT: "
            max_new_tokens = 500
        else:
            print("Prompt type not supported!")
        
        input_by_model = self.model.build_conversation_input_ids(self.tokenizer, query=prompt, images=[image])
        inputs = {
            'input_ids': input_by_model['input_ids'].unsqueeze(0).to(self.device),
            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(self.device),
            'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(self.device),
            'images': [[input_by_model['images'][0].to(self.device).to(self.torch_type)]] if image is not None else None,
        }
        if 'cross_images' in input_by_model and input_by_model['cross_images']:
            inputs['cross_images'] = [[input_by_model['cross_images'][0].to(self.device).to(self.torch_type)]]

        outputs = self.model(**inputs)
        logits = outputs.logits.squeeze()
        a_logits = torch.mean(logits[:, 319]) ## 319 is the token id for 'A' based on llama2 tokenizer
        b_logits = torch.mean(logits[:, 350]) ## 350 is the token id for 'B' based on llama2 tokenizer

        return a_logits, b_logits        

    def evaluate_aro(self, subsets, resume_from_checkpoint=True):
        print(subsets)
        seed = 1
        seed_all(seed)

        dataset_names = subsets
        download = True
        batch_size = 1
        num_workers = 0 #chnage this to 4 when finished debugging
        
        metrics = {}
        if self.evaluation_type == 'logits':
            if self.model_name == "llava-hf/llava-1.5-7b-hf":
                captioner = self.llava_caption_logits 
            elif self.model_name == "Salesforce/blip2-opt-2.7b":
                captioner = self.blip2_caption_logits 
            elif self.model_name == "THUDM/cogvlm-chat-hf":
                captioner = self.cogvlm_caption_logits 
        else:
            if self.model_name == "llava-hf/llava-1.5-7b-hf":
                captioner = self.llava_caption_choice 
            elif self.model_name == "Salesforce/blip2-opt-2.7b":
                captioner = self.blip2_caption_choice
            elif self.model_name == "THUDM/cogvlm-chat-hf":
                captioner = self.cogvlm_caption_choice

        for dataset_name in dataset_names:
            dataset = self.load_dataset(dataset_name, download=download)
            
            # For some models we just pass the PIL images, so we'll need to handle them in the collate_fn. 
            collate_fn = _default_collate if self.processor is None else None
            
            #batch
            # joint_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

            model_name_short = self.model_name.split("/")[1].split('-')[0]
            log_file_path = f'./outputs/log_run/{model_name_short}/aro/{self.evaluation_type}_{dataset_name}_log.csv'
                        
            use_existing_file = os.path.exists(log_file_path) and resume_from_checkpoint
            if use_existing_file:
                with open(log_file_path, 'r') as f:
                    start = int(f.readlines()[-1].split(',')[0]) + 1
            else:
                start = 0
            print(dataset_name, 'i_start', start)
            with open(log_file_path, 'a+') as f:
                if not use_existing_file:
                    f.write('id,correct\n')
                for i, example in tqdm(enumerate(dataset[start:])):
                    if i < start:
                        continue
                    image_options = example['image_options']
                    caption_options = example['caption_options']                
                    if self.evaluation_type == 'logits':
                        answerA, answerB = captioner(image_options[0], caption_options[0], caption_options[1])
                        correct = int(answerA > answerB)
                    elif 'cot' in self.prompt_name:
                        answer = captioner(image_options[0], caption_options[0], caption_options[1])
                        match = re.search('<A>', answer)
                        if match :
                            correct = 1
                        elif re.search(' A ', answer) and not re.search('Caption A', answer):
                            correct = 1
                        else:
                            correct = 0
                    else:
                        answer = captioner(image_options[0], caption_options[0], caption_options[1])
                        if answer[0].lower() == 'a':
                            correct = 1
                        else:
                            correct = 0
                    f.write(f'{i},{correct}\n')

            metrics[dataset_name] = pd.read_csv(log_file_path)['correct'].mean()

        print(metrics)
        return {"ARO_accuracies": metrics}