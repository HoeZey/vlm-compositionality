import sys
sys.path.append("..")

import pandas as pd
import torch
from torch.utils.data import DataLoader
import pandas as pd
import torch

from torch.utils.data import DataLoader
import open_clip

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
            prompt = "USER: <image>\n Given an image and two candidate captions, which caption is the better description of the given image? Give the final answer in the exact format of \"The answer is A/B.\"\n"
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
        else:
            print("Prompt type not supported!")
        
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)

        # Generate
        generate_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        output = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        output = output.split('ASSISTANT:')[1].strip()
        print(output)
        return output



    @torch.no_grad()
    def blip2_caption_choice(self, image, caption_0, caption_1):
        if self.prompt_name == "gpt4-shorterprompt":
            prompt = "USER: <image>\n Given this image and two candidate captions (A and B), which caption is the better description of the given image? Only give a single character answer - 'A' or 'B'.\n"
            prompt += "A. " + caption_0 + "\n"
            prompt += "B. " + caption_1 + "\n"  
            prompt += "ASSISTANT:"
            max_new_tokens = 35
        else:
            print("Prompt type not supported!")
        
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)

        # Generate
        generate_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        output = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        print(output)
        return output


    @torch.no_grad()
    # def cogvlm_caption_choice(self, image, caption_0, caption_1):
    def cogvlm_caption_choice(self, caption, image):

        if self.prompt_name == "gpt4-shorterprompt":
            prompt = "USER: <image>\n Given this image and two candidate captions (A and B), which caption is the better description of the given image? Only give a single character answer - 'A' or 'B'.\n"
            # prompt += "A. " + caption_0 + "\n"
            # prompt += "B. " + caption_1 + "\n"  
            prompt += "Caption: " + caption.strip() + "\n"
            prompt += "ASSISTANT:"
            max_new_tokens = 35

        elif self.prompt_name == "gpt4":
            prompt = "USER: <image>\n Select whether the image matches the caption. Pay close attention to the word order. (Give a short explanation first, then change to a new line give the final answer in the exact format of: \"The answer is Yes/No.\"))\n"
            prompt += "Caption: " + caption.strip() + "\n"
            prompt += "ASSISTANT:"
            max_new_tokens = 35

        elif self.prompt_name == "gpt4-smallerprompt":
            prompt = "USER: <image>\n Select whether the image matches the caption. Pay close attention to the word order. Give the final answer in the exact format of: \"The answer is either Yes or No\"))\n"
            prompt += "Caption: " + caption.strip() + "\n"
            prompt += "ASSISTANT:"
            max_new_tokens = 35

        elif self.prompt_name == "gpt4-evensmallerprompt":
            prompt = "USER: <image>\n Does the image match the caption?. Pay close attention to the word order. Answer in the exact format of: 'Yes' or 'No'\n"
            prompt += "Caption: " + caption.strip() + "\n"
            prompt += "ASSISTANT:"
            max_new_tokens = 35

        elif self.prompt_name == "gpt4-evensmallerprompt2":
            prompt = "USER: <image>\n Does the image match the caption?. Answer in the format of: \"Yes or No.\"))\n"
            prompt += "Caption: " + caption.strip() + "\n"
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


    def evaluate_aro(self):
        seed = 1
        seed_all(seed)

        dataset_names =["VG_Relation", "VG_Attribution", "COCO_Order", "Flickr30k_Order"]
        download = True
        batch_size = 32
        num_workers = 4
        
        metrics = {}

        if self.model_name == "llava-hf/llava-1.5-7b-hf":
            captioner = self.llava_caption_choice
        elif self.model_name == "Salesforce/blip2-opt-2.7b":
            captioner = self.blip2_caption_choice
        elif self.model_name == "THUDM/cogvlm-chat-hf":
            captioner = self.cogvlm_caption_choice

        for dataset_name in dataset_names:
            dataset = self.load_dataset(dataset_name, image_preprocess=self.processor, download=download)
            
            # For some models we just pass the PIL images, so we'll need to handle them in the collate_fn. 
            collate_fn = _default_collate if self.processor is None else None
            
            #batch
            joint_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn) 

            # scores = self.model.get_retrieval_scores_batched(joint_loader)
            # result_records = dataset.evaluate_scores(scores)
            # df = pd.DataFrame(result_records)

            # mean_acc = df['Accuracy'].mean()

            # results[dataset_name] = {
            #     # "Model": self.model_name,
            #     "Accuracy": mean_acc,
            #     "Seed": seed
            # }   
        print(joint_loader)
        # return {"ARO_accuracies": results}