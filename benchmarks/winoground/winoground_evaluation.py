from tqdm import tqdm
import open_clip
import torch
import numpy as np
from PIL import Image
import requests
from datasets import load_dataset
from matplotlib import pyplot as plt
from tqdm import tqdm
import random
import re

from transformers import LlamaTokenizerFast

from transformers import AutoTokenizer, LlamaForCausalLM

class Winoground_evaluation:
    """
    This class is used to evaluate the OpenCLIP model on the Winoground dataset
    """
    def __init__(self, model_name, pretrained):
        self.model_name = model_name
        self.pretrained = pretrained

        
    def text_correct(self, result):
        return result["c0_i0"] > result["c1_i0"] and result["c1_i1"] > result["c0_i1"]

    def image_correct(self, result):
        return result["c0_i0"] > result["c0_i1"] and result["c1_i1"] > result["c1_i0"]

    def group_correct(self, result):
        return self.image_correct(result) and self.text_correct(result)


    def evaluate_open_clip_on_winoground(self):
        """ 
        input: model_name (str) - the name of the model to be used (can be choosen out of open_clip.list_pretrained())
                pretrained (str) - the name of the pre-trined weigths corresponding to the model_name 
        return:
        {
            "Winoground_accuracies": {
                "text_score": x,
                "image_score": y,
                "group_score": z
            }
        }
        where 
        - text_score is the accuracy of the model on matching text with the correct image 
        - image_score is the accuracy of the model on matching image with the correct text
        - group_score is the accuracy of the model if both previous pairings are correct
        """
        auth_token = "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" # Replace with an auth token, which you can get from your huggingface account: Profile -> Settings -> Access Tokens -> New Token
        winoground = load_dataset("facebook/winoground", use_auth_token=auth_token, trust_remote_code=True)["test"]

        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        model, _, preprocess  = open_clip.create_model_and_transforms(self.model_name, self.pretrained, device=device)
        tokenizer = open_clip.get_tokenizer(self.model_name)

        from tqdm import tqdm
        winoground_clip_scores = []
        for example in tqdm(winoground):
            text0 = example["caption_0"]
            text1 = example["caption_1"]
            text0 = tokenizer(text0).to(device)    
            text1 = tokenizer(text1).to(device)          


            image0 = preprocess(example["image_0"].convert("RGB")).unsqueeze(0).to(device)
            image1 = preprocess(example["image_1"].convert("RGB")).unsqueeze(0).to(device)

            
            with torch.no_grad(), torch.cuda.amp.autocast():
                
                image0_features = model.encode_image(image0)
                image1_features = model.encode_image(image1)
                image0_features /= image0_features.norm(dim=-1, keepdim=True)
                image1_features /= image1_features.norm(dim=-1, keepdim=True)
                # print("doing img norm")

                text0_features = model.encode_text(text0)
                text0_features /= text0_features.norm(dim=-1, keepdim=True)
                text1_features = model.encode_text(text1)
                text1_features /= text1_features.norm(dim=-1, keepdim=True)
                

                c0_i0 = (100.0 * image0_features @ text0_features.T)
                c0_i1 = (100.0 * image1_features @ text0_features.T)
                c1_i0 = (100.0 * image0_features @ text1_features.T)
                c1_i1 = (100.0 * image1_features @ text1_features.T)



            winoground_clip_scores.append({"id" : example["id"], "c0_i0": c0_i0, "c0_i1": c0_i1, "c1_i0": c1_i0, "c1_i1": c1_i1})

        # evaluating accuracy
    
        text_correct_count = 0
        image_correct_count = 0
        group_correct_count = 0
        for result in winoground_clip_scores:
            text_correct_count += 1 if self.text_correct(result) else 0
            image_correct_count += 1 if self.image_correct(result) else 0
            group_correct_count += 1 if self.group_correct(result) else 0

            denominator = len(winoground_clip_scores)
        
        acc_val = {"text_score": text_correct_count/denominator, "image_score": image_correct_count/denominator, "group_score": group_correct_count/denominator}

        return {"Winoground_accuracies": acc_val}


class Winoground_generative_evaluation:
    """
    This class is defined to evaluate generative models on winoground dataset.
    """

    def __init__(self, 
                 model_name, 
                 model, 
                 processor=None, 
                 tokenizer=None,
                 torch_type=None,
                 device=None,
                 prompt_name=None, 
                 evaluation_type=None,
                 no_hard_negatives=None,
                 n_shot=8
                 ):
        self.model_name = model_name
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.torch_type = torch_type
        self.device = device
        self.prompt_name = prompt_name  
        self.evaluation_type = evaluation_type
        self.no_hard_negatives = no_hard_negatives
        auth_token = "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" # Replace with an auth token, which you can get from your huggingface account: Profile -> Settings -> Access Tokens -> New Tokens
        winoground = load_dataset("facebook/winoground", use_auth_token=auth_token, trust_remote_code=True)["test"]
        random.seed(2023)
        subset_idx = random.sample(range(len(winoground)), 100)
        fewshot_data = []
        for idx in subset_idx:
            fewshot_data.append(winoground[idx])
        # len(subset_idx[:20])
        #taking the first 20 for time purposes
        # subset_idx = subset_idx[:8]
        self.fewshot_data = fewshot_data[:n_shot]

        ## Retrieval augmented generation
        
        
        im1 = Image.open(
        requests.get(
            "http://images.cocodataset.org/val2017/000000039769.jpg", stream=True
            ).raw
        )
        im2 = Image.open(
            requests.get(
                "http://images.cocodataset.org/test-stuff2017/000000028137.jpg",
                stream=True
            ).raw
        )
        im3 = Image.open(
            requests.get(
                "http://images.cocodataset.org/test-stuff2017/000000028352.jpg", 
                stream=True
            ).raw
        )

        im4 = Image.open(
            requests.get(
                "http://images.cocodataset.org/test-stuff2017/000000000442.jpg", 
                stream=True
            ).raw
        )

        im5 = Image.open(
            requests.get(
                "http://images.cocodataset.org/test-stuff2017/000000000448.jpg", 
                stream=True
            ).raw
        )

        caption_1 = "Two cats sleeping on a purple blanket on top of a couch with two tv remotes next to them."
        caption_2 = "A bathroom with a sink, cabinet, mirror and a shower curtain."
        caption_3 = "A table full of salty and sweet food inside a cozy room."
        caption_4 = "A room full of compute screens and some people working on them."
        caption_5 = "A group of women talking about environmental issues while sitting at a table."

        

        rag_fewshot = []
        rag_fewshot.append({"image": im1, "caption": caption_1})
        rag_fewshot.append({"image": im2, "caption": caption_2})
        rag_fewshot.append({"image": im3, "caption": caption_3})
        rag_fewshot.append({"image": im4, "caption": caption_4})
        rag_fewshot.append({"image": im5, "caption": caption_5})

        self.rag_fewshot = rag_fewshot

        ## Retrieval augmented generation with negatives

        im1 = Image.open(
        requests.get(
            "http://images.cocodataset.org/test-stuff2017/000000000870.jpg", stream=True
            ).raw
        )
        im2 = Image.open(
            requests.get(
                "http://images.cocodataset.org/test-stuff2017/000000000527.jpg",
                stream=True
            ).raw
        )
        im3 = Image.open(
            requests.get(
                "http://images.cocodataset.org/test-stuff2017/000000001227.jpg", 
                stream=True
            ).raw
        )

        im4 = Image.open(
            requests.get(
                "http://images.cocodataset.org/test-stuff2017/000000001331.jpg", 
                stream=True
            ).raw
        )

        im5 = Image.open(
            requests.get(
                "http://images.cocodataset.org/test-stuff2017/000000001574.jpg", 
                stream=True
            ).raw
        )

        caption_1 = "The giraffe is on top of the trees."
        caption_2 = "The zebra is outside the cage."
        caption_3 = "The kite is carrying the lady."
        caption_4 = "The ball is in the air, and the boys are running from it."
        caption_5 = "The computer is under the shelf."

        rag_fewshot_negatives = []
        rag_fewshot_negatives.append({"image": im1, "caption": caption_1})
        rag_fewshot_negatives.append({"image": im2, "caption": caption_2})
        rag_fewshot_negatives.append({"image": im3, "caption": caption_3})
        rag_fewshot_negatives.append({"image": im4, "caption": caption_4})
        rag_fewshot_negatives.append({"image": im5, "caption": caption_5})

        self.rag_fewshot_negatives = rag_fewshot_negatives


        # rag_fewshot["example"]
        # rag_fewshot["image_1"] = im1
        # rag_fewshot["image_2"] = im2
        # rag_fewshot["image_3"] = im3
        # rag_fewshot["image_4"] = im4
        # rag_fewshot["image_5"] = im5
        # rag_fewshot["caption_1"] = caption_1
        # rag_fewshot["caption_2"] = caption_2
        # rag_fewshot["caption_3"] = caption_3
        # rag_fewshot["caption_4"] = caption_4
        # rag_fewshot["caption_5"] = caption_5





        
        # self.pretrained = pretrained
            
    def show_example(self, benchmark, idx):
        ax1 = plt.subplot(1, 3, 1)
        ax1.title.set_text('image_0')
        # plt.imshow(benchmark[idx]["image_0"].convert("RGB"))

        ax2 = plt.subplot(1, 3, 2)
        ax2.title.set_text('image_1')
        # plt.imshow(benchmark[idx]["image_1"].convert("RGB"))

        # plt.show()
        print("index:", idx)
        print("caption_0:", benchmark[idx]["caption_0"])
        print("caption_1:", benchmark[idx]["caption_1"])

    ## define the metrics functions
    def text_correct(self, result):
        return result["c0_i0"] > result["c1_i0"] and result["c1_i1"] > result["c0_i1"]

    def image_correct(self, result):
        return result["c0_i0"] > result["c0_i1"] and result["c1_i1"] > result["c1_i0"]

    def group_correct(self, result):
        return self.image_correct(result) and self.text_correct(result)

    @torch.no_grad()
    def llava_image_to_caption(self, image, caption_0, caption_1):
        # prompt = "USER: <image>\nDescribe the image in one sentence. ASSISTANT:"

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
            max_new_tokens = 15

        
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)

        # Generate
        generate_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        output = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        output = output.split('ASSISTANT:')[1]
        return output

    @torch.no_grad()
    def llava_image_to_caption_binary_match(self, caption, image):

        if self.prompt_name == "gpt4":
            prompt = "USER: <image>\n Select whether the image matches the caption. Pay close attention to the word order. (Give a short explanation first, then change to a new line give the final answer in the exact format of: \"The answer is Yes/No.\"))\n"
            prompt += "Caption: " + caption.strip() + "\n"
            prompt += "ASSISTANT:"
            max_new_tokens = 35
        elif self.prompt_name == "alignment":
            prompt = "USER: <image> Does this image entail the description:" 
            prompt += caption.strip() + "?"
            prompt += "ASSISTANT:"
            max_new_tokens = 35

        elif self.prompt_name == "gpt4-smallerprompt":
            prompt = "USER: <image>\n Select whether the image matches the caption. Pay close attention to the word order. Give the final answer in the exact format of: \"The answer is Yes/No.\"))\n"
            prompt += "Caption: " + caption.strip() + "\n"
            prompt += "ASSISTANT:"
            max_new_tokens = 35

        elif self.prompt_name == "gpt4-evensmallerprompt":
            prompt = "USER: Does the following image match the caption?. Pay close attention to the word order. Answer in the format of \"Yes or No.\"\n"
            prompt += f"Image: <image>. Caption: {caption.strip()}. ASSISTANT:"
            max_new_tokens = 3
        elif self.prompt_name == "gpt4-evensmallerprompt2":
            prompt = "USER: Does the following image match the caption?. Answer in the format of \"Yes or No.\"\n"
            prompt += f"Image: <image>. Caption: {caption.strip()}. ASSISTANT:"
            max_new_tokens = 3
        
        elif self.prompt_name == "cot":  # Chain of Thought Prompting (Option 1)
            prompt = ("USER: <image>\nGiven this image and a caption," 
            "does the caption accurately describe of the given image? Analyze each caption against the image." 
            "Begin by describing the key elements visible in the image. Compare these elements with the details mentioned in the caption." 
            "Answer by stating your choice between 'yes' or 'no', and follow with a detailed explanation of why that caption fits best.\n")
            prompt += "Caption:" + caption.strip() + "\n"
            prompt += "ASSISTANT: Answer: {Answer caption here}. Explanation: {Provide a detailed explanation here}."
            max_new_tokens = 500


        elif self.prompt_name == "auto-cot":  # Chain of Thought Prompting ( Option 2 : (Auto-CoT) Best/structure so far)
            prompt = ("USER: <image>\nGiven this image and a caption," 
            "does the caption accurately describe of the given image? Analyze each caption against the image. Think step-by-step"
            "and analyze the caption against the image. Begin by describing the key elements "
            "visible in the image. Then, compare these elements with the details mentioned in "
            "the caption. After providing a detailed explanation of your reasoning, clearly state your final answer as 'Yes' or 'No'.\n")
            prompt += "Caption: " + caption.strip() + "\n"
            prompt += "ASSISTANT:"
            max_new_tokens = 500


        elif self.prompt_name == "cbe-cot":  # Chain of Thought Prompting (Option 3: Criterion-Based Evaluation)
            prompt = ("USER: <image>\nGiven this image and a caption," 
                    "does the caption accurately describe of the given image? Evaluate the caption "
                    "based on the following criteria: Relevance to the image, accuracy of the details, "
                    "and completeness of the description.\n"
                    "Start by describing the key elements visible in the image. Then proceed as follows:\n")
            prompt += "1. Relevance: How well does the caption relate to the key elements you have described? \n"
            prompt += "2. Accuracy: Are the details mentioned in the caption correct as per the image? \n"
            prompt += "3. Completeness: Does the caption cover all the important aspects of the image? \n"
            prompt += "Conclude with your assessment for each caption and state your final answer as 'Yes' or 'No', "
            prompt += "based on the caption score across these criteria.\n"
            prompt += "Caption: " + caption.strip() + "\n"
            prompt += "ASSISTANT: \n"
            max_new_tokens = 500

        elif self.prompt_name == "ltm-cot":  # Chain of Thought Prompting (Option 4: Least-to-Most Strategy)
            prompt = ("USER: <image>\nGiven this image and a caption," 
                    "does the caption accurately describe of the given image? Begin your analysis by identifying "
                    "the most obvious elements and statements in the captions and image. Gradually move to more detailed "
                    "and subtle aspects.\n"
                    "Start by commenting on the general accuracy and relevance of the caption: \n")
            prompt += "1. Initial Impressions: What are your first thoughts on the caption based on the visible elements? \n"
            prompt += "2. Detailed Analysis: Examine closer details and subtleties in the image. How do these influence the accuracy of the caption? \n"
            prompt += "3. Depth of Description: Consider if the caption provides a comprehensive description of the image. \n"
            prompt += "Conclude with your final analysis, synthesizing all points, and state your final answer as 'Yes' or 'No'.\n"
            prompt += "Caption: " + caption.strip() + "\n"
            prompt += "ASSISTANT: \n"
            max_new_tokens = 500

        elif self.prompt_name == "sc-cot":  # Chain of Thought Prompting (Option 5: Self-Consistency)
            prompt = ("USER: <image>\nGiven this image and a caption," 
                    "does the caption accurately describe of the given image? Use a self-consistency approach by reasoning through the problem three times, "
                    "each time trying to verify your previous conclusions. Begin by identifying the key elements visible in the image, then evaluate the caption against these elements.\n")
            prompt += "Cycle 1: Provide your initial analysis and choose between 'Yes' or 'No'.\n"
            prompt += "Cycle 2: Re-examine the key elements and your previous decision. Provide any new insights or changes in your reasoning.\n"
            prompt += "Cycle 3: Final review and confirmation of your choice. Ensure consistency or revise if necessary.\n"
            prompt += "Conclude with your final, consistent decision on the caption and a summary of your reasoning across all cycles.\n"
            prompt += "Caption: " + caption.strip() + "\n"
            prompt += "ASSISTANT: \n"
            max_new_tokens = 500

        # elif self.prompt_name == "few-shot":
        #     prompt = "USER: Does the image match the caption?. Answer in the format of: \"Yes or No.\"))\n"
        #     fewshot_images = []
        #     for x in self.fewshot_data:
        #         c0, c1 = x['caption_0'], x['caption_1']
        #         fewshot_images.append(x['image_0'])
        #         fewshot_images.append(x['image_1'])
        #         prompt += f"<image>. Caption: {c0.strip()}. ASSISTANT: <answer>\n"
        #         prompt += f"<image>. Caption: {c1.strip()}. ASSISTANT: <answer>\n"
        #     prompt += f"<image>. Caption: {caption.strip()}. ASSISTANT: "
        #     max_new_tokens = 1

        elif self.prompt_name == "rag-few-shot":
            prompt = "USER: Does the image match the caption?. Answer in the format of: \"Yes or No.\"))\n"
            fewshot_images = []
            for x in self.rag_fewshot:
                c0 = x['caption']
                fewshot_images.append(x['image'])
                prompt += f"<image>. The caption is: {c0.strip()}. The Caption matches the image, the answer is <YES>.\n"

            prompt += ("USER: \nGiven this image and a caption," 
            "does the caption accurately describe of the given image? Analyze each caption against the image. Think step-by-step"
            "and analyze the caption against the image. Begin by describing the key elements "
            "visible in the image. Then, compare these elements with the details mentioned in "
            "the caption. After providing a detailed explanation of your reasoning, clearly state your final answer as <Yes> or <No>.\n")
            prompt += f"<image>. The caption is: {caption.strip()}. ASSISTANT: "
            max_new_tokens = 10

        elif self.prompt_name == "rag-few-shot-negatives":
            prompt = "USER: Does the image match the caption?. Answer in the format of: \"Yes or No.\"))\n"
            fewshot_images = []
            for x in self.rag_fewshot:
                c0 = x['caption']
                fewshot_images.append(x['image'])
                # fewshot_images.append(x['image_1'])
                prompt += f"<image>. The caption is: {c0.strip()}. The Caption matches the image, the answer is <YES>.\n"

            for x in self.rag_fewshot_negatives:
                c0 = x['caption']
                fewshot_images.append(x['image'])
                # fewshot_images.append(x['image_1'])
                prompt += f"<image>. The caption is: {c0.strip()}. The Caption does not match the image, the answer is <NO>.\n"

            prompt += ("USER: \nGiven this image and a caption," 
            "does the caption accurately describe of the given image? Analyze each caption against the image. Think step-by-step"
            "and analyze the caption against the image. Begin by describing the key elements "
            "visible in the image. Then, compare these elements with the details mentioned in "
            "the caption. After providing a detailed explanation of your reasoning, clearly state your final answer as <Yes> or <No>.\n")
            prompt += f"<image> The description of the Image is: {caption.strip()}. ASSISTANT: "
            max_new_tokens = 10

            # inputs = self.processor(text=prompt, images=fewshot_images + [image], return_tensors="pt").to(self.device)

        if self.prompt_name == "few-shot" or self.prompt_name == "rag-few-shot" or self.prompt_name == "rag-few-shot-negatives":
            inputs = self.processor(text=prompt, images=fewshot_images + [image], return_tensors="pt").to(self.device)
        else:
            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)

        # use_auth_token = "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"
        # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_auth_token=use_auth_token)
        # prompt = "No"
        # inputs_language = tokenizer(prompt, return_tensors="pt")
        # print("inputs_language", inputs_language)
        
        ##YES: id = 22483
        ##NO: id = 1939
    
        # Generate
        generate_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        
        output = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        # print(output)
        output = output.split('ASSISTANT:')[1]
        return output    

    @torch.no_grad()
    def llava_image_to_caption_logits(self, caption, image):
        
        if self.prompt_name == "gpt4":
            prompt = "USER: <image>\n Select whether the image matches the caption. Pay close attention to the word order. (Give a short explanation first, then change to a new line give the final answer in the exact format of: \"The answer is Yes/No.\"))\n"
            prompt += "Caption: " + caption.strip() + "\n"
            prompt += "ASSISTANT:"
            max_new_tokens = 35
        elif self.prompt_name == "alignment":
            prompt = "USER: <image> Does this image entail the description:" 
            prompt += caption.strip() + "?"
            prompt += "ASSISTANT:"
            max_new_tokens = 35

        elif self.prompt_name == "gpt4-smallerprompt":
            prompt = "USER: <image>\n Select whether the image matches the caption. Pay close attention to the word order. Give the final answer in the exact format of: \"The answer is Yes/No.\"))\n"
            prompt += "Caption: " + caption.strip() + "\n"
            prompt += "ASSISTANT:"
            max_new_tokens = 35

        elif self.prompt_name == "gpt4-evensmallerprompt":
            prompt = "USER: Does the following image match the caption?. Pay close attention to the word order. Answer in the format of \"Yes or No.\"\n"
            prompt += f"Image: <image>. Caption: {caption.strip()}. ASSISTANT:"
            max_new_tokens = 35
        elif self.prompt_name == "gpt4-evensmallerprompt2":
            prompt = "USER: Does the following image match the caption?. Answer in the format of \"Yes or No.\"\n"
            prompt += f"Image: <image>. Caption: {caption.strip()}. ASSISTANT:"
            max_new_tokens = 35
        elif self.prompt_name == "cot":
            prompt = "USER: Does the following image match the caption?. Pay close attention to the word order. Think step-by-step. Answer in the format of \"Yes or No.\", then give a short explanation.\n"
            prompt += f"Image: <image>. Caption: {caption.strip()}. ASSISTANT:"
            max_new_tokens = 50



        if self.prompt_name == "few-shot":
            inputs = self.processor(text=prompt, images=fewshot_images + [image], return_tensors="pt").to(self.device)
        else:
            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        
        outputs = self.model(**inputs)
        logits = outputs.logits.squeeze()
        yes_logits = torch.mean(logits[:, 22483]) ## 22483 is the token id for 'Yes' based on llama2 tokenizer
        no_logtis = torch.mean(logits[:, 1939]) ## 1939 is the token id for 'No' based on llama2 tokenizer
        
        # print("yes_logits", yes_logits.shape)
        # print("no_logits", no_logtis.shape)
        # print("logits", logits.shape)

        return yes_logits

    @torch.no_grad()
    def BLIP2_image_to_caption_binary_match(self, caption, image):

        if self.prompt_name == "gpt4":
            prompt = "Question: \n Select whether the image matches the caption. Pay close attention to the word order. (Give a short explanation first, then change to a new line give the final answer in the exact format of: \"The answer is Yes/No.\"))\n"
            prompt += caption.strip() + "\n"
            prompt += "Answer:"
            max_new_tokens = 35

        elif self.prompt_name == "gpt4-smallerprompt":
            prompt = "Question: \n Select whether the image matches the caption. Pay close attention to the word order. Give the final answer in the exact format of: \"The answer is Yes/No.\"))\n"
            prompt += "Caption: " + caption.strip() + "\n"
            prompt += "Answer:"
            max_new_tokens = 35
        
        elif self.prompt_name == "gpt4-evensmallerprompt":
            prompt = "Question: \n Does the image match the caption?. Pay close attention to the word order. Answer in the format of: \"Yes or No.\"))\n"
            prompt += "Caption: " + caption.strip() + "\n"
            prompt += "Answer:"
            max_new_tokens = 35
        
        elif self.prompt_name == "gpt4-evensmallerprompt2":
            # prompt1 = "Question: \n Does the image match the caption?. Answer in the format of: \"Yes or No.\"))\n"
            # prompt1 += "Caption: " + caption.strip() + "\n"
            # prompt1 += "Answer:"
            prompt = "Question: \n Does the image match the caption?. Answer in the format of: \"Yes or No.\"))\n"
            prompt += "Caption: " + caption + "\n"
            prompt += "Answer:"
            max_new_tokens = 35
            # print("prompt1", prompt1)
            # print("prompt2", prompt2)

        elif self.prompt_name == "alignment":
            prompt = "Question: Does this image entail the description:" 
            prompt += caption.strip() + "?"
            prompt += "Answer:"
            max_new_tokens = 35

        ##icl arises form data
        ##why BLIP2 fails? dataset

        # self.model.to(device)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")


        # inputs = self.tokenizer([prompt], padding=True, return_tensors="pt")
        # text_features = self.model.get_text_features(**inputs)

    
        
        # outputs = self.model(**inputs)

        # language_model = self.model.language_model.ff
        # print("language_model ff", language_model)
        
        # logits = outputs.logits
        # vision_out0 = outputs.vision_outputs
        # vision_out = outputs.qformer_outputs
        # lm_outs = outputs.language_model_outputs
 
        # print("vision_out before Q-former shape", vision_out0.last_hidden_state.shape)
        # print("vision_out after Q-former shape", vision_out.last_hidden_state.shape)

        # print("lm out keys", outputs.language_model_outputs.keys())
        # print("lm out past_key 0 shape", outputs.language_model_outputs.past_key_values[0][0].shape)
        # print("lm out past_key 1 0 shape", outputs.language_model_outputs.past_key_values[1][0].shape)
        # print("lm out past_key -1 0 shape", outputs.language_model_outputs.past_key_values[-1][0].shape)  
        # print("lm out past_key -1 -1 shape", outputs.language_model_outputs.past_key_values[-1][-1].shape)   
        # print("lm wte?", lm_outs.wte)   

        # print("lm out logits shape ", outputs.language_model_outputs.logits.shape)
        
        # print("logits", logits)

        # print("outs.language_model_outputs", outs.language_model_outputs)
        # print("outs.language_model_outputs shape", outs.language_model_outputs.shape)

        # print("vision_out last hidden", vision_out.last_hidden_state)
        # print("-"*10)
        # print("text_out last hidden", text_out.last_hidden_state)

        # print("vision_out shape", vision_out.last_hidden_state.shape)
        # print("text_out shape", text_out.last_hidden_state.shape)
        # print("outs.logits shape", outs.logits.shape)
        # print()
        # logits = logits.squeeze()
        # logits = logits.mean()

        # print("mean logits", logits) 


        # return outs.logits
        
        
        # Generate
        generate_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        # print("generate_ids.logits", generate_ids.logits)
        output = self.processor.decode(generate_ids[0], skip_special_tokens=True)
        # print("output.logits", output.logits)
        # output = output.split('Answer:')[1]
        return output

    @torch.no_grad()
    def BLIP2_image_to_caption_logits(self, caption, image):
        
        if self.prompt_name == "alignment":
            prompt = "Question: Does this image entail the description:" 
            prompt += caption.strip() + "?"
            prompt += "Answer:"
            max_new_tokens = 35
            
        elif self.prompt_name == "gpt4-smallerprompt":
            prompt = "Question: \n Select whether the image matches the caption. Pay close attention to the word order. Give the final answer in the exact format of: \"The answer is Yes/No.\"))\n"
            prompt += "Caption: " + caption.strip() + "\n"
            prompt += "Answer:"
            max_new_tokens = 35
        ##icl arises form data
        ##why BLIP2 fails? dataset

        # self.model.to(device)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")

        # contrastive:
        use_auth_token = "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"
        tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased", use_auth_token=use_auth_token)
        prompt = "Yes"
        inputs_language = tokenizer(prompt, return_tensors="pt")
        print("inputs_language", inputs_language)

        use_auth_token = "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"
        tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased", use_auth_token=use_auth_token)
        prompt = "No"
        inputs_language = tokenizer(prompt, return_tensors="pt")
        print("inputs_language", inputs_language)

        outputs = self.model(**inputs)
        logits = outputs.logits.squeeze()

        # print("logits.shape", logits.shape)
        yes_logits = torch.mean(logits[:, 2748]) ## 1037 is the token id for 'A' based on bert tokenizer
        no_logits = torch.mean(logits[:, 2053]) ## 1038 is the token id for 'B' based on bert tokenizer
        # print("a_logits", a_logits)
        # print("b_logits", b_logits)
        # print("a_logits.shape", a_logits.shape)
        # print("b_logits.shape", b_logits.shape)

        return yes_logits
    
    @torch.no_grad()
    def cogvlm_image_to_caption_binary_match(self, caption, image):

        if self.prompt_name == "gpt4":
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
            prompt = "USER: Does the following image match the caption?. Pay close attention to the word order. Answer in the format of \"Yes or No.\"\n"
            prompt += f"Image: <image>. Caption: {caption.strip()}. ASSISTANT:"
            max_new_tokens = 3
        elif self.prompt_name == "gpt4-evensmallerprompt2":
            prompt = "USER: Does the following image match the caption?. Answer in the format of \"Yes or No.\"\n"
            prompt += f"Image: <image>. Caption: {caption.strip()}. ASSISTANT:"
            max_new_tokens = 3
        elif self.prompt_name == "cot":
            prompt = "USER: Does the following image match the caption?. Pay close attention to the word order. Think step-by-step. Answer in the format of \"Yes or No.\", then give a short explanation.\n"
            prompt += f"Image: <image>. Caption: {caption.strip()}. ASSISTANT:"
            max_new_tokens = 50

        input_by_model = self.model.build_conversation_input_ids(self.tokenizer, query=prompt, images=[image])
        inputs = {
            'input_ids': input_by_model['input_ids'].unsqueeze(0).to(self.device),
            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(self.device),
            'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(self.device),
            'images': [[input_by_model['images'][0].to(self.device).to(self.torch_type)]] if image is not None else None,
        }
        if 'cross_images' in input_by_model and input_by_model['cross_images']:
            inputs['cross_images'] = [[input_by_model['cross_images'][0].to(self.device).to(self.torch_type)]]


        # Contrastive step
        # lm = self.model.language_model.model(**inputs)
        # print("lm", lm)

        # outputs = self.model(**inputs)
        # print("outs keys", outputs.keys())
        # logits = outputs.logits
        # print("logits", logits.shape)

        # Generate
        gen_kwargs = {"max_length": 2048,
                      "do_sample": False} # "temperature": 0.9
        with torch.no_grad():
            output = self.model.generate(**inputs, **gen_kwargs)
            output = output[:, inputs['input_ids'].shape[1]:]
            output = self.tokenizer.decode(output[0])

        output = output.split("</s>")[0]
        return output

    @torch.no_grad()
    def cogvlm_image_to_caption_logits(self, caption, image):
        
        if self.prompt_name == "gpt4":
            prompt = "USER: <image>\n Select whether the image matches the caption. Pay close attention to the word order. (Give a short explanation first, then change to a new line give the final answer in the exact format of: \"The answer is Yes/No.\"))\n"
            prompt += "Caption: " + caption.strip() + "\n"
            prompt += "ASSISTANT:"
            max_new_tokens = 35
        elif self.prompt_name == "alignment":
            prompt = "USER: <image> Does this image entail the description:" 
            prompt += caption.strip() + "?"
            prompt += "ASSISTANT:"
            max_new_tokens = 35

        elif self.prompt_name == "gpt4-smallerprompt":
            prompt = "USER: <image>\n Select whether the image matches the caption. Pay close attention to the word order. Give the final answer in the exact format of: \"The answer is Yes/No.\"))\n"
            prompt += "Caption: " + caption.strip() + "\n"
            prompt += "ASSISTANT:"
            max_new_tokens = 35

        elif self.prompt_name == "gpt4-evensmallerprompt":
            prompt = "USER: Does the following image match the caption?. Pay close attention to the word order. Answer in the format of \"Yes or No.\"\n"
            prompt += f"Image: <image>. Caption: {caption.strip()}. ASSISTANT:"
            max_new_tokens = 35
        elif self.prompt_name == "gpt4-evensmallerprompt2":
            prompt = "USER: Does the following image match the caption?. Answer in the format of \"Yes or No.\"\n"
            prompt += f"Image: <image>. Caption: {caption.strip()}. ASSISTANT:"
            max_new_tokens = 35
        elif self.prompt_name == "cot":
            prompt = "USER: Does the following image match the caption?. Pay close attention to the word order. Think step-by-step. Answer in the format of \"Yes or No.\", then give a short explanation.\n"
            prompt += f"Image: <image>. Caption: {caption.strip()}. ASSISTANT:"
            max_new_tokens = 50

        input_by_model = self.model.build_conversation_input_ids(self.tokenizer, query=prompt, images=[image])
        inputs = {
            'input_ids': input_by_model['input_ids'].unsqueeze(0).to(self.device),
            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(self.device),
            'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(self.device),
            'images': [[input_by_model['images'][0].to(self.device).to(self.torch_type)]] if image is not None else None,
        }
        if 'cross_images' in input_by_model and input_by_model['cross_images']:
            inputs['cross_images'] = [[input_by_model['cross_images'][0].to(self.device).to(self.torch_type)]]


        # if self.prompt_name == "few-shot":
        #     inputs = self.processor(text=prompt, images=fewshot_images + [image], return_tensors="pt").to(self.device)
        # else:
        #     inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)

        outputs = self.model(**inputs)
        logits = outputs.logits.squeeze()
        yes_logits = torch.mean(logits[:, 22483]) ## 22483 is the token id for 'Yes' based on llama2 tokenizer
        no_logtis = torch.mean(logits[:, 1939]) ## 1939 is the token id for 'No' based on llama2 tokenizer

        return yes_logits

    def evaluate_winoground(self):

        auth_token = "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" # Replace with an auth token, which you can get from your huggingface account: Profile -> Settings -> Access Tokens -> New Token
        winoground = load_dataset("facebook/winoground", use_auth_token=auth_token, trust_remote_code=True)["test"]

        ##Evaluation

        ##images are all winoground images
        random.seed(2023)
        subset_idx = random.sample(range(len(winoground)), 300)
        # subset_idx = range(len(winoground))
        subset_idx = subset_idx[:20]
        #taking the first 20 for time purposes
        
        if self.evaluation_type == "logits":
            text_correct_count = 0
            image_correct_count = 0
            group_correct_count = 0
            total = 0
            image_caption_match_results = {}

            for idx in tqdm(subset_idx):
                image_0 = winoground[idx]["image_0"].convert("RGB")
                # image_1 = winoground[idx]["image_1"].convert("RGB")
                caption_0 = winoground[idx]["caption_0"]
                image_1 = winoground[idx]["image_1"].convert("RGB")
                caption_1 = winoground[idx]["caption_1"]
                # print("caption_1", caption_1)
                print ("Example: #", total)
                # self.show_example(benchmark=winoground, idx=idx)
                result = {}
                # try:
                ## map string results to nemurical
                if self.model_name == "llava-hf/llava-1.5-7b-hf": 
                    captioner = self.llava_image_to_caption_logits

                elif self.model_name == "Salesforce/blip2-opt-2.7b":
                    captioner = self.BLIP2_image_to_caption_logits
                
                elif self.model_name == "THUDM/cogvlm-chat-hf":
                    captioner = self.cogvlm_image_to_caption_logits
                
                
                result["c0_i0"] = captioner(caption_0, image_0)
                result["c0_i1"] = captioner(caption_0, image_1)
                result["c1_i0"] = captioner(caption_1, image_0)
                result["c1_i1"] = captioner(caption_1, image_1)

                text_correct_count += 1 if self.text_correct(result) else 0
                image_correct_count += 1 if self.image_correct(result) else 0
                group_correct_count += 1 if self.group_correct(result) else 0

                total += 1
                print ("Current Acc: {}/{} = {}%\n".format(group_correct_count, total, group_correct_count / total * 100))


        if self.evaluation_type == "text_image_group_score":
            text_correct_count = 0
            image_correct_count = 0
            group_correct_count = 0
            total = 0
            image_caption_match_results = {}

            for idx in tqdm(subset_idx):
                image_0 = winoground[idx]["image_0"].convert("RGB")
                # image_1 = winoground[idx]["image_1"].convert("RGB")
                caption_0 = winoground[idx]["caption_0"]
                print("caption_0", caption_0)
                if self.no_hard_negatives:
                    # print("using no Hard negatives")
                    if idx+1 < len(winoground):
                        print("idx+1", idx+1)
                        image_1 = winoground[idx+1]["image_1"].convert("RGB")
                        caption_1 = winoground[idx+1]["caption_1"]
                        print("caption_1", caption_1)
                    else:
                        print("idx-1", idx-1)
                        image_1 = winoground[idx-1]["image_1"].convert("RGB")
                        caption_1 = winoground[idx-1]["caption_1"]
                        print("caption_1", caption_1)
                else:
                    image_1 = winoground[idx]["image_1"].convert("RGB")
                    caption_1 = winoground[idx]["caption_1"]
                    print("caption_1", caption_1)
                    
                print ("Example: #", total)
                # self.show_example(benchmark=winoground, idx=idx)
                result = {}
                # try:
                ## map string results to nemurical
                if self.model_name == "llava-hf/llava-1.5-7b-hf":
                    captioner = self.llava_image_to_caption_binary_match

                elif self.model_name == "Salesforce/blip2-opt-2.7b":
                    captioner = self.BLIP2_image_to_caption_binary_match
                
                elif self.model_name == "THUDM/cogvlm-chat-hf":
                    captioner = self.cogvlm_image_to_caption_binary_match
                
                else:
                    raise ValueError(f"Unknown model name: {self.model_name}")

                # if self.contrastive:
                #     c0_i0 = captioner(caption_0, image_0)
                #     c0_i1 = captioner(caption_0, image_1)
                #     c1_i0 = captioner(caption_1, image_0)
                #     c1_i1 = captioner(caption_1, image_1)

                
                ans_c0_i0 = captioner(caption_0, image_0)
                image_caption_match_results[str(idx)+"_c0_i0"] = ans_c0_i0
                print ("Match between C0 and I0: ", ans_c0_i0.lower())
                                
                match = re.search(' YES ', ans_c0_i0) or re.search('<YES>', ans_c0_i0)
                if match:
                    result["c0_i0"] = 1.0
                else:
                    result["c0_i0"] = 0.0

                ans_c0_i1 = captioner(caption_0, image_1)
                image_caption_match_results[str(idx)+"_c0_i1"] = ans_c0_i1
                print ("Match between C0 and I1: ", ans_c0_i1)
                match = re.search(' YES ', ans_c0_i0) or re.search('<YES>', ans_c0_i0)
                if match:
                    result["c0_i0"] = 1.0
                else:
                    result["c0_i0"] = 0.0
    


                ans_c1_i0 = captioner(caption_1, image_0)
                image_caption_match_results[str(idx)+"_c1_i0"] = ans_c1_i0
                print ("Match between C1 and I0: ", ans_c1_i0)
                if "yes" in ans_c1_i0[:10].lower():
                    result["c1_i0"] = 1.0
                else:
                    result["c1_i0"] = 0.0

                ans_c1_i1 = captioner(caption_1, image_1)
                image_caption_match_results[str(idx)+"_c1_i1"] = ans_c1_i1
                print ("Match between C1 and I1: ", ans_c1_i1)
                if "yes" in ans_c1_i1[:10].lower():
                    result["c1_i1"] = 1.0
                else:
                    result["c1_i1"] = 0.0

                print ("result: ", result)

                text_correct_count += 1 if self.text_correct(result) else 0
                image_correct_count += 1 if self.image_correct(result) else 0
                group_correct_count += 1 if self.group_correct(result) else 0

                total += 1
                print ("Current Acc: {}/{} = {}%\n".format(group_correct_count, total, group_correct_count / total * 100))


        
        return {"text_score": text_correct_count/total, "image_score": image_correct_count/total, "group_score": group_correct_count/total}
