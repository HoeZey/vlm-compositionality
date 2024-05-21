from tqdm import tqdm
import torch
import open_clip
from datasets import load_dataset

from transformers import AutoTokenizer, LlamaForCausalLM, BertTokenizer


class SugarCrepe_evaluation:
    """
    This class is used to evaluate the OpenCLIP model on the SugarCrepe dataset
    """
    def __init__(self, model_name, pretrained, auth_token=""):
        self.model_name = model_name
        self.pretrained = pretrained
        self.auth_token = auth_token
    

    def load_model(self, model_name, pretrained, device):
        model, _, transform = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained,
            cache_dir=None,
            device=device
        )
        model = model.to(device)
        tokenizer = open_clip.get_tokenizer(model_name)
        model.eval()
        return model, tokenizer, transform


    @torch.no_grad()
    def text_retrieval(self, pos_text, neg_text, image, model, tokenizer, transform, device):
        pos_text = tokenizer(pos_text).to(device)
        pos_text_embedding = model.encode_text(pos_text, normalize=True)
        neg_text = tokenizer(neg_text).to(device)
        neg_text_embedding = model.encode_text(neg_text, normalize=True)
        image_embedding = model.encode_image(transform(image).unsqueeze(dim=0).to(device), normalize=True)
        pos_score = pos_text_embedding @ image_embedding.t()
        neg_score = neg_text_embedding @ image_embedding.t()
        return 1 if pos_score.item() > neg_score.item() else 0


    def evaluate(self, dataset, model, tokenizer, transform, device):
        metrics = {}
        for c, data_dict in dataset.items():
            correct_cnt = 0
            for data in tqdm(data_dict, desc=f'evaluating {c}'):
                correct = self.text_retrieval(data['tested_labels'][0], data['tested_labels'][1], 
                                        data['image'], model, tokenizer, transform, device)
                correct_cnt += correct
            count = len(data_dict)
            metrics[c] = correct_cnt / count
        return metrics


    def evaluate_open_clip_on_sugarcrepe(self):
        """
        return:
        {
            "SugarCrepe_accuracies": {
                "add-obj": x,
                "add_att": y,
                "replace_obj": z,
                "replace_att": a,
                "replace_rel": b,
                "swap_obj": c,
                "swap_att": d
            }
        }
        where
        - add-obj is the accuracy of the model on adding an object to the text
        - add_att is the accuracy of the model on adding an attribute to the text
        - replace_obj is the accuracy of the model on replacing an object in the text
        - replace_att is the accuracy of the model on replacing an attribute in the text
        - replace_rel is the accuracy of the model on replacing a relation in the text
        - swap_obj is the accuracy of the model on swapping objects in the text
        - swap_att is the accuracy of the model on swapping attributes in the text
        """
        
        sugarcrepe = {
            'add_obj'    : load_dataset("HuggingFaceM4/SugarCrepe_add_obj", use_auth_token=self.auth_token, trust_remote_code=True)["test"],
            'add_att'    : load_dataset("HuggingFaceM4/SugarCrepe_add_att", use_auth_token=self.auth_token, trust_remote_code=True)["test"],
            'replace_obj': load_dataset("HuggingFaceM4/SugarCrepe_replace_obj", use_auth_token=self.auth_token, trust_remote_code=True)["test"],
            'replace_att': load_dataset("HuggingFaceM4/SugarCrepe_replace_att", use_auth_token=self.auth_token, trust_remote_code=True)["test"],
            'replace_rel': load_dataset("HuggingFaceM4/SugarCrepe_replace_rel", use_auth_token=self.auth_token, trust_remote_code=True)["test"],
            'swap_obj'   : load_dataset("HuggingFaceM4/SugarCrepe_swap_obj", use_auth_token=self.auth_token, trust_remote_code=True)["test"],
            'swap_att'   : load_dataset("HuggingFaceM4/SugarCrepe_swap_att", use_auth_token=self.auth_token, trust_remote_code=True)["test"],
        }

        print(f"Evaluating {self.model_name}-{self.pretrained}")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model, tokenizer, transform = self.load_model(self.model_name, self.pretrained, device)

        acc_val = self.evaluate(sugarcrepe, model, tokenizer, transform, device)
        return {"SugarCrepe_accuracies": acc_val}
    


class SugarCrepe_generative_evaluation:
    """
    This class is used to evaluate the OpenCLIP model on the SugarCrepe dataset
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

    @torch.no_grad()
    def llava_caption_choice(self, image, caption_0, caption_1):
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
    def blip2_caption_choice(self, image, caption_0, caption_1):
        if self.prompt_name == "gpt4-shorterprompt":
            prompt = "USER: \n Given this image and two candidate captions (A and B), which caption is the better description of the given image? Only give a single character answer - 'A' or 'B'.\n"
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
    def blip2_caption_logits(self, image, caption_0, caption_1):
        if self.prompt_name == "gpt4-shorterprompt":
            prompt = "USER: \n Given this image and two candidate captions (A and B), which caption is the better description of the given image? Only give a single character answer - 'A' or 'B'.\n"
            prompt += "A. " + caption_0 + "\n"
            prompt += "B. " + caption_1 + "\n"  
            prompt += "ASSISTANT:"
            max_new_tokens = 35
        else:
            print("Prompt type not supported!")

        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)

        # use_auth_token = "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"
        # tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased", use_auth_token=use_auth_token)
        # prompt = "A"
        # inputs_language = tokenizer(prompt, return_tensors="pt")
        # print("inputs_language", inputs_language)

        # use_auth_token = "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"
        # tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased", use_auth_token=use_auth_token)
        # prompt = "B"
        # inputs_language = tokenizer(prompt, return_tensors="pt")
        # print("inputs_language", inputs_language)


        outputs = self.model(**inputs)
        logits = outputs.logits.squeeze()

        # print("logits.shape", logits.shape)
        a_logits = torch.mean(logits[:, 1037]) ## 1037 is the token id for 'A' based on bert tokenizer
        b_logits = torch.mean(logits[:, 1038]) ## 1038 is the token id for 'B' based on bert tokenizer
        print("a_logits", a_logits)
        print("b_logits", b_logits)
        print("a_logits.shape", a_logits.shape)
        print("b_logits.shape", b_logits.shape)

        return a_logits, b_logits        

    
    @torch.no_grad()
    def blip2_caption_logits(self, image, caption_0, caption_1):
        if self.prompt_name == "gpt4-shorterprompt":
            prompt = "USER: \n Given this image and two candidate captions (A and B), which caption is the better description of the given image? Only give a single character answer - 'A' or 'B'.\n"
            prompt += "A. " + caption_0 + "\n"
            prompt += "B. " + caption_1 + "\n"  
            prompt += "ASSISTANT:"
            max_new_tokens = 35
        else:
            print("Prompt type not supported!")

        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)

        # use_auth_token = "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"
        # tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased", use_auth_token=use_auth_token)
        # prompt = "A"
        # inputs_language = tokenizer(prompt, return_tensors="pt")
        # print("inputs_language", inputs_language)

        # use_auth_token = "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"
        # tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased", use_auth_token=use_auth_token)
        # prompt = "B"
        # inputs_language = tokenizer(prompt, return_tensors="pt")
        # print("inputs_language", inputs_language)


        outputs = self.model(**inputs)
        logits = outputs.logits.squeeze()

        # print("logits.shape", logits.shape)
        a_logits = torch.mean(logits[:, 1037]) ## 1037 is the token id for 'A' based on bert tokenizer
        b_logits = torch.mean(logits[:, 1038]) ## 1038 is the token id for 'B' based on bert tokenizer
        print("a_logits", a_logits)
        print("b_logits", b_logits)
        print("a_logits.shape", a_logits.shape)
        print("b_logits.shape", b_logits.shape)

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


    @torch.no_grad()
    def cogvlm_caption_logits(self, image, caption_0, caption_1):
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

        outputs = self.model(**inputs)
        logits = outputs.logits.squeeze()
        a_logits = torch.mean(logits[:, 319]) ## 319 is the token id for 'A' based on llama2 tokenizer
        b_logits = torch.mean(logits[:, 350]) ## 350 is the token id for 'B' based on llama2 tokenizer

        return a_logits, b_logits        


    @torch.no_grad()
    def cogvlm_caption_logits(self, image, caption_0, caption_1):
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

        outputs = self.model(**inputs)
        logits = outputs.logits.squeeze()
        a_logits = torch.mean(logits[:, 319]) ## 319 is the token id for 'A' based on llama2 tokenizer
        b_logits = torch.mean(logits[:, 350]) ## 350 is the token id for 'B' based on llama2 tokenizer

        return a_logits, b_logits        


    def evaluate_sugarcrepe(self):
        sugarcrepe = {
            'add_obj'    : load_dataset("HuggingFaceM4/SugarCrepe_add_obj", trust_remote_code=True)["test"],
            'add_att'    : load_dataset("HuggingFaceM4/SugarCrepe_add_att", trust_remote_code=True)["test"],
            'replace_obj': load_dataset("HuggingFaceM4/SugarCrepe_replace_obj", trust_remote_code=True)["test"],
            'replace_att': load_dataset("HuggingFaceM4/SugarCrepe_replace_att", trust_remote_code=True)["test"],
            'replace_rel': load_dataset("HuggingFaceM4/SugarCrepe_replace_rel", trust_remote_code=True)["test"],
            'swap_obj'   : load_dataset("HuggingFaceM4/SugarCrepe_swap_obj", trust_remote_code=True)["test"],
            'swap_att'   : load_dataset("HuggingFaceM4/SugarCrepe_swap_att", trust_remote_code=True)["test"],
        }

        metrics = {}

        if self.model_name == "llava-hf/llava-1.5-7b-hf":
            captioner = self.llava_caption_choice
        elif self.model_name == "Salesforce/blip2-opt-2.7b":
            captioner = self.blip2_caption_choice
        elif self.model_name == "THUDM/cogvlm-chat-hf":
            captioner = self.cogvlm_caption_choice

        if self.evaluation_type == "logits":
            if self.model_name == "llava-hf/llava-1.5-7b-hf":
                captioner = self.llava_caption_logits
            # elif self.model_name == "Salesforce/blip2-opt-2.7b":
            #     captioner = self.blip2_caption_choice
            # elif self.model_name == "THUDM/cogvlm-chat-hf":
            #     captioner = self.cogvlm_caption_choice
            for c, data_dict in sugarcrepe.items():
                correct_cnt = 0
                idx_limit = 20
                iter_cnt = 0
                for data in tqdm(data_dict, desc=f'evaluating {c}'):
                    correct = 0
                    answerA, answerB = captioner(data['image'], data['tested_labels'][0], data['tested_labels'][1])
                    if answerA > answerB:
                        correct = 1
                    correct_cnt += correct
                    iter_cnt += 1
                    if iter_cnt >= idx_limit:
                        break
                # count = len(data_dict)
                count = idx_limit
                metrics[c] = correct_cnt / count
                
            print(metrics)
            return {"SugarCrepe_accuracies": metrics}
        else:
            for c, data_dict in sugarcrepe.items():
                correct_cnt = 0
                idx_limit = 20
                iter_cnt = 0
                for data in tqdm(data_dict, desc=f'evaluating {c}'):
                    correct = 0
                    answer = captioner(data['image'], data['tested_labels'][0], data['tested_labels'][1])
                    if answer[0].lower() == 'a':
                        correct = 1
                    correct_cnt += correct
                    iter_cnt += 1
                    if iter_cnt >= idx_limit:
                        break
                # count = len(data_dict)
                count = idx_limit
                metrics[c] = correct_cnt / count
                
            print(metrics)
            return {"SugarCrepe_accuracies": metrics}