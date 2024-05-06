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


    def llava_image_to_caption_binary_match(self, caption, image):

        if self.prompt_name == "gpt4":
            prompt = "USER: <image>\n Select whether the image matches the caption. Pay close attention to the word order. (Give a short explanation first, then change to a new line give the final answer in the exact format of: \"The answer is Yes/No.\"))\n"
            prompt += "Caption: " + caption.strip() + "\n"
            prompt += "ASSISTANT:"
            max_new_tokens = 35

        elif self.prompt_name == "gpt4-smallerprompt":
            prompt = "USER: <image>\n Select whether the image matches the caption. Pay close attention to the word order. Give the final answer in the exact format of: \"The answer is Yes/No.\"))\n"
            prompt += "Caption: " + caption.strip() + "\n"
            prompt += "ASSISTANT:"
            max_new_tokens = 35

        elif self.prompt_name == "gpt4-evensmallerprompt":
            prompt = "USER: <image>\n Does the image match the caption?. Pay close attention to the word order. Answer in the format of: \"Yes or No.\"))\n"
            prompt += "Caption: " + caption.strip() + "\n"
            prompt += "ASSISTANT:"
            max_new_tokens = 35

        elif self.prompt_name == "gpt4-evensmallerprompt2":
            prompt = "USER: <image>\n Does the image match the caption?. Answer in the format of: \"Yes or No.\"))\n"
            prompt += "Caption: " + caption.strip() + "\n"
            prompt += "ASSISTANT:"
            max_new_tokens = 35
        
        elif self.prompt_name == "alignment":
            prompt = "USER: <image> Does this image entail the description:" 
            prompt += caption.strip() + "?"
            prompt += "ASSISTANT:"
            max_new_tokens = 35

        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)

        # Generate
        generate_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        output = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        output = output.split('ASSISTANT:')[1]
        return output    


    def BLIP2_image_to_caption_binary_match(self, caption, image):

        if self.prompt_name == "gpt4":
            prompt = "Question: \n Select whether the image matches the caption. Pay close attention to the word order. (Give a short explanation first, then change to a new line give the final answer in the exact format of: \"The answer is Yes/No.\"))\n"
            prompt += caption.strip() + "\n"
            prompt += "Answer:"
            max_new_tokens = 35


        if self.prompt_name == "gpt4-smallerprompt":
            prompt = "Question: \n Select whether the image matches the caption. Pay close attention to the word order. Give the final answer in the exact format of: \"The answer is Yes/No.\"))\n"
            prompt += "Caption: " + caption.strip() + "\n"
            prompt += "Answer:"
            max_new_tokens = 35
        
        if self.prompt_name == "gpt4-evensmallerprompt":
            prompt = "Question: \n Does the image match the caption?. Pay close attention to the word order. Answer in the format of: \"Yes or No.\"))\n"
            prompt += "Caption: " + caption.strip() + "\n"
            prompt += "Answer:"
            max_new_tokens = 35
        
        if self.prompt_name == "gpt4-evensmallerprompt2":
            prompt = "Question: \n Does the image match the caption?. Answer in the format of: \"Yes or No.\"))\n"
            prompt += "Caption: " + caption.strip() + "\n"
            prompt += "Answer:"
            max_new_tokens = 35

        if self.prompt_name == "alignment":
            prompt = "Question: Does this image entail the description:" 
            prompt += caption.strip() + "?"
            prompt += "Answer:"
            max_new_tokens = 35

        ##icl arises form data
        ##why BLIP2 fails? dataset

        # self.model.to(device)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")

        # Generate
        generate_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        output = self.processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
        # output = output.split('Answer:')[1]
        return output
    

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
            prompt = "USER: <image>\n Does the image match the caption?. Pay close attention to the word order. Answer in the exact format of: 'Yes' or 'No'\n"
            prompt += "Caption: " + caption.strip() + "\n"
            prompt += "ASSISTANT:"
            max_new_tokens = 35

        elif self.prompt_name == "gpt4-evensmallerprompt2":
            prompt = "USER: <image>\n Does the image match the caption?. Answer in the format of: \"Yes or No.\"))\n"
            prompt += "Caption: " + caption.strip() + "\n"
            prompt += "ASSISTANT:"
            max_new_tokens = 35

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
        with torch.no_grad():
            output = self.model.generate(**inputs, **gen_kwargs)
            output = output[:, inputs['input_ids'].shape[1]:]
            output = self.tokenizer.decode(output[0])

        output = output.split("</s>")[0]
        return output
    

    def evaluate_winoground(self):

        auth_token = "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" # Replace with an auth token, which you can get from your huggingface account: Profile -> Settings -> Access Tokens -> New Token
        winoground = load_dataset("facebook/winoground", use_auth_token=auth_token, trust_remote_code=True)["test"]

        ##Evaluation

        ##images are all winoground images
        random.seed(2023)
        subset_idx = random.sample(range(len(winoground)), 100)
        # len(subset_idx[:20])
        #taking the first 20 for time purposes
        subset_idx = subset_idx[:20]
        if self.evaluation_type == "text_image_group_score":
            text_correct_count = 0
            image_correct_count = 0
            group_correct_count = 0
            total = 0
            image_caption_match_results = {}

            for idx in tqdm(subset_idx):
                image_0 = winoground[idx]["image_0"].convert("RGB")
                image_1 = winoground[idx]["image_1"].convert("RGB")
                caption_0 = winoground[idx]["caption_0"]
                caption_1 = winoground[idx]["caption_1"]

                print ("Example: #", total)
                self.show_example(benchmark=winoground, idx=idx)
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

                ans_c0_i0 = captioner(caption_0, image_0)
                image_caption_match_results[str(idx)+"_c0_i0"] = ans_c0_i0
                print ("Match between C0 and I0: ", ans_c0_i0.lower())
                if "yes" in ans_c0_i0.lower():
                    result["c0_i0"] = 1.0
                else:
                    result["c0_i0"] = 0.0

                ans_c0_i1 = captioner(caption_0, image_1)
                image_caption_match_results[str(idx)+"_c0_i1"] = ans_c0_i1
                print ("Match between C0 and I1: ", ans_c0_i1)
                if "yes" in ans_c0_i1.lower():
                    result["c0_i1"] = 1.0
                else:
                    result["c0_i1"] = 0.0   

                ans_c1_i0 = captioner(caption_1, image_0)
                image_caption_match_results[str(idx)+"_c1_i0"] = ans_c1_i0
                print ("Match between C1 and I0: ", ans_c1_i0)
                if "yes" in ans_c1_i0.lower():
                    result["c1_i0"] = 1.0
                else:
                    result["c1_i0"] = 0.0

                ans_c1_i1 = captioner(caption_1, image_1)
                image_caption_match_results[str(idx)+"_c1_i1"] = ans_c1_i1
                print ("Match between C1 and I1: ", ans_c1_i1)
                if "yes" in ans_c1_i1.lower():
                    result["c1_i1"] = 1.0
                else:
                    result["c1_i1"] = 0.0

                print ("result: ", result)

                text_correct_count += 1 if self.text_correct(result) else 0
                image_correct_count += 1 if self.image_correct(result) else 0
                group_correct_count += 1 if self.group_correct(result) else 0

                total += 1
                print ("Current Acc: {}/{} = {}%\n".format(group_correct_count, total, group_correct_count / total * 100))

                # except:
                #     print ("skipped")
                #     continue
            return {"text_score": text_correct_count/total, "image_score": image_correct_count/total, "group_score": group_correct_count/total}
        
        if self.evaluation_type == "accuracy_score":
            correct = 0
            total = 0
            image_to_caption_results = {} ## for saving results

            for idx in tqdm(subset_idx):
                image_0 = winoground[idx]["image_0"]
                image_1 = winoground[idx]["image_1"]
                caption_0 = winoground[idx]["caption_0"]
                caption_1 = winoground[idx]["caption_1"]
                correct_a = False
                correct_b = False

                print ("Example: #", total)
                
                self.show_example(benchmark=winoground, idx=idx)

                # try:
                #     ## match caption for image_0
                #     answer_0 = self.llava_image_to_caption(image_0, caption_0, caption_1)
                #     image_to_caption_results[str(idx)+"_image_0"] = answer_0
                #     print ("\nUsing image_0 to select the better caption: ")
                #     print (answer_0)
                #     if "answer is a" in answer_0.lower():
                #         correct_a = True
                #     print ("\n")

                #     ## match caption for image_1
                #     answer_1 = self.llava_image_to_caption(image_1, caption_0, caption_1)
                #     image_to_caption_results[str(idx)+"_image_1"] = answer_1
                #     print ("\nUsing image_1 to select the better caption: ")
                #     print (answer_1)
                #     if "answer is b" in answer_1.lower():
                #         correct_b = True

                #     ## the example is counted correct only if both matching are correct
                #     if correct_a and correct_b:
                #         correct += 1
                #     total += 1

                #     print ("Current Acc: {}/{} = {}%\n".format(correct, total, correct / total * 100))

            
                ## match caption for image_0
                answer_0 = self.llava_image_to_caption(image_0, caption_0, caption_1)
                image_to_caption_results[str(idx)+"_image_0"] = answer_0
                print ("\nUsing image_0 to select the better caption: ")
                print(answer_0)
                print(answer_0[:4])
                if "A." in answer_0[:4]:
                    correct_a = True
                print ("\n")

                ## match caption for image_1
                answer_1 = self.llava_image_to_caption(image_1, caption_0, caption_1)
                image_to_caption_results[str(idx)+"_image_1"] = answer_1
                print("\nUsing image_1 to select the better caption: ")
                print(answer_1)
                print(answer_1[:4])
                if "B." in answer_1[:4]:
                    correct_b = True

                ## the example is counted correct only if both matching are correct
                if correct_a and correct_b:
                    correct += 1
                total += 1

                print ("Current Acc: {}/{} = {}%\n".format(correct, total, correct / total * 100))
        
            return {"accuracy_score": correct / total}
        
        else:
            raise ValueError(f"Unknown evaluation type: {self.evaluation_type}")
